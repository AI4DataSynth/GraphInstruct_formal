"""Run baseline experiments: generate graphs via LLM and evaluate.

Usage:
    python scripts/run_baseline.py --levels 0 1 --provider vllm --output results/baseline.json
    python scripts/run_baseline.py --strategy few-shot --levels 0 --provider vllm
    python scripts/run_baseline.py --provider openai --api-base https://your-relay/v1 --model gpt-4o-mini
    python scripts/run_baseline.py --resume results/baseline.jsonl  # resume from checkpoint

Supports providers:
- vllm: vLLM server (for local LLaMA models)
- openai: OpenAI-compatible API (GPT-4, etc.)
- claude: Anthropic Claude API (Claude Sonnet/Opus/Haiku)
- mock: Random graph generation (for testing the pipeline)

Supports prompt strategies:
- zero-shot: Direct instruction only
- few-shot: 3 in-context examples before the instruction
- zero-cot: Instruction + "Let's think step by step"
- few-cot: Few-shot examples + CoT reasoning

Intermediate results:
- Each completed instruction is immediately saved to a .jsonl checkpoint file
- Raw LLM outputs are preserved for post-hoc analysis
- Use --resume to continue from the last checkpoint after a crash
"""

from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from graphinstruct.data_loader import Instruction, load_all_levels
from graphinstruct.evaluate import (
    BenchmarkResult,
    GenerationResult,
    evaluate_benchmark,
)
from graphinstruct.metrics.efficiency import TokenRecord
from graphinstruct.scoring import level_score, total_score
from graphinstruct.metrics.l5_eval import apply_graph_edit as apply_graph_edit_shared
from graphinstruct.parser import FORMATS, parse_format, serialize_format
from graphinstruct.parser.code_style import parse

STRATEGIES = ("zero-shot", "few-shot", "zero-cot", "few-cot")
FEW_SHOT_K = 3

# Per-model max_tokens used in the paper. Models not listed default to
# DEFAULT_MAX_TOKENS. Keep this in sync with docs/REPRODUCE.md.
MODEL_MAX_TOKENS = {
    "gpt-3.5-turbo": 4096,  # API hard limit
}
DEFAULT_MAX_TOKENS = 16384


def _resolve_max_tokens(model: str) -> int:
    """Return the paper-aligned max_tokens for a given model name."""
    return MODEL_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

FORMAT_EXAMPLES = {
    "code": (
        "Graph[name='example', nodes=3] {\n"
        "    node_list = ['0', '1', '2'];\n"
        "    edge_list = [('0','1'), ('1','2')];\n"
        "}"
    ),
    "adjacency": ("Nodes: 0, 1, 2\n" "Adjacency list:\n" "0: 1 2\n" "1: 0\n" "2: 0"),
    "edge": ("Nodes: 0, 1, 2\n" "Edges:\n" "0 1\n" "0 2"),
}

FORMAT_NAMES = {
    "code": "code-style",
    "adjacency": "adjacency list",
    "edge": "edge list",
}

_ANTI_LAZY = (
    "\n\nIMPORTANT: Write out every node and edge EXPLICITLY. "
    "Do NOT use ellipsis (...), range(), abbreviations, or shortcuts "
    "in node_list or edge_list. Every element must be listed individually."
)

SYSTEM_PROMPTS = {
    "default": (
        "You are a graph generation assistant. Generate graphs in the following "
        "{format_name} format:\n\n"
        "{format_example}\n\n"
        "If the constraints are mathematically contradictory (e.g., "
        "a d-regular graph requires d*n to be even), explain why "
        "no valid graph exists instead of generating one.\n"
        "Otherwise, output ONLY the graph in the above format, no explanations."
        + _ANTI_LAZY
    ),
    "L3": (
        "You are a graph generation assistant. Generate graphs in the following "
        "{format_name} format. Graphs may include node and edge attributes:\n\n"
        "{format_example}\n\n"
        "Output ONLY the graph in the above format, no explanations." + _ANTI_LAZY
    ),
    "L5": (
        "You are a graph editing assistant. Given an existing graph, apply modifications "
        "using the following incremental edit format:\n\n"
        "GraphEdit[name='edit1', base='original'] {{\n"
        "    add_edges = [('A','D'), ('B','E')];\n"
        "    remove_edges = [('A','B')];\n"
        "    reasoning = 'Added edges to increase connectivity while removing redundant edge.';\n"
        "}}\n\n"
        "If the task requires generating a complete new graph instead, use:\n\n"
        "{format_example}\n\n"
        "Output ONLY the graph/edit in the above format, no extra explanations."
        + _ANTI_LAZY
    ),
}

COT_SUFFIX = (
    "\n\nLet's think step by step:\n"
    "1. Identify the required graph properties from the instruction.\n"
    "2. Determine the node set and edge set that satisfy all constraints.\n"
    "3. Output the graph in the required format."
)


def get_system_prompt(
    level: int,
    strategy: str = "zero-shot",
    graph_format: str = "code",
) -> str:
    """Get the appropriate system prompt for a given level, strategy, and format."""
    if level >= 5:
        template = SYSTEM_PROMPTS["L5"]
    elif level >= 3:
        template = SYSTEM_PROMPTS["L3"]
    else:
        template = SYSTEM_PROMPTS["default"]

    base = template.format(
        format_name=FORMAT_NAMES.get(graph_format, "code-style"),
        format_example=FORMAT_EXAMPLES.get(graph_format, FORMAT_EXAMPLES["code"]),
    )

    if strategy in ("zero-cot", "few-cot"):
        base = base.replace(
            "Output ONLY the graph in the above format, no explanations.",
            "First reason about the required properties step by step, "
            "then output the graph in the above format.",
        )
        base = base.replace(
            "Output ONLY the graph/edit in the above format, no extra explanations.",
            "First reason about what edits are needed step by step, "
            "then output the graph/edit in the above format.",
        )
    return base


_INFEASIBLE_REJECTION = (
    "This is infeasible. A d-regular graph on n nodes requires d*n to be even "
    "(Handshaking Lemma), but the given d*n is odd. No valid graph exists."
)


def _extract_graph_type_from_constraints(
    constraints: tuple[str, ...] | list[str]
) -> str:
    """Extract graph_type value from explicit constraints, or '' if absent."""
    for c in constraints:
        if c.startswith("graph_type="):
            return c.split("=", 1)[1].strip()
    return ""


def _select_few_shot_examples(
    target: Instruction,
    all_instructions: dict[int, list[Instruction]],
    k: int = FEW_SHOT_K,
) -> list[tuple[str, str]]:
    """Select k few-shot examples from the same level, preferring same graph type.

    Selection priority:
    1. Same level + same graph_type (excluding target)
    2. Same level + any graph_type (fallback if not enough same-type)
    3. Adjacent levels (last resort)

    Returns list of (instruction_text, reference_solution) pairs.

    For levels with infeasible instructions (L2), one example slot is reserved
    for an infeasible example with a rejection response, so the LLM learns both
    "generate" and "reject" patterns.
    """
    level = target.level
    target_type = _extract_graph_type_from_constraints(target.explicit_constraints)

    # Separate feasible and infeasible candidates
    all_level = [
        inst for inst in all_instructions.get(level, []) if inst.id != target.id
    ]
    feasible_candidates = [
        inst
        for inst in all_level
        if inst.feasible
        and inst.reference_solutions
        and len(inst.reference_solutions) > 0
    ]
    infeasible_candidates = [inst for inst in all_level if not inst.feasible]

    # Option C: If this level has infeasible instructions, mix 1 infeasible + (k-1) feasible
    if infeasible_candidates:
        inf_example = _random.choice(infeasible_candidates)
        inf_pair = (inf_example.instruction, _INFEASIBLE_REJECTION)
        # For feasible slots, prefer same graph type
        feasible_selected = _select_by_type(feasible_candidates, target_type, k - 1)
        feasible_pairs = [
            (inst.instruction, inst.reference_solutions[0])
            for inst in feasible_selected
        ]
        mixed = feasible_pairs + [inf_pair]
        _random.shuffle(mixed)
        return mixed

    # Normal path: all feasible — prefer same graph type
    candidates = _select_by_type(feasible_candidates, target_type, k)
    if not candidates:
        # Fallback: try adjacent levels
        for delta in [1, -1, 2, -2]:
            adj = level + delta
            adj_feasible = [
                inst
                for inst in all_instructions.get(adj, [])
                if inst.feasible
                and inst.reference_solutions
                and len(inst.reference_solutions) > 0
            ]
            candidates = _select_by_type(adj_feasible, target_type, k)
            if candidates:
                break

    return [(inst.instruction, inst.reference_solutions[0]) for inst in candidates]


def _select_by_type(
    candidates: list[Instruction],
    target_type: str,
    k: int,
) -> list[Instruction]:
    """Select k candidates, preferring those with the same graph_type.

    If enough same-type candidates exist, select all from them.
    Otherwise, fill remaining slots from other-type candidates.
    """
    if not candidates or k <= 0:
        return []

    if target_type:
        same_type = [
            inst
            for inst in candidates
            if _extract_graph_type_from_constraints(inst.explicit_constraints)
            == target_type
        ]
        other_type = [
            inst
            for inst in candidates
            if _extract_graph_type_from_constraints(inst.explicit_constraints)
            != target_type
        ]
    else:
        # No graph_type constraint on target — all candidates are equal
        same_type = candidates
        other_type = []

    selected: list[Instruction] = []
    if len(same_type) >= k:
        selected = _random.sample(same_type, k)
    else:
        selected = list(same_type)
        remaining = k - len(selected)
        if other_type and remaining > 0:
            selected += _random.sample(other_type, min(remaining, len(other_type)))

    return selected


def _build_cot_reasoning(inst_text: str, is_rejection: bool) -> str:
    """Build a concrete CoT reasoning chain from the instruction text.

    Extracts actual constraints from the instruction and produces
    step-by-step verification rather than a generic template sentence.
    """
    if is_rejection:
        return (
            "Reasoning: Let me check constraint compatibility step by step.\n"
            "1. Extract all constraints from the instruction.\n"
            "2. Check mathematical feasibility: a d-regular graph on n nodes "
            "requires d*n to be even; a tree on n nodes has exactly n-1 edges.\n"
            "3. The constraints are contradictory — no valid graph exists."
        )

    steps = ["Reasoning: Let me analyze the requirements step by step."]
    inst_lower = inst_text.lower()

    if "tree" in inst_lower:
        steps.append("1. A tree requires n-1 edges, must be connected and acyclic.")
    elif "cycle" in inst_lower:
        steps.append(
            "1. A cycle on n nodes has exactly n edges, each node has degree 2."
        )
    elif "bipartite" in inst_lower:
        steps.append(
            "1. A bipartite graph has two disjoint node sets with edges only between them."
        )
    elif "complete" in inst_lower:
        steps.append("1. A complete graph K_n has n*(n-1)/2 edges.")
    elif "star" in inst_lower:
        steps.append("1. A star graph has one hub connected to all other nodes.")
    elif "regular" in inst_lower:
        steps.append(
            "1. A d-regular graph requires every node to have exactly degree d."
        )
    else:
        steps.append("1. Identify the required graph type and structural properties.")

    steps.append(
        "2. Determine the node set and compute the required edge count from constraints."
    )
    steps.append(
        "3. Construct edges satisfying all constraints, then verify: "
        "node count, edge count, and structural properties all match."
    )
    return "\n".join(steps)


def _format_few_shot_block(examples: list[tuple[str, str]], cot: bool = False) -> str:
    """Format few-shot examples into a prompt block."""
    parts = ["Here are some examples:\n"]
    for i, (inst_text, ref_sol) in enumerate(examples, 1):
        is_rejection = ref_sol == _INFEASIBLE_REJECTION
        parts.append(f"Example {i}:")
        parts.append(f"Instruction: {inst_text}")
        if cot:
            parts.append(_build_cot_reasoning(inst_text, is_rejection))
        parts.append(f"Output:\n{ref_sol}\n")
    parts.append(
        "Now, generate a graph for the following instruction. "
        "If the constraints are mathematically contradictory, "
        "explain why no valid graph exists instead.\n"
    )
    return "\n".join(parts)


def _convert_examples_format(
    examples: list[tuple[str, str]],
    graph_format: str,
) -> list[tuple[str, str]]:
    """Convert few-shot examples from code-style to target format."""
    converted = []
    for inst_text, ref_sol in examples:
        try:
            parsed = parse(ref_sol)
            if parsed.graph is not None:
                new_sol = serialize_format(parsed.graph, fmt=graph_format)
                converted.append((inst_text, new_sol))
            else:
                converted.append((inst_text, ref_sol))
        except Exception:
            # Keep original if conversion fails
            converted.append((inst_text, ref_sol))
    return converted


def build_prompt(
    instruction: Instruction,
    strategy: str = "zero-shot",
    all_instructions: dict[int, list[Instruction]] | None = None,
    graph_format: str = "code",
) -> str:
    """Build a user prompt from an instruction and strategy.

    Args:
        instruction: The target instruction.
        strategy: One of 'zero-shot', 'few-shot', 'zero-cot', 'few-cot'.
        all_instructions: All loaded instructions (needed for few-shot example selection).
        graph_format: One of 'code', 'adjacency', 'edge'.
    """
    prompt_parts: list[str] = []

    # Few-shot examples
    if strategy in ("few-shot", "few-cot") and all_instructions is not None:
        examples = _select_few_shot_examples(instruction, all_instructions)
        if examples and graph_format != "code":
            # Re-serialize examples into target format
            examples = _convert_examples_format(examples, graph_format)
        if examples:
            prompt_parts.append(
                _format_few_shot_block(examples, cot=(strategy == "few-cot"))
            )

    # Main instruction
    inst_text = instruction.instruction
    if instruction.level == 5 and instruction.base_graph:
        inst_text = (
            f"Base graph:\n\n" f"{instruction.base_graph}\n\n" f"Task: {inst_text}"
        )
    prompt_parts.append(inst_text)

    # CoT suffix
    if strategy in ("zero-cot", "few-cot"):
        prompt_parts.append(COT_SUFFIX)

    return "\n".join(prompt_parts)


def _extract_task_type(constraints: tuple | list) -> str | None:
    """Extract task_type from explicit constraints."""
    for c in constraints:
        if c.startswith("task_type="):
            return c.split("=", 1)[1]
    return None


def apply_graph_edit(
    base_graph: "nx.Graph",
    add_edges: tuple,
    remove_edges: tuple,
    task_type: str | None = None,
) -> "nx.Graph":
    """Apply GraphEdit operations to a base graph, returning new graph.

    Delegates to shared implementation in l5_eval for consistency.
    """
    return apply_graph_edit_shared(base_graph, add_edges, remove_edges, task_type)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def generate_mock(
    instruction: Instruction,
    num_samples: int = 1,
    override_prompt: str | None = None,
    **_ignored: object,
) -> list[GenerationResult]:
    """Mock provider: generates random graphs for pipeline testing.

    ``override_prompt`` is accepted for API parity with real providers but
    ignored (mock never actually calls an LLM). Extra kwargs (strategy,
    graph_format, model, etc.) are swallowed so runners can call mock the
    same way they call real providers.
    """
    import random

    results = []
    for _ in range(num_samples):
        # L5: parse base graph and apply random edits
        if instruction.level == 5 and instruction.base_graph:
            G = _mock_l5_edit(instruction, random)
        else:
            G = _mock_generate(instruction, random)

        # Simulate token costs
        input_tokens = random.randint(80, 150)
        output_tokens = random.randint(30, 100)

        results.append(
            GenerationResult(
                instruction=instruction,
                graph=G,
                raw_output=f"[mock graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges]",
                token_record=TokenRecord(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    valid=G is not None,
                ),
            )
        )

    return results


def _mock_generate(instruction: Instruction, random) -> nx.Graph:
    """Generate a mock graph based on instruction constraints."""
    n = 5
    for c in instruction.explicit_constraints:
        if c.startswith("num_nodes="):
            n = int(c.split("=")[1])

    graph_type = None
    for c in instruction.explicit_constraints:
        if c.startswith("graph_type="):
            graph_type = c.split("=")[1]

    if graph_type == "tree":
        return nx.random_labeled_tree(n)
    elif graph_type == "cycle":
        return nx.cycle_graph(n)
    elif graph_type == "complete":
        return nx.complete_graph(n)
    elif graph_type == "star":
        return nx.star_graph(n - 1)
    elif graph_type == "path":
        return nx.path_graph(n)
    elif graph_type == "complete_bipartite":
        p, q = 3, 3
        for c in instruction.explicit_constraints:
            if c.startswith("bipartite_sides="):
                parts = c.split("=")[1].split(",")
                p, q = int(parts[0]), int(parts[1])
        return nx.complete_bipartite_graph(p, q)
    else:
        m = max(n - 1, n)
        return nx.gnm_random_graph(n, m)


def _mock_l5_edit(instruction: Instruction, random) -> nx.Graph:
    """Mock L5: parse base graph and apply random edge edits."""
    try:
        base_parsed = parse(instruction.base_graph)
        G = base_parsed.graph.copy()
    except Exception:
        return _mock_generate(instruction, random)

    edges = list(G.edges())
    nodes = list(G.nodes())

    # Random remove 1-2 edges
    if edges:
        num_remove = min(random.randint(1, 2), len(edges))
        for e in random.sample(edges, num_remove):
            G.remove_edge(*e)

    # Random add 1-2 edges
    for _ in range(random.randint(1, 2)):
        if len(nodes) >= 2:
            u, v = random.sample(nodes, 2)
            G.add_edge(u, v)

    return G


# Transient HTTP status codes worth retrying
_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504, 530}


def _is_retriable(exc: Exception) -> bool:
    """Return True if the request error is transient and worth retrying."""
    import requests as _req

    if isinstance(exc, (_req.exceptions.Timeout, _req.exceptions.ConnectionError)):
        return True
    if isinstance(exc, _req.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRIABLE_STATUS_CODES
    return False


def _is_qwen35_model(model: str) -> bool:
    m = model.lower()
    return "qwen3.5" in m or "qwen-3.5" in m or "qwen3-5" in m


def generate_vllm(
    instruction: Instruction,
    num_samples: int = 1,
    api_base: str = "http://localhost:8000/v1",
    model: str = "Llama-2-13b-chat-hf",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    strategy: str = "zero-shot",
    all_instructions: dict[int, list[Instruction]] | None = None,
    graph_format: str = "code",
    delay: float = 0.0,
    override_prompt: str | None = None,
) -> list[GenerationResult]:
    """Generate graphs using vLLM server."""
    try:
        import requests
    except ImportError:
        print(
            "ERROR: requests library required for vLLM provider. Install with: pip install requests"
        )
        sys.exit(1)

    results = []
    if override_prompt is not None:
        prompt = override_prompt
    else:
        prompt = build_prompt(instruction, strategy, all_instructions, graph_format)
    system_prompt = get_system_prompt(instruction.level, strategy, graph_format)

    for sample_idx in range(num_samples):
        if delay > 0 and sample_idx > 0:
            time.sleep(delay)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if disable_thinking:
            payload["enable_thinking"] = False

        max_retries = 5
        api_calls = 0
        for attempt in range(max_retries):
            api_calls += 1
            try:
                resp = requests.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()

                raw_output = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                try:
                    if graph_format == "code":
                        parsed = parse(raw_output)
                    else:
                        parsed = parse_format(raw_output, graph_format)
                    graph = parsed.graph
                    # Handle GraphEdit: apply edits to base graph (code format only)
                    if parsed.parse_type == "GraphEdit" and instruction.base_graph:
                        base_parsed = parse(instruction.base_graph)
                        if base_parsed.graph is not None:
                            graph = apply_graph_edit(
                                base_parsed.graph,
                                parsed.add_edges,
                                parsed.remove_edges,
                                task_type=_extract_task_type(
                                    instruction.explicit_constraints
                                ),
                            )
                except Exception:
                    graph = None
                break
            except requests.exceptions.RequestException as e:
                if _is_retriable(e) and attempt < max_retries - 1:
                    wait = min(2**attempt * 5, 60) + _random.uniform(0, 2)
                    time.sleep(wait)
                    continue
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break
            except Exception as e:
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break

        results.append(
            GenerationResult(
                instruction=instruction,
                graph=graph,
                raw_output=raw_output,
                token_record=TokenRecord(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    api_calls=api_calls,
                    valid=graph is not None,
                ),
            )
        )

    return results


def generate_openai(
    instruction: Instruction,
    num_samples: int = 1,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    strategy: str = "zero-shot",
    all_instructions: dict[int, list[Instruction]] | None = None,
    graph_format: str = "code",
    delay: float = 0.0,
    merge_system_prompt: bool = False,
    disable_thinking: bool = False,
    override_prompt: str | None = None,
) -> list[GenerationResult]:
    """Generate graphs using OpenAI-compatible API (GPT-4o, GPT-4o-mini, etc.).

    Uses the official openai SDK (handles SOCKS/HTTP proxies via httpx).
    Requires OPENAI_API_KEY environment variable to be set.
    """
    import os

    try:
        import openai as _openai
    except ImportError:
        print(
            "ERROR: openai library required for OpenAI provider. "
            "Install with: pip install openai"
        )
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = _openai.OpenAI(api_key=api_key, base_url=api_base, timeout=300.0)

    # DashScope streaming for Qwen3.5 MoE series has frequent transient
    # connection drops during 2+ minute inferences. Use more retries with
    # fixed short backoff instead of exponential, to ride out fast-fail drops.
    is_qwen35 = _is_qwen35_model(model)

    results = []
    if override_prompt is not None:
        prompt = override_prompt
    else:
        prompt = build_prompt(instruction, strategy, all_instructions, graph_format)
    system_prompt = get_system_prompt(instruction.level, strategy, graph_format)

    for sample_idx in range(num_samples):
        if delay > 0 and sample_idx > 0:
            time.sleep(delay)

        max_retries = 10 if is_qwen35 else 5
        api_calls = 0
        for attempt in range(max_retries):
            api_calls += 1
            try:
                if merge_system_prompt:
                    # Some API relays silently drop system messages.
                    # Merge system prompt into the user message as a workaround.
                    merged = f"{system_prompt}\n\n---\n\n{prompt}"
                    messages = [{"role": "user", "content": merged}]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                # Claude models via OpenAI-compatible API reject
                # temperature + top_p together; only send temperature.
                kwargs_api = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if "claude" not in model.lower():
                    kwargs_api["top_p"] = top_p
                if disable_thinking:
                    kwargs_api["extra_body"] = {"enable_thinking": False}

                if is_qwen35:
                    # Qwen3.5 inferences take ~2 min. Non-streaming sockets get
                    # killed by network middleboxes during that silence, causing
                    # Connection error while the server still bills tokens. Stream
                    # mode sends SSE chunks continuously, keeping the connection
                    # alive end-to-end.
                    kwargs_api["stream"] = True
                    kwargs_api["stream_options"] = {"include_usage": True}
                    stream = client.chat.completions.create(**kwargs_api)
                    chunks: list[str] = []
                    stream_usage = None
                    for ch in stream:
                        if ch.choices:
                            delta = ch.choices[0].delta
                            if delta and delta.content:
                                chunks.append(delta.content)
                        if getattr(ch, "usage", None):
                            stream_usage = ch.usage
                    raw_output = "".join(chunks)
                    input_tokens = stream_usage.prompt_tokens if stream_usage else 0
                    output_tokens = (
                        stream_usage.completion_tokens if stream_usage else 0
                    )
                else:
                    response = client.chat.completions.create(**kwargs_api)
                    raw_output = response.choices[0].message.content or ""
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0

                try:
                    if graph_format == "code":
                        parsed = parse(raw_output)
                    else:
                        parsed = parse_format(raw_output, graph_format)
                    graph = parsed.graph
                    # Handle GraphEdit: apply edits to base graph (code format only)
                    if parsed.parse_type == "GraphEdit" and instruction.base_graph:
                        base_parsed = parse(instruction.base_graph)
                        if base_parsed.graph is not None:
                            graph = apply_graph_edit(
                                base_parsed.graph,
                                parsed.add_edges,
                                parsed.remove_edges,
                                task_type=_extract_task_type(
                                    instruction.explicit_constraints
                                ),
                            )
                except Exception:
                    graph = None
                break
            except _openai.APIStatusError as e:
                if (
                    e.status_code in (429, 500, 502, 503, 529)
                    and attempt < max_retries - 1
                ):
                    wait = (
                        3.0
                        if is_qwen35
                        else min(2**attempt * 5, 60) + _random.uniform(0, 2)
                    )
                    time.sleep(wait)
                    continue
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break
            except (_openai.APIConnectionError, _openai.APITimeoutError) as e:
                if attempt < max_retries - 1:
                    wait = (
                        3.0
                        if is_qwen35
                        else min(2**attempt * 5, 60) + _random.uniform(0, 2)
                    )
                    time.sleep(wait)
                    continue
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break
            except Exception as e:
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break

        results.append(
            GenerationResult(
                instruction=instruction,
                graph=graph,
                raw_output=raw_output,
                token_record=TokenRecord(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    api_calls=api_calls,
                    valid=graph is not None,
                ),
            )
        )

    return results


def generate_claude(
    instruction: Instruction,
    num_samples: int = 1,
    api_base: str = "https://api.anthropic.com",
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    strategy: str = "zero-shot",
    all_instructions: dict[int, list[Instruction]] | None = None,
    graph_format: str = "code",
    delay: float = 0.0,
    override_prompt: str | None = None,
) -> list[GenerationResult]:
    """Generate graphs using Anthropic Claude API.

    Requires ANTHROPIC_API_KEY environment variable to be set.
    """
    import os

    try:
        import requests
    except ImportError:
        print(
            "ERROR: requests library required for Claude provider. "
            "Install with: pip install requests"
        )
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    results = []
    if override_prompt is not None:
        prompt = override_prompt
    else:
        prompt = build_prompt(instruction, strategy, all_instructions, graph_format)
    system_prompt = get_system_prompt(instruction.level, strategy, graph_format)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    for sample_idx in range(num_samples):
        if delay > 0 and sample_idx > 0:
            time.sleep(delay)

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        max_retries = 5
        api_calls = 0
        for attempt in range(max_retries):
            api_calls += 1
            try:
                resp = requests.post(
                    f"{api_base}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()

                # Claude response: content is a list of blocks
                raw_output = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        raw_output += block["text"]

                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                try:
                    if graph_format == "code":
                        parsed = parse(raw_output)
                    else:
                        parsed = parse_format(raw_output, graph_format)
                    graph = parsed.graph
                    if parsed.parse_type == "GraphEdit" and instruction.base_graph:
                        base_parsed = parse(instruction.base_graph)
                        if base_parsed.graph is not None:
                            graph = apply_graph_edit(
                                base_parsed.graph,
                                parsed.add_edges,
                                parsed.remove_edges,
                                task_type=_extract_task_type(
                                    instruction.explicit_constraints
                                ),
                            )
                except Exception:
                    graph = None
                break
            except requests.exceptions.RequestException as e:
                if _is_retriable(e) and attempt < max_retries - 1:
                    wait = min(2**attempt * 5, 60) + _random.uniform(0, 2)
                    time.sleep(wait)
                    continue
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break
            except Exception as e:
                raw_output = f"ERROR: {e}"
                input_tokens = 0
                output_tokens = 0
                graph = None
                break

        results.append(
            GenerationResult(
                instruction=instruction,
                graph=graph,
                raw_output=raw_output,
                token_record=TokenRecord(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    api_calls=api_calls,
                    valid=graph is not None,
                ),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROVIDERS = {
    "mock": generate_mock,
    "vllm": generate_vllm,
    "openai": generate_openai,
    "claude": generate_claude,
}


def serialize_benchmark_result(result: BenchmarkResult) -> dict:
    """Convert BenchmarkResult to JSON-serializable dict."""
    return {
        "per_instruction": [
            {
                "instruction_id": r.instruction_id,
                "level": r.level,
                "d1_valid_rate": r.d1_valid_rate,
                "d4_instruction_score": r.d4_instruction_score,
                "d5_tokens_per_valid": (
                    r.d5_tokens_per_valid
                    if r.d5_tokens_per_valid != float("inf")
                    else None
                ),
                "dimension_scores": r.dimension_scores,
                "level_score": r.level_score,
            }
            for r in result.per_instruction
        ],
        "per_level_scores": {str(k): v for k, v in result.per_level_scores.items()},
        "total_score": result.total_score,
        "final_score": result.final_score,
    }


# ---------------------------------------------------------------------------
# JSONL checkpoint: incremental save + resume
# ---------------------------------------------------------------------------


def _serialize_generation_result(r: GenerationResult) -> dict:
    """Serialize a single GenerationResult to a JSON-safe dict.

    Preserves raw_output for post-hoc analysis and graph_serialized for
    reliable round-trip recovery (raw_output may not re-parse for mock/edge cases).
    """
    graph_serialized = None
    if r.graph is not None:
        try:
            graph_serialized = serialize_format(r.graph, fmt="code")
        except Exception:
            pass
    return {
        "raw_output": r.raw_output,
        "graph_serialized": graph_serialized,
        "graph_nodes": (r.graph.number_of_nodes() if r.graph is not None else None),
        "graph_edges": (r.graph.number_of_edges() if r.graph is not None else None),
        "valid": r.graph is not None,
        "input_tokens": r.token_record.input_tokens,
        "output_tokens": r.token_record.output_tokens,
        "api_calls": r.token_record.api_calls,
    }


def _write_checkpoint_line(
    jsonl_path: Path,
    inst: "Instruction",
    gen_results: list[GenerationResult],
    method_metadata: dict | None = None,
) -> None:
    """Append one JSONL line for a completed instruction.

    Uses flush + fsync to ensure data reaches disk before returning,
    so a crash after this call never loses the written record.

    ``method_metadata`` is optional; when empty the JSONL schema remains
    bit-for-bit identical to pre-improvement baseline files.
    """
    import os as _os

    record = {
        "instruction_id": inst.id,
        "level": inst.level,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples": [_serialize_generation_result(r) for r in gen_results],
    }
    if method_metadata:
        record["method_metadata"] = method_metadata
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        _os.fsync(f.fileno())


def _load_checkpoint(jsonl_path: Path) -> tuple[set[str], list[dict]]:
    """Load existing checkpoint, return (completed_ids, raw_records).

    Supports two formats:
    1. JSONL checkpoint (one record per line, each with instruction_id + samples)
    2. JSON results file with raw_generations (fallback when JSONL was overwritten)

    Returns:
        completed_ids: Set of instruction IDs already finished.
        raw_records: List of raw dicts for rebuilding GenerationResults.
    """
    completed: set[str] = set()
    records: list[dict] = []
    if not jsonl_path.exists():
        return completed, records

    # Try to detect if file is a JSON results file (not JSONL checkpoint).
    # A JSON results file has both "per_instruction" and "raw_generations" keys.
    if jsonl_path.suffix == ".json":
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and "per_instruction" in data
                and "raw_generations" in data
            ):
                print(f"Detected JSON results file with raw_generations, recovering...")
                for inst_id, samples in data["raw_generations"].items():
                    level_str = inst_id.split("-")[0]  # "L0" -> 0
                    level_num = int(level_str[1:]) if level_str.startswith("L") else 0
                    record = {
                        "instruction_id": inst_id,
                        "level": level_num,
                        "samples": samples,
                    }
                    completed.add(inst_id)
                    records.append(record)
                print(f"Recovered {len(records)} instructions from JSON results")
                return completed, records
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Standard JSONL loading
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                inst_id = record["instruction_id"]
                if inst_id in completed:
                    print(
                        f"WARNING: duplicate instruction {inst_id} at line {line_no}, keeping latest"
                    )
                    # Remove earlier record, keep latest
                    records = [r for r in records if r["instruction_id"] != inst_id]
                completed.add(inst_id)
                records.append(record)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"WARNING: skipping corrupt checkpoint line {line_no}: {e}")
    return completed, records


def _rebuild_generation_results(
    records: list[dict],
    all_instructions: dict[int, list["Instruction"]],
) -> dict[str, list[GenerationResult]]:
    """Rebuild GenerationResult objects from checkpoint records.

    Re-parses raw_output to recover NetworkX graphs.
    """
    # Build instruction lookup
    inst_lookup: dict[str, "Instruction"] = {}
    for insts in all_instructions.values():
        for inst in insts:
            inst_lookup[inst.id] = inst

    all_results: dict[str, list[GenerationResult]] = {}
    for record in records:
        inst_id = record["instruction_id"]
        inst = inst_lookup.get(inst_id)
        if inst is None:
            continue
        gen_list: list[GenerationResult] = []
        for sample in record.get("samples", []):
            # Recover graph from checkpoint.
            # L5 GraphEdit: ALWAYS re-parse from raw_output so apply_graph_edit
            # (with prune_tree fix) runs fresh. graph_serialized from older
            # checkpoints may contain stale graphs without the fix.
            # Other levels: prefer graph_serialized (reliable round-trip).
            graph = None
            serialized = sample.get("graph_serialized")
            raw = sample.get("raw_output", "")
            if inst.level != 5 and serialized:
                try:
                    parsed = parse(serialized)
                    graph = parsed.graph
                except Exception:
                    pass
            if graph is None and raw and not raw.startswith("ERROR:"):
                try:
                    parsed = parse(raw)
                    graph = parsed.graph
                    # L5: apply GraphEdit operations to base graph
                    if (
                        inst.level == 5
                        and parsed.parse_type == "GraphEdit"
                        and inst.base_graph
                    ):
                        base_parsed = parse(inst.base_graph)
                        if base_parsed.graph is not None:
                            graph = apply_graph_edit(
                                base_parsed.graph,
                                parsed.add_edges,
                                parsed.remove_edges,
                                task_type=_extract_task_type(inst.explicit_constraints),
                            )
                except Exception:
                    pass
            gen_list.append(
                GenerationResult(
                    instruction=inst,
                    graph=graph,
                    raw_output=raw,
                    token_record=TokenRecord(
                        input_tokens=sample.get("input_tokens", 0),
                        output_tokens=sample.get("output_tokens", 0),
                        api_calls=sample.get("api_calls", 1),
                        valid=graph is not None,
                    ),
                )
            )
        if gen_list:
            all_results[inst_id] = gen_list
    return all_results


def main():
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Instruction levels to evaluate (default: all 6 levels)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing level_X.json files (default: data/instructions). "
        "Set to e.g. data/instructions/pilot to run on a pilot subset.",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="mock",
        help="Generation provider",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of generation samples per instruction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL for vLLM/OpenAI provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Llama-2-13b-chat-hf",
        help="Model name for API provider",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens. If unset, auto-selects the paper setting per "
        "model (gpt-3.5-turbo=4096; all other models=16384). Override only if "
        "you know what you're doing — mismatched values produce truncated "
        "outputs and leaderboard divergence.",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="zero-shot",
        help="Prompt strategy (default: zero-shot)",
    )
    parser.add_argument(
        "--format",
        choices=list(FORMATS),
        default="code",
        dest="graph_format",
        help="Graph representation format (default: code)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="AUTO",
        default=None,
        help="Resume from checkpoint. Use --resume (auto-detect from --output) "
        "or --resume PATH to specify a .jsonl checkpoint file explicitly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible few-shot selection (default: 42)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of instructions to process in parallel (default: 1, serial)",
    )
    parser.add_argument(
        "--enable-ged",
        action="store_true",
        default=False,
        help="Enable GED diagnostic computation (O(n!), very slow on large graphs)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Skip evaluation after generation (use eval_checkpoint.py separately)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls within each instruction (default: 0)",
    )
    parser.add_argument(
        "--merge-system-prompt",
        action="store_true",
        default=False,
        help="Merge system prompt into user message (workaround for API relays "
        "that silently drop system messages)",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Send enable_thinking=false to disable thinking/reasoning mode "
        "(for Qwen3.5, QwQ, etc.)",
    )
    # ----- Improvement methods (graphinstruct/improvements/) -----
    parser.add_argument(
        "--method",
        choices=["baseline", "retry", "sc", "vgig", "caap", "combined"],
        default="baseline",
        help="Generation method. 'baseline' preserves the original single-pass "
        "behaviour (default). Others wrap the provider with an improvement "
        "runner: retry=E2a, sc=E2b, vgig=E2c, caap=E3, combined=E4.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="T: max iteration rounds for retry/vgig (default: 3).",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="N: number of independent candidates for self-consistency (default: 3).",
    )
    parser.add_argument(
        "--feedback-level",
        choices=["none", "coarse", "fine"],
        default="fine",
        help="VGIG feedback granularity for E6 ablation (default: fine).",
    )
    args = parser.parse_args()

    # Auto-select per-model max_tokens to match the paper's experimental setup.
    # gpt-3.5-turbo: 4096 (API hard limit). All other models: 16384.
    _paper_max_tokens = _resolve_max_tokens(args.model)
    if args.max_tokens is None:
        args.max_tokens = _paper_max_tokens
        print(
            f"[INFO] Auto-selected max_tokens={args.max_tokens} for model "
            f"'{args.model}' (paper setting)."
        )
    elif args.max_tokens != _paper_max_tokens:
        print(
            f"[WARN] --max-tokens={args.max_tokens} differs from the paper "
            f"setting ({_paper_max_tokens}) for model '{args.model}'. "
            f"Results may not match the published leaderboard."
        )

    # Provider-specific api_base defaults: override only when user did NOT
    # supply --api-base explicitly (i.e. it still equals the argparse default).
    _PROVIDER_API_DEFAULTS = {
        "openai": "https://api.openai.com/v1",
        "claude": "https://api.anthropic.com",
        "vllm": "http://localhost:8000/v1",
    }
    _argparse_default_base = "http://localhost:8000/v1"
    if (
        args.api_base == _argparse_default_base
        and args.provider in _PROVIDER_API_DEFAULTS
    ):
        args.api_base = _PROVIDER_API_DEFAULTS[args.provider]

    # Set random seed for reproducibility (affects few-shot example selection)
    _random.seed(args.seed)

    # Load instructions
    all_instructions = load_all_levels(args.levels, args.data_dir)
    total = sum(len(insts) for insts in all_instructions.values())
    print(f"Loaded {total} instructions across levels {args.levels}")
    print(
        f"Provider: {args.provider} | Strategy: {args.strategy} | Format: {args.graph_format}"
    )

    # Determine checkpoint path (sibling to output, same stem + .jsonl)
    output_path = Path(args.output)
    # Force output to .json to prevent overwriting JSONL checkpoint
    if output_path.suffix == ".jsonl":
        output_path = output_path.with_suffix(".json")
        print(f"NOTE: Output changed to {output_path} to avoid overwriting checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume is not None and args.resume != "AUTO":
        resume_path = Path(args.resume)
        # Load from user-specified path (may be .json or .jsonl)
        completed_ids, checkpoint_records = _load_checkpoint(resume_path)
        # Always write new checkpoint lines to .jsonl (never append to .json)
        jsonl_path = resume_path.with_suffix(".jsonl")
    elif args.resume == "AUTO":
        jsonl_path = output_path.with_suffix(".jsonl")
        # Auto-detect: load existing checkpoint if any
        completed_ids, checkpoint_records = _load_checkpoint(jsonl_path)
    else:
        # No --resume: fresh run, do NOT load old checkpoint
        jsonl_path = output_path.with_suffix(".jsonl")
        completed_ids, checkpoint_records = set(), []
        # Truncate any stale checkpoint from a previous run
        if jsonl_path.exists():
            jsonl_path.unlink()
    if completed_ids:
        print(
            f"Resuming: {len(completed_ids)} instructions already completed in {jsonl_path}"
        )

    # Rebuild results from checkpoint
    all_results = _rebuild_generation_results(checkpoint_records, all_instructions)

    # Flatten instruction list for tqdm
    flat_instructions: list["Instruction"] = []
    for _level, instructions in sorted(all_instructions.items()):
        flat_instructions.extend(instructions)

    # Running stats for progress bar
    total_tokens_used = sum(
        s.get("input_tokens", 0) + s.get("output_tokens", 0)
        for rec in checkpoint_records
        for s in rec.get("samples", [])
    )
    total_valid = sum(
        1
        for rec in checkpoint_records
        for s in rec.get("samples", [])
        if s.get("valid", False)
    )
    total_generated = sum(len(rec.get("samples", [])) for rec in checkpoint_records)

    # Generate with progress bar
    provider_fn = PROVIDERS[args.provider]

    # ----- Improvement method dispatch (method != baseline) -----
    runner = None
    if args.method != "baseline":
        from graphinstruct.improvements import (
            CAAPRunner,
            CombinedRunner,
            RetryRunner,
            RunnerConfig,
            SCRunner,
            VGIGRunner,
        )

        _RUNNER_CLS = {
            "retry": RetryRunner,
            "sc": SCRunner,
            "vgig": VGIGRunner,
            "caap": CAAPRunner,
            "combined": CombinedRunner,
        }

        def _runner_generate_fn(
            instruction,
            *,
            num_samples=1,
            strategy=None,
            graph_format=None,
            temperature=None,
            override_prompt=None,
            **extra,
        ):
            """Wrap provider_fn with a standard signature for runners."""
            kwargs = {
                "num_samples": num_samples,
                "strategy": strategy if strategy is not None else args.strategy,
                "all_instructions": all_instructions,
                "graph_format": (
                    graph_format if graph_format is not None else args.graph_format
                ),
                "temperature": (
                    temperature if temperature is not None else args.temperature
                ),
                "override_prompt": override_prompt,
            }
            if args.provider in ("vllm", "openai", "claude"):
                kwargs["api_base"] = args.api_base
                kwargs["model"] = args.model
                kwargs["top_p"] = args.top_p
                kwargs["max_tokens"] = args.max_tokens
                kwargs["delay"] = args.delay
            if args.provider == "openai":
                if args.merge_system_prompt:
                    kwargs["merge_system_prompt"] = True
                if args.disable_thinking:
                    kwargs["disable_thinking"] = True
            kwargs.update(extra)
            return provider_fn(instruction, **kwargs)

        runner_config = RunnerConfig(
            method=args.method,
            max_rounds=args.max_rounds,
            num_candidates=args.num_candidates,
            feedback_level=args.feedback_level,
            graph_format=args.graph_format,
            temperature=args.temperature,
            num_samples=args.samples,
            extras={"strategy": args.strategy},
        )
        runner = _RUNNER_CLS[args.method](
            runner_config,
            generate_fn=_runner_generate_fn,
            all_instructions=all_instructions,
        )
        print(
            f"Method: {args.method} | max_rounds={args.max_rounds} "
            f"| num_candidates={args.num_candidates} "
            f"| feedback={args.feedback_level}"
        )

    remaining = [inst for inst in flat_instructions if inst.id not in completed_ids]

    def _process_one(
        inst: "Instruction",
    ) -> tuple["Instruction", list[GenerationResult], dict]:
        """Process a single instruction (may run in a worker thread)."""
        if runner is not None:
            result = runner.run_instruction(inst, model_name=args.model)
            return inst, list(result.samples), dict(result.method_metadata)
        if args.provider in ("vllm", "openai", "claude"):
            kwargs = dict(
                num_samples=args.samples,
                api_base=args.api_base,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                strategy=args.strategy,
                all_instructions=all_instructions,
                graph_format=args.graph_format,
                delay=args.delay,
            )
            if args.provider == "openai" and args.merge_system_prompt:
                kwargs["merge_system_prompt"] = True
            if args.provider == "openai" and args.disable_thinking:
                kwargs["disable_thinking"] = True
            gen_results = provider_fn(inst, **kwargs)
        else:
            gen_results = provider_fn(inst, num_samples=args.samples)
        return inst, gen_results, {}

    def _update_stats(inst, gen_results, method_metadata=None):
        """Update running stats and checkpoint. MUST be called from main thread only."""
        nonlocal total_tokens_used, total_valid, total_generated
        _write_checkpoint_line(jsonl_path, inst, gen_results, method_metadata)
        all_results[inst.id] = gen_results

        batch_tokens = sum(r.token_record.total_tokens for r in gen_results)
        batch_valid = sum(1 for r in gen_results if r.graph is not None)
        total_tokens_used += batch_tokens
        total_valid += batch_valid
        total_generated += len(gen_results)
        vr = total_valid / total_generated if total_generated > 0 else 0.0
        return vr

    concurrency = args.concurrency
    if concurrency <= 1:
        # Serial mode (original behavior)
        pbar = tqdm(
            remaining,
            desc="Generating",
            unit="inst",
            ncols=120,
            initial=0,
            total=len(remaining),
        )
        for inst in pbar:
            _, gen_results, method_metadata = _process_one(inst)
            vr = _update_stats(inst, gen_results, method_metadata)
            pbar.set_postfix(
                tokens=f"{total_tokens_used:,}",
                valid=f"{vr:.1%}",
                inst=inst.id,
                refresh=False,
            )
        pbar.close()
    else:
        # Concurrent mode — instruction-level parallelism
        print(f"Concurrency: {concurrency} workers")
        failed_count = 0
        pbar = tqdm(
            total=len(remaining),
            desc="Generating",
            unit="inst",
            ncols=120,
        )
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_process_one, inst): inst for inst in remaining}
            for future in as_completed(futures):
                try:
                    inst, gen_results, method_metadata = future.result()
                except Exception as e:
                    inst = futures[future]
                    print(f"\nERROR processing {inst.id}: {e}")
                    failed_count += 1
                    pbar.update(1)
                    continue
                vr = _update_stats(inst, gen_results, method_metadata)
                pbar.update(1)
                pbar.set_postfix(
                    tokens=f"{total_tokens_used:,}",
                    valid=f"{vr:.1%}",
                    inst=inst.id,
                    refresh=False,
                )
        pbar.close()
        if failed_count > 0:
            print(
                f"WARNING: {failed_count} instructions failed, "
                f"use --resume to retry them"
            )
    if completed_ids:
        print(f"Resumed: {len(completed_ids)} previously completed")
    print(f"Checkpoint saved to {jsonl_path}")
    print(
        f"Total tokens: {total_tokens_used:,} | Valid rate: {total_valid}/{total_generated}"
    )

    if args.skip_eval:
        print("\n--skip-eval: skipping evaluation. Run eval_checkpoint.py separately:")
        print(f"  python scripts/eval_checkpoint.py {jsonl_path}")
        return

    # Evaluate
    print("\n=== Evaluation ===")
    benchmark = evaluate_benchmark(
        all_results, enable_ged=args.enable_ged, strategy=args.strategy
    )

    for eval_level, score in sorted(benchmark.per_level_scores.items()):
        print(f"  Level {eval_level} score: {score:.4f}")
    print(f"  Total score: {benchmark.total_score:.4f}")
    print(f"  Final score: {benchmark.final_score:.4f}")

    # Save final results JSON (includes raw_output for recovery)
    result_data = serialize_benchmark_result(benchmark)
    result_data["raw_generations"] = {
        inst_id: [_serialize_generation_result(r) for r in gens]
        for inst_id, gens in all_results.items()
    }
    result_data["metadata"] = {
        "provider": args.provider,
        "model": getattr(args, "model", None),
        "strategy": args.strategy,
        "graph_format": args.graph_format,
        "samples": args.samples,
        "levels": args.levels,
        "seed": args.seed,
        "total_tokens": total_tokens_used,
        "checkpoint": str(jsonl_path),
        "method": args.method,
        "max_rounds": args.max_rounds if args.method != "baseline" else None,
        "num_candidates": (args.num_candidates if args.method != "baseline" else None),
        "feedback_level": (args.feedback_level if args.method != "baseline" else None),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    # Derive quality-only (D1-D4) scores from the combined result,
    # mirroring eval_checkpoint.py --both behaviour.
    quality_path = output_path.with_name(output_path.stem + ".quality.json")
    quality_insts = []
    level_buckets: dict[int, list[float]] = {}
    for inst in result_data["per_instruction"]:
        q_ls = level_score(inst["level"], inst["dimension_scores"], quality_only=True)
        quality_insts.append({**inst, "level_score": q_ls})
        level_buckets.setdefault(inst["level"], []).append(q_ls)

    q_per_level = {lv: sum(s) / len(s) for lv, s in level_buckets.items()}
    if q_per_level:
        q_level_list = [q_per_level.get(lv, 0.0) for lv in range(max(q_per_level) + 1)]
        q_total = total_score(q_level_list)
    else:
        q_total = 0.0

    quality_result = {
        "per_instruction": quality_insts,
        "per_level_scores": {str(k): v for k, v in q_per_level.items()},
        "total_score": q_total,
        "final_score": q_total,
        "metadata": {
            **result_data.get("metadata", {}),
            "eval_mode": "quality-only",
            "derived_from": "combined",
        },
    }
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_result, f, indent=2, ensure_ascii=False)
    print(f"\n=== Quality-only (derived) ===")
    for lv in sorted(q_per_level):
        print(f"  Level {lv}: {q_per_level[lv]:.4f}")
    print(f"  Total score: {q_total:.4f}")
    print(f"Saved to {quality_path}")


if __name__ == "__main__":
    main()
