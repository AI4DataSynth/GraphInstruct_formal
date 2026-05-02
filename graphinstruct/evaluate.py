"""Evaluation pipeline: run metrics on generated graphs against instructions.

Core pipeline:
1. Load instructions from data/instructions/
2. For each instruction + generated graph pair, compute D1/D4/D5 scores
3. Aggregate into level_score -> total_score -> final_score
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import networkx as nx

from graphinstruct.data_loader import Instruction

from graphinstruct.metrics.efficiency import TokenRecord
from graphinstruct.metrics.embedding import d3_score, d3_score_batch
from graphinstruct.metrics.instruction import instruction_score
from graphinstruct.metrics.l4_eval import instruction_score_l4
from graphinstruct.parser.code_style import recover_directed_edges
from graphinstruct.metrics.l5_eval import instruction_score_l5
from graphinstruct.metrics.micro_eval import MicroEvalResult, evaluate_reasoning_chain
from graphinstruct.metrics.structural import d1_combined_score, valid_rate_single
from graphinstruct.metrics.textual import d2_score
from graphinstruct.analysis.pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    pareto_bonus,
)
from graphinstruct.scoring import (
    final_score,
    level_score,
    total_score,
)


@dataclass(frozen=True)
class GenerationResult:
    """Result of a single graph generation attempt.

    Attributes:
        instruction: The instruction that was given.
        graph: The generated graph, or None if parsing failed.
        raw_output: The raw LLM output string.
        token_record: Token usage for this generation.
    """

    instruction: Instruction
    graph: Optional[nx.Graph]
    raw_output: str
    token_record: TokenRecord


@dataclass
class InstructionEvalResult:
    """Evaluation result for a single instruction across multiple generations."""

    instruction_id: str
    level: int
    d1_valid_rate: float
    d4_instruction_score: float
    d5_tokens_per_valid: float
    dimension_scores: dict[str, float]
    level_score: float
    micro_eval: Optional[MicroEvalResult] = None


def evaluate_instruction(
    results: Sequence[GenerationResult],
    enable_ged: bool = False,
    quality_only: bool = False,
    strategy: Optional[str] = None,
) -> InstructionEvalResult:
    """Evaluate all generation results for a single instruction.

    Args:
        results: List of GenerationResult for the same instruction.
            All results must share the same instruction.

    Returns:
        InstructionEvalResult with aggregated scores.
    """
    if not results:
        raise ValueError("No results to evaluate")

    inst = results[0].instruction
    level = inst.level

    # Handle infeasible instructions: LLM correctly refusing = full score
    if not inst.feasible:
        all_none = all(r.graph is None for r in results)
        if all_none:
            # LLM correctly identified infeasible instruction
            return InstructionEvalResult(
                instruction_id=inst.id,
                level=level,
                d1_valid_rate=1.0,
                d4_instruction_score=1.0,
                d5_tokens_per_valid=0.0,
                dimension_scores={
                    "D1": 1.0,
                    "D2": 0.0,
                    "D3": 0.0,
                    "D4": 1.0,
                    "D5": 1.0,
                },
                level_score=level_score(
                    level, {"D1": 1.0, "D2": 0.0, "D3": 0.0, "D4": 1.0, "D5": 1.0}
                ),
            )
        # LLM generated graphs for infeasible instruction — penalize
        return InstructionEvalResult(
            instruction_id=inst.id,
            level=level,
            d1_valid_rate=0.0,
            d4_instruction_score=0.0,
            d5_tokens_per_valid=float("inf"),
            dimension_scores={"D1": 0.0, "D2": 0.0, "D3": 0.0, "D4": 0.0, "D5": 0.0},
            level_score=0.0,
        )

    # D1: Structural quality (Valid Rate + GED + MMD)
    graphs = [r.graph for r in results]

    # Phase 0b: fix directed attribute for ALL levels.
    # When instruction requires directed=true but parser created undirected graph
    # (missing header attribute), recover edge direction from raw_output.
    # Applied before D1 so valid_rate also benefits from the fix.
    _has_directed = any(c.strip() == "directed=true" for c in inst.explicit_constraints)
    if _has_directed:
        fixed_graphs = []
        for g, r in zip(graphs, results):
            if g is not None and not g.is_directed():
                g, _ = recover_directed_edges(g, r.raw_output)
            fixed_graphs.append(g)
        graphs = fixed_graphs

    # Determine graph_type from explicit constraints
    graph_type = _extract_graph_type(inst.explicit_constraints)
    # Constraints excluding graph_type (already checked by validator)
    other_constraints = [
        c for c in inst.explicit_constraints if not c.startswith("graph_type=")
    ]

    # Build reference graphs for GED/MMD comparison
    ref_graphs = _build_reference_graphs(inst)

    # Compute all D1 sub-metrics via d1_combined_score
    d1_result = d1_combined_score(
        graphs,
        ref_graphs,
        level,
        graph_type=graph_type,
        constraints=other_constraints,
        enable_ged=enable_ged,
    )
    d1_score = d1_result["combined"]

    # Build list of D1-valid (graph, result) pairs — preserves alignment
    valid_pairs: list[tuple[nx.Graph, GenerationResult]] = [
        (g, r)
        for g, r in zip(graphs, results)
        if valid_rate_single(g, graph_type=graph_type, constraints=other_constraints)
    ]
    valid_graphs = [g for g, _ in valid_pairs]
    valid_count = len(valid_graphs)

    # D4: Instruction Score (average across D1-valid graphs only)
    # Level-aware scoring: L4 uses domain_structural_fit, L5 uses edit_quality
    d4_scores = []
    if level == 4:
        # L4: D4 = 0.65*constraint_sat + 0.20*domain_structural_fit + 0.15*no_contradiction
        # Note: directed fix already applied globally above
        for g, _r in valid_pairs:
            d4_scores.append(
                instruction_score_l4(
                    g,
                    list(inst.explicit_constraints),
                    reference_graphs=ref_graphs,
                )
            )
    elif level == 5:
        # L5: D4 = 0.45*edit_quality + 0.40*constraint_sat + 0.15*reasoning_alignment
        base_graph = _parse_base_graph(inst)
        for g, r in valid_pairs:
            parsed_out = _try_parse(r.raw_output)
            d4_scores.append(
                instruction_score_l5(
                    g,
                    list(inst.explicit_constraints),
                    base_graph=base_graph,
                    reference_graphs=ref_graphs,
                    raw_output=r.raw_output,
                    parsed_output=parsed_out,
                    strategy=strategy,
                )
            )
    else:
        # L0-L3: original scoring
        for g in valid_graphs:
            d4_scores.append(instruction_score(g, list(inst.explicit_constraints)))
    d4_score = sum(d4_scores) / len(d4_scores) if d4_scores else 0.0

    # D5: Token efficiency (Token/Valid Graph + API Calls/Graph)
    # Use constraint-validated valid_count consistently for both sub-metrics
    total_tokens = sum(r.token_record.total_tokens for r in results)
    total_api_calls = sum(r.token_record.api_calls for r in results)
    if valid_count > 0:
        d5_tpv = total_tokens / valid_count
        d5_api = total_api_calls / valid_count
    else:
        d5_tpv = float("inf")
        d5_api = float("inf")

    # Token efficiency: exponential decay normalization
    if d5_tpv == float("inf"):
        d5_token_score = 0.0
    elif d5_tpv < 0:
        d5_token_score = 0.0  # defensive: should never happen
    else:
        d5_token_score = math.exp(-d5_tpv / 1000.0)  # TPV=0 -> 1.0 (optimal)

    # API call efficiency: exponential decay (1 call optimal)
    if d5_api == float("inf"):
        d5_api_score = 0.0
    elif d5_api < 1:
        d5_api_score = 1.0  # fewer than 1 call means optimal
    else:
        d5_api_score = math.exp(-(d5_api - 1.0) / 2.0)  # 1 call -> 1.0, 3 calls -> 0.37

    # Combined D5: weighted average (token efficiency 0.7, API efficiency 0.3)
    d5_score = 0.7 * d5_token_score + 0.3 * d5_api_score

    # D2: Text quality (active for L4+ only; L3 D2=0.00 — synthetic graphs lack text)
    # D3: Embedding quality (active for L3+)
    d2_val = 0.0
    d3_val = 0.0
    if level >= 3:
        ref_graph = ref_graphs[0] if ref_graphs else None
        if valid_graphs and ref_graph is not None:
            # D3: active for L3+ — batch mode reuses ref GCN (A2)
            d3_results = d3_score_batch(ref_graph, valid_graphs)
            d3_agg = [r["aggregate"] for r in d3_results]
            d3_val = sum(d3_agg) / len(d3_agg) if d3_agg else 0.0
            # D2: active for L4+ only (not L3)
            if level >= 4:
                d2_agg = []
                for vg in valid_graphs:
                    d2_agg.append(d2_score(ref_graph, vg, use_bert=True)["aggregate"])
                d2_val = sum(d2_agg) / len(d2_agg)

    # L5 micro evaluation: GRAPHIA-style step-by-step reasoning assessment
    # Retained as diagnostic (stored in result), but no longer blended into D4.
    # D4 for L5 is now computed by instruction_score_l5 (edit_quality + constraint_sat).
    micro_result: Optional[MicroEvalResult] = None
    if level == 5:
        micro_result = _compute_micro_eval(inst, valid_graphs, ref_graphs)

    # Build dimension scores dict
    dim_scores = {
        "D1": d1_score,
        "D2": d2_val,
        "D3": d3_val,
        "D4": d4_score,
        "D5": d5_score,
    }

    # Compute level score using hierarchical weights
    l_score = level_score(level, dim_scores, quality_only=quality_only)

    return InstructionEvalResult(
        instruction_id=inst.id,
        level=level,
        d1_valid_rate=d1_result.get("valid_rate", 0.0),
        d4_instruction_score=d4_score,
        d5_tokens_per_valid=d5_tpv,
        dimension_scores=dim_scores,
        level_score=l_score,
        micro_eval=micro_result,
    )


@dataclass
class BenchmarkResult:
    """Complete benchmark evaluation result."""

    per_instruction: list[InstructionEvalResult]
    per_level_scores: dict[int, float]
    total_score: float
    final_score: float


def evaluate_benchmark(
    all_results: dict[str, Sequence[GenerationResult]],
    model_label: str = "default",
    other_models: Optional[list[dict]] = None,
    enable_ged: bool = False,
    quality_only: bool = False,
    strategy: Optional[str] = None,
) -> BenchmarkResult:
    """Evaluate the full benchmark across all instructions.

    Automatically computes Pareto bonus based on the model's position
    in the cost-quality space relative to other models.

    Args:
        all_results: Dict mapping instruction_id -> list of GenerationResult.
        model_label: Label for this model/method.
        other_models: Optional list of dicts with keys 'label', 'quality',
            'cost' for Pareto frontier computation against other models.
            If None, single-model mode: Pareto bonus = 1.0 (self is frontier).
        enable_ged: If True, compute GED diagnostic (O(n!), very slow on
            large graphs). Default False.

    Returns:
        BenchmarkResult with complete evaluation.
    """
    import time as _time

    # Group by level for per-level progress reporting
    by_level: dict[int, list[tuple[str, Sequence[GenerationResult]]]] = {}
    for inst_id, results in all_results.items():
        lvl = results[0].instruction.level
        by_level.setdefault(lvl, []).append((inst_id, results))

    per_instruction = []
    level_groups: dict[int, list[float]] = {}

    try:
        from tqdm import tqdm as _tqdm

        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    for lvl in sorted(by_level.keys()):
        items = by_level[lvl]
        t0 = _time.time()
        lvl_scores: list[float] = []
        iterator = (
            _tqdm(items, desc=f"  L{lvl}", unit="inst", ncols=80, leave=False)
            if _has_tqdm
            else items
        )
        for inst_id, results in iterator:
            eval_result = evaluate_instruction(
                results,
                enable_ged=enable_ged,
                quality_only=quality_only,
                strategy=strategy,
            )
            per_instruction.append(eval_result)
            lvl_scores.append(eval_result.level_score)
        elapsed = _time.time() - t0
        avg = sum(lvl_scores) / len(lvl_scores) if lvl_scores else 0.0
        level_groups[lvl] = lvl_scores
        print(
            f"  L{lvl}: {len(items)} instructions, " f"score={avg:.4f}, {elapsed:.1f}s",
            flush=True,
        )

    per_level_scores = {
        lvl: sum(scores) / len(scores) for lvl, scores in level_groups.items()
    }

    # Build ordered level scores for total_score computation
    if per_level_scores:
        ordered_scores = [
            per_level_scores.get(i, 0.0)
            for i in range(max(per_level_scores.keys()) + 1)
        ]
        s_total = total_score(ordered_scores)
    else:
        s_total = 0.0

    # Compute Pareto bonus
    total_cost = sum(
        r.token_record.total_tokens for results in all_results.values() for r in results
    )
    # Use constraint-validated d1_valid_rate (consistent with D5 computation)
    valid_rate = (
        sum(r.d1_valid_rate for r in per_instruction) / len(per_instruction)
        if per_instruction
        else 0.0
    )

    this_point = ParetoPoint(
        label=model_label,
        quality=s_total,
        cost=float(total_cost),
        valid_rate=valid_rate,
    )

    if other_models:
        all_points = [this_point] + [ParetoPoint(**m) for m in other_models]
    else:
        all_points = [this_point]

    frontier = compute_pareto_frontier(all_points)
    p_bonus = pareto_bonus(this_point, frontier)

    s_final = final_score(s_total, pareto_bonus=p_bonus)

    return BenchmarkResult(
        per_instruction=per_instruction,
        per_level_scores=per_level_scores,
        total_score=s_total,
        final_score=s_final,
    )


def _extract_graph_type(constraints: tuple[str, ...] | list[str]) -> Optional[str]:
    """Extract graph_type from explicit constraints."""
    for c in constraints:
        if c.startswith("graph_type="):
            return c.split("=", 1)[1]
    return None


def _build_reference_graphs(inst: Instruction) -> list[nx.Graph]:
    """Build reference graphs for GED diagnostic / MMD comparison.

    For L3/L4 (distribution type): loads from shared reference pool and
    filters by instruction constraints.  Falls back to instruction-level
    reference_solutions if pool is unavailable.

    For L0-L2/L5 (constraint type): uses instruction-level reference
    solutions (only needed for GED diagnostic on L1).
    """
    level = inst.level

    # L3/L4: try shared reference pool first
    if level in (3, 4):
        pool_graphs = _load_reference_pool(inst)
        if pool_graphs:
            return pool_graphs

    # All levels: fall back to instruction-level reference solutions
    from graphinstruct.parser.code_style import parse

    refs: list[nx.Graph] = []
    if inst.reference_solutions:
        for ref_str in inst.reference_solutions:
            try:
                parsed = parse(ref_str)
                if parsed.graph is not None and parsed.graph.number_of_nodes() > 0:
                    refs.append(parsed.graph)
            except Exception:
                continue
    if refs:
        return refs

    # Fallback: build a simple graph from constraints
    num_nodes = None
    for c in inst.explicit_constraints:
        if c.startswith("num_nodes="):
            try:
                num_nodes = int(c.split("=", 1)[1])
            except ValueError:
                pass
    if num_nodes is not None and num_nodes >= 2:
        refs.append(nx.path_graph(num_nodes))
    return refs


# Module-level cache: raw pool graphs by level (A1 — avoid repeated pickle I/O)
_POOL_CACHE: dict[int, list[nx.Graph]] = {}


def _load_reference_pool(inst: Instruction) -> list[nx.Graph]:
    """Load and filter shared reference pool for distribution-type levels.

    Looks for pickled graph lists in data/reference_pools/.
    Filters pool graphs by instruction constraints to produce a
    per-instruction reference subset for MMD computation.

    Raw pool graphs are cached by level so pickle I/O happens only once.

    Returns empty list if pool directory doesn't exist yet (Phase 2b).
    """
    from graphinstruct.validators import check_constraint

    level = inst.level
    if level not in (3, 4):
        return []

    # Load raw pool once per level, cache for subsequent instructions
    if level not in _POOL_CACHE:
        import pickle
        from pathlib import Path

        pool_dir = Path(__file__).resolve().parent.parent / "data" / "reference_pools"
        pool_path = pool_dir / ("l3_synthetic" if level == 3 else "l4_real")

        if not pool_path.exists():
            return []

        all_graphs: list[nx.Graph] = []
        for pkl_file in sorted(pool_path.glob("*.pkl")):
            try:
                with open(pkl_file, "rb") as f:
                    graphs = pickle.load(f)
                if isinstance(graphs, list):
                    all_graphs.extend(g for g in graphs if isinstance(g, nx.Graph))
            except Exception:
                continue
        _POOL_CACHE[level] = all_graphs

    all_graphs = _POOL_CACHE[level]
    if not all_graphs:
        return []

    # Filter by instruction constraints (per-instruction, not cached).
    # Return copies so downstream code cannot mutate cached originals.
    constraints = list(inst.explicit_constraints)
    return [
        g.copy() for g in all_graphs if all(check_constraint(g, c) for c in constraints)
    ]


def _parse_base_graph(inst: Instruction) -> Optional[nx.Graph]:
    """Parse the base graph for L5 instructions."""
    from graphinstruct.parser.code_style import parse

    if not inst.base_graph:
        return None
    try:
        parsed = parse(inst.base_graph)
        return parsed.graph if parsed.graph is not None else None
    except Exception:
        return None


def _try_parse(raw_output: str) -> Optional[object]:
    """Try to parse raw LLM output, returning ParseResult or None."""
    from graphinstruct.parser.code_style import parse

    try:
        return parse(raw_output)
    except Exception:
        return None


def _compute_micro_eval(
    inst: Instruction,
    valid_graphs: list[nx.Graph],
    ref_graphs: list[nx.Graph],
) -> Optional[MicroEvalResult]:
    """Compute GRAPHIA micro evaluation for L5 instructions.

    Parses the base_graph, evaluates each valid generated graph against
    all reference solutions, and averages the micro eval results.

    Args:
        inst: The L5 instruction.
        valid_graphs: D1-valid generated graphs.
        ref_graphs: Parsed reference solution graphs.

    Returns:
        Averaged MicroEvalResult, or None if evaluation not possible.
    """
    from graphinstruct.parser.code_style import parse

    if not valid_graphs or not ref_graphs:
        return None

    # Parse base graph
    base_graph: Optional[nx.Graph] = None
    if inst.base_graph:
        try:
            parsed = parse(inst.base_graph)
            if parsed.graph is not None:
                base_graph = parsed.graph
        except Exception:
            pass

    # Build instruction dict for decompose_instruction
    inst_dict = {
        "id": inst.id,
        "instruction": inst.instruction,
    }

    # Evaluate each valid generated graph
    results: list[MicroEvalResult] = []
    for gen_graph in valid_graphs:
        result = evaluate_reasoning_chain(inst_dict, gen_graph, base_graph, ref_graphs)
        results.append(result)

    if not results:
        return None

    # Average across all valid generated graphs
    avg_sel = sum(r.s_sel for r in results) / len(results)
    avg_edge = sum(r.s_edge for r in results) / len(results)
    avg_micro = sum(r.s_micro for r in results) / len(results)
    total_steps = results[0].steps_total  # Same decomposition for all
    avg_correct = sum(r.steps_correct for r in results) / len(results)

    return MicroEvalResult(
        s_sel=avg_sel,
        s_edge=avg_edge,
        s_micro=avg_micro,
        steps_total=total_steps,
        steps_correct=round(avg_correct),
        step_details=results[0].step_details,  # Use first result's details
    )
