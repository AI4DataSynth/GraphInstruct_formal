"""GRAPHIA-style micro evaluation for L5 multi-step reasoning.

Decomposes L5 instructions into reasoning steps (construct base graph,
apply operations, verify constraints) and evaluates each step:
- S_sel: Node selection correctness (did the model operate on correct nodes?)
- S_edge: Edge operation correctness (add/remove correct edges?)
- S_micro: Average of S_sel and S_edge

Reference: [P2] GRAPHIA micro-macro evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx


@dataclass(frozen=True)
class ReasoningStep:
    """A single step in multi-step graph reasoning.

    Attributes:
        step_index: 1-based step number.
        step_type: Operation type (construct, add_edge, remove_edge, etc.).
        description: Human-readable description of this step.
        target_nodes: Nodes involved in this step.
        target_edges: Edges involved in this step (as (u,v) tuples).
    """

    step_index: int
    step_type: str
    description: str
    target_nodes: Tuple[str, ...] = ()
    target_edges: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StepResult:
    """Evaluation result for a single reasoning step."""

    step_index: int
    step_type: str
    s_sel: float
    s_edge: float
    correct: bool
    detail: str = ""


@dataclass(frozen=True)
class MicroEvalResult:
    """Full micro evaluation result for an L5 instruction."""

    s_sel: float
    s_edge: float
    s_micro: float
    steps_total: int
    steps_correct: int
    step_details: Tuple[StepResult, ...] = ()


def _normalize_edge(u: str, v: str) -> Tuple[str, str]:
    """Normalize an undirected edge to canonical form (smaller first)."""
    return (min(u, v), max(u, v))


def _edge_set(graph: nx.Graph) -> frozenset[Tuple[str, str]]:
    """Extract normalized edge set from a graph."""
    return frozenset(_normalize_edge(str(u), str(v)) for u, v in graph.edges())


def _node_set(graph: nx.Graph) -> frozenset[str]:
    """Extract node set from a graph (as strings)."""
    return frozenset(str(n) for n in graph.nodes())


def decompose_instruction(
    instruction: Dict[str, Any],
    base_graph: Optional[nx.Graph] = None,
    ref_graph: Optional[nx.Graph] = None,
) -> List[ReasoningStep]:
    """Decompose an L5 instruction into reasoning steps.

    Uses base_graph and reference_solution to determine the exact
    operations needed (edge additions, removals, node changes).

    Args:
        instruction: Instruction dict with id, instruction text, etc.
        base_graph: Parsed base graph (starting state).
        ref_graph: Parsed reference solution (target state).

    Returns:
        List of ReasoningStep objects representing the decomposition.
    """
    steps: List[ReasoningStep] = []
    step_idx = 1

    # Step 1: Always start with constructing/understanding the base graph
    steps.append(
        ReasoningStep(
            step_index=step_idx,
            step_type="construct",
            description="Build/understand the initial base graph",
        )
    )
    step_idx += 1

    if base_graph is None or ref_graph is None:
        # Without both graphs, we can only produce the construct step
        steps.append(
            ReasoningStep(
                step_index=step_idx,
                step_type="verify",
                description="Verify final graph satisfies all constraints",
            )
        )
        return steps

    base_edges = _edge_set(base_graph)
    ref_edges = _edge_set(ref_graph)
    base_nodes = _node_set(base_graph)
    ref_nodes = _node_set(ref_graph)

    # Compute diffs
    removed_edges = base_edges - ref_edges
    added_edges = ref_edges - base_edges
    removed_nodes = base_nodes - ref_nodes
    added_nodes = ref_nodes - base_nodes

    # Step 2+: Node removals (do before edge ops since removing nodes
    # also removes their incident edges)
    if removed_nodes:
        steps.append(
            ReasoningStep(
                step_index=step_idx,
                step_type="remove_node",
                description=f"Remove {len(removed_nodes)} node(s)",
                target_nodes=tuple(sorted(removed_nodes)),
            )
        )
        step_idx += 1

    # Step N: Edge removals
    if removed_edges:
        steps.append(
            ReasoningStep(
                step_index=step_idx,
                step_type="remove_edge",
                description=f"Remove {len(removed_edges)} edge(s)",
                target_edges=tuple(sorted(removed_edges)),
            )
        )
        step_idx += 1

    # Step N: Node additions
    if added_nodes:
        steps.append(
            ReasoningStep(
                step_index=step_idx,
                step_type="add_node",
                description=f"Add {len(added_nodes)} node(s)",
                target_nodes=tuple(sorted(added_nodes)),
            )
        )
        step_idx += 1

    # Step N: Edge additions
    if added_edges:
        steps.append(
            ReasoningStep(
                step_index=step_idx,
                step_type="add_edge",
                description=f"Add {len(added_edges)} edge(s)",
                target_edges=tuple(sorted(added_edges)),
            )
        )
        step_idx += 1

    # Final step: Verify constraints
    steps.append(
        ReasoningStep(
            step_index=step_idx,
            step_type="verify",
            description="Verify final graph satisfies all constraints",
        )
    )

    return steps


def _evaluate_construct_step(
    generated: Optional[nx.Graph],
    base_graph: Optional[nx.Graph],
) -> StepResult:
    """Evaluate whether the base graph was correctly understood.

    We check if the generated graph at least preserves the base node set
    (LLM should understand the starting graph). For construct step,
    S_sel = 1 if generated graph is not None, S_edge = 1 if base nodes
    are a subset of generated nodes (or generated has correct node count).
    """
    if generated is None:
        return StepResult(
            step_index=1,
            step_type="construct",
            s_sel=0.0,
            s_edge=0.0,
            correct=False,
            detail="Generated graph is None (parse failure)",
        )

    if base_graph is None:
        # No base to compare — generous: give full score
        return StepResult(
            step_index=1,
            step_type="construct",
            s_sel=1.0,
            s_edge=1.0,
            correct=True,
            detail="No base graph for comparison",
        )

    # Check node overlap with base graph
    gen_nodes = _node_set(generated)
    base_nodes = _node_set(base_graph)

    if not base_nodes:
        return StepResult(
            step_index=1,
            step_type="construct",
            s_sel=1.0,
            s_edge=1.0,
            correct=True,
            detail="Empty base graph",
        )

    # S_sel: How many base nodes appear in generated graph
    overlap = base_nodes & gen_nodes
    s_sel = len(overlap) / len(base_nodes) if base_nodes else 1.0

    # S_edge: give full score for construct step (edges evaluated in later steps)
    s_edge = 1.0

    return StepResult(
        step_index=1,
        step_type="construct",
        s_sel=s_sel,
        s_edge=s_edge,
        correct=(s_sel >= 0.9),
        detail=f"Base nodes preserved: {len(overlap)}/{len(base_nodes)}",
    )


def _evaluate_edge_ops(
    step: ReasoningStep,
    generated: nx.Graph,
    base_graph: nx.Graph,
    ref_graph: nx.Graph,
) -> StepResult:
    """Evaluate edge add/remove operations.

    Compares the edge diff between (base -> generated) vs (base -> reference)
    to determine correctness.
    """
    gen_edges = _edge_set(generated)
    base_edges = _edge_set(base_graph)
    ref_edges = _edge_set(ref_graph)

    if step.step_type == "remove_edge":
        # Expected removals = edges in base but not in reference
        expected_removals = base_edges - ref_edges
        # Actual removals = edges in base but not in generated
        actual_removals = base_edges - gen_edges

        if not expected_removals:
            return StepResult(
                step_index=step.step_index,
                step_type=step.step_type,
                s_sel=1.0,
                s_edge=1.0,
                correct=True,
                detail="No edges to remove",
            )

        # S_edge: What fraction of expected removals were actually done?
        correct_removals = expected_removals & actual_removals
        s_edge = len(correct_removals) / len(expected_removals)

        # S_sel: Were the correct endpoint nodes identified?
        expected_nodes = set()
        for u, v in expected_removals:
            expected_nodes.add(u)
            expected_nodes.add(v)
        actual_nodes = set()
        for u, v in actual_removals:
            actual_nodes.add(u)
            actual_nodes.add(v)
        overlap = expected_nodes & actual_nodes
        s_sel = len(overlap) / len(expected_nodes) if expected_nodes else 1.0

        correct = correct_removals == expected_removals
        detail = (
            f"Removed {len(correct_removals)}/{len(expected_removals)} expected edges"
        )
        return StepResult(
            step_index=step.step_index,
            step_type=step.step_type,
            s_sel=s_sel,
            s_edge=s_edge,
            correct=correct,
            detail=detail,
        )

    elif step.step_type == "add_edge":
        # Expected additions = edges in reference but not in base
        expected_additions = ref_edges - base_edges
        # Actual additions = edges in generated but not in base
        actual_additions = gen_edges - base_edges

        if not expected_additions:
            return StepResult(
                step_index=step.step_index,
                step_type=step.step_type,
                s_sel=1.0,
                s_edge=1.0,
                correct=True,
                detail="No edges to add",
            )

        # S_edge: What fraction of expected additions were actually done?
        correct_additions = expected_additions & actual_additions
        s_edge = len(correct_additions) / len(expected_additions)

        # S_sel: Were the correct endpoint nodes identified?
        expected_nodes = set()
        for u, v in expected_additions:
            expected_nodes.add(u)
            expected_nodes.add(v)
        actual_nodes = set()
        for u, v in actual_additions:
            actual_nodes.add(u)
            actual_nodes.add(v)
        overlap = expected_nodes & actual_nodes
        s_sel = len(overlap) / len(expected_nodes) if expected_nodes else 1.0

        correct = correct_additions == expected_additions
        detail = (
            f"Added {len(correct_additions)}/{len(expected_additions)} expected edges"
        )
        return StepResult(
            step_index=step.step_index,
            step_type=step.step_type,
            s_sel=s_sel,
            s_edge=s_edge,
            correct=correct,
            detail=detail,
        )

    # Fallback for unknown edge-related step
    return StepResult(
        step_index=step.step_index,
        step_type=step.step_type,
        s_sel=0.0,
        s_edge=0.0,
        correct=False,
        detail=f"Unknown edge operation: {step.step_type}",
    )


def _evaluate_node_ops(
    step: ReasoningStep,
    generated: nx.Graph,
    base_graph: nx.Graph,
    ref_graph: nx.Graph,
) -> StepResult:
    """Evaluate node add/remove operations."""
    gen_nodes = _node_set(generated)
    base_nodes = _node_set(base_graph)
    ref_nodes = _node_set(ref_graph)

    if step.step_type == "remove_node":
        expected_removals = base_nodes - ref_nodes
        actual_removals = base_nodes - gen_nodes

        if not expected_removals:
            return StepResult(
                step_index=step.step_index,
                step_type=step.step_type,
                s_sel=1.0,
                s_edge=1.0,
                correct=True,
                detail="No nodes to remove",
            )

        correct_removals = expected_removals & actual_removals
        s_sel = len(correct_removals) / len(expected_removals)
        # S_edge = 1.0 for node ops (no edge evaluation needed)
        s_edge = 1.0

        correct = correct_removals == expected_removals
        detail = (
            f"Removed {len(correct_removals)}/{len(expected_removals)} expected nodes"
        )
        return StepResult(
            step_index=step.step_index,
            step_type=step.step_type,
            s_sel=s_sel,
            s_edge=s_edge,
            correct=correct,
            detail=detail,
        )

    elif step.step_type == "add_node":
        expected_additions = ref_nodes - base_nodes
        actual_additions = gen_nodes - base_nodes

        if not expected_additions:
            return StepResult(
                step_index=step.step_index,
                step_type=step.step_type,
                s_sel=1.0,
                s_edge=1.0,
                correct=True,
                detail="No nodes to add",
            )

        correct_additions = expected_additions & actual_additions
        s_sel = len(correct_additions) / len(expected_additions)
        s_edge = 1.0

        correct = correct_additions == expected_additions
        detail = (
            f"Added {len(correct_additions)}/{len(expected_additions)} expected nodes"
        )
        return StepResult(
            step_index=step.step_index,
            step_type=step.step_type,
            s_sel=s_sel,
            s_edge=s_edge,
            correct=correct,
            detail=detail,
        )

    return StepResult(
        step_index=step.step_index,
        step_type=step.step_type,
        s_sel=0.0,
        s_edge=0.0,
        correct=False,
        detail=f"Unknown node operation: {step.step_type}",
    )


def _evaluate_verify_step(
    step: ReasoningStep,
    generated: nx.Graph,
    ref_graph: nx.Graph,
) -> StepResult:
    """Evaluate the final verification step.

    Checks whether the generated graph matches the reference in terms
    of node set and edge set.
    """
    gen_nodes = _node_set(generated)
    ref_nodes = _node_set(ref_graph)
    gen_edges = _edge_set(generated)
    ref_edges = _edge_set(ref_graph)

    node_match = gen_nodes == ref_nodes
    edge_match = gen_edges == ref_edges

    # S_sel based on node set Jaccard similarity
    union_n = gen_nodes | ref_nodes
    inter_n = gen_nodes & ref_nodes
    s_sel = len(inter_n) / len(union_n) if union_n else 1.0

    # S_edge based on edge set Jaccard similarity
    union_e = gen_edges | ref_edges
    inter_e = gen_edges & ref_edges
    s_edge = len(inter_e) / len(union_e) if union_e else 1.0

    correct = node_match and edge_match
    detail = f"Nodes match={node_match}, Edges match={edge_match}"

    return StepResult(
        step_index=step.step_index,
        step_type=step.step_type,
        s_sel=s_sel,
        s_edge=s_edge,
        correct=correct,
        detail=detail,
    )


def evaluate_step(
    step: ReasoningStep,
    generated: Optional[nx.Graph],
    base_graph: Optional[nx.Graph],
    ref_graph: Optional[nx.Graph],
) -> StepResult:
    """Evaluate a single reasoning step.

    Dispatches to the appropriate handler based on step_type.

    Args:
        step: The reasoning step to evaluate.
        generated: The generated graph (LLM output).
        base_graph: The original base graph.
        ref_graph: The reference solution graph.

    Returns:
        StepResult with s_sel, s_edge, and correctness flag.
    """
    if generated is None:
        return StepResult(
            step_index=step.step_index,
            step_type=step.step_type,
            s_sel=0.0,
            s_edge=0.0,
            correct=False,
            detail="Generated graph is None",
        )

    if step.step_type == "construct":
        return _evaluate_construct_step(generated, base_graph)

    if step.step_type == "verify":
        if ref_graph is None:
            return StepResult(
                step_index=step.step_index,
                step_type=step.step_type,
                s_sel=0.0,
                s_edge=0.0,
                correct=False,
                detail="No reference graph for verification",
            )
        return _evaluate_verify_step(step, generated, ref_graph)

    if base_graph is None or ref_graph is None:
        return StepResult(
            step_index=step.step_index,
            step_type=step.step_type,
            s_sel=0.0,
            s_edge=0.0,
            correct=False,
            detail="Missing base or reference graph",
        )

    if step.step_type in ("add_edge", "remove_edge"):
        return _evaluate_edge_ops(step, generated, base_graph, ref_graph)

    if step.step_type in ("add_node", "remove_node"):
        return _evaluate_node_ops(step, generated, base_graph, ref_graph)

    # Generic fallback: compare final state
    return _evaluate_verify_step(step, generated, ref_graph)


def evaluate_reasoning_chain(
    instruction: Dict[str, Any],
    generated_graph: Optional[nx.Graph],
    base_graph: Optional[nx.Graph],
    ref_graphs: Sequence[nx.Graph],
) -> MicroEvalResult:
    """Full micro evaluation of an L5 instruction.

    Decomposes the instruction into steps, evaluates each step against
    the best-matching reference solution, and aggregates scores.

    Args:
        instruction: Instruction dict with id, instruction, etc.
        generated_graph: The graph generated by the LLM.
        base_graph: The parsed base graph (starting state).
        ref_graphs: List of reference solution graphs.

    Returns:
        MicroEvalResult with s_sel, s_edge, s_micro, and step details.
    """
    if not ref_graphs:
        return MicroEvalResult(
            s_sel=0.0,
            s_edge=0.0,
            s_micro=0.0,
            steps_total=0,
            steps_correct=0,
        )

    # Find the best matching reference (highest s_micro)
    best_result: Optional[MicroEvalResult] = None
    best_micro = -1.0

    for ref_graph in ref_graphs:
        steps = decompose_instruction(instruction, base_graph, ref_graph)
        step_results: List[StepResult] = []

        for step in steps:
            sr = evaluate_step(step, generated_graph, base_graph, ref_graph)
            step_results.append(sr)

        if not step_results:
            continue

        # Aggregate: average S_sel and S_edge across all steps
        avg_sel = sum(sr.s_sel for sr in step_results) / len(step_results)
        avg_edge = sum(sr.s_edge for sr in step_results) / len(step_results)
        s_micro = (avg_sel + avg_edge) / 2.0
        correct_count = sum(1 for sr in step_results if sr.correct)

        candidate = MicroEvalResult(
            s_sel=avg_sel,
            s_edge=avg_edge,
            s_micro=s_micro,
            steps_total=len(step_results),
            steps_correct=correct_count,
            step_details=tuple(step_results),
        )

        if s_micro > best_micro:
            best_micro = s_micro
            best_result = candidate

    if best_result is None:
        return MicroEvalResult(
            s_sel=0.0,
            s_edge=0.0,
            s_micro=0.0,
            steps_total=0,
            steps_correct=0,
        )

    return best_result


def build_reasoning_steps_json(
    instruction: Dict[str, Any],
    base_graph: Optional[nx.Graph],
    ref_graph: Optional[nx.Graph],
) -> List[Dict[str, Any]]:
    """Build reasoning_steps JSON for an L5 instruction.

    Produces a list of step dicts suitable for adding to the instruction
    JSON file.

    Args:
        instruction: Instruction dict.
        base_graph: Parsed base graph.
        ref_graph: Parsed reference solution graph.

    Returns:
        List of step dicts with step, type, description, and targets.
    """
    steps = decompose_instruction(instruction, base_graph, ref_graph)
    result: List[Dict[str, Any]] = []

    for step in steps:
        entry: Dict[str, Any] = {
            "step": step.step_index,
            "type": step.step_type,
            "description": step.description,
        }
        if step.target_nodes:
            entry["target_nodes"] = list(step.target_nodes)
        if step.target_edges:
            entry["target_edges"] = [list(e) for e in step.target_edges]
        result.append(entry)

    return result
