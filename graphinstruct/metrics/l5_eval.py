"""L5 evaluation improvements: edit quality, apply_graph_edit, reasoning alignment.

L5 design intent: evaluate multi-step graph editing capability.
New D4 formula: D4_L5 = 0.45 * edit_quality + 0.40 * constraint_sat + 0.15 * reasoning_alignment

Key signals (validated against raw jsonl):
- edit_quality closeness spread=0.310 (strongest discriminator)
- redundancy: weak 13.1% vs strong 2.1%
- prune_tree isolated node bug confirmed (Phase 0a fix)
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import networkx as nx

from graphinstruct.metrics.instruction import explicit_satisfaction
from graphinstruct.validators import satisfaction_rate


def apply_graph_edit(
    base_graph: nx.Graph,
    add_edges: Sequence[tuple[str, str]],
    remove_edges: Sequence[tuple[str, str]],
    task_type: Optional[str] = None,
) -> nx.Graph:
    """Apply GraphEdit operations to a base graph, returning new graph.

    Phase 0a fix: for prune_tree/extract_subgraph tasks, removes isolated
    nodes after edge deletion. Without this fix, nodes like '4' and '5'
    remain in the graph after their edges are removed, causing num_nodes
    constraint to fail (e.g., n=8 instead of expected n=6).

    Args:
        base_graph: The starting graph.
        add_edges: Edges to add as (u, v) tuples.
        remove_edges: Edges to remove as (u, v) tuples.
        task_type: L5 task type (e.g., 'prune_tree', 'extract_subgraph').

    Returns:
        New graph with edits applied.
    """
    G = base_graph.copy()
    for u, v in remove_edges:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    for u, v in add_edges:
        G.add_edge(u, v)

    # Phase 0a: remove isolated nodes for pruning/extraction tasks
    if task_type in ("prune_tree", "extract_subgraph"):
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

    return G


def edit_quality(
    G_out: nx.Graph,
    base_G: nx.Graph,
    reference_Gs: Sequence[nx.Graph],
    parsed_output: Optional[object] = None,
) -> float:
    """Comprehensive edit quality: closeness to optimal + low redundancy.

    Validated spread=0.310 (strongest L5 discriminator).

    Sub-scores:
    - closeness (0.6): ratio of actual edits to optimal, bidirectional penalty
    - redundancy (0.4): adding edges already in base = redundant

    Args:
        G_out: The generated output graph.
        base_G: The original base graph.
        reference_Gs: Reference solution graphs.
        parsed_output: Parsed output (ParseResult) for GraphEdit detection.

    Returns:
        Score in [0.0, 1.0].
    """
    base_edges = set(frozenset(map(str, e)) for e in base_G.edges())
    gen_edges = set(frozenset(map(str, e)) for e in G_out.edges())

    added = gen_edges - base_edges
    removed = base_edges - gen_edges
    actual_edits = len(added) + len(removed)

    # --- 1. closeness (0.6 weight) ---
    ref_edits = []
    for ref_G in reference_Gs:
        ref_edges = set(frozenset(map(str, e)) for e in ref_G.edges())
        ref_edit = len((base_edges - ref_edges) | (ref_edges - base_edges))
        ref_edits.append(ref_edit)

    optimal = min(ref_edits) if ref_edits else actual_edits

    if optimal == 0:
        closeness = 1.0 if actual_edits == 0 else 0.3
    else:
        ratio = actual_edits / optimal
        if 0.8 <= ratio <= 1.2:
            closeness = 1.0
        elif ratio < 0.8:
            # Too few edits (weak model typical: 10 vs ref 22)
            closeness = max(0.2, ratio)
        else:
            # Too many edits
            closeness = max(0.3, 1.0 - 0.3 * (ratio - 1.2))

    # --- 2. redundancy (0.4 weight) ---
    redundancy_score = 1.0
    if (
        parsed_output is not None
        and getattr(parsed_output, "parse_type", None) == "GraphEdit"
    ):
        stated_add = set(
            frozenset(map(str, e)) for e in getattr(parsed_output, "add_edges", ())
        )
        redundant = stated_add & base_edges
        if stated_add:
            redundancy_rate = len(redundant) / len(stated_add)
            redundancy_score = max(0.3, 1.0 - 3.0 * redundancy_rate)

    return 0.6 * closeness + 0.4 * redundancy_score


def reasoning_alignment(
    raw_output: str,
    strategy: Optional[str] = None,
) -> float:
    """Strategy-aware reasoning + format compliance evaluation.

    ZS/FS: baseline + format compliance bonus.
    ZC/FC: reasoning quality + format compliance.

    Note: v6 validation showed simple keyword extraction has no discriminative
    power. This implementation focuses on format compliance as the primary
    signal, with reasoning quality as secondary for CoT strategies.

    Args:
        raw_output: Raw LLM output string.
        strategy: Prompt strategy ('zero-shot', 'few-shot', 'zero-cot', 'few-cot').

    Returns:
        Score in [0.0, 1.0].
    """
    # Format compliance: GraphEdit > Graph > nothing
    if "GraphEdit[" in raw_output:
        format_score = 1.0
    elif "Graph[" in raw_output:
        format_score = 0.7
    else:
        format_score = 0.3

    if strategy in ("zero-shot", "few-shot", None):
        # ZS/FS: no reasoning process. Format compliance is the only signal.
        return 0.3 + 0.7 * format_score  # range 0.51-1.0

    # CoT strategies: check for explicit reasoning markers
    has_reasoning = bool(
        re.search(r"(remove|add|delete|connect|disconnect|edge|node)", raw_output, re.I)
    )
    if not has_reasoning:
        return 0.3 * format_score

    # Simple reasoning presence score (v6: more refined alignment deferred)
    return 0.6 * 0.7 + 0.4 * format_score  # ~0.7 for CoT with reasoning


def instruction_score_l5(
    graph: Optional[nx.Graph],
    explicit_constraints: list[str],
    base_graph: Optional[nx.Graph] = None,
    reference_graphs: Optional[Sequence[nx.Graph]] = None,
    raw_output: str = "",
    parsed_output: Optional[object] = None,
    strategy: Optional[str] = None,
) -> float:
    """Compute L5 D4 score with edit quality focus.

    D4_L5 = 0.45 * edit_quality + 0.40 * constraint_sat + 0.15 * reasoning_alignment

    Args:
        graph: Generated graph (after apply_graph_edit), or None.
        explicit_constraints: Constraint strings from the instruction.
        base_graph: The original base graph (before editing).
        reference_graphs: Reference solution graphs.
        raw_output: Raw LLM output for reasoning analysis.
        parsed_output: ParseResult for GraphEdit detection.
        strategy: Prompt strategy.

    Returns:
        Combined D4 score in [0.0, 1.0].
    """
    if graph is None:
        return 0.0

    # constraint_sat: explicit + implicit (reuse existing logic)
    s_explicit = explicit_satisfaction(graph, explicit_constraints)
    from graphinstruct.validators import infer_implicit_constraints

    implicit = infer_implicit_constraints(explicit_constraints)
    s_implicit = satisfaction_rate(graph, implicit) if implicit else s_explicit
    constraint_sat = 0.6 * s_explicit + 0.4 * s_implicit

    # edit_quality
    eq = 0.5  # default if no base/ref available
    if base_graph is not None and reference_graphs:
        eq = edit_quality(graph, base_graph, reference_graphs, parsed_output)

    # reasoning_alignment
    ra = reasoning_alignment(raw_output, strategy)

    return 0.45 * eq + 0.40 * constraint_sat + 0.15 * ra
