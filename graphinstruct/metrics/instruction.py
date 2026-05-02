"""D4: Instruction matching metrics.

Evaluates whether generated graphs satisfy instruction requirements:
- explicit_satisfaction: Rate of explicit constraint satisfaction
- implicit_satisfaction: Rate of implicit constraint satisfaction
- no_contradiction: Check for internal consistency issues
- instruction_score: Combined D4 score
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from graphinstruct.validators import (
    check_constraint,
    infer_implicit_constraints,
    satisfaction_rate,
)


def explicit_satisfaction(
    graph: Optional[nx.Graph],
    explicit_constraints: list[str],
) -> float:
    """Compute the fraction of explicit constraints satisfied.

    Args:
        graph: Generated graph, or None if parsing failed.
        explicit_constraints: List of constraint strings from the instruction.

    Returns:
        Satisfaction rate in [0.0, 1.0]. Returns 1.0 for empty constraints.
    """
    if graph is None:
        if not explicit_constraints:
            return 1.0
        return 0.0

    return satisfaction_rate(graph, explicit_constraints)


def implicit_satisfaction(
    graph: Optional[nx.Graph],
    explicit_constraints: list[str],
) -> float:
    """Compute the fraction of implicit constraints satisfied.

    Infers implicit constraints from explicit ones (e.g., tree -> acyclic,
    connected, n-1 edges) and checks them against the graph.

    Args:
        graph: Generated graph, or None if parsing failed.
        explicit_constraints: List of explicit constraint strings.

    Returns:
        Satisfaction rate in [0.0, 1.0]. Returns 1.0 if no implicit
        constraints can be inferred.
    """
    if graph is None:
        implicit = infer_implicit_constraints(explicit_constraints)
        if not implicit:
            return 1.0
        return 0.0

    implicit = infer_implicit_constraints(explicit_constraints)
    if not implicit:
        return 1.0

    return satisfaction_rate(graph, implicit)


def no_contradiction(graph: Optional[nx.Graph]) -> bool:
    """Check if a graph has no internal contradictions (Conflict hallucination).

    Checks for:
    - Self-loops (contradiction for simple graphs)
    - Duplicate/parallel edges (multi-edges in a simple graph context)
    - Edges referencing nodes not in the graph's node set
    - Empty node labels (nodes with None or empty-string names)

    Args:
        graph: Generated graph, or None.

    Returns:
        True if no contradictions found.
    """
    if graph is None:
        return False

    # Check for self-loops
    if nx.number_of_selfloops(graph) > 0:
        return False

    # Check for edges referencing non-existent nodes
    node_set = set(graph.nodes())
    for u, v in graph.edges():
        if u not in node_set or v not in node_set:
            return False

    # Check for empty/None node labels
    for node in graph.nodes():
        if node is None or (isinstance(node, str) and node.strip() == ""):
            return False

    # Check for multi-graph encoded as simple graph (duplicate edges)
    if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
        edge_set = set()
        for u, v, k in graph.edges(keys=True):
            edge_key = (u, v) if graph.is_directed() else tuple(sorted([u, v]))
            if edge_key in edge_set and k > 0:
                return False
            edge_set.add(edge_key)

    # Defensive: undirected graphs must have no one-way edges (design doc Section 3.5).
    # nx.Graph guarantees symmetry internally, so this is a safety net against
    # custom subclasses or future NetworkX changes that might break this invariant.
    if not graph.is_directed():
        for u, v in graph.edges():
            if not graph.has_edge(v, u):
                return False

    return True


def instruction_score(
    graph: Optional[nx.Graph],
    explicit_constraints: list[str],
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute the combined D4 instruction matching score.

    Combines explicit satisfaction, implicit satisfaction, and no-contradiction
    into a single score.

    Args:
        graph: Generated graph, or None.
        explicit_constraints: List of explicit constraint strings.
        weights: Optional weight dict with keys "explicit", "implicit",
                 "contradiction". Defaults to equal weights.

    Returns:
        Combined score in [0.0, 1.0].
    """
    if weights is None:
        weights = {"explicit": 0.4, "implicit": 0.4, "contradiction": 0.2}

    s_explicit = explicit_satisfaction(graph, explicit_constraints)
    s_no_contra = 1.0 if no_contradiction(graph) else 0.0

    # When no implicit constraints can be inferred, redistribute the
    # implicit weight proportionally to explicit and contradiction.
    implicit = infer_implicit_constraints(explicit_constraints)
    if not implicit:
        w_total = weights["explicit"] + weights["contradiction"]
        if w_total > 0:
            return (
                (weights["explicit"] / w_total) * s_explicit
                + (weights["contradiction"] / w_total) * s_no_contra
            )
        return 0.0

    # Compute implicit satisfaction directly from pre-computed list,
    # avoiding a redundant infer_implicit_constraints call inside
    # implicit_satisfaction().
    if graph is None:
        s_implicit = 0.0
    else:
        s_implicit = satisfaction_rate(graph, implicit)

    return (
        weights["explicit"] * s_explicit
        + weights["implicit"] * s_implicit
        + weights["contradiction"] * s_no_contra
    )
