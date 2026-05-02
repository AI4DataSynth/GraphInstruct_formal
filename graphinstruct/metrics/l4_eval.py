"""L4 evaluation improvements: domain structural fit, L4 scoring.

L4 design intent: evaluate domain-semantic graph generation capability.
New D4 formula: D4_L4 = 0.65 * constraint_sat + 0.20 * domain_structural_fit + 0.15 * no_contradiction

Key findings (validated against raw jsonl):
- Constraint satisfaction is positively correlated with model capability (full-data)
- Only acyclic constraint reverses, root cause: parser directed bug (fix in parser/code_style.py)
- domain_structural_fit spread=0.101 (auxiliary, not dominant)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import networkx as nx

from graphinstruct.metrics.instruction import (
    explicit_satisfaction,
    no_contradiction,
)
from graphinstruct.parser.code_style import recover_directed_edges
from graphinstruct.validators import infer_implicit_constraints, satisfaction_rate


def ensure_directed_from_raw(
    G: nx.Graph,
    instruction: dict,
    raw_output: str,
) -> tuple[nx.Graph, bool]:
    """Backward-compatible wrapper. Delegates to parser.recover_directed_edges.

    Checks instruction constraints before delegating. Kept for existing callers.
    """
    requires_directed = any(
        c.strip() == "directed=true"
        for c in instruction.get("explicit_constraints", [])
    )
    if not requires_directed or G.is_directed():
        return G, False
    dg, _ = recover_directed_edges(G, raw_output)
    return dg, True  # True = conversion was performed


def domain_structural_fit(
    G: nx.Graph,
    reference_graphs: Sequence[nx.Graph],
) -> float:
    """Single-graph structural matching against domain references.

    Validated spread=0.101 (auxiliary discriminator, weight 0.20).
    Takes the max score across all references (best match).

    Args:
        G: Generated graph.
        reference_graphs: Reference solution graphs for this domain.

    Returns:
        Score in [0.0, 1.0].
    """
    if not reference_graphs:
        return 1.0

    scores = []
    for ref_G in reference_graphs:
        s = _structural_similarity(G, ref_G)
        scores.append(s)
    return max(scores)


def _structural_similarity(G: nx.Graph, ref_G: nx.Graph) -> float:
    """6-facet structural comparison between generated and reference graph.

    Facets and weights:
    1. Degree distribution (KS test) — 0.25
    2. Clustering coefficient — 0.20
    3. Density ratio — 0.15
    4. Connectivity consistency — 0.15
    5. Degree assortativity — 0.15
    6. Transitivity/reciprocity — 0.10
    """
    from scipy.stats import ks_2samp

    sub_scores: list[tuple[str, float, float]] = []

    # 1. Degree distribution similarity (KS test) — weight 0.25
    deg_G = sorted([d for _, d in G.degree()], reverse=True)
    deg_ref = sorted([d for _, d in ref_G.degree()], reverse=True)
    if deg_G and deg_ref:
        ks_stat = ks_2samp(deg_G, deg_ref).statistic
        deg_score = max(0.0, 1.0 - 1.5 * ks_stat)
    else:
        deg_score = 0.0
    sub_scores.append(("degree_dist", deg_score, 0.25))

    # 2. Clustering coefficient similarity — weight 0.20
    cc_G = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
    cc_ref = nx.average_clustering(ref_G) if ref_G.number_of_nodes() > 0 else 0.0
    cc_score = max(0.0, 1.0 - 3.0 * abs(cc_G - cc_ref))
    sub_scores.append(("clustering", cc_score, 0.20))

    # 3. Density ratio — weight 0.15
    den_G = nx.density(G)
    den_ref = nx.density(ref_G)
    if max(den_G, den_ref) > 0:
        den_ratio = min(den_G, den_ref) / max(den_G, den_ref)
    else:
        den_ratio = 1.0
    sub_scores.append(("density_ratio", den_ratio, 0.15))

    # 4. Connectivity consistency — weight 0.15
    ref_connected = _is_connected(ref_G)
    gen_connected = _is_connected(G)
    if ref_connected and not gen_connected:
        comp_score = 0.3
    else:
        comp_score = 1.0
    sub_scores.append(("connectivity", comp_score, 0.15))

    # 5. Degree assortativity — weight 0.15
    # Regular graphs (all same degree) have undefined assortativity.
    # Both regular → perfect match (1.0); only one regular → mismatch (0.0).
    try:
        a_G = nx.degree_assortativity_coefficient(G)
        if math.isnan(a_G):
            a_G = None
    except (nx.NetworkXError, ZeroDivisionError):
        a_G = None
    try:
        a_ref = nx.degree_assortativity_coefficient(ref_G)
        if math.isnan(a_ref):
            a_ref = None
    except (nx.NetworkXError, ZeroDivisionError):
        a_ref = None
    if a_G is None and a_ref is None:
        assort_score = 1.0
    elif a_G is None or a_ref is None:
        assort_score = 0.0
    else:
        assort_score = max(0.0, 1.0 - 2.0 * abs(a_G - a_ref))
    sub_scores.append(("assortativity", assort_score, 0.15))

    # 6. Transitivity / reciprocity — weight 0.10
    if not G.is_directed():
        trans_G = nx.transitivity(G)
        trans_ref = nx.transitivity(ref_G) if not ref_G.is_directed() else 0.0
    else:
        # Directed: use reciprocity as substitute
        trans_G = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0
        trans_ref = (
            nx.reciprocity(ref_G)
            if ref_G.is_directed() and ref_G.number_of_edges() > 0
            else 0.0
        )
    trans_score = max(0.0, 1.0 - 3.0 * abs(trans_G - trans_ref))
    sub_scores.append(("transitivity", trans_score, 0.10))

    return sum(score * weight for _, score, weight in sub_scores)


def _is_connected(G: nx.Graph) -> bool:
    """Check connectivity for both directed and undirected graphs."""
    if G.number_of_nodes() == 0:
        return False
    if isinstance(G, nx.DiGraph):
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)


def instruction_score_l4(
    graph: Optional[nx.Graph],
    explicit_constraints: list[str],
    reference_graphs: Optional[Sequence[nx.Graph]] = None,
) -> float:
    """Compute L4 D4 score with domain structural fit.

    D4_L4 = 0.65 * constraint_sat + 0.20 * domain_structural_fit + 0.15 * no_contradiction

    Note: Phase 0b directed fix is now applied globally in evaluate.py
    before D1/D4 computation, so this function receives already-fixed graphs.

    Args:
        graph: Generated graph (already directed-fixed if needed), or None.
        explicit_constraints: Constraint strings from the instruction.
        reference_graphs: Reference solution graphs for domain comparison.

    Returns:
        Combined D4 score in [0.0, 1.0].
    """
    if graph is None:
        return 0.0

    # constraint_sat: explicit + implicit
    s_explicit = explicit_satisfaction(graph, explicit_constraints)
    implicit = infer_implicit_constraints(explicit_constraints)
    if implicit:
        s_implicit = satisfaction_rate(graph, implicit)
        constraint_sat = 0.6 * s_explicit + 0.4 * s_implicit
    else:
        constraint_sat = s_explicit

    # domain_structural_fit
    dsf = 1.0
    if reference_graphs:
        dsf = domain_structural_fit(graph, reference_graphs)

    # no_contradiction
    s_no_contra = 1.0 if no_contradiction(graph) else 0.0

    return 0.65 * constraint_sat + 0.20 * dsf + 0.15 * s_no_contra
