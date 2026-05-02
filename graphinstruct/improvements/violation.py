"""Violation detail extraction for VGIG layered feedback.

Upgrades boolean constraint checks (``validators/constraints.check_constraint``)
into structured ``ViolationDetail`` objects that carry enough diagnostic
information to generate precise feedback text.

This module does NOT replace the existing ``check_constraint`` API; it adds a
parallel path used only by VGIG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from graphinstruct.improvements.domain_priors import get_prior
from graphinstruct.validators.constraints import (
    _get_numeric_property,
    check_constraint,
)
from graphinstruct.validators.graph_types import GRAPH_TYPE_VALIDATORS


def _is_nan(x: Any) -> bool:
    """Type-safe NaN check (``math.isnan`` raises on non-float inputs)."""
    return isinstance(x, float) and math.isnan(x)


ConstraintCategory = Literal[
    "structural",  # graph_type, connected, acyclic, bipartite, planar, chromatic, girth
    "degree",  # degree, min_degree, max_degree (k-regular)
    "numerical",  # density, clustering, avg_path_length, diameter, modularity
    "count",  # num_nodes, num_edges
    "directed",  # directed=true/false
    "domain",  # L4 domain structural mismatch
    "unknown",
]

_DEGREE_KEYS = frozenset({"degree", "min_degree", "max_degree"})
_COUNT_KEYS = frozenset({"num_nodes", "num_edges"})
_NUMERICAL_KEYS = frozenset(
    {
        "density",
        "clustering",
        "clustering_coefficient",
        "avg_path_length",
        "average_path_length",
        "diameter",
        "modularity",
        "node_connectivity",
        "edge_connectivity",
    }
)
_STRUCTURAL_KEYS = frozenset(
    {
        "graph_type",
        "connected",
        "acyclic",
        "bipartite",
        "bipartite_sides",
        "planar",
        "chromatic_number",
        "girth",
        "is_tree",
        "is_dag",
    }
)

# _get_numeric_property uses canonical long names; constraint strings may use
# shorter aliases. Normalize before probing the numeric getter.
_NUMERICAL_ALIASES: dict[str, str] = {
    "clustering": "clustering_coefficient",
    "avg_path_length": "average_path_length",
}

_MAX_OFFENDERS = 10
_MAX_COMPONENTS = 5


@dataclass(frozen=True)
class ViolationDetail:
    """A single violated constraint with diagnostic locus.

    Attributes:
        constraint: Original constraint string (e.g. "degree=3").
        category: Routing label used by feedback formatters.
        satisfied: True if this constraint is actually met (may still appear
            in the returned list when ``extract_violations`` is used for full
            diagnostics; feedback generators filter satisfied=True out).
        actual: Measured value from the graph (int/float/list/str).
        expected: Target value or range.
        delta: Signed numerical gap (actual - expected) for numerical
            constraints; None otherwise.
        locus: Category-specific diagnostic dict. Documented shapes:
            degree   -> {"offending_nodes": [3, 7], "degree_table": {0: 3, ...}}
            numerical-> {"direction": "higher"|"lower", "gap": 0.08}
            structural-> {"components": [[0,1,2], [3,4]]} or {"cycles": [[3,7,9]]}
            count    -> {"need_delta": +2}
            domain   -> {"domain": "social", "mismatch": [("clustering",0.1,"0.40-0.70")]}
            parse    -> {"reason": "parse failed"}
    """

    constraint: str
    category: ConstraintCategory
    satisfied: bool
    actual: Any = None
    expected: Any = None
    delta: float | None = None
    locus: dict[str, Any] = field(default_factory=dict)


def classify_constraint(constraint: str) -> ConstraintCategory:
    """Route a constraint string to its category for feedback formatting.

    Relies on the left-hand side of the comparison operator (``=``, ``<=``,
    ``>=``, ``<``, ``>``, ``!=``). Unknown keys map to ``"unknown"``.
    """
    key = _split_key(constraint)
    if key is None:
        return "unknown"
    key = key.lower()
    if key == "directed":
        return "directed"
    if key in _DEGREE_KEYS:
        return "degree"
    if key in _COUNT_KEYS:
        return "count"
    if key in _NUMERICAL_KEYS:
        return "numerical"
    if key in _STRUCTURAL_KEYS:
        return "structural"
    return "unknown"


def extract_violations(
    graph: nx.Graph | nx.DiGraph | None,
    explicit: tuple[str, ...],
    implicit: tuple[str, ...],
    domain: str | None = None,
) -> list[ViolationDetail]:
    """Produce the list of constraint evaluations for feedback generation.

    Returns satisfied and unsatisfied entries alike; feedback formatters
    filter on ``satisfied=False``. When ``graph`` is None (parse failure),
    returns a single sentinel ``ViolationDetail`` with category="structural"
    and constraint="parse" so feedback can still be generated.
    """
    if graph is None:
        return [_parse_failed_sentinel()]

    results: list[ViolationDetail] = []
    seen: set[str] = set()
    for constraint in list(explicit) + list(implicit):
        if constraint in seen:
            continue
        seen.add(constraint)

        key = _split_key(constraint)
        if key is None:
            continue
        key = key.lower()

        if key == "graph_type":
            results.extend(_check_graph_type(graph, constraint))
            continue

        category = classify_constraint(constraint)
        if category == "degree":
            results.extend(_check_degree(graph, constraint))
        elif category == "count":
            results.extend(_check_count(graph, constraint))
        elif category == "numerical":
            results.extend(_check_numerical(graph, constraint))
        elif category == "directed":
            results.extend(_check_directed(graph, constraint))
        elif category == "structural":
            results.extend(_check_structural(graph, constraint))
        # unknown -> skip silently (already exercised by validators tests)

    if domain:
        results.extend(_check_domain(graph, domain))

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_key(constraint: str) -> str | None:
    """Return the LHS of the first comparison operator in a constraint string."""
    for op in ("<=", ">=", "!=", "<", ">", "="):
        if op in constraint:
            return constraint.split(op, 1)[0].strip()
    return None


def _split_value(constraint: str) -> tuple[str, str]:
    """Return (op, value_str) for a constraint, or ("", "") if malformed."""
    for op in ("<=", ">=", "!=", "<", ">", "="):
        if op in constraint:
            _, value_str = constraint.split(op, 1)
            return op, value_str.strip()
    return "", ""


def _parse_target(value_str: str) -> int | float:
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return float("nan")


def _safe_check(constraint: str, G: nx.Graph) -> bool:
    try:
        return check_constraint(G, constraint)
    except Exception:  # pragma: no cover - defensive
        return False


def _check_graph_type(G: nx.Graph, constraint: str) -> list[ViolationDetail]:
    """graph_type=<value>: emits structural category with diagnostic locus."""
    _, value_str = constraint.split("=", 1)
    value_str = value_str.strip()

    validator = GRAPH_TYPE_VALIDATORS.get(value_str)
    if validator is None:
        return [
            ViolationDetail(
                constraint=constraint,
                category="structural",
                satisfied=False,
                actual="unknown",
                expected=value_str,
                locus={"reason": f"unknown graph type: {value_str!r}"},
            )
        ]

    try:
        satisfied = validator(G)
    except Exception:
        satisfied = False

    locus: dict[str, Any] = {}
    if not satisfied:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        is_dir = isinstance(G, nx.DiGraph)
        try:
            is_conn = (
                (nx.is_weakly_connected(G) if is_dir else nx.is_connected(G))
                if n > 0
                else False
            )
        except Exception:
            is_conn = False
        try:
            is_forest = nx.is_forest(G if not is_dir else G.to_undirected())
        except Exception:
            is_forest = False

        if value_str == "tree":
            locus = {
                "num_nodes": n,
                "num_edges_actual": m,
                "num_edges_expected": max(n - 1, 0),
                "connected": is_conn,
                "acyclic": is_forest,
            }
        elif value_str in ("cycle", "star", "path", "complete"):
            locus = {"num_nodes": n, "num_edges": m}
        else:
            locus = {"num_nodes": n, "num_edges": m}

    return [
        ViolationDetail(
            constraint=constraint,
            category="structural",
            satisfied=satisfied,
            actual=None,
            expected=value_str,
            locus=locus,
        )
    ]


def _check_degree(G: nx.Graph, constraint: str) -> list[ViolationDetail]:
    """degree / min_degree / max_degree: fills offending_nodes + degree_table."""
    key = _split_key(constraint)
    if key is None:
        return []
    key = key.lower()

    _, value_str = _split_value(constraint)
    target = _parse_target(value_str)

    degree_table = {n: G.degree(n) for n in G.nodes()}
    if not degree_table:
        return [
            ViolationDetail(
                constraint=constraint,
                category="degree",
                satisfied=False,
                actual=0,
                expected=target,
                locus={"offending_nodes": [], "degree_table": {}, "empty_graph": True},
            )
        ]

    if key == "degree":
        offending = [n for n, d in degree_table.items() if d != target]
        unique_degrees = set(degree_table.values())
        actual: Any = (
            next(iter(unique_degrees)) if len(unique_degrees) == 1 else "non-uniform"
        )
    elif key == "min_degree":
        offending = [n for n, d in degree_table.items() if d < target]
        actual = min(degree_table.values())
    elif key == "max_degree":
        offending = [n for n, d in degree_table.items() if d > target]
        actual = max(degree_table.values())
    else:
        return []

    satisfied = _safe_check(constraint, G)
    trimmed_offenders = list(offending[:_MAX_OFFENDERS])
    sample_table = {
        n: degree_table[n] for n in list(degree_table.keys())[:_MAX_OFFENDERS]
    }

    return [
        ViolationDetail(
            constraint=constraint,
            category="degree",
            satisfied=satisfied,
            actual=actual,
            expected=target,
            locus={
                "offending_nodes": trimmed_offenders,
                "degree_table": sample_table,
                "offender_count": len(offending),
            },
        )
    ]


def _check_count(G: nx.Graph, constraint: str) -> list[ViolationDetail]:
    """num_nodes / num_edges: fills need_delta (signed)."""
    key = _split_key(constraint)
    if key is None:
        return []
    key = key.lower()

    _, value_str = _split_value(constraint)
    target = _parse_target(value_str)

    if key == "num_nodes":
        actual = G.number_of_nodes()
    elif key == "num_edges":
        actual = G.number_of_edges()
    else:
        return []

    satisfied = _safe_check(constraint, G)
    try:
        need_delta = int(target) - int(actual)
    except (TypeError, ValueError):
        need_delta = 0

    return [
        ViolationDetail(
            constraint=constraint,
            category="count",
            satisfied=satisfied,
            actual=actual,
            expected=target,
            delta=float(need_delta),
            locus={"need_delta": need_delta},
        )
    ]


def _check_structural(G: nx.Graph, constraint: str) -> list[ViolationDetail]:
    """connected / acyclic / bipartite / planar: fills components or cycles."""
    key = _split_key(constraint)
    if key is None:
        return []
    key = key.lower()

    if key == "graph_type":
        return _check_graph_type(G, constraint)

    satisfied = _safe_check(constraint, G)
    locus: dict[str, Any] = {}
    actual: Any = None
    expected: Any = None

    if key == "connected":
        try:
            if isinstance(G, nx.DiGraph):
                components = list(nx.weakly_connected_components(G))
            else:
                components = list(nx.connected_components(G))
        except Exception:
            components = []
        components_sorted = sorted(
            [sorted(c) for c in components], key=len, reverse=True
        )
        locus = {"components": components_sorted[:_MAX_COMPONENTS]}
        actual = f"{len(components_sorted)} components"
        expected = "1 connected component"
    elif key == "acyclic":
        cycle_nodes: list[Any] = []
        try:
            edges = list(nx.find_cycle(G))
            cycle_nodes = [e[0] for e in edges]
            if edges:
                cycle_nodes.append(edges[-1][1])
        except nx.NetworkXNoCycle:
            cycle_nodes = []
        except Exception:
            cycle_nodes = []
        if cycle_nodes:
            locus = {"cycles": [cycle_nodes[:_MAX_OFFENDERS]]}
            actual = "has cycle"
        else:
            actual = "acyclic"
        expected = "acyclic"
    elif key == "bipartite":
        try:
            is_bip = nx.is_bipartite(G)
        except Exception:
            is_bip = False
        actual = "bipartite" if is_bip else "non-bipartite"
        expected = "bipartite"
    elif key == "bipartite_sides":
        _, value_str = _split_value(constraint)
        try:
            parts = value_str.split(",")
            expected_sizes = sorted([int(parts[0].strip()), int(parts[1].strip())])
        except Exception:
            expected_sizes = None
        actual_sizes: list[int] | None = None
        try:
            if nx.is_bipartite(G):
                try:
                    top, bottom = nx.bipartite.sets(G)
                except nx.AmbiguousSolution:
                    coloring = nx.bipartite.color(G)
                    top = {n for n, c in coloring.items() if c == 0}
                    bottom = {n for n, c in coloring.items() if c == 1}
                actual_sizes = sorted([len(top), len(bottom)])
        except Exception:
            actual_sizes = None
        actual = actual_sizes if actual_sizes is not None else "non-bipartite"
        expected = expected_sizes if expected_sizes is not None else value_str
        locus = {"actual_sizes": actual_sizes, "expected_sizes": expected_sizes}
    elif key == "planar":
        try:
            is_planar, _ = nx.check_planarity(G)
        except Exception:
            is_planar = False
        actual = "planar" if is_planar else "non-planar"
        expected = "planar"
    elif key in ("chromatic_number", "girth"):
        try:
            actual_num = _get_numeric_property(G, key)
        except Exception:
            actual_num = None
        actual = actual_num
        _, value_str = _split_value(constraint)
        expected = _parse_target(value_str)

    return [
        ViolationDetail(
            constraint=constraint,
            category="structural",
            satisfied=satisfied,
            actual=actual,
            expected=expected,
            locus=locus,
        )
    ]


def _check_directed(G: nx.Graph, constraint: str) -> list[ViolationDetail]:
    """directed=true/false."""
    _, value_str = _split_value(constraint)
    expected_directed = value_str.strip().lower() == "true"
    actual_directed = isinstance(G, nx.DiGraph)
    satisfied = actual_directed == expected_directed
    return [
        ViolationDetail(
            constraint=constraint,
            category="directed",
            satisfied=satisfied,
            actual="directed" if actual_directed else "undirected",
            expected="directed" if expected_directed else "undirected",
        )
    ]


def _check_numerical(G: nx.Graph, constraint: str) -> list[ViolationDetail]:
    """density / clustering / avg_path_length / diameter / modularity."""
    key = _split_key(constraint)
    if key is None:
        return []
    key = key.lower()
    internal_key = _NUMERICAL_ALIASES.get(key, key)

    op, value_str = _split_value(constraint)
    target = _parse_target(value_str)

    try:
        actual = _get_numeric_property(G, internal_key)
    except Exception:
        actual = float("nan")

    satisfied = _safe_check(constraint, G)

    if isinstance(actual, (int, float)) and not _is_nan(actual) and not _is_nan(target):
        delta = float(actual) - float(target)
        if delta > 0:
            direction = "higher"
        elif delta < 0:
            direction = "lower"
        else:
            direction = "match"
        gap: float | None = abs(delta)
    else:
        delta = None
        direction = "unknown"
        gap = None

    return [
        ViolationDetail(
            constraint=constraint,
            category="numerical",
            satisfied=satisfied,
            actual=actual,
            expected=target,
            delta=delta,
            locus={"direction": direction, "gap": gap, "op": op},
        )
    ]


def _check_domain(G: nx.Graph, domain: str) -> list[ViolationDetail]:
    """Compare G's stats against DOMAIN_PRIORS and emit mismatches."""
    prior = get_prior(domain)
    if prior is None:
        return []

    n = G.number_of_nodes()
    if n == 0:
        return [
            ViolationDetail(
                constraint=f"domain={domain}",
                category="domain",
                satisfied=False,
                locus={"domain": domain, "mismatch": [("num_nodes", 0, ">=1")]},
            )
        ]

    try:
        avg_deg = sum(dict(G.degree()).values()) / max(n, 1)
    except Exception:
        avg_deg = 0.0
    try:
        g_undir = G.to_undirected() if isinstance(G, nx.DiGraph) else G
        clustering = nx.average_clustering(g_undir) if n > 0 else 0.0
    except Exception:
        clustering = 0.0
    try:
        density = nx.density(G)
    except Exception:
        density = 0.0

    mismatches: list[tuple[str, float, str]] = []
    for stat_name, actual, prior_range in (
        ("avg_degree", avg_deg, prior.avg_degree),
        ("clustering", clustering, prior.clustering),
        ("density", density, prior.density),
    ):
        lo, hi = prior_range
        if not (lo <= actual <= hi):
            mismatches.append(
                (stat_name, round(float(actual), 3), f"{lo:.2f}-{hi:.2f}")
            )

    return [
        ViolationDetail(
            constraint=f"domain={domain}",
            category="domain",
            satisfied=len(mismatches) == 0,
            locus={"domain": domain, "mismatch": mismatches},
        )
    ]


def _parse_failed_sentinel() -> ViolationDetail:
    """Canonical sentinel for graph=None (parse failure)."""
    return ViolationDetail(
        constraint="parse",
        category="structural",
        satisfied=False,
        actual=None,
        expected="a parseable graph",
        locus={"reason": "parse failed"},
    )
