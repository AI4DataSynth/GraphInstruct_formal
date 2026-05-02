"""Constraint checking and implicit constraint inference.

Handles explicit constraints like "graph_type=tree", "num_nodes=10"
and infers implicit constraints from graph types.
"""

from typing import Any

import networkx as nx

from graphinstruct.validators.graph_types import GRAPH_TYPE_VALIDATORS, is_regular

# Float-valued graph properties that get 5% tolerance on <= / >= comparisons.
_FLOAT_PROPERTIES = frozenset(
    {
        "density",
        "clustering_coefficient",
        "modularity",
        "average_path_length",
    }
)


def check_constraint(G: nx.Graph, constraint: str) -> bool:
    """Check a single constraint string against a graph.

    Constraint format: "key=value" or "key operator value".
    Supported constraints:
        - graph_type=<type>
        - num_nodes=<n>
        - num_edges=<m>
        - connected=<true/false>
        - acyclic=<true/false>
        - directed=<true/false>
        - degree=<k>  (k-regular)
        - min_degree=<k>
        - max_degree=<k>
        - diameter<=<d>
        - bipartite_sides=<p>,<q>
        - node_connectivity>=<k>
        - edge_connectivity>=<k>
        - chromatic_number<=<k>
        - girth>=<g>

    Returns:
        True if the constraint is satisfied.
    """
    constraint = constraint.strip()

    # Handle operators: <=, >=, <, >, =
    for op in ("<=", ">=", "!=", "<", ">", "="):
        if op in constraint:
            key, value_str = constraint.split(op, 1)
            key = key.strip()
            value_str = value_str.strip()
            return _check_kv_constraint(G, key, op, value_str)

    raise ValueError(f"Invalid constraint format: {constraint!r}")


def check_all_constraints(G: nx.Graph, constraints: list[str]) -> dict[str, bool]:
    """Check multiple constraints, returning a dict of results.

    Args:
        G: The graph to check.
        constraints: List of constraint strings.

    Returns:
        Dict mapping each constraint string to True/False.
    """
    return {c: check_constraint(G, c) for c in constraints}


def satisfaction_rate(G: nx.Graph, constraints: list[str]) -> float:
    """Compute the fraction of constraints satisfied.

    Returns:
        Float in [0.0, 1.0]. Returns 1.0 if constraints is empty.
    """
    if not constraints:
        return 1.0
    results = check_all_constraints(G, constraints)
    return sum(results.values()) / len(results)


# ---------------------------------------------------------------------------
# Implicit constraint inference
# ---------------------------------------------------------------------------

# Database: graph_type -> function(explicit_constraints) -> list of implicit constraints
_IMPLICIT_RULES: dict[str, callable] = {}


def _register_implicit(graph_type: str):
    """Decorator to register an implicit constraint rule."""

    def decorator(func):
        _IMPLICIT_RULES[graph_type] = func
        return func

    return decorator


@_register_implicit("tree")
def _tree_implicit(explicit: dict[str, str]) -> list[str]:
    constraints = ["acyclic=true", "connected=true"]
    if "num_nodes" in explicit:
        n = int(explicit["num_nodes"])
        constraints.append(f"num_edges={n - 1}")
    return constraints


@_register_implicit("cycle")
def _cycle_implicit(explicit: dict[str, str]) -> list[str]:
    constraints = ["connected=true", "degree=2"]
    if "num_nodes" in explicit:
        n = int(explicit["num_nodes"])
        constraints.append(f"num_edges={n}")
    return constraints


@_register_implicit("complete")
def _complete_implicit(explicit: dict[str, str]) -> list[str]:
    constraints: list[str] = []
    if "num_nodes" in explicit:
        n = int(explicit["num_nodes"])
        constraints.append(f"num_edges={n * (n - 1) // 2}")
        constraints.append(f"degree={n - 1}")
    return constraints


@_register_implicit("star")
def _star_implicit(explicit: dict[str, str]) -> list[str]:
    constraints = ["connected=true"]
    if "num_nodes" in explicit:
        n = int(explicit["num_nodes"])
        constraints.append(f"num_edges={n - 1}")
    return constraints


@_register_implicit("path")
def _path_implicit(explicit: dict[str, str]) -> list[str]:
    constraints = ["connected=true", "acyclic=true"]
    if "num_nodes" in explicit:
        n = int(explicit["num_nodes"])
        constraints.append(f"num_edges={n - 1}")
    return constraints


@_register_implicit("regular")
def _regular_implicit(explicit: dict[str, str]) -> list[str]:
    constraints: list[str] = []
    if "num_nodes" in explicit and "degree" in explicit:
        n = int(explicit["num_nodes"])
        k = int(explicit["degree"])
        constraints.append(f"num_edges={n * k // 2}")
    return constraints


@_register_implicit("bipartite")
def _bipartite_implicit(explicit: dict[str, str]) -> list[str]:
    constraints: list[str] = ["bipartite=true"]
    if "bipartite_sides" in explicit:
        p, q = explicit["bipartite_sides"].split(",")
        p, q = int(p.strip()), int(q.strip())
        constraints.append(f"num_edges<={p * q}")
    return constraints


@_register_implicit("complete_bipartite")
def _complete_bipartite_implicit(explicit: dict[str, str]) -> list[str]:
    constraints: list[str] = []
    if "bipartite_sides" in explicit:
        p, q = explicit["bipartite_sides"].split(",")
        p, q = int(p.strip()), int(q.strip())
        constraints.append(f"num_nodes={p + q}")
        constraints.append(f"num_edges={p * q}")
    return constraints


def infer_implicit_constraints(explicit_constraints: list[str]) -> list[str]:
    """Infer implicit constraints from a list of explicit constraints.

    Args:
        explicit_constraints: List of constraint strings like
            ["graph_type=tree", "num_nodes=10"].

    Returns:
        List of inferred implicit constraint strings.
    """
    # Parse explicit constraints into a dict for easier lookup
    explicit_dict: dict[str, str] = {}
    for c in explicit_constraints:
        if "=" in c:
            key, value = c.split("=", 1)
            explicit_dict[key.strip()] = value.strip()

    graph_type = explicit_dict.get("graph_type", "")
    if graph_type not in _IMPLICIT_RULES:
        return []

    implicit = _IMPLICIT_RULES[graph_type](explicit_dict)

    # Remove any constraints already explicitly stated
    explicit_set = set(explicit_constraints)
    return [c for c in implicit if c not in explicit_set]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_METADATA_KEYS = frozenset({"task_type", "domain", "graph_model"})


def _check_kv_constraint(G: nx.Graph, key: str, op: str, value_str: str) -> bool:
    """Check a single key-operator-value constraint."""
    # Metadata constraints (classification tags, not verifiable graph properties)
    if key in _METADATA_KEYS:
        return True

    # Graph type check
    if key == "graph_type":
        validator = GRAPH_TYPE_VALIDATORS.get(value_str)
        if validator is None:
            raise ValueError(f"Unknown graph type: {value_str!r}")
        return validator(G)

    # Boolean properties
    if key in (
        "connected",
        "acyclic",
        "directed",
        "planar",
        "bipartite",
        "weakly_connected",
        "strongly_connected",
    ):
        expected = value_str.lower() == "true"
        actual = _get_bool_property(G, key)
        return actual == expected

    # Bipartite sides: bipartite_sides=p,q
    if key == "bipartite_sides":
        if not nx.is_bipartite(G):
            return False
        parts = value_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"bipartite_sides requires 'p,q' format, got {value_str!r}"
            )
        p, q = int(parts[0].strip()), int(parts[1].strip())
        try:
            top, bottom = nx.bipartite.sets(G)
        except nx.AmbiguousSolution:
            # Disconnected bipartite graph — partition is ambiguous.
            # Fall back to coloring-based partition.
            coloring = nx.bipartite.color(G)
            top = {n for n, c in coloring.items() if c == 0}
            bottom = {n for n, c in coloring.items() if c == 1}
        sizes = sorted([len(top), len(bottom)])
        expected = sorted([p, q])
        return sizes == expected

    # Numeric properties
    actual_value = _get_numeric_property(G, key)
    target = _parse_number(value_str)

    # Apply 5% tolerance for float-valued graph properties when using
    # >= or <= comparisons only.  Strict < / > and exact = remain exact.
    tolerance = abs(target) * 0.05 if key in _FLOAT_PROPERTIES else 0.0

    # min_degree/max_degree use >= / <= semantics by default with "="
    if op == "=" and key == "min_degree":
        return actual_value >= target
    if op == "=" and key == "max_degree":
        return actual_value <= target
    if op == "=":
        return actual_value == target
    elif op == "<=":
        return actual_value <= target + tolerance
    elif op == ">=":
        return actual_value >= target - tolerance
    elif op == "<":
        return actual_value < target
    elif op == ">":
        return actual_value > target
    elif op == "!=":
        return actual_value != target

    raise ValueError(f"Unknown operator: {op!r}")


def _get_bool_property(G: nx.Graph, key: str) -> bool:
    """Get a boolean property of the graph."""
    if key == "connected":
        return GRAPH_TYPE_VALIDATORS["connected"](G)
    if key == "acyclic":
        return GRAPH_TYPE_VALIDATORS["acyclic"](G)
    if key == "directed":
        return isinstance(G, nx.DiGraph)
    if key == "planar":
        return GRAPH_TYPE_VALIDATORS["planar"](G)
    if key == "bipartite":
        return GRAPH_TYPE_VALIDATORS["bipartite"](G)
    if key == "weakly_connected":
        if isinstance(G, nx.DiGraph):
            return nx.is_weakly_connected(G)
        # For undirected graphs, weakly_connected == connected
        return nx.is_connected(G)
    if key == "strongly_connected":
        if isinstance(G, nx.DiGraph):
            return nx.is_strongly_connected(G)
        # For undirected graphs, strongly_connected == connected
        return nx.is_connected(G)
    raise ValueError(f"Unknown boolean property: {key!r}")


def _is_graph_connected(G: nx.Graph) -> bool:
    """Check connectivity, handling both directed and undirected graphs."""
    if G.number_of_nodes() == 0:
        return False
    if isinstance(G, nx.DiGraph):
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)


def _get_numeric_property(G: nx.Graph, key: str) -> int | float:
    """Get a numeric property of the graph."""
    if key == "num_nodes":
        return G.number_of_nodes()
    if key == "num_edges":
        return G.number_of_edges()
    if key == "degree":
        # For regularity check: return the common degree if regular,
        # otherwise return float('nan') to fail all comparisons safely
        degrees = set(G.degree(n) for n in G.nodes())
        if len(degrees) == 1:
            return degrees.pop()
        return float("nan")
    if key == "min_degree":
        if G.number_of_nodes() == 0:
            return 0
        # min_degree constraint uses >= semantics: "min_degree=2" means
        # the minimum degree is at least 2. We return the actual min degree
        # and the caller uses >= comparison for this key.
        return min(G.degree(n) for n in G.nodes())
    if key == "max_degree":
        if G.number_of_nodes() == 0:
            return 0
        return max(G.degree(n) for n in G.nodes())
    if key == "diameter":
        if not _is_graph_connected(G):
            return float("inf")
        if isinstance(G, nx.DiGraph):
            return nx.diameter(G.to_undirected())
        return nx.diameter(G)
    if key == "num_components":
        if isinstance(G, nx.DiGraph):
            return nx.number_weakly_connected_components(G)
        return nx.number_connected_components(G)
    if key == "clustering_coefficient":
        if G.number_of_nodes() == 0:
            return 0.0
        return nx.average_clustering(G)
    if key == "average_path_length":
        if not _is_graph_connected(G):
            return float("inf")
        if isinstance(G, nx.DiGraph):
            return nx.average_shortest_path_length(G.to_undirected())
        return nx.average_shortest_path_length(G)
    if key == "modularity":
        # Use greedy modularity to compute approximate modularity
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return 0.0
        g_undir = G.to_undirected() if isinstance(G, nx.DiGraph) else G
        # Sanitize edge weights: LLM-generated graphs may have string weights
        for u, v, d in g_undir.edges(data=True):
            if "weight" in d and not isinstance(d["weight"], (int, float)):
                try:
                    d["weight"] = float(d["weight"])
                except (ValueError, TypeError):
                    del d["weight"]
        communities = nx.community.greedy_modularity_communities(g_undir)
        return nx.community.modularity(g_undir, communities)
    if key == "density":
        return nx.density(G)
    if key == "node_connectivity":
        if G.number_of_nodes() < 2 or not _is_graph_connected(G):
            return 0
        return nx.node_connectivity(G)
    if key == "edge_connectivity":
        if G.number_of_nodes() < 2 or not _is_graph_connected(G):
            return 0
        return nx.edge_connectivity(G)
    if key == "chromatic_number":
        if G.number_of_nodes() == 0:
            return 0
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        return max(coloring.values()) + 1 if coloring else 0
    if key == "girth":
        if G.number_of_nodes() == 0:
            return float("inf")
        try:
            return nx.girth(G)
        except nx.NetworkXError:
            return float("inf")

    raise ValueError(f"Unknown numeric property: {key!r}")


def _parse_number(value_str: str) -> int | float:
    """Parse a string to int or float."""
    try:
        return int(value_str)
    except ValueError:
        return float(value_str)
