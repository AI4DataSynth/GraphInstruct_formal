"""Graph type validators for constraint checking.

Each function takes a NetworkX graph and returns True/False.
"""

import networkx as nx


def is_connected(G: nx.Graph) -> bool:
    """Check if the graph is connected (undirected) or weakly connected (directed)."""
    if G.number_of_nodes() == 0:
        return True
    if isinstance(G, nx.DiGraph):
        return nx.is_weakly_connected(G)
    return nx.is_connected(G)


def is_acyclic(G: nx.Graph) -> bool:
    """Check if the graph has no cycles."""
    if G.number_of_nodes() == 0:
        return True
    if isinstance(G, nx.DiGraph):
        return nx.is_directed_acyclic_graph(G)
    return nx.is_forest(G)


def is_tree(G: nx.Graph) -> bool:
    """Check if the graph is a tree (connected + acyclic)."""
    if G.number_of_nodes() == 0:
        return False
    if isinstance(G, nx.DiGraph):
        return nx.is_tree(G)
    return nx.is_tree(G)


def is_forest(G: nx.Graph) -> bool:
    """Check if the graph is a forest (acyclic, possibly disconnected)."""
    if G.number_of_nodes() == 0:
        return True
    return nx.is_forest(G)


def is_cycle(G: nx.Graph) -> bool:
    """Check if the graph is a simple cycle (every node has degree 2, connected)."""
    if G.number_of_nodes() < 3:
        return False
    if not is_connected(G):
        return False
    return all(G.degree(n) == 2 for n in G.nodes())


def is_path(G: nx.Graph) -> bool:
    """Check if the graph is a simple path."""
    if G.number_of_nodes() == 0:
        return False
    if G.number_of_nodes() == 1:
        return True
    if not is_connected(G):
        return False
    # A path has exactly 2 nodes of degree 1, all others degree 2
    degrees = [G.degree(n) for n in G.nodes()]
    degree_1_count = sum(1 for d in degrees if d == 1)
    degree_2_count = sum(1 for d in degrees if d == 2)
    return degree_1_count == 2 and degree_2_count == G.number_of_nodes() - 2


def is_star(G: nx.Graph) -> bool:
    """Check if the graph is a star graph (one center node connected to all others)."""
    n = G.number_of_nodes()
    if n < 2:
        return n == 1
    if G.number_of_edges() != n - 1:
        return False
    if not is_connected(G):
        return False
    degrees = sorted([G.degree(node) for node in G.nodes()])
    # Star: one node with degree n-1, all others degree 1
    return degrees[-1] == n - 1 and all(d == 1 for d in degrees[:-1])


def is_complete(G: nx.Graph) -> bool:
    """Check if the graph is a complete graph K_n."""
    n = G.number_of_nodes()
    expected_edges = n * (n - 1) // 2
    return G.number_of_edges() == expected_edges and all(
        G.degree(node) == n - 1 for node in G.nodes()
    )


def is_bipartite(G: nx.Graph) -> bool:
    """Check if the graph is bipartite."""
    return nx.is_bipartite(G)


def is_complete_bipartite(G: nx.Graph) -> bool:
    """Check if the graph is a complete bipartite graph K_{p,q}."""
    if G.number_of_nodes() == 0:
        return False
    if not nx.is_bipartite(G):
        return False
    try:
        top, bottom = nx.bipartite.sets(G)
    except nx.AmbiguousSolution:
        coloring = nx.bipartite.color(G)
        top = {n for n, c in coloring.items() if c == 0}
        bottom = {n for n, c in coloring.items() if c == 1}
    p, q = len(top), len(bottom)
    return G.number_of_edges() == p * q


def is_regular(G: nx.Graph, k: int | None = None) -> bool:
    """Check if the graph is regular (all nodes same degree).

    Args:
        G: The graph to check.
        k: If specified, check for k-regular. Otherwise check any regularity.
    """
    if G.number_of_nodes() == 0:
        return True
    degrees = set(G.degree(n) for n in G.nodes())
    if len(degrees) != 1:
        return False
    if k is not None:
        return degrees.pop() == k
    return True


def is_planar(G: nx.Graph) -> bool:
    """Check if the graph is planar."""
    return nx.check_planarity(G)[0]


def is_dag(G: nx.DiGraph) -> bool:
    """Check if a directed graph is a DAG."""
    if not isinstance(G, nx.DiGraph):
        return False
    return nx.is_directed_acyclic_graph(G)


def is_wheel(G: nx.Graph) -> bool:
    """Check if G is a wheel graph W_n.

    A wheel graph has one hub node connected to all nodes of an outer cycle.
    Properties: n nodes, 2(n-1) edges, one node with degree n-1, rest degree 3.
    """
    n = G.number_of_nodes()
    if n < 4:
        return False
    if G.number_of_edges() != 2 * (n - 1):
        return False
    if not nx.is_connected(G):
        return False
    degrees = sorted((d for _, d in G.degree()), reverse=True)
    # Hub has degree n-1, all rim nodes have degree 3
    return degrees[0] == n - 1 and all(d == 3 for d in degrees[1:])


def is_grid(G: nx.Graph) -> bool:
    """Check if G is a 2D grid (lattice) graph p x q.

    Tests all factorizations of n = p*q (p <= q) and checks graph
    isomorphism against ``nx.grid_2d_graph(p, q)``.

    Uses degree-sequence pre-filtering to avoid expensive VF2 isomorphism
    on graphs that cannot possibly be grids.  A p×q grid has exactly:
      - 4 corners with degree 2
      - 2(p-2)+2(q-2) edge nodes with degree 3
      - (p-2)(q-2) interior nodes with degree 4
      - (p-1)*q + p*(q-1) edges
    This pre-check is exact (no false negatives).
    """
    n = G.number_of_nodes()
    if n < 4:
        return False
    if not nx.is_connected(G):
        return False
    import math
    from collections import Counter

    deg_count = Counter(d for _, d in G.degree())
    # Grid nodes only have degree 2, 3, or 4
    if not set(deg_count.keys()).issubset({2, 3, 4}):
        return False

    m = G.number_of_edges()

    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            q = n // p
            # Quick edge count check
            expected_edges = (p - 1) * q + p * (q - 1)
            if m != expected_edges:
                continue
            # Degree sequence check
            expected_deg2 = 4
            expected_deg3 = 2 * (p - 2) + 2 * (q - 2)
            expected_deg4 = (p - 2) * (q - 2)
            if (
                deg_count.get(2, 0) != expected_deg2
                or deg_count.get(3, 0) != expected_deg3
                or deg_count.get(4, 0) != expected_deg4
            ):
                continue
            # Degree sequence matches — VF2 on grid-like graph is fast
            grid = nx.grid_2d_graph(p, q)
            if nx.is_isomorphic(G, grid):
                return True
    return False


# Registry mapping type names to validator functions
GRAPH_TYPE_VALIDATORS: dict[str, callable] = {
    "tree": is_tree,
    "forest": is_forest,
    "cycle": is_cycle,
    "path": is_path,
    "star": is_star,
    "complete": is_complete,
    "bipartite": is_bipartite,
    "complete_bipartite": is_complete_bipartite,
    "regular": is_regular,
    "planar": is_planar,
    "connected": is_connected,
    "acyclic": is_acyclic,
    "dag": is_dag,
    "wheel": is_wheel,
    "grid": is_grid,
}
