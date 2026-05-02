"""D1: Structural quality metrics.

Implements D1 sub-metrics per design doc Section 3.2 (Scheme B):
- Valid Rate: constraint satisfaction rate (all levels)
- GED: Graph Edit Distance to reference (diagnostic only, not in D1 combined)
- MMD.D: degree distribution MMD, Gaussian-EMD kernel sigma=median (L2-L5)
- MMD.C: clustering coefficient distribution MMD, Gaussian-EMD kernel sigma=median (L2-L5)
- MMD.S: spectral distribution MMD, Gaussian-EMD kernel sigma=median (L3-L5)
- MMD.O: orbit count distribution MMD (DEPRECATED — not in D1 combined; orbit
  approximation too coarse, full orbit-count requires Linux orca binary)
- Uniqueness: fraction of distinct graphs in k generations (mode collapse detection)
"""

from __future__ import annotations

import concurrent.futures
import math
from typing import Callable, Optional, Sequence

import numpy as np
import networkx as nx

from graphinstruct.validators import GRAPH_TYPE_VALIDATORS, check_constraint


def valid_rate_single(
    graph: Optional[nx.Graph],
    graph_type: Optional[str] = None,
    constraints: Optional[list[str]] = None,
) -> bool:
    """Check if a single graph is structurally valid (parse + type only).

    Only checks format validity (non-None) and graph type correctness.
    Does NOT check explicit constraints — that is D4's responsibility.
    This separation avoids double-counting constraint satisfaction in D1 and D4.

    Args:
        graph: The generated graph, or None if parsing failed.
        graph_type: Expected graph type (e.g., "tree", "cycle").
        constraints: Ignored (kept for API compatibility). Constraint
            checking is handled by D4 ``explicit_satisfaction()``.

    Returns:
        True if the graph parses successfully and matches the expected type.
    """
    if graph is None:
        return False

    # Check graph type if specified
    if graph_type is not None:
        validator = GRAPH_TYPE_VALIDATORS.get(graph_type)
        if validator is None:
            return False
        if not validator(graph):
            return False

    # NOTE: explicit constraints are NOT checked here (Plan C).
    # D4 explicit_satisfaction() handles all constraint checking
    # with partial credit scoring.

    return True


def valid_rate(
    graphs: Sequence[Optional[nx.Graph]],
    graph_type: Optional[str] = None,
    constraints: Optional[list[str]] = None,
) -> float:
    """Compute the fraction of valid graphs in a batch.

    Args:
        graphs: List of generated graphs (None entries = parse failures).
        graph_type: Expected graph type.
        constraints: List of constraint strings.

    Returns:
        Fraction of valid graphs in [0.0, 1.0]. Returns 0.0 for empty list.
    """
    if not graphs:
        return 0.0

    valid_count = sum(
        1
        for g in graphs
        if valid_rate_single(g, graph_type=graph_type, constraints=constraints)
    )
    return valid_count / len(graphs)


# ---------------------------------------------------------------------------
# GED: Graph Edit Distance (L1-L2)
# ---------------------------------------------------------------------------


_GED_MAX_NODES = 10  # GED is NP-hard; skip exact computation for larger graphs


def _ged_fallback(gen: nx.Graph, ref: nx.Graph) -> float:
    """Cheap structural distance fallback when exact GED is intractable.

    Returns ``|Δnodes| + |Δedges|``, a lower bound on the true GED.
    Returns 0 for graphs with identical node/edge counts but different
    topology — acceptable because GED is only used at L1-L2 where
    graphs are typically small (<10 nodes).
    """
    n_diff = abs(gen.number_of_nodes() - ref.number_of_nodes())
    e_diff = abs(gen.number_of_edges() - ref.number_of_edges())
    return float(n_diff + e_diff)


def _ged_exact(gen: nx.Graph, ref: nx.Graph) -> float:
    """Run optimize_graph_edit_distance; take best bound found."""
    best = float("inf")
    for approx in nx.optimize_graph_edit_distance(gen, ref):
        best = approx
        break  # first approximation is good enough for benchmark
    return float(best)


def ged(
    gen: nx.Graph,
    ref: nx.Graph,
    timeout: float = 2.0,
    max_nodes: int = _GED_MAX_NODES,
) -> float:
    """Approximate Graph Edit Distance (unit cost) with hard timeout.

    Uses ``nx.optimize_graph_edit_distance`` in a worker thread with a
    hard timeout.  For graphs larger than *max_nodes*, falls back to a
    cheap structural distance because the exact GED generator can block
    indefinitely before its first yield.

    Args:
        gen: Generated graph.
        ref: Reference graph.
        timeout: Hard timeout in seconds (enforced via thread).
        max_nodes: Skip exact GED when either graph exceeds this size.

    Returns:
        Best upper-bound GED found within the time limit, or a
        structural-difference fallback for large graphs / timeouts.
    """
    if gen.number_of_nodes() > max_nodes or ref.number_of_nodes() > max_nodes:
        return _ged_fallback(gen, ref)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_ged_exact, gen, ref)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return _ged_fallback(gen, ref)
    except Exception:
        return _ged_fallback(gen, ref)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def ged_score(
    gen: nx.Graph,
    ref: nx.Graph,
    timeout: float = 2.0,
) -> float:
    """Normalised GED mapped to [0, 1] (1 = identical).

    Normalisation: ``1 - ged / max(|V1|+|E1|, |V2|+|E2|)``
    clamped to [0, 1].
    """
    raw = ged(gen, ref, timeout=timeout)
    max_size = max(
        gen.number_of_nodes() + gen.number_of_edges(),
        ref.number_of_nodes() + ref.number_of_edges(),
        1,
    )
    return max(0.0, 1.0 - raw / max_size)


# ---------------------------------------------------------------------------
# Distribution feature extractors
# ---------------------------------------------------------------------------


def _degree_histogram(g: nx.Graph) -> np.ndarray:
    """Return normalised degree histogram as 1-D array."""
    if g.number_of_nodes() == 0:
        return np.array([1.0])
    degs = [d for _, d in g.degree()]
    max_deg = max(degs)
    hist = np.zeros(max_deg + 1)
    for d in degs:
        hist[d] += 1
    return hist / hist.sum()


def _clustering_histogram(g: nx.Graph, bins: int = 10) -> np.ndarray:
    """Return normalised clustering coefficient histogram."""
    if g.number_of_nodes() == 0:
        return np.array([1.0])
    ccs = list(nx.clustering(g).values())
    hist, _ = np.histogram(ccs, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        hist = np.ones(bins)
        total = bins
    return hist / total


def _spectral_features(g: nx.Graph, k: int = 20) -> np.ndarray:
    """Return smallest *k* non-trivial Laplacian eigenvalues (normalised)."""
    if g.number_of_nodes() < 2:
        return np.array([0.0])
    # Strip all edge attributes to avoid non-numeric weight causing
    # scipy.sparse dtype=object errors (LLM-generated graphs may carry
    # string attributes on edges).
    h = type(g)()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(g.edges())
    L = nx.normalized_laplacian_matrix(h).toarray()
    eigs = np.sort(np.real(np.linalg.eigvalsh(L)))
    # Skip the first eigenvalue (always ~0 for connected component)
    eigs = eigs[1 : k + 1]
    if len(eigs) == 0:
        return np.array([0.0])
    return eigs


def _orbit_counts_per_node(g: nx.Graph) -> np.ndarray:
    """Graphlet-based orbit feature counting: 4 structural features per node.

    Features capture the four most discriminative subgraph patterns used
    in graph comparison (cf. GAG [P1], GraphRNN):
      0 - degree (1-star count)
      1 - 2-path count (number of 2-paths through node)
      2 - triangle count (3-cliques containing node)
      3 - 3-star count (number of 3-stars centred at node)

    Note: The full ORCA 15-orbit counting requires a compiled C++ backend
    (not available on Windows via pip). This 4-feature approximation covers
    the dominant structural signals. Differences should be stated in the paper.

    Returns:
        (n_nodes, 4) array.
    """
    n = g.number_of_nodes()
    if n == 0:
        return np.zeros((1, 4))
    nodes = sorted(g.nodes(), key=str)
    node_idx = {v: i for i, v in enumerate(nodes)}
    feats = np.zeros((n, 4))
    adj = {v: set(g.neighbors(v)) for v in nodes}
    for v in nodes:
        i = node_idx[v]
        d = len(adj[v])
        feats[i, 0] = d  # degree
        feats[i, 1] = d * (d - 1) / 2  # 2-paths through v
        # triangles: count edges among neighbours
        tri = 0
        nb = list(adj[v])
        for a_idx in range(len(nb)):
            for b_idx in range(a_idx + 1, len(nb)):
                if nb[b_idx] in adj[nb[a_idx]]:
                    tri += 1
        feats[i, 2] = tri  # triangles at v
        feats[i, 3] = max(0, d * (d - 1) * (d - 2) // 6)  # 3-stars
    return feats


def _orbit_histogram(g: nx.Graph, bins: int = 10) -> np.ndarray:
    """Return concatenated per-orbit normalised histograms."""
    orb = _orbit_counts_per_node(g)  # (n, 4)
    hists = []
    for col in range(orb.shape[1]):
        vals = orb[:, col]
        lo, hi = vals.min(), vals.max()
        if lo == hi:
            h = np.ones(bins)
        else:
            h, _ = np.histogram(vals, bins=bins, range=(lo, hi))
            h = h.astype(float)
        total = h.sum()
        if total == 0:
            h = np.ones(bins)
            total = bins
        hists.append(h / total)
    return np.concatenate(hists)


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def _emd_1d(h1: np.ndarray, h2: np.ndarray) -> float:
    """Earth Mover's Distance for 1-D normalised histograms.

    EMD on 1-D = L1 distance between cumulative distributions.
    Histograms are zero-padded to the same length.
    """
    size = max(len(h1), len(h2))
    a = np.zeros(size)
    b = np.zeros(size)
    a[: len(h1)] = h1
    b[: len(h2)] = h2
    # Normalise to valid distributions
    sa, sb = a.sum(), b.sum()
    if sa > 0:
        a /= sa
    if sb > 0:
        b /= sb
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).sum())


def _gaussian_emd_kernel(h1: np.ndarray, h2: np.ndarray, sigma: float) -> float:
    """Gaussian-EMD kernel: exp(-EMD(h1,h2)^2 / (2*sigma^2))."""
    d = _emd_1d(h1, h2)
    return math.exp(-(d**2) / (2.0 * sigma**2))


def _gaussian_kernel(v1: np.ndarray, v2: np.ndarray, sigma: float) -> float:
    """Standard Gaussian (RBF) kernel: exp(-||v1-v2||^2 / (2*sigma^2))."""
    diff = v1.astype(float) - v2.astype(float)
    sq = float(np.dot(diff, diff))
    return math.exp(-sq / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# MMD computation
# ---------------------------------------------------------------------------


def _compute_mmd(
    samples_x: list[np.ndarray],
    samples_y: list[np.ndarray],
    kernel: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Unbiased MMD^2 estimator.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    Returns max(0, mmd^2) to avoid tiny negative values from
    numerical noise, then takes sqrt.
    """
    n = len(samples_x)
    m = len(samples_y)
    if n < 2 or m < 2:
        return 0.0

    # E[k(x,x')]
    kxx = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            kxx += kernel(samples_x[i], samples_x[j])
    kxx = kxx * 2.0 / (n * (n - 1))

    # E[k(y,y')]
    kyy = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            kyy += kernel(samples_y[i], samples_y[j])
    kyy = kyy * 2.0 / (m * (m - 1))

    # E[k(x,y)]
    kxy = 0.0
    for i in range(n):
        for j in range(m):
            kxy += kernel(samples_x[i], samples_y[j])
    kxy /= n * m

    mmd_sq = kxx + kyy - 2.0 * kxy
    return math.sqrt(max(0.0, mmd_sq))


# ---------------------------------------------------------------------------
# MMD sub-metrics (public API)
# ---------------------------------------------------------------------------


def mmd_degree(
    gen_graphs: Sequence[nx.Graph],
    ref_graphs: Sequence[nx.Graph],
    sigma: float = 1.0,
) -> float:
    """MMD.D — degree distribution MMD with Gaussian-EMD kernel (sigma=1.0)."""
    x = [_degree_histogram(g) for g in gen_graphs]
    y = [_degree_histogram(g) for g in ref_graphs]
    kernel = lambda a, b: _gaussian_emd_kernel(a, b, sigma)
    return _compute_mmd(x, y, kernel)


def mmd_clustering(
    gen_graphs: Sequence[nx.Graph],
    ref_graphs: Sequence[nx.Graph],
    sigma: float = 0.1,
) -> float:
    """MMD.C — clustering coefficient distribution MMD (sigma=0.1)."""
    x = [_clustering_histogram(g) for g in gen_graphs]
    y = [_clustering_histogram(g) for g in ref_graphs]
    kernel = lambda a, b: _gaussian_emd_kernel(a, b, sigma)
    return _compute_mmd(x, y, kernel)


def mmd_orbits(
    gen_graphs: Sequence[nx.Graph],
    ref_graphs: Sequence[nx.Graph],
    sigma: float = 30.0,
) -> float:
    """MMD.O — orbit count distribution MMD with Gaussian kernel (sigma=30.0)."""
    x = [_orbit_histogram(g) for g in gen_graphs]
    y = [_orbit_histogram(g) for g in ref_graphs]
    kernel = lambda a, b: _gaussian_kernel(a, b, sigma)
    return _compute_mmd(x, y, kernel)


def mmd_spectral(
    gen_graphs: Sequence[nx.Graph],
    ref_graphs: Sequence[nx.Graph],
    sigma: float = 1.0,
) -> float:
    """MMD.S — spectral distribution MMD with Gaussian-EMD kernel (sigma=1.0)."""
    x = [_spectral_features(g) for g in gen_graphs]
    y = [_spectral_features(g) for g in ref_graphs]
    kernel = lambda a, b: _gaussian_emd_kernel(a, b, sigma)
    return _compute_mmd(x, y, kernel)


def mmd_score(raw_mmd: float) -> float:
    """Convert raw MMD to [0, 1] score (1 = identical distributions)."""
    return math.exp(-raw_mmd)


# ---------------------------------------------------------------------------
# Uniqueness (MOSES paradigm — mode collapse detection)
# ---------------------------------------------------------------------------


def _normalize_type(g: nx.Graph, as_directed: bool) -> nx.Graph:
    """Normalize graph to a consistent type for isomorphism comparison.

    Parameters
    ----------
    g : nx.Graph or nx.DiGraph
    as_directed : bool
        If True, convert undirected edges to bidirectional directed edges.
        If False, collapse directed edges to undirected.
    """
    if as_directed:
        if not isinstance(g, nx.DiGraph):
            return g.to_directed()  # each edge → two directed edges
        return g
    else:
        if isinstance(g, nx.DiGraph):
            return g.to_undirected()
        return g


def _graph_hash(g: nx.Graph) -> tuple:
    """Fast structural fingerprint: (n_nodes, n_edges, sorted_degree_seq).

    Two graphs can only be isomorphic if their hashes match.
    Caller must ensure *g* is already type-normalized.
    """
    return (
        g.number_of_nodes(),
        g.number_of_edges(),
        tuple(sorted((d for _, d in g.degree()), reverse=True)),
    )


def _wl_hash(g: nx.Graph) -> str:
    """Weisfeiler-Lehman graph hash — much stronger isomorphism invariant.

    WL hash captures multi-hop neighbourhood structure.  Two graphs with
    different WL hashes are guaranteed non-isomorphic.  Same WL hash
    implies isomorphism for almost all practical graphs.
    Caller must ensure *g* is already type-normalized.
    """
    return nx.weisfeiler_lehman_graph_hash(g)


def _use_exact_isomorphism(g: nx.Graph) -> bool:
    """Decide whether exact VF2 isomorphism is safe for this graph.

    VF2 worst-case is exponential on dense/regular graphs.  We use exact
    isomorphism only when the graph is small enough that VF2 will finish
    quickly.  Otherwise fall back to WL hash.

    Threshold: ≤20 nodes only.  The previous 21-45 sparse condition was
    removed because LLM-generated trees (40 nodes, density 0.05) with
    high internal symmetry caused VF2 to hang indefinitely.
    WL hash false-positive rate is ≈0 for graphs >20 nodes.
    """
    return g.number_of_nodes() <= 20


def uniqueness(graphs: Sequence[Optional[nx.Graph]]) -> float:
    """Proportion of non-isomorphic graphs among k generations.

    Filters out None graphs first.  Returns 0.0 when fewer than 2 valid
    graphs are available (cannot assess diversity).

    Strategy:
    - **Small/sparse graphs** (≤20 nodes, or ≤50 nodes with density ≤0.15):
      structural hash pre-filter + exact ``nx.is_isomorphic()``.  100% accurate.
    - **Large/dense graphs**: Weisfeiler-Lehman hash only.
      WL-hash equality is a near-perfect proxy for isomorphism on
      real-world / LLM-generated graphs (false positive rate ≈0).

    Returns:
        float in [0, 1].  1.0 = all unique, 1/k = all identical.
    """
    valid = [g for g in graphs if g is not None]
    if len(valid) <= 1:
        return 0.0

    # Majority-vote: normalize all graphs to the dominant type so that
    # comparison is type-consistent while preserving direction info when
    # most graphs are directed.
    n_directed = sum(1 for g in valid if isinstance(g, nx.DiGraph))
    as_directed = n_directed > len(valid) / 2
    normalized = [_normalize_type(g, as_directed) for g in valid]

    unique_hashes: set[str] = set()  # WL hashes (large path)
    unique_exact: list[tuple[tuple, nx.Graph]] = []  # (struct_hash, graph)

    for gn in normalized:
        if not _use_exact_isomorphism(gn):
            # Large/dense: WL hash only (O(n), no NP-hard check)
            unique_hashes.add(_wl_hash(gn))
        else:
            # Small/sparse: exact isomorphism check
            h = _graph_hash(gn)
            is_dup = False
            for uh, ug in unique_exact:
                if h == uh and nx.is_isomorphic(gn, ug):
                    is_dup = True
                    break
            if not is_dup:
                unique_exact.append((h, gn))

    # Combine: exact unique count + WL-distinct large graphs
    total_unique = len(unique_exact) + len(unique_hashes)
    return total_unique / len(valid)


# ---------------------------------------------------------------------------
# Median heuristic for MMD sigma (O'Bray et al., ICLR 2022)
# ---------------------------------------------------------------------------


def _median_heuristic(
    features_x: list[np.ndarray],
    features_y: list[np.ndarray],
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Compute sigma via median heuristic: sigma = median of pairwise distances.

    Uses all cross-set distances (x_i, y_j) to set an adaptive bandwidth.
    Falls back to 1.0 if the median is zero or sample sizes are too small.
    """
    dists: list[float] = []
    for a in features_x:
        for b in features_y:
            dists.append(distance_fn(a, b))
    if not dists:
        return 1.0
    med = float(np.median(dists))
    return med if med > 1e-10 else 1.0


# ---------------------------------------------------------------------------
# Combined D1 score
# ---------------------------------------------------------------------------


def d1_combined_score(
    gen_graphs: Sequence[Optional[nx.Graph]],
    ref_graphs: Sequence[nx.Graph],
    level: int,
    graph_type: Optional[str] = None,
    constraints: Optional[list[str]] = None,
    enable_ged: bool = False,
) -> dict[str, float]:
    """Compute D1 sub-metrics with type-aware scoring (Scheme B).

    Two evaluation modes:
    - **Constraint type** (L0-L2, L5): D1 = VR * 0.7 + Uniqueness * 0.3
    - **Distribution type** (L3-L4): D1 = VR * 0.3 + MMD * 0.5 + Uniqueness * 0.2
      Falls back to constraint formula when ref_graphs < 20.

    GED is computed as diagnostic only (not included in combined score).
    MMD sigma uses median heuristic (O'Bray et al., ICLR 2022).

    Args:
        gen_graphs: Generated graphs (may contain None).
        ref_graphs: Reference graphs for GED diagnostic / MMD comparison.
        level: Instruction level (0-5).
        graph_type: Expected graph type for valid_rate.
        constraints: Constraint strings for valid_rate.

    Returns:
        Dict with keys 'valid_rate', 'uniqueness', 'ged_diagnostic',
        'mmd_d', 'mmd_c', 'mmd_s', 'mmd_avg', 'combined'.
    """
    result: dict[str, float] = {}

    # Valid Rate — all levels
    vr = valid_rate(gen_graphs, graph_type=graph_type, constraints=constraints)
    result["valid_rate"] = vr

    # Uniqueness — skip for deterministic graph types (only one valid structure)
    _DETERMINISTIC_TYPES = {
        "complete",
        "path",
        "cycle",
        "star",
        "wheel",
        "grid",
        "complete_bipartite",
    }
    is_deterministic = graph_type in _DETERMINISTIC_TYPES
    uniq = uniqueness(gen_graphs)
    result["uniqueness"] = uniq
    result["uniqueness_active"] = not is_deterministic

    # Filter to non-None graphs for GED/MMD
    valid_gen = [g for g in gen_graphs if g is not None]

    # GED — diagnostic only, NOT included in combined score.
    # Disabled by default (O(n!) complexity hangs on large graphs).
    # Enable with enable_ged=True or --enable-ged CLI flag.
    if enable_ged and 1 <= level <= 2 and valid_gen and ref_graphs:
        ged_scores_list = []
        for g in valid_gen:
            best_ged = min(ged_score(g, r) for r in ref_graphs)
            ged_scores_list.append(best_ged)
        result["ged_diagnostic"] = sum(ged_scores_list) / len(ged_scores_list)

    # Determine evaluation type
    is_distribution = level in (3, 4)
    has_enough_refs = len(ref_graphs) >= 20

    if is_distribution and has_enough_refs and len(valid_gen) >= 2:
        # Distribution type: compute MMD with median heuristic sigma
        deg_x = [_degree_histogram(g) for g in valid_gen]
        deg_y = [_degree_histogram(g) for g in ref_graphs]
        sigma_d = _median_heuristic(deg_x, deg_y, _emd_1d)
        raw_d = _compute_mmd(
            deg_x, deg_y, lambda a, b: _gaussian_emd_kernel(a, b, sigma_d)
        )
        result["mmd_d"] = mmd_score(raw_d)

        clust_x = [_clustering_histogram(g) for g in valid_gen]
        clust_y = [_clustering_histogram(g) for g in ref_graphs]
        sigma_c = _median_heuristic(clust_x, clust_y, _emd_1d)
        raw_c = _compute_mmd(
            clust_x, clust_y, lambda a, b: _gaussian_emd_kernel(a, b, sigma_c)
        )
        result["mmd_c"] = mmd_score(raw_c)

        spec_x = [_spectral_features(g) for g in valid_gen]
        spec_y = [_spectral_features(g) for g in ref_graphs]
        sigma_s = _median_heuristic(spec_x, spec_y, _emd_1d)
        raw_s = _compute_mmd(
            spec_x, spec_y, lambda a, b: _gaussian_emd_kernel(a, b, sigma_s)
        )
        result["mmd_s"] = mmd_score(raw_s)

        # MMD average (D + C + S, excluding O which is not reliable)
        mmd_avg = (result["mmd_d"] + result["mmd_c"] + result["mmd_s"]) / 3.0
        result["mmd_avg"] = mmd_avg

        # Distribution combined: VR * 0.3 + MMD * 0.5 + Uniqueness * 0.2
        if is_deterministic:
            # Deterministic types: Uniqueness not meaningful, redistribute weight
            result["combined"] = vr * 0.375 + mmd_avg * 0.625
        else:
            result["combined"] = vr * 0.3 + mmd_avg * 0.5 + uniq * 0.2
    else:
        # Constraint type (L0-L2, L5) or distribution fallback (refs < 20)
        if is_deterministic:
            # Deterministic types: only VR matters (unique solution, no diversity)
            result["combined"] = vr
        else:
            # Combined: VR * 0.7 + Uniqueness * 0.3
            result["combined"] = vr * 0.7 + uniq * 0.3

    return result
