"""Tests for D1 structural quality metrics."""

import math
import unittest

import numpy as np
import networkx as nx

from graphinstruct.metrics.structural import (
    _ged_fallback,
    d1_combined_score,
    ged,
    ged_score,
    mmd_clustering,
    mmd_degree,
    mmd_orbits,
    mmd_score,
    mmd_spectral,
    valid_rate,
    valid_rate_single,
    _degree_histogram,
    _clustering_histogram,
    _spectral_features,
    _orbit_counts_per_node,
    _orbit_histogram,
    _emd_1d,
    _gaussian_emd_kernel,
    _gaussian_kernel,
    _compute_mmd,
)


class TestValidRateSingle(unittest.TestCase):
    """Test single-graph validity check against constraints."""

    def test_valid_tree(self):
        G = nx.path_graph(5)
        self.assertTrue(valid_rate_single(G, graph_type="tree"))

    def test_invalid_tree(self):
        G = nx.cycle_graph(5)
        self.assertFalse(valid_rate_single(G, graph_type="tree"))

    def test_valid_cycle(self):
        G = nx.cycle_graph(6)
        self.assertTrue(valid_rate_single(G, graph_type="cycle"))

    def test_valid_complete(self):
        G = nx.complete_graph(4)
        self.assertTrue(valid_rate_single(G, graph_type="complete"))

    def test_valid_with_constraints(self):
        G = nx.path_graph(5)
        self.assertTrue(
            valid_rate_single(
                G, graph_type="tree", constraints=["num_nodes=5", "num_edges=4"]
            )
        )

    def test_constraint_ignored_by_vr(self):
        """VR no longer checks constraints (Plan C: D4 handles constraints)."""
        G = nx.path_graph(5)
        # Wrong num_nodes constraint — VR should still pass (type is correct)
        self.assertTrue(
            valid_rate_single(G, graph_type="tree", constraints=["num_nodes=10"])
        )

    def test_no_graph_type(self):
        """When no graph_type given, any parseable graph is valid."""
        G = nx.path_graph(5)
        self.assertTrue(valid_rate_single(G, constraints=["num_nodes=5"]))

    def test_none_graph(self):
        """None graph (parse failure) should be invalid."""
        self.assertFalse(valid_rate_single(None, graph_type="tree"))


class TestValidRate(unittest.TestCase):
    """Test batch valid rate computation."""

    def test_all_valid(self):
        graphs = [nx.path_graph(5), nx.star_graph(4), nx.path_graph(3)]
        rate = valid_rate(graphs, graph_type="tree")
        self.assertAlmostEqual(rate, 1.0)

    def test_half_valid(self):
        graphs = [nx.path_graph(5), nx.cycle_graph(5)]
        rate = valid_rate(graphs, graph_type="tree")
        self.assertAlmostEqual(rate, 0.5)

    def test_none_valid(self):
        graphs = [nx.cycle_graph(5), nx.cycle_graph(6)]
        rate = valid_rate(graphs, graph_type="tree")
        self.assertAlmostEqual(rate, 0.0)

    def test_empty_list(self):
        rate = valid_rate([], graph_type="tree")
        self.assertAlmostEqual(rate, 0.0)

    def test_with_none_entries(self):
        """None entries (parse failures) should count as invalid."""
        graphs = [nx.path_graph(5), None, nx.star_graph(3)]
        rate = valid_rate(graphs, graph_type="tree")
        self.assertAlmostEqual(rate, 2.0 / 3.0)

    def test_with_constraints_ignored(self):
        """VR ignores constraints (Plan C). Both are valid trees → VR=1.0."""
        graphs = [nx.path_graph(5), nx.path_graph(3)]
        rate = valid_rate(graphs, graph_type="tree", constraints=["num_nodes=5"])
        # Both are trees → VR=1.0 (constraints not checked by VR)
        self.assertAlmostEqual(rate, 1.0)


# ---------------------------------------------------------------------------
# GED tests
# ---------------------------------------------------------------------------


class TestGED(unittest.TestCase):
    def test_identical_graphs(self):
        G = nx.path_graph(4)
        self.assertAlmostEqual(ged(G, G), 0.0)

    def test_different_graphs(self):
        G1 = nx.path_graph(4)
        G2 = nx.cycle_graph(4)
        d = ged(G1, G2)
        self.assertGreater(d, 0)

    def test_empty_graph(self):
        G1 = nx.Graph()
        G2 = nx.path_graph(3)
        d = ged(G1, G2)
        # Should be num_nodes + num_edges = 3 + 2 = 5
        self.assertGreater(d, 0)

    def test_ged_score_identical(self):
        G = nx.path_graph(5)
        self.assertAlmostEqual(ged_score(G, G), 1.0)

    def test_ged_score_different(self):
        G1 = nx.path_graph(5)
        G2 = nx.complete_graph(5)
        s = ged_score(G1, G2)
        self.assertGreater(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_ged_score_range(self):
        G1 = nx.star_graph(4)
        G2 = nx.cycle_graph(5)
        s = ged_score(G1, G2)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_ged_large_graph_uses_fallback(self):
        """Graphs above max_nodes should use structural fallback."""
        G1 = nx.complete_graph(15)
        G2 = nx.path_graph(15)
        d = ged(G1, G2, max_nodes=10)
        expected = abs(G1.number_of_edges() - G2.number_of_edges())
        self.assertAlmostEqual(d, expected)

    def test_ged_fallback_identical(self):
        G = nx.path_graph(5)
        self.assertAlmostEqual(_ged_fallback(G, G), 0.0)

    def test_ged_fallback_empty_vs_nonempty(self):
        G1 = nx.Graph()
        G2 = nx.path_graph(3)
        # 3 nodes + 2 edges = 5
        self.assertAlmostEqual(_ged_fallback(G1, G2), 5.0)

    def test_ged_fallback_same_size_different_structure(self):
        """Fallback returns 0 for same-size graphs (known limitation)."""
        G1 = nx.cycle_graph(6)
        G2 = nx.path_graph(6)  # 6 nodes, 5 edges vs 6 nodes, 6 edges
        self.assertAlmostEqual(_ged_fallback(G1, G2), 1.0)  # |6-5| edges


# ---------------------------------------------------------------------------
# Distribution extractor tests
# ---------------------------------------------------------------------------


class TestDistributionExtractors(unittest.TestCase):
    def test_degree_histogram_path(self):
        G = nx.path_graph(4)  # degrees: 1, 2, 2, 1
        h = _degree_histogram(G)
        self.assertAlmostEqual(h.sum(), 1.0)
        # Index 1 should have 0.5, index 2 should have 0.5
        self.assertAlmostEqual(h[1], 0.5)
        self.assertAlmostEqual(h[2], 0.5)

    def test_degree_histogram_empty(self):
        G = nx.Graph()
        h = _degree_histogram(G)
        self.assertEqual(len(h), 1)

    def test_clustering_histogram_complete(self):
        G = nx.complete_graph(5)  # All clustering coefficients = 1.0
        h = _clustering_histogram(G)
        self.assertAlmostEqual(h.sum(), 1.0)

    def test_clustering_histogram_empty(self):
        G = nx.Graph()
        h = _clustering_histogram(G)
        self.assertEqual(len(h), 1)

    def test_spectral_features_path(self):
        G = nx.path_graph(5)
        eigs = _spectral_features(G)
        self.assertGreater(len(eigs), 0)
        # All eigenvalues should be non-negative
        self.assertTrue(np.all(eigs >= -1e-10))

    def test_spectral_features_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        eigs = _spectral_features(G)
        self.assertEqual(len(eigs), 1)

    def test_orbit_counts_triangle(self):
        G = nx.cycle_graph(3)  # Triangle
        orb = _orbit_counts_per_node(G)
        self.assertEqual(orb.shape, (3, 4))
        # Each node has degree 2
        self.assertTrue(np.all(orb[:, 0] == 2))
        # Each node participates in 1 triangle
        self.assertTrue(np.all(orb[:, 2] == 1))

    def test_orbit_counts_empty(self):
        G = nx.Graph()
        orb = _orbit_counts_per_node(G)
        self.assertEqual(orb.shape, (1, 4))

    def test_orbit_histogram_shape(self):
        G = nx.path_graph(5)
        h = _orbit_histogram(G, bins=10)
        # 4 features * 10 bins = 40
        self.assertEqual(len(h), 40)
        # Should be normalised per orbit
        for i in range(4):
            segment = h[i * 10 : (i + 1) * 10]
            self.assertAlmostEqual(segment.sum(), 1.0, places=5)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


class TestKernels(unittest.TestCase):
    def test_emd_identical(self):
        h = np.array([0.5, 0.3, 0.2])
        self.assertAlmostEqual(_emd_1d(h, h), 0.0)

    def test_emd_different(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 1.0])
        d = _emd_1d(h1, h2)
        self.assertGreater(d, 0)

    def test_emd_different_lengths(self):
        h1 = np.array([0.5, 0.5])
        h2 = np.array([0.25, 0.25, 0.25, 0.25])
        d = _emd_1d(h1, h2)
        self.assertGreaterEqual(d, 0)

    def test_gaussian_emd_kernel_identical(self):
        h = np.array([0.5, 0.3, 0.2])
        k = _gaussian_emd_kernel(h, h, sigma=1.0)
        self.assertAlmostEqual(k, 1.0)

    def test_gaussian_emd_kernel_range(self):
        h1 = np.array([1.0, 0.0])
        h2 = np.array([0.0, 1.0])
        k = _gaussian_emd_kernel(h1, h2, sigma=1.0)
        self.assertGreater(k, 0.0)
        self.assertLessEqual(k, 1.0)

    def test_gaussian_kernel_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        k = _gaussian_kernel(v, v, sigma=30.0)
        self.assertAlmostEqual(k, 1.0)

    def test_gaussian_kernel_range(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([10.0, 10.0])
        k = _gaussian_kernel(v1, v2, sigma=1.0)
        self.assertGreater(k, 0.0)
        self.assertLessEqual(k, 1.0)


# ---------------------------------------------------------------------------
# MMD tests
# ---------------------------------------------------------------------------


class TestMMD(unittest.TestCase):
    def test_mmd_identical_distributions(self):
        """MMD of identical graph sets should be near 0."""
        graphs = [nx.path_graph(5), nx.path_graph(6), nx.path_graph(7)]
        raw = mmd_degree(graphs, graphs)
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_mmd_different_distributions(self):
        """MMD of different graph types should be > 0."""
        gen = [nx.complete_graph(5), nx.complete_graph(6), nx.complete_graph(7)]
        ref = [nx.path_graph(5), nx.path_graph(6), nx.path_graph(7)]
        raw = mmd_degree(gen, ref)
        self.assertGreater(raw, 0.0)

    def test_mmd_clustering_identical(self):
        graphs = [nx.cycle_graph(6), nx.cycle_graph(7), nx.cycle_graph(8)]
        raw = mmd_clustering(graphs, graphs)
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_mmd_orbits_identical(self):
        graphs = [nx.path_graph(5), nx.path_graph(6)]
        raw = mmd_orbits(graphs, graphs)
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_mmd_spectral_identical(self):
        graphs = [nx.path_graph(5), nx.path_graph(6)]
        raw = mmd_spectral(graphs, graphs)
        self.assertAlmostEqual(raw, 0.0, places=5)

    def test_mmd_score_identical(self):
        self.assertAlmostEqual(mmd_score(0.0), 1.0)

    def test_mmd_score_large(self):
        s = mmd_score(10.0)
        self.assertLess(s, 0.001)

    def test_mmd_score_moderate(self):
        s = mmd_score(0.5)
        self.assertAlmostEqual(s, math.exp(-0.5))


class TestComputeMMDEdgeCases(unittest.TestCase):
    def test_too_few_samples(self):
        """Less than 2 samples should return 0."""
        x = [np.array([1.0])]
        y = [np.array([1.0]), np.array([2.0])]
        result = _compute_mmd(x, y, lambda a, b: 1.0)
        self.assertAlmostEqual(result, 0.0)

    def test_empty_samples(self):
        result = _compute_mmd([], [], lambda a, b: 1.0)
        self.assertAlmostEqual(result, 0.0)


# ---------------------------------------------------------------------------
# Combined D1 score tests
# ---------------------------------------------------------------------------


class TestD1CombinedScore(unittest.TestCase):
    def test_level0_only_valid_rate_and_uniqueness(self):
        """L0: constraint type — VR + Uniqueness, no GED/MMD."""
        graphs = [nx.path_graph(5), nx.path_graph(5)]
        result = d1_combined_score(graphs, [], level=0, graph_type="tree")
        self.assertIn("valid_rate", result)
        self.assertIn("uniqueness", result)
        self.assertNotIn("ged_diagnostic", result)
        self.assertNotIn("mmd_d", result)

    def test_level1_with_ged_diagnostic(self):
        """L1: constraint type — GED is diagnostic only (when enabled)."""
        gen = [nx.path_graph(5), nx.path_graph(5)]
        ref = [nx.path_graph(5)]
        result = d1_combined_score(
            gen, ref, level=1, graph_type="tree", enable_ged=True
        )
        self.assertIn("valid_rate", result)
        self.assertIn("ged_diagnostic", result)
        self.assertAlmostEqual(result["ged_diagnostic"], 1.0)  # identical
        # Combined should NOT include GED
        expected = result["valid_rate"] * 0.7 + result["uniqueness"] * 0.3
        self.assertAlmostEqual(result["combined"], expected)

    def test_level2_constraint_type(self):
        """L2: constraint type — VR*0.7 + Uniq*0.3, no MMD."""
        gen = [nx.path_graph(5), nx.path_graph(5), nx.path_graph(5)]
        ref = [nx.path_graph(5), nx.path_graph(5)]
        result = d1_combined_score(gen, ref, level=2, graph_type="tree")
        self.assertIn("valid_rate", result)
        self.assertIn("uniqueness", result)
        # L2 is constraint type, NOT distribution
        self.assertNotIn("mmd_d", result)

    def test_level3_distribution_with_pool(self):
        """L3 with >= 20 refs: distribution type — includes MMD."""
        gen = [nx.path_graph(5), nx.path_graph(6), nx.path_graph(7)]
        ref = [nx.path_graph(i + 4) for i in range(25)]  # 25 refs >= 20 threshold
        result = d1_combined_score(gen, ref, level=3)
        self.assertIn("mmd_d", result)
        self.assertIn("mmd_c", result)
        self.assertIn("mmd_s", result)
        self.assertIn("mmd_avg", result)
        # Distribution formula: VR*0.3 + MMD*0.5 + Uniq*0.2
        expected = (
            result["valid_rate"] * 0.3
            + result["mmd_avg"] * 0.5
            + result["uniqueness"] * 0.2
        )
        self.assertAlmostEqual(result["combined"], expected)

    def test_combined_in_range(self):
        """Combined score should be in [0, 1]."""
        gen = [nx.path_graph(5), nx.complete_graph(4), None]
        ref = [nx.path_graph(5)]
        result = d1_combined_score(gen, ref, level=1, graph_type="tree")
        self.assertGreaterEqual(result["combined"], 0.0)
        self.assertLessEqual(result["combined"], 1.0)

    def test_all_none_graphs(self):
        """All None graphs: valid_rate=0, uniqueness=0, combined=0."""
        result = d1_combined_score([None, None], [nx.path_graph(3)], level=2)
        self.assertAlmostEqual(result["valid_rate"], 0.0)
        self.assertAlmostEqual(result["uniqueness"], 0.0)
        self.assertAlmostEqual(result["combined"], 0.0)

    def test_no_refs_skips_ged_mmd(self):
        """No reference graphs: constraint formula used."""
        gen = [nx.path_graph(5), nx.path_graph(5)]
        result = d1_combined_score(gen, [], level=2, graph_type="tree")
        self.assertNotIn("ged_diagnostic", result)
        self.assertNotIn("mmd_d", result)
        expected = result["valid_rate"] * 0.7 + result["uniqueness"] * 0.3
        self.assertAlmostEqual(result["combined"], expected)


class TestUniqueness(unittest.TestCase):
    """Test uniqueness() — mode collapse detection."""

    def test_all_unique(self):
        """5 non-isomorphic graphs → 1.0."""
        from graphinstruct.metrics.structural import uniqueness

        graphs = [nx.path_graph(i + 3) for i in range(5)]
        self.assertAlmostEqual(uniqueness(graphs), 1.0)

    def test_all_identical(self):
        """5 identical graphs → 0.2."""
        from graphinstruct.metrics.structural import uniqueness

        g = nx.path_graph(5)
        graphs = [g.copy() for _ in range(5)]
        self.assertAlmostEqual(uniqueness(graphs), 1 / 5)

    def test_some_duplicates(self):
        """A, A, B → 2/3."""
        from graphinstruct.metrics.structural import uniqueness

        a = nx.path_graph(5)
        b = nx.cycle_graph(5)
        graphs = [a.copy(), a.copy(), b.copy()]
        self.assertAlmostEqual(uniqueness(graphs), 2 / 3)

    def test_single_graph(self):
        """1 valid graph → 0.0 (cannot assess diversity)."""
        from graphinstruct.metrics.structural import uniqueness

        self.assertAlmostEqual(uniqueness([nx.path_graph(3)]), 0.0)

    def test_all_none(self):
        """All None → 0.0."""
        from graphinstruct.metrics.structural import uniqueness

        self.assertAlmostEqual(uniqueness([None, None, None]), 0.0)

    def test_mixed_none(self):
        """Some None filtered out: [None, A, A, None, B] → 2/3."""
        from graphinstruct.metrics.structural import uniqueness

        a = nx.path_graph(4)
        b = nx.star_graph(3)
        graphs = [None, a.copy(), a.copy(), None, b.copy()]
        # 3 valid, 2 unique → 2/3
        self.assertAlmostEqual(uniqueness(graphs), 2 / 3)

    def test_mixed_directed_undirected(self):
        """Minority type normalized to majority — same structure → duplicate."""
        from graphinstruct.metrics.structural import uniqueness

        # 1 DiGraph + 1 Graph → majority is tie, as_directed = False
        # Both become undirected path → 1 unique out of 2
        g_undir = nx.path_graph(5)
        g_dir = nx.DiGraph(nx.path_graph(5))
        graphs = [g_undir, g_dir]
        self.assertAlmostEqual(uniqueness(graphs), 1 / 2)

    def test_mixed_directed_different(self):
        """Different structures remain unique after normalization."""
        from graphinstruct.metrics.structural import uniqueness

        g_undir = nx.path_graph(5)
        g_dir = nx.DiGraph(nx.cycle_graph(6))
        graphs = [g_undir, g_dir]
        self.assertAlmostEqual(uniqueness(graphs), 1.0)

    def test_majority_directed_preserves_direction(self):
        """When majority are DiGraph, direction info is preserved."""
        from graphinstruct.metrics.structural import uniqueness

        # star-out (0→1, 0→2) and star-in (1→0, 2→0) have same undirected
        # structure but are NOT directed-isomorphic.  With 2 DiGraphs
        # (majority), direction is preserved → all three unique.
        g1 = nx.DiGraph([(0, 1), (0, 2)])  # star-out
        g2 = nx.DiGraph([(1, 0), (2, 0)])  # star-in
        g3 = nx.star_graph(2)  # minority undirected
        graphs = [g1, g2, g3]
        # g3 → to_directed() → bidirectional star (4 edges, different hash)
        # g1, g2, g3_dir all non-isomorphic → 3/3 = 1.0
        self.assertAlmostEqual(uniqueness(graphs), 1.0)

    def test_all_directed_same(self):
        """All DiGraphs with same structure → detected as duplicates."""
        from graphinstruct.metrics.structural import uniqueness

        g1 = nx.DiGraph([(0, 1), (1, 2)])
        g2 = nx.DiGraph([(0, 1), (1, 2)])
        graphs = [g1, g2]
        self.assertAlmostEqual(uniqueness(graphs), 1 / 2)


class TestD1TypeAware(unittest.TestCase):
    """Test type-aware D1 combined score (Scheme B)."""

    def test_constraint_type_formula(self):
        """L0-L2, L5: D1 = VR * 0.7 + Uniqueness * 0.3."""
        from graphinstruct.metrics.structural import d1_combined_score

        # 3 valid path graphs, all unique
        graphs = [nx.path_graph(5), nx.path_graph(6), nx.path_graph(7)]
        result = d1_combined_score(graphs, [], level=1, graph_type="path")
        # VR = 1.0 (all valid paths), Uniqueness = 1.0 (all different sizes)
        self.assertAlmostEqual(result["valid_rate"], 1.0)
        self.assertAlmostEqual(result["uniqueness"], 1.0)
        self.assertAlmostEqual(result["combined"], 0.7 * 1.0 + 0.3 * 1.0)

    def test_constraint_type_l5(self):
        """L5 with deterministic type: Uniqueness excluded from combined."""
        from graphinstruct.metrics.structural import d1_combined_score

        graphs = [nx.path_graph(5), nx.path_graph(5)]
        result = d1_combined_score(graphs, [], level=5, graph_type="path")
        # path is deterministic → Uniqueness not in combined
        self.assertAlmostEqual(result["uniqueness"], 0.5)
        self.assertFalse(result["uniqueness_active"])
        # Combined = VR only (deterministic type)
        self.assertAlmostEqual(result["combined"], 1.0)

    def test_deterministic_type_uniqueness_excluded(self):
        """Deterministic types (complete/path/cycle/star/wheel/grid/cb):
        Uniqueness computed but NOT included in combined score."""
        from graphinstruct.metrics.structural import d1_combined_score

        for gt in [
            "complete",
            "path",
            "cycle",
            "star",
            "wheel",
            "grid",
            "complete_bipartite",
        ]:
            graphs = [nx.path_graph(5), nx.path_graph(5)]
            result = d1_combined_score(graphs, [], level=1, graph_type=gt)
            self.assertFalse(
                result["uniqueness_active"], f"{gt} should be deterministic"
            )
            # Combined should be VR only
            self.assertAlmostEqual(result["combined"], result["valid_rate"], msg=gt)

    def test_nondeterministic_type_uniqueness_included(self):
        """Non-deterministic types: Uniqueness participates in combined."""
        from graphinstruct.metrics.structural import d1_combined_score

        for gt in ["tree", "forest", "bipartite", "dag", "connected", None]:
            g1 = nx.path_graph(5)
            g2 = nx.star_graph(4)
            graphs = [g1, g2]
            result = d1_combined_score(graphs, [], level=1, graph_type=gt)
            self.assertTrue(
                result["uniqueness_active"], f"{gt} should be non-deterministic"
            )
            # Combined = VR * 0.7 + Uniqueness * 0.3
            expected = result["valid_rate"] * 0.7 + result["uniqueness"] * 0.3
            self.assertAlmostEqual(result["combined"], expected, msg=gt)

    def test_ged_is_diagnostic_only(self):
        """GED should appear as ged_diagnostic, not in combined."""
        from graphinstruct.metrics.structural import d1_combined_score

        ref = [nx.path_graph(5)]
        gen = [nx.path_graph(5), nx.star_graph(4)]
        result = d1_combined_score(gen, ref, level=1, graph_type=None, enable_ged=True)
        self.assertIn("ged_diagnostic", result)
        # Combined should be VR*0.7 + Uniq*0.3, NOT include GED
        expected = result["valid_rate"] * 0.7 + result["uniqueness"] * 0.3
        self.assertAlmostEqual(result["combined"], expected)

    def test_distribution_fallback_no_refs(self):
        """L3 with no refs → falls back to constraint formula."""
        from graphinstruct.metrics.structural import d1_combined_score

        graphs = [nx.path_graph(5), nx.cycle_graph(5)]
        result = d1_combined_score(graphs, [], level=3)
        # No refs → constraint fallback
        self.assertNotIn("mmd_d", result)
        expected = result["valid_rate"] * 0.7 + result["uniqueness"] * 0.3
        self.assertAlmostEqual(result["combined"], expected)


class TestMedianHeuristic(unittest.TestCase):
    """Test _median_heuristic for MMD sigma."""

    def test_basic(self):
        from graphinstruct.metrics.structural import _median_heuristic, _emd_1d

        x = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        y = [np.array([0.5, 0.5]), np.array([0.3, 0.7])]
        sigma = _median_heuristic(x, y, _emd_1d)
        self.assertGreater(sigma, 0)

    def test_fallback_on_empty(self):
        from graphinstruct.metrics.structural import _median_heuristic, _emd_1d

        sigma = _median_heuristic([], [], _emd_1d)
        self.assertAlmostEqual(sigma, 1.0)


if __name__ == "__main__":
    unittest.main()
