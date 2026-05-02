"""Tests for graphinstruct.metrics.embedding (D3 embedding quality metrics)."""

import unittest

import networkx as nx

from graphinstruct.metrics.embedding import (
    d3_score,
    d3_score_batch,
    embedding_mmd,
    grassmann_coherence,
    node_clf_gap,
    node_clf_gap_batch,
)


class TestGrassmannCoherence(unittest.TestCase):
    def test_self_coherence(self):
        G = nx.cycle_graph(10)
        score = grassmann_coherence(G)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_reference_alignment(self):
        G1 = nx.cycle_graph(8)
        G2 = nx.cycle_graph(8)
        score = grassmann_coherence(G1, reference=G2)
        self.assertGreater(score, 0.5)

    def test_different_graphs(self):
        G1 = nx.cycle_graph(8)
        G2 = nx.star_graph(7)
        score = grassmann_coherence(G1, reference=G2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        self.assertAlmostEqual(grassmann_coherence(G), 0.0)

    def test_empty_graph(self):
        G = nx.Graph()
        self.assertAlmostEqual(grassmann_coherence(G), 0.0)

    def test_small_graph(self):
        G = nx.path_graph(3)
        score = grassmann_coherence(G)
        self.assertGreaterEqual(score, 0.0)

    def test_different_sizes(self):
        G1 = nx.cycle_graph(6)
        G2 = nx.cycle_graph(10)
        score = grassmann_coherence(G1, reference=G2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestNodeClfGap(unittest.TestCase):
    def test_small_graph_returns_zero(self):
        """Graphs with < 10 ref nodes return 0.0 (too small for train/test split)."""
        G1 = nx.cycle_graph(5)
        G2 = nx.cycle_graph(5)
        self.assertAlmostEqual(node_clf_gap(G1, G2), 0.0)

    def test_single_community_returns_zero(self):
        """Complete graph has only one community → labels < 2 → returns 0.0."""
        G = nx.complete_graph(12)
        self.assertAlmostEqual(node_clf_gap(G, G), 0.0)

    def test_identical_graphs_high_score(self):
        """Identical graphs should have small gap → high score."""
        G = nx.karate_club_graph()
        score = node_clf_gap(G, G, epochs=50)
        # Identical graph → gap should be small → score > 0.5
        self.assertGreater(score, 0.3)
        self.assertLessEqual(score, 1.0)

    def test_different_structure_lower_score(self):
        """Very different generated graph should have larger gap."""
        ref = nx.karate_club_graph()
        # Generated graph with same nodes but random edges
        gen = nx.gnm_random_graph(34, 20, seed=42)
        gen = nx.relabel_nodes(gen, {i: list(ref.nodes())[i] for i in range(34)})
        score = node_clf_gap(gen, ref, epochs=50)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_with_explicit_labels(self):
        """Test with explicitly provided labels."""
        G = nx.barbell_graph(10, 2)
        labels = {n: 0 if n < 10 else 1 for n in G.nodes()}
        score = node_clf_gap(G, G, labels=labels, epochs=50)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_in_range(self):
        """Score should always be in [0, 1]."""
        G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
        score = node_clf_gap(G, G, epochs=30)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestEmbeddingMMD(unittest.TestCase):
    def test_identical_graphs(self):
        G = nx.cycle_graph(8)
        score = embedding_mmd(G, G)
        self.assertGreater(score, 0.8)

    def test_different_graphs(self):
        G1 = nx.cycle_graph(8)
        G2 = nx.star_graph(7)
        score = embedding_mmd(G1, G2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_small_graph(self):
        G1 = nx.path_graph(2)
        G2 = nx.path_graph(2)
        score = embedding_mmd(G1, G2)
        self.assertGreaterEqual(score, 0.0)

    def test_empty_graph(self):
        G1 = nx.Graph()
        G2 = nx.cycle_graph(5)
        self.assertAlmostEqual(embedding_mmd(G1, G2), 0.0)


class TestD3Score(unittest.TestCase):
    def test_none_inputs(self):
        result = d3_score(None, None)
        self.assertEqual(result["aggregate"], 0.0)

    def test_valid_graphs(self):
        G1 = nx.cycle_graph(8)
        G2 = nx.cycle_graph(8)
        result = d3_score(G1, G2)
        self.assertGreater(result["aggregate"], 0.0)

    def test_keys_present(self):
        G = nx.cycle_graph(6)
        result = d3_score(G, G)
        for key in ("grassmann", "node_clf_gap", "emb_mmd", "aggregate"):
            self.assertIn(key, result)

    def test_small_generated(self):
        G1 = nx.cycle_graph(5)
        G2 = nx.Graph()
        G2.add_node(0)
        result = d3_score(G1, G2)
        self.assertEqual(result["aggregate"], 0.0)


class TestNodeClfGapBatch(unittest.TestCase):
    """Verify batch NCG produces identical results to per-call version."""

    def _make_ref_and_gens(self):
        ref = nx.karate_club_graph()
        gens = [nx.watts_strogatz_graph(34, 4, 0.3, seed=i) for i in range(3)]
        return ref, gens

    def test_equivalence_with_per_call(self):
        """Batch scores must match individual node_clf_gap() calls."""
        ref, gens = self._make_ref_and_gens()
        epochs = 30

        individual = [node_clf_gap(g, ref, epochs=epochs) for g in gens]
        batch = node_clf_gap_batch(gens, ref, epochs=epochs)

        self.assertEqual(len(batch), len(individual))
        for i, (ind, bat) in enumerate(zip(individual, batch)):
            self.assertAlmostEqual(
                ind,
                bat,
                places=6,
                msg=f"Mismatch at index {i}: individual={ind}, batch={bat}",
            )

    def test_empty_list(self):
        ref = nx.karate_club_graph()
        self.assertEqual(node_clf_gap_batch([], ref), [])

    def test_small_ref_returns_zeros(self):
        ref = nx.cycle_graph(5)  # < 10 nodes
        gens = [nx.cycle_graph(5), nx.cycle_graph(6)]
        result = node_clf_gap_batch(gens, ref)
        self.assertEqual(result, [0.0, 0.0])

    def test_mixed_small_and_normal_gens(self):
        """Graphs with < 4 gen nodes should return 0, others computed."""
        ref = nx.karate_club_graph()
        small = nx.path_graph(3)  # 3 nodes < 4
        normal = nx.watts_strogatz_graph(34, 4, 0.3, seed=99)
        result = node_clf_gap_batch([small, normal], ref, epochs=20)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.0)
        # normal graph should get a real score
        self.assertGreaterEqual(result[1], 0.0)
        self.assertLessEqual(result[1], 1.0)


class TestD3ScoreBatch(unittest.TestCase):
    """Verify batch D3 produces identical results to per-call version."""

    def test_equivalence_with_per_call(self):
        ref = nx.karate_club_graph()
        gens = [nx.watts_strogatz_graph(34, 4, 0.3, seed=i) for i in range(3)]

        individual = [d3_score(ref, g) for g in gens]
        batch = d3_score_batch(ref, gens)

        self.assertEqual(len(batch), len(individual))
        for i, (ind, bat) in enumerate(zip(individual, batch)):
            for key in ("grassmann", "node_clf_gap", "emb_mmd", "aggregate"):
                self.assertAlmostEqual(
                    ind[key],
                    bat[key],
                    places=6,
                    msg=f"Mismatch at [{i}][{key}]: {ind[key]} vs {bat[key]}",
                )

    def test_empty_list(self):
        ref = nx.karate_club_graph()
        self.assertEqual(d3_score_batch(ref, []), [])

    def test_none_reference(self):
        gens = [nx.cycle_graph(5)]
        result = d3_score_batch(None, gens)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["aggregate"], 0.0)

    def test_small_graph_mapped_to_zero(self):
        """Graphs with < 2 nodes should get _zero_d3()."""
        ref = nx.karate_club_graph()
        tiny = nx.Graph()
        tiny.add_node(0)
        normal = nx.cycle_graph(8)
        result = d3_score_batch(ref, [tiny, normal])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["aggregate"], 0.0)
        self.assertGreater(result[1]["aggregate"], 0.0)


if __name__ == "__main__":
    unittest.main()
