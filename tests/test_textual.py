"""Tests for graphinstruct.metrics.textual (D2 text quality metrics)."""

import unittest

import networkx as nx

from graphinstruct.metrics.textual import (
    _fallback_bertscore,
    d2_score,
    extract_triples,
    g_bleu,
    g_rouge,
    text_f1,
    text_presence,
    text_similarity,
    triple_to_sentence,
)


def _make_labeled_graph(edges):
    """Create a graph with labeled nodes and edges.

    edges: list of (u, u_label, v, v_label, edge_label)
    """
    G = nx.Graph()
    for u, u_label, v, v_label, edge_label in edges:
        G.add_node(u, label=u_label)
        G.add_node(v, label=v_label)
        G.add_edge(u, v, label=edge_label)
    return G


class TestExtractTriples(unittest.TestCase):
    def test_basic_extraction(self):
        G = _make_labeled_graph(
            [
                (0, "Alice", 1, "Bob", "knows"),
            ]
        )
        triples = extract_triples(G)
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0], ("Alice", "knows", "Bob"))

    def test_fallback_labels(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        triples = extract_triples(G)
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0][0], "0")
        self.assertEqual(triples[0][2], "1")
        self.assertEqual(triples[0][1], "connected_to")

    def test_edge_type_attribute(self):
        G = nx.Graph()
        G.add_node(0, label="A")
        G.add_node(1, label="B")
        G.add_edge(0, 1, type="friend_of")
        triples = extract_triples(G)
        self.assertEqual(triples[0][1], "friend_of")

    def test_empty_graph(self):
        G = nx.Graph()
        self.assertEqual(extract_triples(G), [])


class TestTripleToSentence(unittest.TestCase):
    def test_basic(self):
        s = triple_to_sentence(("Alice", "knows", "Bob"))
        self.assertIn("Alice", s)
        self.assertIn("Bob", s)
        self.assertIn("knows", s)


class TestGBLEU(unittest.TestCase):
    def test_identical_graphs(self):
        G = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel1"),
                (1, "B", 2, "C", "rel2"),
            ]
        )
        score = g_bleu(G, G)
        self.assertAlmostEqual(score, 1.0)

    def test_completely_different(self):
        ref = _make_labeled_graph([(0, "X", 1, "Y", "r1")])
        gen = _make_labeled_graph([(0, "M", 1, "N", "r2")])
        score = g_bleu(ref, gen)
        self.assertLessEqual(score, 0.5)

    def test_empty_graphs(self):
        ref = nx.Graph()
        gen = nx.Graph()
        self.assertEqual(g_bleu(ref, gen), 0.0)

    def test_range(self):
        ref = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel"),
                (1, "B", 2, "C", "rel"),
            ]
        )
        gen = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel"),
            ]
        )
        score = g_bleu(ref, gen)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestGROUGE(unittest.TestCase):
    def test_identical_graphs(self):
        G = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel1"),
                (1, "B", 2, "C", "rel2"),
            ]
        )
        score = g_rouge(G, G)
        self.assertAlmostEqual(score, 1.0)

    def test_no_overlap(self):
        ref = _make_labeled_graph([(0, "X", 1, "Y", "r1")])
        gen = _make_labeled_graph([(0, "M", 1, "N", "r2")])
        score = g_rouge(ref, gen)
        self.assertLessEqual(score, 0.5)

    def test_empty_reference(self):
        ref = nx.Graph()
        gen = _make_labeled_graph([(0, "A", 1, "B", "rel")])
        self.assertEqual(g_rouge(ref, gen), 0.0)


class TestTextF1(unittest.TestCase):
    def test_identical(self):
        G = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel"),
            ]
        )
        self.assertAlmostEqual(text_f1(G, G), 1.0)

    def test_partial_overlap(self):
        ref = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel1"),
                (1, "B", 2, "C", "rel2"),
            ]
        )
        gen = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel1"),
                (1, "B", 2, "D", "rel3"),
            ]
        )
        score = text_f1(ref, gen)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_both_empty(self):
        self.assertAlmostEqual(text_f1(nx.Graph(), nx.Graph()), 1.0)

    def test_one_empty(self):
        G = _make_labeled_graph([(0, "A", 1, "B", "rel")])
        self.assertAlmostEqual(text_f1(G, nx.Graph()), 0.0)


class TestFallbackBERTScore(unittest.TestCase):
    def test_identical(self):
        triples = [("A", "rel", "B"), ("B", "rel", "C")]
        score = _fallback_bertscore(triples, triples)
        self.assertAlmostEqual(score, 1.0)

    def test_different(self):
        ref = [("X", "r1", "Y")]
        gen = [("M", "r2", "N")]
        score = _fallback_bertscore(ref, gen)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestD2Score(unittest.TestCase):
    def test_none_inputs(self):
        result = d2_score(None, None)
        self.assertEqual(result["aggregate"], 0.0)

    def test_identical_graphs(self):
        G = _make_labeled_graph(
            [
                (0, "A", 1, "B", "rel"),
                (1, "B", 2, "C", "rel"),
            ]
        )
        result = d2_score(G, G, use_bert=False)
        self.assertGreater(result["aggregate"], 0.5)

    def test_both_empty_edges(self):
        ref = nx.Graph()
        ref.add_node(0)
        gen = nx.Graph()
        gen.add_node(0)
        result = d2_score(ref, gen)
        # Both empty: text_presence drives aggregate
        self.assertIn("text_presence", result)

    def test_keys_present(self):
        G = _make_labeled_graph([(0, "A", 1, "B", "rel")])
        result = d2_score(G, G)
        for key in (
            "g_bertscore",
            "g_bleu",
            "g_rouge",
            "text_f1",
            "text_presence",
            "text_similarity",
            "aggregate",
        ):
            self.assertIn(key, result)


class TestTextPresence(unittest.TestCase):
    """Tests for v2 D2 text_presence."""

    def test_empty_graph(self):
        G = nx.Graph()
        self.assertEqual(text_presence(G), 0.0)

    def test_label_attr_full(self):
        """All nodes have label attr + edges have explicit type."""
        G = nx.Graph()
        G.add_node("0", label="Alice")
        G.add_node("1", label="Bob")
        G.add_edge("0", "1")
        G.graph["edge_type"] = "friends"
        self.assertAlmostEqual(text_presence(G), 1.0)

    def test_semantic_id_no_label(self):
        """Semantic node IDs (non-numeric) but no label attr."""
        G = nx.Graph()
        G.add_edge("Alice", "Bob")
        score = text_presence(G)
        # 0.6 * 0.7 + 0.4 * 0 = 0.42
        self.assertAlmostEqual(score, 0.42)

    def test_numeric_id_no_label(self):
        """Numeric node IDs, no label attr, no edge type = 0."""
        G = nx.Graph()
        G.add_edge("0", "1")
        self.assertEqual(text_presence(G), 0.0)

    def test_mixed(self):
        """Mix of label attr and semantic/numeric IDs."""
        G = nx.Graph()
        G.add_node("0", label="Alice")  # has label
        G.add_node("Bob")  # semantic ID
        G.add_node("2")  # numeric ID
        G.add_edge("0", "Bob")
        G.add_edge("Bob", "2")
        score = text_presence(G)
        # node: (1*1.0 + 1*0.7 + 0) / 3 = 0.567
        # edge: 0 (no edge_type)
        # total: 0.6 * 0.567 + 0.4 * 0 = 0.34
        self.assertAlmostEqual(score, 0.34, places=2)


class TestTextSimilarity(unittest.TestCase):
    """Tests for v2 D2 text_similarity."""

    def test_identical_labels(self):
        """Same label vocabulary yields 1.0."""
        G = nx.Graph()
        G.add_node("0", label="Alice")
        G.add_node("1", label="Bob")
        G.add_edge("0", "1")
        score = text_similarity(G, G, use_bert=False)
        self.assertAlmostEqual(score, 1.0)

    def test_no_overlap(self):
        """Completely different label vocabulary."""
        ref = nx.Graph()
        ref.add_node("0", label="Alice")
        ref.add_node("1", label="Bob")
        ref.add_edge("0", "1")
        gen = nx.Graph()
        gen.add_node("0", label="Carol")
        gen.add_node("1", label="Dave")
        gen.add_edge("0", "1")
        score = text_similarity(gen, ref, use_bert=False)
        self.assertEqual(score, 0.0)

    def test_numeric_ids_no_labels(self):
        """Both graphs have only numeric IDs -> 0."""
        ref = nx.Graph()
        ref.add_edge("0", "1")
        gen = nx.Graph()
        gen.add_edge("0", "1")
        score = text_similarity(gen, ref, use_bert=False)
        self.assertEqual(score, 0.0)

    def test_gen_semantic_ref_label(self):
        """Gen uses semantic IDs, ref uses label attrs."""
        ref = nx.Graph()
        ref.add_node("0", label="Alice")
        ref.add_node("1", label="Bob")
        ref.add_edge("0", "1")
        gen = nx.Graph()
        gen.add_edge("Alice", "Bob")  # semantic IDs matching ref labels
        score = text_similarity(gen, ref, use_bert=False)
        self.assertAlmostEqual(score, 1.0)

    def test_partial_overlap(self):
        """Partial vocabulary overlap."""
        ref = nx.Graph()
        ref.add_node("0", label="Alice")
        ref.add_node("1", label="Bob")
        ref.add_node("2", label="Carol")
        ref.add_edge("0", "1")
        ref.add_edge("1", "2")
        gen = nx.Graph()
        gen.add_edge("Alice", "Dave")  # Alice matches, Dave doesn't
        score = text_similarity(gen, ref, use_bert=False)
        # Jaccard: {alice} & {alice, bob, carol} / {alice, bob, carol, dave} = 1/4
        self.assertAlmostEqual(score, 0.25)


if __name__ == "__main__":
    unittest.main()
