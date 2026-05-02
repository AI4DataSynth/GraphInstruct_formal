"""Tests for L4/L5 evaluation improvements.

Covers:
- Phase 0a: apply_graph_edit with prune_tree isolated node fix
- Phase 0b: ensure_directed_from_raw
- Phase 1: edit_quality (closeness + redundancy)
- Phase 2: domain_structural_fit (6-facet structural similarity)
- L4/L5 instruction_score integration
"""

import unittest

import networkx as nx

from graphinstruct.metrics.l5_eval import (
    apply_graph_edit,
    edit_quality,
    instruction_score_l5,
    reasoning_alignment,
)
from graphinstruct.metrics.l4_eval import (
    domain_structural_fit,
    ensure_directed_from_raw,
    instruction_score_l4,
)
from graphinstruct.parser.code_style import recover_directed_edges


class TestApplyGraphEdit(unittest.TestCase):
    """Phase 0a: apply_graph_edit with prune_tree fix."""

    def test_basic_remove(self):
        """Basic edge removal works."""
        G = nx.path_graph(5)
        result = apply_graph_edit(G, add_edges=[], remove_edges=[(1, 2)])
        self.assertFalse(result.has_edge(1, 2))
        self.assertEqual(result.number_of_nodes(), 5)

    def test_basic_add(self):
        """Basic edge addition works."""
        G = nx.path_graph(3)
        result = apply_graph_edit(G, add_edges=[(0, 2)], remove_edges=[])
        self.assertTrue(result.has_edge(0, 2))

    def test_does_not_mutate_original(self):
        """Original graph is not modified."""
        G = nx.path_graph(5)
        original_edges = G.number_of_edges()
        apply_graph_edit(G, add_edges=[], remove_edges=[(1, 2)])
        self.assertEqual(G.number_of_edges(), original_edges)

    def test_prune_tree_removes_isolated(self):
        """Phase 0a: prune_tree task removes isolated nodes after edge deletion."""
        # Tree: 0-1-2-3-4-5-6-7 (path)
        G = nx.path_graph(8)
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        # Remove edges to nodes 6 and 7
        result = apply_graph_edit(
            G,
            add_edges=[],
            remove_edges=[("5", "6"), ("6", "7")],
            task_type="prune_tree",
        )
        # Nodes 6 and 7 should be removed (isolated after edge deletion)
        self.assertNotIn("6", result.nodes())
        self.assertNotIn("7", result.nodes())
        self.assertEqual(result.number_of_nodes(), 6)

    def test_extract_subgraph_removes_isolated(self):
        """extract_subgraph also removes isolated nodes."""
        G = nx.star_graph(4)
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        result = apply_graph_edit(
            G,
            add_edges=[],
            remove_edges=[("0", "3"), ("0", "4")],
            task_type="extract_subgraph",
        )
        self.assertNotIn("3", result.nodes())
        self.assertNotIn("4", result.nodes())

    def test_non_prune_keeps_isolated(self):
        """Non-prune tasks keep isolated nodes."""
        G = nx.path_graph(5)
        G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes()})
        result = apply_graph_edit(
            G,
            add_edges=[],
            remove_edges=[("0", "1")],
            task_type="add_edges",
        )
        # Node 0 becomes isolated but should NOT be removed
        self.assertIn("0", result.nodes())
        self.assertEqual(result.number_of_nodes(), 5)


class TestEnsureDirectedFromRaw(unittest.TestCase):
    """Phase 0b: directed attribute inference from raw output."""

    def test_already_directed(self):
        """DiGraph is returned unchanged."""
        DG = nx.DiGraph()
        DG.add_edges_from([("a", "b"), ("b", "c")])
        inst = {"explicit_constraints": ["directed=true"]}
        result, converted = ensure_directed_from_raw(DG, inst, "")
        self.assertFalse(converted)
        self.assertIsInstance(result, nx.DiGraph)

    def test_no_directed_constraint(self):
        """Undirected graph without directed=true constraint is unchanged."""
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c")])
        inst = {"explicit_constraints": ["num_nodes=3"]}
        result, converted = ensure_directed_from_raw(G, inst, "")
        self.assertFalse(converted)
        self.assertNotIsInstance(result, nx.DiGraph)

    def test_recover_from_raw_output(self):
        """Recovers directed edges from raw_output edge_list."""
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.add_edges_from([("a", "b"), ("b", "c")])
        inst = {"explicit_constraints": ["directed=true"]}
        raw = """Graph[name='test', nodes=3] {
    node_list = ['a', 'b', 'c'];
    edge_list = [('a','b'), ('b','c')];
}"""
        result, converted = ensure_directed_from_raw(G, inst, raw)
        self.assertTrue(converted)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertTrue(result.has_edge("a", "b"))
        self.assertTrue(result.has_edge("b", "c"))
        # Direction should be preserved: a->b exists but b->a should not
        self.assertFalse(result.has_edge("b", "a"))

    def test_fallback_simple_convert(self):
        """Falls back to simple conversion when edge_list not found."""
        G = nx.Graph()
        G.add_edges_from([("a", "b")])
        inst = {"explicit_constraints": ["directed=true"]}
        raw = "some output without edge_list"
        result, converted = ensure_directed_from_raw(G, inst, raw)
        self.assertTrue(converted)
        self.assertIsInstance(result, nx.DiGraph)

    def test_preserves_node_attributes(self):
        """Node attributes are preserved after conversion."""
        G = nx.Graph()
        G.add_node("a", label="Alice")
        G.add_node("b", label="Bob")
        G.add_edge("a", "b")
        inst = {"explicit_constraints": ["directed=true"]}
        raw = "edge_list = [('a','b')]"
        result, _ = ensure_directed_from_raw(G, inst, raw)
        self.assertEqual(result.nodes["a"]["label"], "Alice")
        self.assertEqual(result.nodes["b"]["label"], "Bob")


class TestRecoverDirectedEdges(unittest.TestCase):
    """Parser-level recover_directed_edges (called by evaluate.py globally)."""

    def test_recovers_from_edge_list(self):
        """Recovers directed edges from raw_output edge_list."""
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c")])
        raw = "edge_list = [('a','b'), ('b','c')]"
        DG, from_raw = recover_directed_edges(G, raw)
        self.assertIsInstance(DG, nx.DiGraph)
        self.assertTrue(from_raw)
        self.assertTrue(DG.has_edge("a", "b"))
        self.assertFalse(DG.has_edge("b", "a"))

    def test_fallback_without_edge_list(self):
        """Falls back to simple conversion when no edge_list found."""
        G = nx.Graph()
        G.add_edges_from([("x", "y")])
        DG, from_raw = recover_directed_edges(G, "no edges here")
        self.assertIsInstance(DG, nx.DiGraph)
        self.assertFalse(from_raw)

    def test_preserves_graph_attrs(self):
        """Graph-level attributes are preserved."""
        G = nx.Graph()
        G.add_edge("a", "b")
        G.graph["domain"] = "social"
        DG, _ = recover_directed_edges(G, "edge_list = [('a','b')]")
        self.assertEqual(DG.graph["domain"], "social")


class TestEditQuality(unittest.TestCase):
    """Phase 1: edit_quality (closeness + redundancy)."""

    def test_perfect_edit(self):
        """Exact match with reference yields high score."""
        base = nx.path_graph(5)
        base = nx.relabel_nodes(base, {i: str(i) for i in base.nodes()})
        # Reference removes edge (2,3)
        ref = base.copy()
        ref.remove_edge("2", "3")
        # Generated also removes edge (2,3) — perfect
        gen = base.copy()
        gen.remove_edge("2", "3")
        score = edit_quality(gen, base, [ref])
        self.assertGreater(score, 0.9)

    def test_too_few_edits(self):
        """Too few edits (weak model) gets penalized."""
        base = nx.complete_graph(6)
        base = nx.relabel_nodes(base, {i: str(i) for i in base.nodes()})
        # Reference removes 10 edges
        ref = base.copy()
        edges_to_remove = list(ref.edges())[:10]
        ref.remove_edges_from(edges_to_remove)
        # Generated removes only 3 edges
        gen = base.copy()
        gen.remove_edges_from(edges_to_remove[:3])
        score = edit_quality(gen, base, [ref])
        # ratio = 3/10 = 0.3, closeness = max(0.2, 0.3) = 0.3
        self.assertLess(score, 0.7)

    def test_no_edits_when_edits_needed(self):
        """No edits when reference requires edits → low closeness."""
        base = nx.path_graph(5)
        base = nx.relabel_nodes(base, {i: str(i) for i in base.nodes()})
        ref = base.copy()
        ref.remove_edge("2", "3")
        # Generated is same as base — no edits
        gen = base.copy()
        score = edit_quality(gen, base, [ref])
        self.assertLess(score, 0.6)

    def test_redundant_adds_penalized(self):
        """Adding edges that already exist in base is penalized."""

        class FakeParsed:
            parse_type = "GraphEdit"
            add_edges = [("0", "1"), ("1", "2")]  # both already in base

        base = nx.path_graph(5)
        base = nx.relabel_nodes(base, {i: str(i) for i in base.nodes()})
        ref = base.copy()
        gen = base.copy()
        score_redundant = edit_quality(gen, base, [ref], parsed_output=FakeParsed())
        score_clean = edit_quality(gen, base, [ref], parsed_output=None)
        self.assertLess(score_redundant, score_clean)


class TestDomainStructuralFit(unittest.TestCase):
    """Phase 2: domain_structural_fit (6-facet structural similarity)."""

    def test_identical_graph(self):
        """Identical graph to reference yields score close to 1.0."""
        G = nx.barabasi_albert_graph(20, 2, seed=42)
        score = domain_structural_fit(G, [G])
        self.assertGreater(score, 0.95)

    def test_very_different_graph(self):
        """Very different graph (star vs complete) scores low."""
        star = nx.star_graph(19)
        complete = nx.complete_graph(20)
        score = domain_structural_fit(star, [complete])
        self.assertLess(score, 0.7)

    def test_empty_references(self):
        """No references yields 1.0."""
        G = nx.path_graph(5)
        score = domain_structural_fit(G, [])
        self.assertEqual(score, 1.0)

    def test_best_match_selected(self):
        """Max score across multiple references is returned."""
        G = nx.barabasi_albert_graph(20, 2, seed=42)
        ref_similar = nx.barabasi_albert_graph(20, 2, seed=43)
        ref_different = nx.complete_graph(20)
        score = domain_structural_fit(G, [ref_different, ref_similar])
        score_similar_only = domain_structural_fit(G, [ref_similar])
        # Including the similar ref should give at least as good a score
        self.assertGreaterEqual(score, score_similar_only - 0.01)

    def test_directed_graph(self):
        """Works with directed graphs (uses reciprocity facet)."""
        DG = nx.DiGraph()
        DG.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 3)])
        ref = nx.DiGraph()
        ref.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])
        score = domain_structural_fit(DG, [ref])
        self.assertGreater(score, 0.5)


class TestReasoningAlignment(unittest.TestCase):
    """reasoning_alignment format compliance scoring."""

    def test_graphedit_format(self):
        """GraphEdit format gets highest format score."""
        raw = "GraphEdit[name='test'] { add_edges = []; }"
        score = reasoning_alignment(raw, strategy="zero-shot")
        self.assertGreater(score, 0.9)

    def test_graph_format(self):
        """Graph format gets medium format score."""
        raw = "Graph[name='test'] { node_list = ['a']; }"
        score = reasoning_alignment(raw, strategy="zero-shot")
        self.assertGreater(score, 0.7)
        self.assertLess(score, 1.0)

    def test_no_format(self):
        """No recognized format gets low score."""
        raw = "Here is the result: nodes=[1,2,3]"
        score = reasoning_alignment(raw, strategy="zero-shot")
        self.assertLess(score, 0.6)

    def test_cot_with_reasoning(self):
        """CoT with reasoning keywords scores decently."""
        raw = (
            "I need to remove edge (1,2) and add edge (3,4). GraphEdit[name='test'] {}"
        )
        score = reasoning_alignment(raw, strategy="zero-cot")
        self.assertGreater(score, 0.5)


class TestInstructionScoreL4(unittest.TestCase):
    """L4 integrated scoring."""

    def test_basic_l4_score(self):
        """L4 score with all constraints satisfied and good structural fit."""
        G = nx.barabasi_albert_graph(20, 2, seed=42)
        constraints = [f"num_nodes={G.number_of_nodes()}", "connected=true"]
        score = instruction_score_l4(G, constraints, reference_graphs=[G])
        self.assertGreater(score, 0.8)

    def test_none_graph(self):
        """None graph yields 0."""
        score = instruction_score_l4(None, ["num_nodes=10"])
        self.assertEqual(score, 0.0)

    def test_with_directed_graph(self):
        """L4 score works with directed graphs (fix applied upstream in evaluate.py)."""
        DG = nx.DiGraph()
        DG.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        constraints = ["directed=true", "num_nodes=4"]
        score = instruction_score_l4(DG, constraints)
        self.assertGreater(score, 0.0)


class TestInstructionScoreL5(unittest.TestCase):
    """L5 integrated scoring."""

    def test_perfect_l5_score(self):
        """L5 with perfect edits and constraints yields high score."""
        base = nx.path_graph(5)
        base = nx.relabel_nodes(base, {i: str(i) for i in base.nodes()})
        ref = base.copy()
        ref.remove_edge("2", "3")
        gen = base.copy()
        gen.remove_edge("2", "3")
        constraints = [f"num_nodes={gen.number_of_nodes()}", "connected=false"]
        raw = "GraphEdit[name='test'] { remove_edges = [('2','3')]; }"
        score = instruction_score_l5(
            gen,
            constraints,
            base_graph=base,
            reference_graphs=[ref],
            raw_output=raw,
            strategy="zero-shot",
        )
        self.assertGreater(score, 0.7)

    def test_none_graph(self):
        """None graph yields 0."""
        score = instruction_score_l5(None, ["num_nodes=5"])
        self.assertEqual(score, 0.0)

    def test_no_base_no_ref(self):
        """Without base/ref, edit_quality defaults to 0.5."""
        G = nx.path_graph(5)
        constraints = [f"num_nodes={G.number_of_nodes()}", "connected=true"]
        score = instruction_score_l5(
            G, constraints, raw_output="GraphEdit[name='t'] {}"
        )
        # Should still produce a reasonable score
        self.assertGreater(score, 0.3)


if __name__ == "__main__":
    unittest.main()
