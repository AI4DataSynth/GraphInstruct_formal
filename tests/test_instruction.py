"""Tests for D4 instruction matching metrics."""

import unittest

import networkx as nx

from graphinstruct.metrics.instruction import (
    explicit_satisfaction,
    implicit_satisfaction,
    no_contradiction,
    instruction_score,
)


class TestExplicitSatisfaction(unittest.TestCase):
    """Test explicit constraint satisfaction rate."""

    def test_all_satisfied(self):
        G = nx.path_graph(5)
        constraints = ["num_nodes=5", "num_edges=4", "connected=true"]
        self.assertAlmostEqual(explicit_satisfaction(G, constraints), 1.0)

    def test_partial(self):
        G = nx.path_graph(5)
        constraints = ["num_nodes=5", "num_nodes=10"]
        self.assertAlmostEqual(explicit_satisfaction(G, constraints), 0.5)

    def test_none_satisfied(self):
        G = nx.path_graph(5)
        constraints = ["num_nodes=10", "num_edges=10"]
        self.assertAlmostEqual(explicit_satisfaction(G, constraints), 0.0)

    def test_empty_constraints(self):
        G = nx.path_graph(5)
        self.assertAlmostEqual(explicit_satisfaction(G, []), 1.0)

    def test_none_graph(self):
        self.assertAlmostEqual(explicit_satisfaction(None, ["num_nodes=5"]), 0.0)


class TestImplicitSatisfaction(unittest.TestCase):
    """Test implicit constraint satisfaction rate."""

    def test_tree_implicit(self):
        """A valid tree should satisfy all implicit constraints."""
        G = nx.path_graph(10)
        explicit = ["graph_type=tree", "num_nodes=10"]
        self.assertAlmostEqual(implicit_satisfaction(G, explicit), 1.0)

    def test_tree_implicit_fail(self):
        """A cycle labeled as tree should fail implicit constraints."""
        G = nx.cycle_graph(10)
        explicit = ["graph_type=tree", "num_nodes=10"]
        # cycle has 10 edges (not 9), is not acyclic
        self.assertLess(implicit_satisfaction(G, explicit), 1.0)

    def test_no_implicit(self):
        """If no implicit constraints can be inferred, return 1.0."""
        G = nx.path_graph(5)
        explicit = ["num_nodes=5"]  # no graph_type, no implicit
        self.assertAlmostEqual(implicit_satisfaction(G, explicit), 1.0)

    def test_none_graph(self):
        explicit = ["graph_type=tree", "num_nodes=5"]
        self.assertAlmostEqual(implicit_satisfaction(None, explicit), 0.0)


class TestNoContradiction(unittest.TestCase):
    """Test no-contradiction check."""

    def test_consistent_undirected(self):
        """Undirected graph with no contradictions."""
        G = nx.path_graph(5)
        self.assertTrue(no_contradiction(G))

    def test_self_loops(self):
        """Graph with self-loops is a contradiction for simple graphs."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 0)  # self-loop
        self.assertFalse(no_contradiction(G))

    def test_empty_graph(self):
        G = nx.Graph()
        self.assertTrue(no_contradiction(G))

    def test_none_graph(self):
        self.assertFalse(no_contradiction(None))

    def test_directed_graph_ok(self):
        """Directed graph without issues."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        self.assertTrue(no_contradiction(G))

    def test_digraph_unidirectional_ok(self):
        """DiGraph with unidirectional edges is fine (it's directed)."""
        G = nx.DiGraph()
        G.add_edge(0, 1)
        # Directed graph: unidirectional is expected
        self.assertTrue(no_contradiction(G))

    def test_undirected_always_symmetric(self):
        """nx.Graph always stores edges symmetrically — should pass."""
        G = nx.Graph()
        G.add_edge(0, 1)
        self.assertTrue(no_contradiction(G))


class TestInstructionScore(unittest.TestCase):
    """Test combined D4 instruction score."""

    def test_perfect_score(self):
        G = nx.path_graph(10)
        explicit = ["graph_type=tree", "num_nodes=10"]
        score = instruction_score(G, explicit)
        self.assertAlmostEqual(score, 1.0)

    def test_zero_score(self):
        score = instruction_score(None, ["graph_type=tree"])
        self.assertAlmostEqual(score, 0.0)

    def test_partial_score(self):
        G = nx.cycle_graph(10)
        explicit = ["graph_type=tree", "num_nodes=10"]
        score = instruction_score(G, explicit)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_no_implicit_redistributes_weight(self):
        """When no implicit constraints exist, weight is redistributed."""
        G = nx.cycle_graph(5)
        # num_nodes=5 + connected=true → no graph_type → no implicit constraints
        explicit = ["num_nodes=5", "connected=true"]
        score = instruction_score(G, explicit)
        # Expected: (0.4/0.6)*explicit_sat + (0.2/0.6)*1.0
        # Both constraints satisfied → explicit_sat=1.0, no_contradiction=1.0
        # = 0.667*1.0 + 0.333*1.0 = 1.0
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_no_implicit_partial_explicit(self):
        """Redistribution with partial explicit satisfaction."""
        G = nx.cycle_graph(5)
        # num_nodes wrong (5 != 10), connected correct → 1/2 satisfied
        explicit = ["num_nodes=10", "connected=true"]
        score = instruction_score(G, explicit)
        # explicit_sat=0.5, no_contradiction=1.0
        # = 0.667*0.5 + 0.333*1.0 = 0.333 + 0.333 = 0.667
        self.assertAlmostEqual(score, 2.0 / 3.0, places=3)

    def test_no_implicit_none_graph(self):
        """None graph + no implicit → explicit_sat=0, contradiction=0."""
        score = instruction_score(None, ["num_nodes=5", "connected=true"])
        # explicit_sat=0.0, no_contradiction=0.0
        # = 0.667*0.0 + 0.333*0.0 = 0.0
        self.assertAlmostEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
