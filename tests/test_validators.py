"""Tests for graph type validators and constraint checking."""

import unittest

import networkx as nx

from graphinstruct.validators import (
    check_all_constraints,
    check_constraint,
    infer_implicit_constraints,
    is_acyclic,
    is_bipartite,
    is_complete,
    is_complete_bipartite,
    is_connected,
    is_cycle,
    is_forest,
    is_path,
    is_planar,
    is_regular,
    is_star,
    is_tree,
    satisfaction_rate,
)


class TestIsTree(unittest.TestCase):
    def test_valid_tree(self):
        G = nx.path_graph(5)
        self.assertTrue(is_tree(G))

    def test_star_is_tree(self):
        G = nx.star_graph(4)
        self.assertTrue(is_tree(G))

    def test_cycle_is_not_tree(self):
        G = nx.cycle_graph(5)
        self.assertFalse(is_tree(G))

    def test_disconnected_is_not_tree(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        self.assertFalse(is_tree(G))


class TestIsCycle(unittest.TestCase):
    def test_valid_cycle(self):
        G = nx.cycle_graph(6)
        self.assertTrue(is_cycle(G))

    def test_path_is_not_cycle(self):
        G = nx.path_graph(5)
        self.assertFalse(is_cycle(G))

    def test_triangle(self):
        G = nx.cycle_graph(3)
        self.assertTrue(is_cycle(G))

    def test_too_small(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        self.assertFalse(is_cycle(G))


class TestIsPath(unittest.TestCase):
    def test_valid_path(self):
        G = nx.path_graph(5)
        self.assertTrue(is_path(G))

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        self.assertTrue(is_path(G))

    def test_cycle_is_not_path(self):
        G = nx.cycle_graph(5)
        self.assertFalse(is_path(G))

    def test_star_is_not_path(self):
        G = nx.star_graph(3)
        self.assertFalse(is_path(G))


class TestIsStar(unittest.TestCase):
    def test_valid_star(self):
        G = nx.star_graph(5)
        self.assertTrue(is_star(G))

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        self.assertTrue(is_star(G))

    def test_path_is_not_star(self):
        G = nx.path_graph(4)
        self.assertFalse(is_star(G))


class TestIsComplete(unittest.TestCase):
    def test_k4(self):
        G = nx.complete_graph(4)
        self.assertTrue(is_complete(G))

    def test_k1(self):
        G = nx.complete_graph(1)
        self.assertTrue(is_complete(G))

    def test_missing_edge(self):
        G = nx.complete_graph(4)
        G.remove_edge(0, 1)
        self.assertFalse(is_complete(G))


class TestIsBipartite(unittest.TestCase):
    def test_complete_bipartite(self):
        G = nx.complete_bipartite_graph(3, 4)
        self.assertTrue(is_bipartite(G))

    def test_odd_cycle_not_bipartite(self):
        G = nx.cycle_graph(5)
        self.assertFalse(is_bipartite(G))

    def test_even_cycle_is_bipartite(self):
        G = nx.cycle_graph(6)
        self.assertTrue(is_bipartite(G))


class TestIsCompleteBipartite(unittest.TestCase):
    def test_k3_4(self):
        G = nx.complete_bipartite_graph(3, 4)
        self.assertTrue(is_complete_bipartite(G))

    def test_incomplete_bipartite(self):
        G = nx.complete_bipartite_graph(3, 4)
        edges = list(G.edges())
        G.remove_edge(*edges[0])
        self.assertFalse(is_complete_bipartite(G))


class TestIsRegular(unittest.TestCase):
    def test_cycle_is_2_regular(self):
        G = nx.cycle_graph(6)
        self.assertTrue(is_regular(G, k=2))

    def test_complete_k4_is_3_regular(self):
        G = nx.complete_graph(4)
        self.assertTrue(is_regular(G, k=3))

    def test_any_regular(self):
        G = nx.cycle_graph(5)
        self.assertTrue(is_regular(G))

    def test_not_regular(self):
        G = nx.path_graph(3)
        self.assertFalse(is_regular(G))


class TestIsPlanar(unittest.TestCase):
    def test_planar_graph(self):
        G = nx.cycle_graph(5)
        self.assertTrue(is_planar(G))

    def test_k5_not_planar(self):
        G = nx.complete_graph(5)
        self.assertFalse(is_planar(G))

    def test_k33_not_planar(self):
        G = nx.complete_bipartite_graph(3, 3)
        self.assertFalse(is_planar(G))


class TestIsConnectedAndAcyclic(unittest.TestCase):
    def test_connected(self):
        G = nx.path_graph(5)
        self.assertTrue(is_connected(G))

    def test_disconnected(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        self.assertFalse(is_connected(G))

    def test_acyclic_tree(self):
        G = nx.path_graph(5)
        self.assertTrue(is_acyclic(G))

    def test_cycle_not_acyclic(self):
        G = nx.cycle_graph(5)
        self.assertFalse(is_acyclic(G))

    def test_forest_is_acyclic(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        self.assertTrue(is_acyclic(G))
        self.assertTrue(is_forest(G))


class TestCheckConstraint(unittest.TestCase):
    def test_num_nodes(self):
        G = nx.path_graph(5)
        self.assertTrue(check_constraint(G, "num_nodes=5"))
        self.assertFalse(check_constraint(G, "num_nodes=4"))

    def test_num_edges(self):
        G = nx.path_graph(5)
        self.assertTrue(check_constraint(G, "num_edges=4"))

    def test_graph_type_tree(self):
        G = nx.path_graph(5)
        self.assertTrue(check_constraint(G, "graph_type=tree"))

    def test_graph_type_cycle(self):
        G = nx.cycle_graph(5)
        self.assertTrue(check_constraint(G, "graph_type=cycle"))
        self.assertFalse(check_constraint(G, "graph_type=tree"))

    def test_connected_bool(self):
        G = nx.path_graph(3)
        self.assertTrue(check_constraint(G, "connected=true"))

    def test_degree_regular(self):
        G = nx.cycle_graph(6)
        self.assertTrue(check_constraint(G, "degree=2"))

    def test_min_degree(self):
        G = nx.complete_graph(4)
        self.assertTrue(check_constraint(G, "min_degree>=3"))
        self.assertTrue(check_constraint(G, "min_degree=3"))

    def test_max_degree(self):
        G = nx.star_graph(4)
        self.assertTrue(check_constraint(G, "max_degree=4"))

    def test_diameter(self):
        G = nx.path_graph(5)
        self.assertTrue(check_constraint(G, "diameter<=4"))
        self.assertTrue(check_constraint(G, "diameter=4"))


class TestBipartiteSidesConstraint(unittest.TestCase):
    def test_connected_bipartite_sides(self):
        G = nx.complete_bipartite_graph(3, 4)
        self.assertTrue(check_constraint(G, "bipartite_sides=3,4"))
        self.assertTrue(check_constraint(G, "bipartite_sides=4,3"))

    def test_wrong_sides(self):
        G = nx.complete_bipartite_graph(3, 4)
        self.assertFalse(check_constraint(G, "bipartite_sides=2,5"))

    def test_disconnected_bipartite(self):
        """Disconnected bipartite graph should not crash (AmbiguousSolution fix)."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two separate edges, bipartite
        self.assertTrue(check_constraint(G, "bipartite_sides=2,2"))

    def test_non_bipartite_returns_false(self):
        G = nx.cycle_graph(5)  # Odd cycle — not bipartite
        self.assertFalse(check_constraint(G, "bipartite_sides=2,3"))


class TestDegreeConstraintNonRegular(unittest.TestCase):
    def test_degree_equality_fails_for_non_regular(self):
        G = nx.path_graph(4)  # Degrees: 1, 2, 2, 1 — not regular
        self.assertFalse(check_constraint(G, "degree=2"))

    def test_degree_ge_fails_for_non_regular(self):
        """degree>=k on non-regular graph should not match sentinel."""
        G = nx.path_graph(4)
        self.assertFalse(check_constraint(G, "degree>=0"))

    def test_degree_le_fails_for_non_regular(self):
        G = nx.path_graph(4)
        self.assertFalse(check_constraint(G, "degree<=100"))


class TestCheckAllConstraints(unittest.TestCase):
    def test_tree_constraints(self):
        G = nx.path_graph(5)
        results = check_all_constraints(
            G,
            [
                "graph_type=tree",
                "num_nodes=5",
                "num_edges=4",
                "connected=true",
                "acyclic=true",
            ],
        )
        self.assertTrue(all(results.values()))

    def test_mixed_results(self):
        G = nx.cycle_graph(5)
        results = check_all_constraints(
            G,
            [
                "graph_type=cycle",
                "num_nodes=5",
                "graph_type=tree",  # should fail
            ],
        )
        self.assertTrue(results["graph_type=cycle"])
        self.assertTrue(results["num_nodes=5"])
        self.assertFalse(results["graph_type=tree"])


class TestSatisfactionRate(unittest.TestCase):
    def test_all_satisfied(self):
        G = nx.path_graph(5)
        rate = satisfaction_rate(G, ["num_nodes=5", "num_edges=4"])
        self.assertAlmostEqual(rate, 1.0)

    def test_half_satisfied(self):
        G = nx.path_graph(5)
        rate = satisfaction_rate(G, ["num_nodes=5", "num_nodes=10"])
        self.assertAlmostEqual(rate, 0.5)

    def test_empty_constraints(self):
        G = nx.Graph()
        rate = satisfaction_rate(G, [])
        self.assertAlmostEqual(rate, 1.0)


class TestInferImplicitConstraints(unittest.TestCase):
    def test_tree_implicit(self):
        explicit = ["graph_type=tree", "num_nodes=10"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("acyclic=true", implicit)
        self.assertIn("connected=true", implicit)
        self.assertIn("num_edges=9", implicit)

    def test_cycle_implicit(self):
        explicit = ["graph_type=cycle", "num_nodes=8"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("connected=true", implicit)
        self.assertIn("degree=2", implicit)
        self.assertIn("num_edges=8", implicit)

    def test_complete_implicit(self):
        explicit = ["graph_type=complete", "num_nodes=5"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("num_edges=10", implicit)
        self.assertIn("degree=4", implicit)

    def test_star_implicit(self):
        explicit = ["graph_type=star", "num_nodes=6"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("connected=true", implicit)
        self.assertIn("num_edges=5", implicit)

    def test_path_implicit(self):
        explicit = ["graph_type=path", "num_nodes=5"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("connected=true", implicit)
        self.assertIn("acyclic=true", implicit)
        self.assertIn("num_edges=4", implicit)

    def test_no_duplicate_with_explicit(self):
        """Implicit constraints already in explicit list should be excluded."""
        explicit = ["graph_type=tree", "num_nodes=10", "connected=true"]
        implicit = infer_implicit_constraints(explicit)
        self.assertNotIn("connected=true", implicit)

    def test_unknown_type_returns_empty(self):
        explicit = ["graph_type=unknown"]
        implicit = infer_implicit_constraints(explicit)
        self.assertEqual(implicit, [])

    def test_validate_inferred_constraints(self):
        """Inferred constraints should actually be satisfied by a valid graph."""
        # Build a tree with 10 nodes
        # Build a tree using Prufer sequence
        G = nx.from_prufer_sequence([0, 1, 2, 3, 4, 5, 6, 7])
        explicit = ["graph_type=tree", "num_nodes=10"]
        implicit = infer_implicit_constraints(explicit)

        # All implicit constraints should be satisfied
        results = check_all_constraints(G, implicit)
        for constraint, satisfied in results.items():
            self.assertTrue(satisfied, f"Tree failed implicit constraint: {constraint}")

    def test_bipartite_implicit(self):
        """Bipartite graph should imply bipartite=true."""
        explicit = ["graph_type=bipartite", "num_nodes=10"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("bipartite=true", implicit)

    def test_bipartite_implicit_with_sides(self):
        """Bipartite with known sides should imply max edge count."""
        explicit = ["graph_type=bipartite", "num_nodes=7", "bipartite_sides=3,4"]
        implicit = infer_implicit_constraints(explicit)
        self.assertIn("bipartite=true", implicit)
        self.assertIn("num_edges<=12", implicit)

    def test_bipartite_implicit_no_duplicate(self):
        """bipartite=true already explicit should not be duplicated."""
        explicit = ["graph_type=bipartite", "num_nodes=10", "bipartite=true"]
        implicit = infer_implicit_constraints(explicit)
        self.assertNotIn("bipartite=true", implicit)

    def test_validate_bipartite_inferred_constraints(self):
        """Inferred bipartite constraints should be satisfied by a valid bipartite graph."""
        G = nx.complete_bipartite_graph(3, 4)
        explicit = ["graph_type=bipartite", "num_nodes=7", "bipartite_sides=3,4"]
        implicit = infer_implicit_constraints(explicit)
        results = check_all_constraints(G, implicit)
        for constraint, satisfied in results.items():
            self.assertTrue(
                satisfied,
                f"Complete bipartite graph failed implicit constraint: {constraint}",
            )


if __name__ == "__main__":
    unittest.main()
