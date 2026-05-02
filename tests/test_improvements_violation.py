"""API contract tests for graphinstruct.improvements.violation.

Day 1 skeleton: implementation tests are marked SKIP until Day 2.
Only the ``classify_constraint`` routing is exercised today since that
function is already implemented.
"""

import os
import unittest

# OpenMP duplicate lib workaround (see MEMORY.md)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import networkx as nx

from graphinstruct.improvements.violation import (
    ViolationDetail,
    classify_constraint,
    extract_violations,
)


class TestClassifyConstraint(unittest.TestCase):
    def test_degree_is_degree(self):
        self.assertEqual(classify_constraint("degree=3"), "degree")

    def test_min_degree_is_degree(self):
        self.assertEqual(classify_constraint("min_degree=2"), "degree")

    def test_max_degree_is_degree(self):
        self.assertEqual(classify_constraint("max_degree=5"), "degree")

    def test_graph_type_is_structural(self):
        self.assertEqual(classify_constraint("graph_type=tree"), "structural")

    def test_connected_is_structural(self):
        self.assertEqual(classify_constraint("connected=true"), "structural")

    def test_acyclic_is_structural(self):
        self.assertEqual(classify_constraint("acyclic=true"), "structural")

    def test_density_is_numerical(self):
        self.assertEqual(classify_constraint("density<=0.5"), "numerical")

    def test_clustering_is_numerical(self):
        self.assertEqual(
            classify_constraint("clustering_coefficient>=0.4"), "numerical"
        )

    def test_avg_path_length_is_numerical(self):
        self.assertEqual(classify_constraint("avg_path_length<=3.0"), "numerical")

    def test_num_nodes_is_count(self):
        self.assertEqual(classify_constraint("num_nodes=10"), "count")

    def test_num_edges_is_count(self):
        self.assertEqual(classify_constraint("num_edges=15"), "count")

    def test_directed_is_directed(self):
        self.assertEqual(classify_constraint("directed=true"), "directed")

    def test_unknown_key_is_unknown(self):
        self.assertEqual(classify_constraint("zzz_nonexistent=1"), "unknown")


class TestViolationDetailStructure(unittest.TestCase):
    def test_frozen_dataclass(self):
        v = ViolationDetail(
            constraint="degree=3",
            category="degree",
            satisfied=False,
            actual=2,
            expected=3,
        )
        with self.assertRaises(Exception):  # frozen -> FrozenInstanceError
            v.actual = 5  # type: ignore[misc]

    def test_locus_defaults_to_empty_dict(self):
        v = ViolationDetail(
            constraint="num_nodes=10",
            category="count",
            satisfied=False,
        )
        self.assertEqual(v.locus, {})


class TestExtractViolations(unittest.TestCase):
    def test_all_satisfied_returns_empty(self):
        # K3 (3-cycle) satisfies num_nodes=3 and degree=2
        G = nx.cycle_graph(3)
        vs = extract_violations(G, ("num_nodes=3", "degree=2"), ())
        self.assertTrue(all(v.satisfied for v in vs))

    def test_graph_none_returns_parse_sentinel(self):
        vs = extract_violations(None, ("num_nodes=3",), ())
        self.assertEqual(len(vs), 1)
        self.assertEqual(vs[0].category, "structural")
        self.assertEqual(vs[0].constraint, "parse")

    def test_k_regular_reports_offending_nodes(self):
        # 8-node graph where 2 nodes have degree=1 instead of 3
        G = nx.Graph()
        G.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 4),
                (2, 5),
                (3, 4),
                (3, 5),
                (4, 5),
                (6, 7),
            ]
        )
        vs = extract_violations(G, ("degree=3", "num_nodes=8"), ())
        degree_v = next(v for v in vs if v.category == "degree")
        self.assertFalse(degree_v.satisfied)
        self.assertIn("offending_nodes", degree_v.locus)
        self.assertIn("degree_table", degree_v.locus)
        self.assertIn(6, degree_v.locus["offending_nodes"])
        self.assertIn(7, degree_v.locus["offending_nodes"])

    def test_numerical_reports_direction_and_gap(self):
        # Sparse graph (5 nodes, 1 edge) has density 0.1; target 0.5 -> lower
        G = nx.Graph()
        G.add_nodes_from(range(5))
        G.add_edge(0, 1)
        vs = extract_violations(G, ("density=0.5",), ())
        num_v = next(v for v in vs if v.category == "numerical")
        self.assertFalse(num_v.satisfied)
        self.assertEqual(num_v.locus["direction"], "lower")
        self.assertIsNotNone(num_v.locus["gap"])
        self.assertGreater(num_v.locus["gap"], 0)

    def test_non_connected_reports_components(self):
        # Two disconnected triangles
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
        vs = extract_violations(G, ("connected=true",), ())
        conn_v = next(v for v in vs if v.category == "structural")
        self.assertFalse(conn_v.satisfied)
        self.assertIn("components", conn_v.locus)
        self.assertEqual(len(conn_v.locus["components"]), 2)


if __name__ == "__main__":
    unittest.main()
