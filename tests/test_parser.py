"""Tests for the code-style graph format parser."""

import unittest

import networkx as nx

from graphinstruct.parser import ParseResult, parse, serialize


class TestParseBasicGraph(unittest.TestCase):
    """Test parsing basic Graph blocks."""

    def test_simple_tree(self):
        text = """
        Graph[name='task-001', type='tree', nodes=10] {
            node_list = ['n0', 'n1', 'n2', 'n3', 'n4',
                         'n5', 'n6', 'n7', 'n8', 'n9'];
            edge_list = [('n0','n1'), ('n0','n2'), ('n1','n3'), ('n1','n4'),
                         ('n2','n5'), ('n2','n6'), ('n5','n7'), ('n6','n8'), ('n6','n9')];
        }
        """
        result = parse(text)
        self.assertEqual(result.parse_type, "Graph")
        self.assertEqual(result.metadata["name"], "task-001")
        self.assertEqual(result.metadata["type"], "tree")
        self.assertEqual(result.metadata["nodes"], 10)
        self.assertEqual(result.graph.number_of_nodes(), 10)
        self.assertEqual(result.graph.number_of_edges(), 9)
        self.assertIn("n0", result.graph.nodes())
        self.assertIn(("n0", "n1"), result.graph.edges())

    def test_empty_graph(self):
        text = """
        Graph[name='empty', nodes=3] {
            node_list = ['a', 'b', 'c'];
            edge_list = [];
        }
        """
        result = parse(text)
        self.assertEqual(result.graph.number_of_nodes(), 3)
        self.assertEqual(result.graph.number_of_edges(), 0)

    def test_single_node(self):
        text = """
        Graph[name='single', nodes=1] {
            node_list = ['n0'];
            edge_list = [];
        }
        """
        result = parse(text)
        self.assertEqual(result.graph.number_of_nodes(), 1)

    def test_complete_graph_k4(self):
        text = """
        Graph[name='k4', type='complete', nodes=4] {
            node_list = ['n0', 'n1', 'n2', 'n3'];
            edge_list = [('n0','n1'), ('n0','n2'), ('n0','n3'),
                         ('n1','n2'), ('n1','n3'), ('n2','n3')];
        }
        """
        result = parse(text)
        self.assertEqual(result.graph.number_of_nodes(), 4)
        self.assertEqual(result.graph.number_of_edges(), 6)


class TestParseWithAttributes(unittest.TestCase):
    """Test parsing graphs with node attributes and graph properties."""

    def test_node_attributes(self):
        text = """
        Graph[name='task-042', type='community', nodes=4] {
            node_list = ['n0', 'n1', 'n2', 'n3'];
            edge_list = [('n0','n1'), ('n2','n3')];
            # Node attributes
            n0.community = 0; n0.label = "researcher";
            n1.community = 0; n1.label = "student";
            n2.community = 1; n2.label = "engineer";
            n3.community = 1; n3.label = "manager";
        }
        """
        result = parse(text)
        self.assertEqual(result.graph.nodes["n0"]["community"], 0)
        self.assertEqual(result.graph.nodes["n0"]["label"], "researcher")
        self.assertEqual(result.graph.nodes["n3"]["community"], 1)

    def test_graph_properties(self):
        text = """
        Graph[name='task-043', type='community', nodes=4] {
            node_list = ['n0', 'n1', 'n2', 'n3'];
            edge_list = [('n0','n1'), ('n1','n2'), ('n2','n3')];
            clustering_coefficient = 0.42;
            modularity = 0.51;
        }
        """
        result = parse(text)
        self.assertAlmostEqual(result.graph.graph["clustering_coefficient"], 0.42)
        self.assertAlmostEqual(result.graph.graph["modularity"], 0.51)

    def test_comments_ignored(self):
        text = """
        Graph[name='test'] {
            # This is a comment
            node_list = ['a', 'b'];
            # Another comment
            edge_list = [('a','b')];
        }
        """
        result = parse(text)
        self.assertEqual(result.graph.number_of_nodes(), 2)
        self.assertEqual(result.graph.number_of_edges(), 1)


class TestParseGraphEdit(unittest.TestCase):
    """Test parsing GraphEdit blocks."""

    def test_basic_graph_edit(self):
        text = """
        GraphEdit[name='task-088', base='given-graph', operation='add_edges'] {
            add_edges = [('n3','n7'), ('n5','n12')];
            remove_edges = [];
            reasoning = "Adding edges to connect components.";
        }
        """
        result = parse(text)
        self.assertEqual(result.parse_type, "GraphEdit")
        self.assertEqual(result.metadata["name"], "task-088")
        self.assertEqual(result.metadata["base"], "given-graph")
        self.assertEqual(len(result.add_edges), 2)
        self.assertEqual(result.add_edges[0], ("n3", "n7"))
        self.assertEqual(len(result.remove_edges), 0)
        self.assertIn("Adding edges", result.reasoning)

    def test_graph_edit_with_removes(self):
        text = """
        GraphEdit[name='task-089', base='graph-1', operation='modify'] {
            add_edges = [('a','b')];
            remove_edges = [('c','d'), ('e','f')];
            reasoning = "Restructuring the graph.";
        }
        """
        result = parse(text)
        self.assertEqual(len(result.add_edges), 1)
        self.assertEqual(len(result.remove_edges), 2)


class TestParseEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_invalid_no_header(self):
        with self.assertRaises(ValueError):
            parse("not a graph")

    def test_invalid_no_body(self):
        with self.assertRaises(ValueError):
            parse("Graph[name='test']")

    def test_directed_graph(self):
        text = """
        Graph[name='directed', directed=true, nodes=3] {
            node_list = ['a', 'b', 'c'];
            edge_list = [('a','b'), ('b','c')];
        }
        """
        result = parse(text)
        self.assertIsInstance(result.graph, nx.DiGraph)
        self.assertTrue(result.graph.has_edge("a", "b"))

    def test_double_quoted_strings(self):
        text = """
        Graph[name="test-dq", type="tree"] {
            node_list = ['n0', 'n1'];
            edge_list = [('n0','n1')];
        }
        """
        result = parse(text)
        self.assertEqual(result.metadata["name"], "test-dq")


class TestSerialize(unittest.TestCase):
    """Test serialization of NetworkX graphs to code-style format."""

    def test_roundtrip_simple(self):
        """Parse -> serialize -> parse should preserve graph structure."""
        original = """
        Graph[name='roundtrip', type='tree', nodes=5] {
            node_list = ['n0', 'n1', 'n2', 'n3', 'n4'];
            edge_list = [('n0','n1'), ('n0','n2'), ('n1','n3'), ('n1','n4')];
        }
        """
        result1 = parse(original)
        serialized = serialize(result1.graph, name="roundtrip", graph_type="tree")
        result2 = parse(serialized)

        self.assertEqual(
            result1.graph.number_of_nodes(), result2.graph.number_of_nodes()
        )
        self.assertEqual(
            result1.graph.number_of_edges(), result2.graph.number_of_edges()
        )
        self.assertEqual(set(result1.graph.nodes()), set(result2.graph.nodes()))
        self.assertEqual(set(result1.graph.edges()), set(result2.graph.edges()))

    def test_serialize_with_attributes(self):
        G = nx.Graph()
        G.add_nodes_from(["n0", "n1", "n2"])
        G.add_edge("n0", "n1")
        G.add_edge("n1", "n2")
        G.nodes["n0"]["label"] = "first"
        G.nodes["n1"]["label"] = "second"
        G.graph["modularity"] = 0.5

        text = serialize(G, name="attr-test", graph_type="community")
        result = parse(text)

        self.assertEqual(result.graph.nodes["n0"]["label"], "first")
        self.assertAlmostEqual(result.graph.graph["modularity"], 0.5)

    def test_serialize_empty_edges(self):
        G = nx.Graph()
        G.add_nodes_from(["a", "b"])
        text = serialize(G, name="no-edges")
        result = parse(text)
        self.assertEqual(result.graph.number_of_edges(), 0)
        self.assertEqual(result.graph.number_of_nodes(), 2)

    def test_serialize_from_networkx_generator(self):
        """Serialize a graph generated by NetworkX."""
        G = nx.complete_graph(5)
        # Rename nodes to string format
        mapping = {i: f"n{i}" for i in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        text = serialize(G, name="k5", graph_type="complete")
        result = parse(text)
        self.assertEqual(result.graph.number_of_nodes(), 5)
        self.assertEqual(result.graph.number_of_edges(), 10)


if __name__ == "__main__":
    unittest.main()
