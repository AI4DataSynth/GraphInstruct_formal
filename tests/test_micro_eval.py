"""Tests for GRAPHIA micro evaluation (L5 multi-step reasoning)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Optional

import networkx as nx

from graphinstruct.metrics.micro_eval import (
    MicroEvalResult,
    ReasoningStep,
    StepResult,
    build_reasoning_steps_json,
    decompose_instruction,
    evaluate_reasoning_chain,
    evaluate_step,
    _normalize_edge,
    _edge_set,
    _node_set,
)


class TestHelpers(unittest.TestCase):
    """Test low-level helper functions."""

    def test_normalize_edge_ordered(self):
        self.assertEqual(_normalize_edge("a", "b"), ("a", "b"))

    def test_normalize_edge_reversed(self):
        self.assertEqual(_normalize_edge("b", "a"), ("a", "b"))

    def test_normalize_edge_same(self):
        self.assertEqual(_normalize_edge("x", "x"), ("x", "x"))

    def test_edge_set_simple(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        result = _edge_set(G)
        self.assertEqual(result, frozenset({("0", "1"), ("1", "2")}))

    def test_edge_set_normalization(self):
        """Edges should be normalized regardless of insertion order."""
        G = nx.Graph()
        G.add_edge(2, 1)
        result = _edge_set(G)
        self.assertEqual(result, frozenset({("1", "2")}))

    def test_edge_set_empty(self):
        G = nx.Graph()
        G.add_node(0)
        self.assertEqual(_edge_set(G), frozenset())

    def test_node_set_simple(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        result = _node_set(G)
        self.assertEqual(result, frozenset({"0", "1", "2"}))

    def test_node_set_string_nodes(self):
        G = nx.Graph()
        G.add_nodes_from(["A", "B"])
        result = _node_set(G)
        self.assertEqual(result, frozenset({"A", "B"}))


class TestDecomposeInstruction(unittest.TestCase):
    """Test step decomposition from base/ref graph diffs."""

    def test_no_graphs(self):
        """Without graphs, returns construct + verify."""
        steps = decompose_instruction({"id": "test"}, None, None)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].step_type, "construct")
        self.assertEqual(steps[1].step_type, "verify")

    def test_identical_graphs(self):
        """Base == Ref: only construct + verify (no operations needed)."""
        G = nx.path_graph(3)
        steps = decompose_instruction({"id": "test"}, G, G)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].step_type, "construct")
        self.assertEqual(steps[1].step_type, "verify")

    def test_add_edge(self):
        """Ref has one extra edge -> add_edge step."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (0, 2)])

        steps = decompose_instruction({"id": "test"}, base, ref)
        types = [s.step_type for s in steps]
        self.assertIn("construct", types)
        self.assertIn("add_edge", types)
        self.assertIn("verify", types)

        add_step = [s for s in steps if s.step_type == "add_edge"][0]
        self.assertEqual(len(add_step.target_edges), 1)
        self.assertIn(("0", "2"), add_step.target_edges)

    def test_remove_edge(self):
        """Ref has one less edge -> remove_edge step."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2), (0, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2)])

        steps = decompose_instruction({"id": "test"}, base, ref)
        types = [s.step_type for s in steps]
        self.assertIn("remove_edge", types)

        rm_step = [s for s in steps if s.step_type == "remove_edge"][0]
        self.assertEqual(len(rm_step.target_edges), 1)

    def test_add_node(self):
        """Ref has extra nodes -> add_node step."""
        base = nx.Graph()
        base.add_nodes_from([0, 1])
        ref = nx.Graph()
        ref.add_nodes_from([0, 1, 2])
        ref.add_edge(1, 2)

        steps = decompose_instruction({"id": "test"}, base, ref)
        types = [s.step_type for s in steps]
        self.assertIn("add_node", types)
        self.assertIn("add_edge", types)

    def test_remove_node(self):
        """Ref has fewer nodes -> remove_node step."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edge(0, 1)

        steps = decompose_instruction({"id": "test"}, base, ref)
        types = [s.step_type for s in steps]
        self.assertIn("remove_node", types)

    def test_complex_transformation(self):
        """Both add and remove edges -> multiple operation steps."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2), (2, 3)])
        ref = nx.Graph()
        ref.add_edges_from(
            [(0, 1), (0, 3), (1, 3)]
        )  # removed (1,2),(2,3); added (0,3),(1,3)

        steps = decompose_instruction({"id": "test"}, base, ref)
        types = [s.step_type for s in steps]

        self.assertIn("construct", types)
        self.assertIn("remove_node", types)  # node 2 removed
        self.assertIn("add_edge", types)
        self.assertIn("verify", types)

    def test_step_ordering(self):
        """Steps follow order: construct, remove_node, remove_edge, add_node, add_edge, verify."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2), (2, 3)])
        # Remove node 2, add node 4 with edge (3,4)
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (3, 4)])

        steps = decompose_instruction({"id": "test"}, base, ref)
        types = [s.step_type for s in steps]

        # Verify ordering: construct first, verify last
        self.assertEqual(types[0], "construct")
        self.assertEqual(types[-1], "verify")

        # remove_node should appear before add_node
        if "remove_node" in types and "add_node" in types:
            self.assertLess(types.index("remove_node"), types.index("add_node"))


class TestEvaluateStep(unittest.TestCase):
    """Test individual step evaluation."""

    def test_construct_none_graph(self):
        """None generated graph -> all zeros."""
        step = ReasoningStep(1, "construct", "Build base")
        result = evaluate_step(step, None, None, None)
        self.assertEqual(result.s_sel, 0.0)
        self.assertEqual(result.s_edge, 0.0)
        self.assertFalse(result.correct)

    def test_construct_with_base(self):
        """Generated preserves base nodes -> high s_sel."""
        base = nx.path_graph(4)
        gen = nx.path_graph(4)  # Perfect match

        step = ReasoningStep(1, "construct", "Build base")
        result = evaluate_step(step, gen, base, None)
        self.assertEqual(result.s_sel, 1.0)
        self.assertEqual(result.s_edge, 1.0)
        self.assertTrue(result.correct)

    def test_construct_missing_nodes(self):
        """Generated missing some base nodes -> partial s_sel."""
        base = nx.Graph()
        base.add_nodes_from([0, 1, 2, 3])
        gen = nx.Graph()
        gen.add_nodes_from([0, 1])  # Only 2 of 4 nodes

        step = ReasoningStep(1, "construct", "Build base")
        result = evaluate_step(step, gen, base, None)
        self.assertAlmostEqual(result.s_sel, 0.5)  # 2/4

    def test_verify_perfect(self):
        """Generated matches reference exactly."""
        ref = nx.path_graph(3)
        gen = nx.path_graph(3)

        step = ReasoningStep(3, "verify", "Check constraints")
        result = evaluate_step(step, gen, None, ref)
        self.assertEqual(result.s_sel, 1.0)
        self.assertEqual(result.s_edge, 1.0)
        self.assertTrue(result.correct)

    def test_verify_partial_match(self):
        """Generated partially matches reference."""
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (2, 3)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2)])  # Missing node 3 and edge (2,3)

        step = ReasoningStep(3, "verify", "Check constraints")
        result = evaluate_step(step, gen, None, ref)
        self.assertLess(result.s_sel, 1.0)
        self.assertLess(result.s_edge, 1.0)
        self.assertFalse(result.correct)

    def test_add_edge_perfect(self):
        """All expected edge additions done correctly."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (0, 2)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2), (0, 2)])

        step = ReasoningStep(2, "add_edge", "Add edge", target_edges=(("0", "2"),))
        result = evaluate_step(step, gen, base, ref)
        self.assertEqual(result.s_edge, 1.0)
        self.assertTrue(result.correct)

    def test_add_edge_missing(self):
        """Expected edge additions not done."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (0, 2)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2)])  # Didn't add (0,2)

        step = ReasoningStep(2, "add_edge", "Add edge", target_edges=(("0", "2"),))
        result = evaluate_step(step, gen, base, ref)
        self.assertEqual(result.s_edge, 0.0)
        self.assertFalse(result.correct)

    def test_remove_edge_perfect(self):
        """All expected edge removals done correctly."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2), (0, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2)])  # Correctly removed (0,2)

        step = ReasoningStep(
            2, "remove_edge", "Remove edge", target_edges=(("0", "2"),)
        )
        result = evaluate_step(step, gen, base, ref)
        self.assertEqual(result.s_edge, 1.0)
        self.assertTrue(result.correct)

    def test_remove_node_perfect(self):
        """All expected node removals done correctly."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edge(0, 1)
        gen = nx.Graph()
        gen.add_edge(0, 1)

        step = ReasoningStep(2, "remove_node", "Remove node 2", target_nodes=("2",))
        result = evaluate_step(step, gen, base, ref)
        self.assertEqual(result.s_sel, 1.0)
        self.assertTrue(result.correct)

    def test_add_node_perfect(self):
        """All expected node additions done correctly."""
        base = nx.Graph()
        base.add_edge(0, 1)
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2)])

        step = ReasoningStep(2, "add_node", "Add node 2", target_nodes=("2",))
        result = evaluate_step(step, gen, base, ref)
        self.assertEqual(result.s_sel, 1.0)
        self.assertTrue(result.correct)

    def test_verify_no_ref(self):
        """Verify step without reference -> fail."""
        gen = nx.path_graph(3)
        step = ReasoningStep(3, "verify", "Check")
        result = evaluate_step(step, gen, None, None)
        self.assertEqual(result.s_sel, 0.0)
        self.assertEqual(result.s_edge, 0.0)
        self.assertFalse(result.correct)


class TestEvaluateReasoningChain(unittest.TestCase):
    """Test full reasoning chain evaluation."""

    def test_no_refs(self):
        """No reference solutions -> zero scores."""
        gen = nx.path_graph(3)
        result = evaluate_reasoning_chain({"id": "test"}, gen, None, [])
        self.assertEqual(result.s_micro, 0.0)
        self.assertEqual(result.steps_total, 0)

    def test_perfect_match(self):
        """Generated exactly matches reference -> perfect scores."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (0, 2)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2), (0, 2)])

        result = evaluate_reasoning_chain({"id": "test"}, gen, base, [ref])
        self.assertEqual(result.s_sel, 1.0)
        self.assertEqual(result.s_edge, 1.0)
        self.assertEqual(result.s_micro, 1.0)
        self.assertGreater(result.steps_total, 0)

    def test_partial_match(self):
        """Generated partially matches -> partial scores."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2), (2, 3)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2), (2, 3)])  # Didn't add (0,3)

        result = evaluate_reasoning_chain({"id": "test"}, gen, base, [ref])
        self.assertLess(result.s_micro, 1.0)
        self.assertGreater(result.s_micro, 0.0)

    def test_best_match_selected(self):
        """With multiple refs, the best-matching one is used."""
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])

        # ref1: add (0,2) -- generated matches this
        ref1 = nx.Graph()
        ref1.add_edges_from([(0, 1), (1, 2), (0, 2)])
        # ref2: add (0,2) and (0,3) -- generated doesn't match this
        ref2 = nx.Graph()
        ref2.add_edges_from([(0, 1), (1, 2), (0, 2)])
        ref2.add_node(3)
        ref2.add_edge(0, 3)

        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2), (0, 2)])

        result = evaluate_reasoning_chain({"id": "test"}, gen, base, [ref1, ref2])
        # Should pick ref1 as best match (perfect)
        self.assertEqual(result.s_micro, 1.0)

    def test_none_generated(self):
        """None generated graph -> all zeros but steps still counted."""
        base = nx.path_graph(3)
        ref = nx.path_graph(3)
        ref.add_edge(0, 2)

        result = evaluate_reasoning_chain({"id": "test"}, None, base, [ref])
        self.assertEqual(result.s_sel, 0.0)
        self.assertEqual(result.s_edge, 0.0)
        self.assertEqual(result.s_micro, 0.0)

    def test_step_details_populated(self):
        """Step details should be populated with StepResult objects."""
        base = nx.path_graph(3)
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (0, 2)])
        gen = nx.Graph()
        gen.add_edges_from([(0, 1), (1, 2), (0, 2)])

        result = evaluate_reasoning_chain({"id": "test"}, gen, base, [ref])
        self.assertGreater(len(result.step_details), 0)
        for sd in result.step_details:
            self.assertIsInstance(sd, StepResult)


class TestBuildReasoningStepsJson(unittest.TestCase):
    """Test JSON step builder for L5 instructions."""

    def test_simple_add_edge(self):
        base = nx.Graph()
        base.add_edges_from([(0, 1), (1, 2)])
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (1, 2), (0, 2)])

        steps = build_reasoning_steps_json({"id": "test"}, base, ref)
        self.assertIsInstance(steps, list)
        self.assertGreaterEqual(len(steps), 2)

        # First step is always construct
        self.assertEqual(steps[0]["type"], "construct")
        # Last step is always verify
        self.assertEqual(steps[-1]["type"], "verify")

        # Should have an add_edge step
        types = [s["type"] for s in steps]
        self.assertIn("add_edge", types)

    def test_no_graphs(self):
        steps = build_reasoning_steps_json({"id": "test"}, None, None)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["type"], "construct")
        self.assertEqual(steps[1]["type"], "verify")

    def test_target_edges_format(self):
        """target_edges should be list of [u,v] lists."""
        base = nx.Graph()
        base.add_edge(0, 1)
        ref = nx.Graph()
        ref.add_edges_from([(0, 1), (0, 2)])

        steps = build_reasoning_steps_json({"id": "test"}, base, ref)
        add_steps = [s for s in steps if s["type"] == "add_edge"]
        self.assertEqual(len(add_steps), 1)
        # target_edges should be list of lists
        edges = add_steps[0]["target_edges"]
        self.assertIsInstance(edges, list)
        for e in edges:
            self.assertIsInstance(e, list)
            self.assertEqual(len(e), 2)


class TestMicroEvalDataclass(unittest.TestCase):
    """Test dataclass properties."""

    def test_frozen(self):
        r = MicroEvalResult(
            s_sel=0.5, s_edge=0.6, s_micro=0.55, steps_total=3, steps_correct=2
        )
        with self.assertRaises(AttributeError):
            r.s_sel = 0.9  # type: ignore

    def test_step_result_frozen(self):
        sr = StepResult(
            step_index=1, step_type="construct", s_sel=1.0, s_edge=1.0, correct=True
        )
        with self.assertRaises(AttributeError):
            sr.s_sel = 0.0  # type: ignore

    def test_reasoning_step_frozen(self):
        rs = ReasoningStep(step_index=1, step_type="construct", description="test")
        with self.assertRaises(AttributeError):
            rs.step_type = "verify"  # type: ignore


class TestL5InstructionIntegration(unittest.TestCase):
    """Integration tests with actual L5 instruction data."""

    @classmethod
    def setUpClass(cls):
        """Load L5 instructions."""
        path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "instructions"
            / "level_5.json"
        )
        with open(path, encoding="utf-8") as f:
            cls.instructions = json.load(f)

    def test_all_have_reasoning_steps(self):
        """All L5 instructions should have reasoning_steps."""
        for inst in self.instructions:
            self.assertIn(
                "reasoning_steps", inst, f'{inst["id"]} missing reasoning_steps'
            )
            self.assertIsInstance(inst["reasoning_steps"], list)
            self.assertGreaterEqual(
                len(inst["reasoning_steps"]), 2, f'{inst["id"]} has too few steps'
            )

    def test_first_step_is_construct(self):
        """First step of every instruction should be construct."""
        for inst in self.instructions:
            steps = inst["reasoning_steps"]
            self.assertEqual(
                steps[0]["type"],
                "construct",
                f'{inst["id"]} first step is not construct',
            )

    def test_last_step_is_verify(self):
        """Last step of every instruction should be verify."""
        for inst in self.instructions:
            steps = inst["reasoning_steps"]
            self.assertEqual(
                steps[-1]["type"], "verify", f'{inst["id"]} last step is not verify'
            )

    def test_step_indices_sequential(self):
        """Step indices should be sequential starting from 1."""
        for inst in self.instructions:
            steps = inst["reasoning_steps"]
            for i, step in enumerate(steps):
                self.assertEqual(
                    step["step"],
                    i + 1,
                    f'{inst["id"]} step index mismatch at position {i}',
                )

    def test_valid_step_types(self):
        """All step types should be recognized."""
        valid_types = {
            "construct",
            "add_edge",
            "remove_edge",
            "add_node",
            "remove_node",
            "verify",
            "reverse_edge",
            "compute",
        }
        for inst in self.instructions:
            for step in inst["reasoning_steps"]:
                self.assertIn(
                    step["type"],
                    valid_types,
                    f'{inst["id"]} has unknown step type: {step["type"]}',
                )

    def test_sample_l5_001_evaluation(self):
        """Full micro-eval on L5-001 with perfect generated graph."""
        from graphinstruct.parser.code_style import parse

        inst = self.instructions[0]  # L5-001
        self.assertEqual(inst["id"], "L5-001")

        # L5-001 has base_graph=None (random graph, not reconstructable)
        if inst.get("base_graph") is None:
            ref_graph = parse(inst["reference_solutions"][0]).graph
            # Without base graph, use ref as both base and gen for smoke test
            result = evaluate_reasoning_chain(inst, ref_graph, ref_graph, [ref_graph])
            self.assertEqual(result.s_micro, 1.0)
            return

        base_graph = parse(inst["base_graph"]).graph
        ref_graph = parse(inst["reference_solutions"][0]).graph

        # Simulate perfect generation: gen == ref
        result = evaluate_reasoning_chain(inst, ref_graph, base_graph, [ref_graph])
        self.assertEqual(result.s_micro, 1.0)
        self.assertEqual(result.s_sel, 1.0)
        self.assertEqual(result.s_edge, 1.0)

    def test_sample_l5_003_wrong_generation(self):
        """L5-003: generated doesn't match reference -> partial scores."""
        from graphinstruct.parser.code_style import parse

        inst = self.instructions[2]  # L5-003
        self.assertEqual(inst["id"], "L5-003")

        base_graph = parse(inst["base_graph"]).graph
        ref_graphs = []
        for rs in inst["reference_solutions"]:
            ref_graphs.append(parse(rs).graph)

        # Generate the base graph unchanged (no operations performed)
        result = evaluate_reasoning_chain(inst, base_graph, base_graph, ref_graphs)
        # Should not be perfect since base != ref
        self.assertLess(result.s_micro, 1.0)


class TestEvaluatePipelineIntegration(unittest.TestCase):
    """Test integration with evaluate.py pipeline."""

    def test_compute_micro_eval_import(self):
        """The _compute_micro_eval function should be importable."""
        from graphinstruct.evaluate import _compute_micro_eval

        self.assertTrue(callable(_compute_micro_eval))

    def test_instruction_eval_result_has_micro_eval(self):
        """InstructionEvalResult should have micro_eval field."""
        from graphinstruct.evaluate import InstructionEvalResult

        r = InstructionEvalResult(
            instruction_id="test",
            level=5,
            d1_valid_rate=1.0,
            d4_instruction_score=1.0,
            d5_tokens_per_valid=100.0,
            dimension_scores={"D1": 1.0, "D2": 0.0, "D3": 0.0, "D4": 1.0, "D5": 1.0},
            level_score=1.0,
            micro_eval=MicroEvalResult(
                s_sel=0.9,
                s_edge=0.8,
                s_micro=0.85,
                steps_total=3,
                steps_correct=2,
            ),
        )
        self.assertIsNotNone(r.micro_eval)
        self.assertAlmostEqual(r.micro_eval.s_micro, 0.85)

    def test_compute_micro_eval_with_l5_data(self):
        """Test _compute_micro_eval with real L5 instruction data."""
        from graphinstruct.data_loader import Instruction
        from graphinstruct.evaluate import _compute_micro_eval
        from graphinstruct.parser.code_style import parse

        path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "instructions"
            / "level_5.json"
        )
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        inst_data = raw[0]  # L5-001
        inst = Instruction.from_dict(inst_data)

        # Parse reference graphs
        ref_graphs = []
        for rs in inst.reference_solutions:
            parsed = parse(rs)
            if parsed.graph is not None:
                ref_graphs.append(parsed.graph)

        # Use ref as generated (perfect case)
        valid_graphs = [ref_graphs[0]]

        result = _compute_micro_eval(inst, valid_graphs, ref_graphs)
        self.assertIsNotNone(result)
        self.assertEqual(result.s_micro, 1.0)


class TestGraphEditEndToEnd(unittest.TestCase):
    """End-to-end test: GraphEdit output -> apply to base -> evaluate."""

    def setUp(self):
        path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "instructions"
            / "level_5.json"
        )
        with open(path, encoding="utf-8") as f:
            self.raw = json.load(f)

    def _apply_graph_edit(self, base, add_edges, remove_edges):
        """Apply GraphEdit operations to a base graph."""
        G = base.copy()
        for u, v in remove_edges or ():
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        for u, v in add_edges or ():
            G.add_edge(u, v)
        return G

    def test_graph_edit_roundtrip_l5_003(self):
        """Parse GraphEdit ref, apply to base, compare with Graph ref."""
        from graphinstruct.parser.code_style import parse

        inst = next(d for d in self.raw if d["id"] == "L5-003")

        base = parse(inst["base_graph"]).graph
        ref_graph = parse(inst["reference_solutions"][0]).graph
        edit_result = parse(inst["graph_edits"][0])

        self.assertEqual(edit_result.parse_type, "GraphEdit")

        # Apply edit to base
        reconstructed = self._apply_graph_edit(
            base, edit_result.add_edges, edit_result.remove_edges
        )

        # Reconstructed should match reference solution
        ref_edges = {(min(u, v), max(u, v)) for u, v in ref_graph.edges()}
        rec_edges = {
            (min(str(u), str(v)), max(str(u), str(v))) for u, v in reconstructed.edges()
        }
        self.assertEqual(ref_edges, rec_edges, "Reconstructed graph != reference")

    def test_graph_edit_micro_eval_perfect(self):
        """Apply correct GraphEdit to base -> micro eval should score 1.0."""
        from graphinstruct.parser.code_style import parse

        # Use L5-003: remove 1 edge from K5 to make it planar
        inst_data = next(d for d in self.raw if d["id"] == "L5-003")
        base = parse(inst_data["base_graph"]).graph
        edit_result = parse(inst_data["graph_edits"][0])

        generated = self._apply_graph_edit(
            base, edit_result.add_edges, edit_result.remove_edges
        )
        ref_graphs = [parse(rs).graph for rs in inst_data["reference_solutions"]]

        result = evaluate_reasoning_chain(inst_data, generated, base, ref_graphs)
        self.assertEqual(result.s_micro, 1.0)

    def test_graph_edit_micro_eval_wrong(self):
        """Apply no edits (return base as-is) -> micro eval should score < 1."""
        from graphinstruct.parser.code_style import parse

        inst_data = next(d for d in self.raw if d["id"] == "L5-003")
        base = parse(inst_data["base_graph"]).graph
        ref_graphs = [parse(rs).graph for rs in inst_data["reference_solutions"]]

        # Submit base graph unchanged — wrong answer
        result = evaluate_reasoning_chain(inst_data, base, base, ref_graphs)
        self.assertLess(result.s_micro, 1.0)

    def test_all_graph_edits_reconstruct_correctly(self):
        """For all L5 instructions, applying graph_edits to base should match ref."""
        from graphinstruct.parser.code_style import parse

        failures = []
        for inst in self.raw:
            if not inst.get("base_graph") or not inst.get("graph_edits"):
                continue
            base = parse(inst["base_graph"]).graph
            for j, edit_str in enumerate(inst["graph_edits"]):
                edit_r = parse(edit_str)
                reconstructed = self._apply_graph_edit(
                    base, edit_r.add_edges, edit_r.remove_edges
                )
                ref = parse(inst["reference_solutions"][j]).graph

                # Compare node sets (k-core removes nodes)
                ref_nodes = set(str(n) for n in ref.nodes())
                rec_nodes = set(str(n) for n in reconstructed.nodes())

                if ref_nodes != rec_nodes:
                    # Node-removal operation — check that ref is subgraph
                    if not ref_nodes.issubset(rec_nodes):
                        failures.append(
                            f'{inst["id"]} edit[{j}]: ref nodes not subset of reconstructed'
                        )
                    continue

                if base.is_directed():
                    ref_edges = {(str(u), str(v)) for u, v in ref.edges()}
                    rec_edges = {(str(u), str(v)) for u, v in reconstructed.edges()}
                else:
                    ref_edges = {
                        (min(str(u), str(v)), max(str(u), str(v)))
                        for u, v in ref.edges()
                    }
                    rec_edges = {
                        (min(str(u), str(v)), max(str(u), str(v)))
                        for u, v in reconstructed.edges()
                    }

                if ref_edges != rec_edges:
                    failures.append(
                        f'{inst["id"]} edit[{j}]: edges mismatch '
                        f"(missing={len(ref_edges - rec_edges)}, extra={len(rec_edges - ref_edges)})"
                    )

        self.assertEqual(
            failures, [], f"Reconstruction failures:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    unittest.main()
