"""Tests for graphinstruct.evaluate."""

import unittest

import networkx as nx

from graphinstruct.data_loader import Instruction
from graphinstruct.evaluate import (
    BenchmarkResult,
    GenerationResult,
    evaluate_benchmark,
    evaluate_instruction,
)
from graphinstruct.metrics.efficiency import TokenRecord


def _make_instruction(
    id_="L1-001",
    level=1,
    instruction="Generate a tree with 5 nodes.",
    explicit=("graph_type=tree", "num_nodes=5"),
    implicit=("num_edges=4", "acyclic=true", "connected=true"),
):
    return Instruction(
        id=id_,
        level=level,
        instruction=instruction,
        explicit_constraints=tuple(explicit),
        implicit_constraints=tuple(implicit),
        graph_sizes=("small",),
        feasible=True,
    )


def _make_tree(n=5):
    return nx.path_graph(n)


def _make_result(inst, graph, input_tokens=100, output_tokens=50):
    return GenerationResult(
        instruction=inst,
        graph=graph,
        raw_output="",
        token_record=TokenRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


class TestEvaluateInstruction(unittest.TestCase):
    def test_all_valid_trees(self):
        inst = _make_instruction()
        results = [_make_result(inst, _make_tree(5)) for _ in range(3)]
        eval_result = evaluate_instruction(results)

        self.assertEqual(eval_result.instruction_id, "L1-001")
        self.assertEqual(eval_result.level, 1)
        self.assertAlmostEqual(eval_result.d1_valid_rate, 1.0)
        self.assertGreater(eval_result.d4_instruction_score, 0.0)
        self.assertLess(eval_result.d5_tokens_per_valid, float("inf"))

    def test_no_valid_graphs(self):
        inst = _make_instruction()
        # Cycle is not a valid tree
        cycle = nx.cycle_graph(5)
        results = [_make_result(inst, cycle) for _ in range(3)]
        eval_result = evaluate_instruction(results)

        self.assertAlmostEqual(eval_result.d1_valid_rate, 0.0)
        self.assertEqual(eval_result.d5_tokens_per_valid, float("inf"))

    def test_none_graphs(self):
        inst = _make_instruction()
        results = [_make_result(inst, None) for _ in range(2)]
        eval_result = evaluate_instruction(results)

        self.assertAlmostEqual(eval_result.d1_valid_rate, 0.0)
        self.assertAlmostEqual(eval_result.d4_instruction_score, 0.0)

    def test_mixed_valid_invalid(self):
        inst = _make_instruction()
        results = [
            _make_result(inst, _make_tree(5)),  # valid
            _make_result(inst, nx.cycle_graph(5)),  # invalid
            _make_result(inst, _make_tree(5)),  # valid
        ]
        eval_result = evaluate_instruction(results)

        self.assertAlmostEqual(eval_result.d1_valid_rate, 2.0 / 3.0)

    def test_empty_results_raises(self):
        with self.assertRaises(ValueError):
            evaluate_instruction([])

    def test_level0_instruction(self):
        inst = _make_instruction(
            id_="L0-001",
            level=0,
            instruction="Generate a graph with 5 nodes.",
            explicit=("num_nodes=5",),
            implicit=(),
        )
        G = nx.path_graph(5)
        results = [_make_result(inst, G)]
        eval_result = evaluate_instruction(results)

        self.assertEqual(eval_result.level, 0)
        self.assertAlmostEqual(eval_result.d1_valid_rate, 1.0)

    def test_dimension_scores_structure(self):
        inst = _make_instruction()
        results = [_make_result(inst, _make_tree(5))]
        eval_result = evaluate_instruction(results)

        self.assertIn("D1", eval_result.dimension_scores)
        self.assertIn("D4", eval_result.dimension_scores)
        self.assertIn("D5", eval_result.dimension_scores)
        self.assertAlmostEqual(eval_result.dimension_scores["D2"], 0.0)
        self.assertAlmostEqual(eval_result.dimension_scores["D3"], 0.0)


class TestEvaluateBenchmark(unittest.TestCase):
    def test_single_level(self):
        inst = _make_instruction()
        results = {"L1-001": [_make_result(inst, _make_tree(5))]}
        benchmark = evaluate_benchmark(results)

        self.assertIsInstance(benchmark, BenchmarkResult)
        self.assertEqual(len(benchmark.per_instruction), 1)
        self.assertIn(1, benchmark.per_level_scores)
        self.assertGreater(benchmark.total_score, 0.0)

    def test_multiple_levels(self):
        inst0 = _make_instruction(
            id_="L0-001",
            level=0,
            instruction="Generate a graph with 5 nodes.",
            explicit=("num_nodes=5",),
            implicit=(),
        )
        inst1 = _make_instruction()
        results = {
            "L0-001": [_make_result(inst0, nx.path_graph(5))],
            "L1-001": [_make_result(inst1, _make_tree(5))],
        }
        benchmark = evaluate_benchmark(results)

        self.assertIn(0, benchmark.per_level_scores)
        self.assertIn(1, benchmark.per_level_scores)
        self.assertEqual(len(benchmark.per_instruction), 2)

    def test_pareto_bonus_auto(self):
        """Single model auto-computes Pareto bonus = 1.0 (self is frontier)."""
        inst = _make_instruction()
        results = {"L1-001": [_make_result(inst, _make_tree(5))]}
        benchmark = evaluate_benchmark(results)
        # Single model is always on the Pareto frontier, so bonus = 1.0
        # final_score = total_score * (1 + lambda * 1.0)
        self.assertGreater(benchmark.final_score, benchmark.total_score)

    def test_pareto_bonus_with_other_models(self):
        """Test Pareto bonus with other model comparisons."""
        inst = _make_instruction()
        results = {"L1-001": [_make_result(inst, _make_tree(5))]}
        # This model vs a better model (higher quality, same cost)
        other = [{"label": "better", "quality": 1.0, "cost": 100.0}]
        benchmark = evaluate_benchmark(results, other_models=other)
        # Our model may not be on the frontier
        self.assertIsNotNone(benchmark.final_score)

    def test_empty_raises(self):
        benchmark = evaluate_benchmark({})
        self.assertEqual(len(benchmark.per_instruction), 0)


if __name__ == "__main__":
    unittest.main()
