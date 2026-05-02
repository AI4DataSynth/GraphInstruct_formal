"""API contract tests for graphinstruct.improvements.runners."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import networkx as nx

from graphinstruct.data_loader import Instruction
from graphinstruct.improvements.runners import (
    BaseRunner,
    BaselineRunner,
    CAAPRunner,
    CombinedRunner,
    RetryRunner,
    RunnerConfig,
    RunnerResult,
    SCRunner,
    VGIGRunner,
)


def _mk_inst(
    level=0, inst_id="L0-test", feasible=True, explicit=("num_nodes=3",)
) -> Instruction:
    return Instruction(
        id=inst_id,
        level=level,
        instruction="test",
        explicit_constraints=tuple(explicit),
        implicit_constraints=(),
        graph_sizes=("small",),
        feasible=feasible,
    )


def _mock_generate_fn(instruction, num_samples=1, **kwargs):
    """Return num_samples dummy objects with a raw_output attribute-like dict."""
    return [
        {"raw_output": "Graph[] { node_list=[0,1,2]; edge_list=[]; }"}
    ] * num_samples


def _make_programmed_fn(graphs):
    """Return a generate_fn that yields pre-programmed graphs per call.

    Each sample is a SimpleNamespace with .graph / .raw_output / .instruction
    attributes, matching the shape runners expect.
    """
    state = {"calls": 0, "kw_history": []}

    def _fn(instruction, *, num_samples=1, **kw):
        state["kw_history"].append(kw)
        results = []
        for _ in range(num_samples):
            idx = state["calls"]
            if idx < len(graphs):
                g = graphs[idx]
            else:
                g = graphs[-1] if graphs else None
            state["calls"] += 1
            results.append(
                SimpleNamespace(
                    graph=g,
                    raw_output="Graph[] { node_list=[0,1,2]; edge_list=[(0,1)]; }",
                    instruction=instruction,
                )
            )
        return results

    return _fn, state


class TestRunnerConfigDefaults(unittest.TestCase):
    def test_method_required(self):
        c = RunnerConfig(method="vgig")
        self.assertEqual(c.method, "vgig")

    def test_default_max_rounds(self):
        self.assertEqual(RunnerConfig(method="vgig").max_rounds, 3)

    def test_default_num_candidates(self):
        self.assertEqual(RunnerConfig(method="sc").num_candidates, 3)

    def test_default_feedback_level(self):
        self.assertEqual(RunnerConfig(method="vgig").feedback_level, "fine")

    def test_default_num_samples(self):
        self.assertEqual(RunnerConfig(method="baseline").num_samples, 3)

    def test_extras_default_empty_dict(self):
        self.assertEqual(RunnerConfig(method="vgig").extras, {})

    def test_config_is_mutable(self):
        """CLI layer patches num_samples after construction; dataclass must allow it."""
        c = RunnerConfig(method="vgig")
        c.num_samples = 5
        self.assertEqual(c.num_samples, 5)


class TestRunnerResultSchema(unittest.TestCase):
    def test_fields_present(self):
        r = RunnerResult(instruction_id="L0-001", level=0, samples=[])
        self.assertEqual(r.instruction_id, "L0-001")
        self.assertEqual(r.level, 0)
        self.assertEqual(r.samples, [])
        self.assertIsInstance(r.method_metadata, dict)

    def test_method_metadata_default_empty(self):
        r = RunnerResult(instruction_id="L0-001", level=0, samples=[])
        self.assertEqual(r.method_metadata, {})


class TestCombinedRunnerAutoEnablesCAAP(unittest.TestCase):
    """E4 = VGIG + CAAP; CombinedRunner must force use_caap=True."""

    def test_forces_use_caap_true(self):
        config = RunnerConfig(method="combined")
        runner = CombinedRunner(config, _mock_generate_fn, {})
        self.assertTrue(runner.config.extras.get("use_caap"))

    def test_preserves_other_extras(self):
        config = RunnerConfig(method="combined", extras={"strategy": "few-cot"})
        runner = CombinedRunner(config, _mock_generate_fn, {})
        self.assertEqual(runner.config.extras.get("strategy"), "few-cot")
        self.assertTrue(runner.config.extras.get("use_caap"))

    def test_preserves_max_rounds(self):
        config = RunnerConfig(method="combined", max_rounds=5)
        runner = CombinedRunner(config, _mock_generate_fn, {})
        self.assertEqual(runner.config.max_rounds, 5)


class TestRunnerInheritance(unittest.TestCase):
    def test_all_runners_extend_base(self):
        for cls in (
            BaselineRunner,
            RetryRunner,
            SCRunner,
            VGIGRunner,
            CAAPRunner,
            CombinedRunner,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, BaseRunner))

    def test_combined_extends_vgig(self):
        self.assertTrue(issubclass(CombinedRunner, VGIGRunner))


class TestRunnerExecution(unittest.TestCase):
    def test_baseline_runner_returns_k_samples(self):
        fn, state = _make_programmed_fn([nx.cycle_graph(3)] * 3)
        config = RunnerConfig(method="baseline", num_samples=3)
        runner = BaselineRunner(config, fn, {})
        result = runner.run_instruction(_mk_inst(), model_name="gpt-4o-mini")
        self.assertEqual(len(result.samples), 3)
        self.assertEqual(state["calls"], 3)

    def test_retry_runner_generates_t_rounds(self):
        # All failing (empty graph -> fails num_nodes=3)
        fn, state = _make_programmed_fn([nx.Graph()] * 5)
        config = RunnerConfig(method="retry", max_rounds=3, num_samples=3)
        runner = RetryRunner(config, fn, {})
        result = runner.run_instruction(_mk_inst(), model_name="gpt-4o-mini")
        # max(max_rounds, num_samples) = 3 calls (all fail, no early break)
        self.assertEqual(state["calls"], 3)
        self.assertEqual(len(result.samples), 3)
        self.assertIn("scores", result.method_metadata)

    def test_sc_runner_ranks_by_satisfaction_score(self):
        bad = nx.Graph()
        bad.add_nodes_from([0, 1])  # 2 nodes, fails num_nodes=3
        good = nx.cycle_graph(3)  # satisfies num_nodes=3
        # Order: [bad, good, bad]; top-1 should be good
        fn, state = _make_programmed_fn([bad, good, bad])
        config = RunnerConfig(method="sc", num_candidates=3, num_samples=1)
        runner = SCRunner(config, fn, {})
        result = runner.run_instruction(_mk_inst(), model_name="gpt-4o-mini")
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(result.samples[0].graph.number_of_nodes(), 3)

    def test_vgig_early_stops_on_satisfied(self):
        fn, state = _make_programmed_fn([nx.cycle_graph(3)])  # satisfies round 0
        config = RunnerConfig(method="vgig", max_rounds=3, num_samples=1)
        runner = VGIGRunner(config, fn, {})
        result = runner.run_instruction(_mk_inst(), model_name="gpt-4o-mini")
        self.assertEqual(state["calls"], 1)
        self.assertEqual(result.method_metadata["chains"][0]["rounds"], 1)

    def test_vgig_infeasible_short_circuit(self):
        """Infeasible instructions must not iterate (single round only)."""
        fn, state = _make_programmed_fn([nx.Graph()])  # parse-failure-like
        config = RunnerConfig(method="vgig", max_rounds=3, num_samples=1)
        runner = VGIGRunner(config, fn, {})
        result = runner.run_instruction(
            _mk_inst(feasible=False), model_name="gpt-4o-mini"
        )
        self.assertEqual(state["calls"], 1)
        self.assertTrue(result.method_metadata["chains"][0]["infeasible"])

    def test_vgig_produces_per_chain_metadata(self):
        fn, state = _make_programmed_fn([nx.cycle_graph(3)] * 6)
        config = RunnerConfig(method="vgig", max_rounds=1, num_samples=3)
        runner = VGIGRunner(config, fn, {})
        result = runner.run_instruction(_mk_inst(), model_name="gpt-4o-mini")
        chains = result.method_metadata["chains"]
        self.assertEqual(len(chains), 3)
        for c in chains:
            self.assertIn("rounds", c)
            self.assertIn("best_score", c)

    def test_caap_runner_selects_strategy(self):
        fn, state = _make_programmed_fn([nx.cycle_graph(3)] * 3)
        config = RunnerConfig(method="caap", num_samples=3)
        runner = CAAPRunner(config, fn, {})
        # L0 -> zero-shot per CAAP matrix
        result = runner.run_instruction(_mk_inst(level=0), model_name="gpt-4o-mini")
        self.assertEqual(result.method_metadata["caap_strategy"], "zero-shot")

    def test_combined_runner_uses_caap_extras(self):
        fn, state = _make_programmed_fn([nx.cycle_graph(3)])
        config = RunnerConfig(method="combined", num_samples=1)
        runner = CombinedRunner(config, fn, {})
        result = runner.run_instruction(_mk_inst(), model_name="gpt-4o-mini")
        self.assertTrue(result.method_metadata["use_caap"])


if __name__ == "__main__":
    unittest.main()
