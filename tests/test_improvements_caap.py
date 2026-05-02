"""Tests for graphinstruct.improvements.caap decision matrix.

Covers the 24-cell grid (6 levels x 4 tiers) plus the hard prohibition rules
derived from v4 analysis:
    L2 x T3  -> never few-shot   (avg -0.034; GPT-4o-mini crashes to 0.424)
    L3 x T3  -> never few-cot    (avg -0.048 poison)
    L5 x T3  -> never few-cot    (-0.041)
"""

import os
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from graphinstruct.data_loader import Instruction
from graphinstruct.improvements.caap import CAAPDecision, select_strategy
from graphinstruct.improvements.registry import model_tier


def _mk_inst(
    level,
    explicit=(),
    implicit=(),
    inst_id=None,
    domain=None,
) -> Instruction:
    return Instruction(
        id=inst_id or f"L{level}-test",
        level=level,
        instruction="test instruction",
        explicit_constraints=tuple(explicit),
        implicit_constraints=tuple(implicit),
        graph_sizes=("small",),
        feasible=True,
        domain=domain,
    )


# Representative model per tier for cross-tier tests.
_TIER_REP_MODEL = {
    "T1": "claude-sonnet-4-6",
    "T2-GPT": "gpt-4o",
    "T2-open": "qwen3.5-35b-a3b",
    "T3": "gpt-4o-mini",
}


class TestModelTier(unittest.TestCase):
    def test_known_model_returns_tier(self):
        self.assertEqual(model_tier("gpt-4o-mini"), "T3")
        self.assertEqual(model_tier("claude-sonnet-4-6"), "T1")
        self.assertEqual(model_tier("qwen3.5-35b-a3b"), "T2-open")
        self.assertEqual(model_tier("gpt-4o"), "T2-GPT")

    def test_case_insensitive(self):
        self.assertEqual(model_tier("GPT-4o-Mini"), "T3")
        self.assertEqual(model_tier("Claude-Sonnet-4-6"), "T1")

    def test_unknown_defaults_to_t2_gpt(self):
        self.assertEqual(model_tier("nonexistent-model"), "T2-GPT")

    def test_filename_aliases(self):
        self.assertEqual(model_tier("gpt4omini"), "T3")
        self.assertEqual(model_tier("gpt35turbo"), "T3")


class TestL0AllTiers(unittest.TestCase):
    def test_zero_shot_everywhere(self):
        inst = _mk_inst(0)
        for tier_name, model in _TIER_REP_MODEL.items():
            with self.subTest(tier=tier_name):
                d = select_strategy(inst, model)
                self.assertEqual(d.strategy, "zero-shot")


class TestL1Simple(unittest.TestCase):
    def test_tree_all_tiers_zero_shot(self):
        inst = _mk_inst(1, ("graph_type=tree", "num_nodes=10"))
        for tier_name, model in _TIER_REP_MODEL.items():
            with self.subTest(tier=tier_name):
                d = select_strategy(inst, model)
                self.assertEqual(d.strategy, "zero-shot")


class TestL1Complex(unittest.TestCase):
    def test_bipartite_varies_by_tier(self):
        inst = _mk_inst(1, ("graph_type=bipartite", "num_nodes=10"))
        self.assertEqual(
            select_strategy(inst, _TIER_REP_MODEL["T3"]).strategy, "zero-cot"
        )
        self.assertEqual(
            select_strategy(inst, _TIER_REP_MODEL["T2-GPT"]).strategy, "few-shot"
        )
        self.assertEqual(
            select_strategy(inst, _TIER_REP_MODEL["T2-open"]).strategy, "few-cot"
        )
        self.assertEqual(
            select_strategy(inst, _TIER_REP_MODEL["T1"]).strategy, "few-cot"
        )


class TestL2ProhibitsFewShotForT3(unittest.TestCase):
    """L2 x T3 must never use few-shot (v4 data: avg -0.034, mini crashes)."""

    def test_t3_gets_zero_cot_with_checklist(self):
        inst = _mk_inst(2, ("degree=3", "num_nodes=8", "connected=true"))
        d = select_strategy(inst, _TIER_REP_MODEL["T3"])
        self.assertNotEqual(d.strategy, "few-shot")
        self.assertEqual(d.strategy, "zero-cot")
        self.assertIn("checklist", d.extras)
        self.assertTrue(d.extras["checklist"])

    def test_checklist_includes_constraints(self):
        inst = _mk_inst(2, ("degree=3", "num_nodes=8"))
        d = select_strategy(inst, "gpt-4o-mini")
        self.assertIn("degree=3", d.extras["checklist"])
        self.assertIn("num_nodes=8", d.extras["checklist"])

    def test_t1_gets_few_cot(self):
        inst = _mk_inst(2, ("degree=3", "num_nodes=8"))
        d = select_strategy(inst, _TIER_REP_MODEL["T1"])
        self.assertEqual(d.strategy, "few-cot")


class TestL3ProhibitsFewCotForT3(unittest.TestCase):
    """L3 x T3 must never use few-cot (v4 data: avg -0.048 largest negative)."""

    def test_t3_gets_zero_cot_with_formula(self):
        inst = _mk_inst(3, ("density=0.5", "num_nodes=10"))
        d = select_strategy(inst, _TIER_REP_MODEL["T3"])
        self.assertNotEqual(d.strategy, "few-cot")
        self.assertEqual(d.strategy, "zero-cot")
        self.assertIn("formula", d.extras)

    def test_t2_gpt_gets_zero_shot(self):
        inst = _mk_inst(3, ("density=0.5",))
        d = select_strategy(inst, _TIER_REP_MODEL["T2-GPT"])
        self.assertEqual(d.strategy, "zero-shot")

    def test_formula_matches_constraint(self):
        inst_density = _mk_inst(3, ("density=0.5",))
        d = select_strategy(inst_density, "gpt-4o-mini")
        self.assertIn("density", d.extras["formula"].lower())

        inst_cluster = _mk_inst(3, ("clustering_coefficient>=0.4",))
        d2 = select_strategy(inst_cluster, "gpt-4o-mini")
        self.assertIn("clustering", d2.extras["formula"].lower())


class TestL4PrefersFewShot(unittest.TestCase):
    """L4: FS is the only +0.069 strategy across tiers."""

    def test_t3_t2gpt_few_shot(self):
        inst = _mk_inst(4, ("num_nodes=10",), domain="social")
        for tier in ("T3", "T2-GPT"):
            with self.subTest(tier=tier):
                d = select_strategy(inst, _TIER_REP_MODEL[tier])
                self.assertEqual(d.strategy, "few-shot")

    def test_t2_open_t1_few_cot(self):
        inst = _mk_inst(4, ("num_nodes=10",), domain="social")
        for tier in ("T2-open", "T1"):
            with self.subTest(tier=tier):
                d = select_strategy(inst, _TIER_REP_MODEL[tier])
                self.assertEqual(d.strategy, "few-cot")

    def test_domain_extras_injected_when_known(self):
        inst = _mk_inst(4, ("num_nodes=10",), domain="social")
        d = select_strategy(inst, "gpt-4o-mini")
        self.assertIn("domain", d.extras)
        self.assertIn("social", d.extras["domain"].lower())

    def test_unknown_domain_no_extras(self):
        inst = _mk_inst(4, ("num_nodes=10",), domain="nonexistent")
        d = select_strategy(inst, "gpt-4o-mini")
        self.assertNotIn("domain", d.extras)

    def test_no_domain_no_extras(self):
        inst = _mk_inst(4, ("num_nodes=10",), domain=None)
        d = select_strategy(inst, "gpt-4o-mini")
        self.assertNotIn("domain", d.extras)


class TestL5ProhibitsFewCotForT3(unittest.TestCase):
    def test_t3_gets_zero_cot(self):
        inst = _mk_inst(5)
        d = select_strategy(inst, _TIER_REP_MODEL["T3"])
        self.assertNotEqual(d.strategy, "few-cot")
        self.assertEqual(d.strategy, "zero-cot")

    def test_t2_and_t1_get_few_cot(self):
        inst = _mk_inst(5)
        for tier in ("T2-GPT", "T2-open", "T1"):
            with self.subTest(tier=tier):
                d = select_strategy(inst, _TIER_REP_MODEL[tier])
                self.assertEqual(d.strategy, "few-cot")


class TestRationaleAlwaysSet(unittest.TestCase):
    def test_rationale_non_empty_on_all_grid_cells(self):
        """Every (level, tier) decision must carry a human-readable rationale."""
        for level in range(6):
            inst_kwargs: dict = {}
            if level == 1:
                inst_kwargs["explicit"] = ("graph_type=bipartite",)
            elif level == 4:
                inst_kwargs["domain"] = "social"
            elif level in (2, 3):
                inst_kwargs["explicit"] = ("num_nodes=10", "degree=2")
            inst = _mk_inst(level, **inst_kwargs)
            for tier_name, model in _TIER_REP_MODEL.items():
                with self.subTest(level=level, tier=tier_name):
                    d = select_strategy(inst, model)
                    self.assertTrue(
                        d.rationale, f"empty rationale L{level}/{tier_name}"
                    )


if __name__ == "__main__":
    unittest.main()
