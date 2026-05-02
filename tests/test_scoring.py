"""Tests for hierarchical scoring: S_level -> S_total -> S_final."""

import unittest

from graphinstruct.scoring import (
    LEVEL_WEIGHTS,
    DIMENSION_WEIGHTS,
    level_score,
    total_score,
    final_score,
)


class TestConstants(unittest.TestCase):
    """Test weight constants match the design doc."""

    def test_level_weights_sum(self):
        self.assertAlmostEqual(sum(LEVEL_WEIGHTS), 1.0)

    def test_level_weights_count(self):
        self.assertEqual(len(LEVEL_WEIGHTS), 6)

    def test_level_weights_values(self):
        expected = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
        for i, (a, b) in enumerate(zip(LEVEL_WEIGHTS, expected)):
            self.assertAlmostEqual(a, b, msg=f"Level {i} weight mismatch")

    def test_dimension_weights_per_level(self):
        self.assertEqual(len(DIMENSION_WEIGHTS), 6)
        for level, weights in enumerate(DIMENSION_WEIGHTS):
            self.assertAlmostEqual(
                sum(weights.values()),
                1.0,
                msg=f"Level {level} dimension weights don't sum to 1.0",
            )

    def test_d2_d3_zero_for_low_levels(self):
        """D2 and D3 should be 0 weight for levels 0-2."""
        for level in range(3):
            self.assertAlmostEqual(DIMENSION_WEIGHTS[level].get("D2", 0), 0.0)
            self.assertAlmostEqual(DIMENSION_WEIGHTS[level].get("D3", 0), 0.0)

    def test_d2_zero_for_l3(self):
        """L3 D2 should be 0.00 (synthetic graphs lack semantic text)."""
        self.assertAlmostEqual(DIMENSION_WEIGHTS[3]["D2"], 0.0)

    def test_d3_nonzero_for_l3_plus(self):
        """D3 should have positive weight for levels 3-5."""
        for level in range(3, 6):
            self.assertGreater(DIMENSION_WEIGHTS[level]["D3"], 0)

    def test_d2_nonzero_for_l4(self):
        """D2 should have positive weight for L4 only (L5 has no text attributes)."""
        self.assertGreater(DIMENSION_WEIGHTS[4]["D2"], 0)
        self.assertEqual(DIMENSION_WEIGHTS[5]["D2"], 0.0)

    def test_scheme_b_weights(self):
        """Verify all Scheme B dimension weights match design spec."""
        expected = [
            {"D1": 0.10, "D2": 0.00, "D3": 0.00, "D4": 0.60, "D5": 0.30},
            {"D1": 0.15, "D2": 0.00, "D3": 0.00, "D4": 0.70, "D5": 0.15},
            {"D1": 0.15, "D2": 0.00, "D3": 0.00, "D4": 0.70, "D5": 0.15},
            {"D1": 0.15, "D2": 0.00, "D3": 0.15, "D4": 0.50, "D5": 0.20},
            {"D1": 0.10, "D2": 0.15, "D3": 0.05, "D4": 0.55, "D5": 0.15},
            {"D1": 0.15, "D2": 0.00, "D3": 0.15, "D4": 0.50, "D5": 0.20},
        ]
        for level, (actual, exp) in enumerate(zip(DIMENSION_WEIGHTS, expected)):
            for dim in ("D1", "D2", "D3", "D4", "D5"):
                self.assertAlmostEqual(
                    actual[dim],
                    exp[dim],
                    msg=f"L{level} {dim}: got {actual[dim]}, expected {exp[dim]}",
                )


class TestLevelScore(unittest.TestCase):
    """Test S_level computation."""

    def test_perfect_all_dimensions(self):
        scores = {"D1": 1.0, "D2": 1.0, "D3": 1.0, "D4": 1.0, "D5": 1.0}
        for level in range(6):
            self.assertAlmostEqual(level_score(level, scores), 1.0)

    def test_zero_all_dimensions(self):
        scores = {"D1": 0.0, "D2": 0.0, "D3": 0.0, "D4": 0.0, "D5": 0.0}
        for level in range(6):
            self.assertAlmostEqual(level_score(level, scores), 0.0)

    def test_level0_ignores_d2_d3(self):
        """Level 0 should not use D2/D3 even if provided."""
        scores_with = {"D1": 0.5, "D2": 1.0, "D3": 1.0, "D4": 0.8, "D5": 0.6}
        scores_without = {"D1": 0.5, "D2": 0.0, "D3": 0.0, "D4": 0.8, "D5": 0.6}
        self.assertAlmostEqual(
            level_score(0, scores_with), level_score(0, scores_without)
        )

    def test_level0_specific(self):
        """Level 0 weights: D1=0.10, D4=0.60, D5=0.30 (design doc §5.1)."""
        scores = {"D1": 1.0, "D4": 0.5, "D5": 0.0}
        # 0.10*1.0 + 0.60*0.5 + 0.30*0.0 = 0.40
        self.assertAlmostEqual(level_score(0, scores), 0.40)

    def test_missing_dimensions_treated_as_zero(self):
        scores = {"D1": 1.0}
        result = level_score(0, scores)
        # D1=0.10*1.0 + rest=0
        self.assertAlmostEqual(result, 0.10)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            level_score(6, {"D1": 1.0})
        with self.assertRaises(ValueError):
            level_score(-1, {"D1": 1.0})


class TestTotalScore(unittest.TestCase):
    """Test S_total computation."""

    def test_perfect_all_levels(self):
        level_scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertAlmostEqual(total_score(level_scores), 1.0)

    def test_zero_all_levels(self):
        level_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(total_score(level_scores), 0.0)

    def test_weighted_sum(self):
        # Only level 0 scores 1.0, rest 0
        level_scores = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(total_score(level_scores), 0.05)

    def test_partial_levels(self):
        """Support fewer than 6 levels (Phase 1 only has L0-L3)."""
        level_scores = [0.8, 0.7, 0.6, 0.5]
        # 0.05*0.8 + 0.10*0.7 + 0.15*0.6 + 0.20*0.5 = 0.04+0.07+0.09+0.10 = 0.30
        self.assertAlmostEqual(total_score(level_scores), 0.30)


class TestFinalScore(unittest.TestCase):
    """Test S_final with Pareto bonus."""

    def test_no_bonus(self):
        self.assertAlmostEqual(final_score(0.8, pareto_bonus=0.0), 0.8)

    def test_full_bonus(self):
        # S_final = 0.8 * (1 + 0.15 * 1.0) = 0.8 * 1.15 = 0.92
        self.assertAlmostEqual(final_score(0.8, pareto_bonus=1.0), 0.92)

    def test_custom_lambda(self):
        # S_final = 0.8 * (1 + 0.30 * 0.5) = 0.8 * 1.15 = 0.92
        self.assertAlmostEqual(
            final_score(0.8, pareto_bonus=0.5, pareto_lambda=0.30), 0.92
        )

    def test_default_lambda(self):
        """Default lambda should be 0.15."""
        s = final_score(1.0, pareto_bonus=1.0)
        self.assertAlmostEqual(s, 1.15)


if __name__ == "__main__":
    unittest.main()
