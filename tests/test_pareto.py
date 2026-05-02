"""Tests for graphinstruct.analysis.pareto (Pareto frontier analysis)."""

import unittest

from graphinstruct.analysis.pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    cost_at_quality,
    frontier_distance,
    is_on_frontier,
    pareto_bonus,
    quality_at_budget,
    quality_per_ktoken,
)


class TestParetoPoint(unittest.TestCase):
    def test_creation(self):
        pt = ParetoPoint(label="model_A", quality=0.8, cost=1000)
        self.assertEqual(pt.label, "model_A")
        self.assertEqual(pt.quality, 0.8)
        self.assertEqual(pt.cost, 1000)
        self.assertEqual(pt.valid_rate, 1.0)

    def test_frozen(self):
        pt = ParetoPoint(label="A", quality=0.5, cost=500)
        with self.assertRaises(AttributeError):
            pt.quality = 0.9


class TestComputeParetoFrontier(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(compute_pareto_frontier([]), [])

    def test_single_point(self):
        pts = [ParetoPoint("A", 0.8, 1000)]
        frontier = compute_pareto_frontier(pts)
        self.assertEqual(len(frontier), 1)

    def test_domination(self):
        pts = [
            ParetoPoint("A", 0.8, 1000),  # dominates B
            ParetoPoint("B", 0.6, 1500),  # dominated
            ParetoPoint("C", 0.9, 2000),  # on frontier
        ]
        frontier = compute_pareto_frontier(pts)
        labels = {p.label for p in frontier}
        self.assertIn("A", labels)
        self.assertIn("C", labels)
        self.assertNotIn("B", labels)

    def test_no_domination(self):
        pts = [
            ParetoPoint("A", 0.6, 500),
            ParetoPoint("B", 0.8, 1000),
            ParetoPoint("C", 0.95, 2000),
        ]
        frontier = compute_pareto_frontier(pts)
        self.assertEqual(len(frontier), 3)

    def test_sorted_by_cost(self):
        pts = [
            ParetoPoint("C", 0.95, 2000),
            ParetoPoint("A", 0.6, 500),
            ParetoPoint("B", 0.8, 1000),
        ]
        frontier = compute_pareto_frontier(pts)
        costs = [p.cost for p in frontier]
        self.assertEqual(costs, sorted(costs))

    def test_equal_quality_lower_cost_dominates(self):
        pts = [
            ParetoPoint("A", 0.8, 1000),
            ParetoPoint("B", 0.8, 1500),
        ]
        frontier = compute_pareto_frontier(pts)
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].label, "A")


class TestIsOnFrontier(unittest.TestCase):
    def test_on_frontier(self):
        pt = ParetoPoint("A", 0.8, 1000)
        frontier = [pt]
        self.assertTrue(is_on_frontier(pt, frontier))

    def test_not_on_frontier(self):
        pt = ParetoPoint("B", 0.5, 2000)
        frontier = [ParetoPoint("A", 0.8, 1000)]
        self.assertFalse(is_on_frontier(pt, frontier))


class TestParetoBonus(unittest.TestCase):
    def test_on_frontier_bonus(self):
        pt = ParetoPoint("A", 0.8, 1000)
        self.assertAlmostEqual(pareto_bonus(pt, [pt]), 1.0)

    def test_empty_frontier(self):
        pt = ParetoPoint("A", 0.8, 1000)
        self.assertAlmostEqual(pareto_bonus(pt, []), 1.0)

    def test_off_frontier(self):
        pt = ParetoPoint("B", 0.5, 2000)
        frontier = [ParetoPoint("A", 0.9, 500)]
        bonus = pareto_bonus(pt, frontier)
        self.assertGreaterEqual(bonus, 0.0)
        self.assertLess(bonus, 1.0)

    def test_range(self):
        frontier = [
            ParetoPoint("A", 0.6, 500),
            ParetoPoint("C", 0.95, 2000),
        ]
        pt = ParetoPoint("B", 0.7, 1200)
        bonus = pareto_bonus(pt, frontier)
        self.assertGreaterEqual(bonus, 0.0)
        self.assertLessEqual(bonus, 1.0)


class TestQualityPerKtoken(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(quality_per_ktoken(0.8, 2000), 0.4)

    def test_zero_tokens(self):
        self.assertAlmostEqual(quality_per_ktoken(0.8, 0), 0.0)


class TestCostAtQuality(unittest.TestCase):
    def test_achievable(self):
        pts = [
            ParetoPoint("A", 0.6, 500),
            ParetoPoint("B", 0.85, 1000),
            ParetoPoint("C", 0.9, 2000),
        ]
        cost = cost_at_quality(pts, threshold=0.8)
        self.assertEqual(cost, 1000)

    def test_not_achievable(self):
        pts = [ParetoPoint("A", 0.5, 500)]
        self.assertIsNone(cost_at_quality(pts, threshold=0.8))


class TestQualityAtBudget(unittest.TestCase):
    def test_within_budget(self):
        pts = [
            ParetoPoint("A", 0.6, 500),
            ParetoPoint("B", 0.85, 1000),
            ParetoPoint("C", 0.9, 2000),
        ]
        q = quality_at_budget(pts, budget=1500)
        self.assertAlmostEqual(q, 0.85)

    def test_no_budget(self):
        pts = [ParetoPoint("A", 0.6, 500)]
        self.assertAlmostEqual(quality_at_budget(pts, budget=100), 0.0)


class TestFrontierDistance(unittest.TestCase):
    def test_on_frontier(self):
        frontier = [ParetoPoint("A", 0.8, 1000)]
        pt = ParetoPoint("A", 0.8, 1000)
        self.assertAlmostEqual(frontier_distance(pt, frontier), 0.0)

    def test_below_frontier(self):
        frontier = [ParetoPoint("A", 0.9, 1000)]
        pt = ParetoPoint("B", 0.5, 1000)
        dist = frontier_distance(pt, frontier)
        self.assertAlmostEqual(dist, 0.4)

    def test_interpolated(self):
        frontier = [
            ParetoPoint("A", 0.6, 500),
            ParetoPoint("B", 0.9, 1500),
        ]
        pt = ParetoPoint("C", 0.5, 1000)
        dist = frontier_distance(pt, frontier)
        self.assertGreater(dist, 0.0)

    def test_empty_frontier(self):
        pt = ParetoPoint("A", 0.8, 1000)
        self.assertAlmostEqual(frontier_distance(pt, []), 0.0)


if __name__ == "__main__":
    unittest.main()
