"""Tests for D5 token efficiency metrics."""

import unittest

from graphinstruct.metrics.efficiency import (
    tokens_per_valid_graph,
    api_calls_per_graph,
    pareto_score,
    TokenRecord,
)


class TestTokenRecord(unittest.TestCase):
    """Test TokenRecord data structure."""

    def test_total_tokens(self):
        rec = TokenRecord(input_tokens=100, output_tokens=200)
        self.assertEqual(rec.total_tokens, 300)

    def test_defaults(self):
        rec = TokenRecord(input_tokens=50, output_tokens=50)
        self.assertEqual(rec.api_calls, 1)
        self.assertTrue(rec.valid)


class TestTokensPerValidGraph(unittest.TestCase):
    """Test tokens/valid graph computation."""

    def test_all_valid(self):
        records = [
            TokenRecord(input_tokens=100, output_tokens=200, valid=True),
            TokenRecord(input_tokens=150, output_tokens=250, valid=True),
        ]
        # total tokens = 300 + 400 = 700, valid count = 2
        self.assertAlmostEqual(tokens_per_valid_graph(records), 350.0)

    def test_mixed_valid(self):
        records = [
            TokenRecord(input_tokens=100, output_tokens=200, valid=True),
            TokenRecord(input_tokens=100, output_tokens=200, valid=False),
            TokenRecord(input_tokens=100, output_tokens=200, valid=True),
        ]
        # total tokens = 900, valid count = 2
        self.assertAlmostEqual(tokens_per_valid_graph(records), 450.0)

    def test_none_valid(self):
        records = [
            TokenRecord(input_tokens=100, output_tokens=200, valid=False),
        ]
        self.assertEqual(tokens_per_valid_graph(records), float("inf"))

    def test_empty(self):
        self.assertEqual(tokens_per_valid_graph([]), float("inf"))


class TestApiCallsPerGraph(unittest.TestCase):
    """Test API calls per valid graph."""

    def test_single_call_each(self):
        records = [
            TokenRecord(input_tokens=100, output_tokens=200, api_calls=1, valid=True),
            TokenRecord(input_tokens=100, output_tokens=200, api_calls=1, valid=True),
        ]
        self.assertAlmostEqual(api_calls_per_graph(records), 1.0)

    def test_with_retries(self):
        records = [
            TokenRecord(input_tokens=100, output_tokens=200, api_calls=3, valid=True),
            TokenRecord(input_tokens=100, output_tokens=200, api_calls=1, valid=False),
        ]
        # total calls = 4, valid = 1
        self.assertAlmostEqual(api_calls_per_graph(records), 4.0)

    def test_none_valid(self):
        records = [
            TokenRecord(input_tokens=100, output_tokens=200, api_calls=2, valid=False),
        ]
        self.assertEqual(api_calls_per_graph(records), float("inf"))


class TestParetoScore(unittest.TestCase):
    """Test Pareto score: Quality / log(1 + Total_Tokens)."""

    def test_basic(self):
        import math

        quality = 0.8
        total_tokens = 1000
        expected = 0.8 / math.log(1 + 1000)
        self.assertAlmostEqual(pareto_score(quality, total_tokens), expected)

    def test_zero_tokens(self):
        # log(1 + 0) = 0, should handle gracefully
        score = pareto_score(0.5, 0)
        self.assertEqual(score, float("inf"))

    def test_zero_quality(self):
        score = pareto_score(0.0, 1000)
        self.assertAlmostEqual(score, 0.0)

    def test_negative_tokens_raises(self):
        with self.assertRaises(ValueError):
            pareto_score(0.5, -1)


if __name__ == "__main__":
    unittest.main()
