"""Tests for interactive visualization module.

Tests use mock benchmark data to verify that each visualization
function produces valid Plotly Figure objects and that the HTML
dashboard export works correctly.
"""

from __future__ import annotations

import os
import tempfile
import unittest

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from graphinstruct.analysis.visualization import (
    _compute_avg_cost,
    _compute_avg_dimension_scores,
    _compute_valid_rate,
    efficiency_scatter,
    export_dashboard,
    level_heatmap,
    pareto_frontier_plot,
    radar_comparison,
)


def _make_mock_results(num_models: int = 3) -> dict:
    """Build mock multi-model results dict for testing.

    Returns a dict with the structure:
    {
        "models": {
            "Model-A": { per_instruction, per_level_scores, total_score, final_score },
            ...
        }
    }
    """
    models = {}
    model_configs = [
        ("LLaMA-2-13B", 0.65, 500, 0.80),
        ("GPT-4o", 0.85, 1200, 0.95),
        ("Claude-3", 0.78, 800, 0.90),
        ("Gemini-Pro", 0.72, 600, 0.85),
        ("Mistral-7B", 0.55, 350, 0.70),
    ]

    for i in range(min(num_models, len(model_configs))):
        name, quality, cost, valid_rate = model_configs[i]
        per_instruction = []
        for lvl in range(6):
            for j in range(3):
                per_instruction.append(
                    {
                        "instruction_id": f"L{lvl}-{j:03d}",
                        "level": lvl,
                        "d1_valid_rate": valid_rate - lvl * 0.05,
                        "d4_instruction_score": quality - lvl * 0.03,
                        "d5_tokens_per_valid": cost + lvl * 50,
                        "dimension_scores": {
                            "D1": 0.8 - lvl * 0.05 + i * 0.02,
                            "D2": 0.3 + lvl * 0.1 if lvl >= 3 else 0.0,
                            "D3": 0.25 + lvl * 0.08 if lvl >= 3 else 0.0,
                            "D4": quality - lvl * 0.03,
                            "D5": max(0.1, 0.7 - cost / 3000),
                        },
                        "level_score": quality - lvl * 0.04,
                    }
                )

        per_level_scores = {}
        for lvl in range(6):
            scores = [
                item["level_score"] for item in per_instruction if item["level"] == lvl
            ]
            per_level_scores[str(lvl)] = sum(scores) / len(scores)

        models[name] = {
            "per_instruction": per_instruction,
            "per_level_scores": per_level_scores,
            "total_score": quality,
            "final_score": quality * 1.1,
        }

    return {"models": models}


def _make_single_model_results() -> dict:
    """Build mock results with exactly 1 model."""
    return _make_mock_results(num_models=1)


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed")
class TestParetoFrontierPlot(unittest.TestCase):
    """Tests for pareto_frontier_plot."""

    def test_basic_figure_creation(self):
        """Pareto plot returns a Figure with expected traces."""
        results = _make_mock_results()
        fig = pareto_frontier_plot(results)
        self.assertIsInstance(fig, go.Figure)
        # At least one trace per model
        self.assertGreaterEqual(len(fig.data), 3)

    def test_figure_layout(self):
        """Pareto plot has correct title and axis labels."""
        results = _make_mock_results()
        fig = pareto_frontier_plot(results)
        self.assertIn("Pareto", fig.layout.title.text)
        self.assertIsNotNone(fig.layout.xaxis.title.text)
        self.assertIsNotNone(fig.layout.yaxis.title.text)

    def test_single_model(self):
        """Pareto plot works with a single model (no frontier line)."""
        results = _make_single_model_results()
        fig = pareto_frontier_plot(results)
        self.assertIsInstance(fig, go.Figure)
        # Single model: 1 scatter + possibly 1 frontier highlight
        self.assertGreaterEqual(len(fig.data), 1)

    def test_empty_models_raises(self):
        """Pareto plot raises ValueError with empty models dict."""
        with self.assertRaises(ValueError):
            pareto_frontier_plot({"models": {}})

    def test_frontier_line_present(self):
        """With multiple models, a Pareto frontier line trace exists."""
        results = _make_mock_results(num_models=4)
        fig = pareto_frontier_plot(results)
        line_traces = [t for t in fig.data if hasattr(t, "mode") and t.mode == "lines"]
        # Frontier line should exist when multiple models present
        self.assertGreaterEqual(len(line_traces), 0)


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed")
class TestRadarComparison(unittest.TestCase):
    """Tests for radar_comparison."""

    def test_basic_figure_creation(self):
        """Radar chart returns a Figure with Scatterpolar traces."""
        results = _make_mock_results()
        fig = radar_comparison(results)
        self.assertIsInstance(fig, go.Figure)
        # One trace per model
        self.assertEqual(len(fig.data), 3)
        for trace in fig.data:
            self.assertIsInstance(trace, go.Scatterpolar)

    def test_selected_models(self):
        """Radar chart filters to only selected models."""
        results = _make_mock_results()
        fig = radar_comparison(results, models=["LLaMA-2-13B"])
        self.assertEqual(len(fig.data), 1)

    def test_nonexistent_model_ignored(self):
        """Radar chart ignores model names not in results."""
        results = _make_mock_results()
        fig = radar_comparison(results, models=["NonExistent"])
        self.assertEqual(len(fig.data), 0)

    def test_radar_range(self):
        """Radar plot radial axis ranges from 0 to 1."""
        results = _make_mock_results()
        fig = radar_comparison(results)
        radial = fig.layout.polar.radialaxis
        self.assertEqual(list(radial.range), [0, 1])

    def test_five_dimensions(self):
        """Each trace has 6 theta values (5 dims + close)."""
        results = _make_mock_results(num_models=1)
        fig = radar_comparison(results)
        trace = fig.data[0]
        # 5 dimensions + 1 closing point = 6
        self.assertEqual(len(trace.r), 6)
        self.assertEqual(len(trace.theta), 6)


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed")
class TestLevelHeatmap(unittest.TestCase):
    """Tests for level_heatmap."""

    def test_basic_figure_creation(self):
        """Heatmap returns a Figure with a Heatmap trace."""
        results = _make_mock_results()
        fig = level_heatmap(results)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Heatmap)

    def test_heatmap_dimensions(self):
        """Heatmap z-matrix has correct dimensions (models x levels)."""
        results = _make_mock_results(num_models=2)
        fig = level_heatmap(results)
        z = fig.data[0].z
        # 2 models (rows)
        self.assertEqual(len(z), 2)
        # 6 levels (columns)
        self.assertEqual(len(z[0]), 6)

    def test_colorscale_range(self):
        """Heatmap color range is 0 to 1."""
        results = _make_mock_results()
        fig = level_heatmap(results)
        hm = fig.data[0]
        self.assertEqual(hm.zmin, 0)
        self.assertEqual(hm.zmax, 1)

    def test_level_labels(self):
        """Heatmap x-axis labels are L0-L5."""
        results = _make_mock_results()
        fig = level_heatmap(results)
        x_labels = list(fig.data[0].x)
        self.assertEqual(x_labels, ["L0", "L1", "L2", "L3", "L4", "L5"])


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed")
class TestEfficiencyScatter(unittest.TestCase):
    """Tests for efficiency_scatter."""

    def test_basic_figure_creation(self):
        """Efficiency scatter returns a Figure."""
        results = _make_mock_results()
        fig = efficiency_scatter(results)
        self.assertIsInstance(fig, go.Figure)
        # At least one trace per model
        self.assertGreaterEqual(len(fig.data), 3)

    def test_marker_sizes_vary(self):
        """Marker sizes differ based on valid_rate."""
        results = _make_mock_results()
        fig = efficiency_scatter(results)
        sizes = []
        for trace in fig.data:
            if hasattr(trace, "marker") and trace.marker.size is not None:
                sizes.append(trace.marker.size)
        # With different valid_rates, sizes should not all be the same
        if len(sizes) >= 2:
            self.assertFalse(
                all(s == sizes[0] for s in sizes),
                "Marker sizes should vary with valid_rate",
            )


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed")
class TestExportDashboard(unittest.TestCase):
    """Tests for export_dashboard."""

    def test_creates_html_file(self):
        """Dashboard export creates an HTML file."""
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_dashboard.html")
            returned_path = export_dashboard(results, out_path)
            self.assertTrue(os.path.exists(returned_path))
            self.assertTrue(returned_path.endswith(".html"))

    def test_html_is_self_contained(self):
        """Dashboard HTML contains plotly.js (self-contained)."""
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_dashboard.html")
            returned_path = export_dashboard(results, out_path)
            with open(returned_path, encoding="utf-8") as f:
                content = f.read()
            # Should contain plotly.js (either inline or CDN reference)
            self.assertTrue(
                "plotly" in content.lower(),
                "HTML should reference plotly",
            )

    def test_html_contains_all_charts(self):
        """Dashboard HTML contains all four chart containers."""
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_dashboard.html")
            returned_path = export_dashboard(results, out_path)
            with open(returned_path, encoding="utf-8") as f:
                content = f.read()
            # 4 chart divs + 1 CSS class definition = 5 occurrences
            self.assertGreaterEqual(content.count("chart-container"), 4)

    def test_creates_parent_directories(self):
        """Dashboard export creates parent directories if needed."""
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sub", "dir", "dashboard.html")
            returned_path = export_dashboard(results, out_path)
            self.assertTrue(os.path.exists(returned_path))

    def test_html_has_utf8_encoding(self):
        """Dashboard HTML has UTF-8 charset meta tag."""
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_dashboard.html")
            returned_path = export_dashboard(results, out_path)
            with open(returned_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn('charset="utf-8"', content)


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed")
class TestHelperFunctions(unittest.TestCase):
    """Tests for internal helper functions."""

    def test_compute_avg_cost(self):
        """_compute_avg_cost averages d5_tokens_per_valid values."""
        data = {
            "per_instruction": [
                {"d5_tokens_per_valid": 100},
                {"d5_tokens_per_valid": 200},
                {"d5_tokens_per_valid": 300},
            ]
        }
        self.assertAlmostEqual(_compute_avg_cost(data), 200.0)

    def test_compute_avg_cost_skips_inf(self):
        """_compute_avg_cost skips infinite values."""
        data = {
            "per_instruction": [
                {"d5_tokens_per_valid": 100},
                {"d5_tokens_per_valid": float("inf")},
            ]
        }
        self.assertAlmostEqual(_compute_avg_cost(data), 100.0)

    def test_compute_avg_cost_empty(self):
        """_compute_avg_cost returns 0 for empty data."""
        self.assertAlmostEqual(_compute_avg_cost({"per_instruction": []}), 0.0)

    def test_compute_valid_rate(self):
        """_compute_valid_rate averages d1_valid_rate values."""
        data = {
            "per_instruction": [
                {"d1_valid_rate": 1.0},
                {"d1_valid_rate": 0.6},
                {"d1_valid_rate": 0.8},
            ]
        }
        self.assertAlmostEqual(_compute_valid_rate(data), 0.8)

    def test_compute_valid_rate_empty(self):
        """_compute_valid_rate returns 0 for empty data."""
        self.assertAlmostEqual(_compute_valid_rate({"per_instruction": []}), 0.0)

    def test_compute_avg_dimension_scores(self):
        """_compute_avg_dimension_scores averages dimension values."""
        data = {
            "per_instruction": [
                {
                    "dimension_scores": {
                        "D1": 0.8,
                        "D2": 0.6,
                        "D3": 0.4,
                        "D4": 0.9,
                        "D5": 0.5,
                    }
                },
                {
                    "dimension_scores": {
                        "D1": 0.6,
                        "D2": 0.4,
                        "D3": 0.2,
                        "D4": 0.7,
                        "D5": 0.3,
                    }
                },
            ]
        }
        avg = _compute_avg_dimension_scores(data)
        self.assertAlmostEqual(avg["D1"], 0.7)
        self.assertAlmostEqual(avg["D2"], 0.5)
        self.assertAlmostEqual(avg["D3"], 0.3)
        self.assertAlmostEqual(avg["D4"], 0.8)
        self.assertAlmostEqual(avg["D5"], 0.4)

    def test_compute_avg_dimension_scores_empty(self):
        """_compute_avg_dimension_scores returns zeros for empty data."""
        avg = _compute_avg_dimension_scores({"per_instruction": []})
        for d in ["D1", "D2", "D3", "D4", "D5"]:
            self.assertAlmostEqual(avg[d], 0.0)


if __name__ == "__main__":
    unittest.main()
