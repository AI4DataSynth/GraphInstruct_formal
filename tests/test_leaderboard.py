"""Tests for sub-leaderboard generation."""

import json
import os
import tempfile
import unittest

from graphinstruct.analysis.leaderboard import (
    LeaderboardGenerator,
    _assign_ranks,
    _derive_pareto_bonus,
    _mean,
)


def _make_instruction(
    inst_id: str,
    level: int,
    dim_scores: dict,
    d1_vr: float = 0.8,
    d4_is: float = 0.7,
    d5_tpv: float = 500.0,
) -> dict:
    """Create a mock per-instruction evaluation result."""
    return {
        "instruction_id": inst_id,
        "level": level,
        "d1_valid_rate": d1_vr,
        "d4_instruction_score": d4_is,
        "d5_tokens_per_valid": d5_tpv,
        "dimension_scores": dim_scores,
        "level_score": sum(dim_scores.values()) / max(len(dim_scores), 1),
    }


def _make_model_result(
    per_instruction: list[dict],
    total_score: float = 0.7,
    final_score: float = 0.8,
    per_level_scores: dict | None = None,
) -> dict:
    """Create a mock BenchmarkResult dict."""
    if per_level_scores is None:
        # Auto-aggregate from per_instruction
        level_groups: dict[int, list[float]] = {}
        for item in per_instruction:
            lvl = item["level"]
            level_groups.setdefault(lvl, []).append(item["level_score"])
        per_level_scores = {
            str(lvl): sum(scores) / len(scores) for lvl, scores in level_groups.items()
        }
    return {
        "per_instruction": per_instruction,
        "per_level_scores": per_level_scores,
        "total_score": total_score,
        "final_score": final_score,
    }


# ---------------------------------------------------------------------------
# Mock datasets: 3 models, 2 levels (L0, L1)
# ---------------------------------------------------------------------------


def _build_mock_results() -> dict[str, dict]:
    """Build mock results for 3 models across L0 and L1."""
    model_a_insts = [
        _make_instruction(
            "L0-001",
            0,
            {"D1": 0.9, "D2": 0.0, "D3": 0.0, "D4": 0.8, "D5": 0.7},
            d5_tpv=300.0,
        ),
        _make_instruction(
            "L0-002",
            0,
            {"D1": 0.85, "D2": 0.0, "D3": 0.0, "D4": 0.9, "D5": 0.6},
            d5_tpv=350.0,
        ),
        _make_instruction(
            "L1-001",
            1,
            {"D1": 0.7, "D2": 0.0, "D3": 0.0, "D4": 0.6, "D5": 0.5},
            d5_tpv=600.0,
        ),
    ]
    model_b_insts = [
        _make_instruction(
            "L0-001",
            0,
            {"D1": 0.6, "D2": 0.0, "D3": 0.0, "D4": 0.7, "D5": 0.8},
            d5_tpv=200.0,
        ),
        _make_instruction(
            "L0-002",
            0,
            {"D1": 0.65, "D2": 0.0, "D3": 0.0, "D4": 0.6, "D5": 0.9},
            d5_tpv=180.0,
        ),
        _make_instruction(
            "L1-001",
            1,
            {"D1": 0.8, "D2": 0.0, "D3": 0.0, "D4": 0.85, "D5": 0.4},
            d5_tpv=800.0,
        ),
    ]
    model_c_insts = [
        _make_instruction(
            "L0-001",
            0,
            {"D1": 0.95, "D2": 0.0, "D3": 0.0, "D4": 0.95, "D5": 0.5},
            d5_tpv=500.0,
        ),
        _make_instruction(
            "L0-002",
            0,
            {"D1": 0.9, "D2": 0.0, "D3": 0.0, "D4": 0.85, "D5": 0.55},
            d5_tpv=450.0,
        ),
        _make_instruction(
            "L1-001",
            1,
            {"D1": 0.85, "D2": 0.0, "D3": 0.0, "D4": 0.9, "D5": 0.3},
            d5_tpv=1000.0,
        ),
    ]
    return {
        "ModelA": _make_model_result(
            model_a_insts, total_score=0.72, final_score=0.828
        ),
        "ModelB": _make_model_result(
            model_b_insts, total_score=0.65, final_score=0.715
        ),
        "ModelC": _make_model_result(
            model_c_insts, total_score=0.85, final_score=0.9775
        ),
    }


class TestHelpers(unittest.TestCase):
    """Test module-level helper functions."""

    def test_mean_empty(self):
        self.assertAlmostEqual(_mean([]), 0.0)

    def test_mean_single(self):
        self.assertAlmostEqual(_mean([0.5]), 0.5)

    def test_mean_values(self):
        self.assertAlmostEqual(_mean([0.2, 0.4, 0.6]), 0.4)

    def test_assign_ranks_basic(self):
        rows = [
            {"model": "A", "score": 0.5},
            {"model": "B", "score": 0.9},
            {"model": "C", "score": 0.7},
        ]
        ranked = _assign_ranks(rows, "score")
        self.assertEqual(ranked[0]["model"], "B")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["model"], "C")
        self.assertEqual(ranked[1]["rank"], 2)
        self.assertEqual(ranked[2]["model"], "A")
        self.assertEqual(ranked[2]["rank"], 3)

    def test_assign_ranks_ties(self):
        rows = [
            {"model": "A", "score": 0.8},
            {"model": "B", "score": 0.8},
            {"model": "C", "score": 0.5},
        ]
        ranked = _assign_ranks(rows, "score")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 1)  # tie
        self.assertEqual(ranked[2]["rank"], 3)  # skips 2

    def test_derive_pareto_bonus_on_frontier(self):
        # S_final = S_total * (1 + 0.15 * 1.0) => S_final/S_total = 1.15
        bonus = _derive_pareto_bonus(0.8, 0.92)
        self.assertAlmostEqual(bonus, 1.0)

    def test_derive_pareto_bonus_zero(self):
        # S_final == S_total => bonus = 0
        bonus = _derive_pareto_bonus(0.8, 0.8)
        self.assertAlmostEqual(bonus, 0.0)

    def test_derive_pareto_bonus_zero_total(self):
        bonus = _derive_pareto_bonus(0.0, 0.0)
        self.assertAlmostEqual(bonus, 0.0)


class TestMainLeaderboard(unittest.TestCase):
    """Test main leaderboard generation."""

    def setUp(self):
        self.gen = LeaderboardGenerator(_build_mock_results())

    def test_returns_all_models(self):
        board = self.gen.main_leaderboard()
        models = {row["model"] for row in board}
        self.assertEqual(models, {"ModelA", "ModelB", "ModelC"})

    def test_ranked_by_s_final_descending(self):
        board = self.gen.main_leaderboard()
        s_finals = [row["s_final"] for row in board]
        self.assertEqual(s_finals, sorted(s_finals, reverse=True))

    def test_rank_1_is_highest(self):
        board = self.gen.main_leaderboard()
        rank1 = [r for r in board if r["rank"] == 1]
        self.assertEqual(len(rank1), 1)
        self.assertEqual(rank1[0]["model"], "ModelC")  # highest s_final

    def test_has_required_keys(self):
        board = self.gen.main_leaderboard()
        required = {
            "model",
            "s_total",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "pareto_bonus",
            "s_final",
            "rank",
        }
        for row in board:
            self.assertEqual(set(row.keys()), required)

    def test_dimension_scores_are_floats(self):
        board = self.gen.main_leaderboard()
        for row in board:
            for dim in ("D1", "D2", "D3", "D4", "D5"):
                self.assertIsInstance(row[dim], float)


class TestLevelLeaderboards(unittest.TestCase):
    """Test per-level leaderboard generation."""

    def setUp(self):
        self.gen = LeaderboardGenerator(_build_mock_results())

    def test_returns_correct_levels(self):
        boards = self.gen.level_leaderboards()
        self.assertIn("L0", boards)
        self.assertIn("L1", boards)
        self.assertEqual(len(boards), 2)  # only L0 and L1 in mock data

    def test_each_level_has_all_models(self):
        boards = self.gen.level_leaderboards()
        for key, board in boards.items():
            models = {row["model"] for row in board}
            self.assertEqual(models, {"ModelA", "ModelB", "ModelC"})

    def test_level_ranking_is_correct(self):
        boards = self.gen.level_leaderboards()
        for key, board in boards.items():
            scores = [row["level_score"] for row in board]
            self.assertEqual(
                scores,
                sorted(scores, reverse=True),
                f"{key} not sorted by level_score descending",
            )

    def test_has_required_keys(self):
        boards = self.gen.level_leaderboards()
        required = {"model", "level_score", "D1", "D2", "D3", "D4", "D5", "rank"}
        for board in boards.values():
            for row in board:
                self.assertEqual(set(row.keys()), required)


class TestDimensionLeaderboards(unittest.TestCase):
    """Test per-dimension leaderboard generation."""

    def setUp(self):
        self.gen = LeaderboardGenerator(_build_mock_results())

    def test_returns_all_dimensions(self):
        boards = self.gen.dimension_leaderboards()
        self.assertEqual(set(boards.keys()), {"D1", "D2", "D3", "D4", "D5"})

    def test_each_dimension_has_all_models(self):
        boards = self.gen.dimension_leaderboards()
        for dim, board in boards.items():
            models = {row["model"] for row in board}
            self.assertEqual(models, {"ModelA", "ModelB", "ModelC"})

    def test_has_per_level_breakdown(self):
        boards = self.gen.dimension_leaderboards()
        for dim, board in boards.items():
            for row in board:
                for lvl in range(6):
                    self.assertIn(
                        f"L{lvl}", row, f"Missing L{lvl} in {dim} leaderboard"
                    )

    def test_has_overall_and_rank(self):
        boards = self.gen.dimension_leaderboards()
        for dim, board in boards.items():
            for row in board:
                self.assertIn("overall", row)
                self.assertIn("rank", row)

    def test_d1_ranking(self):
        """ModelC has the highest D1 scores overall, should be rank 1."""
        boards = self.gen.dimension_leaderboards()
        d1_board = boards["D1"]
        rank1 = [r for r in d1_board if r["rank"] == 1]
        self.assertEqual(len(rank1), 1)
        self.assertEqual(rank1[0]["model"], "ModelC")

    def test_d2_all_zero(self):
        """D2 should be all zeros for L0/L1 data."""
        boards = self.gen.dimension_leaderboards()
        d2_board = boards["D2"]
        for row in d2_board:
            self.assertAlmostEqual(row["overall"], 0.0)


class TestEfficiencyLeaderboard(unittest.TestCase):
    """Test efficiency leaderboard generation."""

    def setUp(self):
        self.gen = LeaderboardGenerator(_build_mock_results())

    def test_returns_all_models(self):
        board = self.gen.efficiency_leaderboard()
        models = {row["model"] for row in board}
        self.assertEqual(models, {"ModelA", "ModelB", "ModelC"})

    def test_has_required_keys(self):
        required = {
            "model",
            "quality_per_ktoken",
            "pareto_score",
            "cost_at_q08",
            "quality_at_budget_10k",
            "frontier_distance",
            "pareto_bonus",
            "rank",
        }
        board = self.gen.efficiency_leaderboard()
        for row in board:
            self.assertEqual(set(row.keys()), required)

    def test_default_ranked_by_pareto_score(self):
        """Default ranking_mode='log' ranks by pareto_score."""
        board = self.gen.efficiency_leaderboard()
        scores = [row["pareto_score"] for row in board]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_linear_ranked_by_quality_per_ktoken(self):
        """ranking_mode='linear' ranks by quality_per_ktoken."""
        board = self.gen.efficiency_leaderboard(ranking_mode="linear")
        scores = [row["quality_per_ktoken"] for row in board]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_invalid_ranking_mode(self):
        with self.assertRaises(ValueError):
            self.gen.efficiency_leaderboard(ranking_mode="invalid")

    def test_pareto_bonus_range(self):
        board = self.gen.efficiency_leaderboard()
        for row in board:
            self.assertGreaterEqual(row["pareto_bonus"], 0.0)
            self.assertLessEqual(row["pareto_bonus"], 1.0)

    def test_pareto_score_positive(self):
        board = self.gen.efficiency_leaderboard()
        for row in board:
            self.assertGreaterEqual(row["pareto_score"], 0.0)

    def test_frontier_distance_non_negative(self):
        board = self.gen.efficiency_leaderboard()
        for row in board:
            self.assertGreaterEqual(row["frontier_distance"], 0.0)


class TestMarkdownFormat(unittest.TestCase):
    """Test Markdown table formatting."""

    def test_basic_table(self):
        rows = [
            {"model": "A", "score": 0.9, "rank": 1},
            {"model": "B", "score": 0.7, "rank": 2},
        ]
        md = LeaderboardGenerator.to_markdown(rows, title="Test Board")
        self.assertIn("## Test Board", md)
        self.assertIn("| model | score | rank |", md)
        self.assertIn("| --- | --- | --- |", md)
        self.assertIn("| A | 0.9000 | 1 |", md)

    def test_empty_rows(self):
        md = LeaderboardGenerator.to_markdown([], title="Empty")
        self.assertIn("(No data)", md)

    def test_none_values(self):
        rows = [{"model": "A", "value": None, "rank": 1}]
        md = LeaderboardGenerator.to_markdown(rows)
        self.assertIn("N/A", md)


class TestExport(unittest.TestCase):
    """Test export_all writes files correctly."""

    def setUp(self):
        self.gen = LeaderboardGenerator(_build_mock_results())
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_creates_files(self):
        written = self.gen.export_all(self.tmpdir)
        # At minimum: main + 2 levels + 5 dimensions + efficiency = 9 boards
        # Each board produces .json and .md = 18 files minimum
        self.assertGreaterEqual(len(written), 18)

    def test_export_json_valid(self):
        self.gen.export_all(self.tmpdir)
        json_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".json")]
        self.assertGreater(len(json_files), 0)
        for jf in json_files:
            path = os.path.join(self.tmpdir, jf)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)

    def test_export_md_valid(self):
        self.gen.export_all(self.tmpdir)
        md_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".md")]
        self.assertGreater(len(md_files), 0)
        for mf in md_files:
            path = os.path.join(self.tmpdir, mf)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("|", content)  # Has table

    def test_main_leaderboard_file(self):
        self.gen.export_all(self.tmpdir)
        path = os.path.join(self.tmpdir, "main_leaderboard.json")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # All 3 models present
        models = {row["model"] for row in data}
        self.assertEqual(models, {"ModelA", "ModelB", "ModelC"})


class TestSingleModel(unittest.TestCase):
    """Test leaderboard with only one model."""

    def setUp(self):
        insts = [
            _make_instruction(
                "L0-001", 0, {"D1": 0.9, "D2": 0.0, "D3": 0.0, "D4": 0.8, "D5": 0.7}
            ),
        ]
        self.gen = LeaderboardGenerator(
            {
                "OnlyModel": _make_model_result(
                    insts, total_score=0.5, final_score=0.575
                ),
            }
        )

    def test_main_leaderboard_single(self):
        board = self.gen.main_leaderboard()
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0]["rank"], 1)

    def test_efficiency_leaderboard_single(self):
        board = self.gen.efficiency_leaderboard()
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0]["rank"], 1)
        self.assertEqual(board[0]["pareto_bonus"], 1.0)

    def test_level_leaderboard_single(self):
        boards = self.gen.level_leaderboards()
        self.assertIn("L0", boards)
        self.assertEqual(len(boards["L0"]), 1)


class TestEmptyResults(unittest.TestCase):
    """Test edge case: empty results."""

    def test_empty_generator(self):
        gen = LeaderboardGenerator({})
        self.assertEqual(gen.main_leaderboard(), [])
        self.assertEqual(gen.level_leaderboards(), {})
        self.assertEqual(gen.efficiency_leaderboard(), [])

    def test_dimension_leaderboards_empty(self):
        gen = LeaderboardGenerator({})
        boards = gen.dimension_leaderboards()
        for dim in ("D1", "D2", "D3", "D4", "D5"):
            self.assertEqual(boards[dim], [])


if __name__ == "__main__":
    unittest.main()
