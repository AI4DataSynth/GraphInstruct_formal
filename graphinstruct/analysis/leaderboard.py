"""Sub-leaderboard generation for GraphInstruct benchmark.

Generates 4 types of leaderboards from evaluation results:
1. Main leaderboard: Ranked by S_final
2. Level leaderboards: Per-level rankings (L0-L5)
3. Dimension leaderboards: Per-dimension rankings (D1-D5)
4. Efficiency leaderboard: Cost-quality trade-off rankings

Input format: Dict mapping model_name -> serialized BenchmarkResult
(same JSON structure as visualize_results.py expects).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from graphinstruct.analysis.pareto import (
    ParetoPoint,
    compute_pareto_frontier,
    cost_at_quality,
    frontier_distance,
    pareto_bonus,
    quality_at_budget,
    quality_per_ktoken,
)
from graphinstruct.metrics.efficiency import pareto_score as _pareto_score_log


def _mean(values: Sequence[float]) -> float:
    """Compute mean of a numeric sequence, returning 0.0 if empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _assign_ranks(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    """Assign dense ranks to rows based on score_key (descending).

    Returns a new list of dicts with a 'rank' field added.
    Ties receive the same rank; the next rank skips accordingly.
    """
    sorted_rows = sorted(rows, key=lambda r: r[score_key], reverse=True)
    ranked: list[dict[str, Any]] = []
    prev_score: Optional[float] = None
    prev_rank = 0
    for i, row in enumerate(sorted_rows):
        new_row = dict(row)
        if prev_score is None or abs(row[score_key] - prev_score) > 1e-9:
            prev_rank = i + 1
        new_row["rank"] = prev_rank
        prev_score = row[score_key]
        ranked.append(new_row)
    return ranked


class LeaderboardGenerator:
    """Generate 4 types of leaderboards from evaluation results.

    Expected input structure per model (matches evaluate.py JSON output):
    {
        "per_instruction": [
            {
                "instruction_id": "L1-042",
                "level": 1,
                "d1_valid_rate": 0.8,
                "d4_instruction_score": 0.7,
                "d5_tokens_per_valid": 500.0,
                "dimension_scores": {"D1": 0.8, "D2": 0.0, ...},
                "level_score": 0.75
            }, ...
        ],
        "per_level_scores": {"0": 0.9, "1": 0.85, ...},
        "total_score": 0.72,
        "final_score": 0.83
    }
    """

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        """Initialize with multi-model results.

        Args:
            results: Dict mapping model_name -> evaluation result dict.
        """
        self.results = results

    # ------------------------------------------------------------------
    # 1. Main Leaderboard
    # ------------------------------------------------------------------

    def main_leaderboard(self) -> list[dict[str, Any]]:
        """Generate main leaderboard sorted by S_final.

        Returns:
            List of dicts with keys:
            model, s_total, D1..D5, pareto_bonus, s_final, rank
        """
        rows: list[dict[str, Any]] = []
        for model, data in self.results.items():
            # Aggregate dimension scores across all instructions
            dim_means = self._aggregate_dimensions(data)
            # Derive pareto_bonus from s_total vs s_final
            s_total = data.get("total_score", 0.0)
            s_final = data.get("final_score", 0.0)
            p_bonus = _derive_pareto_bonus(s_total, s_final)
            rows.append(
                {
                    "model": model,
                    "s_total": round(s_total, 4),
                    "D1": round(dim_means.get("D1", 0.0), 4),
                    "D2": round(dim_means.get("D2", 0.0), 4),
                    "D3": round(dim_means.get("D3", 0.0), 4),
                    "D4": round(dim_means.get("D4", 0.0), 4),
                    "D5": round(dim_means.get("D5", 0.0), 4),
                    "pareto_bonus": round(p_bonus, 4),
                    "s_final": round(s_final, 4),
                }
            )
        return _assign_ranks(rows, "s_final")

    # ------------------------------------------------------------------
    # 2. Level Leaderboards
    # ------------------------------------------------------------------

    def level_leaderboards(self) -> dict[str, list[dict[str, Any]]]:
        """Generate per-level leaderboards (one per level).

        Returns:
            Dict mapping "L0".."L5" -> ranked list of model dicts.
            Each dict has: model, level_score, D1..D5, rank
        """
        all_levels: set[int] = set()
        for data in self.results.values():
            for key in data.get("per_level_scores", {}).keys():
                all_levels.add(int(key))

        boards: dict[str, list[dict[str, Any]]] = {}
        for lvl in sorted(all_levels):
            rows: list[dict[str, Any]] = []
            for model, data in self.results.items():
                lvl_score = data.get("per_level_scores", {}).get(str(lvl), 0.0)
                dim_means = self._aggregate_dimensions_for_level(data, lvl)
                row: dict[str, Any] = {
                    "model": model,
                    "level_score": round(lvl_score, 4),
                }
                for d in ("D1", "D2", "D3", "D4", "D5"):
                    row[d] = round(dim_means.get(d, 0.0), 4)
                rows.append(row)
            boards[f"L{lvl}"] = _assign_ranks(rows, "level_score")
        return boards

    # ------------------------------------------------------------------
    # 3. Dimension Leaderboards
    # ------------------------------------------------------------------

    def dimension_leaderboards(self) -> dict[str, list[dict[str, Any]]]:
        """Generate per-dimension leaderboards (one per dimension D1-D5).

        Returns:
            Dict mapping "D1".."D5" -> ranked list of model dicts.
            Each dict has: model, overall, L0..L5 scores, rank
        """
        boards: dict[str, list[dict[str, Any]]] = {}
        for dim in ("D1", "D2", "D3", "D4", "D5"):
            rows: list[dict[str, Any]] = []
            for model, data in self.results.items():
                level_dim_scores = self._dimension_per_level(data, dim)
                overall = (
                    _mean(list(level_dim_scores.values())) if level_dim_scores else 0.0
                )
                row: dict[str, Any] = {
                    "model": model,
                    "overall": round(overall, 4),
                }
                for lvl in range(6):
                    row[f"L{lvl}"] = round(level_dim_scores.get(lvl, 0.0), 4)
                rows.append(row)
            boards[dim] = _assign_ranks(rows, "overall")
        return boards

    # ------------------------------------------------------------------
    # 4. Efficiency Leaderboard
    # ------------------------------------------------------------------

    def efficiency_leaderboard(self, ranking_mode: str = "log") -> list[dict[str, Any]]:
        """Generate efficiency leaderboard.

        Args:
            ranking_mode: Ranking formula — "log" (default, design doc:
                Quality / log(1 + Total_Tokens)) or "linear"
                (Quality / (Total_Tokens / 1000)).

        Returns:
            List of dicts with keys:
            model, quality_per_ktoken, pareto_score, cost_at_q08,
            quality_at_budget_10k, frontier_distance, pareto_bonus, rank
        """
        if ranking_mode not in ("log", "linear"):
            raise ValueError(
                f"ranking_mode must be 'log' or 'linear', got {ranking_mode!r}"
            )
        # Build ParetoPoints for all models
        points: list[ParetoPoint] = []
        model_data: dict[str, dict[str, Any]] = {}
        for model, data in self.results.items():
            total_tokens = self._total_tokens(data)
            s_total = data.get("total_score", 0.0)
            pt = ParetoPoint(
                label=model,
                quality=s_total,
                cost=float(total_tokens),
            )
            points.append(pt)
            model_data[model] = {
                "total_tokens": total_tokens,
                "s_total": s_total,
            }

        frontier = compute_pareto_frontier(points) if points else []

        rows: list[dict[str, Any]] = []
        for pt in points:
            model = pt.label
            info = model_data[model]
            total_tokens = info["total_tokens"]
            s_total = info["s_total"]

            q_per_kt = quality_per_ktoken(s_total, total_tokens)
            p_score = _pareto_score_log(s_total, total_tokens)
            c_at_q = cost_at_quality(points, threshold=0.8)
            q_at_b = quality_at_budget(points, budget=10000)
            f_dist = frontier_distance(pt, frontier)
            p_bonus = pareto_bonus(pt, frontier)

            rows.append(
                {
                    "model": model,
                    "quality_per_ktoken": round(q_per_kt, 4),
                    "pareto_score": round(p_score, 4),
                    "cost_at_q08": round(c_at_q, 4) if c_at_q is not None else None,
                    "quality_at_budget_10k": round(q_at_b, 4),
                    "frontier_distance": round(f_dist, 4),
                    "pareto_bonus": round(p_bonus, 4),
                    "rank": 0,  # placeholder
                }
            )
        rank_key = "pareto_score" if ranking_mode == "log" else "quality_per_ktoken"
        return _assign_ranks(rows, rank_key)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_all(self, output_dir: str) -> dict[str, str]:
        """Export all leaderboards to JSON and Markdown files.

        Args:
            output_dir: Directory to write output files.

        Returns:
            Dict mapping filename -> absolute path of written files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}

        # Main leaderboard
        main = self.main_leaderboard()
        written.update(self._write_board(out, "main_leaderboard", main))

        # Level leaderboards
        level_boards = self.level_leaderboards()
        for key, board in level_boards.items():
            written.update(self._write_board(out, f"level_{key}_leaderboard", board))

        # Dimension leaderboards
        dim_boards = self.dimension_leaderboards()
        for key, board in dim_boards.items():
            written.update(
                self._write_board(out, f"dimension_{key}_leaderboard", board)
            )

        # Efficiency leaderboard
        eff = self.efficiency_leaderboard()
        written.update(self._write_board(out, "efficiency_leaderboard", eff))

        return written

    # ------------------------------------------------------------------
    # Markdown formatting
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(rows: list[dict[str, Any]], title: str = "") -> str:
        """Format a leaderboard as a Markdown table.

        Args:
            rows: List of row dicts (all must have same keys).
            title: Optional title to prepend as a heading.

        Returns:
            Markdown-formatted string.
        """
        if not rows:
            return f"## {title}\n\n(No data)\n" if title else "(No data)\n"

        keys = list(rows[0].keys())
        lines: list[str] = []
        if title:
            lines.append(f"## {title}\n")

        # Header
        header = "| " + " | ".join(keys) + " |"
        separator = "| " + " | ".join("---" for _ in keys) + " |"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for row in rows:
            cells = []
            for k in keys:
                val = row[k]
                if val is None:
                    cells.append("N/A")
                elif isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_dimensions(self, data: dict[str, Any]) -> dict[str, float]:
        """Compute mean dimension scores across all instructions."""
        dim_vals: dict[str, list[float]] = defaultdict(list)
        for item in data.get("per_instruction", []):
            for dim, val in item.get("dimension_scores", {}).items():
                dim_vals[dim].append(val)
        return {dim: _mean(vals) for dim, vals in dim_vals.items()}

    def _aggregate_dimensions_for_level(
        self, data: dict[str, Any], level: int
    ) -> dict[str, float]:
        """Compute mean dimension scores for a specific level."""
        dim_vals: dict[str, list[float]] = defaultdict(list)
        for item in data.get("per_instruction", []):
            if item.get("level") == level:
                for dim, val in item.get("dimension_scores", {}).items():
                    dim_vals[dim].append(val)
        return {dim: _mean(vals) for dim, vals in dim_vals.items()}

    def _dimension_per_level(self, data: dict[str, Any], dim: str) -> dict[int, float]:
        """Compute mean of a single dimension per level."""
        level_vals: dict[int, list[float]] = defaultdict(list)
        for item in data.get("per_instruction", []):
            lvl = item.get("level")
            val = item.get("dimension_scores", {}).get(dim, 0.0)
            if lvl is not None:
                level_vals[lvl].append(val)
        return {lvl: _mean(vals) for lvl, vals in level_vals.items()}

    def _total_tokens(self, data: dict[str, Any]) -> int:
        """Sum d5_tokens_per_valid across all instructions as proxy for total cost."""
        total = 0
        for item in data.get("per_instruction", []):
            tpv = item.get("d5_tokens_per_valid", 0.0)
            if tpv is not None and tpv != float("inf") and tpv > 0:
                total += int(tpv)
        return max(total, 1)  # Avoid division by zero

    @staticmethod
    def _write_board(
        out_dir: Path, name: str, rows: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Write a single leaderboard to JSON and Markdown files."""
        written: dict[str, str] = {}

        json_path = out_dir / f"{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        written[f"{name}.json"] = str(json_path)

        md_path = out_dir / f"{name}.md"
        title = name.replace("_", " ").title()
        md_content = LeaderboardGenerator.to_markdown(rows, title=title)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        written[f"{name}.md"] = str(md_path)

        return written


def _derive_pareto_bonus(
    s_total: float, s_final: float, pareto_lambda: float = 0.15
) -> float:
    """Derive Pareto bonus from s_total and s_final.

    Since S_final = S_total * (1 + lambda * bonus),
    bonus = (S_final / S_total - 1) / lambda.
    """
    if s_total <= 0 or pareto_lambda <= 0:
        return 0.0
    bonus = (s_final / s_total - 1.0) / pareto_lambda
    return max(0.0, min(1.0, bonus))
