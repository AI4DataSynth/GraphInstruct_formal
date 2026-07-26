"""B1B: model rankings under dimension subsets (D1-only / D4-only / D1+D4).

Directly answers the ask "how do rankings change under D4-only, D1-only,
and D1+D4 subsets": each model's best quality-only baseline run is held
fixed, and that same run is re-scored using only the named dimension
subset (unweighted mean over the subset's per-instruction scores, then
the default level weights). Holding the run fixed isolates the metric
effect from strategy re-selection.

Writes B1B_dimension_subset_rankings.json into the rebuttal artifact dir.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    MODEL_NAMES,
    best_baseline_runs,
    kendall_tau_b,
    load_quality_file,
    quality_path,
    rank_desc,
    rank_positions,
    safe_float,
    spearman,
    top_overlap,
    write_json,
)

SUBSETS: dict[str, tuple[str, ...]] = {
    "D1_only": ("D1",),
    "D4_only": ("D4",),
    "D1_D4": ("D1", "D4"),
}


def record_dim(record: dict[str, Any], dimension: str) -> float:
    scores = record.get("dimension_scores", {})
    if dimension == "D4":
        return safe_float(record.get("d4_instruction_score", scores.get("D4")))
    return safe_float(scores.get(dimension))


def subset_total(data: dict[str, Any], dims: tuple[str, ...]) -> float:
    level_sums: dict[int, list[float]] = {level: [] for level in range(6)}
    for record in data["per_instruction"]:
        level = int(record["level"])
        values = [record_dim(record, d) for d in dims]
        level_sums[level].append(sum(values) / len(values))
    total = 0.0
    for level in range(6):
        cells = level_sums[level]
        level_score = sum(cells) / len(cells) if cells else 0.0
        total += LEVEL_WEIGHTS[level] * level_score
    return total


def main() -> None:
    selected = best_baseline_runs(list(LEVEL_WEIGHTS))
    reference_totals = {m: float(selected[m]["total"]) for m in BASELINE_MODELS}
    reference_order = rank_desc(reference_totals)
    reference_pos = rank_positions(reference_totals)

    loaded = {
        m: load_quality_file(str(quality_path(m, str(selected[m]["strategy"]))))
        for m in BASELINE_MODELS
    }

    subsets_out: dict[str, Any] = {}
    for name, dims in SUBSETS.items():
        totals = {m: subset_total(loaded[m], dims) for m in BASELINE_MODELS}
        order = rank_desc(totals)
        pos = rank_positions(totals)
        moves = [
            {
                "model": MODEL_NAMES[m],
                "reference_rank": reference_pos[m],
                "subset_rank": pos[m],
            }
            for m in reference_order
            if reference_pos[m] != pos[m]
        ]
        vs = {
            "spearman_rho": spearman(reference_totals, totals),
            "kendall_tau_b": kendall_tau_b(reference_totals, totals),
            "top_retention": [
                top_overlap(reference_order, order, k) for k in (5, 9, 12)
            ],
            "max_rank_displacement": max(
                abs(reference_pos[m] - pos[m]) for m in BASELINE_MODELS
            ),
            "rank_moves": moves,
        }
        subsets_out[name] = {
            "dimensions": list(dims),
            "totals": {MODEL_NAMES[m]: round(totals[m], 4) for m in order},
            "ranking": [MODEL_NAMES[m] for m in order],
            "vs_full_quality_only": vs,
        }
        print(
            f"{name:8s} top5 {vs['top_retention'][0]['retained']}/5  "
            f"top9 {vs['top_retention'][1]['retained']}/9  "
            f"rho {vs['spearman_rho']:.4f}  tau {vs['kendall_tau_b']:.4f}  "
            f"max_shift {vs['max_rank_displacement']}"
        )

    write_json(
        "B1B_dimension_subset_rankings.json",
        {
            "analysis": "B1B dimension-subset leaderboards (D1-only / D4-only / D1+D4)",
            "zero_api_zero_model_calls": True,
            "construction": (
                "each model's best quality-only baseline run held fixed; the "
                "run is re-scored as the unweighted per-instruction mean over "
                "the subset dimensions, aggregated with default level weights"
            ),
            "reference": "full quality-only (D1-D4 renormalized) leaderboard",
            "reference_ranking": [MODEL_NAMES[m] for m in reference_order],
            "subsets": subsets_out,
        },
    )
    print("Written: B1B_dimension_subset_rankings.json")


if __name__ == "__main__":
    main()
