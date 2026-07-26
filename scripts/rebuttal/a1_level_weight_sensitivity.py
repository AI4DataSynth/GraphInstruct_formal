"""A1: quality-only model ranking sensitivity to level-weight schemes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    MODEL_NAMES,
    best_baseline_runs,
    kendall_tau_b,
    normalize,
    rank_desc,
    rank_positions,
    spearman,
    top_overlap,
    write_json,
)


def schemes() -> dict[str, list[float]]:
    default = list(LEVEL_WEIGHTS)
    return {
        "default": default,
        "uniform": [1 / 6] * 6,
        "reversed": list(reversed(default)),
        "plus50": normalize([weight * (1.5 if level in (4, 5) else 1.0) for level, weight in enumerate(default)]),
        "minus50": normalize([weight * (0.5 if level in (4, 5) else 1.0) for level, weight in enumerate(default)]),
        "n_proportional": normalize([100, 200, 200, 150, 100, 50]),
    }


def main() -> None:
    choices: dict[str, dict[str, object]] = {}
    for scheme_name, level_weights in schemes().items():
        selected = best_baseline_runs(level_weights)
        totals = {model: float(selected[model]["total"]) for model in BASELINE_MODELS}
        order = rank_desc(totals)
        choices[scheme_name] = {
            "level_weights": level_weights,
            "quality_only_totals": totals,
            "ranking": order,
            "rank_positions": rank_positions(totals),
            "best_strategy_per_model": {model: str(selected[model]["strategy"]) for model in BASELINE_MODELS},
        }

    default_totals = choices["default"]["quality_only_totals"]
    default_order = choices["default"]["ranking"]
    default_positions = choices["default"]["rank_positions"]
    for scheme_name, entry in choices.items():
        totals = entry["quality_only_totals"]
        ranking = entry["ranking"]
        positions = entry["rank_positions"]
        entry["vs_default"] = {
            "spearman_rho": spearman(default_totals, totals),
            "kendall_tau_b": kendall_tau_b(default_totals, totals),
            "top_retention": [top_overlap(default_order, ranking, k) for k in (5, 9, 15)],
            "default_top15_effective_n": min(15, len(default_order)),
            "default_top15_max_rank_displacement": max(
                abs(default_positions[model] - positions[model]) for model in default_order[: min(15, len(default_order))]
            ),
        }

    write_json(
        "A1_level_weight_sensitivity.json",
        {
            "analysis": "A1 level-weight sensitivity of model quality-only rankings",
            "zero_api_zero_model_calls": True,
            "metric_definition": "D5=0; repository DIMENSION_WEIGHTS D1-D4 renormalized per level; best baseline strategy selected separately under each level-weight scheme.",
            "models": [{"slug": model, "name": MODEL_NAMES[model]} for model in BASELINE_MODELS],
            "note": "Only 12 baseline models exist, so requested top-15 retention is reported with effective_k=12.",
            "schemes": choices,
        },
    )


if __name__ == "__main__":
    main()
