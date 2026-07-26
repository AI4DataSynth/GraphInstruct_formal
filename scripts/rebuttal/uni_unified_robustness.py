"""UNI: one table of tier-gap robustness controls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    Q_TIERS,
    best_baseline_runs,
    d4_level_means,
    normalize,
    peak_summary,
    split_data_tier_results,
    tier_gap_from_level_vectors,
    write_json,
)

SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)


def quality_gaps(weights: list[float]) -> tuple[list[float], dict[str, str]]:
    selected = best_baseline_runs(weights)
    vectors = {model: list(selected[model]["level_means"]) for model in BASELINE_MODELS}
    return tier_gap_from_level_vectors(vectors, Q_TIERS["T1"], Q_TIERS["T3"]), {model: str(selected[model]["strategy"]) for model in BASELINE_MODELS}


def main() -> None:
    default_weights = list(LEVEL_WEIGHTS)
    uniform_weights = [1 / 6] * 6
    n_weights = normalize([100, 200, 200, 150, 100, 50])
    default_gaps, default_strategies = quality_gaps(default_weights)
    uniform_gaps, uniform_strategies = quality_gaps(uniform_weights)
    n_gaps, n_strategies = quality_gaps(n_weights)
    default_selected = best_baseline_runs(default_weights)
    d4_vectors = {model: d4_level_means(default_selected[model]["data"]) for model in BASELINE_MODELS}
    d4_gaps = tier_gap_from_level_vectors(d4_vectors, Q_TIERS["T1"], Q_TIERS["T3"])
    split = split_data_tier_results(SPLIT_SEEDS)
    columns = {
        "q_tier_default_quality_only": {
            "label": "① Q-tier + default quality-only",
            "gaps": default_gaps,
            "level_weights": default_weights,
            "selected_strategies": default_strategies,
            **peak_summary(default_gaps),
        },
        "q_tier_continuous_d4": {
            "label": "② Q-tier + continuous D4",
            "gaps": d4_gaps,
            "selected_strategies": {model: default_selected[model]["strategy"] for model in BASELINE_MODELS},
            **peak_summary(d4_gaps),
        },
        "split_data_continuous_d4": {
            "label": "③ split-data tier + continuous D4",
            "gaps": split["mean_d4_tier_gaps"],
            "std_gaps": split["std_d4_tier_gaps"],
            "seeds": list(SPLIT_SEEDS),
            "l2_peak_frequency": split["l2_peak_frequency"],
            **peak_summary(split["mean_d4_tier_gaps"]),
        },
        "q_tier_uniform_quality_only": {
            "label": "④ Q-tier + uniform-level quality-only",
            "gaps": uniform_gaps,
            "level_weights": uniform_weights,
            "selected_strategies": uniform_strategies,
            **peak_summary(uniform_gaps),
        },
        "q_tier_n_proportional_quality_only": {
            "label": "⑤ Q-tier + N-proportional quality-only",
            "gaps": n_gaps,
            "level_weights": n_weights,
            "selected_strategies": n_strategies,
            **peak_summary(n_gaps),
        },
    }
    write_json(
        "UNI_unified_robustness.json",
        {
            "analysis": "UNI unified tier-gap robustness table",
            "zero_api_zero_model_calls": True,
            "levels": list(range(6)),
            "tier_gap_definition": "mean(T1) - mean(T3); Q-tier groups are the supplied paper groups. Quality-only means D5=0 with D1-D4 weights renormalized from graphinstruct.scoring.",
            "columns": columns,
            "l2_is_peak_in_all_columns": all(column["l2_is_peak"] for column in columns.values()),
            "l2_is_unique_peak_in_all_columns": all(column["l2_is_unique_peak"] for column in columns.values()),
        },
    )


if __name__ == "__main__":
    main()
