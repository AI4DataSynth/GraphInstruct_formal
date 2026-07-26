"""C1: continuous D4 tier gaps under Q, split-data, and parameter tiers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    ACTIVE_PARAMS_B,
    BASELINE_MODELS,
    Q_TIERS,
    best_baseline_runs,
    d4_level_means,
    peak_summary,
    split_data_tier_results,
    tier_gap_from_level_vectors,
    write_json,
)

SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)
    vectors = {model: d4_level_means(selected[model]["data"]) for model in BASELINE_MODELS}
    q_gaps = tier_gap_from_level_vectors(vectors, Q_TIERS["T1"], Q_TIERS["T3"])
    ordered_params = sorted(ACTIVE_PARAMS_B, key=lambda model: (-ACTIVE_PARAMS_B[model], model))
    high_params, low_params = ordered_params[:3], ordered_params[3:]
    param_gaps = tier_gap_from_level_vectors(vectors, high_params, low_params)
    split = split_data_tier_results(SPLIT_SEEDS)
    write_json(
        "C1_d4_only_tiergap.json",
        {
            "analysis": "C1 continuous D4 tier gaps across three tier definitions",
            "zero_api_zero_model_calls": True,
            "metric": "Mean d4_instruction_score, a continuous [0,1] value present across all six levels.",
            "best_baseline_selection": {
                "criterion": "default LEVEL_WEIGHTS quality-only total (D5 excluded; D1-D4 renormalized)",
                "strategies": {model: selected[model]["strategy"] for model in BASELINE_MODELS},
            },
            "q_based_tier": {
                "groups": Q_TIERS,
                "d4_tier_gaps_T1_minus_T3": q_gaps,
                **peak_summary(q_gaps),
            },
            "split_data_tier": {
                "seeds": list(SPLIT_SEEDS),
                "mean_d4_tier_gaps_T1_minus_T3": split["mean_d4_tier_gaps"],
                "std_d4_tier_gaps": split["std_d4_tier_gaps"],
                "l2_peak_frequency": split["l2_peak_frequency"],
                "peak_levels": split["peak_levels"],
                "l2_is_peak": split["l2_is_peak"],
                "l2_is_unique_peak": split["l2_is_unique_peak"],
                "selection_analysis_disjoint": True,
            },
            "external_param_tier": {
                "active_params_billions": ACTIVE_PARAMS_B,
                "groups": {"high": high_params, "low": low_params},
                "d4_tier_gaps_high_minus_low": param_gaps,
                **peak_summary(param_gaps),
                "coverage_limitation": "Only six open-weight models have the supplied active-parameter anchor; all closed models are excluded.",
            },
        },
    )


if __name__ == "__main__":
    main()
