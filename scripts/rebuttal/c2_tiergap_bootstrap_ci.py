"""C2: paired per-instruction bootstrap CIs for Q-tier D4 gaps."""

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
    paired_bootstrap_tier_gap,
    peak_summary,
    records_to_metric_map,
    safe_float,
    write_json,
)

N_BOOTSTRAP = 2000
BASE_SEED = 3970


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)
    output = {}
    observed_gaps = []
    for level in range(6):
        metric_maps = {
            model: records_to_metric_map(
                selected[model]["data"],
                level,
                lambda record: safe_float(record.get("d4_instruction_score", record.get("dimension_scores", {}).get("D4"))),
            )
            for model in BASELINE_MODELS
        }
        result = paired_bootstrap_tier_gap(metric_maps, Q_TIERS["T1"], Q_TIERS["T3"], BASE_SEED + level, N_BOOTSTRAP)
        output[str(level)] = result
        observed_gaps.append(result["observed_gap"])
    l2_lower = output["2"]["ci_95_percentile"][0]
    l1_upper = output["1"]["ci_95_percentile"][1]
    l3_upper = output["3"]["ci_95_percentile"][1]
    write_json(
        "C2_tiergap_bootstrap_ci.json",
        {
            "analysis": "C2 bootstrap 95% CI for continuous-D4 Q-tier gap",
            "zero_api_zero_model_calls": True,
            "selection": "Default LEVEL_WEIGHTS quality-only best baseline strategy per model, selected once on the full dataset; Q tiers are externally specified by the paper.",
            "selected_strategies": {model: selected[model]["strategy"] for model in BASELINE_MODELS},
            "tiers": Q_TIERS,
            "bootstrap_method": "Paired cluster bootstrap over common per-instruction IDs within each level; all model scores for a sampled instruction remain paired. Unit is the five-sample-aggregated per_instruction score, because per-sample quality detail is unavailable.",
            "levels": output,
            "observed_gap_peak": peak_summary(observed_gaps),
            "l2_ci_lower_exceeds_neighbor_ci_upper": {
                "vs_L1": l2_lower > l1_upper,
                "vs_L3": l2_lower > l3_upper,
                "both": l2_lower > l1_upper and l2_lower > l3_upper,
                "values": {"L2_lower": l2_lower, "L1_upper": l1_upper, "L3_upper": l3_upper},
            },
        },
    )


if __name__ == "__main__":
    main()
