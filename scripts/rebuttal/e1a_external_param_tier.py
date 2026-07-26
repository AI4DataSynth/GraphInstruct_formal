"""E1a: open-model active-parameter tier control."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import ACTIVE_PARAMS_B, best_baseline_runs, peak_summary, tier_gap_from_level_vectors, write_json


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)
    vectors = {model: list(selected[model]["level_means"]) for model in ACTIVE_PARAMS_B}
    ordered = sorted(ACTIVE_PARAMS_B, key=lambda model: (-ACTIVE_PARAMS_B[model], model))
    high, low = ordered[:3], ordered[3:]
    gaps = tier_gap_from_level_vectors(vectors, high, low)
    write_json(
        "E1a_external_param_tier.json",
        {
            "analysis": "E1a active-parameter high/low tier control",
            "zero_api_zero_model_calls": True,
            "metric": "Quality-only D1-D4 level score, using imported repository dimension weights and default LEVEL_WEIGHTS for best-baseline selection.",
            "active_params_billions": ACTIVE_PARAMS_B,
            "groups": {"high": high, "low": low},
            "selected_strategies": {model: selected[model]["strategy"] for model in ACTIVE_PARAMS_B},
            "quality_only_tier_gaps_high_minus_low": gaps,
            **peak_summary(gaps),
            "coverage_limitation": "This external anchor covers only six open-weight models. Closed models have no supplied active-parameter value and are intentionally excluded.",
        },
    )


if __name__ == "__main__":
    main()
