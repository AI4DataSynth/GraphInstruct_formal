"""L4D: cross-model L4 dimensional decomposition."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import BASELINE_MODELS, best_baseline_runs, dimension_level_means, mean, write_json


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)
    per_model = {
        model: {
            "strategy": selected[model]["strategy"],
            "means": {dimension: dimension_level_means(selected[model]["data"])[4][dimension] for dimension in ("D1", "D2", "D3", "D4")},
        }
        for model in BASELINE_MODELS
    }
    dimensions = {}
    for dimension in ("D1", "D2", "D3", "D4"):
        values = [per_model[model]["means"][dimension] for model in BASELINE_MODELS]
        average = mean(values)
        dimensions[dimension] = {
            "mean": average,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "std_sample": statistics.stdev(values),
            "cv_sample": statistics.stdev(values) / average if average else None,
        }
    d2 = dimensions["D2"]
    other_cvs = [dimensions[dimension]["cv_sample"] for dimension in ("D1", "D3", "D4") if dimensions[dimension]["cv_sample"] is not None]
    write_json(
        "L4D_l4_dimension_decomp.json",
        {
            "analysis": "L4D L4 cross-model dimension decomposition",
            "zero_api_zero_model_calls": True,
            "selection": "One default-LEVEL_WEIGHTS quality-only best baseline strategy per model.",
            "per_model": per_model,
            "dimensions": dimensions,
            "factual_pattern_flags": {
                "d2_lowest_mean": d2["mean"] == min(dimensions[dimension]["mean"] for dimension in dimensions),
                "d2_highest_cv_among_defined": d2["cv_sample"] is not None and d2["cv_sample"] == max(other_cvs + [d2["cv_sample"]]),
                "d1_d3_d4_mean_above_d2": all(dimensions[dimension]["mean"] > d2["mean"] for dimension in ("D1", "D3", "D4")),
            },
        },
    )


if __name__ == "__main__":
    main()
