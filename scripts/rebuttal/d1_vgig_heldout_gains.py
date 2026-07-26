"""D1: VGIG/combined gains on non-D4 dimensions by level."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    AVAILABLE_DIMENSIONS,
    best_baseline_runs,
    dimension_level_means,
    mean,
    quality_path,
    load_quality_file,
    write_json,
)

TARGETS = ("deepseekaiDeepSeekV3", "gpt4omini")
METHODS = ("vgig", "combined")
HELD_OUT_DIMENSIONS = ("D1", "D2", "D3")


def main() -> None:
    baselines = best_baseline_runs(LEVEL_WEIGHTS)
    targets: dict[str, object] = {}
    for target in TARGETS:
        baseline = baselines[target]
        baseline_dimensions = dimension_level_means(baseline["data"])
        methods: dict[str, object] = {}
        for method in METHODS:
            path = quality_path(target, method)
            if not path.exists():
                methods[method] = {"available": False, "reason": f"missing {path.name}"}
                continue
            method_dimensions = dimension_level_means(load_quality_file(str(path)))
            by_level = {}
            all_gains = []
            for level in range(6):
                gains = {}
                for dimension in HELD_OUT_DIMENSIONS:
                    if dimension in AVAILABLE_DIMENSIONS[level]:
                        gain = method_dimensions[level][dimension] - baseline_dimensions[level][dimension]
                        gains[dimension] = gain
                        all_gains.append(gain)
                    else:
                        gains[dimension] = None
                by_level[str(level)] = {
                    "baseline_means": {dim: baseline_dimensions[level][dim] if dim in AVAILABLE_DIMENSIONS[level] else None for dim in HELD_OUT_DIMENSIONS},
                    "method_means": {dim: method_dimensions[level][dim] if dim in AVAILABLE_DIMENSIONS[level] else None for dim in HELD_OUT_DIMENSIONS},
                    "gains": gains,
                }
            methods[method] = {
                "available": True,
                "per_level_non_D4_dimension_gains": by_level,
                "mean_gain_over_all_applicable_D1_D2_D3_cells": mean(all_gains),
                "positive_cells": sum(gain > 0 for gain in all_gains),
                "n_applicable_cells": len(all_gains),
                "all_applicable_cells_positive": all(gain > 0 for gain in all_gains),
            }
        targets[target] = {
            "baseline": {
                "strategy": baseline["strategy"],
                "quality_only_total": baseline["total"],
                "selection": "highest default-LEVEL_WEIGHTS quality-only total among available baseline strategies",
            },
            "methods": methods,
        }
    write_json(
        "D1_vgig_heldout_gains.json",
        {
            "analysis": "D1 VGIG held-out dimension gains",
            "zero_api_zero_model_calls": True,
            "held_out_definition": "D1/D2/D3 only; D4 is deliberately excluded. Null means the dimension is not defined at that level.",
            "targets": targets,
        },
    )


if __name__ == "__main__":
    main()
