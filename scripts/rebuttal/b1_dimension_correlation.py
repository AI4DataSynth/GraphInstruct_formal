"""B1: pooled level-wise D1--D4 correlation and orthogonal failure cases."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.common import AVAILABLE_DIMENSIONS, all_quality_runs, pearson, safe_float, spearman_from_lists, write_json


def main() -> None:
    runs = all_quality_runs()
    by_level = {
        level: {dimension: [] for dimension in AVAILABLE_DIMENSIONS[level]}
        for level in range(6)
    }
    first_predicate: dict[int, list[dict[str, object]]] = {level: [] for level in range(6)}
    second_predicate: dict[int, list[dict[str, object]]] = {level: [] for level in range(6)}
    for run_name, data in runs:
        for record in data["per_instruction"]:
            level = int(record["level"])
            scores = record.get("dimension_scores", {})
            for dimension in AVAILABLE_DIMENSIONS[level]:
                by_level[level][dimension].append(safe_float(scores.get(dimension)))
            d1 = safe_float(scores.get("D1"))
            d4 = safe_float(record.get("d4_instruction_score", scores.get("D4")))
            case = {"run": run_name, "instruction_id": record["instruction_id"], "D1": d1, "D4": d4}
            if math_is_one(d1) and d4 < 1.0 - 1e-12:
                first_predicate[level].append(case)
            if d4 >= 0.7 and d1 < 0.5:
                second_predicate[level].append(case)

    levels: dict[str, object] = {}
    d1_d4_curve: list[dict[str, object]] = []
    for level in range(6):
        correlations = {}
        for first, second in itertools.combinations(AVAILABLE_DIMENSIONS[level], 2):
            xs, ys = by_level[level][first], by_level[level][second]
            correlations[f"{first}__{second}"] = {
                "n": len(xs),
                "pearson_r": pearson(xs, ys),
                "spearman_rho": spearman_from_lists(xs, ys),
            }
        d1_d4 = correlations["D1__D4"]
        d1_not_d4 = sorted(first_predicate[level], key=lambda item: (item["D4"], item["run"], item["instruction_id"]))[:3]
        d4_not_d1 = sorted(second_predicate[level], key=lambda item: (item["D1"], -item["D4"], item["run"], item["instruction_id"]))[:3]
        levels[str(level)] = {
            "available_dimensions": list(AVAILABLE_DIMENSIONS[level]),
            "pairwise_correlations": correlations,
            "orthogonal_failure_cases": {
                "D1_equals_1_D4_below_1": d1_not_d4,
                "D4_high_D1_low": d4_not_d1,
                "empty_predicates": {
                    "D1_equals_1_D4_below_1": not d1_not_d4,
                    "D4_high_D1_low": not d4_not_d1,
                },
            },
        }
        d1_d4_curve.append({"level": level, **d1_d4})
    write_json(
        "B1_dimension_correlation.json",
        {
            "analysis": "B1 pooled dimension correlations by level",
            "zero_api_zero_model_calls": True,
            "pooling": "All readable results/*.quality.json runs; each per_instruction value is a five-sample aggregate, not an independent sample-level observation.",
            "n_quality_files": len(runs),
            "d1_d4_curve": d1_d4_curve,
            "levels": levels,
        },
    )


def math_is_one(value: float) -> bool:
    return abs(value - 1.0) <= 1e-12


if __name__ == "__main__":
    main()
