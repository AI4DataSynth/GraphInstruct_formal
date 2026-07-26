"""A2: cross-check model rankings under four aggregations of level scores."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import BASELINE_MODELS, best_baseline_runs, kendall_tau_b, rank_desc, rank_positions, top_overlap, write_json


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)
    level_scores = {model: list(selected[model]["level_means"]) for model in BASELINE_MODELS}
    n_models = len(BASELINE_MODELS)
    per_level_positions = [
        rank_positions({model: level_scores[model][level] for model in BASELINE_MODELS})
        for level in range(6)
    ]
    score_sets = {
        "weighted_sum": {
            model: sum(weight * value for weight, value in zip(LEVEL_WEIGHTS, level_scores[model]))
            for model in BASELINE_MODELS
        },
        "average_rank": {
            model: -sum(positions[model] for positions in per_level_positions) / 6
            for model in BASELINE_MODELS
        },
        "borda_count": {
            model: sum(n_models - positions[model] for positions in per_level_positions)
            for model in BASELINE_MODELS
        },
        "geometric_mean": {
            model: math.prod(max(value, 1e-12) for value in level_scores[model]) ** (1 / 6)
            for model in BASELINE_MODELS
        },
    }
    aggregations = {
        name: {"scores": scores, "ranking": rank_desc(scores), "rank_positions": rank_positions(scores)}
        for name, scores in score_sets.items()
    }
    pairwise = {}
    aggregation_names = list(aggregations)
    for index, first in enumerate(aggregation_names):
        for second in aggregation_names[index + 1 :]:
            first_order = aggregations[first]["ranking"]
            second_order = aggregations[second]["ranking"]
            pairwise[f"{first}__vs__{second}"] = {
                "kendall_tau_b": kendall_tau_b(score_sets[first], score_sets[second]),
                "top5_intersection": top_overlap(first_order, second_order, 5),
                "top10_intersection": top_overlap(first_order, second_order, 10),
            }
    write_json(
        "A2_aggregation_crosscheck.json",
        {
            "analysis": "A2 aggregation cross-check",
            "zero_api_zero_model_calls": True,
            "selection": "Each model first uses its default LEVEL_WEIGHTS quality-only best baseline strategy; all four aggregations use that same selected run.",
            "selected_strategies": {model: selected[model]["strategy"] for model in BASELINE_MODELS},
            "per_model_quality_only_level_scores": level_scores,
            "aggregations": aggregations,
            "pairwise_agreement": pairwise,
        },
    )


if __name__ == "__main__":
    main()
