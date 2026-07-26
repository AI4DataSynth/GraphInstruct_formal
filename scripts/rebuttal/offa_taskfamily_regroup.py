"""OFF-A: task-family regrouping of the tier-gap discrimination metric.

The question: does the discrimination conclusion survive a grouping by *task
family* rather than by the L0-L5 ladder.  We regroup the six levels into five
task families motivated by inspecting the instructions:

    Format               = {L0}       bare "make a graph of size n" (mean 1.8 constraints)
    StructuralConstraint = {L1, L2}   explicit structural constraints, single (L1, ~2.5)
                                       and composed / multi-constraint (L2, ~3.9)
    NumericProperty      = {L3}       distributional targets: density / modularity /
                                       clustering / path length (~6.9 constraints)
    DomainSemantic       = {L4}       domain-grounded graphs (social/citation/...) with
                                       a ``domain`` label (~5.4 constraints)
    EditingReasoning     = {L5}       edit a given base graph / multi-step task_type

The "multi-constraint composition family" of the paper is L2, which lives inside
StructuralConstraint.  We recompute the continuous-D4 tier gap (T1 minus T3,
paper Q-tiers), pooled over each family's instructions, with a paired
instruction bootstrap CI, and check whether StructuralConstraint remains the
discrimination peak.  Reads only results/*.quality.json (best quality-only
baseline per model, as in C1).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    MODEL_NAMES,
    Q_TIERS,
    best_baseline_runs,
    d4_level_means,
    instruction_records_by_level,
    paired_bootstrap_tier_gap,
    quality_level_score,
    safe_float,
    tier_gap_from_level_vectors,
    write_json,
)

# Task-family -> member levels. StructuralConstraint holds the multi-constraint
# composition family (L2) together with single-constraint L1.
FAMILIES: dict[str, list[int]] = {
    "Format": [0],
    "StructuralConstraint": [1, 2],
    "NumericProperty": [3],
    "DomainSemantic": [4],
    "EditingReasoning": [5],
}
# A finer reference scheme that isolates the L2 composition family, so the peak
# can be attributed even after L1+L2 are merged above.
FINE_FAMILIES: dict[str, list[int]] = {
    "Format": [0],
    "SingleConstraint": [1],
    "MultiConstraint": [2],
    "NumericProperty": [3],
    "DomainSemantic": [4],
    "EditingReasoning": [5],
}
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 5000


def _record_metric(record: Mapping[str, Any], kind: str) -> float:
    if kind == "d4":
        return safe_float(
            record.get(
                "d4_instruction_score", record.get("dimension_scores", {}).get("D4")
            )
        )
    return quality_level_score(int(record["level"]), record.get("dimension_scores", {}))


def family_metric_per_model(
    data: Mapping[str, Any], levels: Sequence[int], kind: str
) -> float:
    """Pooled (instruction-weighted) mean metric over a family's instructions."""
    grouped = instruction_records_by_level(data)
    values = [
        _record_metric(record, kind) for level in levels for record in grouped[level]
    ]
    return sum(values) / len(values)


def family_id_metric_map(
    data: Mapping[str, Any], levels: Sequence[int], kind: str
) -> dict[str, float]:
    grouped = instruction_records_by_level(data)
    return {
        str(record["instruction_id"]): _record_metric(record, kind)
        for level in levels
        for record in grouped[level]
    }


def family_tier_gaps(
    selected: Mapping[str, Any], families: Mapping[str, list[int]], kind: str
) -> dict[str, Any]:
    high, low = Q_TIERS["T1"], Q_TIERS["T3"]
    gaps: dict[str, float] = {}
    high_means: dict[str, float] = {}
    low_means: dict[str, float] = {}
    boot: dict[str, Any] = {}
    for family, levels in families.items():
        per_model = {
            model: family_metric_per_model(selected[model]["data"], levels, kind)
            for model in BASELINE_MODELS
        }
        high_mean = sum(per_model[m] for m in high) / len(high)
        low_mean = sum(per_model[m] for m in low) / len(low)
        gaps[family] = high_mean - low_mean
        high_means[family] = high_mean
        low_means[family] = low_mean
        maps = {
            model: family_id_metric_map(selected[model]["data"], levels, kind)
            for model in list(high) + list(low)
        }
        boot[family] = paired_bootstrap_tier_gap(
            maps, high, low, BOOTSTRAP_SEED, N_BOOTSTRAP
        )
    ordered = sorted(gaps, key=lambda f: (-gaps[f], f))
    peak = ordered[0]
    runner_up = ordered[1]
    peak_ci = boot[peak]["ci_95_percentile"]
    runner_ci = boot[runner_up]["ci_95_percentile"]
    return {
        "metric": "continuous d4_instruction_score"
        if kind == "d4"
        else "quality-only level_score",
        "tier_gaps_T1_minus_T3": gaps,
        "T1_family_means": high_means,
        "T3_family_means": low_means,
        "bootstrap_ci_95": boot,
        "ranking_desc": ordered,
        "peak_family": peak,
        "peak_is_structural_constraint": peak == "StructuralConstraint",
        "peak_significant_over_runner_up": peak_ci[0] > runner_ci[1],
        "runner_up_family": runner_up,
    }


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)

    # Per-level D4 gaps for transparency (identical basis to C1).
    d4_vectors = {
        model: d4_level_means(selected[model]["data"]) for model in BASELINE_MODELS
    }
    per_level_d4_gaps = tier_gap_from_level_vectors(
        d4_vectors, Q_TIERS["T1"], Q_TIERS["T3"]
    )

    write_json(
        "OFFA_taskfamily_regroup.json",
        {
            "analysis": "OFF-A task-family regrouping of the tier-gap discrimination metric",
            "zero_api_zero_model_calls": True,
            "question": "Does the discrimination peak survive a grouping by task family rather than by L-level?",
            "tiering": {
                "scheme": "paper coarse Q-tiers",
                "T1": Q_TIERS["T1"],
                "T3": Q_TIERS["T3"],
            },
            "family_mapping": FAMILIES,
            "family_mapping_justification": {
                "Format": "L0: bare 'graph with n nodes' (mean 1.8 explicit constraints).",
                "StructuralConstraint": "L1 single structural constraint (~2.5) + L2 composed multi-constraint (~3.9); contains the paper's multi-constraint composition family (L2).",
                "NumericProperty": "L3 distributional targets density/modularity/clustering/path-length (~6.9).",
                "DomainSemantic": "L4 domain-grounded graphs with a domain label (~5.4).",
                "EditingReasoning": "L5 edit-a-base-graph / multi-step task_type.",
            },
            "best_baseline_strategies": {
                m: selected[m]["strategy"] for m in BASELINE_MODELS
            },
            "per_level_d4_tier_gaps_reference": {
                "levels": list(range(6)),
                "gaps_T1_minus_T3": per_level_d4_gaps,
            },
            "primary_continuous_d4": family_tier_gaps(selected, FAMILIES, "d4"),
            "secondary_quality_only": family_tier_gaps(selected, FAMILIES, "quality"),
            "fine_scheme_continuous_d4": family_tier_gaps(
                selected, FINE_FAMILIES, "d4"
            ),
        },
    )


if __name__ == "__main__":
    main()
