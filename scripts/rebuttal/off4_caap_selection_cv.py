"""OFF-4: cross-validation that CAAP-style per-(level,tier) selection is not overfit.

The CAAP decision rule ("pick the strategy that works best for this level and
capability tier") is derived from the same benchmark it is later evaluated on,
which was flagged as circular.  This script breaks the circularity two ways and
reports the held-out gain of the learned selection over each model's best single
fixed strategy:

1. Leave-one-model-out (primary): for each held-out model, learn the best
   strategy per (level, coarse Q-tier) from the OTHER eleven models (only its
   tier-mates inform its tier), then apply that rule to the held-out model.
   The held-out model never informs its own selection, so a positive held-out
   gain cannot be selection overfitting to that model.

2. Five-fold by instruction (corroboration): learn the best strategy per
   (level, tier) on 4/5 of the instructions (all models pooled per tier), apply
   it to the held-out instruction fold, and average the per-model gain.

Coarse Q-tiers (T1/T2/T3) are used so that leaving one model out always leaves
at least one tier-mate; the production CAAP table additionally splits T2 into
T2-GPT / T2-open.  Everything reads only results/*.quality.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    BASELINE_STRATEGIES,
    MODEL_NAMES,
    Q_TIERS,
    all_instruction_ids_by_level,
    baseline_runs,
    best_baseline_runs,
    mean,
    quality_level_means,
    quality_level_score,
    split_instruction_ids,
    weighted_total,
    write_json,
)

MODEL_TIER = {model: tier for tier, models in Q_TIERS.items() for model in models}
CV_SEED = 2026


def _model_strategy_level_means(
    runs: Mapping[str, Any],
) -> dict[str, dict[str, list[float]]]:
    """{model: {strategy: [level0..level5 quality-only means]}}."""
    return {
        model: {
            strategy: quality_level_means(data) for strategy, data in strategies.items()
        }
        for model, strategies in runs.items()
    }


def _learn_selection(
    train_models: Sequence[str],
    cache: Mapping[str, Mapping[str, Sequence[float]]],
    selected_ids: Mapping[int, set[str]] | None = None,
    runs: Mapping[str, Any] | None = None,
) -> dict[tuple[int, str], str]:
    """Best strategy per (level, tier) averaged over training tier-mates.

    When ``selected_ids`` is given the level means are recomputed on that
    instruction subset (instruction-fold mode); otherwise the full-run cache is
    used (leave-one-model-out mode).
    """
    learned: dict[tuple[int, str], str] = {}
    for tier in ("T1", "T2", "T3"):
        tier_models = [m for m in train_models if MODEL_TIER[m] == tier]
        for level in range(6):
            strat_scores: dict[str, list[float]] = defaultdict(list)
            for model in tier_models:
                strategies = runs[model] if runs is not None else None
                for strategy in BASELINE_STRATEGIES:
                    if (
                        strategies is not None
                        and strategy in strategies
                        and selected_ids is not None
                    ):
                        means = quality_level_means(strategies[strategy], selected_ids)
                        strat_scores[strategy].append(means[level])
                    elif strategy in cache[model]:
                        strat_scores[strategy].append(cache[model][strategy][level])
            averaged = {s: sum(v) / len(v) for s, v in strat_scores.items() if v}
            best_value = max(averaged.values())
            learned[(level, tier)] = sorted(
                s for s in averaged if averaged[s] == best_value
            )[0]
    return learned


def leave_one_model_out(
    runs: Mapping[str, Any], baselines: Mapping[str, Any]
) -> dict[str, Any]:
    cache = _model_strategy_level_means(runs)
    per_model: dict[str, Any] = {}
    gains: list[float] = []
    gains_vs_zero: list[float] = []
    for held in BASELINE_MODELS:
        train = [m for m in BASELINE_MODELS if m != held]
        learned = _learn_selection(train, cache)
        held_tier = MODEL_TIER[held]
        available = cache[held]
        fallback = baselines[held]["strategy"]
        applied_means: list[float] = []
        applied_strategy: dict[str, str] = {}
        for level in range(6):
            chosen = learned[(level, held_tier)]
            if chosen not in available:
                chosen = fallback
            applied_means.append(available[chosen][level])
            applied_strategy[str(level)] = chosen
        cv_total = weighted_total(applied_means, LEVEL_WEIGHTS)
        baseline_total = baselines[held]["total"]
        zero_total = (
            weighted_total(available["zero-shot"], LEVEL_WEIGHTS)
            if "zero-shot" in available
            else None
        )
        gain = cv_total - baseline_total
        gains.append(gain)
        if zero_total is not None:
            gains_vs_zero.append(cv_total - zero_total)
        per_model[held] = {
            "display_name": MODEL_NAMES[held],
            "tier": held_tier,
            "learned_strategy_per_level": applied_strategy,
            "cv_selected_total": cv_total,
            "best_single_strategy": {
                "strategy": baselines[held]["strategy"],
                "total": baseline_total,
            },
            "gain_vs_best_single_strategy": gain,
            "gain_vs_zero_shot": (cv_total - zero_total)
            if zero_total is not None
            else None,
        }
    return {
        "per_model": per_model,
        "mean_gain_vs_best_single_strategy": mean(gains),
        "mean_gain_vs_zero_shot": mean(gains_vs_zero) if gains_vs_zero else None,
        "positive_gain_vs_best_single": {
            "positive": sum(1 for g in gains if g > 1e-9),
            "nonnegative": sum(1 for g in gains if g > -1e-9),
            "total": len(gains),
        },
        "positive_gain_vs_zero_shot": {
            "positive": sum(1 for g in gains_vs_zero if g > 1e-9),
            "total": len(gains_vs_zero),
        },
    }


def five_fold_by_instruction(runs: Mapping[str, Any]) -> dict[str, Any]:
    cache = _model_strategy_level_means(runs)
    ids_by_level = all_instruction_ids_by_level()
    # Deterministic 5 folds: reuse the level-stratified splitter with 5 seeds to
    # emulate folds (each seed yields a distinct held-out B half is 50/50; for a
    # cleaner 5-fold we partition ids into 5 contiguous stratified chunks).
    import random

    rng = random.Random(CV_SEED)
    folds: list[dict[int, set[str]]] = [dict() for _ in range(5)]
    for level in range(6):
        ids = sorted(map(str, ids_by_level[level]))
        rng.shuffle(ids)
        for index, iid in enumerate(ids):
            folds[index % 5].setdefault(level, set())
            folds[index % 5][level].add(iid)
    # Ensure every fold has every level populated.
    for fold in folds:
        for level in range(6):
            fold.setdefault(level, set())

    fold_reports: list[dict[str, Any]] = []
    all_gains: list[float] = []
    for k in range(5):
        held_ids = folds[k]
        train_ids: dict[int, set[str]] = {
            level: set().union(*(folds[j][level] for j in range(5) if j != k))
            for level in range(6)
        }
        learned = _learn_selection(
            BASELINE_MODELS, cache, selected_ids=train_ids, runs=runs
        )
        per_model_gain: dict[str, float] = {}
        for model in BASELINE_MODELS:
            tier = MODEL_TIER[model]
            available = runs[model]
            # CV-selected total on held-out fold
            cv_means: list[float] = []
            for level in range(6):
                chosen = learned[(level, tier)]
                if chosen not in available:
                    chosen = sorted(available)[0]
                cv_means.append(quality_level_means(available[chosen], held_ids)[level])
            cv_total = weighted_total(cv_means, LEVEL_WEIGHTS)
            # Best single strategy evaluated on the SAME held-out fold.
            single_totals = {
                strategy: weighted_total(
                    quality_level_means(data, held_ids), LEVEL_WEIGHTS
                )
                for strategy, data in available.items()
            }
            best_single = max(single_totals.values())
            per_model_gain[model] = cv_total - best_single
        fold_mean = mean(per_model_gain.values())
        all_gains.extend(per_model_gain.values())
        fold_reports.append(
            {
                "fold": k,
                "counts_by_level": {
                    str(level): len(held_ids[level]) for level in range(6)
                },
                "learned_strategy_per_tier_level": {
                    f"{tier}|L{level}": learned[(level, tier)]
                    for tier in ("T1", "T2", "T3")
                    for level in range(6)
                },
                "per_model_gain_vs_best_single": per_model_gain,
                "mean_gain_vs_best_single": fold_mean,
            }
        )
    return {
        "seed": CV_SEED,
        "n_folds": 5,
        "folds": fold_reports,
        "overall_mean_gain_vs_best_single": mean(all_gains),
        "positive_share": {
            "positive": sum(1 for g in all_gains if g > 1e-9),
            "total": len(all_gains),
        },
    }


def main() -> None:
    runs = baseline_runs()
    baselines = best_baseline_runs(LEVEL_WEIGHTS)
    lomo = leave_one_model_out(runs, baselines)
    five_fold = five_fold_by_instruction(runs)
    write_json(
        "OFF4_caap_selection_cv.json",
        {
            "analysis": "OFF-4 cross-validation of CAAP-style per-(level,tier) strategy selection",
            "zero_api_zero_model_calls": True,
            "tiering": {"scheme": "paper coarse Q-tiers", "groups": Q_TIERS},
            "level_weights": list(LEVEL_WEIGHTS),
            "method_notes": {
                "leave_one_model_out": "Best strategy per (level, tier) learned from the other 11 models "
                "(tier-mates only), applied to the held-out model; gain vs the held-out model's own best "
                "single fixed strategy. The held-out model never informs its selection.",
                "five_fold_by_instruction": "Best strategy per (level, tier) learned on 4/5 of instructions "
                "(all models pooled), applied to the held-out fifth; per-model gain vs best single strategy "
                "measured on the same held-out instructions.",
                "caveat": "Sonnet-4 has only zero-shot, so its held-out selection falls back to zero-shot and "
                "its gain vs best-single-strategy is ~0 by construction.",
            },
            "leave_one_model_out": lomo,
            "five_fold_by_instruction": five_fold,
        },
    )


if __name__ == "__main__":
    main()
