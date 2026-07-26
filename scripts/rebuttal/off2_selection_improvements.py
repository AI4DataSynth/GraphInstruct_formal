"""OFF-2: selection-based improvements across all twelve baseline models.

Three fully offline, zero-API selection strategies, each reported as a
quality-only total and a gain over the model's best single-strategy baseline:

1. Oracle (per-instruction) -- for every instruction take the maximum
   quality-only ``level_score`` across the available baseline strategies.
   Recomputed from ``dimension_scores`` (identical to ``best_baseline_runs``),
   so it is exact.  A weaker per-level oracle (the ``build_oracle_baseline``
   convention) is reported alongside for the file consistency check.

2. SC-best-of-5 -- re-parse the five samples of the model's best-strategy
   ``.jsonl``, recompute per-sample ``satisfaction_rate`` exactly as
   ``runners._score_sample`` does, and measure the best-of-five uplift
   (max minus mean satisfaction).  Because the stored D4 metric is *not*
   raw satisfaction (it adds no-contradiction and L4/L5 terms; empirical
   per-level MAE ~0.08-0.19), we do NOT reconstruct an absolute D4.  Instead
   the uplift is applied through the renormalized D4 weight on top of the
   faithful stored baseline total.  Baseline and SC therefore use one
   internally consistent satisfaction metric, so the *delta* is meaningful
   even though the absolute satisfaction does not equal the stored D4.

3. CAAP-selection-only -- use ``caap.select_strategy`` to pick a strategy per
   (instruction, model) and take that strategy's existing per-instruction
   quality score.  This is the NO-AUGMENTATION lower bound: real CAAP also
   injects L2 checklists / L3 formulas / L4 domain priors, which require new
   generation and are excluded here.

Consistency sanity-checks (deepseekaiDeepSeekV3 / gpt4omini / qwen3535ba3b)
compare each offline recompute against the already-run
``-oracle`` / ``-sc`` / ``-caap`` quality files and report the delta.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.data_loader import Instruction, load_all_levels
from graphinstruct.improvements.caap import select_strategy
from graphinstruct.parser.code_style import parse
from graphinstruct.scoring import DIMENSION_WEIGHTS, LEVEL_WEIGHTS
from graphinstruct.validators import satisfaction_rate
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    MODEL_NAMES,
    RESULTS_DIR,
    baseline_runs,
    best_baseline_runs,
    load_quality_file,
    mean,
    quality_level_means,
    quality_level_score,
    quality_path,
    weighted_total,
    write_json,
)

# Baseline slug -> a model name the CAAP registry resolves to the correct tier.
# All twelve resolve to the paper's coarse tier (T1/T2/T3); the T2 split into
# T2-GPT / T2-open follows the registry.  Sonnet-4 has only zero-shot data, so
# any non-zero-shot CAAP pick falls back to zero-shot and its selection total
# equals its baseline regardless of the (immaterial) T2-GPT assignment.
SLUG_TO_CAAP_NAME = {
    "claudesonnet46": "claude-sonnet-4-6",
    "qwen35397ba17b": "qwen3.5-397b",
    "qwen35122ba10b": "qwen3.5-122b",
    "qwen3535ba3b": "qwen3.5-35b-a3b",
    "gpt41": "gpt41",
    "gpt4o": "gpt4o",
    "deepseekaiDeepSeekV3": "deepseek-v3",
    "metallamaLlama3370BInstructTurbo": "meta-llama/llama-3.3-70b-instruct-turbo",
    "claudesonnet420250514": "claude-sonnet-4",  # unknown key -> registry default T2-GPT
    "gpt35turbo": "gpt35turbo",
    "gpt4omini": "gpt4omini",
    "metallamaMetaLlama318BInstruct": "llama-3-8b",
}

CONSISTENCY_MODELS = ("deepseekaiDeepSeekV3", "gpt4omini", "qwen3535ba3b")


def d4_quality_weight(level: int) -> float:
    """Renormalized D4 weight in the quality-only (D5=0) scheme for a level."""
    weights = dict(DIMENSION_WEIGHTS[level])
    weights["D5"] = 0.0
    denominator = sum(weights.values())
    return weights["D4"] / denominator


def _inst_qls_map(data: Mapping[str, Any]) -> dict[tuple[int, str], float]:
    """Map (level, instruction_id) -> quality-only level_score from a run."""
    result: dict[tuple[int, str], float] = {}
    for record in data["per_instruction"]:
        level = int(record["level"])
        iid = str(record["instruction_id"])
        result[(level, iid)] = quality_level_score(
            level, record.get("dimension_scores", {})
        )
    return result


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["instruction_id"])] = record
    return records


def oracle_per_instruction(strategies: Mapping[str, Any]) -> dict[str, Any]:
    """Per-instruction best-of-strategies quality-only total (exact)."""
    maps = {strategy: _inst_qls_map(data) for strategy, data in strategies.items()}
    keys: set[tuple[int, str]] = set()
    for values in maps.values():
        keys |= set(values)
    per_level: dict[int, list[float]] = defaultdict(list)
    choice: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for level, iid in keys:
        candidates = {
            strategy: values[(level, iid)]
            for strategy, values in maps.items()
            if (level, iid) in values
        }
        best_value = max(candidates.values())
        best_strategy = sorted(s for s in candidates if candidates[s] == best_value)[0]
        per_level[level].append(best_value)
        choice[level][best_strategy] += 1
    level_means = [mean(per_level[level]) for level in range(6)]
    total = weighted_total(level_means, LEVEL_WEIGHTS)
    return {
        "quality_only_total": total,
        "level_means": level_means,
        "strategy_choice_counts": {
            str(level): dict(choice[level]) for level in range(6)
        },
    }


def oracle_per_level(strategies: Mapping[str, Any]) -> dict[str, Any]:
    """Per-level best-strategy oracle (the build_oracle_baseline convention)."""
    strategy_level_means = {
        strategy: quality_level_means(data) for strategy, data in strategies.items()
    }
    level_means: list[float] = []
    level_strategy: dict[str, str] = {}
    for level in range(6):
        best_value = max(means[level] for means in strategy_level_means.values())
        best_strategy = sorted(
            s for s, means in strategy_level_means.items() if means[level] == best_value
        )[0]
        level_means.append(best_value)
        level_strategy[str(level)] = best_strategy
    total = weighted_total(level_means, LEVEL_WEIGHTS)
    return {
        "quality_only_total": total,
        "level_means": level_means,
        "level_strategy": level_strategy,
    }


def sc_best_of_5(
    model: str, base_strategy: str, insts: dict[int, list[Instruction]]
) -> dict[str, Any] | None:
    """Best-of-5 satisfaction uplift over a base-strategy jsonl.

    Returns per-level D4-satisfaction uplift (max-of-5 minus mean-of-5) and the
    quality-only total uplift obtained by folding that uplift through the
    renormalized D4 weight.  ``None`` if the jsonl is unavailable.
    """
    jsonl_path = RESULTS_DIR / f"{model}-{base_strategy}.jsonl"
    if not jsonl_path.exists():
        return None
    records = _load_jsonl(jsonl_path)
    inst_by_id = {inst.id: inst for level in insts for inst in insts[level]}

    per_level_uplift: dict[int, list[float]] = defaultdict(list)
    per_level_mean_sat: dict[int, list[float]] = defaultdict(list)
    per_level_bo5_sat: dict[int, list[float]] = defaultdict(list)

    for iid, record in records.items():
        inst = inst_by_id.get(iid)
        if inst is None:
            continue
        level = inst.level
        constraints = list(inst.explicit_constraints) + list(inst.implicit_constraints)
        sats: list[float] = []
        for sample in record.get("samples", []):
            serialized = sample.get("graph_serialized")
            graph = None
            if serialized:
                try:
                    graph = parse(serialized).graph
                except Exception:
                    graph = None
            if graph is None:
                sats.append(0.0)
            elif not constraints:
                sats.append(1.0)
            else:
                # Defensive guard mirrors runners._score_sample: some parsed
                # graphs are degenerate (e.g. null graph) and crash validators.
                try:
                    sats.append(satisfaction_rate(graph, constraints))
                except Exception:
                    sats.append(0.0)
        if not sats:
            continue
        mean_sat = sum(sats) / len(sats)
        max_sat = max(sats)
        per_level_uplift[level].append(max_sat - mean_sat)
        per_level_mean_sat[level].append(mean_sat)
        per_level_bo5_sat[level].append(max_sat)

    level_uplift: dict[str, float] = {}
    total_uplift = 0.0
    for level in range(6):
        uplift = mean(per_level_uplift[level]) or 0.0
        level_uplift[str(level)] = uplift
        total_uplift += LEVEL_WEIGHTS[level] * d4_quality_weight(level) * uplift
    return {
        "base_strategy": base_strategy,
        "d4_satisfaction_uplift_by_level": level_uplift,
        "mean_sat_by_level": {
            str(l): (mean(per_level_mean_sat[l]) or 0.0) for l in range(6)
        },
        "bo5_sat_by_level": {
            str(l): (mean(per_level_bo5_sat[l]) or 0.0) for l in range(6)
        },
        "quality_only_total_uplift": total_uplift,
    }


def caap_selection(
    model: str,
    strategies: Mapping[str, Any],
    insts: dict[int, list[Instruction]],
    fallback: str,
) -> dict[str, Any]:
    """CAAP-picked strategy per instruction, scored on existing baselines."""
    caap_name = SLUG_TO_CAAP_NAME[model]
    available = set(strategies)
    maps = {strategy: _inst_qls_map(data) for strategy, data in strategies.items()}
    fallback_strategy = fallback if fallback in available else sorted(available)[0]

    per_level: dict[int, list[float]] = defaultdict(list)
    choice: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fallback_count = 0
    for level in range(6):
        for inst in insts[level]:
            chosen = select_strategy(inst, caap_name).strategy
            if chosen not in available:
                fallback_count += 1
                chosen = fallback_strategy
            score = maps[chosen].get((level, inst.id))
            if score is None:  # defensive: chosen strategy lacks this id
                score = maps[fallback_strategy].get((level, inst.id))
                chosen = fallback_strategy
            per_level[level].append(score)
            choice[level][chosen] += 1
    level_means = [mean(per_level[level]) for level in range(6)]
    total = weighted_total(level_means, LEVEL_WEIGHTS)
    return {
        "quality_only_total": total,
        "level_means": level_means,
        "fallback_count": fallback_count,
        "strategy_choice_counts": {
            str(level): dict(choice[level]) for level in range(6)
        },
    }


def existing_file_total(model: str, method: str) -> float | None:
    path = quality_path(model, method)
    if not path.exists():
        return None
    return weighted_total(
        quality_level_means(load_quality_file(str(path))), LEVEL_WEIGHTS
    )


def main() -> None:
    insts = load_all_levels([0, 1, 2, 3, 4, 5])
    runs = baseline_runs()
    baselines = best_baseline_runs(LEVEL_WEIGHTS)

    per_model: dict[str, Any] = {}
    skipped_models: list[str] = []
    gains = {
        "oracle_per_instruction": [],
        "oracle_per_level": [],
        "sc_best_of_5": [],
        "caap_selection": [],
    }

    for model in BASELINE_MODELS:
        # ext16 new models (GLM-4.6 / Gemma-3-27B / Phi-4 / Mistral-Small-24B) were added
        # for baseline robustness only; they have no CAAP/SC improvement runs. Selection-
        # improvement analysis is defined solely on the improvement-study subset -> skip them.
        if model not in SLUG_TO_CAAP_NAME:
            skipped_models.append(model)
            continue
        strategies = runs[model]
        baseline_total = baselines[model]["total"]
        baseline_strategy = baselines[model]["strategy"]

        oracle_pi = oracle_per_instruction(strategies)
        oracle_pl = oracle_per_level(strategies)
        sc = sc_best_of_5(model, baseline_strategy, insts)
        caap = caap_selection(model, strategies, insts, fallback="zero-shot")

        oracle_pi_gain = oracle_pi["quality_only_total"] - baseline_total
        oracle_pl_gain = oracle_pl["quality_only_total"] - baseline_total
        sc_total = baseline_total + (sc["quality_only_total_uplift"] if sc else 0.0)
        sc_gain = sc["quality_only_total_uplift"] if sc else None
        caap_gain = caap["quality_only_total"] - baseline_total

        gains["oracle_per_instruction"].append(oracle_pi_gain)
        gains["oracle_per_level"].append(oracle_pl_gain)
        if sc_gain is not None:
            gains["sc_best_of_5"].append(sc_gain)
        gains["caap_selection"].append(caap_gain)

        per_model[model] = {
            "display_name": MODEL_NAMES[model],
            "baseline": {
                "strategy": baseline_strategy,
                "quality_only_total": baseline_total,
            },
            "oracle_per_instruction": {**oracle_pi, "gain_vs_baseline": oracle_pi_gain},
            "oracle_per_level": {**oracle_pl, "gain_vs_baseline": oracle_pl_gain},
            "sc_best_of_5": (
                {
                    **sc,
                    "quality_only_total": sc_total,
                    "gain_vs_baseline": sc_gain,
                }
                if sc
                else {"available": False, "reason": "best-strategy jsonl missing"}
            ),
            "caap_selection": {**caap, "gain_vs_baseline": caap_gain},
        }

    means_across_12 = {key: mean(values) for key, values in gains.items()}
    positive_counts = {
        key: {"positive": sum(1 for v in values if v > 1e-9), "total": len(values)}
        for key, values in gains.items()
    }

    # Consistency sanity checks against already-run improvement files.
    consistency: dict[str, Any] = {}
    for model in CONSISTENCY_MODELS:
        strategies = runs[model]
        zero_sc = sc_best_of_5(model, "zero-shot", insts)
        zero_shot_total = (
            weighted_total(quality_level_means(strategies["zero-shot"]), LEVEL_WEIGHTS)
            if "zero-shot" in strategies
            else None
        )
        my_sc_zero = (
            zero_shot_total + zero_sc["quality_only_total_uplift"]
            if (zero_sc and zero_shot_total is not None)
            else None
        )
        existing_sc = existing_file_total(model, "sc")
        existing_oracle = existing_file_total(model, "oracle")
        existing_caap = existing_file_total(model, "caap")

        my_oracle_pl = per_model[model]["oracle_per_level"]["quality_only_total"]
        my_oracle_pi = per_model[model]["oracle_per_instruction"]["quality_only_total"]
        my_caap = per_model[model]["caap_selection"]["quality_only_total"]

        consistency[model] = {
            "oracle": {
                "offline_per_level": my_oracle_pl,
                "offline_per_instruction": my_oracle_pi,
                "existing_file": existing_oracle,
                "delta_per_level_vs_file": (
                    my_oracle_pl - existing_oracle
                    if existing_oracle is not None
                    else None
                ),
                "note": "existing -oracle is a per-level cherry-pick re-evaluated end-to-end; "
                "the offline per-level oracle should closely match; per-instruction is a "
                "strictly stronger upper bound.",
            },
            "sc_best_of_5": {
                "offline_on_zero_shot_base": my_sc_zero,
                "offline_zero_shot_uplift": (
                    zero_sc["quality_only_total_uplift"] if zero_sc else None
                ),
                "existing_file": existing_sc,
                "delta_vs_file": (
                    my_sc_zero - existing_sc
                    if (my_sc_zero is not None and existing_sc is not None)
                    else None
                ),
                "note": "existing -sc uses a zero-shot base re-evaluated across all "
                "dimensions; the offline uplift touches only the D4/satisfaction term, so "
                "some divergence is expected.",
            },
            "caap_selection": {
                "offline_selection_only": my_caap,
                "existing_file": existing_caap,
                "delta_vs_file": (
                    my_caap - existing_caap if existing_caap is not None else None
                ),
                "note": "NOT directly comparable: offline is pure strategy selection over the "
                "four existing baselines (no augmentation); existing -caap adds L2/L3/L4 prompt "
                "augmentation on a zero-shot base with fresh generation. Offline is a lower bound.",
            },
        }

    write_json(
        "OFF2_selection_improvements_all12.json",
        {
            "analysis": "OFF-2 selection-based improvements (Oracle / SC-best-of-5 / CAAP-selection) across twelve baseline models",
            "zero_api_zero_model_calls": True,
            "baseline_definition": "Each model's highest quality-only (D5=0, D1-D4 renormalized) single-strategy total.",
            "level_weights": list(LEVEL_WEIGHTS),
            "d4_quality_weight_by_level": {
                str(l): d4_quality_weight(l) for l in range(6)
            },
            "caap_name_mapping": SLUG_TO_CAAP_NAME,
            "method_notes": {
                "oracle_per_instruction": "Exact: per-instruction max quality-only level_score across available strategies.",
                "oracle_per_level": "build_oracle_baseline convention: per-level best strategy.",
                "sc_best_of_5": "Best-strategy jsonl; per-sample satisfaction_rate (== runners._score_sample); "
                "uplift = max-of-5 minus mean-of-5, folded through renormalized D4 weight and added to the "
                "faithful stored baseline total. Absolute satisfaction != stored D4 (per-level MAE ~0.08-0.19), "
                "so only the internally-consistent delta is used.",
                "caap_selection": "NO-AUGMENTATION lower bound: caap.select_strategy picks a strategy; that "
                "strategy's existing per-instruction score is used. Real CAAP additionally injects "
                "L2 checklist / L3 formulas / L4 domain priors requiring new generation (excluded).",
            },
            "per_model": per_model,
            "means_across_12": means_across_12,
            "n_models_analyzed": len(per_model),
            "skipped_models": [
                {"slug": m, "display_name": MODEL_NAMES.get(m, m)}
                for m in skipped_models
            ],
            "skipped_reason": (
                "ext16 baseline-robustness models without CAAP/SC improvement runs; "
                "selection-improvement analysis is defined only on the improvement-study subset."
            ),
            "positive_counts": positive_counts,
            "consistency_checks": consistency,
        },
    )


if __name__ == "__main__":
    main()
