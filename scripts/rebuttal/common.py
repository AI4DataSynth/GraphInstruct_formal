"""Shared, read-only calculations for the rebuttal analysis scripts.

This module deliberately reads only ``results/*.quality.json``.  It contains no
client imports, subprocesses, or model/API execution paths.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import DIMENSION_WEIGHTS, LEVEL_WEIGHTS

RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "rebuttal_analyses" / "results"
BASELINE_STRATEGIES = ("zero-shot", "few-shot", "zero-cot", "few-cot")
BASELINE_MODELS = (
    "claudesonnet46",
    "qwen35397ba17b",
    "qwen35122ba10b",
    "qwen3535ba3b",
    "gpt41",
    "gpt4o",
    "deepseekaiDeepSeekV3",
    "metallamaLlama3370BInstructTurbo",
    "claudesonnet420250514",
    "gpt35turbo",
    "gpt4omini",
    "metallamaMetaLlama318BInstruct",
)
MODEL_NAMES = {
    "claudesonnet46": "Sonnet-4.6",
    "qwen35397ba17b": "Qwen3.5-397B-A17B",
    "qwen35122ba10b": "Qwen3.5-122B-A10B",
    "qwen3535ba3b": "Qwen3.5-35B-A3B",
    "gpt41": "GPT-4.1",
    "gpt4o": "GPT-4o",
    "deepseekaiDeepSeekV3": "DeepSeek-V3",
    "metallamaLlama3370BInstructTurbo": "Llama-3.3-70B",
    "claudesonnet420250514": "Sonnet-4",
    "gpt35turbo": "GPT-3.5-turbo",
    "gpt4omini": "GPT-4o-mini",
    "metallamaMetaLlama318BInstruct": "Llama-3.1-8B",
}
Q_TIERS = {
    "T1": ("claudesonnet46", "qwen35397ba17b", "qwen35122ba10b"),
    "T2": (
        "qwen3535ba3b",
        "gpt41",
        "gpt4o",
        "deepseekaiDeepSeekV3",
        "metallamaLlama3370BInstructTurbo",
        "claudesonnet420250514",
    ),
    "T3": ("gpt35turbo", "gpt4omini", "metallamaMetaLlama318BInstruct"),
}
ACTIVE_PARAMS_B = {
    "qwen35397ba17b": 17,
    "qwen35122ba10b": 10,
    "qwen3535ba3b": 3,
    "deepseekaiDeepSeekV3": 37,
    "metallamaLlama3370BInstructTurbo": 70,
    "metallamaMetaLlama318BInstruct": 8,
}

# --- ext16 model-set expansion (opt-in via env REBUTTAL_MODELSET=ext16) ---
# Rebuttal 增量:纳入 4 个 4-策略齐全的新模型(GLM-4.6/Gemma-3-27B/Phi-4/Mistral-Small-24B),
# 重算稳健性分析。orig12 模式(默认)完全不受影响;ext16 覆盖模型集/参数/显示名/输出目录。
# Q_TIERS 的 ext16 版在文件末尾按质量分自动分层(需 best_baseline_runs,故延后到函数定义之后)。
import os as _os

_EXT16_ACTIVE = _os.environ.get("REBUTTAL_MODELSET", "").strip().lower() == "ext16"
if _EXT16_ACTIVE:
    _EXT16_NEW = {
        # slug: (display_name, active_params_billions)
        "zaiorgGLM46": ("GLM-4.6", 32),
        "googlegemma327bit": ("Gemma-3-27B", 27),
        "microsoftphi4": ("Phi-4", 14),
        "mistralaiMistralSmall24BInstruct2501": ("Mistral-Small-24B", 24),
    }
    BASELINE_MODELS = BASELINE_MODELS + tuple(_EXT16_NEW)
    MODEL_NAMES = {
        **MODEL_NAMES,
        **{slug: meta[0] for slug, meta in _EXT16_NEW.items()},
    }
    ACTIVE_PARAMS_B = {
        **ACTIVE_PARAMS_B,
        **{slug: meta[1] for slug, meta in _EXT16_NEW.items()},
    }
    ARTIFACT_DIR = ROOT / "rebuttal_analyses" / "results" / "ext16"

LEVEL_COUNTS = (100, 200, 200, 150, 100, 50)
AVAILABLE_DIMENSIONS = {
    0: ("D1", "D4"),
    1: ("D1", "D4"),
    2: ("D1", "D4"),
    3: ("D1", "D3", "D4"),
    4: ("D1", "D2", "D3", "D4"),
    5: ("D1", "D3", "D4"),
}
QUALITY_DIMENSIONS = ("D1", "D2", "D3", "D4")


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or a documented scoring default of zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def normalize(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return [float(weight) / total for weight in weights]


def quality_level_score(level: int, dimension_scores: Mapping[str, Any]) -> float:
    """Apply the repository's D1--D4 quality-only scoring convention.

    The weights are imported from ``graphinstruct.scoring``.  D5 is explicitly
    set to zero and the remaining dimensional weights are renormalized.
    """
    weights = dict(DIMENSION_WEIGHTS[level])
    weights["D5"] = 0.0
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError(f"no quality-only dimension weights at level {level}")
    return sum(
        weights[dimension] / denominator * safe_float(dimension_scores.get(dimension))
        for dimension in QUALITY_DIMENSIONS
    )


@lru_cache(maxsize=None)
def load_quality_file(path_string: str) -> dict[str, Any]:
    """Load one local quality artifact; deliberately no network or model calls."""
    path = Path(path_string)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("per_instruction"), list):
        raise ValueError(f"{path} has no per_instruction list")
    return data


def quality_path(model: str, strategy: str) -> Path:
    direct = RESULTS_DIR / f"{model}-{strategy}.quality.json"
    if direct.exists():
        return direct
    # Released-artifact layout keeps quality files under results/quality/.
    nested = RESULTS_DIR / "quality" / f"{model}-{strategy}.quality.json"
    return nested if nested.exists() else direct


@lru_cache(maxsize=1)
def baseline_runs() -> dict[str, dict[str, dict[str, Any]]]:
    """Load all available baseline runs for the stipulated twelve models."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    missing_models: list[str] = []
    for model in BASELINE_MODELS:
        strategies: dict[str, dict[str, Any]] = {}
        for strategy in BASELINE_STRATEGIES:
            path = quality_path(model, strategy)
            if path.exists():
                strategies[strategy] = load_quality_file(str(path))
        if not strategies:
            missing_models.append(model)
        result[model] = strategies
    if missing_models:
        raise FileNotFoundError(f"missing all baseline runs for: {missing_models}")
    return result


def available_baseline_strategies() -> dict[str, list[str]]:
    return {model: sorted(strategies) for model, strategies in baseline_runs().items()}


def instruction_records_by_level(
    data: Mapping[str, Any], selected_ids: Mapping[int, set[str]] | None = None
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {level: [] for level in range(6)}
    for record in data["per_instruction"]:
        level = int(record["level"])
        instruction_id = str(record["instruction_id"])
        if selected_ids is not None and instruction_id not in selected_ids[level]:
            continue
        grouped[level].append(record)
    return grouped


def quality_level_means(
    data: Mapping[str, Any], selected_ids: Mapping[int, set[str]] | None = None
) -> list[float]:
    grouped = instruction_records_by_level(data, selected_ids)
    values: list[float] = []
    for level in range(6):
        scores = [
            quality_level_score(level, record.get("dimension_scores", {}))
            for record in grouped[level]
        ]
        if not scores:
            raise ValueError(f"no selected instructions at level {level}")
        values.append(sum(scores) / len(scores))
    return values


def d4_level_means(
    data: Mapping[str, Any], selected_ids: Mapping[int, set[str]] | None = None
) -> list[float]:
    grouped = instruction_records_by_level(data, selected_ids)
    means: list[float] = []
    for level in range(6):
        values = [
            safe_float(
                record.get(
                    "d4_instruction_score", record.get("dimension_scores", {}).get("D4")
                )
            )
            for record in grouped[level]
        ]
        if not values:
            raise ValueError(f"no selected instructions at level {level}")
        means.append(sum(values) / len(values))
    return means


def dimension_level_means(
    data: Mapping[str, Any], selected_ids: Mapping[int, set[str]] | None = None
) -> dict[int, dict[str, float]]:
    grouped = instruction_records_by_level(data, selected_ids)
    result: dict[int, dict[str, float]] = {}
    for level, records in grouped.items():
        if not records:
            raise ValueError(f"no selected instructions at level {level}")
        result[level] = {
            dimension: sum(
                safe_float(record.get("dimension_scores", {}).get(dimension))
                for record in records
            )
            / len(records)
            for dimension in ("D1", "D2", "D3", "D4", "D5")
        }
    return result


def weighted_total(
    level_means: Sequence[float], level_weights: Sequence[float]
) -> float:
    if len(level_means) != 6 or len(level_weights) != 6:
        raise ValueError("six level means and six level weights are required")
    return sum(score * weight for score, weight in zip(level_means, level_weights))


def best_baseline_runs(
    level_weights: Sequence[float] = LEVEL_WEIGHTS,
    selected_ids: Mapping[int, set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select each model's highest quality-only baseline under this exact subset."""
    chosen: dict[str, dict[str, Any]] = {}
    for model, strategies in baseline_runs().items():
        candidates: list[dict[str, Any]] = []
        for strategy, data in strategies.items():
            level_means = quality_level_means(data, selected_ids)
            candidates.append(
                {
                    "strategy": strategy,
                    "data": data,
                    "level_means": level_means,
                    "total": weighted_total(level_means, level_weights),
                }
            )
        candidates.sort(key=lambda value: (-value["total"], value["strategy"]))
        chosen[model] = candidates[0]
    return chosen


def rank_desc(values: Mapping[str, float]) -> list[str]:
    """Stable descending rank order; lexical key order resolves exact ties."""
    return sorted(values, key=lambda key: (-values[key], key))


def rank_positions(values: Mapping[str, float]) -> dict[str, int]:
    return {key: index + 1 for index, key in enumerate(rank_desc(values))}


def average_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Descending average ranks, suitable for tied Spearman correlations."""
    sorted_items = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(sorted_items):
        stop = start + 1
        while (
            stop < len(sorted_items) and sorted_items[stop][1] == sorted_items[start][1]
        ):
            stop += 1
        average_rank = (start + 1 + stop) / 2
        for key, _ in sorted_items[start:stop]:
            ranks[key] = average_rank
        start = stop
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0 or y_scale == 0:
        return None
    return numerator / (x_scale * y_scale)


def spearman_from_lists(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        return None
    x_ranks = average_ranks({str(index): value for index, value in enumerate(xs)})
    y_ranks = average_ranks({str(index): value for index, value in enumerate(ys)})
    return pearson(
        [x_ranks[str(index)] for index in range(len(xs))],
        [y_ranks[str(index)] for index in range(len(xs))],
    )


def spearman(
    values_a: Mapping[str, float], values_b: Mapping[str, float]
) -> float | None:
    keys = sorted(set(values_a) & set(values_b))
    if len(keys) < 2:
        return None
    return spearman_from_lists(
        [values_a[key] for key in keys], [values_b[key] for key in keys]
    )


def kendall_tau_b(
    values_a: Mapping[str, float], values_b: Mapping[str, float]
) -> float | None:
    keys = sorted(set(values_a) & set(values_b))
    if len(keys) < 2:
        return None
    concordant = discordant = ties_a = ties_b = 0
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            dx = values_a[key_a] - values_a[key_b]
            dy = values_b[key_a] - values_b[key_b]
            if dx == 0 and dy == 0:
                ties_a += 1
                ties_b += 1
            elif dx == 0:
                ties_a += 1
            elif dy == 0:
                ties_b += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    total_pairs = len(keys) * (len(keys) - 1) / 2
    denominator = math.sqrt((total_pairs - ties_a) * (total_pairs - ties_b))
    return (concordant - discordant) / denominator if denominator else None


def top_overlap(
    order_a: Sequence[str], order_b: Sequence[str], requested_k: int
) -> dict[str, int]:
    effective_k = min(requested_k, len(order_a), len(order_b))
    return {
        "requested_k": requested_k,
        "effective_k": effective_k,
        "retained": len(set(order_a[:effective_k]) & set(order_b[:effective_k])),
    }


def all_instruction_ids_by_level() -> dict[int, list[str]]:
    first_model = BASELINE_MODELS[0]
    first_strategy = sorted(baseline_runs()[first_model])[0]
    grouped = instruction_records_by_level(baseline_runs()[first_model][first_strategy])
    ids = {
        level: sorted(str(record["instruction_id"]) for record in records)
        for level, records in grouped.items()
    }
    if tuple(len(ids[level]) for level in range(6)) != LEVEL_COUNTS:
        raise ValueError(
            f"unexpected instruction counts: {[len(ids[level]) for level in range(6)]}"
        )
    return ids


def split_instruction_ids(
    ids_by_level: Mapping[int, Sequence[str]], seed: int
) -> dict[str, dict[int, set[str]]]:
    """Make a deterministic level-stratified half split (400/400 for this data)."""
    rng = random.Random(seed)
    a_ids: dict[int, set[str]] = {}
    b_ids: dict[int, set[str]] = {}
    for level in sorted(ids_by_level):
        ids = sorted(map(str, ids_by_level[level]))
        rng.shuffle(ids)
        cutoff = len(ids) // 2
        a_ids[level] = set(ids[:cutoff])
        b_ids[level] = set(ids[cutoff:])
    return {"A": a_ids, "B": b_ids}


def tier_gap_from_level_vectors(
    vectors: Mapping[str, Sequence[float]],
    high_models: Sequence[str],
    low_models: Sequence[str],
) -> list[float]:
    if not high_models or not low_models:
        raise ValueError("both high and low tier must be nonempty")
    return [
        sum(vectors[model][level] for model in high_models) / len(high_models)
        - sum(vectors[model][level] for model in low_models) / len(low_models)
        for level in range(6)
    ]


def peak_summary(gaps: Sequence[float]) -> dict[str, Any]:
    max_gap = max(gaps)
    peak_levels = [
        index
        for index, gap in enumerate(gaps)
        if math.isclose(gap, max_gap, abs_tol=1e-12)
    ]
    return {
        "peak_levels": peak_levels,
        "l2_is_peak": 2 in peak_levels,
        "l2_is_unique_peak": peak_levels == [2],
    }


def split_data_tier_results(seeds: Sequence[int]) -> dict[str, Any]:
    """Strict split-data control: select strategy/tier on A, evaluate only B."""
    ids_by_level = all_instruction_ids_by_level()
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        split = split_instruction_ids(ids_by_level, seed)
        selected = best_baseline_runs(LEVEL_WEIGHTS, split["A"])
        selection_totals = {model: entry["total"] for model, entry in selected.items()}
        ordered = rank_desc(selection_totals)
        high, middle, low = ordered[:3], ordered[3:9], ordered[9:]
        b_vectors = {
            model: d4_level_means(entry["data"], split["B"])
            for model, entry in selected.items()
        }
        gaps = tier_gap_from_level_vectors(b_vectors, high, low)
        per_seed.append(
            {
                "seed": seed,
                "split": {
                    "A_ids": sorted(
                        identifier
                        for values in split["A"].values()
                        for identifier in values
                    ),
                    "B_ids": sorted(
                        identifier
                        for values in split["B"].values()
                        for identifier in values
                    ),
                    "counts_by_level": {
                        str(level): {
                            "A": len(split["A"][level]),
                            "B": len(split["B"][level]),
                        }
                        for level in range(6)
                    },
                },
                "selection_on_A": {
                    "quality_only_totals": selection_totals,
                    "selected_strategies": {
                        model: selected[model]["strategy"] for model in BASELINE_MODELS
                    },
                    "tiers": {"T1": high, "T2": middle, "T3": low},
                },
                "analysis_on_B": {"d4_tier_gaps": gaps, **peak_summary(gaps)},
            }
        )
    mean_gaps = [
        mean(result["analysis_on_B"]["d4_tier_gaps"][level] for result in per_seed)
        for level in range(6)
    ]
    std_gaps = [
        statistics.stdev(
            [result["analysis_on_B"]["d4_tier_gaps"][level] for result in per_seed]
        )
        if len(per_seed) > 1
        else 0.0
        for level in range(6)
    ]
    return {
        "seeds": list(seeds),
        "split_method": "Level-stratified random 50/50 A/B; strategy and tier selected on A only, D4 gaps evaluated on B only.",
        "tier_sizes": {"T1": 3, "T2": 6, "T3": 3},
        "seed_results": per_seed,
        "mean_d4_tier_gaps": mean_gaps,
        "std_d4_tier_gaps": std_gaps,
        "l2_peak_frequency": sum(
            result["analysis_on_B"]["l2_is_peak"] for result in per_seed
        )
        / len(per_seed),
        **peak_summary(mean_gaps),
    }


def records_to_metric_map(
    data: Mapping[str, Any], level: int, metric: Callable[[Mapping[str, Any]], float]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in data["per_instruction"]:
        if int(record["level"]) == level:
            result[str(record["instruction_id"])] = metric(record)
    return result


def percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_tier_gap(
    model_metric_maps: Mapping[str, Mapping[str, float]],
    high_models: Sequence[str],
    low_models: Sequence[str],
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Bootstrap common instruction IDs, retaining cross-model pairing."""
    all_models = list(high_models) + list(low_models)
    ids = sorted(
        set.intersection(*(set(model_metric_maps[model]) for model in all_models))
    )
    if not ids:
        raise ValueError("tiers have no common instruction IDs")
    observed = sum(
        sum(model_metric_maps[model][identifier] for model in high_models)
        / len(high_models)
        for identifier in ids
    ) / len(ids) - sum(
        sum(model_metric_maps[model][identifier] for model in low_models)
        / len(low_models)
        for identifier in ids
    ) / len(ids)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        high_mean = sum(
            sum(model_metric_maps[model][identifier] for model in high_models)
            / len(high_models)
            for identifier in sampled
        ) / len(sampled)
        low_mean = sum(
            sum(model_metric_maps[model][identifier] for model in low_models)
            / len(low_models)
            for identifier in sampled
        ) / len(sampled)
        replicates.append(high_mean - low_mean)
    return {
        "observed_gap": observed,
        "ci_95_percentile": [percentile(replicates, 2.5), percentile(replicates, 97.5)],
        "n_instruction_ids": len(ids),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def all_quality_runs() -> list[tuple[str, dict[str, Any]]]:
    """All readable quality files in results/, for B1's all-run pooling only.

    Scans both ``results/`` and the released artifact's ``results/quality/``
    layout; duplicate stems prefer the ``results/`` copy.
    """
    seen: dict[str, Path] = {}
    for path in sorted((RESULTS_DIR / "quality").glob("*.quality.json")) + sorted(
        RESULTS_DIR.glob("*.quality.json")
    ):
        stem = path.stem.removesuffix(".quality")
        if stem.startswith("_"):  # smoke-test fixtures, not runs
            continue
        seen[stem] = path
    return [(stem, load_quality_file(str(path))) for stem, path in sorted(seen.items())]


def write_json(filename: str, payload: Mapping[str, Any]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_DIR / filename.lower()
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    print(destination.relative_to(ROOT))
    return destination


def write_markdown(filename: str, content: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_DIR / filename.lower()
    destination.write_text(content, encoding="utf-8", newline="\n")
    print(destination.relative_to(ROOT))
    return destination


def _compute_auto_tiers(models: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """ext16: 按 best-baseline quality-only total 降序自动分层(16 模型 → 4/8/4)。

    透明、可复现、数据变自适应,不手工硬编码。各脚本把 groups 字段写进 JSON 供审计。
    仅在 ext16 模式于文件末尾调用(此时 best_baseline_runs/rank_desc 已定义)。
    """
    selected = best_baseline_runs()
    totals = {model: selected[model]["total"] for model in models}
    ordered = rank_desc(totals)
    n = len(ordered)
    edge = max(1, round(n / 4))
    return {
        "T1": tuple(ordered[:edge]),
        "T2": tuple(ordered[edge : n - edge]),
        "T3": tuple(ordered[n - edge :]),
    }


if _EXT16_ACTIVE:
    # ext16 的 Q-tier 覆盖(orig12 保持顶部硬编码 3/6/3,不受影响)。
    Q_TIERS = _compute_auto_tiers(BASELINE_MODELS)
