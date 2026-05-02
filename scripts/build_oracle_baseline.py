"""Build E1b Oracle baseline by cherry-picking best strategy per level.

For each (model, level), select the strategy with the highest ``level_score``
from the existing ``{model}-{strategy}.quality.json`` files, then synthesise
an oracle JSONL by concatenating the per-instruction samples from each
best-strategy JSONL.

Outputs:
    results/{model_prefix}-oracle.jsonl
    results/{model_prefix}-oracle.json
    results/{model_prefix}-oracle.quality.json

The re-evaluation step reuses ``graphinstruct.evaluate.evaluate_benchmark``
so no custom evaluation code is needed.

Usage:
    python scripts/build_oracle_baseline.py --model-prefix gpt4omini
    python scripts/build_oracle_baseline.py --model-prefix deepseekaiDeepSeekV3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path (mirrors scripts/run_baseline.py convention)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Offline flags match eval_checkpoint.py to avoid HF hub lookups during D2/D3.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

_STRATEGIES = ("zero-shot", "few-shot", "zero-cot", "few-cot")
_BENCHMARK_LEVELS = (0, 1, 2, 3, 4, 5)


def build_oracle(model_prefix: str, results_dir: Path) -> Path:
    """Build the oracle JSONL for a single model. Returns the output path.

    Orchestrates the full pipeline:
        1. Discover which ``{prefix}-{strategy}.quality.json`` files exist.
        2. Pick the best strategy per level based on ``per_level_scores``.
        3. Concatenate the per-level slices from the best-strategy JSONLs.
        4. Re-run evaluation (combined + quality-only) on the oracle JSONL
           to produce ``.json`` and ``.quality.json`` siblings.

    Raises:
        FileNotFoundError: if no strategy files exist for the prefix.
    """
    results_dir = Path(results_dir)

    # Step 1: discover available quality files (skip strategies missing either
    # the quality.json OR the matching jsonl — both are needed downstream).
    quality_files: dict[str, Path] = {}
    for strategy in _STRATEGIES:
        quality_path = results_dir / f"{model_prefix}-{strategy}.quality.json"
        jsonl_path = results_dir / f"{model_prefix}-{strategy}.jsonl"
        if not quality_path.exists():
            logger.info(
                "Skip strategy %s: quality file %s not found",
                strategy,
                quality_path,
            )
            continue
        if not jsonl_path.exists():
            logger.info(
                "Skip strategy %s: jsonl file %s not found (cannot extract samples)",
                strategy,
                jsonl_path,
            )
            continue
        quality_files[strategy] = quality_path
        logger.info("Found strategy %s: %s", strategy, quality_path)

    if not quality_files:
        raise FileNotFoundError(
            f"No strategy files found for prefix '{model_prefix}' in {results_dir}. "
            f"Expected at least one of: "
            f"{', '.join(f'{model_prefix}-{s}.quality.json' for s in _STRATEGIES)}"
        )

    # Step 2: compute per-level best strategy
    level_to_strategy = _pick_best_strategy_per_level(quality_files)
    logger.info("Oracle composition: %s", level_to_strategy)

    # Step 3: concatenate oracle jsonl
    oracle_jsonl = _concatenate_oracle_jsonl(
        model_prefix, level_to_strategy, results_dir
    )
    logger.info("Oracle jsonl written to %s", oracle_jsonl)

    # Step 4: re-evaluate (combined + quality-only)
    _evaluate_oracle_jsonl(oracle_jsonl, model_prefix)

    return oracle_jsonl


def _pick_best_strategy_per_level(
    quality_files: dict[str, Path],
) -> dict[int, str]:
    """Given {strategy -> quality.json path}, return {level -> best_strategy}.

    Reads ``per_level_scores`` (falling back to ``level_score`` for backward
    compatibility) from each quality file. Missing levels per strategy are
    skipped. Ties are broken by alphabetical order of strategy name.
    """
    # strategy -> {level: score}
    strategy_level_scores: dict[str, dict[int, float]] = {}
    for strategy, path in quality_files.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_scores = data.get("per_level_scores") or data.get("level_score") or {}
        if not raw_scores:
            logger.warning(
                "Strategy %s (%s) has neither per_level_scores nor level_score; skipping",
                strategy,
                path,
            )
            continue
        # keys may be str or int — normalise to int
        level_scores: dict[int, float] = {}
        for k, v in raw_scores.items():
            try:
                level_scores[int(k)] = float(v)
            except (TypeError, ValueError):
                logger.warning(
                    "Strategy %s: cannot parse level key %r -> skipping", strategy, k
                )
        strategy_level_scores[strategy] = level_scores
        logger.info("Loaded %s level scores: %s", strategy, level_scores)

    level_to_strategy: dict[int, str] = {}
    for level in _BENCHMARK_LEVELS:
        # Collect (score, strategy) pairs for this level, skipping strategies
        # that do not report it.
        candidates = [
            (score, strategy)
            for strategy, scores in strategy_level_scores.items()
            for lvl, score in scores.items()
            if lvl == level
        ]
        if not candidates:
            logger.info("Level %d: no strategy has a score; skipping", level)
            continue
        # Max by score, tie-break alphabetically by strategy name. Python's
        # sort is stable, so sort alphabetically first then take max by score.
        candidates.sort(key=lambda sc: sc[1])  # alphabetical asc
        best_score, best_strategy = max(candidates, key=lambda sc: sc[0])
        logger.info(
            "Level %d: best=%s (score=%.4f), all=%s",
            level,
            best_strategy,
            best_score,
            {strategy: f"{score:.4f}" for score, strategy in candidates},
        )
        level_to_strategy[level] = best_strategy

    return level_to_strategy


def _concatenate_oracle_jsonl(
    model_prefix: str,
    level_to_strategy: dict[int, str],
    results_dir: Path,
) -> Path:
    """Merge per-instruction records from per-level best-strategy JSONLs.

    For each level in sorted order, reads
    ``{model_prefix}-{level_to_strategy[level]}.jsonl`` and appends lines
    whose top-level ``level`` (falling back to ``instruction.level``) matches.
    The record order within each level follows the source file order.
    Overwrites any existing oracle.jsonl.
    """
    results_dir = Path(results_dir)
    output_path = results_dir / f"{model_prefix}-oracle.jsonl"

    # Cache per-strategy lines to avoid reading the same file multiple times
    # when a strategy is best for several levels.
    strategy_lines_cache: dict[str, list[tuple[int, dict]]] = {}

    def _record_level(record: dict) -> int | None:
        lvl = record.get("level")
        if lvl is None and isinstance(record.get("instruction"), dict):
            lvl = record["instruction"].get("level")
        if lvl is None:
            return None
        try:
            return int(lvl)
        except (TypeError, ValueError):
            return None

    with open(output_path, "w", encoding="utf-8") as out_f:
        for level in sorted(level_to_strategy):
            strategy = level_to_strategy[level]
            src_path = results_dir / f"{model_prefix}-{strategy}.jsonl"

            if strategy not in strategy_lines_cache:
                logger.info("Reading %s", src_path)
                cached: list[tuple[int, dict]] = []
                with open(src_path, "r", encoding="utf-8") as src_f:
                    for raw in src_f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except json.JSONDecodeError as e:
                            logger.warning("Skip malformed line in %s: %s", src_path, e)
                            continue
                        lvl = _record_level(rec)
                        if lvl is None:
                            continue
                        cached.append((lvl, rec))
                strategy_lines_cache[strategy] = cached

            level_records = [
                rec for lvl, rec in strategy_lines_cache[strategy] if lvl == level
            ]
            logger.info(
                "Level %d: copied %d records from %s",
                level,
                len(level_records),
                strategy,
            )
            for rec in level_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return output_path


def _evaluate_oracle_jsonl(jsonl_path: Path, model_prefix: str) -> None:
    """Re-evaluate the oracle jsonl, producing sibling .json and .quality.json.

    Mirrors the --both mode of ``scripts/eval_checkpoint.py``: a single
    combined evaluate_benchmark() call + a derived quality-only pass
    computed from the combined dimension_scores (no second eval).
    """
    # Imports deferred: these pull in torch / transformers and are slow.
    from graphinstruct.data_loader import load_all_levels
    from graphinstruct.evaluate import evaluate_benchmark
    from graphinstruct.scoring import (
        level_score as compute_level_score,
        total_score as compute_total_score,
    )
    from scripts.run_baseline import (
        _load_checkpoint,
        _rebuild_generation_results,
        serialize_benchmark_result,
    )

    all_instructions = load_all_levels(list(_BENCHMARK_LEVELS))
    _, records = _load_checkpoint(jsonl_path)
    all_results = _rebuild_generation_results(records, all_instructions)
    logger.info(
        "Re-evaluating oracle: %d instructions loaded from %s",
        len(all_results),
        jsonl_path,
    )

    # Combined evaluation (D1-D5)
    benchmark = evaluate_benchmark(all_results, strategy="oracle")
    for lvl, score in sorted(benchmark.per_level_scores.items()):
        logger.info("Oracle combined Level %s: %.4f", lvl, score)
    logger.info("Oracle combined total_score=%.4f", benchmark.total_score)
    logger.info("Oracle combined final_score=%.4f", benchmark.final_score)

    combined_path = jsonl_path.with_suffix(".json")
    combined_result = serialize_benchmark_result(benchmark)
    combined_result["metadata"] = {
        "model": model_prefix,
        "strategy": "oracle",
        "eval_mode": "combined",
        "samples": 5,
        "checkpoint": str(jsonl_path),
        "derived_from": "per-level best strategy cherry-pick",
    }
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_result, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", combined_path)

    # Quality-only (D1-D4) derived from combined dimension_scores.
    quality_path = jsonl_path.with_name(jsonl_path.stem + ".quality.json")
    quality_insts: list[dict] = []
    level_buckets: dict[int, list[float]] = {}
    for inst in combined_result["per_instruction"]:
        q_ls = compute_level_score(
            inst["level"], inst["dimension_scores"], quality_only=True
        )
        quality_insts.append({**inst, "level_score": q_ls})
        level_buckets.setdefault(inst["level"], []).append(q_ls)

    q_per_level = {lv: sum(s) / len(s) for lv, s in level_buckets.items()}
    if q_per_level:
        q_level_list = [q_per_level.get(lv, 0.0) for lv in range(max(q_per_level) + 1)]
        q_total = compute_total_score(q_level_list)
    else:
        q_total = 0.0

    quality_result = {
        "per_instruction": quality_insts,
        "per_level_scores": {str(k): v for k, v in q_per_level.items()},
        "total_score": q_total,
        "final_score": q_total,  # no Pareto bonus in quality mode
        "metadata": {
            "model": model_prefix,
            "strategy": "oracle",
            "eval_mode": "quality-only",
            "samples": 5,
            "checkpoint": str(jsonl_path),
            "derived_from": "combined",
        },
    }
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_result, f, indent=2, ensure_ascii=False)
    for lv in sorted(q_per_level):
        logger.info("Oracle quality Level %d: %.4f", lv, q_per_level[lv])
    logger.info("Oracle quality total_score=%.4f", q_total)
    logger.info("Wrote %s", quality_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E1b Oracle baseline")
    parser.add_argument(
        "--model-prefix",
        required=True,
        help=(
            "File prefix used by run_baseline outputs, e.g. 'gpt4omini' or "
            "'deepseekaiDeepSeekV3'. Oracle output uses the same prefix."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing existing per-strategy .jsonl / .quality.json files.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    out = build_oracle(args.model_prefix, args.results_dir)
    logger.info("Oracle baseline written to %s", out)


if __name__ == "__main__":
    main()
