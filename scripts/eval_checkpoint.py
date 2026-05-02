"""Evaluate a completed JSONL checkpoint without re-running generation.

Usage:
    # Combined evaluation (D1-D5, default)
    python scripts/eval_checkpoint.py results/gpt4omini-zero-shot.jsonl

    # Quality-only (D1-D4, no token efficiency)
    python scripts/eval_checkpoint.py results/gpt4omini-zero-shot.jsonl --quality-only

    # Both modes at once
    python scripts/eval_checkpoint.py results/gpt4omini-zero-shot.jsonl --both
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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


def run_eval(all_results, ckpt_path, out_path, model, strategy, quality_only):
    """Run one evaluation pass and save results."""
    mode = "quality-only" if quality_only else "combined"
    print(f"\n=== Evaluation: {mode} ===", flush=True)

    benchmark = evaluate_benchmark(
        all_results, quality_only=quality_only, strategy=strategy
    )
    for level, score in sorted(benchmark.per_level_scores.items()):
        print(f"  Level {level}: {score:.4f}")
    print(f"  Total score: {benchmark.total_score:.4f}")
    print(f"  Final score: {benchmark.final_score:.4f}")

    result = serialize_benchmark_result(benchmark)
    result["metadata"] = {
        "model": model,
        "strategy": strategy,
        "eval_mode": mode,
        "samples": 5,
        "checkpoint": str(ckpt_path),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate JSONL checkpoint")
    parser.add_argument("checkpoint", help="Path to .jsonl checkpoint file")
    parser.add_argument(
        "--output", help="Output JSON path (default: auto from checkpoint name)"
    )
    parser.add_argument("--model", default="unknown", help="Model name for metadata")
    parser.add_argument("--strategy", default="unknown", help="Strategy for metadata")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quality-only",
        action="store_true",
        help="Exclude D5 (token efficiency), only score D1-D4",
    )
    mode_group.add_argument(
        "--both",
        action="store_true",
        help="Run both combined and quality-only evaluations",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        sys.exit(1)

    # Load data once, reuse for both modes
    all_instructions = load_all_levels([0, 1, 2, 3, 4, 5])
    _, records = _load_checkpoint(ckpt_path)
    all_results = _rebuild_generation_results(records, all_instructions)
    print(f"Loaded {len(all_results)} instructions from {ckpt_path}")

    if args.both:
        # Run combined evaluation once (D1-D5)
        out_combined = (
            Path(args.output) if args.output else ckpt_path.with_suffix(".json")
        )
        run_eval(all_results, ckpt_path, out_combined, args.model, args.strategy, False)

        # Derive quality-only (D1-D4) by re-scoring from combined dimension_scores.
        # This avoids a second full evaluate_benchmark() call — D1-D4 raw scores
        # are identical, only level_score weights differ (D5 zeroed + renormalized).
        out_quality = ckpt_path.with_name(ckpt_path.stem + ".quality.json")
        if not out_combined.exists():
            print(
                f"ERROR: Combined output {out_combined} not found, "
                "cannot derive quality scores",
                flush=True,
            )
            sys.exit(1)
        with open(out_combined, encoding="utf-8") as f:
            combined = json.load(f)

        quality_insts = []
        level_buckets: dict[int, list[float]] = {}
        for inst in combined["per_instruction"]:
            q_ls = compute_level_score(
                inst["level"], inst["dimension_scores"], quality_only=True
            )
            quality_insts.append({**inst, "level_score": q_ls})
            level_buckets.setdefault(inst["level"], []).append(q_ls)

        q_per_level = {lv: sum(s) / len(s) for lv, s in level_buckets.items()}
        if q_per_level:
            q_level_list = [
                q_per_level.get(lv, 0.0) for lv in range(max(q_per_level) + 1)
            ]
            q_total = compute_total_score(q_level_list)
        else:
            q_total = 0.0

        quality_result = {
            "per_instruction": quality_insts,
            "per_level_scores": {str(k): v for k, v in q_per_level.items()},
            "total_score": q_total,
            "final_score": q_total,  # no Pareto bonus in quality mode
            "metadata": {
                "model": args.model,
                "strategy": args.strategy,
                "eval_mode": "quality-only",
                "samples": 5,
                "checkpoint": str(ckpt_path),
                "derived_from": "combined",
            },
        }
        with open(out_quality, "w", encoding="utf-8") as f:
            json.dump(quality_result, f, indent=2, ensure_ascii=False)
        print(f"\n=== Quality-only (derived from combined) ===", flush=True)
        for lv in sorted(q_per_level):
            print(f"  Level {lv}: {q_per_level[lv]:.4f}")
        print(f"  Total score: {q_total:.4f}")
        print(f"Saved to {out_quality}", flush=True)
    elif args.quality_only:
        out_path = (
            Path(args.output)
            if args.output
            else ckpt_path.with_name(ckpt_path.stem + ".quality.json")
        )
        run_eval(all_results, ckpt_path, out_path, args.model, args.strategy, True)
    else:
        out_path = Path(args.output) if args.output else ckpt_path.with_suffix(".json")
        run_eval(all_results, ckpt_path, out_path, args.model, args.strategy, False)


if __name__ == "__main__":
    main()
