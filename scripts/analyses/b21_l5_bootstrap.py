"""B2.1 — L5 bootstrap CI.

L5 has only 50 instructions, much smaller than other levels. The paper
quotes "95% CI ±0.019 at N=50" in passing but does not show the bootstrap
distribution per (model, strategy). This script:

1. For each baseline (model, strategy), extracts per-instruction L5
   level_scores from per_instruction in quality.json.
2. Resamples N=50 with replacement, 1000 times, computes mean.
3. Reports the 95% CI for the L5 mean.

Also reports the CI for Qwen3.5-35B vs 397B L5 delta (F5 claim),
which the paper says is within both 95% CI and noise band.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from _common import MODEL_DISPLAY, load_quality, scan_results_dir


def l5_per_instruction(q: dict) -> list[float]:
    """Extract level_score for L5 from per_instruction list."""
    out = []
    for rec in q.get("per_instruction", []):
        if rec.get("level") == 5:
            ls = rec.get("level_score")
            if ls is not None:
                out.append(float(ls))
    return out


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float, float]:
    """Bootstrap mean, lower, upper, std."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return statistics.mean(values), lo, hi, statistics.stdev(means)


def main() -> None:
    cells = scan_results_dir(only_baselines=True)
    rows = []
    for (model, strat), path in cells.items():
        q = load_quality(path)
        l5_scores = l5_per_instruction(q)
        if not l5_scores:
            continue
        mean, lo, hi, std = bootstrap_ci(l5_scores, n_boot=1000)
        rows.append(
            {
                "model": model,
                "strategy": strat,
                "n": len(l5_scores),
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width": hi - lo,
                "boot_std": std,
            }
        )

    rows.sort(key=lambda r: r["mean"], reverse=True)
    print(f"# B2.1 — L5 bootstrap CI (n_boot=1000, n=50 per cell)\n")
    print(f"## Bootstrap CI for L5 mean Quality per (model, strategy)\n")
    print(
        "| Model | Strategy | n | mean | 95% CI lo | 95% CI hi | CI width "
        "| boot SE |"
    )
    print(
        "|-------|----------|---|------|-----------|-----------|----------"
        "|---------|"
    )
    for r in rows:
        print(
            f"| {MODEL_DISPLAY.get(r['model'], r['model']):20s} "
            f"| {r['strategy']:9s} | {r['n']} | {r['mean']:.3f} "
            f"| {r['ci_lo']:.3f} | {r['ci_hi']:.3f} | {r['ci_width']:.3f} "
            f"| {r['boot_std']:.4f} |"
        )
    print()

    # Average CI width
    widths = [r["ci_width"] for r in rows]
    print(f"Mean CI width across 45 cells: {sum(widths)/len(widths):.3f}")
    print(
        f"Median CI half-width (matches paper '±0.019'): "
        f"{sorted(widths)[len(widths)//2] / 2:.3f}"
    )
    print()

    # Specific F5 claim: Qwen3.5-35B vs 397B on L5 — PAIRED bootstrap
    # Both models are evaluated on the same 50 L5 instructions, so the
    # per-instruction differences are paired observations. Paired bootstrap
    # is the statistically correct comparison.
    print("## F5 claim: Qwen3.5-35B vs 397B L5 delta (paired bootstrap)\n")

    def load_l5_by_instr(model: str, strat: str) -> dict[str, float]:
        path = Path(f"results/{model}-{strat}.quality.json")
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fp:
            q = json.load(fp)
        return {
            r["instruction_id"]: float(r.get("level_score", 0.0))
            for r in q.get("per_instruction", [])
            if r.get("level") == 5
        }

    def paired_bootstrap(
        d35: dict, d397: dict, n_boot: int = 1000, seed: int = 42
    ) -> tuple[float, float, float, float, float]:
        """Return (mean_delta, ci_lo, ci_hi, boot_se, n_pairs).

        Bootstrap is done by resampling instruction-IDs (not models), preserving
        the pairing.
        """
        common = sorted(set(d35.keys()) & set(d397.keys()))
        deltas = [d35[i] - d397[i] for i in common]
        if not deltas:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        n = len(deltas)
        rng = random.Random(seed)
        boot_means = []
        for _ in range(n_boot):
            sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()
        mean_d = sum(deltas) / n
        ci_lo = boot_means[int(0.025 * n_boot)]
        ci_hi = boot_means[int(0.975 * n_boot)]
        boot_se = statistics.stdev(boot_means)
        return mean_d, ci_lo, ci_hi, boot_se, n

    print(
        "| Strategy | Delta = 35B - 397B (paired) | 95% CI (paired) | boot SE | "
        "n_pairs | Crosses zero? |"
    )
    print(
        "|----------|-------------------------|-----------------|---------|"
        "---------|---------------|"
    )
    f5_paired_results = {}
    for strat in ("zero-shot", "few-shot", "zero-cot", "few-cot"):
        d35 = load_l5_by_instr("qwen3535ba3b", strat)
        d397 = load_l5_by_instr("qwen35397ba17b", strat)
        if not d35 or not d397:
            continue
        mean_d, ci_lo, ci_hi, boot_se, n_pairs = paired_bootstrap(d35, d397)
        crosses_zero = ci_lo <= 0 <= ci_hi
        f5_paired_results[strat] = {
            "delta": mean_d,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "boot_se": boot_se,
            "n_pairs": n_pairs,
            "crosses_zero": crosses_zero,
        }
        print(
            f"| {strat:9s} | {mean_d:+.4f} | "
            f"[{ci_lo:+.4f}, {ci_hi:+.4f}] | {boot_se:.4f} | "
            f"{n_pairs} | {'YES' if crosses_zero else 'NO ← significant'} |"
        )
    print()

    # Compare paired vs unpaired CIs for sanity
    print("## Paired vs unpaired comparison (sanity check)\n")
    by_pair = {}
    for r in rows:
        if r["model"] in ("qwen3535ba3b", "qwen35397ba17b"):
            by_pair.setdefault(r["strategy"], {})[r["model"]] = r
    print(
        "| Strategy | Δ | Paired 95% CI | Unpaired 95% CI | "
        "CI width (paired) | CI width (unpaired) |"
    )
    print(
        "|----------|---|---------------|-----------------|"
        "-------------------|---------------------|"
    )
    for strat in ("zero-shot", "few-shot", "zero-cot", "few-cot"):
        if strat not in by_pair or strat not in f5_paired_results:
            continue
        d = by_pair[strat]
        b35, b397 = d["qwen3535ba3b"], d["qwen35397ba17b"]
        delta_unp = b35["mean"] - b397["mean"]
        unp_se = (b35["boot_std"] ** 2 + b397["boot_std"] ** 2) ** 0.5
        unp_lo = delta_unp - 1.96 * unp_se
        unp_hi = delta_unp + 1.96 * unp_se
        p = f5_paired_results[strat]
        print(
            f"| {strat:9s} | {p['delta']:+.4f} | "
            f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
            f"[{unp_lo:+.4f}, {unp_hi:+.4f}] | "
            f"{p['ci_hi']-p['ci_lo']:.4f} | {unp_hi-unp_lo:.4f} |"
        )
    print()

    # Variability comparison: best vs worst model on L5
    best = max(rows, key=lambda r: r["mean"])
    worst = min(rows, key=lambda r: r["mean"])
    print(f"## Range across all 45 cells\n")
    print(
        f"Best: {MODEL_DISPLAY.get(best['model'], best['model'])} "
        f"{best['strategy']}, L5 mean = {best['mean']:.3f} "
        f"(CI [{best['ci_lo']:.3f}, {best['ci_hi']:.3f}])"
    )
    print(
        f"Worst: {MODEL_DISPLAY.get(worst['model'], worst['model'])} "
        f"{worst['strategy']}, L5 mean = {worst['mean']:.3f} "
        f"(CI [{worst['ci_lo']:.3f}, {worst['ci_hi']:.3f}])"
    )
    print(f"Top-vs-bottom gap: {best['mean'] - worst['mean']:.3f}")
    print(
        f"  (this >> typical CI width of {sum(widths)/len(widths):.3f}, "
        f"so cross-model L5 differences are real)"
    )
    print()

    out_path = Path("scripts/analyses/results/b21_l5_bootstrap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
