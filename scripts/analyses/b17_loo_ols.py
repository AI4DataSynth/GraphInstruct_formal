"""B1.7 — Leave-one-model-out OLS for F2 (sigma_strat ~ mean Q).

F2 claims: prompt sensitivity inversely scales with capability
(beta = -0.27, R^2 = 0.62, p < 10^-3, on 11 fully-evaluated models).

With n=11, the OLS is fragile to a single leverage point. This script:
1. Replicates the full-data OLS.
2. Runs leave-one-model-out (11 LOO fits).
3. Reports beta range, R^2 range, and which model is the most leveraged.
4. Computes bootstrap 95% CI for beta over n_boot=10000 resamples.

For Sonnet-4 (zero-shot only), sigma_strat is undefined, so it is excluded
(same as the paper).
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from _common import (
    BASELINE_STRATS,
    MODEL_DISPLAY,
    MODEL_TIER,
    compute_total_score,
    load_quality,
    per_level_scores,
    scan_results_dir,
)

SONNET4_KEY = "claudesonnet420250514"


def gather_q_and_sigma() -> list[tuple[str, float, float]]:
    """For each fully-evaluated model, compute (mean Q across 4 strategies,
    sigma_strat = std of Q across 4 strategies)."""
    cells = scan_results_dir(only_baselines=True)
    by_model: dict[str, dict[str, float]] = {}
    for (model, strat), path in cells.items():
        q = load_quality(path)
        Q = q.get("total_score", compute_total_score(per_level_scores(q)))
        by_model.setdefault(model, {})[strat] = Q

    rows: list[tuple[str, float, float]] = []
    for model, strat_qs in by_model.items():
        if model == SONNET4_KEY:
            continue  # zero-shot only, no sigma
        if not all(s in strat_qs for s in BASELINE_STRATS):
            continue  # missing strategies
        qs = [strat_qs[s] for s in BASELINE_STRATS]
        mean_q = statistics.mean(qs)
        # Use population stdev (ddof=0), matching the paper's figure-
        # generation code paper_figures.py::fig_F4_capability_variance,
        # which uses numpy's default np.std. The GraphInstruct paper reports
        # this slope (-0.073) rather than the sample-stdev slope (-0.085).
        sigma = statistics.pstdev(qs)
        rows.append((model, mean_q, sigma))
    return rows


def ols_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Simple OLS fit y = alpha + beta*x. Returns (alpha, beta, R^2)."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0:
        return my, 0.0, 0.0
    beta = sxy / sxx
    alpha = my - beta * mx
    # R^2
    ss_res = sum((y[i] - (alpha + beta * x[i])) ** 2 for i in range(n))
    r2 = 1 - (ss_res / syy) if syy > 0 else 0.0
    return alpha, beta, r2


def ols_pvalue_t(x: list[float], y: list[float], beta: float) -> float:
    """Approximate t-test p-value for beta != 0 using two-sided normal CDF.

    For n=11 (df=9) the t-distribution differs slightly from normal; we use
    the t-statistic and a normal approximation tail (acceptable for our
    purposes; the precise NumPy/scipy version is in the SUPP if needed).
    """
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    ss_res = sum((y[i] - (my + beta * (x[i] - mx))) ** 2 for i in range(n))
    if n - 2 <= 0 or sxx == 0:
        return float("nan")
    se = math.sqrt(ss_res / (n - 2) / sxx)
    if se == 0:
        return 0.0
    t = beta / se
    # Two-sided normal tail
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return p


def main() -> None:
    rows = gather_q_and_sigma()
    print(f"# B1.7 — Leave-one-model-out OLS for F2\n")
    print(f"## Raw data: {len(rows)} fully-evaluated models\n")
    print("| Model | Tier | mean Q | sigma_strat |")
    print("|-------|------|--------|-------------|")
    for model, q, sigma in sorted(rows, key=lambda r: r[1], reverse=True):
        print(
            f"| {MODEL_DISPLAY.get(model, model):20s} | "
            f"{MODEL_TIER.get(model, '??')} | {q:.3f} | {sigma:.4f} |"
        )
    print()

    # Full-data OLS
    x_full = [r[1] for r in rows]
    y_full = [r[2] for r in rows]
    alpha, beta, r2 = ols_fit(x_full, y_full)
    p = ols_pvalue_t(x_full, y_full, beta)
    print(f"## Full-data OLS (n={len(rows)})\n")
    print(f"sigma_strat = {alpha:.4f} + {beta:.4f} * mean_Q")
    print(f"R^2 = {r2:.3f}")
    print(f"p (approx, two-sided) = {p:.2e}")
    print(f"Paper reports: beta = -0.27, R^2 = 0.62, p < 10^-3")
    print()

    # Leave-one-out
    print(f"## Leave-one-out OLS (n=10 each, 11 fits)\n")
    print("| Excluded | mean_Q | sigma | beta (LOO) | R^2 (LOO) | Δ beta vs full |")
    print("|----------|--------|-------|------------|----------|----------------|")
    loo_betas = []
    loo_r2 = []
    for i, (model, q, sigma) in enumerate(rows):
        x_loo = [r[1] for j, r in enumerate(rows) if j != i]
        y_loo = [r[2] for j, r in enumerate(rows) if j != i]
        _, beta_loo, r2_loo = ols_fit(x_loo, y_loo)
        loo_betas.append(beta_loo)
        loo_r2.append(r2_loo)
        d = beta_loo - beta
        print(
            f"| {MODEL_DISPLAY.get(model, model):20s} | {q:.3f} | {sigma:.4f} "
            f"| {beta_loo:+.4f} | {r2_loo:.3f} | {d:+.4f} |"
        )
    print()
    print(f"LOO beta range: [{min(loo_betas):.4f}, {max(loo_betas):.4f}]")
    print(f"LOO R^2 range: [{min(loo_r2):.3f}, {max(loo_r2):.3f}]")
    print(
        f"Sign stability: {sum(1 for b in loo_betas if b < 0)}/{len(loo_betas)} negative"
    )
    print()

    # Bootstrap CI for beta
    n_boot = 10000
    rng = random.Random(42)
    boot_betas = []
    n = len(rows)
    for _ in range(n_boot):
        sample_idx = [rng.randint(0, n - 1) for _ in range(n)]
        x_b = [x_full[i] for i in sample_idx]
        y_b = [y_full[i] for i in sample_idx]
        try:
            _, b, _ = ols_fit(x_b, y_b)
            boot_betas.append(b)
        except (ZeroDivisionError, statistics.StatisticsError):
            continue
    boot_betas.sort()
    ci_lo = boot_betas[int(0.025 * len(boot_betas))]
    ci_hi = boot_betas[int(0.975 * len(boot_betas))]
    print(f"## Bootstrap 95% CI for beta (n_boot={n_boot})\n")
    print(f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    n_negative = sum(1 for b in boot_betas if b < 0)
    print(f"P(beta < 0) ≈ {n_negative/len(boot_betas):.4f}")
    print(f"  → CI excludes zero: {ci_lo < 0 and ci_hi < 0}")
    print()

    out = {
        "full_data": {
            "n": len(rows),
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "p_approx": p,
        },
        "loo": [
            {"excluded": MODEL_DISPLAY.get(m, m), "beta": b, "r2": r2}
            for (m, _, _), b, r2 in zip(rows, loo_betas, loo_r2)
        ],
        "loo_beta_range": [min(loo_betas), max(loo_betas)],
        "loo_r2_range": [min(loo_r2), max(loo_r2)],
        "bootstrap": {
            "n": n_boot,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_negative": n_negative / len(boot_betas),
        },
        "raw_data": [
            {"model": m, "tier": MODEL_TIER.get(m, "??"), "mean_Q": q, "sigma_strat": s}
            for m, q, s in rows
        ],
    }
    out_path = Path("scripts/analyses/results/b17_loo_ols.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
