"""B2.3 — Sign-count table for strategy effects.

F3 reports averaged-across-11-models strategy deltas (FS-ZS, ZC-ZS, FC-ZS)
per level (Tab. strategy-delta). Averages hide whether the sign is robust
across models. This script counts, for each (level, strategy) cell:
- # models with Δ > 0
- # models with Δ < 0
- # models with |Δ| <= 0.005 (within stability band)

So users can verify whether "FS net-negative at L2 (-0.034)" is
e.g. 11/11 or 7/4 split.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import (
    BASELINE_STRATS,
    MODEL_DISPLAY,
    compute_total_score,
    load_quality,
    per_level_scores,
    scan_results_dir,
)

NOISE_BAND = 0.005


def main() -> None:
    cells = scan_results_dir(only_baselines=True)
    # by_model[model][strat] = quality.json
    by_model: dict[str, dict[str, dict]] = {}
    for (model, strat), path in cells.items():
        q = load_quality(path)
        by_model.setdefault(model, {})[strat] = q

    # Filter to models with all 4 strategies (exclude Sonnet-4 zero-shot-only)
    fully_eval = [
        m for m, sd in by_model.items() if all(s in sd for s in BASELINE_STRATS)
    ]
    print(f"# B2.3 — Sign-count table for strategy effects\n")
    print(
        f"# {len(fully_eval)} fully-evaluated models: "
        f"{sorted(MODEL_DISPLAY[m] for m in fully_eval)}\n"
    )
    print(f"Noise band: ±{NOISE_BAND} (effects within this are reported as 'tie')\n")

    # Per (level, strategy_vs_zs) collect Δs per model
    per_cell: dict[tuple[int, str], list[tuple[str, float]]] = {}
    for model in fully_eval:
        sd = by_model[model]
        pls_zs = per_level_scores(sd["zero-shot"])
        for strat_other in ("few-shot", "zero-cot", "few-cot"):
            pls_o = per_level_scores(sd[strat_other])
            for lvl in range(6):
                d = pls_o.get(lvl, 0) - pls_zs.get(lvl, 0)
                per_cell.setdefault((lvl, strat_other), []).append((model, d))

    # For each level, output sign-count
    label_map = {"few-shot": "FS-ZS", "zero-cot": "ZC-ZS", "few-cot": "FC-ZS"}
    print(
        "## Sign-count: # of fully-evaluated models in each cell with positive / "
        "negative / tie effect\n"
    )
    print("| Level | Strategy | mean Δ | # pos | # neg | # tie | dominant sign |")
    print("|-------|----------|--------|-------|-------|-------|---------------|")
    rows_summary = []
    for lvl in range(6):
        for strat in ("few-shot", "zero-cot", "few-cot"):
            ds = per_cell.get((lvl, strat), [])
            if not ds:
                continue
            mean_d = sum(d for _, d in ds) / len(ds)
            n_pos = sum(1 for _, d in ds if d > NOISE_BAND)
            n_neg = sum(1 for _, d in ds if d < -NOISE_BAND)
            n_tie = len(ds) - n_pos - n_neg
            if n_pos > n_neg + 1:
                dom = f"POS ({n_pos}/{len(ds)})"
            elif n_neg > n_pos + 1:
                dom = f"NEG ({n_neg}/{len(ds)})"
            else:
                dom = f"MIXED ({n_pos}+/{n_neg}-/{n_tie}=)"
            print(
                f"| L{lvl} | {label_map[strat]:5s} | {mean_d:+.3f} "
                f"| {n_pos} | {n_neg} | {n_tie} | {dom} |"
            )
            rows_summary.append(
                {
                    "level": lvl,
                    "strategy": strat,
                    "label": label_map[strat],
                    "mean_delta": mean_d,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_tie": n_tie,
                    "dominant": dom,
                }
            )
    print()

    # Highlight cells where mean and dominant sign disagree
    print("## Cells where the mean and the dominant sign disagree (or close call)\n")
    print("| Level | Strategy | mean Δ | # pos | # neg | # tie | comment |")
    print("|-------|----------|--------|-------|-------|-------|---------|")
    flagged = []
    for r in rows_summary:
        n = r["n_pos"] + r["n_neg"] + r["n_tie"]
        mean = r["mean_delta"]
        # Disagreement: mean is positive but more negs, or vice versa; or close call
        sign_mean = 1 if mean > NOISE_BAND else (-1 if mean < -NOISE_BAND else 0)
        sign_count = (
            1 if r["n_pos"] > r["n_neg"] else (-1 if r["n_neg"] > r["n_pos"] else 0)
        )
        margin = abs(r["n_pos"] - r["n_neg"])
        if sign_mean != sign_count or margin <= 2:
            cmt = (
                f"mean says {'+' if sign_mean > 0 else ('-' if sign_mean<0 else '0')}, "
                f"vote says {'+' if sign_count > 0 else ('-' if sign_count<0 else '0')}, "
                f"margin={margin}"
            )
            print(
                f"| L{r['level']} | {r['label']:5s} | {r['mean_delta']:+.3f} "
                f"| {r['n_pos']} | {r['n_neg']} | {r['n_tie']} | {cmt} |"
            )
            flagged.append(r)
    if not flagged:
        print("| (none) | | | | | | |")
    print()

    # Per-model strategy effects (the data behind paper Tab. strategy-delta)
    print("## Per-model per-level signed effects (FC-ZS only, showing F4 evidence)\n")
    print("| Model | L0 | L1 | L2 | L3 | L4 | L5 |")
    print("|-------|------|------|------|------|------|------|")
    for model in fully_eval:
        sd = by_model[model]
        zs = per_level_scores(sd["zero-shot"])
        fc = per_level_scores(sd["few-cot"])
        cells_row = [f"{fc.get(l, 0) - zs.get(l, 0):+.3f}" for l in range(6)]
        print(f"| {MODEL_DISPLAY[model]:20s} | " + " | ".join(cells_row) + " |")
    print()

    out = {
        "n_models": len(fully_eval),
        "noise_band": NOISE_BAND,
        "sign_counts": rows_summary,
        "flagged_close_calls": flagged,
        "per_model_fc_minus_zs": {
            MODEL_DISPLAY[m]: {
                str(l): per_level_scores(by_model[m]["few-cot"]).get(l, 0)
                - per_level_scores(by_model[m]["zero-shot"]).get(l, 0)
                for l in range(6)
            }
            for m in fully_eval
        },
    }
    out_path = Path("scripts/analyses/results/b23_sign_count.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
