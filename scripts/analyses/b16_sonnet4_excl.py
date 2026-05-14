"""B1.6 — Sonnet-4 exclusion sensitivity.

Sonnet-4 is included in some analyses (tier-mean, Pareto frontier, top-N
leaderboard) but excluded from strategy-effect analyses (RQ3/4/5) because
only zero-shot data is available. This script recomputes the affected
quantities with Sonnet-4 entirely excluded to verify the conclusions are
not driven by this single zero-shot-only configuration.

Quantities recomputed:
1. Tier-level means and T1-T3 gap (RQ1, Tab. tier-gap)
2. 45-cell Pareto frontier (RQ6)
3. Top-15 Stot leaderboard (App. profiles)
4. Top-5 Sfin leaderboard (RQ6)
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import (
    LEVEL_WEIGHTS,
    MODEL_DISPLAY,
    MODEL_TIER,
    PARETO_LAMBDA,
    compute_total_score,
    compute_tpv_api,
    find_pareto_front,
    load_quality,
    per_level_scores,
    scan_results_dir,
)

SONNET4_KEY = "claudesonnet420250514"


def aggregate_with_or_without_sonnet4(
    cells: dict[tuple[str, str], Path],
) -> tuple[dict, dict]:
    """Compute aggregate stats with and without Sonnet-4.

    Returns (with_sonnet4_dict, without_sonnet4_dict), each containing:
    - per_cell: dict[(model, strat) -> dict of stats]
    - tier_means: dict[tier -> dict[level -> mean]]
    - pareto_keys: set of (model, strat) on Pareto frontier
    """

    def collect(include_sonnet4: bool):
        per_cell = {}
        per_tier_per_level = {"T1": [], "T2": [], "T3": []}
        per_tier_per_level = {t: {l: [] for l in range(6)} for t in ("T1", "T2", "T3")}

        for (model, strat), path in cells.items():
            if not include_sonnet4 and model == SONNET4_KEY:
                continue
            q = load_quality(path)
            pls = per_level_scores(q)
            total = q.get("total_score", compute_total_score(pls))

            # Compute TPV and Sfin from jsonl
            jsonl_path = path.parent / path.name.replace(".quality.json", ".jsonl")
            tpv, api, _ = compute_tpv_api(jsonl_path)

            per_cell[(model, strat)] = {
                "total_score": total,
                "per_level": pls,
                "tpv": tpv,
                "api": api,
            }
            tier = MODEL_TIER.get(model, "??")
            if tier in per_tier_per_level:
                for lvl in range(6):
                    if lvl in pls:
                        per_tier_per_level[tier][lvl].append(pls[lvl])

        tier_means = {
            t: {lvl: sum(vs) / len(vs) if vs else 0.0 for lvl, vs in lvls.items()}
            for t, lvls in per_tier_per_level.items()
        }
        tier_gap = {
            lvl: tier_means["T1"][lvl] - tier_means["T3"][lvl] for lvl in range(6)
        }

        # Pareto frontier on (TPV, total_score)
        pts = [
            (d["tpv"], d["total_score"], key)
            for key, d in per_cell.items()
            if d["tpv"] != float("inf")
        ]
        pareto = find_pareto_front(pts)

        return {
            "per_cell": per_cell,
            "tier_means": tier_means,
            "tier_gap": tier_gap,
            "pareto_keys": pareto,
            "n_cells": len(per_cell),
        }

    return collect(include_sonnet4=True), collect(include_sonnet4=False)


def fmt_cell(key: tuple[str, str]) -> str:
    m, s = key
    return f"{MODEL_DISPLAY.get(m, m)} {s}"


def main() -> None:
    # Only baseline cells (zero-shot, few-shot, zero-cot, few-cot)
    cells = scan_results_dir(only_baselines=True)
    print(f"# Baseline cells found: {len(cells)}")
    print()

    with_s, without_s = aggregate_with_or_without_sonnet4(cells)

    print("## Tier-level means + T1-T3 gap (with vs without Sonnet-4)")
    print()
    print(
        "| Level | T1 (with) | T1 (w/o) | T2 (with) | T2 (w/o) | "
        "T3 (with) | T3 (w/o) | Gap (with) | Gap (w/o) | Δ Gap |"
    )
    print(
        "|-------|-----------|----------|-----------|----------|"
        "-----------|----------|------------|-----------|-------|"
    )
    for lvl in range(6):
        t1_w = with_s["tier_means"]["T1"][lvl]
        t1_n = without_s["tier_means"]["T1"][lvl]
        t2_w = with_s["tier_means"]["T2"][lvl]
        t2_n = without_s["tier_means"]["T2"][lvl]
        t3_w = with_s["tier_means"]["T3"][lvl]
        t3_n = without_s["tier_means"]["T3"][lvl]
        g_w = with_s["tier_gap"][lvl]
        g_n = without_s["tier_gap"][lvl]
        d_g = g_n - g_w
        print(
            f"| L{lvl} | {t1_w:.3f} | {t1_n:.3f} | {t2_w:.3f} | {t2_n:.3f} "
            f"| {t3_w:.3f} | {t3_n:.3f} | {g_w:.3f} | {g_n:.3f} | {d_g:+.4f} |"
        )
    print()

    # Pareto frontier comparison
    print("## Pareto frontier size and composition")
    print()
    print(f"With Sonnet-4    : {len(with_s['pareto_keys'])} Pareto-optimal cells")
    print(f"Without Sonnet-4 : {len(without_s['pareto_keys'])} Pareto-optimal cells")
    only_with = with_s["pareto_keys"] - without_s["pareto_keys"]
    only_without = without_s["pareto_keys"] - with_s["pareto_keys"]
    common = with_s["pareto_keys"] & without_s["pareto_keys"]
    print(f"Common Pareto cells: {len(common)}")
    if only_with:
        print(
            f"  Cells lost by removing Sonnet-4 (i.e., Sonnet-4 itself or "
            f"its near-frontier neighbors): {[fmt_cell(k) for k in only_with]}"
        )
    if only_without:
        print(
            f"  Cells newly Pareto after removing Sonnet-4: "
            f"{[fmt_cell(k) for k in only_without]}"
        )
    print()

    # Top-15 leaderboard impact
    print("## Top-15 Stot leaderboard impact")
    print()
    with_sorted = sorted(
        with_s["per_cell"].items(), key=lambda kv: kv[1]["total_score"], reverse=True
    )
    without_sorted = sorted(
        without_s["per_cell"].items(), key=lambda kv: kv[1]["total_score"], reverse=True
    )
    with_top15 = [k for k, _ in with_sorted[:15]]
    without_top15 = [k for k, _ in without_sorted[:15]]
    overlap = set(with_top15) & set(without_top15)
    print(f"Top-15 overlap: {len(overlap)} / 15")
    sonnet4_in_top15 = sum(1 for k in with_top15 if k[0] == SONNET4_KEY)
    print(f"Sonnet-4 entries in top-15 (with-version): {sonnet4_in_top15}")
    print()

    # Max shift in rank inside top-15
    rank_with = {k: i + 1 for i, k in enumerate(with_top15)}
    max_shift = 0
    max_shift_key = None
    for i, k in enumerate(without_top15):
        if k in rank_with:
            shift = abs(rank_with[k] - (i + 1))
            if shift > max_shift:
                max_shift = shift
                max_shift_key = k
    if max_shift_key:
        print(
            f"Max rank shift inside top-15: {max_shift} positions "
            f"({fmt_cell(max_shift_key)})"
        )
    print()

    # Save full output
    out = {
        "with_sonnet4": {
            "n_cells": with_s["n_cells"],
            "tier_means": {
                t: {str(l): v for l, v in lvls.items()}
                for t, lvls in with_s["tier_means"].items()
            },
            "tier_gap": {str(l): v for l, v in with_s["tier_gap"].items()},
            "pareto_cells": [list(k) for k in with_s["pareto_keys"]],
            "top_15": [
                [k[0], k[1], with_s["per_cell"][k]["total_score"]] for k in with_top15
            ],
        },
        "without_sonnet4": {
            "n_cells": without_s["n_cells"],
            "tier_means": {
                t: {str(l): v for l, v in lvls.items()}
                for t, lvls in without_s["tier_means"].items()
            },
            "tier_gap": {str(l): v for l, v in without_s["tier_gap"].items()},
            "pareto_cells": [list(k) for k in without_s["pareto_keys"]],
            "top_15": [
                [k[0], k[1], without_s["per_cell"][k]["total_score"]]
                for k in without_top15
            ],
        },
        "summary": {
            "top15_overlap": len(overlap),
            "pareto_size_with": len(with_s["pareto_keys"]),
            "pareto_size_without": len(without_s["pareto_keys"]),
            "max_l2_gap_change": without_s["tier_gap"][2] - with_s["tier_gap"][2],
        },
    }
    out_path = Path("scripts/analyses/results/b16_sonnet4_excl.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
