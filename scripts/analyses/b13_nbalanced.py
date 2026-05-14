"""B1.3 — N-balanced level-weight ablation.

L5 has only 50 instructions but carries weight 0.25 (same as L4 with 100,
and 5× L0 with 100). Per-instruction influence on Stot is therefore 20× L0.
This ablation recomputes the 45-cell leaderboard under several alternative
weight schemes to verify the ranking is robust to this design choice.

Schemes:
- Default: w = (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)  [paper]
- N-balanced (1/n_l): w ∝ (1/100, 1/200, 1/200, 1/150, 1/100, 1/50), normalised
- Uniform: w = (1/6,)*6
- D4-mass-balanced: w ∝ n_l, normalised — gives small levels less voice

For each scheme, compute:
- Top-9 retained from default
- Top-15 retained from default
- Top-15 Jaccard
- Max rank shift within top-15
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import (
    LEVEL_N,
    LEVEL_WEIGHTS,
    MODEL_DISPLAY,
    compute_total_score,
    load_quality,
    per_level_scores,
    scan_results_dir,
)

# Define alternative weight schemes
SCHEMES = {
    "default": LEVEL_WEIGHTS,
    "N-balanced (1/n_l)": tuple(  # noqa: E501
        w / sum(1 / n for n in LEVEL_N) for w in [1 / n for n in LEVEL_N]
    ),
    "Uniform (1/6)": (1 / 6,) * 6,
    "Mass-balanced (n_l)": tuple(n / sum(LEVEL_N) for n in LEVEL_N),
}


def main() -> None:
    print("# B1.3 — N-balanced level-weight ablation\n")
    print("## Weight schemes\n")
    print("| Scheme | L0 | L1 | L2 | L3 | L4 | L5 | Sum |")
    print("|--------|------|------|------|------|------|------|------|")
    for name, w in SCHEMES.items():
        print(
            f"| {name:25s} | {w[0]:.3f} | {w[1]:.3f} | {w[2]:.3f} "
            f"| {w[3]:.3f} | {w[4]:.3f} | {w[5]:.3f} | {sum(w):.3f} |"
        )
    print()

    # Load all baseline cells
    cells = scan_results_dir(only_baselines=True)
    print(f"# Loaded {len(cells)} baseline cells\n")

    # For each cell, compute total under each scheme
    per_cell_totals: dict[tuple[str, str], dict[str, float]] = {}
    for key, path in cells.items():
        q = load_quality(path)
        pls = per_level_scores(q)
        per_cell_totals[key] = {
            name: compute_total_score(pls, w) for name, w in SCHEMES.items()
        }

    # Compute rank lists under each scheme
    scheme_rankings: dict[str, list[tuple[str, str]]] = {}
    for scheme in SCHEMES:
        ranked = sorted(
            per_cell_totals.items(), key=lambda kv: kv[1][scheme], reverse=True
        )
        scheme_rankings[scheme] = [k for k, _ in ranked]

    default_top9 = set(scheme_rankings["default"][:9])
    default_top15 = set(scheme_rankings["default"][:15])
    default_top20 = set(scheme_rankings["default"][:20])
    default_rank_top15 = {
        k: i + 1 for i, k in enumerate(scheme_rankings["default"][:15])
    }

    print("## Sensitivity matrix (overlap with default ranking)\n")
    print(
        "| Scheme | Top-9 retained | Top-15 retained | Top-20 retained "
        "| Top-15 Jaccard | Max rank shift in top-15 |"
    )
    print(
        "|--------|----------------|-----------------|-----------------"
        "|----------------|--------------------------|"
    )
    for scheme, rank_list in scheme_rankings.items():
        top9 = set(rank_list[:9])
        top15 = set(rank_list[:15])
        top20 = set(rank_list[:20])
        n9 = len(top9 & default_top9)
        n15 = len(top15 & default_top15)
        n20 = len(top20 & default_top20)
        jac15 = len(top15 & default_top15) / len(top15 | default_top15)
        # Max rank shift inside default top-15
        max_shift = 0
        for i, k in enumerate(rank_list[:20]):
            if k in default_rank_top15:
                shift = abs((i + 1) - default_rank_top15[k])
                max_shift = max(max_shift, shift)
        print(
            f"| {scheme:25s} | {n9}/9 | {n15}/15 | {n20}/20 "
            f"| {jac15:.3f} | {max_shift} |"
        )
    print()

    # Show top-15 for each scheme side by side
    print("## Top-15 leaderboards per scheme\n")
    print("| # | Default | N-balanced | Uniform | Mass-balanced |")
    print("|---|---------|------------|---------|---------------|")
    for i in range(15):
        cells_row = []
        for scheme in [
            "default",
            "N-balanced (1/n_l)",
            "Uniform (1/6)",
            "Mass-balanced (n_l)",
        ]:
            k = scheme_rankings[scheme][i]
            model, strat = k
            disp = f"{MODEL_DISPLAY.get(model, model)} {strat}"
            score = per_cell_totals[k][scheme]
            cells_row.append(f"{disp} ({score:.3f})")
        print(f"| {i+1} | " + " | ".join(cells_row) + " |")
    print()

    # Quantify L5-influence test: does removing L5 entirely change top-9?
    no_l5_weights = (0.05 / 0.75, 0.10 / 0.75, 0.15 / 0.75, 0.20 / 0.75, 0.25 / 0.75, 0)
    print("## L5-only ablation: zero L5 weight (renormalised D1-D4 to sum 1)\n")
    per_cell_no_l5 = {
        k: compute_total_score(per_level_scores(load_quality(path)), no_l5_weights)
        for k, path in cells.items()
    }
    ranked_no_l5 = [
        k for k, _ in sorted(per_cell_no_l5.items(), key=lambda kv: kv[1], reverse=True)
    ]
    no_l5_top9 = set(ranked_no_l5[:9])
    no_l5_top15 = set(ranked_no_l5[:15])
    print(f"Top-9 retained without L5: {len(no_l5_top9 & default_top9)}/9")
    print(f"Top-15 retained without L5: {len(no_l5_top15 & default_top15)}/15")
    print()

    # Persist
    out = {
        "schemes": {name: list(w) for name, w in SCHEMES.items()},
        "default_top15": [
            [k[0], k[1], per_cell_totals[k]["default"]]
            for k in scheme_rankings["default"][:15]
        ],
        "n_balanced_top15": [
            [k[0], k[1], per_cell_totals[k]["N-balanced (1/n_l)"]]
            for k in scheme_rankings["N-balanced (1/n_l)"][:15]
        ],
        "sensitivity_matrix": {
            scheme: {
                "top9_retained": len(set(rl[:9]) & default_top9),
                "top15_retained": len(set(rl[:15]) & default_top15),
                "top15_jaccard": len(set(rl[:15]) & default_top15)
                / len(set(rl[:15]) | default_top15),
            }
            for scheme, rl in scheme_rankings.items()
        },
    }
    out_path = Path("scripts/analyses/results/b13_nbalanced.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
