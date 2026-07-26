"""A1B — cell-level (model x strategy) leaderboard weight sensitivity.

Complements A1 (model-level, best-baseline leaderboard) with the cell-level
view: every (model, baseline-strategy) run is a leaderboard entry, and the
level-weight vector is perturbed with the sample-size-motivated schemes
(N-balanced 1/n_l, uniform, mass-balanced n_l) plus an L5-zero ablation
(L5 weight set to 0, L0-L4 renormalised).

The universe is frozen to the 12 baseline models' baseline-strategy cells
(45 cells: 11 fully-evaluated models x 4 strategies + Sonnet-4 zero-shot),
matching the released supplement's Table S7 analysis. It deliberately does
NOT honour REBUTTAL_MODELSET=ext16 -- the point is to back the published
cell-level numbers with full detail (exact dropped/entered cells, rank
shifts, Spearman/Kendall), which the original run printed but never
persisted.

Run from the repository root:
    python scripts/rebuttal/a1b_cell_weight_sensitivity.py
Writes rebuttal_analyses/results/a1b_cell_weight_sensitivity.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analyses"))

from _common import (  # noqa: E402
    LEVEL_WEIGHTS,
    MODEL_DISPLAY,
    compute_total_score,
    load_quality,
    per_level_scores,
    scan_results_dir,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.common import (  # noqa: E402
    BASELINE_MODELS,
    MODEL_NAMES,
    best_baseline_runs,
    kendall_tau_b,
    rank_desc,
    rank_positions,
    spearman,
    top_overlap,
)

try:
    from scipy.stats import kendalltau, spearmanr

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a project dependency
    HAVE_SCIPY = False

BASELINE_12 = frozenset(
    {
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
    }
)

LEVEL_N = (100, 200, 200, 150, 100, 50)

SCHEMES: dict[str, tuple[float, ...]] = {
    "default": tuple(LEVEL_WEIGHTS),
    "N-balanced (1/n_l)": tuple((1 / n) / sum(1 / m for m in LEVEL_N) for n in LEVEL_N),
    "Uniform (1/6)": (1 / 6,) * 6,
    "Mass-balanced (n_l)": tuple(n / sum(LEVEL_N) for n in LEVEL_N),
    "L5-zero (renorm)": (
        0.05 / 0.75,
        0.10 / 0.75,
        0.15 / 0.75,
        0.20 / 0.75,
        0.25 / 0.75,
        0.0,
    ),
}

Cell = tuple[str, str]


def disp(key: Cell) -> str:
    model, strat = key
    return f"{MODEL_DISPLAY.get(model, model)} {strat}"


def cell_moves(
    cells: list[Cell],
    d_rank_all: dict[Cell, int],
    r_all: dict[Cell, int],
) -> list[dict[str, Any]]:
    return [
        {
            "cell": list(k),
            "display": disp(k),
            "default_rank": d_rank_all[k],
            "new_rank": r_all[k],
        }
        for k in cells
    ]


def main() -> None:
    cells = {
        k: p
        for k, p in scan_results_dir(ROOT / "results", only_baselines=True).items()
        if k[0] in BASELINE_12
    }
    print(f"cells (12-model universe) = {len(cells)}")

    pls_cache = {k: per_level_scores(load_quality(p)) for k, p in cells.items()}
    totals = {
        name: {k: compute_total_score(pls, w) for k, pls in pls_cache.items()}
        for name, w in SCHEMES.items()
    }
    rankings = {
        name: [k for k, _ in sorted(t.items(), key=lambda kv: kv[1], reverse=True)]
        for name, t in totals.items()
    }

    d_rank_all = {k: i + 1 for i, k in enumerate(rankings["default"])}
    d_top9 = set(rankings["default"][:9])
    d_top15 = set(rankings["default"][:15])
    d_rank15 = {k: i + 1 for i, k in enumerate(rankings["default"][:15])}
    cell_order = rankings["default"]

    per_scheme: dict[str, Any] = {}
    for name in SCHEMES:
        rl = rankings[name]
        r_all = {k: i + 1 for i, k in enumerate(rl)}
        top9, top15 = set(rl[:9]), set(rl[:15])
        n9 = len(top9 & d_top9)
        n15 = len(top15 & d_top15)
        jac = len(top15 & d_top15) / len(top15 | d_top15)
        max_shift = max(abs(r_all[k] - d_rank15[k]) for k in d_rank15)
        dropped9 = sorted(d_top9 - top9, key=lambda k: d_rank_all[k])
        entered9 = sorted(top9 - d_top9, key=lambda k: r_all[k])
        dropped15 = sorted(d_top15 - top15, key=lambda k: d_rank_all[k])
        entered15 = sorted(top15 - d_top15, key=lambda k: r_all[k])

        if HAVE_SCIPY:
            v_def = [totals["default"][k] for k in cell_order]
            v_alt = [totals[name][k] for k in cell_order]
            rho = float(spearmanr(v_def, v_alt).statistic)
            tau = float(kendalltau(v_def, v_alt).statistic)
        else:
            rho = tau = float("nan")

        print(
            f"{name:22s} top9 {n9}/9  top15 {n15}/15  jac {jac:.3f}  "
            f"shift {max_shift}  rho {rho:.4f}  tau {tau:.4f}"
        )

        per_scheme[name] = {
            "top9_retained": n9,
            "top15_retained": n15,
            "top15_jaccard": round(jac, 4),
            "max_rank_shift_top15": max_shift,
            "spearman_rho_45cells": round(rho, 4),
            "kendall_tau_45cells": round(tau, 4),
            "top15_ranking": [[k[0], k[1], round(totals[name][k], 4)] for k in rl[:15]],
            "top9_dropped": cell_moves(dropped9, d_rank_all, r_all),
            "top9_entered": cell_moves(entered9, d_rank_all, r_all),
            "top15_dropped": cell_moves(dropped15, d_rank_all, r_all),
            "top15_entered": cell_moves(entered15, d_rank_all, r_all),
        }

    # Model-level L5-zero: identical construction to A1 (best baseline
    # strategy selected separately under each weight vector).
    ml_totals: dict[str, dict[str, float]] = {}
    ml_order: dict[str, list[str]] = {}
    for name in ("default", "L5-zero (renorm)"):
        selected = best_baseline_runs(list(SCHEMES[name]))
        ml_totals[name] = {m: float(selected[m]["total"]) for m in BASELINE_MODELS}
        ml_order[name] = rank_desc(ml_totals[name])
    d_pos = rank_positions(ml_totals["default"])
    z_pos = rank_positions(ml_totals["L5-zero (renorm)"])
    model_level: dict[str, Any] = {
        "construction": "best baseline strategy per model under each scheme",
        "default_ranking": [
            [m, MODEL_NAMES[m], round(ml_totals["default"][m], 4)]
            for m in ml_order["default"]
        ],
        "l5_zero_ranking": [
            [m, MODEL_NAMES[m], round(ml_totals["L5-zero (renorm)"][m], 4)]
            for m in ml_order["L5-zero (renorm)"]
        ],
        "vs_default": {
            "spearman_rho": spearman(
                ml_totals["default"], ml_totals["L5-zero (renorm)"]
            ),
            "kendall_tau_b": kendall_tau_b(
                ml_totals["default"], ml_totals["L5-zero (renorm)"]
            ),
            "top_retention": [
                top_overlap(ml_order["default"], ml_order["L5-zero (renorm)"], k)
                for k in (5, 9, 12)
            ],
            "max_rank_displacement": max(
                abs(d_pos[m] - z_pos[m]) for m in BASELINE_MODELS
            ),
            "rank_moves": [
                {
                    "model": MODEL_NAMES[m],
                    "default_rank": d_pos[m],
                    "l5_zero_rank": z_pos[m],
                }
                for m in ml_order["default"]
                if d_pos[m] != z_pos[m]
            ],
        },
    }
    print(
        "model-level L5-zero: "
        f"top5 {model_level['vs_default']['top_retention'][0]}  "
        f"top9 {model_level['vs_default']['top_retention'][1]}  "
        f"rho {model_level['vs_default']['spearman_rho']:.4f}  "
        f"tau {model_level['vs_default']['kendall_tau_b']:.4f}  "
        f"max_shift {model_level['vs_default']['max_rank_displacement']}"
    )

    out = {
        "universe": (
            "12 baseline models x baseline strategies (45 cells; frozen, no ext16)"
        ),
        "n_cells": len(cells),
        "schemes": {n: list(w) for n, w in SCHEMES.items()},
        "default_ranking_top20": [
            [k[0], k[1], round(totals["default"][k], 4)]
            for k in rankings["default"][:20]
        ],
        "per_scheme": per_scheme,
        "model_level_l5_zero": model_level,
    }
    out_path = ROOT / "rebuttal_analyses" / "results" / "a1b_cell_weight_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
