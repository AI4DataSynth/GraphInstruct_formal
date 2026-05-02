"""Aggregate the 45 quality.json files in results/quality/ into 3 CSV
leaderboards that map directly to paper Tables 1, 2, 3.

Run from the repo root:
    python scripts/_build_leaderboards.py

The underscore prefix marks this as a one-shot build helper, not a
reviewer-facing CLI tool. After it produces the 3 CSVs, you can delete
this file (or keep it for re-derivation if quality.json files change).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
SRC_QUALITY = WORK / "results" / "quality"
OUT_DIR = WORK / "results" / "leaderboards"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "claudesonnet46": "Sonnet-4.6",
    "claudesonnet420250514": "Sonnet-4",
    "qwen35397ba17b": "Qwen3.5-397B-A17B",
    "qwen35122ba10b": "Qwen3.5-122B-A10B",
    "qwen3535ba3b": "Qwen3.5-35B-A3B",
    "deepseekaiDeepSeekV3": "DeepSeek-V3",
    "gpt41": "GPT-4.1",
    "gpt4o": "GPT-4o",
    "gpt4omini": "GPT-4o-mini",
    "gpt35turbo": "GPT-3.5-turbo",
    "metallamaLlama3370BInstructTurbo": "Llama-3.3-70B",
    "metallamaMetaLlama318BInstruct": "Llama-3.1-8B",
}
STRATEGY_SHORT = {
    "zero-shot": "ZS",
    "few-shot": "FS",
    "zero-cot": "ZC",
    "few-cot": "FC",
}
LEVEL_DIM_W = {
    0: {"D1": 0.10, "D2": 0.0, "D3": 0.0, "D4": 0.60, "D5": 0.30},
    1: {"D1": 0.15, "D2": 0.0, "D3": 0.0, "D4": 0.70, "D5": 0.15},
    2: {"D1": 0.15, "D2": 0.0, "D3": 0.0, "D4": 0.70, "D5": 0.15},
    3: {"D1": 0.15, "D2": 0.0, "D3": 0.15, "D4": 0.50, "D5": 0.20},
    4: {"D1": 0.10, "D2": 0.15, "D3": 0.05, "D4": 0.55, "D5": 0.15},
    5: {"D1": 0.15, "D2": 0.0, "D3": 0.15, "D4": 0.50, "D5": 0.20},
}
LEVEL_W = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]


def parse_filename(stem: str):
    for strat in ("few-cot", "few-shot", "zero-cot", "zero-shot"):
        if stem.endswith("-" + strat):
            model_key = stem[: -len("-" + strat)]
            return MODEL_NAMES.get(model_key, model_key), strat
    return None, None


def load_cells():
    cells = []
    for fp in sorted(SRC_QUALITY.glob("*.quality.json")):
        stem = fp.stem.replace(".quality", "")
        model, strategy = parse_filename(stem)
        if not model:
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        pls = data.get("per_level_scores", {})
        pi = data.get("per_instruction", [])
        tpv_values = [
            r.get("d5_tokens_per_valid", 0) or 0
            for r in pi
            if (r.get("d5_tokens_per_valid") or 0) > 0
        ]
        tpv = sum(tpv_values) / max(1, len(tpv_values))
        cells.append(
            {
                "stem": stem,
                "model": model,
                "strategy": strategy,
                "strategy_short": STRATEGY_SHORT.get(strategy, strategy),
                "L0": pls.get("0", 0),
                "L1": pls.get("1", 0),
                "L2": pls.get("2", 0),
                "L3": pls.get("3", 0),
                "L4": pls.get("4", 0),
                "L5": pls.get("5", 0),
                "Total": data.get("total_score", 0),
                "Final": data.get("final_score", 0),
                "TPV": tpv,
                "_path": fp,
            }
        )
    return cells


def write_tab1(cells):
    sorted_q = sorted(cells, key=lambda c: c["Total"], reverse=True)
    out = OUT_DIR / "tab1_quality_top15.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Rank",
                "Model",
                "Strategy",
                "Quality (Total)",
                "Final (Pareto-adj)",
                "L0",
                "L1",
                "L2",
                "L3",
                "L4",
                "L5",
                "TPV",
            ]
        )
        for i, c in enumerate(sorted_q[:15], start=1):
            w.writerow(
                [
                    i,
                    c["model"],
                    c["strategy_short"],
                    f'{c["Total"]:.4f}',
                    f'{c["Final"]:.4f}',
                    f'{c["L0"]:.4f}',
                    f'{c["L1"]:.4f}',
                    f'{c["L2"]:.4f}',
                    f'{c["L3"]:.4f}',
                    f'{c["L4"]:.4f}',
                    f'{c["L5"]:.4f}',
                    f'{c["TPV"]:.0f}',
                ]
            )
    print(
        f"  Tab 1: {out.relative_to(WORK)}  top-1 = {sorted_q[0]['model']} {sorted_q[0]['strategy_short']} Q={sorted_q[0]['Total']:.4f}"
    )


def write_tab2(cells):
    sorted_f = sorted(cells, key=lambda c: c["Final"], reverse=True)
    out = OUT_DIR / "tab2_pareto_efficiency.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["Sfin Rank", "Model", "Strategy", "Final", "Quality", "TPV", "Q/kTPV"]
        )
        for i, c in enumerate(sorted_f[:20], start=1):
            q_per_ktpv = c["Total"] / (c["TPV"] / 1000) if c["TPV"] > 0 else 0
            w.writerow(
                [
                    i,
                    c["model"],
                    c["strategy_short"],
                    f'{c["Final"]:.4f}',
                    f'{c["Total"]:.4f}',
                    f'{c["TPV"]:.0f}',
                    f"{q_per_ktpv:.3f}",
                ]
            )
    print(
        f"  Tab 2: {out.relative_to(WORK)}  Sfin top-1 = {sorted_f[0]['model']} {sorted_f[0]['strategy_short']} Sfin={sorted_f[0]['Final']:.4f}"
    )


def d5_score(tpv, api, s_t, s_a):
    if tpv <= 0 or api < 1:
        return 0.0
    return 0.7 * math.exp(-tpv / s_t) + 0.3 * math.exp(-max(0, api - 1) / s_a)


def total_q_with_perturbed_d5(quality_json_path, s_t, s_a):
    with open(quality_json_path, encoding="utf-8") as f:
        q = json.load(f)
    per_level = defaultdict(list)
    for inst in q["per_instruction"]:
        L = inst["level"]
        tpv = inst.get("d5_tokens_per_valid") or 0
        api = 1.0
        new_d5 = d5_score(tpv, api, s_t, s_a)
        ds = inst["dimension_scores"]
        wts = LEVEL_DIM_W[L]
        ls = (
            wts["D1"] * ds["D1"]
            + wts["D2"] * ds.get("D2", 0)
            + wts["D3"] * ds.get("D3", 0)
            + wts["D4"] * ds["D4"]
            + wts["D5"] * new_d5
        )
        per_level[L].append(ls)
    return sum(
        LEVEL_W[L] * (sum(per_level[L]) / max(1, len(per_level[L])))
        for L in range(6)
        if per_level[L]
    )


def spearman(a, b):
    keys = list(a.keys())
    ra = sorted(keys, key=lambda k: -a[k])
    rb = sorted(keys, key=lambda k: -b[k])
    rank_a = {k: i for i, k in enumerate(ra)}
    rank_b = {k: i for i, k in enumerate(rb)}
    n = len(keys)
    if n < 2:
        return 1.0
    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def write_tab3(cells):
    grid = [(s_t, s_a) for s_t in (500, 1000, 2000) for s_a in (1, 2, 4)]
    qmap = {g: {} for g in grid}
    for c in cells:
        for g in grid:
            qmap[g][c["stem"]] = total_q_with_perturbed_d5(c["_path"], *g)
    default = qmap[(1000, 2)]
    default_top1 = max(default, key=default.get)
    default_top5 = set(sorted(default, key=default.get, reverse=True)[:5])

    out = OUT_DIR / "tab3_d5_robustness.csv"
    rho_min = 1.0
    rho_max = 0.0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "s_T (token scale)",
                "s_A (API scale)",
                "Spearman rho",
                "Kendall tau",
                "Top-1 stable",
                "Top-5 Jaccard",
            ]
        )
        for s_t, s_a in grid:
            q = qmap[(s_t, s_a)]
            rho = spearman(default, q)
            rho_min = min(rho_min, rho)
            rho_max = max(rho_max, rho)
            names = list(q.keys())
            pairs = concord = 0
            for i, x in enumerate(names):
                for y in names[i + 1 :]:
                    pairs += 1
                    if (default[x] - default[y]) * (q[x] - q[y]) > 0:
                        concord += 1
            kendall = (2 * concord - pairs) / pairs if pairs else 1.0
            top1_ok = "yes" if max(q, key=q.get) == default_top1 else "no"
            top5 = set(sorted(q, key=q.get, reverse=True)[:5])
            jacc = len(top5 & default_top5) / len(top5 | default_top5)
            w.writerow(
                [s_t, s_a, f"{rho:.4f}", f"{kendall:.4f}", top1_ok, f"{jacc:.3f}"]
            )
    print(
        f"  Tab 3: {out.relative_to(WORK)}  rho range = [{rho_min:.4f}, {rho_max:.4f}]"
    )


def main():
    cells = load_cells()
    print(f"Loaded {len(cells)} baseline cells from results/quality/")
    write_tab1(cells)
    write_tab2(cells)
    write_tab3(cells)
    print("\nAll 3 leaderboards generated.")


if __name__ == "__main__":
    main()
