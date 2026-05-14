"""D5 robustness: recompute Quality rankings under perturbed exponential scales.

For each (s_T, s_A) in a 3x3 grid, recompute D5 = 0.7*exp(-TPV/s_T) + 0.3*exp(-(API-1)/s_A)
per instruction, propagate through level scoring, and rank (model, strategy) cells.
Report Spearman rho vs the default (1000, 2) ranking.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVEL_DIM_W: dict[int, dict[str, float]] = {
    0: {"D1": 0.10, "D2": 0.0, "D3": 0.0, "D4": 0.60, "D5": 0.30},
    1: {"D1": 0.15, "D2": 0.0, "D3": 0.0, "D4": 0.70, "D5": 0.15},
    2: {"D1": 0.15, "D2": 0.0, "D3": 0.0, "D4": 0.70, "D5": 0.15},
    3: {"D1": 0.15, "D2": 0.0, "D3": 0.15, "D4": 0.50, "D5": 0.20},
    4: {"D1": 0.10, "D2": 0.15, "D3": 0.05, "D4": 0.55, "D5": 0.15},
    5: {"D1": 0.15, "D2": 0.0, "D3": 0.15, "D4": 0.50, "D5": 0.20},
}
LEVEL_W = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]


def load_api_per_inst(jsonl_path: str) -> dict[str, float]:
    """Mean api_calls per sample, per instruction (1.0 if file missing)."""
    api_per_inst: dict[str, float] = {}
    if not os.path.exists(jsonl_path):
        return api_per_inst
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples = rec.get("samples", [])
            if not samples:
                continue
            total_api = sum(s.get("api_calls", 1) for s in samples)
            api_per_inst[rec["instruction_id"]] = total_api / len(samples)
    return api_per_inst


def d5_score(tpv: float | None, api: float | None, s_t: float, s_a: float) -> float:
    if tpv is None or api is None:
        return 0.0
    if tpv == float("inf") or api == float("inf"):
        return 0.0
    if tpv <= 0 or api < 1:
        return 0.0
    return 0.7 * math.exp(-tpv / s_t) + 0.3 * math.exp(-(api - 1) / s_a)


def total_quality(quality_path: str, s_t: float, s_a: float) -> float:
    with open(quality_path, encoding="utf-8") as f:
        q = json.load(f)
    jsonl_path = quality_path.replace(".quality.json", ".jsonl")
    apis = load_api_per_inst(jsonl_path)
    per_level: dict[int, list[float]] = defaultdict(list)
    for inst in q["per_instruction"]:
        L = inst["level"]
        tpv = inst["d5_tokens_per_valid"]
        api = apis.get(inst["instruction_id"], 1.0)
        new_d5 = d5_score(tpv, api, s_t, s_a)
        ds = inst["dimension_scores"]
        w = LEVEL_DIM_W[L]
        ls = (
            w["D1"] * ds["D1"]
            + w["D2"] * ds.get("D2", 0.0)
            + w["D3"] * ds.get("D3", 0.0)
            + w["D4"] * ds["D4"]
            + w["D5"] * new_d5
        )
        per_level[L].append(ls)
    total = 0.0
    for L in range(6):
        if per_level[L]:
            total += LEVEL_W[L] * (sum(per_level[L]) / len(per_level[L]))
    return total


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
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


def main() -> None:
    # Prefer the curated layout `results/quality/*.quality.json`; fall back to
    # the legacy flat layout `results/*.quality.json` for backward compatibility.
    files = sorted(
        glob.glob(os.path.join(ROOT, "results", "quality", "*.quality.json"))
    )
    if not files:
        files = sorted(glob.glob(os.path.join(ROOT, "results", "*.quality.json")))
    baseline_strategies = ("-zero-shot", "-few-shot", "-zero-cot", "-few-cot")
    runs: list[str] = []
    for fp in files:
        name = os.path.basename(fp).replace(".quality.json", "")
        if not any(name.endswith(s) for s in baseline_strategies):
            continue
        runs.append(fp)
    print(f"# Robustness over {len(runs)} (model, strategy) baseline cells")
    print(f"# Default exponential scales: TPV/1000, (API-1)/2")

    grid_t = [500, 1000, 2000]
    grid_a = [1, 2, 4]
    cell_q: dict[tuple[float, float], dict[str, float]] = {}
    for s_t, s_a in product(grid_t, grid_a):
        q_dict = {}
        for fp in runs:
            name = os.path.basename(fp).replace(".quality.json", "")
            q_dict[name] = total_quality(fp, s_t, s_a)
        cell_q[(s_t, s_a)] = q_dict

    default = cell_q[(1000, 2)]
    print()
    print(
        f"{'s_T':>6} {'s_A':>6}  {'Spearman':>10}  {'Kendall':>10}  {'Top-1 stable':>14}  {'Top-5 jaccard':>14}"
    )
    rho_grid = []
    default_top1 = max(default, key=default.get)
    default_top5 = set(sorted(default, key=default.get, reverse=True)[:5])
    for s_t, s_a in product(grid_t, grid_a):
        q = cell_q[(s_t, s_a)]
        rho = spearman(default, q)
        rho_grid.append(rho)
        top1_stable = "yes" if max(q, key=q.get) == default_top1 else "no"
        top5 = set(sorted(q, key=q.get, reverse=True)[:5])
        jacc = len(top5 & default_top5) / len(top5 | default_top5)
        kendall_pairs = 0
        concordant = 0
        names = list(q.keys())
        for i, x in enumerate(names):
            for y in names[i + 1 :]:
                kendall_pairs += 1
                if (default[x] - default[y]) * (q[x] - q[y]) > 0:
                    concordant += 1
        kendall = (
            (2 * concordant - kendall_pairs) / kendall_pairs if kendall_pairs else 1.0
        )
        print(
            f"{s_t:>6} {s_a:>6}  {rho:>10.4f}  {kendall:>10.4f}  {top1_stable:>14}  {jacc:>14.3f}"
        )
    print()
    print(f"# Min Spearman across grid: {min(rho_grid):.4f}")
    print(f"# Max Spearman across grid: {max(rho_grid):.4f}")
    print(f"# Default top-1: {default_top1} (Q={default[default_top1]:.4f})")
    print(f"# Default top-5: {sorted(default, key=default.get, reverse=True)[:5]}")


if __name__ == "__main__":
    main()
