"""B1.5 — Failure and truncation rates by (model, strategy, level).

Scans all baseline JSONL files in `results/`, counts:
- parse_fail % : fraction of samples with valid=False
- trunc % : fraction where output_tokens >= max_tokens × 0.99
- total_generations : sum over samples

Also reports the exact total generation count (180,000 nominal vs actual)
to address the 74-generation gap for B2.10.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Final

# Model max-output-tokens map from the GraphInstruct paper App. F
MAX_TOKENS: Final[dict[str, int]] = {
    "gpt35turbo": 4096,
}
DEFAULT_MAX_TOKENS: Final[int] = 16384

# Baseline strategy patterns
BASELINE_STRATS: Final[set[str]] = {"zero-shot", "few-shot", "zero-cot", "few-cot"}


def parse_filename(stem: str) -> tuple[str, str] | None:
    """Extract (model, strategy) from a results filename stem.

    Examples:
        gpt4omini-zero-shot          -> ("gpt4omini", "zero-shot")
        claudesonnet46-few-cot       -> ("claudesonnet46", "few-cot")
        deepseekaiDeepSeekV3-zero-shot -> ("deepseekaiDeepSeekV3", "zero-shot")

    Returns None if the strategy is not a baseline strategy.
    """
    parts = stem.split("-")
    for i in range(len(parts) - 1, 0, -1):
        cand = "-".join(parts[i:])
        if cand in BASELINE_STRATS:
            model = "-".join(parts[:i])
            return model, cand
    return None


def model_max_tokens(model: str) -> int:
    """Return the max_tokens setting for a model."""
    for k, v in MAX_TOKENS.items():
        if k in model.lower():
            return v
    return DEFAULT_MAX_TOKENS


def scan_jsonl(path: Path) -> dict[str, dict[int, dict[str, int]]]:
    """Scan one jsonl file, returning per-level stats.

    Returns dict mapping level -> {total, parse_fail, truncated}.
    """
    per_level: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "parse_fail": 0, "truncated": 0}
    )

    parsed = parse_filename(path.stem)
    if parsed is None:
        return {}
    model, _ = parsed
    max_tok = model_max_tokens(model)
    trunc_threshold = int(max_tok * 0.99)

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            level = rec.get("level", -1)
            for s in rec.get("samples", []):
                per_level[level]["total"] += 1
                if not s.get("valid", True):
                    per_level[level]["parse_fail"] += 1
                if s.get("output_tokens", 0) >= trunc_threshold:
                    per_level[level]["truncated"] += 1

    return dict(per_level)


def main() -> None:
    results_dir = Path("results")
    files = sorted(results_dir.glob("*.jsonl"))

    by_cell: dict[tuple[str, str], dict[int, dict[str, int]]] = {}
    for f in files:
        parsed = parse_filename(f.stem)
        if parsed is None:
            continue
        model, strat = parsed
        by_cell[(model, strat)] = scan_jsonl(f)

    # Aggregate global totals
    total_gens = sum(
        per_lvl["total"] for cell in by_cell.values() for per_lvl in cell.values()
    )
    total_fail = sum(
        per_lvl["parse_fail"] for cell in by_cell.values() for per_lvl in cell.values()
    )
    total_trunc = sum(
        per_lvl["truncated"] for cell in by_cell.values() for per_lvl in cell.values()
    )

    print(f"# Total baseline cells scanned: {len(by_cell)}")
    print(f"# Total generations (sum samples): {total_gens:,}")
    print(f"#   nominal (45 × 800 × 5):       180,000")
    print(f"#   gap:                          {180_000 - total_gens}")
    print(
        f"# Total parse failures:           {total_fail:,} "
        f"({100*total_fail/total_gens:.2f}%)"
    )
    print(
        f"# Total truncations (≥99% max):   {total_trunc:,} "
        f"({100*total_trunc/total_gens:.2f}%)"
    )
    print()

    # Top-K cells with highest parse_fail rate by level
    print("## Top-15 cells by parse_fail rate (per-level)")
    print()
    print("| Model | Strategy | Level | n_gen | parse_fail % | trunc % |")
    print("|-------|----------|-------|-------|--------------|---------|")
    rows = []
    for (model, strat), lvls in by_cell.items():
        for lvl, st in lvls.items():
            if st["total"] == 0:
                continue
            fail_pct = 100 * st["parse_fail"] / st["total"]
            trunc_pct = 100 * st["truncated"] / st["total"]
            rows.append((model, strat, lvl, st["total"], fail_pct, trunc_pct))
    rows.sort(key=lambda r: r[4], reverse=True)
    for r in rows[:15]:
        print(
            f"| {r[0]:20s} | {r[1]:9s} | L{r[2]} | {r[3]} "
            f"| {r[4]:.2f} | {r[5]:.2f} |"
        )
    print()

    # Per-cell summary: how many cells have non-zero failures?
    cells_with_fail = sum(
        1
        for (m, s), lvls in by_cell.items()
        if any(st["parse_fail"] > 0 for st in lvls.values())
    )
    cells_with_trunc = sum(
        1
        for (m, s), lvls in by_cell.items()
        if any(st["truncated"] > 0 for st in lvls.values())
    )
    print(f"# Cells with any parse failure: {cells_with_fail} / {len(by_cell)}")
    print(f"# Cells with any truncation:    {cells_with_trunc} / {len(by_cell)}")
    print()

    # Per-model summary: max truncation rate across strategies / levels
    print("## Per-model max-truncation summary (highlights)")
    print()
    print("| Model | Max trunc % | (at strategy, level) | n_gen affected |")
    print("|-------|-------------|----------------------|----------------|")
    per_model_max = defaultdict(lambda: (0.0, "", -1, 0))
    for (model, strat), lvls in by_cell.items():
        for lvl, st in lvls.items():
            if st["total"] == 0:
                continue
            tr = 100 * st["truncated"] / st["total"]
            if tr > per_model_max[model][0]:
                per_model_max[model] = (tr, strat, lvl, st["truncated"])
    for model, (tr, strat, lvl, n) in sorted(
        per_model_max.items(), key=lambda kv: kv[1][0], reverse=True
    ):
        if tr > 0.5:  # Only show non-trivial
            print(f"| {model:20s} | {tr:.2f}% | ({strat}, L{lvl}) | {n} |")

    # Save to JSON for SUPP_ANALYSES.md integration
    out = {
        "global": {
            "total_cells": len(by_cell),
            "total_generations": total_gens,
            "nominal_generations": 180_000,
            "gap": 180_000 - total_gens,
            "total_parse_failures": total_fail,
            "total_truncations": total_trunc,
        },
        "per_cell": [
            {
                "model": m,
                "strategy": s,
                "level": lvl,
                "n_gen": st["total"],
                "parse_fail_pct": (
                    100 * st["parse_fail"] / st["total"] if st["total"] else 0.0
                ),
                "trunc_pct": (
                    100 * st["truncated"] / st["total"] if st["total"] else 0.0
                ),
                "n_parse_fail": st["parse_fail"],
                "n_trunc": st["truncated"],
            }
            for (m, s), lvls in by_cell.items()
            for lvl, st in lvls.items()
        ],
    }
    out_path = Path("scripts/analyses/results/b15_failure_rates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
