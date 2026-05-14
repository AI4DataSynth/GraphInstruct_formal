"""B1.4 — Cost-adjusted method comparison.

For each target model in the method experiments (GPT-4o-mini, DeepSeek-V3,
Qwen3.5-35B), compute for ZS / Oracle / VGIG / CAAP / Combined / retry / SC:
- Q (total quality)
- TPV (mean tokens per valid graph)
- mean API calls
- D5 (efficiency score)
- Sfin (with Pareto bonus, if applicable)
- Q/kTPV (scale-free efficiency)
- API multiplier vs ZS

Key narrative: Combined achieves +0.035-0.050 over Oracle but at ~10-15× cost.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import (
    PARETO_LAMBDA,
    compute_tpv_api,
    d5_score,
    find_pareto_front,
    load_quality,
    q_per_ktpv,
    sfin,
)

TARGETS = [
    ("gpt4omini", "GPT-4o-mini", "T3"),
    ("deepseekaiDeepSeekV3", "DeepSeek-V3", "T2"),
    ("qwen3535ba3b", "Qwen3.5-35B", "T2"),
]

METHOD_CONDITIONS = [
    ("zero-shot", "ZS"),
    ("oracle", "Oracle"),
    ("retry", "retry"),
    ("sc", "SC"),
    ("caap", "CAAP"),
    ("vgig", "VGIG"),
    ("combined", "Combined"),
]


def gather_for_model(model: str) -> list[dict]:
    """For one target model, build per-condition cost-quality stats."""
    rows = []
    for strat, label in METHOD_CONDITIONS:
        qpath = Path(f"results/{model}-{strat}.quality.json")
        jpath = Path(f"results/{model}-{strat}.jsonl")
        if not qpath.exists() or not jpath.exists():
            print(f"  [skip] {model}-{strat}: missing file")
            continue
        q = load_quality(qpath)
        tpv, api, n_valid = compute_tpv_api(jpath)
        Q = q.get("total_score", 0.0)
        d5 = d5_score(tpv, api)
        qk = q_per_ktpv(Q, tpv) if tpv != float("inf") else 0.0
        rows.append(
            {
                "model": model,
                "condition": strat,
                "label": label,
                "Q": Q,
                "TPV": tpv,
                "API": api,
                "n_valid": n_valid,
                "D5": d5,
                "Q_per_kTPV": qk,
            }
        )
    return rows


def main() -> None:
    print("# B1.4 — Cost-adjusted method comparison\n")
    all_rows = []
    for model, display, tier in TARGETS:
        print(f"\n## {display} ({tier})\n")
        rows = gather_for_model(model)
        if not rows:
            continue
        for r in rows:
            r["display"] = display
            r["tier"] = tier

        # API multiplier and TPV multiplier vs ZS
        zs_row = next((r for r in rows if r["condition"] == "zero-shot"), None)
        if zs_row:
            zs_tpv = zs_row["TPV"]
            zs_api = zs_row["API"]
            for r in rows:
                r["TPV_mult_vs_ZS"] = (r["TPV"] / zs_tpv) if zs_tpv > 0 else 0
                r["API_mult_vs_ZS"] = (r["API"] / zs_api) if zs_api > 0 else 0

        # Pareto over the conditions within this model: (TPV, Q)
        pts = [
            (r["TPV"], r["Q"], r["condition"]) for r in rows if r["TPV"] != float("inf")
        ]
        pareto_keys = find_pareto_front(pts)
        for r in rows:
            r["is_pareto_within_model"] = r["condition"] in pareto_keys
            r["Sfin"] = sfin(r["Q"], r["is_pareto_within_model"])

        # Print table
        print(
            "| Condition | Q     | ΔQ vs ZS | ΔQ vs Oracle | TPV   | TPV× vs ZS | API   | API× vs ZS | D5    | Sfin  | Q/kTPV |"
        )
        print(
            "|-----------|-------|----------|--------------|-------|------------|-------|------------|-------|-------|--------|"
        )
        oracle_q = next((r["Q"] for r in rows if r["condition"] == "oracle"), None)
        for r in rows:
            dq_zs = r["Q"] - zs_row["Q"] if zs_row else 0
            dq_orc = (r["Q"] - oracle_q) if oracle_q is not None else 0
            print(
                f"| {r['label']:9s} | {r['Q']:.3f} | {dq_zs:+.3f}   | {dq_orc:+.3f}       "
                f"| {r['TPV']:5.0f} | {r['TPV_mult_vs_ZS']:5.1f}×     | {r['API']:.2f}  "
                f"| {r['API_mult_vs_ZS']:5.2f}×     | {r['D5']:.3f} | {r['Sfin']:.3f} "
                f"| {r['Q_per_kTPV']:.2f}   |"
            )
        all_rows.extend(rows)

    # Summary insights
    print("\n## Key narratives\n")
    for model, display, tier in TARGETS:
        rows = [r for r in all_rows if r["model"] == model]
        if not rows:
            continue
        zs = next((r for r in rows if r["condition"] == "zero-shot"), None)
        oracle = next((r for r in rows if r["condition"] == "oracle"), None)
        comb = next((r for r in rows if r["condition"] == "combined"), None)
        if not (zs and oracle and comb):
            continue
        print(
            f"- **{display}**: Combined achieves Q={comb['Q']:.3f} "
            f"(+{comb['Q']-oracle['Q']:+.3f} over Oracle) at "
            f"TPV={comb['TPV']:.0f} ({comb['TPV_mult_vs_ZS']:.1f}× ZS), "
            f"API={comb['API']:.1f} ({comb['API_mult_vs_ZS']:.1f}× ZS), "
            f"Sfin={comb['Sfin']:.3f}, Q/kTPV={comb['Q_per_kTPV']:.2f}."
        )
        print(
            f"  - ZS  baseline:  Q={zs['Q']:.3f}, TPV={zs['TPV']:.0f}, "
            f"Sfin={zs['Sfin']:.3f}, Q/kTPV={zs['Q_per_kTPV']:.2f}"
        )
        print(
            f"  - Oracle:        Q={oracle['Q']:.3f}, TPV={oracle['TPV']:.0f}, "
            f"Sfin={oracle['Sfin']:.3f}, Q/kTPV={oracle['Q_per_kTPV']:.2f}"
        )
        # Is Combined Pareto-dominant?
        comb_q_per_k = comb["Q_per_kTPV"]
        zs_q_per_k = zs["Q_per_kTPV"]
        verdict = "lower" if comb_q_per_k < zs_q_per_k else "higher"
        ratio = comb_q_per_k / zs_q_per_k if zs_q_per_k > 0 else 0
        print(
            f"  - **Combined Q/kTPV is {ratio:.2f}× ZS Q/kTPV** "
            f"({'Combined is quality-prioritized, not free' if ratio < 1 else 'Combined Pareto-dominates ZS'})\n"
        )

    # Persist results
    out_path = Path("scripts/analyses/results/b14_sfin.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
