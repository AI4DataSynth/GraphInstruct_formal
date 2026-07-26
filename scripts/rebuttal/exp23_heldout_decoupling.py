"""EXP23: held-out exemplar anti-leakage (EXP-2) + decoupled-verifier (EXP-3) summary.

Reproduces the two established tables from current quality.json files so the numbers
stay consistent after re-eval. Model-set independent (fixed model lists) -> run once,
writes rebuttal_analysis/EXP23_heldout_decoupling.json.

EXP-2 (held-out): quality-only total; delta = heldout(ref1) - original(ref0).
EXP-3 (decoupled): best-baseline quality-only total vs vgig-structural; delta = struct - BL.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "rebuttal_analysis" / "EXP23_heldout_decoupling.json"

BASELINE_STRATEGIES = ("zero-shot", "few-shot", "zero-cot", "few-cot")

# EXP-2: (slug, display, strategy) whose {slug}-{strategy}-heldout.quality.json exists.
HELDOUT_TARGETS = [
    ("deepseekaiDeepSeekV3", "DeepSeek-V3", "few-shot"),
    ("deepseekaiDeepSeekV3", "DeepSeek-V3", "few-cot"),
    ("gpt4omini", "GPT-4o-mini", "few-shot"),
    ("gpt4omini", "GPT-4o-mini", "few-cot"),
    ("qwen3535ba3b", "Qwen3.5-35B-A3B", "few-cot"),
]

# EXP-3: decoupled verifier. vgig-structural slug casing differs per model (real filenames).
DECOUPLE_TARGETS = [
    ("gpt4omini", "GPT-4o-mini", "gpt4omini-vgig", "gpt4omini-vgig-structural"),
    (
        "deepseekaiDeepSeekV3",
        "DeepSeek-V3",
        "deepseekaiDeepSeekV3-vgig",
        "deepseekaiDeepSeekV3-vgig-structural",
    ),
    (
        "qwen3535ba3b",
        "Qwen3.5-35B-A3B",
        "qwen3535ba3b-vgig",
        "QwenQwen3535BA3B-vgig-structural",
    ),
]


def _total(stem: str) -> float | None:
    """quality-only total_score from results/{stem}.quality.json (None if missing)."""
    p = RESULTS / f"{stem}.quality.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("total_score")


def _best_baseline_total(slug: str) -> tuple[float | None, str | None]:
    """max quality-only total over the 4 standard strategies -> (total, strategy)."""
    best_t, best_s = None, None
    for s in BASELINE_STRATEGIES:
        t = _total(f"{slug}-{s}")
        if t is not None and (best_t is None or t > best_t):
            best_t, best_s = t, s
    return best_t, best_s


def _round(x, n=4):
    return None if x is None else round(x, n)


def main() -> None:
    # ---- EXP-2 held-out ----
    exp2 = []
    for slug, disp, strat in HELDOUT_TARGETS:
        ref0 = _total(f"{slug}-{strat}")
        ref1 = _total(f"{slug}-{strat}-heldout")
        delta = None if (ref0 is None or ref1 is None) else ref1 - ref0
        exp2.append(
            {
                "model": disp,
                "slug": slug,
                "strategy": strat,
                "ref0_original": _round(ref0),
                "ref1_heldout": _round(ref1),
                "delta": _round(delta),
                "clean": (
                    strat == "few-shot"
                ),  # few-shot (no CoT) = clean leakage test
            }
        )
    few_shot_deltas = [
        r["delta"]
        for r in exp2
        if r["strategy"] == "few-shot" and r["delta"] is not None
    ]
    exp2_summary = {
        "few_shot_mean_delta": _round(sum(few_shot_deltas) / len(few_shot_deltas))
        if few_shot_deltas
        else None,
        "few_shot_all_near_zero": all(abs(d) < 0.01 for d in few_shot_deltas)
        if few_shot_deltas
        else None,
        "n_models_support_heldout": len(
            {r["slug"] for r in exp2 if r["delta"] is not None}
        ),
        "note": "few-shot held-out delta ~0 -> no exemplar-into-context leakage; few-cot small drop (CoT length sensitivity), not leakage.",
    }

    # ---- EXP-3 decoupled verifier ----
    exp3 = []
    for slug, disp, vgig_stem, struct_stem in DECOUPLE_TARGETS:
        bl_t, bl_s = _best_baseline_total(slug)
        vgig_t = _total(vgig_stem)
        struct_t = _total(struct_stem)
        d_struct = None if (struct_t is None or bl_t is None) else struct_t - bl_t
        d_vgig = None if (vgig_t is None or bl_t is None) else vgig_t - bl_t
        exp3.append(
            {
                "model": disp,
                "slug": slug,
                "best_baseline": _round(bl_t),
                "best_baseline_strategy": bl_s,
                "vgig_d4": _round(vgig_t),
                "delta_vgig_vs_bl": _round(d_vgig),
                "vgig_structural": _round(struct_t),
                "delta_struct_vs_bl": _round(d_struct),
            }
        )
    struct_deltas = [
        r["delta_struct_vs_bl"] for r in exp3 if r["delta_struct_vs_bl"] is not None
    ]
    exp3_summary = {
        "struct_mean_delta": _round(sum(struct_deltas) / len(struct_deltas))
        if struct_deltas
        else None,
        "struct_gain_near_zero": all(abs(d) < 0.02 for d in struct_deltas)
        if struct_deltas
        else None,
        "note": "decoupled to pure structural signal -> gain shrinks to ~0; VGIG gain comes from D4 constraint-satisfaction signal (cheap, non-leakage).",
    }

    out = {
        "analysis": "EXP23: held-out exemplar anti-leakage (EXP-2) + decoupled verifier (EXP-3)",
        "metric": "quality-only total_score (D1-D4, D5=0 renormalized)",
        "exp2_heldout": exp2,
        "exp2_summary": exp2_summary,
        "exp3_decoupled": exp3,
        "exp3_summary": exp3_summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EXP23 written -> {OUT}")
    print(
        f"  EXP-2 few-shot mean delta: {exp2_summary['few_shot_mean_delta']} (all near-zero: {exp2_summary['few_shot_all_near_zero']})"
    )
    for r in exp2:
        print(
            f"    {r['model']:16s} {r['strategy']:9s} ref0={r['ref0_original']} ref1={r['ref1_heldout']} delta={r['delta']}"
        )
    print(
        f"  EXP-3 struct mean delta: {exp3_summary['struct_mean_delta']} (near-zero: {exp3_summary['struct_gain_near_zero']})"
    )
    for r in exp3:
        print(
            f"    {r['model']:16s} BL={r['best_baseline']}({r['best_baseline_strategy']}) vgig={r['vgig_d4']} struct={r['vgig_structural']} dStruct={r['delta_struct_vs_bl']}"
        )


if __name__ == "__main__":
    main()
