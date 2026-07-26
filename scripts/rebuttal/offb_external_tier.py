"""OFF-B: external independent-capability anchor tier control (Elo / MMLU-Pro).

Re-tiers models by PUBLISHED external capability scores (LMArena Elo, MMLU-Pro) that are
fully independent of GraphInstruct Q -> anti-circularity control for the L2 discriminative
peak (directly answers the tier-circularity concern). Reuses the C1 continuous-D4 tier-gap
machinery. orig12 reproduces the established ad-hoc product; ext16 adds the 4 new models
where they have reliable public anchors (honest confidence labels; frontier models w/o
clean Elo enter the MMLU-Pro variant only, as the 12-model version already did).

Anchors are hardcoded published values -> zero API / zero model calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    BASELINE_MODELS,
    MODEL_NAMES,
    Q_TIERS,
    best_baseline_runs,
    d4_level_means,
    peak_summary,
    quality_level_means,
    tier_gap_from_level_vectors,
    write_json,
)

# External published capability scores, independent of GraphInstruct Q.
# Elo: ALL 16 from ONE consistent source (Arena.ai Text Overall leaderboard, snapshot 2026-07-21,
#   user-verified 2026-07-26) -> no frontier model excluded (supersedes the earlier metatext.io elo8
#   subset that dropped the 4 frontier models for aggregator inconsistency). MMLU-Pro per-model sources noted.
EXTERNAL_SCORES: dict[str, dict] = {
    "claudesonnet420250514": {
        "elo": 1389,
        "mmlu_pro": 80.0,
        "confidence": "HIGH",
        "src": "Arena Text Overall 1389±4 (2026-07-21); MMLU-Pro 80.0 estimate MED",
    },
    "claudesonnet46": {
        "elo": 1473,
        "mmlu_pro": 78.5,
        "confidence": "HIGH",
        "src": "Arena Text Overall 1473±4 (2026-07-21, clean); MMLU-Pro 78.5 benchlm.ai MED",
    },
    "deepseekaiDeepSeekV3": {
        "elo": 1396,
        "mmlu_pro": 75.9,
        "confidence": "HIGH",
        "src": "Arena Text Overall DeepSeek-V3-0324 1396±4 (2026-07-21); MMLU-Pro tech report 2412.19437",
    },
    "gpt35turbo": {
        "elo": 1224,
        "mmlu_pro": 43.3,
        "confidence": "HIGH",
        "src": "Arena Text Overall GPT-3.5-Turbo-0125 1224±5 (2026-07-21); MMLU-Pro leaderboard historical",
    },
    "gpt41": {
        "elo": 1414,
        "mmlu_pro": 80.5,
        "confidence": "HIGH",
        "src": "Arena Text Overall GPT-4.1-2025-04-14 1414±4 (2026-07-21); MMLU-Pro llm-stats.com",
    },
    "gpt4o": {
        "elo": 1443,
        "mmlu_pro": 72.6,
        "confidence": "HIGH",
        "src": "Arena Text Overall GPT-4o Mar-2025 1443±3 (2026-07-21); MMLU-Pro paper 2406.01574",
    },
    "gpt4omini": {
        "elo": 1318,
        "mmlu_pro": 64.0,
        "confidence": "HIGH",
        "src": "Arena Text Overall GPT-4o-mini 1318±4 (2026-07-21); MMLU-Pro public leaderboard MED",
    },
    "metallamaLlama3370BInstructTurbo": {
        "elo": 1318,
        "mmlu_pro": 68.9,
        "confidence": "HIGH",
        "src": "Arena Text Overall Llama-3.3-70B 1318±3 (2026-07-21); MMLU-Pro artificialanalysis.ai",
    },
    "metallamaMetaLlama318BInstruct": {
        "elo": 1211,
        "mmlu_pro": 48.3,
        "confidence": "HIGH",
        "src": "Arena Text Overall Llama-3.1-8B 1211±4 (2026-07-21); MMLU-Pro artificialanalysis.ai",
    },
    "qwen35122ba10b": {
        "elo": 1417,
        "mmlu_pro": 86.1,
        "confidence": "HIGH",
        "src": "Arena Text Overall Qwen3.5-122B-A10B 1417±4 (2026-07-21, clean); MMLU-Pro benchlm.ai MED",
    },
    "qwen3535ba3b": {
        "elo": 1396,
        "mmlu_pro": 85.0,
        "confidence": "HIGH",
        "src": "Arena Text Overall Qwen3.5-35B-A3B 1396±4 (2026-07-21, clean); MMLU-Pro search approx LOW",
    },
    "qwen35397ba17b": {
        "elo": 1442,
        "mmlu_pro": 87.8,
        "confidence": "HIGH",
        "src": "Arena Text Overall Qwen3.5-397B-A17B 1442±4 (2026-07-21, clean); MMLU-Pro digitalapplied/benchlm.ai MED",
    },
    # ---- ext16 new models (researched 2026-07-26) ----
    "googlegemma327bit": {
        "elo": 1366,
        "mmlu_pro": 67.5,
        "confidence": "HIGH",
        "src": "Arena.ai Text Elo 1366±4 rank168; MMLU-Pro 67.5 Gemma-3 tech report 2503.19786",
    },
    "microsoftphi4": {
        "elo": 1256,
        "mmlu_pro": 71.4,
        "confidence": "HIGH",
        "src": "Arena.ai Text Elo 1256±5 rank282; MMLU-Pro 71.4 Phi-4 tech report 2412.08905",
    },
    "mistralaiMistralSmall24BInstruct2501": {
        "elo": 1274,
        "mmlu_pro": 66.3,
        "confidence": "HIGH",
        "src": "Arena.ai Text Elo 1274±6 rank270; MMLU-Pro 66.3 langdb/n8n benchmark",
    },
    "zaiorgGLM46": {
        "elo": 1425,
        "mmlu_pro": 83.2,
        "confidence": "HIGH-ELO/MED-MMLU",
        "src": "Arena.ai Text Elo 1425±4 rank91 (clean); MMLU-Pro 83.2 official z.ai (AA independent 78.4, diff=prompt/sampling/judge); cross-model tables use 83.2",
    },
}


def build_tiering(
    vec_d4, vec_q, models, anchor: str, split: tuple[int, int, int], label: str
) -> dict:
    """Tier models by an external anchor (desc), gap = mean(T1)-mean(T3) per level, both metrics."""
    avail = [m for m in models if EXTERNAL_SCORES.get(m, {}).get(anchor) is not None]
    ordered = sorted(avail, key=lambda m: (-EXTERNAL_SCORES[m][anchor], m))
    n1, n2, n3 = split
    if n1 + n2 + n3 != len(ordered):
        # keep splits exact: rescale bottom group to consume the remainder
        n3 = len(ordered) - n1 - n2
    t1, t2, t3 = ordered[:n1], ordered[n1 : n1 + n2], ordered[n1 + n2 : n1 + n2 + n3]
    d4_gaps = tier_gap_from_level_vectors(vec_d4, t1, t3)
    q_gaps = tier_gap_from_level_vectors(vec_q, t1, t3)
    return {
        "anchor": anchor,
        "split": f"{n1}-{n2}-{n3}",
        "n_models": len(ordered),
        "groups": {
            "T1": [MODEL_NAMES.get(m, m) for m in t1],
            "T2": [MODEL_NAMES.get(m, m) for m in t2],
            "T3": [MODEL_NAMES.get(m, m) for m in t3],
        },
        "order_desc": [MODEL_NAMES.get(m, m) for m in ordered],
        "d4": {"gaps": d4_gaps, **peak_summary(d4_gaps)},
        "quality_only": {"gaps": q_gaps, **peak_summary(q_gaps)},
    }


def main() -> None:
    selected = best_baseline_runs(LEVEL_WEIGHTS)
    models = list(BASELINE_MODELS)
    vec_d4 = {m: d4_level_means(selected[m]["data"]) for m in models}
    vec_q = {m: quality_level_means(selected[m]["data"]) for m in models}

    ext16 = (
        "googlegemma327bit" in models
    )  # common.py expands BASELINE_MODELS to 16 under ext16

    tierings: dict[str, dict] = {}
    if ext16:
        # All 16 models now have clean Arena Text Overall Elo -> full 16-model Elo + MMLU-Pro tiering.
        tierings["elo16_4-8-4"] = build_tiering(
            vec_d4, vec_q, models, "elo", (4, 8, 4), "elo16_4-8-4"
        )
        tierings["mmlu16_4-8-4"] = build_tiering(
            vec_d4, vec_q, models, "mmlu_pro", (4, 8, 4), "mmlu16_4-8-4"
        )
    else:
        # All 12 orig models now have clean Arena Elo -> full 12-model Elo tiering (supersedes elo8 subset).
        tierings["elo12_3-6-3"] = build_tiering(
            vec_d4, vec_q, models, "elo", (3, 6, 3), "elo12_3-6-3"
        )
        tierings["mmlu12_3-6-3"] = build_tiering(
            vec_d4, vec_q, models, "mmlu_pro", (3, 6, 3), "mmlu12_3-6-3"
        )

    # Reference: paper's own Q-tier (circular anchor) for comparison.
    q_ref_d4 = tier_gap_from_level_vectors(
        vec_d4, list(Q_TIERS["T1"]), list(Q_TIERS["T3"])
    )
    q_ref_q = tier_gap_from_level_vectors(
        vec_q, list(Q_TIERS["T1"]), list(Q_TIERS["T3"])
    )

    l2_all = all(t["d4"]["l2_is_unique_peak"] for t in tierings.values())

    write_json(
        "OFF-B_external_tier.json",
        {
            "analysis": "OFF-B external independent-capability anchor tier control (Elo / MMLU-Pro)",
            "zero_api_zero_model_calls": True,
            "tiering_key": "EXTERNAL published capability scores (LMArena Elo / MMLU-Pro), independent of GraphInstruct Q.",
            "metric": "Continuous d4_instruction_score per-level tier gap (primary) + quality-only D1-D4 level gap. Best baseline per model.",
            "n_total_models": len(models),
            "model_set": "ext16" if ext16 else "orig12",
            "external_scores": {
                m: EXTERNAL_SCORES[m] for m in models if m in EXTERNAL_SCORES
            },
            "tierings": tierings,
            "l2_unique_peak_all_external_tierings": l2_all,
            "reference_Q_tier": {
                "anchor": "GraphInstruct Q (paper own, circular)",
                "d4": {"gaps": q_ref_d4, **peak_summary(q_ref_d4)},
                "quality_only": {"gaps": q_ref_q, **peak_summary(q_ref_q)},
            },
            "selected_strategies": {m: selected[m]["strategy"] for m in models},
        },
    )
    print(
        f"OFF-B written ({'ext16' if ext16 else 'orig12'}, {len(models)} models). L2 unique peak all external tierings: {l2_all}"
    )
    for name, t in tierings.items():
        print(
            f"  {name}: d4 L2 gap={t['d4']['gaps'][2]:.4f} peak={t['d4']['peak_levels']} unique_L2={t['d4']['l2_is_unique_peak']}"
        )


if __name__ == "__main__":
    main()
