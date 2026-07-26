"""EXP-6: L2 discrimination-peak robustness under LLM paraphrase (offline).

Reproduces rebuttal_analysis/SUMMARY_ext_EXP6.md from persisted data:
  - paraphrased runs:  results/exp6/{model}-para-zero-shot.quality.json
  - original baseline: results/{model}-zero-shot.quality.json  (matched IDs only)

Discrimination = across-model standard deviation of per-level mean D4 (tier-split
free, non-gameable). Higher std = models more separated at that level. We test
whether L2 is the discrimination peak in BOTH original and paraphrased data.

Zero API / zero model calls: reads only local quality.json files.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.common import (
    RESULTS_DIR,
    load_quality_file,
    safe_float,
    write_json,
)

PARA_DIR = RESULTS_DIR / "exp6"
LEVELS = (1, 2, 3)

# Models with BOTH a paraphrased run and an original zero-shot baseline.
# Mistral-Small-24B is excluded until its para run is re-scored (original EXP6
# hit a NetworkX scorer crash on one of its graphs -> no para quality.json yet).
CANDIDATE_MODELS = (
    "deepseekaiDeepSeekV3",
    "metallamaLlama3370BInstructTurbo",
    "googlegemma327bit",
    "metallamaMetaLlama318BInstruct",
    "microsoftphi4",
    "mistralaiMistralSmall24BInstruct2501",
)


def _d4_means_for_ids(
    data: dict, ids_by_level: dict[int, set[str]]
) -> dict[int, float]:
    """Mean D4 per level, restricted to the supplied matched instruction IDs."""
    means: dict[int, float] = {}
    for level in LEVELS:
        vals = [
            safe_float(
                rec.get(
                    "d4_instruction_score", rec.get("dimension_scores", {}).get("D4")
                )
            )
            for rec in data["per_instruction"]
            if int(rec["level"]) == level
            and str(rec["instruction_id"]) in ids_by_level[level]
        ]
        means[level] = sum(vals) / len(vals) if vals else float("nan")
    return means


def _para_ids_by_level(para_data: dict) -> dict[int, set[str]]:
    ids: dict[int, set[str]] = {level: set() for level in LEVELS}
    for rec in para_data["per_instruction"]:
        level = int(rec["level"])
        if level in ids:
            ids[level].add(str(rec["instruction_id"]))
    return ids


def _across_model_std(per_model: dict[str, dict[int, float]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for level in LEVELS:
        vals = [per_model[m][level] for m in per_model]
        out[level] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def _peak_level(std_by_level: dict[int, float]) -> int:
    return max(std_by_level, key=lambda lv: std_by_level[lv])


def main() -> None:
    available = []
    for model in CANDIDATE_MODELS:
        para_path = PARA_DIR / f"{model}-para-zero-shot.quality.json"
        orig_path = RESULTS_DIR / f"{model}-zero-shot.quality.json"
        if para_path.exists() and orig_path.exists():
            available.append((model, para_path, orig_path))

    orig_d4: dict[str, dict[int, float]] = {}
    para_d4: dict[str, dict[int, float]] = {}
    for model, para_path, orig_path in available:
        para_data = load_quality_file(str(para_path))
        orig_data = load_quality_file(str(orig_path))
        ids = _para_ids_by_level(para_data)  # 50 sampled IDs/level define the match
        para_d4[model] = _d4_means_for_ids(para_data, ids)
        orig_d4[model] = _d4_means_for_ids(orig_data, ids)

    std_orig = _across_model_std(orig_d4)
    std_para = _across_model_std(para_d4)
    peak_orig = _peak_level(std_orig)
    peak_para = _peak_level(std_para)

    payload = {
        "analysis": "EXP-6 paraphrase robustness of the L2 discrimination peak",
        "metric": "across-model std of per-level mean D4 (tier-split-free discrimination)",
        "levels": list(LEVELS),
        "n_models_matched": len(available),
        "models_matched": [m for m, _, _ in available],
        "excluded": [m for m in CANDIDATE_MODELS if m not in {a[0] for a in available}],
        "per_model_d4": {
            m: {"original": orig_d4[m], "paraphrased": para_d4[m]} for m in orig_d4
        },
        "across_model_std": {
            "original": std_orig,
            "paraphrased": std_para,
        },
        "peak_level_original": peak_orig,
        "peak_level_paraphrased": peak_para,
        "l2_is_peak_original": peak_orig == 2,
        "l2_is_peak_paraphrased": peak_para == 2,
        "l2_peak_survives_paraphrase": peak_orig == 2 and peak_para == 2,
        "zero_api_zero_model_calls": True,
        "note": (
            "Discrimination compresses under paraphrase (std shrinks) but the peak "
            "LEVEL is the claim, not its magnitude. SUMMARY_ext_EXP6.md used a 4-model "
            "matched subset; this script uses all models with both runs present."
        ),
    }
    write_json("EXP6_paraphrase.json", payload)

    print(f"matched models ({len(available)}): {[m for m, _, _ in available]}")
    print(
        f"orig std L1/L2/L3: {[round(std_orig[l], 4) for l in LEVELS]}  peak=L{peak_orig}"
    )
    print(
        f"para std L1/L2/L3: {[round(std_para[l], 4) for l in LEVELS]}  peak=L{peak_para}"
    )
    print(f"L2 peak survives paraphrase: {payload['l2_peak_survives_paraphrase']}")


if __name__ == "__main__":
    main()
