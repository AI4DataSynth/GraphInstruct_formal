"""D2: quality-only VGIG/combined comparison against retry and self-consistency."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphinstruct.scoring import LEVEL_WEIGHTS
from scripts.rebuttal.common import (
    best_baseline_runs,
    quality_level_means,
    quality_path,
    load_quality_file,
    weighted_total,
    write_json,
)

TARGETS = ("deepseekaiDeepSeekV3", "gpt4omini")
METHODS = ("vgig", "retry", "sc", "combined")


def quality_total_from_path(path: Path) -> float:
    return weighted_total(quality_level_means(load_quality_file(str(path))), LEVEL_WEIGHTS)


def main() -> None:
    baselines = best_baseline_runs(LEVEL_WEIGHTS)
    targets: dict[str, object] = {}
    for target in TARGETS:
        baseline = baselines[target]
        method_rows = {}
        for method in METHODS:
            path = quality_path(target, method)
            if not path.exists():
                method_rows[method] = {"available": False, "reason": f"missing {path.name}"}
                continue
            total = quality_total_from_path(path)
            method_rows[method] = {
                "available": True,
                "quality_only_total": total,
                "gain_vs_best_baseline": total - baseline["total"],
            }
        vgig = method_rows.get("vgig", {})
        for comparison in ("retry", "sc"):
            comparator = method_rows.get(comparison, {})
            if vgig.get("available") and comparator.get("available"):
                vgig[f"gain_over_{comparison}"] = vgig["quality_only_total"] - comparator["quality_only_total"]
        targets[target] = {
            "baseline": {"strategy": baseline["strategy"], "quality_only_total": baseline["total"]},
            "methods": method_rows,
        }

    verify_only: dict[str, object]
    target = "gpt4omini"
    needed = {name: quality_path(target, name) for name in ("zero-shot", "vgig-fbnone", "vgig-fbcoarse", "vgig")}
    if all(path.exists() for path in needed.values()):
        totals = {name: quality_total_from_path(path) for name, path in needed.items()}
        best = baselines[target]["total"]
        fine_total = totals["vgig"]
        zero_ratio = (totals["vgig-fbnone"] - totals["zero-shot"]) / (fine_total - totals["zero-shot"]) if fine_total != totals["zero-shot"] else None
        best_ratio = (totals["vgig-fbnone"] - best) / (fine_total - best) if fine_total != best else None
        verify_only = {
            "available": True,
            "files": {name: path.name for name, path in needed.items()},
            "quality_only_totals": totals,
            "formula": "verify-only proportion = (fbnone - baseline)/(fine_vgig - baseline); fbnone is the local no-feedback verifier/early-stop ablation.",
            "proportion_vs_zero_shot": zero_ratio,
            "proportion_vs_best_baseline": best_ratio,
            "best_baseline_strategy": baselines[target]["strategy"],
        }
    else:
        verify_only = {"available": False, "reason": "Missing one or more local gpt4omini E6 feedback ablation quality files."}

    write_json(
        "D2_vgig_vs_retry_sc.json",
        {
            "analysis": "D2 VGIG versus retry/SC quality-only comparison",
            "zero_api_zero_model_calls": True,
            "metric": "D5=0 and D1-D4 reweighted from repository constants; all values are recomputed from per_instruction scores.",
            "selection_signal_note": "This artifact compares only scored outputs. The quality JSON schema has no method-selection-signal field, so satisfaction-rate selection is not independently asserted from these files.",
            "targets": targets,
            "verify_only_feedback_ablation": verify_only,
        },
    )


if __name__ == "__main__":
    main()
