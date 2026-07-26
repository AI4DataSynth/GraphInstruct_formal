"""S1: tier-free cross-model discrimination (headline L2-peak evidence, zero tiers).

Per level, computes the cross-model spread (mean / population-std / range) of zero-shot
quality-only level scores. Uses NO tier partition at all -> the purest answer to the tier-circularity concern
("circular tier definition") and AC concern-1 ("L2 peak is a metric artifact"): the
multi-constraint level (L2) simply spreads models the most. Model-set follows BASELINE_MODELS
(12 orig / 16 ext16). Zero API / zero model calls.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.common import (
    BASELINE_MODELS,
    MODEL_NAMES,
    RESULTS_DIR,
    load_quality_file,
    write_json,
)

LEVEL_NAMES = {
    0: "format",
    1: "single-constraint",
    2: "multi-constraint",
    3: "property",
    4: "semantic",
    5: "multi-step",
}


def _zero_shot_level_scores(slug: str) -> dict[str, float]:
    path = RESULTS_DIR / f"{slug}-zero-shot.quality.json"
    return load_quality_file(str(path))["per_level_scores"]


def main() -> None:
    models = list(BASELINE_MODELS)
    per_model = {m: _zero_shot_level_scores(m) for m in models}

    per_level = []
    stds = []
    for level in range(6):
        vals = [per_model[m][str(level)] for m in models]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals)
        stds.append(std)
        lo, hi = min(vals), max(vals)
        per_level.append(
            {
                "level": level,
                "name": LEVEL_NAMES[level],
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(lo, 4),
                "max": round(hi, 4),
                "range": round(hi - lo, 4),
            }
        )

    max_std = max(stds)
    peak_levels = [i for i, s in enumerate(stds) if abs(s - max_std) < 1e-12]
    l2_std = stds[2]
    runner_up = sorted(stds, reverse=True)[1]

    write_json(
        "S1_tierfree_discrimination.json",
        {
            "analysis": "S1 tier-free cross-model discrimination (zero-shot quality-only level scores)",
            "zero_api_zero_model_calls": True,
            "metric": "Per-level cross-model population std / mean / range of zero-shot quality-only level_score. No tiers.",
            "n_models": len(models),
            "models": [MODEL_NAMES.get(m, m) for m in models],
            "per_level": per_level,
            "std_by_level": [round(s, 4) for s in stds],
            "peak_levels": peak_levels,
            "l2_is_peak": 2 in peak_levels,
            "l2_is_unique_peak": peak_levels == [2],
            "l2_std_over_runnerup": round(l2_std / runner_up, 3) if runner_up else None,
        },
    )
    print(
        f"S1 written ({len(models)} models). L2 unique std-peak: {peak_levels == [2]} "
        f"(L2 std={l2_std:.4f} = {l2_std / runner_up:.2f}x runner-up)"
    )
    for row in per_level:
        mark = " <-- PEAK" if row["level"] in peak_levels else ""
        print(
            f"  L{row['level']} ({row['name']:17s}): mean={row['mean']:.3f} std={row['std']:.4f} range={row['range']:.3f}{mark}"
        )


if __name__ == "__main__":
    main()
