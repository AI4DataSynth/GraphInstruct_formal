"""E1b: Kriegeskorte-style split-data tier construction and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuttal.common import split_data_tier_results, write_json

SEEDS = (2026, 2027, 2028, 2029, 2030)


def main() -> None:
    result = split_data_tier_results(SEEDS)
    write_json(
        "E1b_splitdata_tier.json",
        {
            "analysis": "E1b split-data tier control",
            "zero_api_zero_model_calls": True,
            "metric": "A-half selection uses default-LEVEL_WEIGHTS quality-only totals; B-half analysis uses continuous d4_instruction_score tier gaps.",
            "selection_analysis_disjoint": True,
            **result,
        },
    )


if __name__ == "__main__":
    main()
