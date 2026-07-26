"""Shared utilities for B-bucket analyses.

Defines:
- Canonical model/tier mapping
- Strategy parsing from filenames
- Loading quality.json + per_instruction records
- Level weight schedule
- Token-cost (TPV / API) → D5 / Sfin computation
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

# Level weights from the GraphInstruct paper Eq. (2)
LEVEL_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)

# Per-level instruction counts (data/instructions/level_X.json)
LEVEL_N: Final[tuple[int, ...]] = (100, 200, 200, 150, 100, 50)

# D5 hyperparameters from the GraphInstruct paper Eq. (1)
D5_TPV_SCALE: Final[float] = 1000.0
D5_API_SCALE: Final[float] = 2.0
D5_TPV_WEIGHT: Final[float] = 0.7
D5_API_WEIGHT: Final[float] = 0.3
PARETO_LAMBDA: Final[float] = 0.15

# Tier definitions from the GraphInstruct paper §3.1 (post-hoc descriptive)
MODEL_TIER: Final[dict[str, str]] = {
    "claudesonnet46": "T1",
    "qwen35397ba17b": "T1",
    "qwen35122ba10b": "T1",
    "claudesonnet420250514": "T2",
    "qwen3535ba3b": "T2",
    "gpt41": "T2",
    "gpt4o": "T2",
    "deepseekaiDeepSeekV3": "T2",
    "metallamaLlama3370BInstructTurbo": "T2",
    "gpt35turbo": "T3",
    "gpt4omini": "T3",
    "metallamaMetaLlama318BInstruct": "T3",
}

# Pretty model names for tables
MODEL_DISPLAY: Final[dict[str, str]] = {
    "claudesonnet46": "Sonnet-4.6",
    "claudesonnet420250514": "Sonnet-4",
    "qwen35397ba17b": "Qwen3.5-397B",
    "qwen35122ba10b": "Qwen3.5-122B",
    "qwen3535ba3b": "Qwen3.5-35B",
    "gpt41": "GPT-4.1",
    "gpt4o": "GPT-4o",
    "deepseekaiDeepSeekV3": "DeepSeek-V3",
    "metallamaLlama3370BInstructTurbo": "Llama-3.3-70B",
    "gpt35turbo": "GPT-3.5",
    "gpt4omini": "GPT-4o-mini",
    "metallamaMetaLlama318BInstruct": "Llama-3.1-8B",
}

# Family groupings
FAMILY: Final[dict[str, str]] = {
    "claudesonnet46": "Anthropic",
    "claudesonnet420250514": "Anthropic",
    "qwen35397ba17b": "Qwen3.5",
    "qwen35122ba10b": "Qwen3.5",
    "qwen3535ba3b": "Qwen3.5",
    "gpt41": "GPT",
    "gpt4o": "GPT",
    "gpt35turbo": "GPT",
    "gpt4omini": "GPT",
    "deepseekaiDeepSeekV3": "DeepSeek",
    "metallamaLlama3370BInstructTurbo": "Llama",
    "metallamaMetaLlama318BInstruct": "Llama",
}

BASELINE_STRATS: Final[tuple[str, ...]] = (
    "zero-shot",
    "few-shot",
    "zero-cot",
    "few-cot",
)
METHOD_STRATS: Final[tuple[str, ...]] = (
    "caap",
    "combined",
    "vgig",
    "oracle",
    "retry",
    "sc",
)


def parse_stem(stem: str) -> tuple[str, str] | None:
    """Parse a results file stem into (model, strategy).

    Returns None if the strategy is not recognized.
    """
    parts = stem.split("-")
    # Try longest strategy suffix first (e.g. "vgig-T15")
    for i in range(len(parts) - 1, 0, -1):
        cand = "-".join(parts[i:])
        if cand in BASELINE_STRATS or cand in METHOD_STRATS:
            return "-".join(parts[:i]), cand
        if cand.startswith("vgig-"):  # e.g. vgig-T15, vgig-fbcoarse
            return "-".join(parts[:i]), cand
    return None


def load_quality(path: Path) -> dict:
    """Load a results/<model>-<strategy>.quality.json file."""
    return json.loads(path.read_text(encoding="utf-8"))


def per_level_scores(q: dict) -> dict[int, float]:
    """Extract per_level_scores keyed by int level."""
    pls = q.get("per_level_scores", {})
    return {int(k): v for k, v in pls.items()}


def compute_total_score(
    level_scores: dict[int, float], weights: tuple[float, ...] = LEVEL_WEIGHTS
) -> float:
    """Compute total Quality from per-level scores given weights."""
    return sum(weights[l] * level_scores.get(l, 0.0) for l in range(6))


def d5_score(tpv: float, api_calls: float = 1.0) -> float:
    """Compute D5 efficiency score per Eq. (1).

    D5 = 0.7 * exp(-TPV/1000) + 0.3 * exp(-(API-1)/2)
    """
    return D5_TPV_WEIGHT * math.exp(-tpv / D5_TPV_SCALE) + D5_API_WEIGHT * math.exp(
        -(api_calls - 1) / D5_API_SCALE
    )


def sfin(total_q: float, is_pareto: bool, lam: float = PARETO_LAMBDA) -> float:
    """Compute Sfin = Q * (1 + lambda * ParetoBonus)."""
    return total_q * (1 + lam * (1.0 if is_pareto else 0.0))


def q_per_ktpv(total_q: float, tpv: float) -> float:
    """Compute Q / kTPV = Q / (TPV / 1000)."""
    return total_q / (tpv / 1000.0) if tpv > 0 else 0.0


def find_pareto_front(points: list[tuple[float, float, str]]) -> set[str]:
    """Find Pareto-optimal points given list of (TPV, Q, key).

    A point (TPV_a, Q_a) is dominated by (TPV_b, Q_b) iff
    TPV_b <= TPV_a AND Q_b >= Q_a AND at least one strict.

    Returns set of keys that are non-dominated.
    """
    keys = set()
    for tpv_a, q_a, key_a in points:
        dominated = False
        for tpv_b, q_b, key_b in points:
            if key_a == key_b:
                continue
            if tpv_b <= tpv_a and q_b >= q_a and (tpv_b < tpv_a or q_b > q_a):
                dominated = True
                break
        if not dominated:
            keys.add(key_a)
    return keys


def scan_results_dir(
    results_dir: Path = Path("results"), only_baselines: bool = False
) -> dict[tuple[str, str], Path]:
    """Find all quality.json files; map (model, strategy) -> path.

    Files may live either directly under ``results/`` or in the
    ``results/quality/`` subdirectory used by the released artifact.
    """
    out: dict[tuple[str, str], Path] = {}
    candidates = sorted(results_dir.glob("*.quality.json")) + sorted(
        (results_dir / "quality").glob("*.quality.json")
    )
    for f in candidates:
        stem = f.name.removesuffix(".quality.json")
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        model, strat = parsed
        if only_baselines and strat not in BASELINE_STRATS:
            continue
        out[(model, strat)] = f
    return out


def compute_tpv_api(jsonl_path: Path) -> tuple[float, float, int]:
    """Compute mean TPV (per valid graph) and mean API-calls across a jsonl.

    Returns (mean_tpv, mean_api_calls, n_valid_samples).
    Counts only samples where valid=True for TPV (matches D5 definition).
    """
    total_tokens = 0
    total_api = 0
    n_valid = 0
    n_total = 0
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for s in rec.get("samples", []):
                n_total += 1
                api = s.get("api_calls", 1)
                total_api += api
                if s.get("valid", False):
                    total_tokens += s.get("output_tokens", 0)
                    n_valid += 1
    mean_tpv = total_tokens / n_valid if n_valid else float("inf")
    mean_api = total_api / n_total if n_total else 1.0
    return mean_tpv, mean_api, n_valid
