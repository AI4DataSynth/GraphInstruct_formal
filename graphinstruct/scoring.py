"""Hierarchical scoring: S_level -> S_total -> S_final.

Implements the scoring formulas from the design doc:
- level_score: Weighted sum of dimension scores for a single level
- total_score: Weighted sum of level scores across all levels
- final_score: Pareto-adjusted final ranking score
"""

from __future__ import annotations

from typing import Optional, Sequence

# Level weights: w_l = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
LEVEL_WEIGHTS: list[float] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]

# Dimension weights per level (from design doc Table 5.1)
# Keys: D1=Structure, D2=Text, D3=Embedding, D4=Instruction, D5=Efficiency
DIMENSION_WEIGHTS: list[dict[str, float]] = [
    # Level 0: Format generation — constraint type
    {"D1": 0.10, "D2": 0.00, "D3": 0.00, "D4": 0.60, "D5": 0.30},
    # Level 1: Explicit constraint — constraint type
    {"D1": 0.15, "D2": 0.00, "D3": 0.00, "D4": 0.70, "D5": 0.15},
    # Level 2: Multi-constraint — constraint type
    {"D1": 0.15, "D2": 0.00, "D3": 0.00, "D4": 0.70, "D5": 0.15},
    # Level 3: Property constraint — distribution type (D2=0: synthetic graphs lack text)
    # v2: D1 0.25→0.15 (spread=0.038, no discrimination), D4 0.40→0.50 (sole discriminator)
    {"D1": 0.15, "D2": 0.00, "D3": 0.15, "D4": 0.50, "D5": 0.20},
    # Level 4: Semantic constraint — distribution type
    # v2: D1 0.20→0.10 (spread=0.059), D3 0.15→0.05 (spread=0.015), D4 0.35→0.55
    # D2=0.15: text_presence + text_similarity v2 metric
    {"D1": 0.10, "D2": 0.15, "D3": 0.05, "D4": 0.55, "D5": 0.15},
    # Level 5: Multi-step reasoning — constraint type (D2=0: no text attributes)
    {"D1": 0.15, "D2": 0.00, "D3": 0.15, "D4": 0.50, "D5": 0.20},
]


def level_score(
    level: int,
    dimension_scores: dict[str, float],
    quality_only: bool = False,
) -> float:
    """Compute S_level for a single level.

    Formula: S_level_l = sum(w_{l,d} * S_{d,l}) for d in D1..D5

    Args:
        level: Instruction level (0-5).
        dimension_scores: Dict mapping dimension keys (D1-D5) to scores.
            Missing dimensions are treated as 0.0.
        quality_only: If True, exclude D5 (token efficiency) and
            renormalize D1-D4 weights to sum to 1.0.

    Returns:
        Weighted score for this level.
    """
    if not 0 <= level < len(DIMENSION_WEIGHTS):
        raise ValueError(f"level must be 0-{len(DIMENSION_WEIGHTS)-1}, got {level}")
    weights = dict(DIMENSION_WEIGHTS[level])

    if quality_only:
        weights["D5"] = 0.0
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

    return sum(
        weights.get(dim, 0.0) * dimension_scores.get(dim, 0.0)
        for dim in ("D1", "D2", "D3", "D4", "D5")
    )


def total_score(level_scores: Sequence[float]) -> float:
    """Compute S_total from per-level scores.

    Formula: S_total = sum(w_l * S_level_l) for l in 0..5

    Supports partial level lists (e.g., only L0-L3 in Phase 1).

    Args:
        level_scores: List of per-level scores. Can be shorter than 6.

    Returns:
        Weighted total score.
    """
    return sum(LEVEL_WEIGHTS[i] * s for i, s in enumerate(level_scores))


def final_score(
    s_total: float,
    pareto_bonus: float = 0.0,
    pareto_lambda: float = 0.15,
) -> float:
    """Compute S_final with Pareto bonus adjustment.

    Formula: S_final = S_total * (1 + lambda * Pareto_Bonus)

    Args:
        s_total: Total quality score.
        pareto_bonus: Pareto bonus in [0, 1]. 1.0 = on Pareto frontier.
        pareto_lambda: Bonus coefficient in [0, 0.3]. Default 0.15.

    Returns:
        Pareto-adjusted final score.
    """
    return s_total * (1 + pareto_lambda * pareto_bonus)
