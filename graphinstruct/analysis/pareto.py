"""Pareto frontier analysis for cost-quality trade-off evaluation.

Computes Pareto frontiers, Pareto bonuses, and efficiency rankings
as described in Section 4 of the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ParetoPoint:
    """A point in the cost-quality space.

    Attributes:
        label: Model/method identifier (e.g., "GPT-4o + few-shot").
        quality: Quality score (S_total), higher is better.
        cost: Total token cost, lower is better.
        valid_rate: Generation success rate (for visualization sizing).
    """

    label: str
    quality: float
    cost: float
    valid_rate: float = 1.0


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------


def compute_pareto_frontier(
    points: Sequence[ParetoPoint],
) -> list[ParetoPoint]:
    """Find the Pareto-optimal points (maximize quality, minimize cost).

    A point is Pareto-optimal if no other point has both
    higher-or-equal quality AND lower-or-equal cost (with at least
    one strict inequality).

    Args:
        points: Collection of (label, quality, cost) points.

    Returns:
        List of Pareto-optimal points sorted by cost ascending.
    """
    if not points:
        return []

    frontier: list[ParetoPoint] = []

    for pt in points:
        dominated = any(
            fp.quality >= pt.quality
            and fp.cost <= pt.cost
            and (fp.quality > pt.quality or fp.cost < pt.cost)
            for fp in frontier
        )
        if dominated:
            continue
        # Remove existing frontier points dominated by this one
        frontier = [
            fp
            for fp in frontier
            if not (
                pt.quality >= fp.quality
                and pt.cost <= fp.cost
                and (pt.quality > fp.quality or pt.cost < fp.cost)
            )
        ]
        frontier.append(pt)

    return sorted(frontier, key=lambda p: p.cost)


def is_on_frontier(
    point: ParetoPoint,
    frontier: Sequence[ParetoPoint],
) -> bool:
    """Check if a point is on the given Pareto frontier."""
    for fp in frontier:
        if abs(fp.quality - point.quality) < 1e-9 and abs(fp.cost - point.cost) < 1e-9:
            return True
    return False


# ---------------------------------------------------------------------------
# Pareto bonus
# ---------------------------------------------------------------------------


def pareto_bonus(
    point: ParetoPoint,
    frontier: Sequence[ParetoPoint],
) -> float:
    """Compute Pareto bonus for a point.

    Returns 1.0 if the point is on the frontier, otherwise a value
    in [0, 1) based on the normalized distance to the frontier.

    Args:
        point: The point to evaluate.
        frontier: The Pareto frontier.

    Returns:
        Bonus in [0.0, 1.0].
    """
    if not frontier:
        return 1.0

    if is_on_frontier(point, frontier):
        return 1.0

    all_points = list(frontier) + [point]
    max_cost = max(p.cost for p in all_points)
    min_cost = min(p.cost for p in all_points)
    max_q = max(p.quality for p in all_points)
    min_q = min(p.quality for p in all_points)

    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    q_range = max_q - min_q if max_q > min_q else 1.0

    norm_cost = (point.cost - min_cost) / cost_range
    norm_q = (point.quality - min_q) / q_range

    min_dist = float("inf")
    for fp in frontier:
        fc = (fp.cost - min_cost) / cost_range
        fq = (fp.quality - min_q) / q_range
        dist = ((norm_cost - fc) ** 2 + (norm_q - fq) ** 2) ** 0.5
        min_dist = min(min_dist, dist)

    # Max normalized distance is sqrt(2) ≈ 1.414
    return max(0.0, 1.0 - min_dist / 1.414)


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------


def quality_per_ktoken(quality: float, total_tokens: int) -> float:
    """Quality per 1000 tokens.

    Higher is better — more quality per unit cost.
    """
    if total_tokens <= 0:
        return 0.0
    return quality / (total_tokens / 1000)


def cost_at_quality(
    points: Sequence[ParetoPoint],
    threshold: float = 0.8,
) -> Optional[float]:
    """Minimum token cost to achieve quality >= threshold.

    Returns None if no point meets the threshold.
    """
    qualifying = [p for p in points if p.quality >= threshold]
    if not qualifying:
        return None
    return min(p.cost for p in qualifying)


def quality_at_budget(
    points: Sequence[ParetoPoint],
    budget: float = 10000,
) -> float:
    """Maximum quality achievable within a token budget.

    Returns 0.0 if no point is within budget.
    """
    within_budget = [p for p in points if p.cost <= budget]
    if not within_budget:
        return 0.0
    return max(p.quality for p in within_budget)


def frontier_distance(
    point: ParetoPoint,
    frontier: Sequence[ParetoPoint],
) -> float:
    """Vertical (quality) distance from a point to the Pareto frontier.

    Uses linear interpolation between frontier points.
    Positive = below frontier (room to improve).
    Zero = on frontier.

    Returns:
        Non-negative distance value.
    """
    if not frontier:
        return 0.0

    sorted_f = sorted(frontier, key=lambda p: p.cost)

    # Outside frontier range: use nearest endpoint
    if point.cost <= sorted_f[0].cost:
        return max(0.0, sorted_f[0].quality - point.quality)
    if point.cost >= sorted_f[-1].cost:
        return max(0.0, sorted_f[-1].quality - point.quality)

    # Interpolate between surrounding frontier points
    for i in range(len(sorted_f) - 1):
        if sorted_f[i].cost <= point.cost <= sorted_f[i + 1].cost:
            t = (point.cost - sorted_f[i].cost) / (
                sorted_f[i + 1].cost - sorted_f[i].cost
            )
            frontier_q = sorted_f[i].quality + t * (
                sorted_f[i + 1].quality - sorted_f[i].quality
            )
            return max(0.0, frontier_q - point.quality)

    return 0.0
