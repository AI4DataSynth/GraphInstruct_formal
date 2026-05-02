"""D5: Token efficiency metrics.

Evaluates the cost of graph generation:
- tokens_per_valid_graph: Average tokens consumed per valid graph
- api_calls_per_graph: Average API calls per valid graph
- pareto_score: Quality / log(1 + total_tokens)
- TokenRecord: Data structure for recording generation costs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class TokenRecord:
    """Record of token usage for a single graph generation attempt.

    Attributes:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        api_calls: Number of LLM API calls (including retries).
        valid: Whether the generated graph was valid.
    """

    input_tokens: int
    output_tokens: int
    api_calls: int = 1
    valid: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def tokens_per_valid_graph(records: Sequence[TokenRecord]) -> float:
    """Compute average tokens consumed per valid graph.

    Formula: sum(all tokens) / count(valid graphs)

    Args:
        records: List of token records from generation attempts.

    Returns:
        Average tokens per valid graph. Returns inf if no valid graphs.
    """
    if not records:
        return float("inf")

    total_tokens = sum(r.total_tokens for r in records)
    valid_count = sum(1 for r in records if r.valid)

    if valid_count == 0:
        return float("inf")

    return total_tokens / valid_count


def api_calls_per_graph(records: Sequence[TokenRecord]) -> float:
    """Compute average API calls per valid graph.

    Formula: sum(all api_calls) / count(valid graphs)

    Args:
        records: List of token records.

    Returns:
        Average API calls per valid graph. Returns inf if no valid graphs.
    """
    if not records:
        return float("inf")

    total_calls = sum(r.api_calls for r in records)
    valid_count = sum(1 for r in records if r.valid)

    if valid_count == 0:
        return float("inf")

    return total_calls / valid_count


def pareto_score(quality: float, total_tokens: int) -> float:
    """Compute Pareto score: Quality / log(1 + Total_Tokens).

    This is a **leaderboard auxiliary metric** used by ``leaderboard.py``
    for cross-model efficiency comparison. It is NOT the D5 dimension score.

    The D5 dimension score is computed in ``evaluate.py`` using an exponential
    decay formula: ``0.7 * exp(-TPV/1000) + 0.3 * exp(-(API-1)/2)``.
    The two formulas serve different purposes:
    - This function: cross-model ranking (higher quality per token = better)
    - D5 dimension: per-instruction efficiency score in [0, 1]

    Args:
        quality: Quality score (e.g., S_total), typically in [0, 1].
        total_tokens: Total tokens consumed.

    Returns:
        Pareto score. Returns inf if total_tokens is 0 and quality > 0.
    """
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    denominator = math.log(1 + total_tokens)
    if denominator == 0:
        return float("inf") if quality > 0 else 0.0
    return quality / denominator
