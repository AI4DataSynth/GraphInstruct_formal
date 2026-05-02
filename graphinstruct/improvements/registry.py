"""Model tier registry and infeasibility lookup.

Centralised facts for CAAP strategy selection and VGIG short-circuit logic.

Tier labels derive from ``docs/model-capability-analysis-v4.md``:
    T1        best_total > 0.87 (Sonnet-4.6, Qwen3.5-397B/122B)
    T2-GPT    0.80-0.87 and CoT negative response (GPT-4o, GPT-4.1)
    T2-open   0.80-0.87 and CoT positive response (Qwen3.5-35B, DeepSeek-V3, Llama-70B)
    T3        best_total < 0.80 (GPT-3.5, GPT-4o-mini, Llama-8B)
"""

from __future__ import annotations

from typing import Literal

from graphinstruct.data_loader import Instruction

ModelTier = Literal["T1", "T2-GPT", "T2-open", "T3"]


# Canonical model name -> tier. All keys must be lower-case; lookup normalises
# input via ``.lower()``. Aliases cover filename-prefix variants used in
# results/ directory (e.g. "gpt4omini" vs "gpt-4o-mini").
MODEL_TIERS: dict[str, ModelTier] = {
    # T1: strongest tier
    "claude-sonnet-4-6": "T1",
    "claude-sonnet-4-5": "T1",
    "claude-sonnet-4-5-20250929": "T1",
    "qwen3.5-397b": "T1",
    "qwen-3.5-397b": "T1",
    "qwen3.5-122b": "T1",
    "qwen-3.5-122b": "T1",
    # T2-GPT: GPT family, CoT-negative
    "gpt-4o": "T2-GPT",
    "gpt4o": "T2-GPT",
    "gpt-4.1": "T2-GPT",
    "gpt-4-1": "T2-GPT",
    "gpt41": "T2-GPT",
    # T2-open: open-weight, CoT-positive
    "qwen3.5-35b-a3b": "T2-open",
    "qwen-3.5-35b-a3b": "T2-open",
    "qwen3.5-35b": "T2-open",
    "deepseek-ai/deepseek-v3": "T2-open",
    "deepseek-v3": "T2-open",
    "deepseekaideepseekv3": "T2-open",
    "llama-3-70b": "T2-open",
    "meta-llama/meta-llama-3-70b": "T2-open",
    # T3: weakest tier
    "gpt-3.5-turbo": "T3",
    "gpt-3-5-turbo": "T3",
    "gpt35turbo": "T3",
    "gpt-4o-mini": "T3",
    "gpt4omini": "T3",
    "gpt-4o-mini-2024-07-18": "T3",
    "llama-3-8b": "T3",
    "meta-llama/meta-llama-3-8b": "T3",
}


def model_tier(model_name: str) -> ModelTier:
    """Return the tier label for a model name.

    Case-insensitive lookup. Returns ``"T2-GPT"`` as a safe default for unknown
    model names (avoids crashing downstream callers; users should extend
    MODEL_TIERS when introducing a new model).
    """
    key = (model_name or "").strip().lower()
    return MODEL_TIERS.get(key, "T2-GPT")


def is_infeasible(instruction: Instruction) -> bool:
    """Return True if an instruction is known to be mathematically unsolvable.

    This delegates to ``Instruction.feasible`` which is pre-tagged in the
    dataset files (see data/instructions/level_2.json for k-regular
    odd-times-odd cases).
    """
    return not instruction.feasible
