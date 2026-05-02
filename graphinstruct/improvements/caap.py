"""Constraint-Aware Adaptive Prompting (CAAP) 2.0 decision matrix.

Encodes the strategy-selection rules derived from the v4 benchmark analysis
(``docs/model-capability-analysis-v4.md``) as a (level, tier) lookup.

Key data-driven constraints:
    L2 x any-tier  -> never few-shot (avg -0.034; GPT-4o-mini crashes to 0.424)
    L3 x T3        -> never few-cot (poison, -0.048)
    L3 x T2-GPT    -> prefer zero-shot (FC bad, ZS neutral)
    L4             -> few-shot is the only +0.069 strategy; domain prior always injected
    L5 x T3        -> never few-cot (-0.041)

See ``docs/improvement-proposal-and-timeline-v2.md`` §2 for the full table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from graphinstruct.data_loader import Instruction
from graphinstruct.improvements.domain_priors import build_domain_text
from graphinstruct.improvements.registry import ModelTier, model_tier

Strategy = Literal["zero-shot", "few-shot", "zero-cot", "few-cot"]


@dataclass(frozen=True)
class CAAPDecision:
    """Selected strategy plus prompt-augmenting extras.

    ``extras`` keys (all optional):
        checklist  -- L2 per-constraint verification text
        formula    -- L3 numerical property formulas
        domain     -- L4 domain structural prior text
        step_hint  -- L5 edit task step breakdown
    """

    strategy: Strategy
    extras: dict[str, str] = field(default_factory=dict)
    rationale: str = ""


_SIMPLE_TYPES = frozenset({"tree", "cycle", "star", "path", "complete"})


def select_strategy(instruction: Instruction, model_name: str) -> CAAPDecision:
    """Pick the optimal (strategy, extras) pair for an instruction and model.

    Dispatches by instruction level; per-level deciders encode the v4 rules.
    """
    tier = model_tier(model_name)
    return _DECISION_TABLE[instruction.level](instruction, tier)


# ---------------------------------------------------------------------------
# Per-level deciders
# ---------------------------------------------------------------------------


def _decide_l0(inst: Instruction, tier: ModelTier) -> CAAPDecision:
    return CAAPDecision(
        strategy="zero-shot",
        rationale="L0 format only; all tier deltas <0.025",
    )


def _decide_l1(inst: Instruction, tier: ModelTier) -> CAAPDecision:
    if _is_simple_type(inst):
        return CAAPDecision(
            strategy="zero-shot",
            rationale="L1 simple type (tree/cycle/star/path/complete): >90% baseline",
        )
    # Complex type: forest / dag / bipartite / grid / wheel / etc.
    if tier == "T3":
        return CAAPDecision(
            strategy="zero-cot",
            rationale="T3 L1 complex: ZC is the safest incremental strategy",
        )
    if tier == "T2-GPT":
        return CAAPDecision(
            strategy="few-shot",
            rationale="T2-GPT L1 complex: FS works, GPT family dislikes FC",
        )
    return CAAPDecision(
        strategy="few-cot",
        rationale="T2-open/T1 L1 complex: FC positive (e.g. Qwen3.5 +0.052)",
    )


def _decide_l2(inst: Instruction, tier: ModelTier) -> CAAPDecision:
    # Hard rule: FS is toxic at L2 (avg -0.034; GPT-4o-mini collapses to 0.424).
    if tier == "T3":
        return CAAPDecision(
            strategy="zero-cot",
            extras={"checklist": _build_constraint_checklist(inst)},
            rationale="T3 L2: FS toxic (GPT-4o-mini drops to 0.424); ZC + checklist",
        )
    if tier == "T2-GPT":
        return CAAPDecision(
            strategy="few-shot",
            rationale="T2-GPT L2: tolerates FS in v4 (confirm in pilot)",
        )
    return CAAPDecision(
        strategy="few-cot",
        rationale="T2-open/T1 L2: FC works in v4",
    )


def _decide_l3(inst: Instruction, tier: ModelTier) -> CAAPDecision:
    # Hard rule: FC is the largest negative effect at L3 (avg -0.048).
    if tier == "T3":
        return CAAPDecision(
            strategy="zero-cot",
            extras={"formula": _build_formula_hints(inst)},
            rationale="T3 L3: FC toxic (-0.048); ZC + explicit formulas",
        )
    if tier == "T2-GPT":
        return CAAPDecision(
            strategy="zero-shot",
            rationale="T2-GPT L3: ZS neutral, FC clearly bad",
        )
    return CAAPDecision(
        strategy="zero-cot",
        rationale="T2-open/T1 L3: ZC has a small positive effect (+0.004 avg)",
    )


def _decide_l4(inst: Instruction, tier: ModelTier) -> CAAPDecision:
    # L4: FS is the uniquely strong positive strategy (+0.069 avg).
    domain_txt = build_domain_text(inst.domain, _try_num_nodes(inst))
    extras: dict[str, str] = {}
    if domain_txt:
        extras["domain"] = domain_txt
    if tier in ("T3", "T2-GPT"):
        return CAAPDecision(
            strategy="few-shot",
            extras=extras,
            rationale="L4: FS is the only +0.069 strategy across all tiers",
        )
    return CAAPDecision(
        strategy="few-cot",
        extras=extras,
        rationale="T2-open/T1 L4: FC + domain prior",
    )


def _decide_l5(inst: Instruction, tier: ModelTier) -> CAAPDecision:
    # L5: FC best overall (+0.045) but T3 weak models are actually hurt (-0.041).
    if tier == "T3":
        return CAAPDecision(
            strategy="zero-cot",
            rationale="T3 L5: FC toxic (-0.041); ZC is safest",
        )
    return CAAPDecision(
        strategy="few-cot",
        rationale="T2-GPT/T2-open/T1 L5: FC has the highest positive delta (+0.045)",
    )


_DECISION_TABLE = {
    0: _decide_l0,
    1: _decide_l1,
    2: _decide_l2,
    3: _decide_l3,
    4: _decide_l4,
    5: _decide_l5,
}


# ---------------------------------------------------------------------------
# Extras builders
# ---------------------------------------------------------------------------


def _is_simple_type(inst: Instruction) -> bool:
    """True if the instruction's graph_type is one of the simple primitives."""
    for c in inst.explicit_constraints:
        if c.startswith("graph_type="):
            gtype = c.split("=", 1)[1].strip().lower()
            return gtype in _SIMPLE_TYPES
    return False


def _try_num_nodes(inst: Instruction) -> Optional[int]:
    """Best-effort extraction of num_nodes for domain prior sizing."""
    for c in inst.explicit_constraints:
        if c.startswith("num_nodes="):
            try:
                return int(c.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _build_constraint_checklist(inst: Instruction) -> str:
    """Explicit-constraint verification checklist for L2 prompts."""
    lines = [
        "Before outputting, verify that your graph satisfies ALL constraints:",
    ]
    for i, c in enumerate(inst.explicit_constraints, 1):
        lines.append(f"{i}. [ ] {c}")
    lines.append("Think step by step and count carefully for each item.")
    return "\n".join(lines)


def _build_formula_hints(inst: Instruction) -> str:
    """Emit formulas only for numerical properties present in this instruction."""
    constraint_blob = " ".join(inst.explicit_constraints).lower()
    hints: list[str] = []
    if "density" in constraint_blob:
        hints.append("- density = 2|E| / (|V|(|V|-1))  (undirected graphs)")
    if "clustering" in constraint_blob:
        hints.append(
            "- clustering coefficient = 3 * (#triangles) / (#connected_triples)"
        )
    if "avg_path_length" in constraint_blob or "average_path_length" in constraint_blob:
        hints.append(
            "- avg_path_length = mean shortest-path distance over all connected node pairs"
        )
    if "diameter" in constraint_blob:
        hints.append("- diameter = max shortest-path distance over all node pairs")
    if "modularity" in constraint_blob:
        hints.append(
            "- modularity measures how well a partitioning separates densely "
            "connected clusters"
        )
    if not hints:
        return ""
    return "Numerical property formulas relevant to this instruction:\n" + "\n".join(
        hints
    )
