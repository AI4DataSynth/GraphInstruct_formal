"""Prompt augmentation for improvement runners.

Builds on top of the legacy ``scripts.run_baseline.build_prompt`` to:
    1. Inject CAAP extras (checklist / formula / domain / step_hint) into a
       per-instruction prompt at fixed positions.
    2. Build VGIG refinement prompts (round t >= 1) that include the previous
       attempt and verifier feedback.

The import of ``build_prompt`` is deferred to call-time to tolerate test
environments where ``scripts/`` is not on ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from graphinstruct.data_loader import Instruction


def _import_base_build_prompt():
    """Lazy import of scripts.run_baseline.build_prompt.

    Adds the project root to ``sys.path`` if necessary so the import
    succeeds when called outside of ``run_baseline.py``.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from scripts.run_baseline import build_prompt  # type: ignore[import-not-found]

    return build_prompt


def build_enhanced_prompt(
    instruction: Instruction,
    strategy: str,
    all_instructions: dict,
    graph_format: str = "code",
    extras: Optional[dict[str, str]] = None,
) -> str:
    """Return a prompt augmented with CAAP extras.

    Injection order (top to bottom in the final prompt text):
        1. domain    - prefix as context block
        2. <base prompt from scripts.run_baseline.build_prompt>
        3. formula   - appended before checklist
        4. checklist - appended
        5. step_hint - appended (L5 only)

    When ``extras`` is None or empty, this is equivalent to calling
    ``build_prompt`` directly.
    """
    build_prompt = _import_base_build_prompt()
    base = build_prompt(instruction, strategy, all_instructions, graph_format)
    if not extras:
        return base
    parts: list[str] = []
    if extras.get("domain"):
        parts.append(extras["domain"])
        parts.append("")
    parts.append(base)
    for key in ("formula", "checklist", "step_hint"):
        value = extras.get(key)
        if value:
            parts.append("")
            parts.append(value)
    return "\n".join(parts)


def build_refinement_prompt(
    instruction: Instruction,
    previous_graph_serialized: str,
    feedback: str,
    round_index: int,
    graph_format: str = "code",
    extras: Optional[dict[str, str]] = None,
) -> str:
    """Build the VGIG round-t prompt (t >= 1) with verifier feedback.

    The prompt uses markdown-style section markers that LLMs reliably
    recognise. Keeps the revised graph in the same serialisation format.
    CAAP extras (checklist/formula/domain/step_hint) are re-injected so
    that augmentation persists across refinement rounds.
    """
    parts: list[str] = []
    parts.append(f"[Original instruction]\n{instruction.instruction}")

    all_constraints = list(instruction.explicit_constraints) + list(
        instruction.implicit_constraints
    )
    if all_constraints:
        parts.append("[Constraints]\n" + "\n".join(f"- {c}" for c in all_constraints))

    parts.append(
        f"[Your previous attempt (round {round_index})]\n"
        f"{previous_graph_serialized}"
    )
    parts.append(f"[Feedback from verifier]\n{feedback}")

    if extras:
        extra_parts: list[str] = []
        for key in ("domain", "formula", "checklist", "step_hint"):
            value = extras.get(key)
            if value:
                extra_parts.append(value)
        if extra_parts:
            parts.append("[Reference]\n" + "\n\n".join(extra_parts))

    parts.append(
        "[Task]\n"
        "Please output a REVISED graph that addresses ALL of the feedback "
        "above. Use the same "
        f"{graph_format}-style format as before. Only output the final "
        "graph, not commentary."
    )
    return "\n\n".join(parts)
