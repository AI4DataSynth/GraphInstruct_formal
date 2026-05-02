"""Runner implementations for the improvement methods.

Each runner exposes ``run_instruction(instruction, model_name) -> RunnerResult``
and delegates LLM calls to an injected ``generate_fn`` with signature::

    generate_fn(
        instruction,
        *,
        num_samples=1,
        strategy="zero-shot",
        graph_format="code",
        temperature=0.7,
        override_prompt=None,
    ) -> list[GenerationResult]

``scripts.run_baseline.py`` wraps the concrete provider (mock / vllm /
openai / claude) into a closure that satisfies this signature, so runners
never need to know about api_base / model / top_p / max_tokens.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any

from graphinstruct.data_loader import Instruction
from graphinstruct.improvements.feedback import FeedbackLevel

logger = logging.getLogger(__name__)

# ``GenerationResult`` lives in ``scripts.run_baseline`` and is imported
# lazily inside runner methods to avoid a circular import at module load.
GenerationResult = Any
GenerateFn = Callable[..., Sequence[GenerationResult]]


@dataclass
class RunnerConfig:
    """Tunable configuration for all improvement runners."""

    method: str
    max_rounds: int = 3
    num_candidates: int = 3
    feedback_level: FeedbackLevel = "fine"
    graph_format: str = "code"
    temperature: float = 0.7
    num_samples: int = 3
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerResult:
    """Per-instruction runner output."""

    instruction_id: str
    level: int
    samples: list[Any]
    method_metadata: dict[str, Any] = field(default_factory=dict)


class BaseRunner(ABC):
    """Shared plumbing: config, injected provider, instruction index."""

    def __init__(
        self,
        config: RunnerConfig,
        generate_fn: GenerateFn,
        all_instructions: dict[int, list[Instruction]],
    ) -> None:
        self.config = config
        self.generate_fn = generate_fn
        self.all_instructions = all_instructions

    @abstractmethod
    def run_instruction(
        self, instruction: Instruction, model_name: str
    ) -> RunnerResult:
        """Process a single instruction and return ``num_samples`` samples."""

    def _score_sample(self, instruction: Instruction, sample: Any) -> float:
        """Return the constraint satisfaction rate for a sample.

        A sample is ``GenerationResult(instruction, graph, raw_output, ...)``.
        Returns 0.0 when the graph could not be parsed.
        """
        graph = getattr(sample, "graph", None)
        if graph is None:
            return 0.0
        all_constraints = list(instruction.explicit_constraints) + list(
            instruction.implicit_constraints
        )
        if not all_constraints:
            return 1.0
        try:
            from graphinstruct.validators.constraints import satisfaction_rate

            return satisfaction_rate(graph, all_constraints)
        except Exception:  # pragma: no cover - defensive
            return 0.0

    def _default_strategy(self) -> str:
        return self.config.extras.get("strategy", "zero-shot")

    def _build_base_prompt(self, instruction: Instruction, strategy: str) -> str:
        from graphinstruct.improvements.prompts import _import_base_build_prompt

        build_prompt = _import_base_build_prompt()
        return build_prompt(
            instruction,
            strategy,
            self.all_instructions,
            self.config.graph_format,
        )


class BaselineRunner(BaseRunner):
    """Reference implementation that mirrors the non-runner baseline path."""

    def run_instruction(
        self, instruction: Instruction, model_name: str
    ) -> RunnerResult:
        strategy = self._default_strategy()
        results = list(
            self.generate_fn(
                instruction,
                num_samples=self.config.num_samples,
                strategy=strategy,
                graph_format=self.config.graph_format,
                temperature=self.config.temperature,
                override_prompt=None,
            )
        )
        return RunnerResult(
            instruction_id=instruction.id,
            level=instruction.level,
            samples=results,
            method_metadata={"strategy": strategy},
        )


class RetryRunner(BaseRunner):
    """E2a: pure retry. T independent single-sample generations, no feedback.

    Early-breaks on full constraint satisfaction. Returns the top-
    ``num_samples`` collected samples ranked by satisfaction score.
    """

    def run_instruction(
        self, instruction: Instruction, model_name: str
    ) -> RunnerResult:
        strategy = self._default_strategy()
        max_rounds = max(self.config.max_rounds, self.config.num_samples)
        collected: list[Any] = []
        scores: list[float] = []
        rounds_used = 0

        for _ in range(max_rounds):
            rounds_used += 1
            batch = list(
                self.generate_fn(
                    instruction,
                    num_samples=1,
                    strategy=strategy,
                    graph_format=self.config.graph_format,
                    temperature=self.config.temperature,
                    override_prompt=None,
                )
            )
            if not batch:
                continue
            sample = batch[0]
            score = self._score_sample(instruction, sample)
            collected.append(sample)
            scores.append(score)
            if score >= 1.0 and len(collected) >= self.config.num_samples:
                break

        order = sorted(range(len(collected)), key=lambda i: scores[i], reverse=True)
        top_indices = order[: self.config.num_samples]
        top_samples = [collected[i] for i in top_indices]

        return RunnerResult(
            instruction_id=instruction.id,
            level=instruction.level,
            samples=top_samples,
            method_metadata={
                "strategy": strategy,
                "rounds_used": rounds_used,
                "scores": scores,
                "best_score": max(scores) if scores else 0.0,
            },
        )


class SCRunner(BaseRunner):
    """E2b: Self-Consistency.

    Draws ``num_candidates`` independent samples in one batched provider call,
    scores each, and returns the top-``num_samples``.
    """

    def run_instruction(
        self, instruction: Instruction, model_name: str
    ) -> RunnerResult:
        strategy = self._default_strategy()
        candidates = list(
            self.generate_fn(
                instruction,
                num_samples=self.config.num_candidates,
                strategy=strategy,
                graph_format=self.config.graph_format,
                temperature=self.config.temperature,
                override_prompt=None,
            )
        )
        scores = [self._score_sample(instruction, s) for s in candidates]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        top_indices = order[: self.config.num_samples]
        top_samples = [candidates[i] for i in top_indices]

        return RunnerResult(
            instruction_id=instruction.id,
            level=instruction.level,
            samples=top_samples,
            method_metadata={
                "strategy": strategy,
                "num_candidates": self.config.num_candidates,
                "scores": scores,
                "best_score": max(scores) if scores else 0.0,
            },
        )


class VGIGRunner(BaseRunner):
    """E2c: Verification-Guided Iterative Generation.

    Runs ``num_samples`` independent refinement chains. Each chain does at
    most ``max_rounds + 1`` LLM calls (1 for round 0, up to ``max_rounds``
    refinements). Returns the best-so-far sample per chain.
    """

    _CHAIN_TEMP_OFFSETS: tuple[float, ...] = (-0.1, 0.0, 0.1, -0.05, 0.05)
    _REFINEMENT_TEMP_FACTOR: float = 0.6
    _REFINEMENT_TEMP_MIN: float = 0.2

    def _chain_temperature(self, chain_idx: int) -> float:
        """Per-chain temperature with jitter for diversity (fix #5).

        When ``temperature == 0`` (greedy), jitter is skipped to preserve
        deterministic decoding intent.
        """
        base = self.config.temperature
        if base == 0.0:
            return 0.0
        offsets = self._CHAIN_TEMP_OFFSETS
        offset = offsets[chain_idx % len(offsets)]
        return max(0.1, min(1.5, base + offset))

    def _refinement_temperature(self, chain_idx: int) -> float:
        """Lowered temperature for refinement rounds (fix #7).

        Greedy (temp 0) is preserved as-is.
        """
        base = self._chain_temperature(chain_idx)
        if base == 0.0:
            return 0.0
        return max(
            self._REFINEMENT_TEMP_MIN,
            base * self._REFINEMENT_TEMP_FACTOR,
        )

    @staticmethod
    def _serialize_best(best: Any) -> str:
        """Re-serialize parsed graph for clean feedback; fall back to raw_output (fix #4)."""
        graph = getattr(best, "graph", None)
        if graph is not None:
            try:
                from graphinstruct.parser.code_style import serialize

                return serialize(graph)
            except Exception:
                pass
        return getattr(best, "raw_output", "") or ""

    def run_instruction(
        self, instruction: Instruction, model_name: str
    ) -> RunnerResult:
        from graphinstruct.improvements.caap import select_strategy
        from graphinstruct.improvements.feedback import generate_feedback
        from graphinstruct.improvements.prompts import (
            build_enhanced_prompt,
            build_refinement_prompt,
        )

        use_caap = bool(self.config.extras.get("use_caap"))
        if use_caap:
            decision = select_strategy(instruction, model_name)
            round0_strategy = decision.strategy
            round0_extras = decision.extras or {}
            caap_rationale = decision.rationale
            round0_prompt = build_enhanced_prompt(
                instruction,
                round0_strategy,
                self.all_instructions,
                self.config.graph_format,
                round0_extras,
            )
        else:
            round0_strategy = self._default_strategy()
            round0_extras = {}
            caap_rationale = ""
            round0_prompt = self._build_base_prompt(instruction, round0_strategy)

        max_rounds = self.config.max_rounds
        num_samples = self.config.num_samples
        feedback_level = self.config.feedback_level

        chain_samples: list[Any] = []
        chain_metadata: list[dict[str, Any]] = []

        for chain_idx in range(num_samples):
            chain_temp = self._chain_temperature(chain_idx)

            # Round 0
            round0 = list(
                self.generate_fn(
                    instruction,
                    num_samples=1,
                    strategy=round0_strategy,
                    graph_format=self.config.graph_format,
                    temperature=chain_temp,
                    override_prompt=round0_prompt,
                )
            )
            if not round0:
                continue
            best = round0[0]
            best_score = self._score_sample(instruction, best)
            round_scores = [best_score]
            round_sources = ["round0"]

            # Skip iteration for infeasible instructions (no point refining)
            feasible = getattr(instruction, "feasible", True)
            if not feasible:
                chain_samples.append(best)
                chain_metadata.append(
                    {
                        "chain": chain_idx,
                        "rounds": 1,
                        "scores": round_scores,
                        "sources": round_sources,
                        "best_score": best_score,
                        "infeasible": True,
                    }
                )
                continue

            refine_temp = self._refinement_temperature(chain_idx)
            rounds_run = 1
            consecutive_parse_fails = 0
            for t in range(1, max_rounds + 1):
                if best_score >= 1.0:
                    break

                # Fix #6: parse failure → regenerate fresh instead of
                # feeding broken output into refinement.
                best_graph = getattr(best, "graph", None)
                if best_graph is None:
                    consecutive_parse_fails += 1
                    if consecutive_parse_fails >= 2:
                        logger.warning(
                            "chain %d: %d consecutive parse failures, stopping",
                            chain_idx,
                            consecutive_parse_fails,
                        )
                        break
                    retry_batch = list(
                        self.generate_fn(
                            instruction,
                            num_samples=1,
                            strategy=round0_strategy,
                            graph_format=self.config.graph_format,
                            temperature=chain_temp,
                            override_prompt=round0_prompt,
                        )
                    )
                    rounds_run += 1
                    if retry_batch:
                        candidate = retry_batch[0]
                        candidate_score = self._score_sample(instruction, candidate)
                        round_scores.append(candidate_score)
                        round_sources.append(f"retry{t}")
                        if candidate_score > best_score:
                            best = candidate
                            best_score = candidate_score
                    continue

                consecutive_parse_fails = 0
                feedback = generate_feedback(
                    best_graph, instruction, level=feedback_level
                )
                if not feedback:
                    break

                prev_serialised = self._serialize_best(best)
                refinement_prompt = build_refinement_prompt(
                    instruction,
                    prev_serialised,
                    feedback,
                    t,
                    self.config.graph_format,
                    extras=round0_extras,
                )
                new_batch = list(
                    self.generate_fn(
                        instruction,
                        num_samples=1,
                        strategy=round0_strategy,
                        graph_format=self.config.graph_format,
                        temperature=refine_temp,
                        override_prompt=refinement_prompt,
                    )
                )
                rounds_run += 1
                if not new_batch:
                    continue
                candidate = new_batch[0]
                candidate_score = self._score_sample(instruction, candidate)
                round_scores.append(candidate_score)
                round_sources.append(f"round{t}")
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score

            chain_samples.append(best)
            chain_metadata.append(
                {
                    "chain": chain_idx,
                    "rounds": rounds_run,
                    "scores": round_scores,
                    "sources": round_sources,
                    "best_score": best_score,
                }
            )

        return RunnerResult(
            instruction_id=instruction.id,
            level=instruction.level,
            samples=chain_samples,
            method_metadata={
                "use_caap": use_caap,
                "strategy": round0_strategy,
                "caap_extras": list(round0_extras.keys()) if round0_extras else [],
                "caap_rationale": caap_rationale,
                "feedback_level": feedback_level,
                "chains": chain_metadata,
            },
        )


class CAAPRunner(BaseRunner):
    """E3: CAAP-only. Picks (strategy, extras) via the decision matrix and
    runs a single generation round (no iteration).
    """

    def run_instruction(
        self, instruction: Instruction, model_name: str
    ) -> RunnerResult:
        from graphinstruct.improvements.caap import select_strategy
        from graphinstruct.improvements.prompts import build_enhanced_prompt

        decision = select_strategy(instruction, model_name)
        enhanced_prompt = build_enhanced_prompt(
            instruction,
            decision.strategy,
            self.all_instructions,
            self.config.graph_format,
            decision.extras,
        )
        results = list(
            self.generate_fn(
                instruction,
                num_samples=self.config.num_samples,
                strategy=decision.strategy,
                graph_format=self.config.graph_format,
                temperature=self.config.temperature,
                override_prompt=enhanced_prompt,
            )
        )
        return RunnerResult(
            instruction_id=instruction.id,
            level=instruction.level,
            samples=results,
            method_metadata={
                "caap_strategy": decision.strategy,
                "caap_extras": list(decision.extras.keys()) if decision.extras else [],
                "caap_rationale": decision.rationale,
            },
        )


class CombinedRunner(VGIGRunner):
    """E4: VGIG + CAAP + domain priors."""

    def __init__(
        self,
        config: RunnerConfig,
        generate_fn: GenerateFn,
        all_instructions: dict[int, list[Instruction]],
    ) -> None:
        merged_config = dataclass_replace(
            config, extras={**config.extras, "use_caap": True}
        )
        super().__init__(merged_config, generate_fn, all_instructions)
