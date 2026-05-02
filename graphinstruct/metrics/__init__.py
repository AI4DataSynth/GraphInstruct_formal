"""Evaluation metrics for GraphInstruct benchmark.

Dimensions:
- D1 (structural): Valid Rate, GED, MMD
- D2 (textual): G-BERTScore, G-BLEU, G-ROUGE, Text-F1 (active at L3+)
- D3 (embedding): Grassmann Coherence, Node Clf. Gap, Emb. MMD (active at L3+)
- D4 (instruction): Explicit/Implicit Satisfaction, No-Contradiction
- D5 (efficiency): Tokens/Valid Graph, API Calls, Pareto Score
"""

from graphinstruct.metrics.structural import valid_rate, valid_rate_single, uniqueness
from graphinstruct.metrics.instruction import (
    explicit_satisfaction,
    implicit_satisfaction,
    instruction_score,
    no_contradiction,
)
from graphinstruct.metrics.efficiency import (
    TokenRecord,
    api_calls_per_graph,
    pareto_score,
    tokens_per_valid_graph,
)
from graphinstruct.metrics.textual import (
    d2_score,
    extract_triples,
    g_bertscore,
    g_bleu,
    g_rouge,
    text_f1,
)
from graphinstruct.metrics.embedding import (
    d3_score,
    embedding_mmd,
    grassmann_coherence,
    node_clf_gap,
)
from graphinstruct.metrics.micro_eval import (
    MicroEvalResult,
    ReasoningStep,
    StepResult,
    build_reasoning_steps_json,
    decompose_instruction,
    evaluate_reasoning_chain,
    evaluate_step,
)

__all__ = [
    # D1
    "valid_rate",
    "valid_rate_single",
    "uniqueness",
    # D2
    "d2_score",
    "extract_triples",
    "g_bertscore",
    "g_bleu",
    "g_rouge",
    "text_f1",
    # D3
    "d3_score",
    "embedding_mmd",
    "grassmann_coherence",
    "node_clf_gap",
    # D4
    "explicit_satisfaction",
    "implicit_satisfaction",
    "instruction_score",
    "no_contradiction",
    # D5
    "TokenRecord",
    "api_calls_per_graph",
    "pareto_score",
    "tokens_per_valid_graph",
    # Micro evaluation (L5)
    "MicroEvalResult",
    "ReasoningStep",
    "StepResult",
    "build_reasoning_steps_json",
    "decompose_instruction",
    "evaluate_reasoning_chain",
    "evaluate_step",
]
