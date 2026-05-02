"""GraphInstruct improvements: VGIG, CAAP, domain priors, SC/Retry baselines.

Importable components:
    Runners:  BaselineRunner, RetryRunner, SCRunner, VGIGRunner, CAAPRunner,
              CombinedRunner
    Config:   RunnerConfig, RunnerResult
    Feedback: generate_feedback, FeedbackLevel
    Violation: ViolationDetail, ConstraintCategory, extract_violations,
               classify_constraint
    CAAP:     select_strategy, CAAPDecision
    Domain:   DOMAIN_PRIORS, DomainPrior, build_domain_text, get_prior
    Registry: MODEL_TIERS, ModelTier, model_tier, is_infeasible
    Prompts:  build_enhanced_prompt, build_refinement_prompt
"""

from graphinstruct.improvements.caap import (
    CAAPDecision,
    Strategy,
    select_strategy,
)
from graphinstruct.improvements.domain_priors import (
    DOMAIN_PRIORS,
    DomainPrior,
    build_domain_text,
    get_prior,
)
from graphinstruct.improvements.feedback import (
    FeedbackLevel,
    generate_feedback,
)
from graphinstruct.improvements.prompts import (
    build_enhanced_prompt,
    build_refinement_prompt,
)
from graphinstruct.improvements.registry import (
    MODEL_TIERS,
    ModelTier,
    is_infeasible,
    model_tier,
)
from graphinstruct.improvements.runners import (
    BaseRunner,
    BaselineRunner,
    CAAPRunner,
    CombinedRunner,
    RetryRunner,
    RunnerConfig,
    RunnerResult,
    SCRunner,
    VGIGRunner,
)
from graphinstruct.improvements.violation import (
    ConstraintCategory,
    ViolationDetail,
    classify_constraint,
    extract_violations,
)

__all__ = [
    # runners
    "BaseRunner",
    "BaselineRunner",
    "RetryRunner",
    "SCRunner",
    "VGIGRunner",
    "CAAPRunner",
    "CombinedRunner",
    "RunnerConfig",
    "RunnerResult",
    # feedback / violation
    "generate_feedback",
    "FeedbackLevel",
    "ViolationDetail",
    "ConstraintCategory",
    "extract_violations",
    "classify_constraint",
    # caap
    "select_strategy",
    "CAAPDecision",
    "Strategy",
    # domain priors
    "DOMAIN_PRIORS",
    "DomainPrior",
    "build_domain_text",
    "get_prior",
    # registry
    "MODEL_TIERS",
    "ModelTier",
    "model_tier",
    "is_infeasible",
    # prompts
    "build_enhanced_prompt",
    "build_refinement_prompt",
]
