"""brain 包：分阶段认知架构提示词（人造大脑初始化 / 任务 / 注入）与扰动引擎。"""

from .architecture import (
    brain_fingerprint,
    build_brain_prompt,
    build_brain_prompt_steps,
    build_brain_turn,
    build_init_prompt,
)
from . import prompts
from .perturb import (
    LHS_LOW,
    Perturbation,
    compose,
    from_values,
    generic_degradation_control,
    intact,
    intact_values,
    interaction_sign,
    sample_design,
    sample_lhs,
    sample_pairs,
    sample_singles,
    single,
)

__all__ = [
    "LHS_LOW",
    "Perturbation",
    "brain_fingerprint",
    "build_brain_prompt",
    "build_brain_prompt_steps",
    "build_brain_turn",
    "build_init_prompt",
    "prompts",
    "compose",
    "from_values",
    "generic_degradation_control",
    "intact",
    "intact_values",
    "interaction_sign",
    "sample_design",
    "sample_lhs",
    "sample_pairs",
    "sample_singles",
    "single",
]
