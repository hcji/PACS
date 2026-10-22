"""agents 包：虚拟被试、对话探针与 LLM 涌现表型测量。"""

from .measure import MeasureResult, measure_phenotype, readings_from_feedback
from .verbalize import (
    anti_circularity_check,
    build_assessments,
    build_descriptions,
    verbalize_profile,
)
from .virtual_agent import BrainInitResult, VirtualAgent

__all__ = [
    "BrainInitResult",
    "MeasureResult",
    "VirtualAgent",
    "anti_circularity_check",
    "build_assessments",
    "build_descriptions",
    "measure_phenotype",
    "readings_from_feedback",
    "verbalize_profile",
]
