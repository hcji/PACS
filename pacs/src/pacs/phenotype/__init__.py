"""表型词表与剖面：6 个表型维度 + 1 个对照通道，以及「θ → B」的剖面记录。

表型一律由 LLM 读出（提问者提问 → 心智作答 → 分析者逐维读数），本包不含任何规则式测量。
"""

from .conclusion import (
    ASD_LIKENESS_LEVELS,
    ASD_LIKENESS_LEVEL_ZH,
    CONCLUSION_CHARS_MAX,
    CONCLUSION_CHARS_MIN,
    CONCLUSION_CHARS_TARGET,
    CONCLUSION_MAX_CHARS,
    DIAGNOSIS_WORDS,
    diagnosis_wording,
    normalize_asd_likeness,
    normalize_level,
)
from .dims import (
    CONTROL_DIM,
    CONTROL_DIM_ZH,
    DIM_ANCHORS,
    PHENOTYPE_DIMS,
    SCORED_CHANNELS,
    channel_label,
)
from .profile import (
    PhenotypeProfile,
    aggregate_baseline,
    all_controls_intact,
    control_channel_value,
    now_iso,
    synthetic_count,
    uncovered_counts,
)

__all__ = [
    "ASD_LIKENESS_LEVELS",
    "ASD_LIKENESS_LEVEL_ZH",
    "CONCLUSION_CHARS_MAX",
    "CONCLUSION_CHARS_MIN",
    "CONCLUSION_CHARS_TARGET",
    "CONCLUSION_MAX_CHARS",
    "CONTROL_DIM",
    "CONTROL_DIM_ZH",
    "DIAGNOSIS_WORDS",
    "DIM_ANCHORS",
    "PHENOTYPE_DIMS",
    "SCORED_CHANNELS",
    "PhenotypeProfile",
    "aggregate_baseline",
    "all_controls_intact",
    "channel_label",
    "control_channel_value",
    "diagnosis_wording",
    "normalize_asd_likeness",
    "normalize_level",
    "now_iso",
    "synthetic_count",
    "uncovered_counts",
]
