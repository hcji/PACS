"""行为语言化（FR-8）：`B → 标准化自然语言描述`，供 Model 1 消费。

**反循环要求（关键）**：描述必须**只由可观测行为**生成，绝不能读出 θ 的数值。
否则 Model 1 学到的将是「模板的逆运算」而不是「行为的逆运算」，整条链路就失去意义。
`anti_circularity_check()` 是这条红线的机器化守卫，且有单元测试覆盖。

同时产出**量表锚点**（`assessments`）：把表型维度映射到引擎既有量表的 `tool_id`
（仅作**字符串引用**，不复制引擎题库），作为 Mode B 大幅评估入口。
"""

from __future__ import annotations

import re
from typing import Optional

from .. import DEFAULT_AGE_MONTHS
from ..phenotype import PhenotypeProfile

# 维度 → 措辞模板：按高/中/低三档给出行为化描述。
# 措辞只依赖表型（正确率），不依赖任何 θ 数值。
DIM_TEMPLATES: dict[str, dict[str, str]] = {
    "social_interaction_motivation": {
        "low": "主动发起社交互动较少，他人的社交回应有限；在可选择时较少优先注意社会性信息。",
        "mid": "在熟悉情境下会参与社交，主动发起与信息寻求的主动性一般。",
        "high": "会主动寻求社会性信息，他人表情与互动本身具有吸引力。",
    },
    "social_cognition": {
        "low": "在推断他人想法方面存在明显困难：当他人的看法与自己已知的事实不一致时，倾向于按自己知道的情况作答。",
        "mid": "推断他人想法时表现不稳定，简单情形可以应对，涉及他人持有不同信息时容易出错。",
        "high": "能够区分自己与他人所掌握的信息，可以据此推断他人会怎么想。",
    },
    "social_communication_pragmatics": {
        "low": "对语言的理解偏向字面：对间接提出的请求、反讽与依赖语境的表达容易按字面意思理解；会话中话题维持困难。",
        "mid": "日常交流基本可行，但对间接表达和反讽的理解不够稳定，话题维持需要一些支持。",
        "high": "能够结合语境理解间接表达、反讽与言外之意，并能维持话题。",
    },
    "cognitive_flexibility_adaptation": {
        "low": "在规则或活动顺序发生变化时明显不适应，倾向于坚持原有做法，切换代价高。",
        "mid": "面对变化时有一定抵触，需要一些时间才能调整过来。",
        "high": "能够较顺利地适应规则变化与新做法。",
    },
    "repetitive_restricted_behavior": {
        "low": "反复出现重复动作或仪式化流程，对同一性要求强，兴趣范围狭窄，流程被打破时反应强烈。",
        "mid": "有一些重复偏好或固定流程，被打破时需要一点时间才能恢复。",
        "high": "行为方式多样，没有明显的仪式化要求或狭窄兴趣。",
    },
    "novelty_uncertainty_sensory_behavior": {
        "low": "回避新异与不确定的情境，感觉反应过强或过弱（如对声音过度敏感、寻求特定感觉输入）。",
        "mid": "对新异与不确定持谨慎态度，但多数情况下可以参与。",
        "high": "愿意尝试新异事物，能容忍不确定，感觉反应与刺激强度相称。",
    },
    "general_cognition": {
        "low": "一般认知表现低于同龄水平（含语言、推理、工作记忆与事实知识）。",
        "mid": "一般认知表现接近同龄水平。",
        "high": "一般认知表现与同龄儿童相当。",
    },
}

# 通道 → 引擎既有量表 tool_id（**字符串引用**，不复制题库）
DIM_TO_TOOL: dict[str, str] = {
    "social_interaction_motivation": "SRS",
    "social_cognition": "SRS",
    "social_communication_pragmatics": "Vineland",
    "cognitive_flexibility_adaptation": "SRS",
    "repetitive_restricted_behavior": "SRS",
    "novelty_uncertainty_sensory_behavior": "SRS",
    "general_cognition": "MSEL",
}

# 分档阈值针对 **chance-corrected** 得分（完整心智 ≈ 0.92；强扰动可降至 0.4 以下）
LOW_BAND = 0.60
HIGH_BAND = 0.80


def _band(value: Optional[float]) -> str:
    if value is None:
        return "mid"
    if value < LOW_BAND:
        return "low"
    if value >= HIGH_BAND:
        return "high"
    return "mid"


def _age_phrase(age_months: int) -> str:
    years, months = divmod(max(0, age_months), 12)
    if years and months:
        return f"{years} 岁 {months} 个月"
    if years:
        return f"{years} 岁"
    return f"{months} 个月"


def verbalize_profile(profile: PhenotypeProfile) -> str:
    """把表型转成一段标准化自然语言描述。

    年龄用常量而不是 `profile.age_months`：心智年龄固定（见 `DEFAULT_AGE_MONTHS`），
    读剖面字段会把旧数据里别的月龄写进描述，让同一批描述看起来像跨年龄段的研究。
    """
    sentences = [f"该儿童 {_age_phrase(DEFAULT_AGE_MONTHS)}。"]
    for dim, templates in DIM_TEMPLATES.items():
        sentences.append(templates[_band(profile.dims.get(dim))])
    return "".join(sentences)


def build_descriptions(
    profile: PhenotypeProfile,
    *,
    source: str = "PACS 仿真描述（模板生成）",
    update_kind: str = "small",
    observation_confidence: float = 0.5,
    observation_strength: float = 0.3,
) -> list[dict]:
    """构造引擎 `DescriptionEntry` 对齐的描述条目（PRD §6.4）。"""
    return [
        {
            "text": verbalize_profile(profile),
            "source": source,
            "ts": profile.ts,
            "update_kind": update_kind,
            "observation_confidence": observation_confidence,
            "observation_strength": observation_strength,
        }
    ]


def build_assessments(profile: PhenotypeProfile) -> list[dict]:
    """构造量表锚点（Mode B 入口）。`answers` 为空对象：本管线不产生真实逐题作答。"""
    seen: list[str] = []
    for dim in profile.dims:
        tool = DIM_TO_TOOL.get(dim)
        if tool and tool not in seen:
            seen.append(tool)
    return [
        {
            "tool_id": tool,
            "ts": profile.ts,
            "answers": {},
            "observation_confidence": 0.8,
            "observation_strength": 0.8,
        }
        for tool in seen
    ]


# ---------------------------------------------------------------------------
# 反循环守卫（FR-8 要求 5）
# ---------------------------------------------------------------------------

_NUMERIC_ASSIGNMENT = re.compile(r"[=＝]\s*0?\.\d+")
_PRIMITIVE_ID_HINT = re.compile(r"\b[a-z]+_[a-z_]+\b")


def anti_circularity_check(text: str, primitive_ids: tuple[str, ...] | None = None) -> list[str]:
    """检查描述文本是否泄露了 θ 的内部信息。返回问题清单（空 = 合规）。"""
    problems: list[str] = []
    if _NUMERIC_ASSIGNMENT.search(text):
        problems.append("描述中出现了「= 0.xx」形式的数值赋值，可能泄露维度数值")
    from ..ontology.primitives import DEFAULT_ACTIVE

    ids = primitive_ids or DEFAULT_ACTIVE
    for pid in ids:
        if pid in text:
            problems.append(f"描述中出现了 primitive id「{pid}」，泄露扰动内部构造")
    for token in _PRIMITIVE_ID_HINT.findall(text):
        if token in ids:
            problems.append(f"描述中出现了 primitive 标识「{token}」")
    return sorted(set(problems))
