"""表型词表：6 个表型维度 + 1 个对照通道（全项目唯一的维度口径）。

**表型维度（6 维，`PHENOTYPE_DIMS`）** —— ③ 表型观察的读数与雷达图、提问者/分析者的
观察目标、条目库的维度归属、机制分析的端点维度都用它：

  social_interaction_motivation          社会互动与社会动机
  social_cognition                       社会认知
  social_communication_pragmatics        社会沟通与语用
  cognitive_flexibility_adaptation       认知灵活性与适应
  repetitive_restricted_behavior         重复性与限制性行为
  novelty_uncertainty_sensory_behavior   新颖性、不确定性与感觉相关行为

**对照通道（`CONTROL_DIM`，不是表型维度）** —— `general_cognition` 是 FR-12「阴性预测」与
FR-11 Control 1 的测量基础：任何针对某个机制的扰动都**不该**把一般认知一起搞坏，
否则说明发生了非特异的通用退化，该扰动不合格。因此它与表型维度分列两处：

- 它**不进雷达图**（雷达图画的是 6 个表型维度的形状）；
- 但它**要进分析者的评分通道**（`SCORED_CHANNELS`）—— 机制分析的特异性判定需要一个
  「应该不变」的对照读数，缺了它 `specificity.evaluate_specificity` 会对每个 primitive
  都判 `insufficient`（见 `analysis/specificity.py`）。落进剖面才有阴性预测可谈。

表型一律由 LLM 从对话里读出；注册表独立成模块 —— 词表不该依附于某个测量工具。
"""

from __future__ import annotations

#: 表型维度（6 维）：id → 中文名。③ 的雷达图、观察目标、条目库维度归属都用它。
PHENOTYPE_DIMS: dict[str, str] = {
    "social_interaction_motivation": "社会互动与社会动机",
    "social_cognition": "社会认知",
    "social_communication_pragmatics": "社会沟通与语用",
    "cognitive_flexibility_adaptation": "认知灵活性与适应",
    "repetitive_restricted_behavior": "重复性与限制性行为",
    "novelty_uncertainty_sensory_behavior": "新颖性、不确定性与感觉相关行为",
}

#: 每个通道的「看什么 / 低分什么样 / 高分什么样」。进提问者与分析者的提示词 ——
#: 6 维之间的边界比旧 5 维细（灵活 / 重复 / 新颖—不确定—感觉 要分开），
#: 只给 id 与标签会让分析者按各自的理解打分，故把锚点写死在注册表里。
#: 对照通道的锚点也在其中：它要被问到、被评分，就得先写清楚「看什么」。
DIM_ANCHORS: dict[str, dict[str, str]] = {
    "social_interaction_motivation": {
        "watch": "是否主动发起互动、是否把社会性信息当作有价值的目标（分享、招呼、注视、共同注意）。",
        "low": "极少主动发起社交，他人的表情与示好不足以驱动行为。",
        "high": "主动寻求社会性信息，互动本身具有吸引力。",
    },
    "social_cognition": {
        "watch": "能否表征他人的知识、意图与情绪并据此推断（错误信念、视角采择、意图与情绪推断）。",
        "low": "难以表征他人持有与自己不同的信息，倾向按自己知道的情况作答。",
        "high": "能区分自己与他人所知，并据此推断他人会怎么想、要做什么。",
    },
    "social_communication_pragmatics": {
        "watch": "在真实语境里如何使用与理解语言：话题维持、间接请求、反讽与字面义的取舍。",
        "low": "偏向字面理解，间接表达、反讽与语境依赖的说法容易误解；话题维持困难。",
        "high": "能结合语境理解间接表达与言外之意，会话连贯。",
    },
    "cognitive_flexibility_adaptation": {
        "watch": "规则或情境变化时能否调整：切换、更新、接受替代方案、从抗拒中恢复。",
        "low": "变化时明显不适应，坚持原有做法，切换代价高。",
        "high": "能较顺利地适应规则变化与新做法。",
    },
    "repetitive_restricted_behavior": {
        "watch": "重复动作与言语、仪式化流程、对同一性的要求、兴趣范围是否狭窄。",
        "low": "频繁的重复动作或仪式化要求，兴趣范围窄，被打破时反应强烈。",
        "high": "行为方式多样，没有明显的仪式化要求或狭窄兴趣。",
    },
    "novelty_uncertainty_sensory_behavior": {
        "watch": "对新颖与不确定的态度，以及感觉通道的反应（声音、触觉、光线、气味、疼痛）。",
        "low": "回避新异与不确定，感觉反应过强或过弱（如对声音过度敏感、寻求特定感觉输入）。",
        "high": "能容忍不确定并愿意尝试新异事物，感觉反应与刺激强度相称。",
    },
    # ---- 对照通道（会被评分，但不进雷达图）----
    "general_cognition": {
        "watch": "与本次社会性情境无关的通用能力：是否听懂任务要求、能否数数/用词/造句/简单推理、"
                 "能否记住并按要求操作。",
        "low": "连任务理解、语言表达与简单推理都明显吃力 —— 这提示整体资源下降，"
               "而不是某个机制的选择性影响。",
        "high": "任务理解、语言表达与简单推理与同龄相当。",
    },
}

#: 对照通道（**不是**表型维度）：用于阴性预测与特异性质检，不进雷达图。
CONTROL_DIM = "general_cognition"
CONTROL_DIM_ZH = "一般认知控制（对照通道）"

#: 全部会被计分并落库的通道 = 6 个表型维度 + 对照通道。
#: 分析者按这套通道评分；雷达图只画 `PHENOTYPE_DIMS` 那 6 条轴。
SCORED_CHANNELS: dict[str, str] = {**PHENOTYPE_DIMS, CONTROL_DIM: CONTROL_DIM_ZH}


def channel_label(channel: str) -> str:
    """通道 id → 中文标签（未知 id 原样返回，便于新通道先跑起来再补标签）。"""
    return SCORED_CHANNELS.get(channel, channel)
