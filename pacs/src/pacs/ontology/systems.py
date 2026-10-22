"""Level 1：高阶认知的**组合定义**（FR-2）。

方法论红线：ToM / 认知灵活性 / 语用**不是**可扰动的基本单元，而是由 Level 0 primitive
组合**涌现**的高阶能力。本模块把每个高阶构念写成显式的 primitive 组合，并给出：

- `emergence_rule`：组合逻辑（哪一个成员承担什么角色）；
- `falsifiable_expectation`：可证伪预期 —— 单独扰动某成员应产生**部分**损伤，
  同时扰动多个成员应产生**更强**损伤。这由 FR-5 的组合扰动实测检验。

研究目标因此是「**ToM 为什么下降**」而不是「ToM 是否下降」。
"""

from __future__ import annotations

from dataclasses import dataclass

from .primitives import PRIMITIVE_INDEX


@dataclass(frozen=True)
class CognitiveSystem:
    """一个高阶认知构念 = 一组 primitive 的组合。"""

    id: str
    name_zh: str
    name_en: str
    members: tuple[str, ...]
    emergence_rule: str
    falsifiable_expectation: str
    behavioral_signature: str

    def member_weights(self, weights: dict[str, float] | None = None) -> dict[str, float]:
        """成员权重：缺省等权；可传入覆盖（用于特异化组合，如某成员承担主角色）。"""
        if weights:
            return {m: float(weights.get(m, 1.0)) for m in self.members}
        return {m: 1.0 for m in self.members}


SYSTEMS: tuple[CognitiveSystem, ...] = (
    CognitiveSystem(
        id="theory_of_mind",
        name_zh="心智理论",
        name_en="Theory of Mind",
        members=(
            "self_other_distinction",
            "other_state_representation",
            "agent_state_updating",
            "perspective_transformation",
            "representation_updating",
            "belief_updating",
            "relational_representation",
            "attention_shifting",
        ),
        emergence_rule=(
            "self_other_distinction 为表征打上「谁的」标签 → other_state_representation 构建"
            "他人当前的状态模型 → agent_state_updating 让该模型随他人获得的新信息推进 → "
            "perspective_transformation 以他人位置为原点重算「他看到的」 → representation_updating "
            "维持他人版本的信念而不被自身知识覆写 → belief_updating 依他人所见证据推进其信念 → "
            "relational_representation 编码「谁相对谁」的二阶关系 → attention_shifting 在自我与"
            "他人视角间往复（递归推断）。"
        ),
        falsifiable_expectation=(
            "单独扰动 self_other_distinction 或 belief_updating 应使错误信念任务部分损伤；"
            "同时扰动二者应产生**超加性**损伤（因为二者是错误信念的必要条件）。"
            "视角采择类任务应对 perspective_transformation 特别敏感，错误信念类任务应对 "
            "agent_state_updating 特别敏感 —— 二者可分离是 ToM 内部结构可检验的证据。"
        ),
        behavioral_signature="错误信念、意图推断、视角采择、情绪推断成绩下降。",
    ),
    CognitiveSystem(
        id="cognitive_flexibility",
        name_zh="认知灵活性",
        name_en="cognitive flexibility",
        members=(
            "representation_maintenance",
            "representation_updating",
            "inhibition",
            "prediction_error_sensitivity",
            "evidence_weighting",
            "switching_cost",
            "rule_updating",
            "goal_maintenance",
            "feature_binding",
        ),
        emergence_rule=(
            "representation_maintenance 保持当前规则 → prediction_error_sensitivity 标记规则"
            "失效 → evidence_weighting 判定新证据是否压过既有规则 → inhibition 压制旧规则的"
            "优势反应 → representation_updating 写入新规则 → rule_updating 完成控制参数替换 → "
            "switching_cost 决定任务集重设的代价阈值 → goal_maintenance 把新目标维持在线上以防"
            "被即时线索替换 → feature_binding 按新维度重组同一批特征。"
        ),
        falsifiable_expectation=(
            "单独扰动 switching_cost 应主要抬高切换代价（切换 > 重复试次）；同时扰动 inhibition 与 "
            "switching_cost 应出现**坚持性错误**（perseveration）而非仅变慢 —— 这是可区分的两种模式。"
            "单独扰动 rule_updating 应表现为「反馈已变仍按旧规则作答」，goal_maintenance 受损应表现为"
            "被即时线索牵引 —— 这两种错误类型与纯切换代价升高可区分。"
        ),
        behavioral_signature="规则坚持、切换代价升高、更新延迟增加、重复偏好上升。",
    ),
    CognitiveSystem(
        id="pragmatic_interpretation",
        name_zh="语用解释",
        name_en="pragmatic interpretation",
        members=(
            "representation_maintenance",
            "salience_assignment",
            "information_relevance_weighting",
            "self_other_distinction",
            "other_state_representation",
            "belief_updating",
            "ambiguity_tolerance",
        ),
        emergence_rule=(
            "representation_maintenance 保持语境 → salience_assignment 让语境线索（而非字面"
            "词义）成为显著输入 → information_relevance_weighting 判定哪些线索与说话人意图"
            "相关 → self_other_distinction 表征说话人状态 → other_state_representation 构建"
            "说话人此刻的意图模型 → belief_updating 推断说话人意图 → ambiguity_tolerance "
            "允许多解共存直至语境消解。"
        ),
        falsifiable_expectation=(
            "单独扰动 ambiguity_tolerance 应产生**字面化**倾向（隐喻/反讽/间接请求成绩下降）"
            "而语言基础能力（词法、句法）不受影响 —— 这正是 FR-9 的特异性检验目标。"
        ),
        behavioral_signature="字面解释、间接言语与反讽失败、语境依赖解读下降。",
    ),
)

SYSTEM_INDEX: dict[str, CognitiveSystem] = {s.id: s for s in SYSTEMS}


def get(system_id: str) -> CognitiveSystem:
    try:
        return SYSTEM_INDEX[system_id]
    except KeyError:
        known = ", ".join(sorted(SYSTEM_INDEX))
        raise KeyError(f"未知 Level 1 构念：{system_id}（已知：{known}）") from None


def systems_of(primitive_id: str) -> tuple[str, ...]:
    """某个 primitive 参与了哪些高阶构念（用于「扰动一个 primitive → 多个高阶能力受影响」）。"""
    return tuple(s.id for s in SYSTEMS if primitive_id in s.members)


def validate_systems() -> list[str]:
    """自检：所有成员必须在 Level 0 注册（FR-2 要求 2）。"""
    problems: list[str] = []
    seen: set[str] = set()
    for s in SYSTEMS:
        if s.id in seen:
            problems.append(f"system id 重复：{s.id}")
        seen.add(s.id)
        if len(s.members) < 2:
            problems.append(f"{s.id} 的成员少于 2 个，不构成组合定义")
        for m in s.members:
            if m not in PRIMITIVE_INDEX:
                problems.append(f"{s.id} 引用了未注册的 primitive：{m}")
        if not s.falsifiable_expectation.strip():
            problems.append(f"{s.id} 缺少可证伪预期（FR-2 要求 3）")
    return problems


def combination_candidates(system_id: str, levels: int = 2) -> list[tuple[str, ...]]:
    """返回该高阶构念的 k 阶组合候选（用于 FR-5 组合扰动设计）。

    levels=2 → 所有二元组合（P_i + P_j）。这是检验「可加性 / 超加性」的最小设计。
    """
    import itertools

    s = get(system_id)
    return [tuple(c) for c in itertools.combinations(s.members, levels)]
