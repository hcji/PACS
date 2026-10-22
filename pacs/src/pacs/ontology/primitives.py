"""Level 0：认知基本单元本体（FR-1）。

31 个可独立扰动的 cognitive primitive，归入 8 大家族。每个 primitive 必须满足五项标准
（FR-1）：
  1. 有定义      —— `definition` 可写成一句话的计算职责；
  2. 有独立 assay —— 见 `assays.py`（由条目库按敏感度派生）；
  3. 可单独扰动   —— `Perturbation` 只作用于目标 primitive；
  4. 有 dose-response —— `dose_levels` 连续梯度；
  5. 有 specificity control —— `assays.py` 给出零敏感度对照条目。

**红线一**：Level 1 高阶认知（ToM / 认知灵活性 / 语用）不在本表内，它们由这些 primitive
组合涌现（`systems.py`），不作为扰动目标。

**红线二**：本表**不声明** primitive 与测量维度之间的对应关系。既没有「该机制影响哪个
外部模型维度」的字段，也没有「该机制应体现在哪个表型维度」的字段。
测量维度读数一律由分析者 LLM 从对话得出，判据由数据本身给出；
见 `assays.py` 的模块红线与 `README.md`「扰动与表型之间不允许有任何已知条件的关联」。
"""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES: dict[str, str] = {
    "A_information_processing": "A 信息处理",
    "B_representation": "B 表征",
    "C_attention": "C 注意",
    "D_prediction": "D 预测",
    "E_uncertainty": "E 不确定性",
    "F_value": "F 价值",
    "G_control": "G 控制",
    "H_agent_social_state": "H 主体/社会状态表征",
}

DEFAULT_DOSE_LEVELS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
INTACT = 1.0


@dataclass(frozen=True)
class Primitive:
    """一个认知基本单元。

    `impairment_signature` 描述该 primitive 被削弱时，行为上应当出现什么 ——
    这是提示词扰动算子与后续特异性检验的文本依据，也是与临床描述对接的桥梁。
    """

    id: str
    name_zh: str
    name_en: str
    family: str
    definition: str
    computational_role: str
    impairment_signature: str
    dose_levels: tuple[float, ...] = DEFAULT_DOSE_LEVELS
    status: str = "core"                    # core | candidate | rejected

    @property
    def family_zh(self) -> str:
        return FAMILIES.get(self.family, self.family)


# ---------------------------------------------------------------------------
# 31 个 core primitive（V2 全量本体；候补项见 CANDIDATES）
# ---------------------------------------------------------------------------

PRIMITIVES: tuple[Primitive, ...] = (
    # ---- A 信息处理 ----
    Primitive(
        id="salience_assignment",
        name_zh="显著性分配",
        name_en="salience assignment",
        family="A_information_processing",
        definition="决定哪些输入被赋予高显著性并优先进入后续加工。",
        computational_role="在输入的显著度竞争中分配注意资源，决定什么进入加工瓶颈。",
        impairment_signature="社会性显著线索（面孔、声音、注视方向）不再优先；细节与物理特征占据显著性。",
    ),
    Primitive(
        id="signal_noise_discrimination",
        name_zh="信噪判别",
        name_en="signal/noise discrimination",
        family="A_information_processing",
        definition="从背景噪声中分离出信号。",
        computational_role="估计输入通道的信噪比，决定信号的可提取程度。",
        impairment_signature="信噪比失衡——感觉通道过载，微弱变化被放大为强信号。",
    ),
    Primitive(
        id="information_relevance_weighting",
        name_zh="信息相关性加权",
        name_en="information relevance weighting",
        family="A_information_processing",
        definition="按与当前目标的相关性对输入信息加权。",
        computational_role="估计输入与当前目标、任务与语境的相关度，据此调节其加工权重。",
        impairment_signature="相关性加权失准——无关细节与目标相关信息被同等加工，难以筛出要紧的部分。",
    ),
    Primitive(
        id="local_global_weighting",
        name_zh="局部—整体信息权衡",
        name_en="local–global weighting",
        family="A_information_processing",
        definition="在局部细节与整体构型之间分配加工权重。",
        computational_role="决定当前判断由局部特征主导还是由整体构型主导。",
        impairment_signature="局部权重过高，整体构型（global form）被忽略；只见细节不见全貌。",
    ),
    # ---- B 表征 ----
    Primitive(
        id="representation_maintenance",
        name_zh="表征维持",
        name_en="representation maintenance",
        family="B_representation",
        definition="在工作表征中保持信息不衰减。",
        computational_role="在时间上维持当前活跃表征，供后续操作使用。",
        impairment_signature="表征易流失，需要外部重复与仪式化来维持；偏好可预测的重复。",
    ),
    Primitive(
        id="representation_updating",
        name_zh="表征更新",
        name_en="representation updating",
        family="B_representation",
        definition="按新输入改写既有表征。",
        computational_role="用新证据替换或修正当前表征内容。",
        impairment_signature="表征更新困难，坚持旧表征；环境变化后行为不随之调整。",
    ),
    Primitive(
        id="feature_binding",
        name_zh="特征绑定",
        name_en="feature binding",
        family="B_representation",
        definition="把分散的特征整合为单一客体表征。",
        computational_role="把颜色、形状、位置、动作等特征通道绑定到同一客体上。",
        impairment_signature="特征绑定松散，客体被拆成互不相关的属性；难以按当前维度重组同一批特征。",
    ),
    Primitive(
        id="relational_representation",
        name_zh="关系表征",
        name_en="relational representation",
        family="B_representation",
        definition="表征对象之间的关系而非单个对象本身。",
        computational_role="把「谁相对谁、谁在谁之前、谁与谁同属」编码为可操作的关系结构。",
        impairment_signature="关系结构难以提取，倾向逐个对象处理；类比、比喻与二阶关系理解受损。",
    ),
    Primitive(
        id="self_other_distinction",
        name_zh="自我—他人区分",
        name_en="self–other distinction",
        family="B_representation",
        definition="区分自身状态与他人状态。",
        computational_role="为表征打上「谁的」标签，是意图推断与视角采择的前提。",
        impairment_signature="自我—他人边界模糊，难以表征他人持有与自己不同的信息。",
    ),
    # ---- C 注意 ----
    Primitive(
        id="attention_shifting",
        name_zh="注意转移",
        name_en="attention shifting",
        family="C_attention",
        definition="在对象或规则之间解除并转移注意。",
        computational_role="解除当前注意锁定并把资源重新分配到新目标。",
        impairment_signature="注意粘滞，转移代价高；任务切换时停留在先前规则上。",
    ),
    Primitive(
        id="attention_persistence",
        name_zh="注意维持",
        name_en="attention persistence",
        family="C_attention",
        definition="在一段时间内持续锁定同一对象。",
        computational_role="抵抗干扰，保持注意在目标上不漂移。",
        impairment_signature="过度维持——兴趣范围狭窄，难以离开当前活动。",
    ),
    Primitive(
        id="selective_attention",
        name_zh="选择性注意",
        name_en="selective attention",
        family="C_attention",
        definition="在竞争性输入中选出一个通道并压制其余通道。",
        computational_role="按当前目标在多个并行输入中做取舍，抑制非目标通道。",
        impairment_signature="选择失衡——非目标通道无法被压制，注意被环境中最强的物理刺激拉走。",
    ),
    # ---- D 预测 ----
    Primitive(
        id="prior_strength",
        name_zh="先验强度",
        name_en="prior strength",
        family="D_prediction",
        definition="既有预期在知觉与推断中的权重。",
        computational_role="为后续预测提供初始分布，并决定该分布被改动所需的证据量。",
        impairment_signature="先验过强——既有预期难以被输入改写，变化被当作误差而非新信息。",
    ),
    Primitive(
        id="prediction_error_sensitivity",
        name_zh="预测误差敏感性",
        name_en="prediction-error sensitivity",
        family="D_prediction",
        definition="对违反预期的输入的敏感程度。",
        computational_role="计算预测与实际输入之间的偏差，并决定其权重。",
        impairment_signature="对微小预期违背反应过强，变化引发强烈应激。",
    ),
    Primitive(
        id="evidence_weighting",
        name_zh="证据权重",
        name_en="evidence weighting",
        family="D_prediction",
        definition="对新证据相对既有信念的权重赋值。",
        computational_role="按证据的可靠性与时效性决定其在推断中的份量。",
        impairment_signature="证据权重失当——新证据不足以改写判断，或以过强权重覆盖稳定知识。",
    ),
    Primitive(
        id="belief_updating",
        name_zh="信念更新",
        name_en="belief updating",
        family="D_prediction",
        definition="依据新证据改写信念。",
        computational_role="按证据强度对信念分布做后验修正。",
        impairment_signature="信念更新不足，坚持已知答案与既有规则。",
    ),
    # ---- E 不确定性 ----
    Primitive(
        id="ambiguity_tolerance",
        name_zh="歧义耐受",
        name_en="ambiguity tolerance",
        family="E_uncertainty",
        definition="容忍多义或不确定的信息而不拒绝加工。",
        computational_role="在信息不完整或存在多解时维持加工而不崩溃。",
        impairment_signature="歧义耐受低，要求字面与唯一确定的解释，拒绝隐喻、反讽与间接表达。",
    ),
    Primitive(
        id="uncertainty_estimation",
        name_zh="不确定性估计",
        name_en="uncertainty estimation",
        family="E_uncertainty",
        definition="估计当前判断的不确定程度。",
        computational_role="为当前表征与预测输出一个不确定性量，供后续策略选择使用。",
        impairment_signature="不确定性估计失准——或低估不确定而草率结论，或高估而不敢行动。",
    ),
    Primitive(
        id="information_seeking_under_uncertainty",
        name_zh="不确定性下的信息寻求",
        name_en="information-seeking under uncertainty",
        family="E_uncertainty",
        definition="主动取样以降低不确定性。",
        computational_role="在不确定时产生信息获取行为（提问、探索、检验）。",
        impairment_signature="不主动提问或探索，不确定性长期滞留。",
    ),
    Primitive(
        id="confidence_calibration",
        name_zh="置信度校准",
        name_en="confidence calibration",
        family="E_uncertainty",
        definition="使主观置信度与实际正确率保持一致。",
        computational_role="把内部置信信号对齐到外部可验证的准确率上。",
        impairment_signature="校准失配——高置信地坚持错误解读，或对已掌握的判断缺乏确定感。",
    ),
    # ---- F 价值 ----
    Primitive(
        id="social_reward_weighting",
        name_zh="社会奖励加权",
        name_en="social reward weighting",
        family="F_value",
        definition="社会性线索相对非社会线索的价值权重。",
        computational_role="在价值计算中为社会来源的输入施加相对增益。",
        impairment_signature="社会信息的主观价值降低，社交互动不足以驱动行为。",
    ),
    Primitive(
        id="novelty_valuation",
        name_zh="新颖性估值",
        name_en="novelty valuation",
        family="F_value",
        definition="新颖刺激相对熟悉刺激的价值。",
        computational_role="为尚无表征的刺激分配探索价值。",
        impairment_signature="新颖性估值低，偏好熟悉与可预测，回避新异。",
    ),
    Primitive(
        id="familiarity_valuation",
        name_zh="熟悉性估值",
        name_en="familiarity valuation",
        family="F_value",
        definition="熟悉刺激所携带的确定感与可用性价值。",
        computational_role="为已编码的刺激赋予检索优势与确定感，使其在竞争中优先被取用。",
        impairment_signature="熟悉性既不能带来确定感也不能带来便利——惯用表达与既有经验难以被调用。",
    ),
    Primitive(
        id="information_valuation",
        name_zh="信息价值",
        name_en="information valuation",
        family="F_value",
        definition="信息本身（而非其带来的奖赏）所具有的价值。",
        computational_role="为信息缺口与信息获取行为分配内在价值，驱动非工具性求知。",
        impairment_signature="信息本身不再有价值，只追求结果性奖赏，缺少「想知道」的驱动。",
    ),
    # ---- G 控制 ----
    Primitive(
        id="inhibition",
        name_zh="抑制",
        name_en="inhibition",
        family="G_control",
        definition="抑制优势或预激活的反应。",
        computational_role="在竞争性反应中压制不合当前目标的那一个。",
        impairment_signature="抑制弱，优势反应难以压制；或过度抑制，反应迟滞僵化。",
    ),
    Primitive(
        id="switching_cost",
        name_zh="切换成本",
        name_en="switching cost",
        family="G_control",
        definition="在规则或反应集之间切换所需付出的代价。",
        computational_role="按当前情境线索重设任务集，并决定切换的代价阈值。",
        impairment_signature="切换代价高——分离（separation）与坚持性（perseveration）错误增多。",
    ),
    Primitive(
        id="goal_maintenance",
        name_zh="目标维持",
        name_en="goal maintenance",
        family="G_control",
        definition="在干扰与延迟中保持当前目标在线。",
        computational_role="把当前目标维持为活跃控制信号，抑制与之竞争的其它目标。",
        impairment_signature="目标易被当前刺激替换，行为被即时线索牵引而偏离原定目标。",
    ),
    Primitive(
        id="rule_updating",
        name_zh="规则更新",
        name_en="rule updating",
        family="G_control",
        definition="按反馈替换当前生效的任务规则。",
        computational_role="监测反馈并把控制参数切换到新的规则集。",
        impairment_signature="规则固化——反馈已指示规则变了，仍在旧规则下作答。",
    ),
    # ---- H 主体/社会状态表征 ----
    Primitive(
        id="other_state_representation",
        name_zh="他人状态表征",
        name_en="other-state representation",
        family="H_agent_social_state",
        definition="表征他人当前的内部状态（情绪、意图、知识、期待）。",
        computational_role="从外显线索构建关于他人在此刻处于什么状态的模型。",
        impairment_signature="他人内部状态表征薄弱，只看行为不看意图；社会线索不被读作状态信息。",
    ),
    Primitive(
        id="agent_state_updating",
        name_zh="主体状态更新",
        name_en="agent-state updating",
        family="H_agent_social_state",
        definition="随新信息改写关于某个主体的状态模型。",
        computational_role="当他人获得新信息或处境变化时，更新其状态模型而非沿用旧值。",
        impairment_signature="他人状态模型不随新信息改写，按对方「本来知道/本来不知道」作答。",
    ),
    Primitive(
        id="perspective_transformation",
        name_zh="视角转换",
        name_en="perspective transformation",
        family="H_agent_social_state",
        definition="把他人的视角变换为自己的表征可供使用。",
        computational_role="以他人的位置与知识为原点重算「他看到的/他知道的」。",
        impairment_signature="视角变换失败，默认他人与自己看到相同的世界。",
    ),
)

# ---------------------------------------------------------------------------
# 候补项（FR-1「M1 定稿说明」）：需按五项标准裁定，进则替出。
# ---------------------------------------------------------------------------

CANDIDATES: tuple[Primitive, ...] = (
    Primitive(
        id="context_integration",
        name_zh="语境整合",
        name_en="context integration",
        family="E_uncertainty",
        definition="把上下文信息整合进当前输入的解读。",
        computational_role="以语境为约束，消解输入的多种可能解释。",
        impairment_signature="解读脱离语境，倾向字面化；语用损伤的直接来源。",
        dose_levels=DEFAULT_DOSE_LEVELS,
        status="candidate",
    ),
)

# ---------------------------------------------------------------------------
# 索引与校验
# ---------------------------------------------------------------------------

PRIMITIVE_INDEX: dict[str, Primitive] = {p.id: p for p in PRIMITIVES}

# 默认纳入扰动的 primitive 集合（V2「全量」= 31 个 core）
DEFAULT_ACTIVE: tuple[str, ...] = tuple(p.id for p in PRIMITIVES)

FAMILY_MEMBERS: dict[str, list[str]] = {}
for _p in PRIMITIVES:
    FAMILY_MEMBERS.setdefault(_p.family, []).append(_p.id)


def get(primitive_id: str) -> Primitive:
    """按 id 取 primitive，未知 id 直接报错（避免静默漂移）。"""
    try:
        return PRIMITIVE_INDEX[primitive_id]
    except KeyError:
        known = ", ".join(sorted(PRIMITIVE_INDEX))
        raise KeyError(f"未知 primitive：{primitive_id}（已知：{known}）") from None


def all_ids(include_candidates: bool = False) -> tuple[str, ...]:
    ids = [p.id for p in PRIMITIVES]
    if include_candidates:
        ids += [p.id for p in CANDIDATES]
    return tuple(ids)


def validate_ontology() -> list[str]:
    """本体自检：返回问题清单（空 = 通过）。用于测试与启动守卫。

    **不校验任何「primitive → 测量维度」的对应**：这类预设对应已被明令禁止
    （测量维度只由分析者从「提问 → 作答」的对话里总结）。因此这里只校验定义、
    家族、dose 梯度等本体自身的完整性。
    """
    problems: list[str] = []
    seen: set[str] = set()
    for p in (*PRIMITIVES, *CANDIDATES):
        if p.id in seen:
            problems.append(f"primitive id 重复：{p.id}")
        seen.add(p.id)
        if p.family not in FAMILIES:
            problems.append(f"{p.id} 的 family 未注册：{p.family}")
        for required, label in (
            (p.definition, "definition"),
            (p.computational_role, "computational_role"),
            (p.impairment_signature, "impairment_signature"),
        ):
            if not required or not required.strip():
                problems.append(f"{p.id} 缺少 {label}（FR-1 标准 1）")
        if len(p.dose_levels) < 2:
            problems.append(f"{p.id} 的 dose_levels 少于 2 档（FR-1 标准 4）")
        if any(not (0.0 < d <= 1.0) for d in p.dose_levels):
            problems.append(f"{p.id} 的 dose_levels 必须落在 (0, 1]")
    return problems


def summary_table() -> str:
    """人类可读的家族 × primitive 表（CLI `pacs ontology` 使用）。"""
    lines = ["家族 / primitive 本体（Level 0）", "=" * 62]
    for fam, fam_zh in FAMILIES.items():
        members = FAMILY_MEMBERS.get(fam, [])
        if not members:
            continue
        lines.append(f"\n{fam_zh}（{len(members)}）")
        for pid in members:
            p = PRIMITIVE_INDEX[pid]
            lines.append(f"  - {p.id:<34} {p.name_zh}")
    lines.append(f"\n合计 {len(PRIMITIVES)} 个 core primitive；候补 {len(CANDIDATES)} 个。")
    return "\n".join(lines)
