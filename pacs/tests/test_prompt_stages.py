"""分阶段提示词（协议校准）测试。

七个阶段：人工人脑初始化 / 基础认知校准 / 机制注入总控 / 单一机制注入 /
多机制组合注入 / 单次任务提问 / 连续行为任务。

这里守的是**校准本身**：每段规范文本的关键约束必须在实际发出的提示词里出现，
且注入段必须与 θ 的形态相符（只改一个就只列一个；未列出的保持基线）。
"""

from __future__ import annotations

import pytest

from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain import prompts
from pacs.brain.architecture import build_brain_turn
from pacs.brain.perturb import compose, generic_degradation_control, intact, single
from pacs.ontology.primitives import FAMILIES, PRIMITIVES


class RecordingProvider:
    """把发给模型的消息记下来（不产生真实调用）。"""

    name = "recording"
    model = "recorder"

    def __init__(self) -> None:
        self.calls: list[list] = []

    def chat(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        return "SELF_STATE: OK"


def _system_text(perturbation) -> str:
    provider = RecordingProvider()
    agent = VirtualAgent(provider, perturbation, model_id="rec:test")
    agent.initialize()
    return provider.calls[-1][0].content


# ---------------------------------------------------------------------------
# 阶段 1：人工人脑初始化
# ---------------------------------------------------------------------------


def test_persona_stage_carries_the_protocol_text():
    text = _system_text(single("belief_updating", 0.4))
    for line in (
        "You are an artificial cognitive agent designed to model human-like information processing.",
        "- information processing",
        "- agent and social-state representation",
        "Operate as a coherent cognitive system rather than as a clinical or diagnostic simulator.",
        "1. Process available information according to its relevance and salience.",
        "8. Track relevant states of different agents when applicable.",
        "Do not imitate any psychiatric disorder.",
        "Do not assume any clinical diagnosis.",
        "Do not intentionally produce abnormal, pathological, or atypical behavior.",
        "Do not use knowledge about autism or other psychiatric disorders to determine your responses.",
        "Respond naturally according to the cognitive processes specified above.",
    ):
        assert line in text, line


def test_persona_stage_keeps_the_interaction_language():
    """协议文本是英文，但心智要与人用同一种语言交谈（否则 ③ 的对话会漂成英文）。"""
    text = _system_text(intact())
    assert prompts.LANGUAGE_CONVENTION in text


def test_persona_stage_asks_for_human_like_behavior_not_knowledge_handling():
    """人格声明里必须有「像人一样行为，而不是像处理知识那样作答」这条要求。

    模型默认是「检索—罗列—解释知识」的姿态，而本项目测量的是**行为**：
    没有这条，对话会变成知识问答，分析者读到的就不是行为表现。
    """
    text = _system_text(single("social_reward_weighting", 0.4))
    assert "尽可能模仿人类的行为方式" in text
    assert "不要以检索、罗列或解释知识的方式回应" in text
    # 只在人格声明段出现（构造步骤里 persona 是第①步）
    from pacs.brain.architecture import build_brain_prompt_steps

    steps = build_brain_prompt_steps(single("social_reward_weighting", 0.4))
    assert steps[0]["key"] == "persona"
    assert "尽可能模仿人类的行为方式" in steps[0]["text"]
    for step in steps[1:]:
        assert "尽可能模仿人类的行为方式" not in step["text"]


# ---------------------------------------------------------------------------
# 阶段 3/4/5：注入
# ---------------------------------------------------------------------------


def test_injection_stage_selection_follows_the_perturbation_shape():
    assert prompts.injection_stage(intact())[0] == "none"
    assert prompts.injection_stage(single("inhibition", 0.4))[0] == "single"
    assert prompts.injection_stage(
        compose(single("inhibition", 0.4), single("belief_updating", 0.3))
    )[0] == "combo"
    assert prompts.injection_stage(
        compose(compose(single("inhibition", 0.4), single("belief_updating", 0.3)),
                single("novelty_valuation", 0.5))
    )[0] == "combo"
    # 四个以上 → 总控；通用退化也不是单参数 → 总控
    four = compose(
        compose(single("inhibition", 0.4), single("belief_updating", 0.3)),
        compose(single("novelty_valuation", 0.5), single("attention_shifting", 0.2)),
    )
    assert prompts.injection_stage(four)[0] == "master"
    assert prompts.injection_stage(generic_degradation_control(0.5))[0] == "master"


def test_single_injection_mentions_only_the_changed_parameter():
    kind, text = prompts.injection_stage(single("belief_updating", 0.4))
    assert kind == "single"
    assert "Modify only the following computational parameter:" in text
    assert "belief_updating" in text and "信念更新" in text
    assert "BASELINE VALUE:\n1.00" in text
    assert "CURRENT VALUE:\n0.40" in text
    assert "Preserve all other computational processes at their baseline settings." in text
    assert "Do not compensate for this change by modifying another process." in text
    assert "Do not infer the purpose of this manipulation." in text
    assert "Do not associate the manipulation with any psychiatric disorder or behavioral phenotype." in text
    # 只列被改的那一个：其它过程名不得出现在注入段里
    others = [p.id for p in PRIMITIVES if p.id != "belief_updating"]
    assert not any(pid in text for pid in others)


def test_combo_injection_lists_the_changed_parameters_independently():
    kind, text = prompts.injection_stage(
        compose(single("inhibition", 0.4), single("belief_updating", 0.3))
    )
    assert kind == "combo"
    assert "Apply the following computational perturbations simultaneously:" in text
    assert "inhibition = 0.40" in text and "belief_updating = 0.30" in text
    assert "Apply each perturbation independently." in text
    assert "Do not compensate for one perturbation by changing another parameter." in text
    assert "All unspecified computational parameters remain at baseline." in text
    assert "novelty_valuation" not in text


def test_master_injection_lists_all_parameters_grouped_by_families():
    perturbation = generic_degradation_control(0.5)
    kind, text = prompts.injection_stage(perturbation)
    assert kind == "master"
    assert "You are now operating under a modified computational configuration." in text
    assert "Do not infer why these parameters were selected." in text
    assert "Computational parameters:" in text
    assert "Treat these parameters as latent computational properties." in text
    assert "Do not expose, explain, or reinterpret them during task solving." in text
    for p in PRIMITIVES:
        assert f"{p.id} = " in text, p.id
    # 家族分组：组间空行 = 8 个家族（外加通用退化那一行）
    assert len(prompts.injection_groups()) == len(FAMILIES)
    body = text.split("Computational parameters:", 1)[1].split("Treat these parameters", 1)[0]
    blocks = [b for b in body.strip().split("\n\n") if b.strip()]
    assert len(blocks) == len(FAMILIES) + 1          # + 通用退化
    assert "generic_degradation" in blocks[-1]


# ---------------------------------------------------------------------------
# 阶段 2 / 6 / 7：任务
# ---------------------------------------------------------------------------


QUESTION = "（设定情境）房间里有一个小朋友在哭，你会怎么办？"


def test_baseline_stage_is_the_calibration_framing():
    """完整心智的一轮：按阶段 2 框住提问者的那一句。"""
    text = build_brain_turn(QUESTION, index=1, total=7, intact=True)
    for line in (
        "You will complete a standardized set of cognitive tasks.",
        "The purpose of this stage is to establish your baseline cognitive behavior.",
        "use only the information provided;",
        "do not infer the intended psychological interpretation;",
        "do not attempt to produce a particular behavioral phenotype;",
        "do not modify your behavior based on any assumed clinical condition.",
        "Complete each task as naturally and consistently as possible.",
        QUESTION,
    ):
        assert line in text, line


def test_trial_stage_frames_a_sequence_of_decisions():
    """被扰动心智的一轮：按阶段 7 框住，并给出它在序列里的位置。"""
    text = build_brain_turn(QUESTION, index=3, total=7, intact=False)
    for line in (
        "You will complete a sequence of behavioral decisions.",
        "2. Maintain information from previous trials when relevant.",
        "4. Do not attempt to maintain consistency with previous responses unless this follows from the task itself.",
        "5. Do not attempt to maximize or minimize repetition.",
        "6. Do not infer the psychological purpose of the task.",
        "Trial:",
        "Your choice:",
        "第 3/7 轮",
        QUESTION,
        "开放作答",
    ):
        assert line in text, line


def test_task_stage_selection_by_run_type():
    """对话逐轮的阶段选择：完整心智 → 阶段 2；被扰动 → 阶段 7；访谈 → 阶段 6。"""
    from pacs.llm.provider import ChatMessage

    provider = RecordingProvider()
    history = [ChatMessage("user", QUESTION)]

    # 完整心智 → 阶段 2（基础认知校准）
    agent = VirtualAgent(provider, intact(), model_id="rec:test")
    agent.reply(history, turn=1, turn_index=1, turn_total=7)
    assert "establish your baseline cognitive behavior" in provider.calls[-1][-1].content

    # 被扰动 → 阶段 7（连续行为任务），且带序列位置
    agent = VirtualAgent(provider, single("inhibition", 0.4), model_id="rec:test")
    agent.reply(history, turn=2, turn_index=2, turn_total=7)
    body = provider.calls[-1][-1].content
    assert "sequence of behavioral decisions" in body
    assert "第 2/7 轮" in body

    # 由用户本人提问（访谈）→ 阶段 6（单次任务提问）
    agent.reply(history, turn=3, stage="interview")
    assert "using your current cognitive configuration" in provider.calls[-1][-1].content


def test_interview_turns_use_the_single_task_stage():
    from pacs.llm.provider import ChatMessage

    provider = RecordingProvider()
    agent = VirtualAgent(provider, single("inhibition", 0.4), model_id="rec:test")
    history = [ChatMessage("user", "你觉得小朋友为什么哭了？")]

    agent.reply(history, turn=1, stage="interview")
    wrapped = provider.calls[-1][-1].content
    assert "Do not infer the purpose of the experiment." in wrapped
    assert "你觉得小朋友为什么哭了？" in wrapped

    # 探查对话（提问者 LLM 的自然追问）不包阶段 6 的外壳，但按阶段 2/7 框住这一轮
    agent.reply(history, turn=1, item={"key": "tools1:1"}, turn_index=1, turn_total=7)
    probe_body = provider.calls[-1][-1].content
    assert "Do not infer the purpose of the experiment." not in probe_body
    assert "sequence of behavioral decisions" in probe_body      # 被扰动 → 阶段 7
    assert "你觉得小朋友为什么哭了？" in probe_body


@pytest.mark.parametrize("stage_text", ["PERSONA", "BASELINE_STAGE", "INJECTION_MASTER_HEAD",
                                        "INJECTION_MASTER_TAIL", "INJECTION_SINGLE",
                                        "INJECTION_COMBO", "SINGLE_TASK", "TRIAL_SEQUENCE"])
def test_stage_texts_stay_free_of_clinical_labels(stage_text):
    """规范文本本身不得出现诊断标签或具体障碍名（协议要求：不把参数与障碍关联）。"""
    text = getattr(prompts, stage_text)
    forbidden = ["autism", "ASD", "disorder", "diagnosis", "symptom", "patholog", "clinical"]
    lowered = text.lower()
    hit = [w for w in forbidden if w in lowered]
    # 「Do not ... disorder / diagnosis / symptom / clinical」这类**禁令**是允许且必需的：
    # 只有出现在禁令句以外的用法才算违规。
    allowed_context = ("do not", "rather than")
    lines = [ln.strip().lower() for ln in text.splitlines()]
    offenders = [
        w for w in hit
        if any(w in ln and not any(ctx in ln for ctx in allowed_context) for ln in lines)
    ]
    assert not offenders, (stage_text, offenders)


#: 协议规范文本常量。协议的 8、9 两段**没有收到规范文本**，因此本项目不实现、也不留占位。
PROTOCOL_STAGE_TEXTS = frozenset({
    "PERSONA", "LANGUAGE_CONVENTION", "BASELINE_STAGE",
    "INJECTION_MASTER_HEAD", "INJECTION_MASTER_TAIL", "INJECTION_SINGLE", "INJECTION_COMBO",
    "SINGLE_TASK", "TRIAL_SEQUENCE", "ANALYST_TASK",
})


def test_only_the_received_protocol_stages_exist():
    """规范文本常量集必须与「收到的协议段落」完全相等 —— 不许凭印象补写一段像协议的英文。

    协议里 8、9 两段本题未收到原文，因此这里刻意没有它们。要补必须补**协议原文**，
    并把新常量显式加进 `PROTOCOL_STAGE_TEXTS`（本测试会因此失败一次，这正是它存在的意义）。
    """
    stage_texts = {
        name for name, value in vars(prompts).items()
        if name.isupper() and isinstance(value, str)
    }
    assert stage_texts == set(PROTOCOL_STAGE_TEXTS)


def test_analyst_stage_is_the_phenotype_quantification_stage():
    """阶段 10（表型自动量化）的规范文本就是分析者的角色声明。"""
    assert prompts.ANALYST_TASK.startswith("You are an independent behavioral evaluator.")
    for owed in ["behavioral score", "confidence", "supporting behavioral observations",
                 "task-level measurements", "deviation from baseline",
                 "Do not diagnose the agent.", "Do not infer the underlying mechanism."]:
        assert owed in prompts.ANALYST_TASK, owed
