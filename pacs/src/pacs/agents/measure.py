"""LLM 涌现表型测量：表型向量 `B` 直接取自对话探针的分析者读数。

为什么不再用固定行为电池：强制选择题（forced-choice）对一个人工心智是不自然的任务，
它的正确率同时受「认知完整度」和「是否愿意/能够按格式作答」影响，两件事在 LLM 上分不开。
本项目的论点是**表型由扰动自然涌现**，那么测量也应该走同一条通道：让提问者 LLM 与
被扰动的心智自由对话，再由分析者 LLM 按维度给出典型性读数 —— 这个读数就是 `B`。

方向约定与 θ 一致：**1.0 = 与同龄典型发展无异，0.0 = 严重受损**。这正好落在下游分析
（特异性 / 剂量-反应 / 景观 / 分型）使用的 0..1 区间里，因此那些分析不必区分来源。

盲法仍然成立：分析者只读对话记录，提示词里没有 θ；本模块也**不把 θ 回灌**给提问者或
分析者 —— 它只把分析者的输出搬进 `PhenotypeProfile.dims`。
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from .. import DEFAULT_AGE_MONTHS, DEFAULT_TEXT_LIMIT
from ..phenotype import PhenotypeProfile, SCORED_CHANNELS
from .dialogue import (
    DEFAULT_N_TURNS,
    MAX_N_TURNS,
    DialogueSession,
    analyze_dialogue,
    run_dialogue,
)


def _clamp01(value) -> Optional[float]:
    """把分析者的读数收敛到 [0, 1]；非数值一律视为「未给出」。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def provider_kind_of(*providers) -> str:
    """这批 provider 是不是离线规则式替身。

    只要参与测量的一环是 `mock`（提问者/分析者/大脑任一），读数就**不是** LLM 涌现的，
    剖面必须标成合成 —— 否则规则构造出来的表型会被当成真实测量结果往下游走。
    """
    for provider in providers:
        name = str(getattr(provider, "name", "") or "")
        if name == "mock":
            return "mock"
    return "llm"


def readings_from_feedback(
    feedback: dict, goals: Sequence[str] = ()
) -> dict[str, Optional[float]]:
    """从分析者反馈里取出各维度读数，缺失的维度填 `None`。

    为什么缺失要显式填 `None` 而不是干脆不出现：分析者偶尔只覆盖部分维度，
    下游按维度取均值时必须能区分「这一维没有读数」与「这一维得分 0」。
    """
    raw = feedback.get("dims") or {}
    dims = tuple(goals) if goals else tuple(SCORED_CHANNELS)
    return {dim: _clamp01(raw.get(dim)) for dim in dims}


@dataclass
class MeasureResult:
    """一个被试的 LLM 涌现测量结果。"""

    profile: PhenotypeProfile
    session: DialogueSession
    feedback: dict
    problems: list[str] = field(default_factory=list)


def measure_phenotype(
    agent,
    examiner,
    analyst,
    *,
    run_id: str = "",
    probes: Sequence[dict] = (),
    probe_mode: str = "auto",
    n_turns: int | None = None,
    max_turns: int | None = None,
    goals: Sequence[str] = (),
    seed: int = 42,
    rubric_items: Sequence[dict] = (),
    brain_id: str = "",
    session_id: str | None = None,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    on_turn=None,
) -> MeasureResult:
    """对一个被试跑一次「对话探针 → 分析者读数」，并把它作为表型向量。

    `agent` 需满足 `measure` 所用的鸭子类型接口：`.theta_id`、`.model_id`、
    `.perturbation`、`.reply(history, turn=)`。提问者与分析者都不接触 θ。
    `n_turns=None` 时取基础轮数（每个计分通道各 5 个问题的下限）；`max_turns` 是补问上限。
    **年龄不是参数**：`PhenotypeProfile.age_months` 一律写 `DEFAULT_AGE_MONTHS`（固定口径，
    见该常量）—— 参照线随参数漂移，读数就不可比。
    """
    goals = tuple(goals) if goals else tuple(SCORED_CHANNELS)
    n_turns = n_turns or max(DEFAULT_N_TURNS, len(goals))
    max_turns = max_turns or max(MAX_N_TURNS, n_turns)
    session = run_dialogue(
        agent,
        examiner,
        probe_items=list(probes),
        probe_mode=probe_mode,
        n_turns=n_turns,
        max_turns=max_turns,
        goals=goals,
        seed=seed,
        session_id=session_id,
        brain_id=brain_id,
        text_limit=text_limit,
        on_turn=on_turn,
    )
    perturbation = agent.perturbation
    feedback = analyze_dialogue(
        analyst,
        session,
        goals=tuple((dim, SCORED_CHANNELS.get(dim, dim)) for dim in goals),
        rubric_items=list(rubric_items),
    )
    dims = readings_from_feedback(feedback, goals)
    answered = session.n_brain_turns
    problems: list[str] = list(session.problems)
    if feedback.get("error"):
        problems.append(f"分析者输出解析失败：{feedback['error']}")
    if not any(v is not None for v in dims.values()) and not feedback.get("error"):
        problems.append("分析者没有给出任何可用的维度读数")
    # 两件事分开算，别混成一个数：
    # - `planned` 是**实际发起**的轮数（含补问轮），用作 `parse_rate` 的分母 ——
    #   拿基础轮数当分母，一旦补问这个比值就会大于 1，它的含义（回合完成率）也就废了；
    # - 「跑没跑满」要跟**协议承诺的基础轮数**比：基础 35 轮是每通道 ≥5 个问题的下限，
    #   提前停（提问者产不出问题等）就是协议问题，必须在 problems 里说出来。
    planned = max(1, session.n_turns_run or n_turns)
    if answered < max(1, n_turns):
        problems.append(
            f"对话只完成了 {answered}/{n_turns} 个基础轮，读数覆盖率偏低"
        )

    # 没有证据支撑的通道：整段对话没探查到的（`run_dialogue` 结构化记下）加上分析者
    # 没有给出读数的 —— 显式记下来，使下游不会把「没问过」当成「表现典型」。
    uncovered = tuple(
        ch for ch in goals
        if ch in session.uncovered_channels or dims.get(ch) is None
    )

    profile = PhenotypeProfile(
        theta_id=agent.theta_id,
        model_id=agent.model_id,
        run_id=run_id,
        age_months=DEFAULT_AGE_MONTHS,
        theta=dict(perturbation.values),
        generic_degradation=perturbation.generic_degradation,
        dims=dims,
        # n_items 记**实际发起**的轮数（含补问轮）、n_parsed 记**完成**轮数 →
        # parse_rate 即对话回合完成率，且恒 ≤ 1
        n_items=planned,
        n_parsed=answered,
        design=perturbation.design,
        note=(
            f"LLM 涌现读数：{answered} 轮对话"
            + (
                f"（含补问 {answered - n_turns} 轮）"
                if answered > n_turns else ""
            )
            + f" · 分析者 {feedback.get('model_id') or '—'}"
            + (f" · 探针 {len(session.probe_keys)} 条" if session.probe_keys else "")
        ),
        instrument="dialogue",
        confidence=_clamp01(feedback.get("confidence")),
        provider_kind=provider_kind_of(agent, analyst),
        uncovered_channels=uncovered,
    )
    return MeasureResult(profile=profile, session=session, feedback=feedback, problems=problems)


def with_replicate(profile: PhenotypeProfile, replicate: int) -> PhenotypeProfile:
    """标出同一 θ 的第几次重复测量（参照条件重复取平均时要区分）。"""
    return replace(profile, replicate=replicate)


def stamp_run(result: MeasureResult, run_id: str) -> MeasureResult:
    """把运行号写进剖面 —— 服务端在落盘前调用。"""
    return MeasureResult(
        profile=replace(result.profile, run_id=run_id),
        session=result.session,
        feedback=result.feedback,
        problems=result.problems,
    )


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
