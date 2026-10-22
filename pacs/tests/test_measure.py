"""LLM 涌现表型测量测试（表型 = 对话探针的分析者读数，不再来自行为电池）。

守住的几件事：
- 分析者读数被收进 `PhenotypeProfile.dims`，并**逐维区分**（选择性受损必须看得见，
  否则下游的机制异质性/亚型分析没有可用信号）；
- 校验/钳位：越界值、非数值一律不当作读数，缺失维度显式记 `None`；
- `instrument` 与 `confidence` 能随剖面往返存取（历史库兼容）；
- 引擎在 `instrument="dialogue"` 下不产生选择题作答，但每位被试留下独立会话。
"""

from __future__ import annotations

import pytest

from pacs.agents.measure import measure_phenotype, readings_from_feedback
from pacs.agents.virtual_agent import VirtualAgent
from pacs.phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS
from pacs.phenotype import PhenotypeProfile
from pacs.brain import examiner_bank as EB
from pacs.brain.perturb import intact, single
from pacs.engine import run_design
from pacs.llm.provider import MockProvider

ALL_DIMS = tuple(PHENOTYPE_DIMS)      # 表型维度（雷达图的轴）
SCORED = tuple(SCORED_CHANNELS)        # 计分通道（含对照通道）


def _agent(perturbation) -> VirtualAgent:
    return VirtualAgent(MockProvider(), perturbation, model_id="mock:m", seed=42)


def _probes(n: int = 5):
    return EB.dialogue_probes(goals=ALL_DIMS, n=n, seed=42)


# ---------------------------------------------------------------------------
# 读数归一
# ---------------------------------------------------------------------------


def test_readings_are_clamped_and_missing_dims_are_none():
    feedback = {
        "dims": {
            "social_cognition": 1.4,        # 越界 → 钳到 1.0
            "social_communication_pragmatics": -0.3,   # 越界 → 钳到 0.0
            "repetitive_restricted_behavior": "0.5",                   # 非数值 → 视作未给出
        }
    }
    dims = readings_from_feedback(feedback, ALL_DIMS)
    assert dims["social_cognition"] == 1.0
    assert dims["social_communication_pragmatics"] == 0.0
    assert dims["repetitive_restricted_behavior"] is None
    assert dims["social_interaction_motivation"] is None      # 分析者没提过 → 显式记 None
    assert set(dims) == set(ALL_DIMS)


def test_readings_treat_booleans_as_missing():
    """`True` 是 int 的子类，不能被当成读数 1.0。"""
    dims = readings_from_feedback({"dims": {"social_cognition": True}}, ALL_DIMS)
    assert dims["social_cognition"] is None


def test_readings_survive_analyst_failure():
    """分析者输出解析失败时，读数必须全为「未给出」，而不是悄悄变成 0（那会被读成重度受损）。"""
    dims = readings_from_feedback({"dims": {}, "error": "bad json"}, ALL_DIMS)
    assert all(v is None for v in dims.values())


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------


def test_measure_returns_llm_emergent_profile():
    perturbation = single("social_reward_weighting", 0.2, seed=7)
    result = measure_phenotype(
        _agent(perturbation),
        MockProvider(),
        MockProvider(),
        run_id="run_x",
        probes=_probes(),
        n_turns=5,
        seed=42,
    )
    profile = result.profile
    assert profile.instrument == "dialogue"
    assert profile.run_id == "run_x"
    assert profile.theta_id == perturbation.theta_id
    assert set(profile.dims) == set(SCORED)   # 计分通道（含对照通道）
    assert all(v is not None for v in profile.dims.values())
    # 自适应：基础 5 轮走完后按「每通道 ≥5 个问题」补问，直到每个通道都够数。
    # n_items = 实际发起轮数（含补问轮）、n_parsed = 完成轮数 → parse_rate 恒 ≤ 1。
    assert profile.n_items == profile.n_parsed
    assert profile.n_items >= 5
    assert result.session.question_counts
    assert all(v >= 5 for v in result.session.question_counts.values())
    assert profile.age_months == 36          # 年龄是固定常量，不是参数
    assert 0.0 <= profile.confidence <= 1.0
    assert "LLM 涌现读数" in profile.note
    assert not result.problems


def test_emergent_readings_are_dimension_selective():
    """关键性质：扰动一个过程，只有它支撑的维度掉下去。

    如果所有维度一起掉（旧 mock 的行为），机制异质性分析拿不到任何区分维度 ——
    所以这个性质是「表型由 LLM 涌现」这条路能不能用的问题，不是美观问题。
    """
    probes = _probes(5)
    # 自适应补问（不设 max_turns）：每个通道都拿到证据，读数才没有 None 可比较
    intact_dims = measure_phenotype(
        _agent(intact(seed=42)), MockProvider(), MockProvider(),
        probes=probes, n_turns=5, seed=42,
    ).profile.dims
    impaired_dims = measure_phenotype(
        _agent(single("social_reward_weighting", 0.2, seed=7)), MockProvider(), MockProvider(),
        probes=probes, n_turns=5, seed=42,
    ).profile.dims

    dropped = [d for d in ALL_DIMS if impaired_dims[d] < intact_dims[d] - 0.10]
    held = [d for d in ALL_DIMS if impaired_dims[d] >= 0.75]
    assert dropped, f"没有任何维度随扰动下降：{impaired_dims}"
    assert held, f"所有维度一起塌陷，读数没有区分度：{impaired_dims}"


class _FadingExaminer(MockProvider):
    """提问者退化：产出前 `ok_turns` 个可解析问题后，一直返回不可解析输出。"""

    def __init__(self, ok_turns: int = 2) -> None:
        super().__init__()
        self.ok_turns = ok_turns
        self.calls = 0

    def chat(self, messages, **kwargs):  # type: ignore[override]
        if kwargs.get("role") == "examiner":
            self.calls += 1
            if self.calls > self.ok_turns:
                return ""
        return super().chat(messages, **kwargs)


def test_measure_records_short_dialogue_as_a_problem():
    """轮数没跑满就结束 → 记为协议问题（读数覆盖率低），而不是静默当作完整测量。"""
    result = measure_phenotype(
        _agent(intact(seed=1)), _FadingExaminer(ok_turns=2), MockProvider(),
        probes=_probes(5), n_turns=5, max_turns=5, seed=42,
    )
    assert result.session.n_brain_turns == 2
    assert any("只完成了" in p for p in result.problems)
    assert any("没有产出可解析的问题" in p for p in result.problems)


# ---------------------------------------------------------------------------
# 剖面字段往返
# ---------------------------------------------------------------------------


def test_profile_roundtrip_keeps_instrument_and_confidence():
    profile = PhenotypeProfile(
        theta_id="t", model_id="m", run_id="r", theta={"a": 1.0},
        generic_degradation=0.0, dims={"social_cognition": 0.8}, n_items=5, n_parsed=5,
        instrument="dialogue", confidence=0.75,
    )
    restored = PhenotypeProfile.from_dict(profile.to_dict())
    assert restored.instrument == "dialogue"
    assert restored.confidence == 0.75


def test_legacy_profile_payload_defaults_to_dialogue():
    """老库里没有 instrument / confidence 字段的行仍要能读出来。"""
    legacy = {
        "theta_id": "t", "model_id": "m", "run_id": "r", "age_months": 36,
        "theta": {"a": 1.0}, "dims": {"social_cognition": 0.5}, "n_items": 7, "n_parsed": 7,
    }
    restored = PhenotypeProfile.from_dict(legacy)
    assert restored.instrument == "dialogue"
    assert restored.confidence is None


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


def test_run_design_measures_every_subject_by_dialogue():
    """群体运行的测量通道只有一条：每位被试一段对话，不产生逐题作答。"""
    designs = [intact(seed=42), single("salience_assignment", 0.4, seed=3)]
    seen: list[str] = []
    run = run_design(
        MockProvider(), designs, run_id="run_d", model_id="mock:m",
        probes=_probes(3), dialogue_turns=3, seed=42,
        on_dialogue=lambda session, _feedback: seen.append(session.session_id),
    )
    assert run.instrument == "dialogue"
    assert len(run.profiles) == 2
    assert all(p.instrument == "dialogue" for p in run.profiles)
    # 会话 id 必须逐被试唯一，否则批量运行会互相覆盖
    assert len(set(seen)) == len(seen) == 2


def test_run_design_has_no_instrument_switch():
    """测量工具不再可切换：`instrument` 参数随电池一起删除（传了就该报错）。"""
    with pytest.raises(TypeError):
        run_design(MockProvider(), [intact(seed=1)], instrument="battery")


# ---------------------------------------------------------------------------
# 扰动描述的界面语言：中文过程名，不出现英文 primitive id
# ---------------------------------------------------------------------------


def test_perturbation_description_uses_chinese_primitive_names():
    """描述进的是界面与 CLI（还可能是大脑的默认名），因此不能出现英文 id。"""
    from pacs.brain.perturb import (
        compose, generic_degradation_control, intact, single,
    )

    intact_text = intact(seed=1).describe()
    assert intact_text == "完整心智（无扰动）"

    single_text = single("salience_assignment", 0.3, seed=1).describe()
    assert single_text == "[单因素] 显著性分配 0.30"
    assert "salience_assignment" not in single_text

    # 不带设计标签：用于生成大脑默认名称
    bare = single("salience_assignment", 0.3, seed=1).describe(include_design=False)
    assert bare == "显著性分配 0.30"

    # 组合与通用退化对照也要中文
    combined = compose(single("switching_cost", 0.4, seed=2), single("inhibition", 0.2, seed=3))
    assert combined.describe().startswith("[组合] ")
    assert "switching_cost" not in combined.describe() and "inhibition" not in combined.describe()
    assert generic_degradation_control(0.5).describe() == "[通用退化对照] 通用退化 0.50"

    # 原始 id 仍可追溯，但只经由 describe_ids()（界面把它放进 title）
    assert single("salience_assignment", 0.3, seed=1).describe_ids() == "salience_assignment=0.30"


def test_primitive_zh_tolerates_unknown_ids():
    """本体外的 id（候补 primitive）原样返回，不静默变成空字符串。"""
    from pacs.brain.perturb import primitive_zh

    assert primitive_zh("context_integration")          # 候补项仍在本体里有名字
    assert primitive_zh("no_such_primitive") == "no_such_primitive"
