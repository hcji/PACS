"""轮数配额：每通道至少 5 个问题（基础 35 轮 + 自动补问，上限可配）。

计数口径是逐轮按提问者**自报的焦点**记 1 个问题（一条题库条目覆盖多维度也只记 1）——
这条口径既是「问够了没有」的判据，也是补问轮选目标的依据，因此必须可审计：
会话里留着 `question_counts`，界面与 CLI 都把它打出来。

守的几件事：
1. 基础轮数 = 通道数 × 每通道问题数（默认 35）；跑完基础轮后不足 5 的通道会被补问；
2. 补问问的是**缺口最紧的那个通道**（并列按目标顺序），且提示词里点名「本轮必须探查」；
3. 到上限就停（不无限烧额度）；`adapt=False` 或固定清单模式下是定长对话；
4. 逐通道计数与三类通道清单（未探查 / 不足配额）如实记录；
5. 入口校验：`n_turns ≤ max_turns ≤ 硬顶`，字数上限单独校验。
"""

from __future__ import annotations

import json

import pytest

from pacs import QUESTIONS_PER_CHANNEL
from pacs.agents.dialogue import (
    ABSOLUTE_MAX_TURNS,
    DEFAULT_N_TURNS,
    MAX_N_TURNS,
    DialogueSession,
    _quota_deficits,
    run_dialogue,
)
from pacs.agents.measure import measure_phenotype
from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain import examiner_bank as EB
from pacs.brain.architecture import build_examiner_system_prompt, build_examiner_turn_prompt
from pacs.brain.perturb import intact, single
from pacs.llm.provider import MockProvider
from pacs.phenotype import SCORED_CHANNELS

ALL = tuple(SCORED_CHANNELS)


def _brain(perturbation=None):
    return VirtualAgent(MockProvider(), perturbation or intact(), model_id="mock:m", seed=7)


def _probes(n: int = DEFAULT_N_TURNS):
    return EB.dialogue_probes(goals=ALL, n=n, seed=42)


class _RoundRobinExaminer(MockProvider):
    """轮流自报每个通道的提问者：35 轮正好每通道 5 个问题（确定性地填满配额）。"""

    def __init__(self, goals) -> None:
        super().__init__()
        self.goals = list(goals)
        self.calls = 0

    def chat(self, messages, **kwargs):
        if kwargs.get("role") == "examiner":
            target = self.goals[self.calls % len(self.goals)]
            self.calls += 1
            question = f"我们聊聊这件事好吗（第 {self.calls} 次）？"
            return f"{question}\nQUESTION: {question}\nFOCUS: {target}"
        return super().chat(messages, **kwargs)


# ---------------------------------------------------------------------------
# 常量与计数口径
# ---------------------------------------------------------------------------


def test_base_turns_are_channels_times_questions_per_channel():
    assert QUESTIONS_PER_CHANNEL == 5
    assert DEFAULT_N_TURNS == QUESTIONS_PER_CHANNEL * len(SCORED_CHANNELS) == 35
    assert DEFAULT_N_TURNS < MAX_N_TURNS < ABSOLUTE_MAX_TURNS


def test_quota_deficits_prefers_the_thinnest_channel_and_is_deterministic():
    """补问目标：缺口最紧的优先，并列时按目标顺序 —— 否则补问内容会随字典序漂移。"""
    counts = {ch: QUESTIONS_PER_CHANNEL for ch in ALL}      # 先假定都够数
    counts["social_cognition"] = 1
    counts["general_cognition"] = 1
    counts["repetitive_restricted_behavior"] = 4
    deficits = _quota_deficits(counts, ALL)
    # 只有这三个还有缺口；两个「1 个」的并列，按目标顺序（social_cognition 在前）
    assert deficits == [
        "social_cognition", "general_cognition", "repetitive_restricted_behavior",
    ]
    # 全部够数时没有缺口（调用方据此收工）
    assert _quota_deficits({ch: QUESTIONS_PER_CHANNEL for ch in ALL}, ALL) == []


def test_counts_by_self_reported_focus_only():
    """计数按自报焦点：一条条目覆盖多个维度也只记 1，未自报的轮次不计入任何通道。"""
    session = run_dialogue(_brain(), MockProvider(), probe_items=_probes(), n_turns=35)
    examiner_turns = [t for t in session.turns if t["role"] == "examiner"]
    assert sum(session.question_counts.values()) <= len(examiner_turns)
    for turn in examiner_turns:
        if turn["focus"]:
            assert turn["focus"] in session.question_counts
    assert set(session.question_counts) == set(ALL)
    assert session.n_turns_run == len(examiner_turns)


# ---------------------------------------------------------------------------
# 补问循环
# ---------------------------------------------------------------------------


def test_base_phase_is_never_cut_short():
    """基础 35 轮必须走完：它是「每通道 ≥5 个问题」的下限，不能提前收工。"""
    session = run_dialogue(_brain(), MockProvider(), probe_items=_probes(), n_turns=35)
    assert session.n_turns_run >= DEFAULT_N_TURNS
    assert all(v >= QUESTIONS_PER_CHANNEL for v in session.question_counts.values())
    assert session.under_quota_channels == [] and session.uncovered_channels == []
    assert not any("不足" in p for p in session.problems)


def test_no_top_up_when_the_base_phase_already_fills_every_quota():
    """基础轮正好够数时**一轮都不补** —— 多问一轮就是多两次模型调用。"""
    session = run_dialogue(_brain(), _RoundRobinExaminer(ALL), n_turns=DEFAULT_N_TURNS)
    assert session.n_turns_run == DEFAULT_N_TURNS
    assert session.question_counts == {ch: QUESTIONS_PER_CHANNEL for ch in ALL}


def test_short_base_phase_is_topped_up_to_the_quota():
    """基础轮不够（这里只给 7 轮）→ 继续补问到每通道 5 个问题。"""
    session = run_dialogue(_brain(), MockProvider(), probe_items=_probes(7), n_turns=7)
    assert session.n_turns_run > 7
    assert all(v >= QUESTIONS_PER_CHANNEL for v in session.question_counts.values())
    assert session.under_quota_channels == []


def test_top_up_stops_at_max_turns_and_says_so():
    """到上限就停，并把「哪几个方面还没问够」如实写进 problems。"""
    session = run_dialogue(
        _brain(), MockProvider(), probe_items=_probes(4), n_turns=4, max_turns=4,
    )
    assert session.n_turns_run == 4
    assert session.under_quota_channels
    note = next(p for p in session.problems if "不足" in p)
    for ch in session.under_quota_channels:
        assert SCORED_CHANNELS[ch] in note
    assert "上限 4" in note


def test_two_turn_run_leaves_channels_never_probed_at_all():
    """一个问题都没问到的通道单独记：它与「问到了但不够」是两件事。"""
    session = run_dialogue(_brain(), MockProvider(), n_turns=2, max_turns=2)
    assert session.uncovered_channels
    assert set(session.uncovered_channels) | set(session.under_quota_channels) <= set(ALL)
    assert any("没有证据支撑" in p for p in session.problems)


def test_fixed_question_list_is_always_fixed_length():
    """固定清单没有「再生成一个问题」的来源 → 不做补问。"""
    session = run_dialogue(_brain(), None, questions=["Q1?", "Q2?"], n_turns=9)
    assert session.n_turns_run == 2
    assert session.question_counts == {ch: 0 for ch in ALL}


def test_adapt_false_keeps_the_fixed_behaviour():
    session = run_dialogue(_brain(), MockProvider(), n_turns=7, max_turns=35, adapt=False)
    assert session.n_turns_run == 7


def test_top_up_targets_only_the_thinnest_channel():
    """补问轮把缺口写进提问者的当轮指令（点名「本轮必须探查」），而不是让它自己猜。"""
    text = build_examiner_turn_prompt(
        turn_index=36, turn_budget=35, transcript="提问者：你会怎么做？\n孩子：我不知道。",
        must_focus="social_cognition",
    )
    assert "本轮必须探查" in text
    assert f"{SCORED_CHANNELS['social_cognition']}（social_cognition）" in text
    assert "至少 35 轮" in text          # 基础轮数是下限，不是总数
    # 基础轮内不点名（前几轮按缺口指派会让开场反复问同一件事）
    assert "本轮必须探查" not in build_examiner_turn_prompt(turn_index=2, turn_budget=35)


def test_examiner_system_prompt_states_the_quota_and_the_ceiling():
    text = build_examiner_system_prompt(turn_budget=DEFAULT_N_TURNS)
    assert f"每个方面至少要有 {QUESTIONS_PER_CHANNEL} 个问题" in text
    assert f"至少 {DEFAULT_N_TURNS} 轮" in text


# ---------------------------------------------------------------------------
# 与测量/落库的一致性
# ---------------------------------------------------------------------------


def test_measure_reports_actual_turns_and_keeps_parse_rate_bounded():
    result = measure_phenotype(
        _brain(single("inhibition", 0.2)), MockProvider(), MockProvider(),
        probes=_probes(7), n_turns=7, seed=7,
    )
    assert result.session.n_turns_run > 7
    assert result.profile.n_items == result.session.n_turns_run
    assert 0.0 < result.profile.parse_rate <= 1.0
    assert "含补问" in result.profile.note
    assert not result.profile.uncovered_channels


def test_session_round_trip_keeps_the_quota_records():
    session = run_dialogue(_brain(), MockProvider(), probe_items=_probes(7), n_turns=7)
    restored = DialogueSession.from_dict(json.loads(json.dumps(session.to_dict())))
    assert restored.question_counts == session.question_counts
    assert restored.n_turns_run == session.n_turns_run
    assert restored.max_turns == session.max_turns
    assert restored.under_quota_channels == session.under_quota_channels


# ---------------------------------------------------------------------------
# 入口校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"n_turns": 0},
    {"n_turns": ABSOLUTE_MAX_TURNS + 1},
    {"n_turns": 40, "max_turns": 20},            # 上限低于基础轮数
    {"n_turns": 5, "max_turns": ABSOLUTE_MAX_TURNS + 1},
])
def test_run_dialogue_rejects_impossible_turn_settings(kwargs):
    with pytest.raises(ValueError):
        run_dialogue(_brain(), MockProvider(), **kwargs)


def test_mock_examiner_never_repeats_its_own_improvised_question():
    """补问同一通道时要换例子：重复的场景对儿童式探查没有新信息。

    只查**自造题**（`probe_key is None`）：题库里本来就有多条条目的探查语句逐字相同，
    那是题库的性质，不是提问者重复提问。
    """
    session = run_dialogue(_brain(), MockProvider(), probe_items=_probes(3), n_turns=3)
    improvised = [t["question"] for t in session.turns
                  if t["role"] == "examiner" and not t["probe_key"]]
    assert len(improvised) > 5
    assert len(improvised) == len(set(improvised))
