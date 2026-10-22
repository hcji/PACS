"""问题与回答的字数上限：提示词约束 + 超长告警，**不截断、不重试**。

为什么是「约束 + 告警」而不是「截断」：这段对话是**证据**。截断会让分析者读到一个
不存在的回答，读数就不可信了；重试则会悄悄改变被试的表现（同一轮问两次，
模型第二次的回应不是第一轮的延续）。所以超长如实记录：轮次上标 `over_limit`，
会话里记一条汇总，界面与 CLI 都显示出来，让「证据比预期长」这件事可见。

守的几件事：
1. 两处提示词（提问者的问题、心智的回答）用**同一个数**渲染；
2. 默认 100，可配（`text_limit`），越界值拒绝；
3. 超长只告警：内容原样保留，轮数不重跑；
4. 字数上限也记进会话（落库后可回看当次的口径）。
"""

from __future__ import annotations

import pytest

from pacs import DEFAULT_TEXT_LIMIT, MAX_TEXT_LIMIT
from pacs.agents.dialogue import DialogueSession, run_dialogue
from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain.architecture import (
    build_brain_prompt,
    build_brain_prompt_steps,
    build_examiner_system_prompt,
)
from pacs.brain.perturb import intact
from pacs.llm.provider import MockProvider


def _brain(text_limit: int = DEFAULT_TEXT_LIMIT):
    return VirtualAgent(
        MockProvider(), intact(), model_id="mock:m", seed=7, text_limit=text_limit,
    )


class _LongWindedExaminer(MockProvider):
    """每轮都问一段超长的话（模拟真实模型不守字数上限）。"""

    def chat(self, messages, **kwargs):
        if kwargs.get("role") == "examiner":
            long_question = "今天在幼儿园里，" + "然后呢" * 60 + "你会怎么做？"
            return f"{long_question}\nQUESTION: {long_question}\nFOCUS: social_cognition"
        return super().chat(messages, **kwargs)


# ---------------------------------------------------------------------------
# 提示词：同一个数渲染两处
# ---------------------------------------------------------------------------


def test_both_prompts_render_the_same_text_limit():
    examiner = build_examiner_system_prompt(turn_budget=35, text_limit=123)
    assert "不超过 123 个字" in examiner
    brain = build_brain_prompt(intact(), text_limit=123)
    assert "不超过 123 个字" in brain
    # 界面上看到的构造过程与实际发送的提示词同源
    steps = build_brain_prompt_steps(intact(), text_limit=123)
    assert "\n\n".join(s["text"] for s in steps) == brain


def test_default_limit_is_100_and_stated_in_the_brain_prompt():
    assert DEFAULT_TEXT_LIMIT == 100
    assert "不超过 100 个字" in build_brain_prompt(intact())
    assert "不超过 100 个字" in build_examiner_system_prompt(turn_budget=35)


def test_brain_prompt_steps_stay_in_sync_when_the_limit_changes():
    for limit in (40, 100, 300):
        assert build_brain_prompt(intact(), text_limit=limit) \
            == "\n\n".join(s["text"] for s in build_brain_prompt_steps(intact(), text_limit=limit))


# ---------------------------------------------------------------------------
# 运行时：告警而不截断
# ---------------------------------------------------------------------------


def test_over_limit_turns_are_flagged_but_kept_verbatim():
    session = run_dialogue(
        _brain(), _LongWindedExaminer(), n_turns=6, max_turns=6, text_limit=100,
    )
    over = [t for t in session.turns if t["role"] == "examiner" and t["over_limit"]]
    assert over, "超长的问题必须被标出来"
    # 内容原样保留（不截断）：字数就是原文的字数
    assert all(t["chars"] == len(t["content"]) > 100 for t in over)
    assert session.over_limit_turns == len({t["turn"] for t in over})
    note = next(p for p in session.problems if "超过" in p)
    assert "100 字上限" in note and "未截断" in note


def test_within_limit_turns_are_not_flagged():
    session = run_dialogue(_brain(), MockProvider(), n_turns=6, max_turns=6)
    assert not any(t.get("over_limit") for t in session.turns)
    assert session.over_limit_turns == 0
    assert not any("超过" in p for p in session.problems)


def test_limit_is_recorded_in_the_session_and_round_trips():
    session = run_dialogue(
        _brain(80), MockProvider(), n_turns=6, max_turns=6, text_limit=80,
    )
    assert session.text_limit == 80
    restored = DialogueSession.from_dict(session.to_dict())
    assert restored.text_limit == 80
    # 旧记录没有这个键 → 回到默认口径（不沿用别人的设置）
    legacy = session.to_dict()
    legacy.pop("text_limit")
    assert DialogueSession.from_dict(legacy).text_limit == DEFAULT_TEXT_LIMIT


@pytest.mark.parametrize("limit", [0, -1, MAX_TEXT_LIMIT + 1])
def test_out_of_range_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        run_dialogue(_brain(), MockProvider(), n_turns=2, max_turns=2, text_limit=limit)


def test_tight_limit_flags_more_turns_than_a_loose_one():
    """口径真的在起作用：同一个提问者，上限越紧告警越多。"""
    tight = run_dialogue(
        _brain(20), _LongWindedExaminer(), n_turns=4, max_turns=4, text_limit=20,
    )
    loose = run_dialogue(
        _brain(400), _LongWindedExaminer(), n_turns=4, max_turns=4, text_limit=400,
    )
    assert tight.over_limit_turns > 0
    assert loose.over_limit_turns == 0
