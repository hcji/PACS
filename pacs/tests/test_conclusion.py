"""探查结论：约 350 字的病例报告式总结 + 结构化表型印象，一路到界面。

结论是用户在界面上要看到的那一段（「是否是孤独症、程度如何、能力方面表现如何」），
因此它必须：由分析者写出、字数有口径、**只依据对话**、并且**不是临床诊断**。
这几件事都是可测的：

1. 提示词里有结论段与 JSON 契约（含字数口径与两条措辞红线）；
2. 解析层归一化：字数、越带提示、诊断式用词告警、程度档位（含中文）；
3. 反馈里带着结论与表型印象；`brain-report` 的读数载荷把它透传给界面；
4. mock 也产出结论，并明写「离线规则式产出，不是 LLM 的判断」；
5. 界面的结论卡片挂在读数卡上（`rd-conclusion`），访谈视图复用同一个渲染函数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pacs
from pacs.agents.dialogue import analyze_dialogue, run_dialogue
from pacs.agents.virtual_agent import VirtualAgent
from pacs.analysis.brain_report import _readings_payload
from pacs.brain.architecture import build_analyst_system_prompt
from pacs.brain.perturb import single
from pacs.llm.parse import parse_analyst_json
from pacs.llm.provider import MockProvider
from pacs.phenotype.conclusion import (
    ASD_LIKENESS_LEVEL_ZH,
    CONCLUSION_CHARS_MAX,
    CONCLUSION_CHARS_MIN,
    CONCLUSION_CHARS_TARGET,
    CONCLUSION_MAX_CHARS,
    diagnosis_wording,
    normalize_asd_likeness,
    normalize_level,
)

_WEB = Path(pacs.__file__).parent / "web" / "static" / "js"


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------


def test_analyst_prompt_requires_a_case_report_style_conclusion():
    text = build_analyst_system_prompt()
    assert "## 探查结论（必须写，病例报告口吻）" in text
    assert "是否表现出孤独症样表型特征" in text
    assert "能力方面的相对表现" in text
    assert f"约 {CONCLUSION_CHARS_TARGET} 字" in text
    assert f"{CONCLUSION_CHARS_MIN}–{CONCLUSION_CHARS_MAX} 字" in text
    # JSON 契约里必须有这两个键
    assert '"conclusion"' in text and '"asd_likeness"' in text
    assert '"level": "<none|mild|moderate|marked>"' in text
    # 措辞红线：可以写表型特征与程度，但不许写成临床诊断
    assert "不是临床诊断" in text
    assert "不得使用「诊断」「确诊」「患病」「患者」" in text


def test_analyst_prompt_still_carries_the_protocol_wording():
    """协议原文逐字保留：改写措辞不能把协议断言也改掉。"""
    text = build_analyst_system_prompt()
    assert "Do not diagnose the agent." in text
    assert "Do not infer the underlying mechanism." in text


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


def test_level_normalization_accepts_ids_english_and_chinese():
    for level in ("none", "mild", "moderate", "marked"):
        assert normalize_level(level) == level
        assert normalize_level(ASD_LIKENESS_LEVEL_ZH[level]) == level
    assert normalize_level("SEVERE") == "marked"
    assert normalize_level("轻度") == "mild"
    assert normalize_level("") == "unknown"
    assert normalize_level("挺明显的吧") == "marked"


def test_asd_likeness_defaults_to_unknown_never_to_absent():
    """缺字段时不许替分析者下「未表现出」的结论。"""
    empty = normalize_asd_likeness(None)
    assert empty == {"present": None, "level": "unknown", "basis": []}
    assert normalize_asd_likeness({"present": None, "level": None})["present"] is None
    yes = normalize_asd_likeness({"present": "是", "level": "中度", "basis": ["a", "b"]})
    assert yes["present"] is True and yes["level"] == "moderate"
    # present=false 而没给程度 → 程度就是「无」（同一件事的另一种写法）
    assert normalize_asd_likeness({"present": False})["level"] == "none"
    # 给了程度就能推出 present
    assert normalize_asd_likeness({"level": "marked"})["present"] is True
    # basis 最多 6 条，空串丢掉
    many = normalize_asd_likeness({"basis": ["", "x"] + [f"b{i}" for i in range(10)]})
    assert len(many["basis"]) == 6


def test_diagnosis_wording_flags_clinical_language_but_allows_disclaimers():
    assert diagnosis_wording("这孩子可以确诊为孤独症谱系障碍") == ["确诊"]
    assert diagnosis_wording("从临床角度看表现明显") == ["临床"]
    # 明确声明「不是诊断」不算违规（那正是我们要的口径）
    assert diagnosis_wording("这是行为学印象，不是诊断") == []
    assert diagnosis_wording("") == []


def test_parse_normalizes_the_conclusion_and_records_its_length():
    payload = json.dumps({
        "dimensions": {"social_cognition": {"score": 70}},
        "conclusion": "观察到的表现" * 40,
        "asd_likeness": {"present": True, "level": "轻度", "basis": ["回应简短"]},
    })
    parsed, error = parse_analyst_json(payload)
    assert error == ""
    assert parsed["conclusion_chars"] == len(parsed["conclusion"]) == 240
    assert parsed["conclusion_target_chars"] == CONCLUSION_CHARS_TARGET
    # 240 字不在 300–400 里 → 记一行说明，但**不截断、不改写**
    assert any("超出" in n for n in parsed["conclusion_notes"])
    assert parsed["asd_likeness"]["level"] == "mild"
    assert parsed["asd_likeness"]["present"] is True


def test_parse_truncates_only_a_runaway_conclusion():
    parsed, _ = parse_analyst_json(json.dumps({"conclusion": "长" * 5000}))
    assert len(parsed["conclusion"]) == CONCLUSION_MAX_CHARS
    assert any("已截断" in n for n in parsed["conclusion_notes"])


def test_parse_flags_diagnosis_wording_in_the_conclusion():
    parsed, _ = parse_analyst_json(json.dumps({"conclusion": "该儿童确诊为孤独症。"}))
    assert any("诊断式用词" in n for n in parsed["conclusion_notes"])


def test_missing_conclusion_is_empty_not_invented():
    parsed, _ = parse_analyst_json(json.dumps({"dimensions": {}}))
    assert parsed["conclusion"] == ""
    assert parsed["conclusion_chars"] == 0
    assert parsed["asd_likeness"]["present"] is None
    # 正则兜底也救不回结论（它是一整段自然语言），但键必须在，界面才知道「没给出」
    fallback, error = parse_analyst_json("dims 是 social_cognition: 0.5 之类")
    assert error
    assert fallback["conclusion"] == "" and fallback["asd_likeness"]["present"] is None


# ---------------------------------------------------------------------------
# 端到端：反馈 → 读数载荷 → 界面
# ---------------------------------------------------------------------------


def _mock_feedback():
    brain = VirtualAgent(MockProvider(), single("inhibition", 0.2), model_id="mock:m", seed=7)
    session = run_dialogue(brain, MockProvider(), n_turns=7, max_turns=7, seed=7)
    return analyze_dialogue(MockProvider(), session)


def test_feedback_carries_the_conclusion_and_the_likeness():
    feedback = _mock_feedback()
    assert feedback["conclusion"]
    assert feedback["conclusion_chars"] == len(feedback["conclusion"])
    assert CONCLUSION_CHARS_MIN <= feedback["conclusion_chars"] <= CONCLUSION_CHARS_MAX
    assert not feedback["conclusion_notes"]
    likeness = feedback["asd_likeness"]
    assert likeness["present"] in (True, False)
    assert likeness["level"] in ASD_LIKENESS_LEVEL_ZH
    assert likeness["basis"]
    # 结论里不许出现诊断式用词（mock 也不许）
    assert diagnosis_wording(feedback["conclusion"]) == []
    # 如实声明这是离线规则式产出
    assert "mock" in feedback["conclusion"]


def test_readings_payload_passes_the_conclusion_to_the_ui():
    feedback = _mock_feedback()
    payload = _readings_payload(feedback)
    for key in ("conclusion", "conclusion_chars", "conclusion_target_chars",
                "conclusion_notes", "asd_likeness", "n_turns_run", "max_turns",
                "question_counts", "under_quota_channels"):
        assert key in payload, key
    assert payload["conclusion"] == feedback["conclusion"]
    assert payload["asd_likeness"]["level"] == feedback["asd_likeness"]["level"]


def test_ui_renders_the_conclusion_card_and_reuses_it_in_the_interview():
    readings_js = (_WEB / "ui" / "readings.js").read_text(encoding="utf-8")
    assert 'id="rd-conclusion"' in readings_js
    assert "export function renderConclusionInto(" in readings_js
    assert "不是临床诊断" in readings_js
    assert "虚拟被试" in readings_js
    phenotype_js = (_WEB / "views" / "phenotype.js").read_text(encoding="utf-8")
    assert "readings.template()" in phenotype_js       # 读数卡（含结论块）挂在 ③ 上
    interview_js = (_WEB / "views" / "interview.js").read_text(encoding="utf-8")
    assert "renderConclusionInto(" in interview_js      # 访谈复用同一个渲染函数
    assert 'id="itv-conclusion"' in interview_js
