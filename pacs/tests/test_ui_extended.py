"""四界面改造的核心不变量测试：提示词构造过程、大脑库、量表分流、对话探针、盲法。

这些测试盯住的都是**容易悄悄坏掉**的地方：
- 界面上看到的「提示词构造过程」必须与实际发送的提示词一致（同源，不许漂移）；
- 提问者 / 分析者必须对 θ 盲法（否则对话探针退化成对已知答案的复述）；
- 保存量表的模式分流必须显式，不能静默把无客观指标的题塞进定量管线；
- 同一个 θ 不应在大脑库里存成两条。
"""

from __future__ import annotations

import json

import pytest

from pacs.agents.dialogue import (
    ABSOLUTE_MAX_TURNS,
    MAX_N_TURNS,
    DialogueSession,
    analyze_dialogue,
    run_dialogue,
)
from pacs.agents.virtual_agent import VirtualAgent
from pacs.phenotype import SCORED_CHANNELS
from pacs.scales import (
    SCORING_LLM,
    parse_item_set,
)
from pacs.brain.architecture import (
    build_analyst_prompt,
    build_analyst_system_prompt,
    build_brain_prompt,
    build_brain_prompt_steps,
    build_examiner_system_prompt,
    build_examiner_turn_prompt,
)
from pacs.brain.perturb import compose, generic_degradation_control, intact, single
from pacs.llm.parse import parse_analyst_json, parse_examiner_question
from pacs.llm.provider import MockProvider
from pacs.ontology.primitives import PRIMITIVES
from pacs.store.db import Store

PERTURBATION_CASES = [
    ("intact", lambda: intact()),
    ("single_low", lambda: single("attention_shifting", 0.2)),
    ("single_mid", lambda: single("belief_updating", 0.6)),
    ("generic", lambda: generic_degradation_control(0.5)),
    ("composed", lambda: compose(single("inhibition", 0.2), single("switching_cost", 0.4))),
    ("multi", lambda: __import__("pacs.brain.perturb", fromlist=["from_values"]).from_values(
        {"salience_assignment": 0.3, "novelty_valuation": 0.7}
    )),
]


# ---------------------------------------------------------------------------
# 提示词构造过程（规则 4：过程可视化不得与实际发送内容漂移）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,build", PERTURBATION_CASES, ids=[c[0] for c in PERTURBATION_CASES])
def test_prompt_steps_render_to_exact_prompt(label, build):
    """各步原文拼接必须**逐字节**等于最终提示词（界面所见 = 实际发送）。"""
    perturbation = build()
    steps = build_brain_prompt_steps(perturbation)
    assert "\n\n".join(step["text"] for step in steps) == build_brain_prompt(perturbation)


def test_prompt_steps_are_structured_for_visualization():
    steps = build_brain_prompt_steps(single("attention_shifting", 0.2))
    keys = [s["key"] for s in steps]
    # 被扰动的 θ：人格声明 → 过程清单 → 机制注入（协议阶段 4/5）→ 完整度 → 后果 → 涌现 → 准则 → 指纹
    assert keys == ["persona", "primitives", "injection", "levels", "consequences",
                    "emergence", "conduct", "fingerprint"]
    for step in steps:
        assert step["title"] and step["why"] and step["text"]
        assert step["kind"] in ("text", "rows", "lines")
        if step["kind"] == "rows":
            assert step["rows"], f"{step['key']} 声明为 rows 却没有内容"
        if step["kind"] == "lines":
            assert step["lines"], f"{step['key']} 声明为 lines 却没有内容"

    by_key = {s["key"]: s for s in steps}
    assert len(by_key["primitives"]["rows"]) == 31
    levels = {r["id"]: r for r in by_key["levels"]["rows"]}
    assert levels["attention_shifting"]["level"] == 0.2
    assert levels["attention_shifting"]["impaired"] is True
    assert levels["inhibition"]["impaired"] is False
    # 受损后果只列出真正受损的过程
    assert [r["id"] for r in by_key["consequences"]["rows"]] == ["attention_shifting"]


def test_intact_prompt_has_no_impairment_section():
    steps = build_brain_prompt_steps(intact())
    by_key = {s["key"]: s for s in steps}
    assert by_key["consequences"]["kind"] == "text"
    assert "所有基础过程都完整" in by_key["consequences"]["text"]
    assert by_key["levels"]["rows"][0]["impaired"] is False
    # 完整心智没有「被修改的计算配置」这回事 → 不出现注入段
    assert "injection" not in by_key


def test_generic_degradation_appears_as_level_row():
    steps = build_brain_prompt_steps(generic_degradation_control(0.5))
    rows = {s["key"]: s for s in steps}["levels"]["rows"]
    assert any(r["id"] == "(generic)" and r["level"] == 0.5 for r in rows)


# ---------------------------------------------------------------------------
# 盲法（规则 5）：提问者 / 分析者看不到 θ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,build", PERTURBATION_CASES, ids=[c[0] for c in PERTURBATION_CASES])
def test_examiner_and_analyst_prompts_are_blind_to_theta(label, build):
    """θ 的数值、primitive id、完整度措辞都不得出现在提问者/分析者的提示词里。"""
    perturbation = build()
    forbidden = [f"{v:.2f}" for v in perturbation.values.values()]
    forbidden += list(perturbation.values.keys())
    forbidden += ["完整度", "generic_degradation", "综合退化"]

    texts = [
        build_examiner_system_prompt(turn_budget=6),
        build_examiner_turn_prompt(turn_index=2, turn_budget=6, transcript="提问者：A\n孩子：B"),
        build_analyst_system_prompt(),
        build_analyst_prompt("提问者：A\n孩子：B"),
    ]
    for text in texts:
        for token in forbidden:
            assert token not in text, f"盲法泄漏：{token!r} 出现在提示词里"


def test_dialogue_prompts_never_contain_theta_even_after_running():
    """跑完一整段对话后，提问者的每一轮提示词仍然不含 θ。"""
    seen_prompts: list[str] = []
    perturbation = single("switching_cost", 0.2)

    class SpyProvider(MockProvider):
        def chat(self, messages, *, temperature=None, item=None, perturbation=None,
                 seed=0, role="", context=None, max_tokens=None):
            if role == "examiner":
                assert "switching_cost" not in messages[1].content
                assert "0.20" not in messages[1].content
                seen_prompts.append(messages[1].content)
            return super().chat(messages, temperature=temperature, item=item,
                                perturbation=perturbation, seed=seed, role=role,
                                context=context, max_tokens=max_tokens)

    brain = VirtualAgent(MockProvider(), perturbation, model_id="mock:m")
    run_dialogue(brain, SpyProvider(), n_turns=3, max_turns=3)
    assert len(seen_prompts) == 3
    assert all("switching_cost" not in p for p in seen_prompts)


# ---------------------------------------------------------------------------
# 解析回退
# ---------------------------------------------------------------------------


def test_parse_examiner_question_variants():
    assert parse_examiner_question("你今天过得怎么样？\nQUESTION: 你今天过得怎么样？") == "你今天过得怎么样？"
    assert parse_examiner_question("问题：你愿意分享吗？") == "你愿意分享吗？"
    assert parse_examiner_question("好的，那么你愿意分享吗？") == "那么你愿意分享吗？"
    assert parse_examiner_question("") == ""
    # 裸剥单字会篡改真正的问题，因此「那」「好」这类开头必须保留
    assert parse_examiner_question("那天发生了什么？") == "那天发生了什么？"
    assert parse_examiner_question("好孩子会怎么做？") == "好孩子会怎么做？"
    # 多段时取首段，而不是把整段说明都当成问题
    assert parse_examiner_question("你愿意分享吗？\n\n（说明：这是第 1 轮）") == "你愿意分享吗？"


def test_parse_analyst_json_valid_and_fallbacks():
    good = json.dumps({"dims": {"social_cognition": 0.4}, "confidence": 0.7,
                       "evidence": [{"dim": "social_cognition", "quote": "q", "reading": "r"}],
                       "narrative": "n", "notes": ""}, ensure_ascii=False)
    payload, error = parse_analyst_json(good)
    assert error == "" and payload["dims"] == {"social_cognition": 0.4}
    assert payload["evidence"][0]["quote"] == "q"

    # 代码块包裹
    payload, error = parse_analyst_json(f"```json\n{good}\n```")
    assert error == "" and payload["dims"]["social_cognition"] == 0.4

    # 非 JSON：正则兜底救回 dims，且必须显式报告格式问题
    payload, error = parse_analyst_json("我给出的判断是 social_cognition: 0.35 左右")
    assert payload is not None and error
    assert payload["dims"]["social_cognition"] == 0.35

    # 完全无法解析：返回 None + 说明，不能静默当成「一切正常」
    payload, error = parse_analyst_json("完全不是结构化输出")
    assert payload is None and error


def test_parse_analyst_json_clamps_out_of_range():
    payload, _ = parse_analyst_json('{"dims": {"repetitive_restricted_behavior": 3.5, "x": -2}}')
    assert payload["dims"] == {"repetitive_restricted_behavior": 1.0, "x": 0.0}


# ---------------------------------------------------------------------------
# Mock 三角色
# ---------------------------------------------------------------------------


def test_mock_examiner_returns_parsable_question():
    provider = MockProvider()
    for turn in range(6):
        question = parse_examiner_question(provider.chat([], role="examiner", seed=turn))
        assert question and len(question) > 5


def test_mock_analyst_returns_parsable_json_with_requested_dims():
    provider = MockProvider()
    from pacs.llm.provider import ChatMessage

    messages = [
        ChatMessage("system", build_analyst_system_prompt()),
        ChatMessage("user", build_analyst_prompt("提问者：A\n孩子：我不知道。")),
    ]
    payload, error = parse_analyst_json(provider.chat(messages, role="analyst"))
    assert error == ""
    # 默认观察目标 = 计分通道（含对照通道）
    assert set(payload["dims"]) == set(SCORED_CHANNELS)


def test_mock_brain_reply_tracks_theta_impairment():
    """mock 大脑的回应质量必须随 θ 单调下降（否则离线管线无法验证）。"""
    rich = MockProvider().chat([], perturbation=intact(), seed=0)
    poor = MockProvider().chat([], perturbation=single("switching_cost", 0.2), seed=0)
    assert len(rich) > len(poor)


# ---------------------------------------------------------------------------
# 对话探针
# ---------------------------------------------------------------------------


def test_run_dialogue_alternates_and_streams_turns():
    seen: list[dict] = []
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=4, max_turns=4, on_turn=seen.append)

    assert session.n_brain_turns == 4
    assert len(session.turns) == 8
    assert [t["role"] for t in session.turns] == ["examiner", "brain"] * 4
    assert [t["turn"] for t in session.turns] == [1, 1, 2, 2, 3, 3, 4, 4]
    # 逐轮回调（界面据此做实时可视化）
    assert len(seen) == 8
    # 4 轮问不完 7 个方面，且每方面都只问到 1 个问题：两类缺口都必须写进 problems
    # （不静默），而不是假装覆盖全了。定长模式（max_turns == n_turns）下没法补问，
    # 因此「不足配额」这一条是如实的结论，不是流程失败。
    assert any("都没有被探查到" in p and "覆盖 4/7 个方面" in p for p in session.problems)
    assert any("不足 5 个" in p for p in session.problems)
    assert session.uncovered_channels and session.under_quota_channels


def test_run_dialogue_reports_no_coverage_gap_when_every_channel_is_probed():
    """每个通道都问到了 → 没有「整段没探查到」的缺口（配额不足另记，见 test_turn_quota）。"""
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    goals = list(SCORED_CHANNELS)[:3]
    session = run_dialogue(brain, MockProvider(), n_turns=3, max_turns=3, goals=goals)
    assert session.uncovered_channels == []
    assert not any("都没有被探查到" in p for p in session.problems)
    assert [t["focus"] for t in session.turns if t["role"] == "examiner"] == goals
    assert session.focus_channels == goals


def test_run_dialogue_records_the_focus_of_every_examiner_turn():
    """自行出题的轮次没有题库条目，`focus`（提问者自报）是该轮唯一的维度归属线索。"""
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=7, max_turns=7)
    examiner = [t for t in session.turns if t["role"] == "examiner"]
    assert all(t["focus"] in SCORED_CHANNELS for t in examiner)
    assert len({t["focus"] for t in examiner}) == 7
    assert session.focus_channels == [t["focus"] for t in examiner]


def test_run_dialogue_uses_fixed_question_list_when_given():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, None, questions=["Q1?", "Q2?"], n_turns=5)
    assert [t["content"] for t in session.turns if t["role"] == "examiner"] == ["Q1?", "Q2?"]
    assert session.problems and "问题清单已用尽" in session.problems[0]


def test_run_dialogue_stops_gracefully_on_empty_questions():
    class SilentExaminer(MockProvider):
        def chat(self, messages, *, role="", **kwargs):
            if role == "examiner":
                return ""
            return super().chat(messages, role=role, **kwargs)

    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, SilentExaminer(), n_turns=3)
    assert session.turns == []           # 没有半截轮次
    assert session.problems and "没有产出可解析的问题" in session.problems[0]


def test_run_dialogue_rejects_out_of_range_turns():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    with pytest.raises(ValueError):
        run_dialogue(brain, MockProvider(), n_turns=0)
    with pytest.raises(ValueError):
        run_dialogue(brain, MockProvider(), n_turns=ABSOLUTE_MAX_TURNS + 1)


def test_analyze_dialogue_feedback_carries_no_theta():
    """观测通路上不允许出现 θ：分析者反馈里既没有基准真值，也没有任何 θ 的痕迹。

    这是「扰动只输入给 LLM、表型只由提问与观察得到」两条规则在代码上的落点 ——
    `analyze_dialogue` 的签名里根本没有 θ，所以它不可能把 θ 写进读数，
    读数也因此不可能与扰动构成「已知条件的关联」。
    """
    perturbation = single("attention_shifting", 0.2)
    brain = VirtualAgent(MockProvider(), perturbation, model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=4, max_turns=4)
    feedback = analyze_dialogue(MockProvider(), session)

    assert feedback["error"] == ""
    # 分析者按**计分通道**评分：6 个表型维度 + 对照通道（后者不进雷达图，但必须落库）。
    # 没探查到的通道没有读数（记为空，不按典型计），未覆盖情况结构化记在 session 里。
    assert set(feedback["dims"]) <= set(SCORED_CHANNELS)
    assert 0.0 <= feedback["mean_reading"] <= 1.0
    assert set(session.uncovered_channels) == set(SCORED_CHANNELS) - set(feedback["dims"])

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    primitive_ids = {p.id for p in PRIMITIVES}
    leaked = sorted(set(keys(feedback)) & primitive_ids)
    assert not leaked, f"分析者反馈里出现了 θ 的过程名：{leaked}"
    assert "blind_audit" not in feedback
    # 受损心智的盲法读数应明显低于完整心智
    intact_brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    intact_fb = analyze_dialogue(MockProvider(), run_dialogue(intact_brain, MockProvider(), n_turns=4, max_turns=4))
    assert intact_fb["mean_reading"] > feedback["mean_reading"]


def test_dialogue_session_round_trips_through_dict():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=2, max_turns=2, goals=("social_cognition",))
    restored = DialogueSession.from_dict(session.to_dict())
    assert restored.turns == session.turns
    assert restored.goals == ("social_cognition",)
    assert restored.transcript() == session.transcript()


# ---------------------------------------------------------------------------
# 量表上传与模式分流
# ---------------------------------------------------------------------------

NUMERIC_ITEM = {
    "id": "sc1", "battery_dim": "social_cognition",
    "prompt": "他会怎么想？", "options": ["A", "B"], "key": 1,
    "primitive_sensitivity": {"belief_updating": 0.4},
}
CUSTOM_DIM_ITEM = {
    "id": "cx1", "battery_dim": "某个自定义维度",
    "prompt": "自由作答", "options": ["A", "B"], "key": 0,
    "primitive_sensitivity": {"inhibition": 0.3},
}


def test_itemset_is_always_llm_scored():
    """只作出处信息保留：带 key / 敏感权重的量表也按 LLM 逐条打分，且要显式说明。"""
    set_id, normalized, problems = parse_item_set(
        {"name": "量表A", "items": [NUMERIC_ITEM, dict(NUMERIC_ITEM, id="sc2")]}
    )
    assert problems == []
    assert set_id
    assert normalized["scoring_mode"] == SCORING_LLM
    assert [i["id"] for i in normalized["items"]] == ["sc1", "sc2"]
    # 旧字段只作出处信息保留，且必须有可见提示（不能让人以为还在按正确率算）
    assert normalized["items"][0]["legacy"]["primitive_sensitivity"] == {"belief_updating": 0.4}
    assert any("只作出处信息保留" in n for n in normalized["notices"])


def test_itemset_marks_custom_dimensions_as_out_of_downstream_scope():
    """自定义维度照常打分，但要说清下游机制分析不覆盖它们。"""
    _, normalized, problems = parse_item_set({"name": "量表B", "items": [CUSTOM_DIM_ITEM]})
    assert problems == []
    assert normalized["scoring_mode"] == SCORING_LLM
    assert any("自定义维度" in n for n in normalized["notices"])
    assert [d["builtin"] for d in normalized["dims"]] == [False]


def test_itemset_validation_reports_every_problem():
    _, _, problems = parse_item_set({"items": [
        {"id": "", "battery_dim": "repetitive_restricted_behavior", "prompt": "p", "options": ["A", "B"]},        # 缺 id
        {"id": "x", "battery_dim": "repetitive_restricted_behavior", "prompt": "p", "options": ["A"]},            # 选项不足
        {"id": "y", "battery_dim": "repetitive_restricted_behavior", "prompt": "p", "options": ["A", "B"], "key": 5},  # key 越界
        {"id": "x", "battery_dim": "repetitive_restricted_behavior", "prompt": "p", "options": ["A", "B"]},       # id 重复
        {"id": "z", "battery_dim": "repetitive_restricted_behavior", "options": ["A", "B"]},                      # 缺题干
    ]})
    joined = "；".join(problems)
    assert "缺少 id" in joined
    assert "至少要有 2 个选项" in joined
    assert "超出选项范围" in joined
    assert "id 重复" in joined
    assert "缺少 prompt" in joined


def test_itemset_rejects_non_list_and_empty():
    assert parse_item_set({"items": "not a list"})[2]
    assert parse_item_set({"items": []})[2]
    assert parse_item_set("not a dict")[2]


# ---------------------------------------------------------------------------
# 大脑库（规则 7）
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    with Store(tmp_path / "t.db") as s:
        yield s


def _save(store: Store, name: str, perturbation, **kwargs) -> str:
    from pacs.api.server import brain_payload, _new_brain_id

    return store.save_brain(_new_brain_id(), name, perturbation.theta_id,
                            brain_payload(name, perturbation, **kwargs),
                            source=kwargs.get("source", "user"))


def test_brain_save_list_rename_delete(store):
    brain_id = _save(store, "完整心智", intact())
    assert store.get_meta("active_brain_id") is None

    store.set_meta("active_brain_id", brain_id)
    assert store.get_meta("active_brain_id") == brain_id

    brains = store.list_brains()
    assert len(brains) == 1
    assert brains[0]["name"] == "完整心智"
    assert brains[0]["payload"]["describe"] == "完整心智（无扰动）"

    assert store.rename_brain(brain_id, "基线") is True
    assert store.load_brain(brain_id)["name"] == "基线"

    assert store.delete_brain(brain_id) is True
    assert store.list_brains() == []
    assert store.load_brain(brain_id) is None


def test_brain_dedupes_by_theta():
    """同一个 θ 重复保存只更新名字，不产生两条记录（否则大脑库会被刷屏）。"""
    with Store(":memory:") as store:
        store.db_path = ":memory:"          # 仅用于占位，实际用内存连接
        first = _save(store, "全脑-初版", single("switching_cost", 0.2))
        second = _save(store, "全脑-改名", single("switching_cost", 0.2))
        assert first == second
        brains = store.list_brains()
        assert len(brains) == 1
        assert brains[0]["name"] == "全脑-改名"


def test_brain_perturbation_round_trip():
    from pacs.api.server import brain_payload, perturbation_of_brain

    original = compose(single("inhibition", 0.2), single("switching_cost", 0.4))
    record = {"brain_id": "b1", "name": "n", "theta_id": original.theta_id,
              "payload": brain_payload("n", original)}
    restored = perturbation_of_brain(record)
    assert restored.theta_id == original.theta_id
    assert restored.values == original.values


def test_dialogue_and_feedback_persist(store):
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=2, max_turns=2)
    store.save_dialogue_session(session.session_id, session.to_dict())
    for index, turn in enumerate(session.turns):
        store.save_dialogue_turn(session.session_id, index, turn["role"], turn["content"], turn)

    sessions = store.list_dialogue_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session.session_id
    assert len(store.load_dialogue_turns(session.session_id)) == 4

    store.save_feedback(session.session_id, session.theta_id, "mock:m", {"dims": {"repetitive_restricted_behavior": 0.5}})
    assert store.load_feedback(session.session_id)["dims"] == {"repetitive_restricted_behavior": 0.5}


def test_item_set_persist_and_delete(store):
    set_id, normalized, _ = parse_item_set({"name": "量表E", "items": [NUMERIC_ITEM]})
    store.save_item_set(set_id, normalized["name"], normalized["scoring_mode"], normalized)
    sets = store.list_item_sets()
    assert len(sets) == 1 and sets[0]["scoring_mode"] == SCORING_LLM
    assert store.delete_item_set(set_id) is True
    assert store.list_item_sets() == []


def test_counts_include_new_tables(store):
    counts = store.counts()
    for table in ("brains", "dialogues", "feedback", "item_sets"):
        assert table in counts
