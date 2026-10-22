"""按维度自己出题：提问者的授权、`FOCUS` 契约、逐轮归属与覆盖缺口。

这一批测试守住的是「题库从题目清单降为参考材料」之后的接线：
- 提示词必须**明确授权并鼓励**提问者按维度自己设计问题（否则退化回念题库）；
- 自己出的题没有条目可查，「这一轮问的是哪个方面」只能靠提问者自报 `FOCUS:` ——
  这个自报必须被解析、被校验（不能把中文名当成通道 id）、被逐轮记录；
- 逐轮归属（`turn_channels`）与覆盖度统计**共用同一条规则**，不能两处各算一遍；
- 整段对话没覆盖到的方面必须写进 `session.problems`（不静默：没有证据的通道，
  分析者只能给保守值，这件事必须留在记录里）；
- 离线 mock 在自行出题的轮次上按「通道 signature」估算敏感度 —— 它覆盖不到哪些
  primitive 是**已知局限**，在这里钉成断言，条目库一变就会暴露。
"""

from __future__ import annotations

from pacs.agents.dialogue import _brain_probe, analyze_dialogue, run_dialogue, turn_channels
from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain import examiner_bank as EB
from pacs.brain.architecture import build_examiner_system_prompt, build_examiner_turn_prompt
from pacs.brain.perturb import intact, single
from pacs.llm.parse import parse_examiner_focus, parse_examiner_turn, presentation_of
from pacs.llm.provider import ChatMessage, MockProvider
from pacs.ontology.primitives import PRIMITIVES
from pacs.phenotype import CONTROL_DIM, SCORED_CHANNELS

#: 「全部轮次都由提问者自造」的离线跑法下，channel signature 覆盖不到的 primitive。
#: 这是 mock 的近似（真实模型自行出题时考察什么，离线无从得知），不是机制主张；
#: 一旦条目库或 signature 算法变化，这个集合就会变，测试会立刻报出来。
OFFLINE_UNCOVERED = frozenset({
    "belief_updating",
    "confidence_calibration",
    "evidence_weighting",
    "familiarity_valuation",
    "information_valuation",
    "novelty_valuation",
    "selective_attention",
})


# ---------------------------------------------------------------------------
# 提问者提示词：题库仅供参考 + 明确授权自拟
# ---------------------------------------------------------------------------


def test_examiner_prompt_marks_the_bank_as_reference_material_only():
    prompt = build_examiner_system_prompt(
        goals=("social_cognition",), probe_plan="1. [social_cognition] 他为什么这么做？"
    )
    assert "仅供参考" in prompt
    assert "参考材料" in prompt
    # 计划里的条目原样进入提示词（可视化 ≡ 实际发送）
    assert "1. [social_cognition] 他为什么这么做？" in prompt


def test_examiner_prompt_authorises_self_authored_questions_without_a_plan():
    prompt = build_examiner_system_prompt(goals=("social_cognition",))
    assert "鼓励自己出题" in prompt
    assert "自己设计" in prompt
    assert "计划探查的方面" not in prompt      # 没有计划就不该出现这一节


def test_examiner_prompt_declares_the_focus_contract():
    prompt = build_examiner_system_prompt(goals=("social_cognition", CONTROL_DIM))
    assert "FOCUS:" in prompt
    assert "不要写中文名" in prompt
    # 合法的 FOCUS 取值必须在提示词里出现过，否则提问者无从知道能写哪些 id
    for channel in ("social_cognition", CONTROL_DIM):
        assert channel in prompt


def test_turn_prompt_names_the_uncovered_channels():
    prompt = build_examiner_turn_prompt(
        turn_index=3, turn_budget=7, remaining=("social_cognition", CONTROL_DIM)
    )
    assert "还没探查过的方面" in prompt
    assert SCORED_CHANNELS["social_cognition"] in prompt
    assert "social_cognition" in prompt
    assert CONTROL_DIM in prompt
    assert "自己设计" in prompt
    # 都覆盖到了 → 不再提示缺口
    assert "还没探查过的方面" not in build_examiner_turn_prompt(turn_index=3, turn_budget=7)


# ---------------------------------------------------------------------------
# FOCUS 的解析
# ---------------------------------------------------------------------------


def test_parse_examiner_focus_accepts_only_channel_ids():
    assert parse_examiner_focus("你愿意分享吗？\nQUESTION: 你愿意分享吗？\nFOCUS: social_cognition") \
        == "social_cognition"
    assert parse_examiner_focus("FOCUS：general_cognition。") == "general_cognition"
    assert parse_examiner_focus("FOCUS: `attention_shifting`") == "attention_shifting"
    # 中文名 / 整句话一律当作「没报」，交给上层按题库条目兜底
    assert parse_examiner_focus("FOCUS: 社会认知") == ""
    assert parse_examiner_focus("FOCUS: 社会认知（social_cognition）") == ""
    assert parse_examiner_focus("没有这一行") == ""
    assert parse_examiner_focus("") == ""


def test_parse_examiner_turn_keeps_the_scenario_in_the_presentation():
    raw = (
        "（设定情境）积木被推倒了\n他为什么这么做？\n"
        "QUESTION: 他为什么这么做？\nFOCUS: social_cognition"
    )
    question, focus = parse_examiner_turn(raw)
    assert question == "他为什么这么做？"
    assert focus == "social_cognition"
    presentation = presentation_of(raw, question)
    assert "积木被推倒了" in presentation          # 情境不能被丢掉
    assert "QUESTION:" not in presentation and "FOCUS" not in presentation


# ---------------------------------------------------------------------------
# 逐轮归属：自行出题的轮次按自报的方面归属
# ---------------------------------------------------------------------------


def test_turn_channels_prefers_the_self_reported_focus_when_it_is_off_plan():
    item = {"dims": {"social_cognition": 1.0}}
    # 计划条目问的是社会认知，但提问者自报问的是重复行为 → 以自报为准（它没照条目问）
    assert turn_channels(item, "repetitive_restricted_behavior") == ["repetitive_restricted_behavior"]
    # 自报落在条目维度里 → 按条目（条目带逐维度权重，比单通道更准）
    assert turn_channels(item, "social_cognition") == ["social_cognition"]


def test_turn_channels_falls_back_to_the_item_then_the_focus_then_nothing():
    item = {"dims": {"social_cognition": 1.0, "social_interaction_motivation": 1.0}}
    assert turn_channels(item, "") == ["social_cognition", "social_interaction_motivation"]
    assert turn_channels(None, "social_cognition") == ["social_cognition"]
    assert turn_channels(None, "") == []


# ---------------------------------------------------------------------------
# 覆盖度：缺口写进 problems，不静默
# ---------------------------------------------------------------------------


def test_coverage_note_names_exactly_the_channels_that_were_never_probed():
    goals = list(SCORED_CHANNELS)[:4]
    # 定长（max_turns == n_turns）才能造出「整段没问到」的通道：自适应模式会一直补问
    session = run_dialogue(
        VirtualAgent(MockProvider(), intact(), model_id="mock:m"),
        MockProvider(),
        n_turns=2,
        max_turns=2,
        goals=goals,
    )
    probed = [t["focus"] for t in session.turns if t["role"] == "examiner"]
    missing = [g for g in goals if g not in probed]
    assert missing, "这两轮不可能覆盖 4 个方面，测试前提有误"
    note = next(p for p in session.problems if "没有证据支撑" in p)
    assert f"覆盖 {len(goals) - len(missing)}/{len(goals)} 个方面" in note
    for channel in missing:
        assert SCORED_CHANNELS[channel] in note
    for channel in probed:
        assert SCORED_CHANNELS[channel] not in note


# ---------------------------------------------------------------------------
# 离线 mock：自行出题的轮次如何拿到敏感度
# ---------------------------------------------------------------------------


def test_mock_examiner_always_self_reports_a_valid_focus():
    provider = MockProvider()
    prompt = build_examiner_system_prompt()
    for turn in range(1, 8):
        raw = provider.chat(
            [ChatMessage("system", prompt), ChatMessage("user", "第 1 轮")],
            role="examiner",
            seed=turn,
        )
        assert parse_examiner_focus(raw) in SCORED_CHANNELS


def test_improvised_probe_uses_the_channel_signature():
    probe = _brain_probe([], 1, "social_cognition")
    assert probe["improvised"] is True
    assert set(probe["primitives"]) == set(EB.channel_signature("social_cognition"))
    assert set(probe["primitives"].values()) == {1.0}
    assert probe["dims"] == {"social_cognition": 1.0}


def test_improvised_probe_keeps_the_control_channel_insensitive():
    probe = _brain_probe([], 1, CONTROL_DIM)
    # 对照通道按构造不对任何单一机制敏感：只有通用退化该压低它（否则阴性预测在离线上就废了）
    assert probe["insensitive"] is True
    assert probe["primitives"] == {}
    # 认不出的方面 id → 不给条目，退回「没有条目信息」的兜底
    assert _brain_probe([], 1, "not_a_channel") is None
    assert _brain_probe([], 1, "") is None


def test_channel_signature_exists_for_every_scored_channel_but_the_control():
    for channel in SCORED_CHANNELS:
        signature = EB.channel_signature(channel)
        if channel == CONTROL_DIM:
            assert signature == ()
        else:
            assert len(signature) >= EB.SIGNATURE_TOP


def test_channel_signature_gap_is_the_documented_set():
    """已知局限钉死：条目库或 signature 算法一变，这条断言就会失败并要求更新文档。"""
    covered = {p for channel in SCORED_CHANNELS for p in EB.channel_signature(channel)}
    assert {p.id for p in PRIMITIVES} - covered == set(OFFLINE_UNCOVERED)


def test_improvised_dialogue_shows_selective_impairment_offline():
    """全程自行出题的离线跑法下，受损仍须可见，且只落在该 primitive 相关的通道上。"""
    impaired = "social_reward_weighting"
    goals = list(SCORED_CHANNELS)

    def readings(perturbation):
        provider = MockProvider()
        brain = VirtualAgent(provider, perturbation, model_id="mock:m")
        session = run_dialogue(brain, provider, n_turns=len(goals), goals=goals)
        assert session.problems == []
        return analyze_dialogue(provider, session)["dims"]

    intact_dims = readings(intact())
    impaired_dims = readings(single(impaired, 0.2))

    assert sum(impaired_dims.values()) < sum(intact_dims.values())
    dropped = {d for d, v in impaired_dims.items() if v < intact_dims[d]}
    related = {c for c in goals if impaired in EB.channel_signature(c)}
    assert dropped and dropped <= related, f"掉分通道 {dropped} 超出相关通道 {related}"


# ---------------------------------------------------------------------------
# 「问的话 / 记的条目 / 自报的方面」三者必须一致
# ---------------------------------------------------------------------------


def _plan_line(items: list[dict], n: int) -> str:
    plan = EB.render_probe_plan(items, mode="auto")
    return next(line for line in plan.splitlines() if line.strip().startswith(f"{n}."))


def test_bank_has_identical_text_twins_with_different_dims():
    """题库里有多条条目的探查语句**逐字相同**、维度声明却不同。

    这是 mock 不能靠文本回查定位条目的原因，也是这组测试的前提 —— 条目库换了要重看这里。
    """
    groups: dict[str, list[dict]] = {}
    for item in EB.load_bank()["items"]:
        text = (item.get("probe_zh") or "").strip()
        if text:
            groups.setdefault(text, []).append(item)
    twins = [g for g in groups.values() if len(g) > 1]
    assert twins, "条目库不再有同文条目 —— 本文件关于『不能靠文本回查』的假设需要复核"
    ambiguous = [
        g for g in twins
        if len({tuple(sorted(i.get("dims") or {})) for i in g}) > 1
    ]
    assert ambiguous, "同文条目的维度声明都相同 —— 回查错条目也不会改变 FOCUS，假设需复核"


def test_mock_examiner_takes_the_round_item_from_the_context():
    """给了 `probe_item` 时，mock 必须按**这一条**报 FOCUS，而不是按同文的孪生条目。"""
    twins = [
        g for g in (
            [i for i in EB.load_bank()["items"]
             if (i.get("probe_zh") or "") == (other.get("probe_zh") or "") and (i.get("probe_zh") or "")]
            for other in EB.load_bank()["items"]
        ) if len(g) > 1
    ]
    assert twins
    provider = MockProvider()
    system = build_examiner_system_prompt(
        goals=tuple(SCORED_CHANNELS), probe_plan=EB.render_probe_plan(twins[0], mode="auto")
    )
    turn = build_examiner_turn_prompt(turn_index=1, turn_budget=7)
    for item in twins[0]:
        raw = provider.chat(
            [ChatMessage("system", system), ChatMessage("user", turn)],
            role="examiner", seed=0,
            context={"probe_item": item, "plan_dims": list(item.get("dims") or ())},
        )
        dims = item.get("dims") or {}
        expected = max(dims.items(), key=lambda kv: (kv[1], kv[0]))[0]
        assert parse_examiner_focus(raw) == expected


def test_dialogue_keeps_item_question_and_focus_consistent():
    """整段对话里：这一轮自报的方面必须落在该轮条目的维度里，下发给大脑的也是同一条。

    不一致就会出现「答题按 A 条目估、证据记到 B 通道」的错配；对照通道那一轮尤其致命 ——
    条目权重会让它随扰动塌下去，而它必须对单一机制不敏感（阴性预测）。
    """
    goals = tuple(SCORED_CHANNELS)
    items = EB.dialogue_probes(goals=goals, n=7, seed=42)
    provider = MockProvider()
    session = run_dialogue(
        VirtualAgent(provider, intact(), model_id="mock:m"), provider,
        n_turns=7, goals=goals, probe_items=items, seed=42,
    )
    assert session.problems == []
    for turn in (t for t in session.turns if t["role"] == "examiner"):
        item = EB.item_by_key(turn["probe_key"]) if turn["probe_key"] else None
        probe = _brain_probe(list(items), turn["turn"], turn["focus"])
        if item is not None and turn["focus"] in (item.get("dims") or {}):
            assert probe is item, "自报方面落在条目维度里时，下发的应当就是这条条目"
            continue
        # 自报落在条目维度之外 = 提问者没照条目问。对照通道**总是**这种（题库里没有它的
        # 条目）；补问轮也可能这样 —— 那时提问者按配额要求换了话题，同样算自造题。
        assert (probe or {}).get("improvised") is True
        assert turn["probe_key"] is None, "自造题的轮次不该记一条实际没用的条目"
        # 自造题的伪条目只归属它自报的那一个通道（下发给大脑的权重也按它）
        assert set((probe or {}).get("dims") or {}) == {turn["focus"]}
