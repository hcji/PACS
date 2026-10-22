"""观察者内置条目库测试：完整性、映射规则、盲法、选题、以及「不是随手编的」。

条目库由 `tools/build_examiner_bank.py` 从 `resources/asdkb/`（AsdKB 知识库精选）
生成后随包分发。这里守住的几件事：
- 随包 JSON 与它自己声明的统计**自洽**（条数、来源摘要、各层数量）；
- 每个条目都带出处与施测方式，且**儿童面向语句只给人工重写过的条目**；
- 维度判定遵守「先内容维度、后对照通道」的优先级（否则社交条目会被误判到对照通道）；
- 用条目库构造的提问者/分析者提示词**仍然对 θ 盲法**；
- 选题随年龄与目标变化、可复现、支持用户指定条目。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pacs.agents.dialogue import analyze_dialogue, run_dialogue
from pacs.agents.virtual_agent import VirtualAgent
from pacs.phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS
from pacs.brain import examiner_bank as EB
from pacs.brain.architecture import build_analyst_system_prompt, build_examiner_system_prompt
from pacs.brain.perturb import intact, single
from pacs.llm.provider import MockProvider

ALL_DIMS = tuple(PHENOTYPE_DIMS)


@pytest.fixture(scope="module")
def bank():
    return EB.load_bank()


# ---------------------------------------------------------------------------
# 完整性与自洽
# ---------------------------------------------------------------------------


def test_bank_ships_with_package(bank):
    """条目库必须在包内（随包分发），而不是依赖外部 resources 目录。"""
    assert EB.BANK_PATH.exists()
    assert "src" in str(EB.BANK_PATH) and "data" in str(EB.BANK_PATH)
    assert len(bank["items"]) == 441


def test_bank_has_only_one_source(bank):
    """唯一来源为 AsdKB：不得再残留任何其它来源（例如被移出的量表 PDF）。"""
    meta = bank["meta"]
    assert meta["source"]["id"] == "ASDKB"
    assert set(meta["source"]["sha256_16"]) == {
        "tools.json", "symptoms.json", "diagnostic_criteria.json", "treatments.json"
    }
    raw = json.dumps(bank, ensure_ascii=False)
    for gone in ("WS/T", "WST580", "发育行为评估量表", "wst580"):
        assert gone not in raw, f"仍残留旧来源痕迹：{gone}"


def test_bank_stats_are_self_consistent(bank):
    items, stats = bank["items"], bank["stats"]
    assert stats["items_total"] == len(items) == 441
    assert stats["probe_ready"] == sum(1 for i in items if i.get("probe_zh"))
    assert stats["curated_probes"] == sum(1 for i in items if i["probe_basis"] == "curated")
    assert stats["observation_only"] == sum(1 for i in items if not i.get("probe_zh"))
    # 探查方式分面：直接提问 / 情境式 / 不可探查，三者之和为总数
    modes = stats["by_probe_mode"]
    assert set(modes) == {"question", "scenario", "none"}
    assert sum(modes.values()) == len(items)
    assert modes["scenario"] > modes["question"]        # 情境式是本库的主要施测形态
    assert stats["scenario_ready"] == sum(1 for i in items if i.get("scenario_zh"))
    assert stats["scales"] == 20
    assert stats["scales_with_items"] == 14
    assert len(stats["metadata_only_scales"]) == 6
    assert (stats["symptoms"], stats["diagnostic_criteria"], stats["treatments"]) == (65, 43, 28)
    # 运行时算出来的统计与文件里写的一致（同一口径）
    live = EB.bank_stats()
    assert live["items_total"] == len(items)
    assert live["by_admin"] == stats["by_admin"]
    assert live["by_dim"] == stats["by_dim"]


def test_reference_layers_are_complete(bank):
    assert len(bank["symptoms"]) == 65
    assert len(bank["criteria"]) == 43
    assert len(bank["treatments"]) == 28
    assert EB.symptoms()[0]["dims"] and EB.symptoms()[0]["category"]
    assert all(t["type"] == "EBP" for t in EB.treatments())
    # 症状分类已归一到三类（源数据的 RRB 拼写分裂必须已被修复）
    categories = {s["category"] for s in bank["symptoms"]}
    assert categories == {
        "impairments_in_social_interaction",
        "restrictive_repetitive_and_stereotyped_behaviors",
        "other_symptoms",
    }


def test_every_item_carries_provenance(bank):
    for item in bank["items"]:
        assert item["key"] and item["source_id"] and item["source_name"]
        assert item["source_ref"], f"{item['key']} 缺少出处定位"
        assert item["admin"] in ("caregiver_report", "clinician_rating",
                                 "self_report", "observation")
        assert set(item["mapping_basis"]) == {"dims", "primitives", "probe", "symptoms"}
        assert item["dims"], f"{item['key']} 没有维度归属"


def test_sources_are_recorded_with_digests(bank):
    source = bank["meta"]["source"]
    assert source["stats"]["scales"] == 20
    assert source["stats"]["items"] == 441
    assert all(len(v) == 16 for v in source["sha256_16"].values())


def test_mapping_rules_are_documented(bank):
    """映射规则必须可复核：每条维度映射有理由、每条 primitive 规则有关键词与依据。"""
    rules = bank["mapping_rules"]
    assert set(rules["dim_keywords"]) == set(ALL_DIMS)
    assert all(kw for kw in rules["dim_keywords"].values())
    # 维度优先级就是 6 个表型维度；对照通道已不是表型维度，不应出现在里面
    assert tuple(rules["dim_priority"]) == tuple(ALL_DIMS)
    assert "general_cognition" not in rules["dim_priority"]
    assert rules["dim_priority_note"]
    assert set(rules["scale_dims"]) == {f"tools{i}" for i in range(20)}
    for sid, dims in rules["scale_dims"].items():
        assert dims and sid in rules["scale_dims_rationale"]
    assert len(rules["primitive_keywords"]) == 31
    for pid, rule in rules["primitive_keywords"].items():
        assert rule["keywords"] and rule["why"].strip()
    assert rules["criteria_domain_rule"]
    assert set(rules["symptom_category_dims"]) == {
        "impairments_in_social_interaction",
        "restrictive_repetitive_and_stereotyped_behaviors",
        "other_symptoms",
    }
    assert set(rules["broken_age_fields"]) == {"tools3", "tools5"}


# ---------------------------------------------------------------------------
# 源数据的性质必须体现在数据里
# ---------------------------------------------------------------------------


def test_no_scale_is_child_administered(bank):
    """AsdKB 的 20 套量表没有一套是儿童直接施测的 —— 该事实必须体现在数据里。"""
    admins = {i["admin"] for i in bank["items"]}
    assert admins <= {"caregiver_report", "clinician_rating", "self_report", "observation"}
    assert "direct" not in admins
    # 自评量表（AQ / AQ-10）必须被标出来，不能当幼儿条目用
    for sid in ("tools3", "tools4"):
        aq = [i for i in bank["items"] if i["source_id"] == sid]
        assert aq and all(i["admin"] == "self_report" for i in aq)
        # 自评量表存的是成人第一人称措辞，不作为儿童直接提问的来源
        assert all(not i.get("question_zh") for i in aq)
    # CARS-2 是纯临床构念，不可直接询问儿童
    cars = [i for i in bank["items"] if i["source_id"] == "tools7"]
    assert cars and all(i["admin"] == "clinician_rating" for i in cars)


def test_chat_section_b_is_marked_as_examiner_task(bank):
    items = {i["key"]: i for i in bank["items"]}
    assert items["tools5:8"]["admin"] == "caregiver_report"      # A 段（0–8）
    assert items["tools5:9"]["admin"] == "observation"           # B 段（9–13）
    assert items["tools5:12"]["probe_zh"]


def test_scale_level_symptom_caveat_is_recorded(bank):
    sample = next(i for i in bank["items"] if i["source_id"] == "tools19")
    note = sample["symptoms_source_scale_coverage"]["note"]
    assert "量表级覆盖" in note
    assert "117" in note and "319" in note       # 把不可用的证据也写清楚


def test_criteria_domain_is_rerouted_for_mislabeled_entries(bank):
    """源数据把 D29–D42（受限重复行为）错标成社交互动分类，本库须重新判定。"""
    criteria = {c["id"]: c for c in bank["criteria"]}
    assert criteria["D31"]["domain"] == "rrb"                    # 简单的躯体刻板运动
    assert criteria["D1"]["domain"] == "social_communication"     # 社交情感互动中的缺陷
    assert all(c["domain"] in ("rrb", "social_communication") for c in bank["criteria"])


def test_probe_basis_is_explicit(bank):
    """每条都要说清探查从哪来：人工撰写 / 模板派生 / 无；标注必须与数据自洽。"""
    for item in bank["items"]:
        basis = item["probe_basis"]
        assert basis in ("curated", "template", "none"), item["key"]
        if basis == "none":
            assert item["probe_mode"] == "none" and not item["probe_zh"]
            assert item["observation"], f"{item['key']} 既无探查又无观察要点"
            continue
        # 有探查：方式必须是提问或情境之一
        assert item["probe_mode"] in ("question", "scenario"), item["key"]
        if item["probe_mode"] == "scenario":
            assert item["scenario_zh"] and item["scenario"], item["key"]
            assert "情境：" in item["scenario_zh"] and "提问：" in item["scenario_zh"]
        if basis == "template":
            # 模板派生：没有人工撰写的内容，情境必须来自模板
            assert not item["question_zh"] and item["scenario"]["basis"] == "template", item["key"]
        else:
            # 人工撰写：要么有手写问句，要么情境是手写的
            handwritten = bool(item["question_zh"]) or (
                item.get("scenario") or {}
            ).get("basis") == "curated"
            assert handwritten, f"{item['key']} 标注为人工撰写但找不到人工内容"


def test_dim_priority_prefers_content_dims(bank):
    """含社交语义的条目不得被判到对照通道（对照通道是对照，不是兜底去处）。"""
    regression = {"tools1:2", "tools1:14", "tools14:1", "tools19:42"}
    items = {i["key"]: i for i in bank["items"]}
    for key in regression:
        item = items.get(key)
        if item is None:
            continue
        assert "general_cognition" not in item["dims"], f"{key} 被误判到对照通道：{item['text_zh']}"


def test_dims_basis_is_recorded(bank):
    """每条都要说明维度是逐题命中还是回落到量表先验。"""
    for item in bank["items"]:
        assert item["dims_basis"] in ("keyword", "scale")
        if item["dims_basis"] == "keyword":
            assert item["dims_matched_terms"], f"{item['key']} 标为关键词命中但没有命中词"
        else:
            assert item["dims"] == item["scale_dims"]
    basis = bank["stats"]["by_dims_basis"]
    assert basis["keyword"] + basis["scale"] == 441


def test_control_channel_is_not_overused(bank):
    """对照通道只应接收真正的一般认知条目，不应成为兜底去处。"""
    control = [i for i in bank["items"] if "general_cognition" in i["dims"]]
    assert len(control) < 60, f"对照通道条目过多（{len(control)}），疑似泛化关键词污染"


# ---------------------------------------------------------------------------
# 盲法：带条目库内容时依然不能泄漏 θ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [
    lambda: single("attention_shifting", 0.2),
    lambda: single("belief_updating", 0.4),
])
def test_bank_informed_prompts_stay_blind(build):
    perturbation = build()
    probes = EB.dialogue_probes(goals=ALL_DIMS, n=5)
    rubric = EB.observation_rubric(goals=ALL_DIMS)
    texts = [
        build_examiner_system_prompt(turn_budget=5,
                                     probe_plan=EB.render_probe_plan(probes)),
        build_analyst_system_prompt(rubric=EB.render_rubric(rubric)),
    ]
    forbidden = [f"{v:.2f}" for v in perturbation.values.values()] + list(perturbation.values)
    forbidden += ["完整度", "generic_degradation", "theta_"]
    for text in texts:
        for token in forbidden:
            assert token not in text, f"盲法泄漏：{token!r}"


def test_probe_plan_and_rubric_are_rendered_for_prompts():
    probes = EB.dialogue_probes(goals=ALL_DIMS, n=3)
    plan = EB.render_probe_plan(probes)
    for index, item in enumerate(probes, start=1):
        assert f"{index}. [" in plan
    rubric = EB.render_rubric(EB.observation_rubric(limit=3))
    assert rubric.startswith("- [")
    assert "知情者报告" in rubric or "临床评定" in rubric


def test_scenario_plan_renders_setup_and_how_to_administer():
    """情境式计划必须写明：情境、处理方式提问、观察目标与施测方式。"""
    probes = EB.dialogue_probes(goals=ALL_DIMS, n=3, mode="scenario")
    assert probes and all(p.get("scenario") for p in probes)
    plan = EB.render_probe_plan(probes, mode="scenario")
    assert "情境：" in plan and "提问：" in plan
    assert "观察目标：" in plan
    assert "这种情况你会怎么处理" in plan
    # 不照读、按回答追问，是对提问者的硬要求
    assert "不要照读" in plan


# ---------------------------------------------------------------------------
# 选题
# ---------------------------------------------------------------------------


def test_dialogue_probes_are_age_appropriate():
    probes = EB.dialogue_probes(goals=ALL_DIMS, n=6, seed=42)
    assert len(probes) == 6
    assert all(p["probe_zh"] for p in probes)
    assert all(EB.age_fit(p, 36) != "above" for p in probes)


def test_dialogue_probes_cover_each_goal():
    probes = EB.dialogue_probes(goals=ALL_DIMS, n=5, seed=3)
    covered = {d for p in probes for d in p["dims"]}
    assert set(ALL_DIMS) <= covered


def test_dialogue_probes_are_deterministic_per_seed():
    first = [p["key"] for p in EB.dialogue_probes(goals=ALL_DIMS, n=6, seed=11)]
    second = [p["key"] for p in EB.dialogue_probes(goals=ALL_DIMS, n=6, seed=11)]
    assert first == second
    # 6 个目标正好用满 6 个名额 → 选题由「逐目标覆盖」决定，种子没有发挥空间
    assert set(first) == {p["key"] for p in EB.dialogue_probes(
        goals=ALL_DIMS, n=6, seed=12)}
    # 名额多于目标数时才轮到随机补齐，此时不同种子应给出不同计划
    more = [p["key"] for p in EB.dialogue_probes(goals=ALL_DIMS, n=8, seed=12)]
    assert len(more) == 8 and more[:6] == first
    assert more != [p["key"] for p in EB.dialogue_probes(
        goals=ALL_DIMS, n=8, seed=99)]


def test_dialogue_probes_shift_with_age():
    young = {p["key"] for p in EB.dialogue_probes(age_months=18, goals=(), n=6, seed=5)}
    mid = {p["key"] for p in EB.dialogue_probes(goals=(), n=6, seed=5)}
    older = {p["key"] for p in EB.dialogue_probes(age_months=84, goals=(), n=6, seed=5)}
    assert mid != young
    assert len({frozenset(young), frozenset(mid), frozenset(older)}) >= 2


def test_dialogue_probes_respect_explicit_keys():
    keys = ["tools1:3", "tools14:4", "tools5:12"]
    probes = EB.dialogue_probes(keys=keys)
    assert [p["key"] for p in probes] == keys


def test_items_by_keys_ignores_unknown_keys():
    assert EB.items_by_keys(["tools1:0", "不存在"])[0]["key"] == "tools1:0"
    assert EB.items_by_keys(["不存在"]) == []


def test_observation_rubric_prefers_clinician_items():
    rubric = EB.observation_rubric(goals=("social_cognition",), limit=6)
    assert rubric
    assert all(r["admin"] != "self_report" for r in rubric)
    assert rubric[0]["admin"] == "clinician_rating"        # 临床评定条目最能锚定评分


def test_age_band_handles_broken_age_field():
    items = {i["key"]: i for i in EB.load_bank()["items"]}
    broken = next(i for i in items.values() if i["age_field_broken"])
    assert EB.age_band(broken) == (None, None)
    assert EB.age_fit(broken, 36) == "unknown"             # 不据污染字段判断适配


# ---------------------------------------------------------------------------
# 浏览筛选
# ---------------------------------------------------------------------------


def test_list_items_filters():
    assert EB.list_items(limit=1)["total"] == 441
    probe = EB.list_items(probe_ready=True, limit=1)
    assert probe["total"] == EB.bank_stats()["probe_ready"]
    assert all(i["probe_zh"] for i in probe["items"])
    scenarios = EB.list_items(probe_mode="scenario", limit=3)
    assert scenarios["total"] == EB.bank_stats()["by_probe_mode"]["scenario"]
    assert all(i["probe_mode"] == "scenario" for i in scenarios["items"])
    by_dim = EB.list_items(dim="social_communication_pragmatics", limit=5)
    assert by_dim["total"] > 0
    assert all("social_communication_pragmatics" in i["dims"] for i in by_dim["items"])
    by_prim = EB.list_items(primitive="switching_cost", limit=5)
    assert all("switching_cost" in i["primitives"] for i in by_prim["items"])
    by_admin = EB.list_items(admin="self_report", limit=3)
    assert all(i["admin"] == "self_report" for i in by_admin["items"])
    by_src = EB.list_items(source_id="tools19", limit=3)
    assert all(i["source_id"] == "tools19" for i in by_src["items"])
    assert EB.list_items(query="假装", limit=3)["total"] > 0
    assert EB.list_items(age=36, limit=3)["total"] > 0
    paged = EB.list_items(limit=10, offset=10)
    assert len(paged["items"]) == 10
    assert paged["items"][0]["key"] != EB.list_items(limit=10)["items"][0]["key"]


def test_bank_stats_expose_facets():
    stats = EB.bank_stats()
    assert set(stats["by_dim"]) == set(ALL_DIMS)
    assert len(stats["by_primitive"]) == 31
    assert stats["by_admin"]["caregiver_report"] > 0
    assert sum(stats["by_scale"].values()) == 441


# ---------------------------------------------------------------------------
# 与对话探针的接线
# ---------------------------------------------------------------------------


def test_dialogue_with_probe_items_records_provenance():
    probes = EB.dialogue_probes(goals=ALL_DIMS, n=3, seed=42)
    brain = VirtualAgent(MockProvider(), single("attention_shifting", 0.2), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), probe_items=probes, n_turns=3,
                           max_turns=3, goals=ALL_DIMS)

    assert session.probe_keys == [p["key"] for p in probes]
    assert session.probe_sources                               # 出处可追溯
    assert session.probe_dims                                  # 覆盖的维度
    asked = [t["content"] for t in session.turns if t["role"] == "examiner"]
    assert len(asked) == len(probes)
    examiner_turns = [t for t in session.turns if t["role"] == "examiner"]
    for turn, probe in zip(examiner_turns, probes):
        # 呈现里必须带上该条目的内容（情境式则为情境与提问两段），且能解析出干净的问题
        head = probe["probe_zh"].split("\n")[0].lstrip("情境：")
        assert head.split(" ")[0] in turn["content"] or turn["question"]
        assert turn["question"]
        # 每轮要么用位置上的这条条目，要么是提问者自己出题（`probe_key=None` + 自报方面）。
        # 「自造题也照抄位置条目的 key」是假的出处，已改掉。
        assert turn["probe_key"] == (probe["key"]
                                     if turn["focus"] in (probe.get("dims") or {}) else None)
    assert any(t["probe_key"] for t in examiner_turns)     # 计划里的条目确实被用到过


def test_dialogue_falls_back_to_free_questions_without_bank():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=2, max_turns=2)
    assert session.probe_keys == []
    asked = [t["content"] for t in session.turns if t["role"] == "examiner"]
    assert len(asked) == 2 and all(asked)


def test_analysis_records_rubric():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    session = run_dialogue(brain, MockProvider(), n_turns=2, max_turns=2)
    feedback = analyze_dialogue(
        MockProvider(), session,
        rubric_items=EB.observation_rubric(limit=4),
    )
    assert len(feedback["rubric_keys"]) == 4
    assert feedback["error"] == ""
    # 分析者按**计分通道**评分（含对照通道）；题库本身只覆盖 6 个表型维度。
    # 没有被探查到的通道记为空读数（不在 dims 里），并在 session 里结构化列出 ——
    # 「没问过」不等于「表现典型」。
    assert set(feedback["dims"]) <= set(SCORED_CHANNELS)
    assert feedback["dims"], "至少应有被探查到的通道读数"
    assert set(session.uncovered_channels) <= set(SCORED_CHANNELS)
    assert "expectation_keys" not in feedback      # 该概念已随旧来源一并移除


# ---------------------------------------------------------------------------
# 生成器可复现（需要 asdkb 源数据时才跑）
# ---------------------------------------------------------------------------

_KB = Path(r"D:\ASD Agent\resources\asdkb")


@pytest.mark.skipif(not (_KB / "tools.json").is_file(), reason="本机没有 asdkb 源数据（CI 上跳过）")
def test_generator_reproduces_identical_bank(tmp_path):
    """用同一份源数据重跑生成器，条目/统计/映射规则必须逐字段一致。

    只比对**内容**：来源摘要里的 `path` 会随源目录位置变化，但 sha256 与各层计数必须相同 ——
    这样才能证明「随包条目库」确实由这份源数据生成，而不是手工改出来的。
    """
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "tools" / "build_examiner_bank.py"
    out = tmp_path / "regen.json"
    subprocess.run(
        [sys.executable, str(script), "--kb", str(_KB), "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    regenerated = json.loads(out.read_text(encoding="utf-8"))
    shipped = json.loads(EB.BANK_PATH.read_text(encoding="utf-8"))

    assert regenerated["items"] == shipped["items"]
    assert regenerated["stats"] == shipped["stats"]
    assert regenerated["mapping_rules"] == shipped["mapping_rules"]
    assert regenerated["symptoms"] == shipped["symptoms"]
    assert regenerated["criteria"] == shipped["criteria"]
    assert regenerated["treatments"] == shipped["treatments"]
    assert (regenerated["meta"]["source"]["sha256_16"]
            == shipped["meta"]["source"]["sha256_16"])
