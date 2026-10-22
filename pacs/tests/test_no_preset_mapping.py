"""红线守卫：测量维度与构造属性之间**不得有任何预设对应**。

这些测试是「构造效度」这一条要求的可执行形式（`README.md`：扰动与表型之间不允许有任何
已知条件的关联）。它们守四件事：

1. **无映射**：代码里不存在任何「primitive → 测量维度」的表（含旧的 `dim_map` /
   `dim_map_targets` / `target_dim` 家族），探查登记里也没有任何维度字段；
2. **盲法**：提问者与分析者的提示词里不出现 primitive id、θ 数值、也不出现任何映射表；
3. **读数只由提问与观察得到**：表型向量 `B` 的每一维都能追到「分析者读了这段对话」，
   没被探查到的通道记为空读数（不是「表现典型」）；
4. **合成数据不得冒充涌现**：离线 mock 的剖面必须被标成 `synthetic`，并在 Atlas 里
   带免责行。

用 AST 而不是纯文本扫描：docstring 里解释「什么被删掉了」是必要的记载，不算代码使用。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pacs
from pacs import freeze as freeze_mod
from pacs.agents.dialogue import analyze_dialogue, run_dialogue
from pacs.agents.measure import measure_phenotype, provider_kind_of
from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain.architecture import (
    build_analyst_prompt,
    build_analyst_system_prompt,
    build_examiner_system_prompt,
    build_examiner_turn_prompt,
)
from pacs.brain import examiner_bank as EB
from pacs.brain.perturb import intact, single
from pacs.llm.provider import MockProvider
from pacs.ontology import assays
from pacs.ontology.primitives import PRIMITIVES
from pacs.phenotype import SCORED_CHANNELS

#: 一旦回流就说明预设对应又出现了的代码名字
BANNED_NAMES = {
    "dim_map",
    "dim_map_targets",
    "target_dim",
    "DIM_MAP",
    "DIM_INDEX",
    "DIM_ORDER",
    "dim_order",              # 旧导出清单里的 13 维顺序
    "project_theta",
    "validate_dim_map",
    "primitive_targets",
    "TYPICAL_ANCHORS",
    "ASD_ANCHORS",
    "structural_correlation",
    "cse_contract",
    "export_cases",
    "phenotype_mapping",      # 冻结快照里旧的第 3 类对象名
    "unmapped_dims",          # 旧描述集里的缺口字段
}

_SOURCE_ROOT = Path(pacs.__file__).parent


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """所有 docstring 字面量的 id（它们允许提到旧名字，用于记载「什么被删掉了」）。"""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _banned_uses(path: Path) -> list[str]:
    """模块里对禁用名字的**代码级**使用（import / 属性 / 变量名 / 字符串常量键）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_nodes(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            hits.append(f"{path.name}:{node.lineno} Name({node.id})")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            hits.append(f"{path.name}:{node.lineno} Attribute(.{node.attr})")
        elif isinstance(node, ast.Constant) and id(node) not in docstrings \
                and node.value in BANNED_NAMES:
            # 例如 getattr(p, "target_dim") 或 entry["dim_order"] —— 字符串键同样是使用
            hits.append(f"{path.name}:{node.lineno} Constant({node.value!r})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] in BANNED_NAMES:
                    hits.append(f"{path.name}:{node.lineno} Import({alias.name})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[-1] in BANNED_NAMES:
                hits.append(f"{path.name}:{node.lineno} ImportFrom({node.module})")
            for alias in node.names:
                if alias.name in BANNED_NAMES:
                    hits.append(f"{path.name}:{node.lineno} ImportFrom({alias.name})")
        elif isinstance(node, ast.keyword) and node.arg in BANNED_NAMES:
            hits.append(f"{path.name}:{node.lineno} keyword({node.arg}=)")
    return hits


def test_no_mapping_tables_anywhere_in_source():
    """全源码扫描：不存在任何 primitive → 维度 映射的代码级使用。"""
    hits: list[str] = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        hits += _banned_uses(path)
    assert not hits, "预设对应回流了：\n" + "\n".join(hits)


def test_ontology_module_has_no_mapping_module():
    with pytest.raises(ModuleNotFoundError):
        __import__("pacs.ontology.dim_map")
    with pytest.raises(ModuleNotFoundError):
        __import__("pacs.export")


def test_probe_registry_has_no_dimension_fields():
    """探查登记只回答「用什么探查、哪些是对照、dose 档位」。"""
    for a in assays.ASSAYS:
        assert not hasattr(a, "target_dim")
        assert not hasattr(a.primary_probe, "target_dim")
        assert not hasattr(a.primary_probe, "dims")
        assert not hasattr(a.primary_probe, "primitives")


def test_primitives_declare_no_dimension_targets():
    for p in PRIMITIVES:
        assert not hasattr(p, "dim_map_targets")


# ---------------------------------------------------------------------------
# 盲法：提问者 / 分析者的提示词里不能有 θ 或任何映射
# ---------------------------------------------------------------------------


def _leak_tokens(perturbation) -> list[str]:
    """会泄漏 θ 或映射的记号。"""
    tokens = [f"{v:.2f}" for v in perturbation.values.values()]
    tokens += list(perturbation.values)
    tokens += ["完整度", "generic_degradation", "theta_", "BRAIN_FINGERPRINT"]
    tokens += sorted(BANNED_NAMES)
    return tokens


@pytest.mark.parametrize("perturbation", [
    single("attention_shifting", 0.2),
    single("belief_updating", 0.4),
    intact(),
])
def test_examiner_and_analyst_prompts_stay_blind(perturbation):
    probes = EB.dialogue_probes(goals=tuple(SCORED_CHANNELS), n=7)
    rubric = EB.observation_rubric(goals=tuple(SCORED_CHANNELS), limit=6)
    texts = {
        "examiner": build_examiner_system_prompt(
            turn_budget=7, probe_plan=EB.render_probe_plan(probes)
        ),
        "examiner_turn": build_examiner_turn_prompt(
            turn_index=2, turn_budget=7, transcript="提问者：你叫什么？\n孩子：我叫小明。"
        ),
        "analyst": build_analyst_system_prompt(rubric=EB.render_rubric(rubric)),
        "analyst_input": build_analyst_prompt(
            "提问者：今天换了座位你会怎么想？\n孩子：我会先看看新同桌。",
            tasks=list(probes[:3]),
            uncovered=(("general_cognition", "一般认知控制（对照通道）"),),
        ),
    }
    for name, text in texts.items():
        for token in _leak_tokens(perturbation):
            assert token not in text, f"{name} 提示词里泄漏了：{token!r}"
    # primitive id 一个都不能出现（θ 的过程名本身就是构造属性）
    primitive_ids = {p.id for p in PRIMITIVES}
    for name, text in texts.items():
        assert not (primitive_ids & set(text.split())), name
        for pid in primitive_ids:
            assert pid not in text, f"{name} 提示词里出现了 primitive id：{pid}"


def test_conclusion_path_stays_blind_to_theta():
    """结论这条路也要盲法：分析者的输入（含覆盖披露）与它写出的结论里都不能有 θ。

    结论是要显示给人看的一段话，最容易被顺手写成「因为抑制受损，所以……」——
    那就不再是测量，而是把 ground truth 抄了一遍。这里用带扰动的真跑一遍来守。
    """
    from pacs.llm.provider import ChatMessage

    perturbation = single("salience_assignment", 0.2)
    brain = VirtualAgent(MockProvider(), perturbation, model_id="mock:m", seed=11)
    seen: list[str] = []

    class Spy(MockProvider):
        def chat(self, messages, **kwargs):
            if kwargs.get("role") == "analyst":
                seen.extend(m.content for m in messages)
            return super().chat(messages, **kwargs)

    session = run_dialogue(brain, Spy(), n_turns=7, max_turns=7, seed=11)
    feedback = analyze_dialogue(Spy(), session)

    assert seen, "分析者的输入没有经过 spy"
    assert feedback["conclusion"]
    for token in _leak_tokens(perturbation):
        for text in seen:
            assert token not in text, f"分析者输入泄漏了 {token!r}"
        assert token not in feedback["conclusion"], f"结论泄漏了 {token!r}"
    for p in PRIMITIVES:
        assert p.id not in feedback["conclusion"]
    # 表型印象是结构化的，且不含 θ
    assert set(feedback["asd_likeness"]) == {"present", "level", "basis"}


def test_analyst_input_discloses_coverage_but_not_theta():
    """覆盖披露（没问到 / 问得不够）不是 θ：它说的是证据，不是预期的维度归属。"""
    covered = run_dialogue(
        VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=9),
        MockProvider(), n_turns=7, max_turns=7, goals=tuple(SCORED_CHANNELS)[:3], seed=9,
    )
    text = build_analyst_prompt(
        covered.transcript(),
        uncovered=(("general_cognition", "一般认知控制（对照通道）"),),
        under_quota=(("social_cognition", "社会认知", 2),),
        quota=5,
    )
    assert "没有被探查到的方面" in text
    assert "问题数少于 5 个" in text and "只问到 2 个问题" in text
    # 披露里出现的是通道 id/中文名与**证据量**，不是「某个机制该体现在这里」
    assert "机制" not in text and "primitive" not in text


def test_examiner_is_told_which_aspects_have_no_bank_item():
    """题库里没有条目的方面必须被点名要求自造题 —— 覆盖情况不能是静默的。"""
    from pacs.phenotype import SCORED_CHANNELS as CH

    probes = EB.dialogue_probes(goals=tuple(CH), n=7)
    covered = tuple(sorted({d for i in probes for d in (i.get("dims") or {})}))
    assert "general_cognition" not in covered          # 对照通道确实没有内置条目
    text = build_examiner_system_prompt(
        goals=tuple(CH), turn_budget=7,
        probe_plan=EB.render_probe_plan(probes), covered_dims=covered,
    )
    assert "没有**对应条目的方面" in text
    assert "general_cognition" in text
    # 不传 covered_dims 时不打印这一段（覆盖情况未知就不许声称）
    silent = build_examiner_system_prompt(
        goals=tuple(CH), turn_budget=7,
        probe_plan=EB.render_probe_plan(probes),
    )
    assert "没有**对应条目的方面" not in silent


def test_analyst_input_discloses_missing_evidence_without_naming_mechanisms():
    """告诉分析者「哪些方面没被探查到」是证据披露，不是维度期望，也不能带机制名。"""
    text = build_analyst_prompt(
        "提问者：你叫什么？\n孩子：小明。",
        uncovered=(("general_cognition", "一般认知控制（对照通道）"),),
    )
    assert "没有被探查到的方面" in text
    assert "general_cognition" in text
    assert "保守" in text
    for p in PRIMITIVES:
        assert p.id not in text


# ---------------------------------------------------------------------------
# 读数只由提问与观察得到
# ---------------------------------------------------------------------------


def _mock_session(n_turns: int, *, adapt: bool = False):
    """跑一段对话并拿分析者反馈。

    默认**定长**（`max_turns == n_turns`）：需要「某些通道没有被探查到」这类情形时，
    自适应补问会把缺口填满，反而造不出被测场景。要测补问本身就传 `adapt=True`。
    """
    brain = VirtualAgent(MockProvider(), single("salience_assignment", 0.2), model_id="mock:m")
    session = run_dialogue(
        brain, MockProvider(), n_turns=n_turns,
        max_turns=(None if adapt else n_turns),
        goals=tuple(SCORED_CHANNELS),
    )
    return session, analyze_dialogue(MockProvider(), session)


def test_uncovered_channels_are_empty_readings_not_typical_values():
    """没问过的通道必须记为空读数并显式列出 —— 「没有证据」不等于「表现典型」。"""
    session, feedback = _mock_session(3)
    assert session.uncovered_channels
    for ch in session.uncovered_channels:
        assert feedback["dims"].get(ch) is None, f"{ch} 没有证据却给了读数"
        assert ch not in feedback.get("dimensions", {})
    # 未覆盖的通道要在 problems 里说出来
    assert any("没有证据支撑" in p for p in session.problems)


def test_every_reading_traces_back_to_a_conversation_turn():
    """每一维读数都必须能追到至少一轮对话（probe_dims 里有它的归属）。"""
    session, feedback = _mock_session(int(len(SCORED_CHANNELS)))
    covered = {ch for turn in session.turns if turn["role"] == "examiner" for ch in
               ([turn["focus"]] if turn.get("focus") else [])}
    assert set(feedback["dims"]) <= set(SCORED_CHANNELS)
    assert set(session.uncovered_channels) == set(SCORED_CHANNELS) - set(feedback["dims"])
    for ch in feedback["dims"]:
        assert ch in covered or ch in set(session.probe_dims), ch


def test_measurement_records_provider_kind():
    assert provider_kind_of(MockProvider()) == "mock"
    assert provider_kind_of(object()) == "llm"
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    result = measure_phenotype(brain, MockProvider(), MockProvider(), n_turns=3)
    assert result.profile.provider_kind == "mock"
    assert result.profile.synthetic is True
    assert result.profile.to_dict()["synthetic"] is True


def test_synthetic_profiles_are_disclaimed_in_atlas():
    """离线 mock 的剖面必须在 Atlas 里带显著免责行。"""
    from pacs.report import SYNTHETIC_DISCLAIMER, build_atlas, render_atlas

    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m")
    result = measure_phenotype(brain, MockProvider(), MockProvider(), n_turns=3)
    atlas = build_atlas([result.profile])
    assert atlas.has_synthetic
    assert any("离线 mock" in n for n in atlas.notes)
    md = render_atlas(atlas)
    assert SYNTHETIC_DISCLAIMER in md


def test_legacy_rows_infer_synthetic_from_model_id():
    """旧库里的剖面没有 provider_kind：必须从 model_id 前缀回推，否则规则构造的读数
    会被当成 LLM 涌现的证据。"""
    from pacs.phenotype import PhenotypeProfile

    base = {
        "theta_id": "t", "run_id": "r", "age_months": 36, "theta": {},
        "generic_degradation": 0.0, "dims": {"general_cognition": 0.9},
        "n_items": 7, "n_parsed": 7,
    }
    assert PhenotypeProfile.from_dict({**base, "model_id": "mock:mock-cognitive-mind"}).synthetic
    assert not PhenotypeProfile.from_dict({**base, "model_id": "openai:gpt-4o-mini"}).synthetic


def test_mechanism_profile_is_not_frozen():
    """估计出的机制—维度剖面是测量产物，禁止进冻结快照（否则会变成先验）。"""
    assert "mechanism_profile" not in freeze_mod._FROZEN_OBJECTS
    snapshot = freeze_mod.build_freeze(perturbations=[], convergence=None)
    assert "mechanism_profile" not in snapshot.objects
