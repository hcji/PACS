"""冻结协议测试（导出与相关矩阵已随 13 维链整体删除）。

保留的断言集中在：冻结五类对象的语义与摘要、篡改检测、以及**冻结对象里不含任何
primitive → 测量维度 的对应**。"""

from __future__ import annotations

import json

import pytest

from pacs.agents.verbalize import anti_circularity_check, verbalize_profile
from pacs.phenotype import SCORED_CHANNELS
from pacs.phenotype import PhenotypeProfile
from pacs.freeze import (
    build_freeze,
    compare_freeze,
    digest,
    save_freeze,
    verify_freeze,
)
from pacs.ontology.primitives import DEFAULT_ACTIVE

#: 计分通道 = 6 个表型维度 + 对照通道（对照通道不是表型维度，但会落进剖面）
ALL_DIMS = tuple(SCORED_CHANNELS)


def _profile(theta=None, dims=None, *, theta_id="theta_test", generic=0.0) -> PhenotypeProfile:
    theta = theta if theta is not None else {p: 1.0 for p in DEFAULT_ACTIVE}
    dims = dims or {d: 0.75 for d in ALL_DIMS}
    return PhenotypeProfile(
        theta_id=theta_id,
        model_id="mock:test",
        run_id="run_test",
        theta=theta,
        generic_degradation=generic,
        dims=dict(dims),
        n_items=35,
        n_parsed=35,
    )


# ---------------------------------------------------------------------------
# 描述集契约
# ---------------------------------------------------------------------------


def test_verbalize_output_has_no_numeric_assignment_or_primitive_id():
    text = verbalize_profile(_profile())
    assert anti_circularity_check(text) == []


def test_verbalize_bands_reflect_phenotype():
    low = verbalize_profile(_profile(dims={**{d: 0.75 for d in ALL_DIMS}, "repetitive_restricted_behavior": 0.20}))
    high = verbalize_profile(_profile(dims={**{d: 0.75 for d in ALL_DIMS}, "repetitive_restricted_behavior": 0.95}))
    assert low != high
    assert "仪式化流程" in low         # 低分档的措辞（重复性与限制性行为维度）
    assert "行为方式多样" in high    # 高分档的措辞


def test_anti_circularity_catches_leaked_numeric_assignment():
    problems = anti_circularity_check("该儿童 social_reward = 0.32，表现偏弱。")
    assert problems


def test_anti_circularity_catches_primitive_id():
    assert anti_circularity_check("注意 switching_cost 出现困难。")
    assert anti_circularity_check("他的 switching_cost 受损。") != []


# ---------------------------------------------------------------------------
# 冻结协议
# ---------------------------------------------------------------------------


def _freeze():
    from pacs.analysis.heterogeneity import find_convergence

    profiles = [
        _profile(),
        _profile({"ambiguity_tolerance": 0.2}, {**{d: 0.9 for d in ALL_DIMS}, "social_communication_pragmatics": 0.45}, theta_id="a"),
        _profile({"novelty_valuation": 0.2}, {**{d: 0.9 for d in ALL_DIMS}, "social_communication_pragmatics": 0.48, "repetitive_restricted_behavior": 0.4}, theta_id="b"),
    ]
    convergence = find_convergence(profiles, "social_communication_pragmatics")
    return build_freeze(
        perturbations=[{"theta_id": p.theta_id, "design": p.design, "theta": p.theta} for p in profiles],
        convergence=convergence,
    )


def test_freeze_freezes_five_object_classes():
    """五类冻结对象；第三类已从「表型映射」重新定义为「表型词表与探查登记」。"""
    snapshot = _freeze()
    assert set(snapshot.digests) == {
        "primitive_definitions",
        "perturbation_parameters",
        "phenotype_vocabulary",
        "candidate_mechanisms",
        "human_predictions",
    }
    vocab = snapshot.objects["phenotype_vocabulary"]
    assert set(vocab) == {"channels", "partition", "anchors", "probes", "systems"}
    assert list(vocab["channels"]) == list(SCORED_CHANNELS)
    # 探查登记以 primitive 为键（「用什么探查这个机制」），但每个值里**只能有探查元数据**，
    # 不得出现任何测量通道名；也不得有 dim_order / mappings 这类映射表。
    for pid, entry in vocab["probes"].items():
        assert set(entry) == {"primary_probe", "n_controls", "dose_levels"}, (pid, entry)
        assert not (set(entry) & set(SCORED_CHANNELS)), pid
    assert "dim_order" not in vocab and "mappings" not in vocab


def test_frozen_primitive_definitions_have_no_dimension_targets():
    snapshot = _freeze()
    for entry in snapshot.objects["primitive_definitions"].values():
        assert "dim_map_targets" not in entry


def test_freeze_verifies_clean():
    assert verify_freeze(_freeze()) == []


def test_freeze_id_is_deterministic():
    assert _freeze().freeze_id == _freeze().freeze_id


def test_freeze_includes_negative_predictions():
    snapshot = _freeze()
    predictions = snapshot.objects["human_predictions"]
    assert "negative_predictions" in predictions


def test_tampering_is_detected():
    snapshot = _freeze()
    snapshot.objects["primitive_definitions"]["switching_cost"]["definition"] = "被篡改"
    assert any("摘要与内容不符" in p for p in verify_freeze(snapshot))


def test_changing_any_frozen_object_changes_freeze_id():
    a = _freeze()
    b = _freeze()
    b.objects["phenotype_vocabulary"]["channels"] = ["x"]
    b.digests["phenotype_vocabulary"] = digest(b.objects["phenotype_vocabulary"])
    b.freeze_id = "freeze_" + digest(b.digests)
    assert a.freeze_id != b.freeze_id
    comparison = compare_freeze(a, b)
    assert comparison["identical"] is False
    assert "phenotype_vocabulary" in comparison["changed_objects"]


def test_save_freeze_writes_snapshot_and_predictions(tmp_path):
    out = save_freeze(_freeze(), tmp_path)
    assert (out / "snapshot.json").exists()
    assert (out / "human_predictions.json").exists()
    payload = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["freeze_id"].startswith("freeze_")


def test_digest_is_key_order_independent():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
