"""本体层测试：primitive / Level 1 组合 / 探查登记（assay）。

**不测任何映射**：primitive → Model 1 13 维 与 primitive → 表型维度 两条预设对应都已
整体删除（构造属性与测量维度之间不允许有预设对应）。这里额外守住「探查登记里不含
任何维度声称」，防止映射以别的名字回流。
"""

from __future__ import annotations

import pytest

from pacs.ontology import assays, primitives, systems


def test_primitive_count_is_thirty_one():
    assert len(primitives.PRIMITIVES) == 31


def test_ontology_self_check_is_clean():
    assert primitives.validate_ontology() == []


def test_systems_self_check_is_clean():
    assert systems.validate_systems() == []


def test_assays_self_check_is_clean():
    """每个 primitive 都必须有独立 assay + 特异性对照（FR-1 标准 2、5）。"""
    assert assays.validate_assays() == []
    assert assays.primitives_without_assay() == ()


def test_every_primitive_has_multiple_dose_levels():
    for p in primitives.PRIMITIVES:
        assert len(p.dose_levels) >= 2, p.id
        assert all(0.0 < d <= 1.0 for d in p.dose_levels), p.id
        assert 1.0 in p.dose_levels


def test_assays_carry_no_dimension_claims():
    """红线：探查登记只说明「用什么探查、哪些是对照、dose 档位」，不含维度字段。"""
    for a in assays.ASSAYS:
        assert not hasattr(a, "target_dim")
        assert not hasattr(a, "primary_probe") or not hasattr(a.primary_probe, "target_dim")
        assert not hasattr(a.primary_probe, "dims")
        assert "维度" not in a.expected_direction.replace("哪个测量维度", "")
        assert len(a.dose_levels) >= 2
        assert a.n_controls > 0


def test_ontology_declares_no_dimension_targets():
    """本体层不得再有「primitive 影响哪些维度」的字段或表。"""
    for p in (*primitives.PRIMITIVES, *primitives.CANDIDATES):
        assert not hasattr(p, "dim_map_targets")
    assert not hasattr(primitives, "DIM_MAP")
    assert "dim_map" not in dir(primitives)


def test_unknown_primitive_raises():
    with pytest.raises(KeyError):
        primitives.get("no_such_primitive")


def test_level1_systems_are_combinations_not_primitives():
    """红线：ToM / 认知灵活性 / 语用不得出现在 Level 0 本体中。"""
    level0 = set(primitives.all_ids())
    for s in systems.SYSTEMS:
        assert s.id not in level0
        assert len(s.members) >= 2
    # Level 1 成员必须全部在 Level 0 注册
    for s in systems.SYSTEMS:
        for m in s.members:
            assert m in primitives.PRIMITIVE_INDEX, (s.id, m)


def test_systems_of_maps_primitive_to_high_level():
    assert "theory_of_mind" in systems.systems_of("belief_updating")
    assert "pragmatic_interpretation" in systems.systems_of("ambiguity_tolerance")
