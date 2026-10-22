"""扰动引擎测试：θ 的构造、采样与序列化。"""

from __future__ import annotations

import pytest

from pacs.brain import perturb as P


# ---------------------------------------------------------------------------
# 扰动引擎
# ---------------------------------------------------------------------------


def test_intact_has_all_primitives_at_one():
    p = P.intact()
    assert p.is_intact
    assert all(v == 1.0 for v in p.values.values())
    assert len(p.values) == 31


def test_single_perturbs_exactly_one_primitive():
    """FR-1 标准 3：可以单独扰动。"""
    p = P.single("social_reward_weighting", 0.4)
    active = p.active_perturbations()
    assert list(active) == ["social_reward_weighting"]
    assert active["social_reward_weighting"] == 0.4
    assert len(p.values) == 31


def test_single_rejects_bad_dose_and_unknown_primitive():
    with pytest.raises(ValueError):
        P.single("switching_cost", 0.0)
    with pytest.raises(ValueError):
        P.single("switching_cost", 1.5)
    with pytest.raises(KeyError):
        P.single("not_a_primitive", 0.5)


def test_theta_id_is_deterministic_and_value_derived():
    a = P.single("switching_cost", 0.4)
    b = P.single("switching_cost", 0.4)
    c = P.single("switching_cost", 0.6)
    assert a.theta_id == b.theta_id          # 同 θ 同 id
    assert a.theta_id != c.theta_id          # 不同 θ 不同 id
    # 与设计/seed 无关（仅由取值决定）
    assert P.single("switching_cost", 0.4, seed=999).theta_id == a.theta_id


def test_roundtrip_serialization():
    p = P.single("inhibition", 0.3)
    restored = P.Perturbation.from_dict(p.to_dict())
    assert restored.theta_id == p.theta_id
    assert restored.values == p.values


def test_compose_takes_weakest_level():
    """组合扰动：同一 primitive 取最弱（min），通用退化取 max。"""
    a = P.single("switching_cost", 0.6)
    b = P.single("switching_cost", 0.3)
    combined = P.compose(a, b)
    assert combined.level_of("switching_cost") == 0.3


def test_compose_unions_different_primitives():
    combined = P.compose(P.single("switching_cost", 0.4), P.single("inhibition", 0.5))
    assert set(combined.active_perturbations()) == {"switching_cost", "inhibition"}


def test_generic_degradation_keeps_primitives_intact():
    """Control 1：通用退化下所有 primitive 应保持完整。"""
    p = P.generic_degradation_control(0.3)
    assert p.is_intact is False
    assert p.active_perturbations() == {}
    assert p.generic_degradation == 0.3


def test_sample_singles_covers_each_primitive_and_dose():
    designs = P.sample_singles()
    intacts = [d for d in designs if d.is_intact]
    perturbed = [d for d in designs if not d.is_intact]
    # 5 个基线重复（降低基线噪声）+ 31 primitive × 4 个非完整 dose
    assert len(intacts) == 5
    assert len(perturbed) == 31 * 4
    per_primitive = {}
    for d in perturbed:
        (pid,) = d.active_perturbations()
        per_primitive.setdefault(pid, []).append(d.level_of(pid))
    assert len(per_primitive) == 31
    for pid, doses in per_primitive.items():
        assert sorted(doses) == [0.2, 0.4, 0.6, 0.8]


def test_sample_lhs_is_deterministic_and_in_range():
    a = P.sample_lhs(n_samples=12, seed=7)
    b = P.sample_lhs(n_samples=12, seed=7)
    c = P.sample_lhs(n_samples=12, seed=8)
    assert [d.theta_id for d in a] == [d.theta_id for d in b]
    assert [d.theta_id for d in a] != [d.theta_id for d in c]
    for d in a:
        for value in d.values.values():
            assert P.LHS_LOW <= value <= 1.0


def test_sample_lhs_stratifies_each_dimension():
    """LHS 的每个维度应在每层各落一个样本（分层性质）。"""
    n = 10
    designs = P.sample_lhs(primitives=("switching_cost",), n_samples=n, seed=3)
    levels = sorted(d.level_of("switching_cost") for d in designs)
    # 分层 → 任一层内不应出现两个样本
    strata = {min(n - 1, int((v - P.LHS_LOW) / (1.0 - P.LHS_LOW) * n)) for v in levels}
    assert len(strata) == n


def test_sample_pairs_are_pairs():
    designs = P.sample_pairs(primitives=("switching_cost", "inhibition", "novelty_valuation"), dose=0.4)
    assert len(designs) == 3          # C(3,2)
    for d in designs:
        assert len(d.active_perturbations()) == 2


def test_interaction_sign_known_cases():
    # 基线 0.9；单独各掉到 0.7（损伤量 0.2 each）；纯可加 → 组合 0.5
    assert P.interaction_sign(0.7, 0.7, 0.5, 0.9) == "additive"
    assert P.interaction_sign(0.7, 0.7, 0.3, 0.9) == "superadditive"   # 比可加更差
    assert P.interaction_sign(0.7, 0.7, 0.8, 0.9) == "subadditive"     # 比可加更好
