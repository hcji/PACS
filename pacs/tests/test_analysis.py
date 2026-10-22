"""分析层测试：统计、剂量-反应、特异性、机制异质性、亚型、贝叶斯、反事实。"""

from __future__ import annotations

import pytest

from pacs.analysis import (
    bayes,
    counterfactual,
    dose_response,
    heterogeneity,
    landscape,
    specificity,
    stats,
    subtypes,
)
from pacs.phenotype import SCORED_CHANNELS
from pacs.phenotype import PhenotypeProfile


def _profile(
    theta: dict,
    dims: dict,
    *,
    theta_id: str = "",
    model_id: str = "mock:test",
    generic: float = 0.0,
) -> PhenotypeProfile:
    return PhenotypeProfile(
        theta_id=theta_id or f"theta_{abs(hash(tuple(sorted(theta.items())))) % 10**8}",
        model_id=model_id,
        run_id="r",
        theta=theta,
        generic_degradation=generic,
        dims=dict(dims),
        n_items=35,
        n_parsed=35,
    )


#: 计分通道 = 6 个表型维度 + 对照通道（对照通道不是表型维度，但会落进剖面）
ALL_DIMS = tuple(SCORED_CHANNELS)


def _dims(**overrides) -> dict:
    base = {d: 0.90 for d in ALL_DIMS}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 统计（已知算例）
# ---------------------------------------------------------------------------


def test_mean_stdev():
    assert stats.mean([1, 2, 3]) == 2
    assert stats.mean([]) != stats.mean([])          # nan
    assert stats.stdev([2, 2, 2]) == 0.0
    # 样本标准差：1..5 → sqrt(2.5)
    assert stats.stdev([1, 2, 3, 4, 5]) == pytest.approx(2.5 ** 0.5)


def test_pearson_known_values():
    assert stats.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert stats.pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert stats.pearson([1, 2], [1, 2]) == 0.0      # n<3 → 0


def test_spearman_handles_monotone_nonlinear():
    xs = [1, 2, 3, 4, 5]
    ys = [1, 4, 9, 16, 25]                            # 严格单调但非线性
    assert stats.spearman(xs, ys) == pytest.approx(1.0)


def test_cohen_d_direction_and_zero():
    assert stats.cohen_d([1, 1, 1], [1, 1, 1]) == 0.0
    d = stats.cohen_d([0, 0, 0, 0], [1, 1, 1, 1])
    assert d < 0                                      # treatment 更低 → 负
    assert abs(d) > 0.8


def test_bootstrap_ci_is_deterministic_and_ordered():
    values = [0.2, 0.4, 0.6, 0.8, 1.0] * 4
    lo1, hi1 = stats.bootstrap_ci(values, seed=11)
    lo2, hi2 = stats.bootstrap_ci(values, seed=11)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1


def test_permutation_test_extremes():
    a = [1.0] * 8
    b = [0.0] * 8
    assert stats.permutation_test(a, b, n_permutations=500, seed=5) < 0.05
    same = [0.5] * 8
    assert stats.permutation_test(a[:4] + same[:4], a[:4] + same[:4], n_permutations=200) >= 0.5


def test_linear_fit_known_line():
    slope, intercept, r2 = stats.linear_fit([0, 1, 2, 3], [1, 3, 5, 7])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_is_monotonic():
    assert stats.is_monotonic([0.1, 0.3, 0.5])[0] is True
    assert stats.is_monotonic([0.5, 0.3, 0.1])[1] == "decreasing"
    assert stats.is_monotonic([0.1, 0.5, 0.3])[0] is False
    assert stats.is_monotonic([0.5, 0.5, 0.5])[1] == "flat"


def test_normal_conjugate_matches_closed_form():
    """先验 N(0,1)，观测 n 个 x̄，σ=1 → 后验均值 = n·x̄/(n+1)。"""
    mu, sigma = stats.normal_conjugate(0.0, 1.0, [1.0, 1.0, 1.0], 1.0)
    assert mu == pytest.approx(3 / 4)
    assert sigma == pytest.approx((1 / 4) ** 0.5)


# ---------------------------------------------------------------------------
# 剂量-反应
# ---------------------------------------------------------------------------


def test_dose_response_passes_on_clean_monotone_data():
    designs = [P_intact()]
    for dose in (0.8, 0.6, 0.4, 0.2):
        designs.append(
            _profile(
                {"ambiguity_tolerance": dose},
                _dims(social_communication_pragmatics=0.9 - 0.15 * (1 - dose) / 0.8 * 4 * 0.25),
            )
        )
    fits = dose_response.build_dose_response(designs, primitives=("ambiguity_tolerance",))
    assert len(fits) == 1
    assert fits[0].monotonic is True
    assert fits[0].slope > 0


def test_dose_response_flags_insufficient_data():
    fit = dose_response.fit_dose_response([_profile({"switching_cost": 0.4}, _dims())], "switching_cost")
    assert fit.verdict == "insufficient"


def P_intact():
    from pacs.ontology.primitives import DEFAULT_ACTIVE

    return _profile({p: 1.0 for p in DEFAULT_ACTIVE}, _dims(), theta_id="theta_intact")


# ---------------------------------------------------------------------------
# 特异性
# ---------------------------------------------------------------------------


def test_specificity_passes_when_one_dim_drops_and_control_holds():
    """主导维度由数据给出：只降一个维度、对照稳定 → pass（不再预设目标维度）。"""
    baseline = P_intact()
    probe = _profile(
        {"ambiguity_tolerance": 0.2},
        _dims(social_communication_pragmatics=0.45, general_cognition=0.88),   # 一个维度降、对照稳
    )
    verdicts = specificity.evaluate_specificity([baseline, probe])
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict == "pass"
    # 主导维度是从数据里选出来的（这里就是唯一大幅下降的那一维）
    assert v.dominant_dim == "social_communication_pragmatics"
    assert v.concentration > 0
    # 完整 Δ 剖面必须可见，而不是只留一个被挑出来的维度
    assert set(v.profile) >= set(ALL_DIMS)


def test_specificity_is_data_driven_not_preset():
    """同一个 primitive 的扰动，若数据里降的是**另一个**维度，判定就跟着那一维走。

    这是「构造属性与测量维度之间没有预设对应」的可执行证据：判定不查任何映射表，
    只查数据。
    """
    baseline = P_intact()
    probe = _profile(
        {"ambiguity_tolerance": 0.2},
        _dims(cognitive_flexibility_adaptation=0.40, general_cognition=0.88),
    )
    verdicts = specificity.evaluate_specificity([baseline, probe])
    assert verdicts[0].verdict == "pass"
    assert verdicts[0].dominant_dim == "cognitive_flexibility_adaptation"


def test_specificity_fails_when_control_also_drops():
    baseline = P_intact()
    probe = _profile(
        {"ambiguity_tolerance": 0.2},
        _dims(social_communication_pragmatics=0.45, general_cognition=0.40),   # 对照也崩
    )
    verdicts = specificity.evaluate_specificity([baseline, probe])
    assert verdicts[0].verdict == "fail_control"


def test_specificity_reports_no_effect_when_nothing_drops():
    baseline = P_intact()
    probe = _profile({"ambiguity_tolerance": 0.2}, _dims(social_communication_pragmatics=0.89))
    verdicts = specificity.evaluate_specificity([baseline, probe])
    assert verdicts[0].verdict == "no_effect"


def test_specificity_flags_diffuse_effect():
    """多个维度一起下降 → 不能归因于该单一机制（效应不集中）。"""
    baseline = P_intact()
    probe = _profile(
        {"ambiguity_tolerance": 0.2},
        _dims(social_communication_pragmatics=0.45, cognitive_flexibility_adaptation=0.46,
              general_cognition=0.88),
    )
    verdicts = specificity.evaluate_specificity([baseline, probe])
    assert verdicts[0].verdict == "fail_diffuse"


def test_mechanism_profile_is_measured_and_never_written_back():
    """机制—维度剖面的唯一来源是测量；它不得被当作先验回写（notes 里必须写死）。"""
    from pacs.analysis import mechanism_profile as mp

    baseline = P_intact()
    probe = _profile(
        {"ambiguity_tolerance": 0.2},
        _dims(social_communication_pragmatics=0.45, general_cognition=0.88),
    )
    report = mp.estimate_mechanism_profile([baseline, probe])
    assert report.rows
    row = report.rows[0]
    assert row.primitive_id == "ambiguity_tolerance"
    assert row.dominant_channel == "social_communication_pragmatics"
    assert row.deltas["general_cognition"] == pytest.approx(-0.02, abs=0.01)
    assert any("不得回写" in note for note in report.notes)
    assert "mechanism_profile" not in __import__("pacs.freeze", fromlist=["x"])._FROZEN_OBJECTS


def test_generic_degradation_control_detects_nonspecific_drop():
    baseline = P_intact()
    generic = _profile({}, _dims(**{d: 0.45 for d in ALL_DIMS}), generic=0.3)
    verdict = specificity.evaluate_generic_degradation([baseline, generic])
    assert verdict is not None
    assert verdict.is_nonspecific is True


def test_generic_degradation_control_not_triggered_when_only_control_dips():
    """只有对照通道掉、其它维度稳 → 应判为噪声而非通用退化。"""
    baseline = P_intact()
    generic = _profile({}, _dims(general_cognition=0.50), generic=0.3)
    verdict = specificity.evaluate_generic_degradation([baseline, generic])
    assert verdict is not None
    assert verdict.is_nonspecific is False


# ---------------------------------------------------------------------------
# 机制异质性
# ---------------------------------------------------------------------------


def _heterogeneity_fixture():
    baseline = P_intact()
    profiles = [
        baseline,
        # 机制 A：只坏 social_communication_pragmatics，且 social_cognition 也受损
        _profile(
            {"ambiguity_tolerance": 0.2},
            _dims(social_communication_pragmatics=0.45, social_cognition=0.50, repetitive_restricted_behavior=0.88),
            theta_id="th_a",
        ),
        # 机制 B：同样坏 social_communication_pragmatics，但 repetitive_restricted_behavior 受损、social_cognition 稳
        _profile(
            {"novelty_valuation": 0.2},
            _dims(social_communication_pragmatics=0.48, social_cognition=0.88, repetitive_restricted_behavior=0.40),
            theta_id="th_b",
        ),
    ]
    return profiles


def test_find_convergence_detects_two_mechanisms():
    result = heterogeneity.find_convergence(_heterogeneity_fixture(), "social_communication_pragmatics")
    assert result is not None
    assert result.n_candidates == 2
    assert result.is_heterogeneous is True


def test_convergence_excludes_endpoint_and_control_from_separation():
    """判别维度必须排除终点维度与对照组通道（否则会用噪声区分机制）。"""
    result = heterogeneity.find_convergence(_heterogeneity_fixture(), "social_communication_pragmatics")
    dims = [d for d, _ in result.distinguishing_dims]
    assert "social_communication_pragmatics" not in dims
    assert "general_cognition" not in dims
    assert "social_cognition" in dims and "repetitive_restricted_behavior" in dims


def test_minimal_task_set_is_subset_of_distinguishing_dims():
    result = heterogeneity.find_convergence(_heterogeneity_fixture(), "social_communication_pragmatics")
    assert result.minimal_task_set
    assert set(result.minimal_task_set) <= {d for d, _ in result.distinguishing_dims}


def test_negative_predictions_record_unchanged_dims():
    result = heterogeneity.find_convergence(_heterogeneity_fixture(), "social_communication_pragmatics")
    # 机制 B 在 social_cognition 上未变 → 应出现在阴性预测里
    assert "social_cognition" in result.negative_predictions["novelty_valuation"]


def test_convergence_reports_single_candidate_without_claiming_heterogeneity():
    profiles = [
        P_intact(),
        _profile({"ambiguity_tolerance": 0.2}, _dims(social_communication_pragmatics=0.45), theta_id="x"),
    ]
    result = heterogeneity.find_convergence(profiles, "social_communication_pragmatics")
    assert result.n_candidates == 1
    assert result.is_heterogeneous is False
    assert any("≥2" in n for n in result.notes)


def test_interactions_match_same_dose_only():
    """可加性比较只在单因素与组合**同 dose** 时进行。"""
    from pacs.ontology.primitives import DEFAULT_ACTIVE

    baseline = P_intact()
    profiles = [
        baseline,
        _profile({**{p: 1.0 for p in DEFAULT_ACTIVE}, "switching_cost": 0.2}, _dims(repetitive_restricted_behavior=0.70)),
        _profile({**{p: 1.0 for p in DEFAULT_ACTIVE}, "inhibition": 0.2}, _dims(repetitive_restricted_behavior=0.70)),
        _profile(
            {**{p: 1.0 for p in DEFAULT_ACTIVE}, "switching_cost": 0.2, "inhibition": 0.2},
            _dims(repetitive_restricted_behavior=0.50),
        ),
    ]
    results = heterogeneity.evaluate_interactions(profiles, "repetitive_restricted_behavior")
    assert len(results) == 1
    assert results[0].kind == "additive"      # 0.7+0.7-0.9 = 0.5 恰好可加
    assert results[0].interaction == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 亚型
# ---------------------------------------------------------------------------


def test_kmeans_separates_two_clear_clusters():
    vectors = [[0.0, 0.0]] * 5 + [[10.0, 10.0]] * 5
    labels, centroids = subtypes.kmeans(vectors, 2, seed=1)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]


def test_silhouette_high_for_well_separated_clusters():
    vectors = [[0.0, 0.0]] * 5 + [[10.0, 10.0]] * 5
    labels, _ = subtypes.kmeans(vectors, 2, seed=1)
    assert subtypes.silhouette(vectors, labels) > 0.8


def test_cluster_profiles_reports_mechanism_signature():
    profiles = []
    for i in range(6):
        profiles.append(
            _profile(
                {"switching_cost": 0.2 if i < 3 else 1.0, "inhibition": 1.0 if i < 3 else 0.2},
                _dims(repetitive_restricted_behavior=0.4 if i < 3 else 0.9),
                theta_id=f"theta_{i}",
            )
        )
    clones, scores = subtypes.cluster_profiles(profiles, k=2, seed=3)
    assert len(clones) == 2
    assert all(c.mechanism_signature for c in clones)
    assert all(c.impaired_primitives for c in clones)


def test_cluster_requires_minimum_samples():
    clones, scores = subtypes.cluster_profiles([_profile({}, _dims())])
    assert clones == [] and scores == {}


# ---------------------------------------------------------------------------
# 贝叶斯机制比较
# ---------------------------------------------------------------------------


def test_mechanism_posterior_prefers_matching_mechanism():
    models = [
        bayes.MechanismModel("m_a", 0.2, {d: 0.45 for d in ALL_DIMS}),
        bayes.MechanismModel("m_b", 0.2, {d: 0.90 for d in ALL_DIMS}),
    ]
    observed = {d: 0.44 for d in ALL_DIMS}       # 更贴近 m_a
    result = bayes.mechanism_posterior(observed, models, sigma=0.1)
    assert result.best == "M[m_a@0.2]"
    assert result.posterior["M[m_a@0.2]"] > 0.99


def test_mechanism_posterior_uniform_when_indistinguishable():
    models = [
        bayes.MechanismModel("m_a", 0.2, {d: 0.50 for d in ALL_DIMS}),
        bayes.MechanismModel("m_b", 0.2, {d: 0.50 for d in ALL_DIMS}),
    ]
    result = bayes.mechanism_posterior({d: 0.5 for d in ALL_DIMS}, models)
    assert result.posterior["M[m_a@0.2]"] == pytest.approx(0.5)
    assert result.entropy == pytest.approx(0.6931, abs=1e-3)


def test_mechanism_posterior_ignores_missing_dims():
    models = [bayes.MechanismModel("m_a", 0.2, {d: 0.5 for d in ALL_DIMS})]
    result = bayes.mechanism_posterior({"social_communication_pragmatics": 0.5}, models)
    assert result.posterior  # 不因缺失维度而崩


def test_build_mechanism_models_groups_by_primitive_and_dose():
    profiles = [
        _profile({"switching_cost": 0.2}, _dims(repetitive_restricted_behavior=0.4), theta_id="a"),
        _profile({"switching_cost": 0.4}, _dims(repetitive_restricted_behavior=0.7), theta_id="b"),
    ]
    models = bayes.build_mechanism_models(profiles)
    labels = {m.label for m in models}
    assert labels == {"M[switching_cost@0.2]", "M[switching_cost@0.4]"}


# ---------------------------------------------------------------------------
# 反事实与实验设计
# ---------------------------------------------------------------------------


def test_counterfactual_recovery_pattern_differs_between_mechanisms():
    result = heterogeneity.find_convergence(_heterogeneity_fixture(), "social_communication_pragmatics")
    cfs = counterfactual.build_counterfactuals(result)
    assert len(cfs) == 2
    # 两个机制的恢复模式必须不同 —— 这是机制可区分的核心证据
    assert cfs[0].recovered != cfs[1].recovered


def test_experiment_design_lists_tasks_with_expectations():
    result = heterogeneity.find_convergence(_heterogeneity_fixture(), "social_communication_pragmatics")
    design = counterfactual.build_experiment_design(result)
    assert design["n_tasks"] == len(result.minimal_task_set)
    for task in design["tasks"]:
        assert task["measured_dim"] != "social_communication_pragmatics"    # 终点不用于判别
        assert task["expected"]


# ---------------------------------------------------------------------------
# 景观
# ---------------------------------------------------------------------------


def test_landscape_skips_generic_and_multi_primitive_profiles():
    profiles = [
        P_intact(),
        _profile({"switching_cost": 0.4}, _dims()),
        _profile({"switching_cost": 0.4, "inhibition": 0.4}, _dims()),
        _profile({}, _dims(), generic=0.3),
    ]
    cells = landscape.landscape_cells(profiles)
    assert len(cells) == 1
    assert cells[0].primitive_id == "switching_cost"


def test_heatmap_render_contains_all_primitives():
    cells = landscape.landscape_cells([_profile({"switching_cost": 0.4}, _dims())])
    text = landscape.render_heatmap(cells, "repetitive_restricted_behavior")
    assert "switching_cost" in text
