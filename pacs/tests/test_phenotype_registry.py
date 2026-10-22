"""表型词表与剖面的测试（原 `test_perturb_battery.py` 的「电池与表型」部分）。

行为电池删除后，这里守的是仍然成立的那部分契约：
1. **词表结构**：6 个表型维度 + 1 个对照通道，对照通道不是表型维度、但必须进评分通道；
2. **对照通道的锚点**：它要被问到、被评分，就必须写清「看什么」；
3. **profiles 的基线聚合**：完整心智剖面取平均，以及完成率口径。

被删掉的是强制选择题那一套（题目敏感度、chance-corrected 正确率、解析失败排除分母）——
那套口径随电池一起废除，表型一律由分析者 LLM 从对话里读出（见 test_measure.py）。
"""

from __future__ import annotations

import pytest

from pacs.brain import perturb as P
from pacs.phenotype import (
    CONTROL_DIM,
    DIM_ANCHORS,
    PHENOTYPE_DIMS,
    SCORED_CHANNELS,
    PhenotypeProfile,
    aggregate_baseline,
    control_channel_value,
    all_controls_intact,
)


def test_registry_has_the_six_phenotype_dimensions_plus_control():
    assert set(PHENOTYPE_DIMS) == {
        "social_interaction_motivation",
        "social_cognition",
        "social_communication_pragmatics",
        "cognitive_flexibility_adaptation",
        "repetitive_restricted_behavior",
        "novelty_uncertainty_sensory_behavior",
    }
    # 对照通道不是表型维度（不进雷达图），但必须进评分通道（否则特异性判定没有参照）
    assert CONTROL_DIM not in PHENOTYPE_DIMS
    assert set(SCORED_CHANNELS) == set(PHENOTYPE_DIMS) | {CONTROL_DIM}


def test_every_scored_channel_has_an_observation_anchor():
    """每个要打分的通道都得写清「看什么 / 低分什么样 / 高分什么样」。

    缺锚点时分析者会按各自的理解打分 —— 通道之间的边界就不可比。
    """
    for channel in SCORED_CHANNELS:
        anchor = DIM_ANCHORS.get(channel)
        assert anchor, channel
        for key in ("watch", "low", "high"):
            assert (anchor.get(key) or "").strip(), (channel, key)


def test_observation_goals_cover_every_scored_channel():
    """观察目标 = 计分通道：漏掉对照通道，Control 1 就会永远判「无法评估」。"""
    from pacs.brain.architecture import DEFAULT_GOALS
    from pacs.agents.dialogue import DEFAULT_N_TURNS
    from pacs import QUESTIONS_PER_CHANNEL

    assert {dim for dim, _label in DEFAULT_GOALS} == set(SCORED_CHANNELS)
    # 基础轮数 = 每个通道各「每通道最低问题数」个问题，否则某些通道从头到尾没有证据
    assert DEFAULT_N_TURNS == QUESTIONS_PER_CHANNEL * len(SCORED_CHANNELS)


def _profile(**over) -> PhenotypeProfile:
    data = dict(
        theta_id="theta_intact",
        model_id="mock:t",
        run_id="r",
        theta={p: 1.0 for p in P.DEFAULT_ACTIVE},
        generic_degradation=0.0,
        dims={d: 0.7 for d in SCORED_CHANNELS},
        n_items=7,
        n_parsed=7,
    )
    data.update(over)
    return PhenotypeProfile(**data)


def test_aggregate_baseline_averages_replicates():
    """聚合基线应等于各重复的均值（并降低噪声）。"""
    profiles = [
        _profile(dims={d: value for d in SCORED_CHANNELS}, replicate=idx)
        for idx, value in enumerate((0.5, 0.7, 0.9))
    ]
    merged = aggregate_baseline(profiles)
    assert merged is not None
    assert merged.dims["social_communication_pragmatics"] == pytest.approx(0.7)
    assert merged.dims[CONTROL_DIM] == pytest.approx(0.7)
    assert merged.theta_id.startswith("baseline_mean_of_3")


def test_aggregate_baseline_skips_perturbed_profiles():
    """基线只由完整心智构成：任何被削弱的 θ 都不该混进来。"""
    intact = _profile(dims={d: 0.8 for d in SCORED_CHANNELS})
    perturbed = _profile(
        theta_id="theta_x",
        theta={**{p: 1.0 for p in P.DEFAULT_ACTIVE}, "inhibition": 0.4},
        dims={d: 0.2 for d in SCORED_CHANNELS},
    )
    merged = aggregate_baseline([intact, perturbed])
    assert merged is not None
    assert merged.dims["social_cognition"] == pytest.approx(0.8)


def test_control_channel_value_and_intactness():
    profile = _profile(dims={**{d: 0.9 for d in PHENOTYPE_DIMS}, CONTROL_DIM: 0.6})
    assert control_channel_value(profile) == pytest.approx(0.6)
    assert all_controls_intact(profile) is False       # 低于 0.75 阈值
    ok = _profile(dims={**{d: 0.9 for d in PHENOTYPE_DIMS}, CONTROL_DIM: 0.9})
    assert all_controls_intact(ok) is True
    # 缺读数时不能当成「完好」—— 那是把没测到当成测到了
    missing = _profile(dims={d: 0.9 for d in PHENOTYPE_DIMS})
    assert control_channel_value(missing) is None
    assert all_controls_intact(missing) is False


def test_parse_rate_is_conversation_completion_rate():
    """n_items 记计划轮数、n_parsed 记完成轮数 → parse_rate 即对话完成率。"""
    profile = _profile(n_items=7, n_parsed=5)
    assert profile.parse_rate == pytest.approx(5 / 7)
    assert _profile(n_items=0, n_parsed=0).parse_rate == 0.0


def test_legacy_battery_profile_still_loads():
    """删除电池之前落库的剖面仍能读出来（instrument 保留为出处标记）。"""
    legacy = {
        "theta_id": "theta_old", "model_id": "mock:old", "run_id": "r0", "age_months": 36,
        "theta": {"inhibition": 0.4}, "generic_degradation": 0.0,
        "dims": {"social_cognition": 0.31}, "accuracy": {"social_cognition": 0.44},
        "n_items": 36, "n_parsed": 36, "instrument": "battery",
    }
    profile = PhenotypeProfile.from_dict(legacy)
    assert profile.instrument == "battery"
    assert profile.dims["social_cognition"] == pytest.approx(0.31)
    assert profile.parse_rate == pytest.approx(1.0)
