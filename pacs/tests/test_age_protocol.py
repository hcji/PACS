"""心智年龄固定 36 个月：任何入口都不允许改它。

为什么要用测试把这件事钉死：心智年龄是「与同龄典型发展无异」这条参照线的**定义**。
只要它还能从 UI / CLI / API / config 传进来，两批读数就可能是在两条参照线上量的，
而剖面里看不出差别 —— 那是最难发现的一类不可比。这里守四件事：

1. 常量唯一来源 `pacs.DEFAULT_AGE_MONTHS`，且为 36；
2. **代码链路上没有 age 参数**：`run_dialogue` / `measure_phenotype` / `VirtualAgent` /
   提问者提示词构造器都不接受 `age_months`；
3. 配置文件里的月龄被忽略（旧 config.json 残留不会污染运行）；
4. 提示词与剖面记录都写死 36（剖面字段保留只是为了旧数据可读）。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import pacs
from pacs import DEFAULT_AGE_MONTHS
from pacs.agents.dialogue import analyze_dialogue, run_dialogue
from pacs.agents.measure import measure_phenotype
from pacs.agents.virtual_agent import VirtualAgent
from pacs.brain.architecture import (
    build_analyst_prompt,
    build_analyst_system_prompt,
    build_brain_prompt,
    build_brain_prompt_steps,
    build_examiner_system_prompt,
    build_examiner_turn_prompt,
)
from pacs.brain.perturb import intact
from pacs.config import Settings
from pacs.llm.provider import MockProvider
from pacs.phenotype import PhenotypeProfile


def test_age_constant_is_36_and_zero_entry_points():
    """常量是唯一口径，且被登记为公开常量（下游要读它，而不是自己写 36）。"""
    assert DEFAULT_AGE_MONTHS == 36
    assert "DEFAULT_AGE_MONTHS" in pacs.__all__
    # 源码里不该再出现「把 36 当默认值写字面上」的入口：核心链路的签名里没有 age 参数
    assert "age_months" not in inspect.signature(measure_phenotype).parameters


@pytest.mark.parametrize("func", [
    run_dialogue,
    measure_phenotype,
    VirtualAgent.__init__,
    build_examiner_system_prompt,
    build_examiner_turn_prompt,
    build_brain_prompt,
    build_brain_prompt_steps,
])
def test_no_age_parameter_anywhere_in_the_chain(func):
    """年龄不是参数：任何一环接受 age_months 都会重新打开这条漂移路径。"""
    params = inspect.signature(func).parameters
    assert "age_months" not in params, f"{func} 又接受 age_months 了"


def test_config_ignores_age_from_file_and_env(tmp_path, monkeypatch):
    """旧 config.json 里的观察段（含月龄）一律忽略：改了文件也不会改变运行口径。"""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "llm": {"provider": "mock"},
        "observation": {"age_months": 84},
        "battery": {"age_months": 18},       # 更旧的段名，同样忽略
    }, ensure_ascii=False), encoding="utf-8")
    settings = Settings.load(cfg, base_dir=tmp_path)
    assert not hasattr(settings, "age_months")

    # 保存也不写回这一项（否则配置里会一直留着一个看起来能改的旋钮）
    path = settings.save()
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "observation" not in saved
    assert "age_months" not in json.dumps(saved, ensure_ascii=False)


def test_prompts_state_the_fixed_age():
    examiner = build_examiner_system_prompt(turn_budget=7)
    assert f"约 {DEFAULT_AGE_MONTHS} 个月大的孩子" in examiner
    # 分析者看不到任何「年龄」以外的内部设置，但参照线必须是同一个数
    analyst = build_analyst_system_prompt()
    assert "同龄典型" in analyst


def test_brain_prompt_and_steps_stay_in_sync_about_age():
    """心智侧提示词不写月龄（它只写行为准则），但两侧的字数口径必须一致。"""
    steps = build_brain_prompt_steps(intact())
    assert "\n\n".join(s["text"] for s in steps) == build_brain_prompt(intact())


def test_measure_writes_the_constant_into_the_profile():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=3)
    result = measure_phenotype(
        brain, MockProvider(), MockProvider(), n_turns=7, max_turns=7, seed=3,
    )
    assert result.profile.age_months == DEFAULT_AGE_MONTHS
    assert result.session.age_months == DEFAULT_AGE_MONTHS


def test_legacy_profile_rows_still_load_their_own_age():
    """旧数据可读：字段保留，历史值照原样出来（不被常量覆盖）。"""
    old = PhenotypeProfile(
        theta_id="t", model_id="mock:m", run_id="r",
        theta={}, generic_degradation=0.0, dims={}, n_items=1, n_parsed=1,
        age_months=48,
    )
    restored = PhenotypeProfile.from_dict(old.to_dict())
    assert restored.age_months == 48
    # 不传就用常量（新写入一律 36）
    fresh = PhenotypeProfile(
        theta_id="t", model_id="mock:m", run_id="r",
        theta={}, generic_degradation=0.0, dims={}, n_items=1, n_parsed=1,
    )
    assert fresh.age_months == DEFAULT_AGE_MONTHS


def test_analyst_prompt_and_session_round_trip_without_age_drift():
    brain = VirtualAgent(MockProvider(), intact(), model_id="mock:m", seed=5)
    session = run_dialogue(brain, MockProvider(), n_turns=7, max_turns=7, seed=5)
    assert analyze_dialogue(MockProvider(), session)["error"] == ""
    assert "age_months" in session.to_dict()
    # 即使把旧记录里的月龄改成别的值，反序列化也一律回到常量
    payload = session.to_dict()
    payload["age_months"] = 90
    from pacs.agents.dialogue import DialogueSession

    assert DialogueSession.from_dict(payload).age_months == DEFAULT_AGE_MONTHS
