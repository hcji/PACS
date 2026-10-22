"""端到端与 CLI 测试：离线（Mock）跑通全链路并核验全部交付物。"""

from __future__ import annotations

import json


from pacs.cli import main


def _run(argv: list[str]) -> int:
    return main(argv)


def test_ontology_command_prints_self_check(capsys):
    assert _run(["ontology"]) == 0
    out = capsys.readouterr().out
    assert "A 信息处理" in out
    assert "心智理论" in out
    assert "自检" in out


def test_vocab_command_prints_channels_and_probes(capsys):
    """测量口径与探查登记；打印内容里不得出现任何「机制 → 维度」的映射声明。"""
    assert _run(["vocab"]) == 0
    out = capsys.readouterr().out
    assert "计分通道" in out
    assert "general_cognition" in out             # 对照通道必须列出
    assert "观察锚点" in out
    assert "探查登记" in out
    assert "不声称维度" in out
    for token in ("Model 1", "dim_map", "覆盖率校验"):
        assert token not in out, token


def test_prompt_command_shows_perturbed_architecture(capsys):
    assert _run(["prompt", "--primitive", "ambiguity_tolerance", "--dose", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "认知架构提示词" in out
    assert "语用" in out or "认知架构" in out
    assert "0.20" in out                  # 受损档位可见
    assert "BRAIN_FINGERPRINT" in out


def test_init_brain_works_offline_with_mock(tmp_path, capsys):
    assert _run(["--base-dir", str(tmp_path), "init-brain", "--provider", "mock"]) == 0
    out = capsys.readouterr().out
    assert "心智的自我报告" in out
    assert "SELF_STATE: OK" in out


def test_sample_reports_scale_without_calling_models(tmp_path, capsys):
    assert _run(["--base-dir", str(tmp_path), "sample", "--mode", "single"]) == 0
    out = capsys.readouterr().out
    assert "被试数" in out
    assert "预计模型调用数" in out


def test_full_demo_pipeline_produces_all_deliverables(tmp_path, capsys):
    """核心端到端测试：无网络跑通 run → analyze → export → report → freeze。"""
    assert (
        _run(
            [
                "--base-dir",
                str(tmp_path),
                "demo",
                "--turns",
                "4",
                "--max-pairs",
                "20",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "运行完成" in out
    assert "交付物" in out
    assert "机制异质性摘要" in out

    exports = tmp_path / "data" / "exports"
    assert (exports / "atlas.md").exists()
    assert (exports / "atlas.json").exists()
    assert (tmp_path / "data" / "pacs.db").exists()
    # 导出功能已整体删除：不再产出任何描述集或相关矩阵
    assert not (exports / "cse_descriptions.jsonl").exists()
    assert not (exports / "correlation.json").exists()

    # Atlas JSON 九章齐备
    atlas = json.loads((exports / "atlas.json").read_text(encoding="utf-8"))
    for key in (
        "dose_response",
        "specificity",
        "generic_degradation",
        "landscape",
        "convergence",
        "interactions",
        "counterfactuals",
        "experiment_design",
        "subtypes",
        "cross_model_convergence",
        "model_profiles",
        "mechanism_profile",           # 由测量估计的机制—维度剖面（对应关系的唯一来源）
        "dominant_channel_distribution",
    ):
        assert key in atlas, key
    assert atlas["mechanism_profile"]["rows"]
    assert any("不得回写" in n for n in atlas["mechanism_profile"]["notes"])
    # 对照通道由分析者从同一段对话里评分（观察目标里有「一般认知控制」）→ Control 1 判得了
    verdict = atlas["generic_degradation"]
    assert verdict is not None, "对照通道没有读数时 Control 1 应显式说明原因，而不是静默缺席"
    assert verdict["control_value"] is not None
    assert atlas["notes"] == [] or not any("Control 1（通用退化对照）无法评估" in n for n in atlas["notes"])


def test_demo_atlas_markdown_has_all_nine_sections(tmp_path):
    _run(["--base-dir", str(tmp_path), "demo", "--turns", "3", "--max-pairs", "15"])
    md = (tmp_path / "data" / "exports" / "atlas.md").read_text(encoding="utf-8")
    for heading in (
        "## ① 认知基本单元本体",
        "## ② 构造效度",
        "## ③ 表型景观",
        "## ④ 机制异质性",
        "## ⑤ 计算亚型",
        "## ⑥ 四个关键 controls",
        "## ⑦ 跨模型收敛矩阵",
        "## ⑧ 实测表型出口与维度口径",
        "## ⑨ 人类实验设计建议与冻结",
    ):
        assert heading in md, heading
    assert "不用于医学诊断" in md
    assert "Model 1" not in md


def test_demo_freeze_snapshot_is_verifiable(tmp_path):
    _run(["--base-dir", str(tmp_path), "demo", "--turns", "3", "--max-pairs", "10"])
    exports = tmp_path / "data" / "exports"
    freeze_dirs = [p for p in exports.iterdir() if p.name.startswith("freeze_")]
    assert len(freeze_dirs) == 1
    payload = json.loads((freeze_dirs[0] / "snapshot.json").read_text(encoding="utf-8"))
    assert set(payload["digests"]) == {
        "primitive_definitions",
        "perturbation_parameters",
        "phenotype_vocabulary",
        "candidate_mechanisms",
        "human_predictions",
    }

    from pacs.freeze import FreezeSnapshot, verify_freeze

    snapshot = FreezeSnapshot(
        freeze_id=payload["freeze_id"],
        created_at=payload["created_at"],
        digests=payload["digests"],
        objects=payload["objects"],
        experiment_design=payload["experiment_design"],
    )
    assert verify_freeze(snapshot) == []


def test_runs_command_lists_completed_run(tmp_path, capsys):
    _run(["--base-dir", str(tmp_path), "demo", "--turns", "3", "--max-pairs", "5"])
    capsys.readouterr()
    assert _run(["--base-dir", str(tmp_path), "runs"]) == 0
    out = capsys.readouterr().out
    assert "demo_" in out
    assert "phenotypes" in out


def test_analyze_command_runs_offline(tmp_path, capsys):
    _run(["--base-dir", str(tmp_path), "demo", "--turns", "3", "--max-pairs", "10"])
    capsys.readouterr()
    assert (
        _run(
            [
                "--base-dir",
                str(tmp_path),
                "analyze",
                "--endpoint",
                "social_communication_pragmatics",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "扰动特异性" in out
    assert "机制异质性" in out


def test_analyze_without_data_fails_cleanly(tmp_path, capsys):
    assert _run(["--base-dir", str(tmp_path), "analyze"]) == 1
    assert "库里没有" in capsys.readouterr().out


def test_real_provider_requires_confirmation(tmp_path, capsys):
    """真实 provider 下大规模调用必须显式 --yes，避免误花额度。"""
    # 没有 Key 时 provider 构造就会失败 → 返回 2；这里验证的是守卫逻辑不静默放行
    code = _run(
        [
            "--base-dir",
            str(tmp_path),
            "run",
            "--provider",
            "openai",
            "--api-key",
            "",
            "--mode",
            "single",
        ]
    )
    assert code in (1, 2)

def test_bank_command_lists_builtin_bank(capsys):
    """内置条目库要能直接从命令行浏览（含探查语句与出处）。"""
    assert _run(["bank", "--probe-ready", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "内置条目库" in out
    assert "来源 AsdKB" in out
    assert "量表      : 20 套" in out
    assert "可探查条目" in out and "情境式" in out
    assert "出处：" in out


def test_bank_command_filters_and_shows_rules(capsys):
    assert _run(["bank", "--scale", "tools19", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "筛选后 65 条" in out
    assert _run(["bank", "--age", "36", "--admin", "clinician_rating", "--limit", "2"]) == 0
    assert "筛选后" in capsys.readouterr().out
    assert _run(["bank", "--rules"]) == 0
    rules = capsys.readouterr().out
    assert "映射规则" in rules
    assert "维度判定优先级" in rules
    assert "tools19 →" in rules


def test_bank_command_prints_reference_layers(capsys):
    """AsdKB 的三个参考层（症状 / 诊断标准 / 循证干预）也要能打印。"""
    assert _run(["bank", "--layer", "symptoms"]) == 0
    symptoms = capsys.readouterr().out
    assert "症状层（65 条）" in symptoms
    assert "量表级覆盖" in symptoms

    assert _run(["bank", "--layer", "criteria"]) == 0
    criteria = capsys.readouterr().out
    assert "诊断标准条目（43 条）" in criteria
    assert "受限重复行为域" in criteria

    assert _run(["bank", "--layer", "treatments"]) == 0
    treatments = capsys.readouterr().out
    assert "循证干预（28 条" in treatments


def test_dialogue_command_uses_builtin_bank(capsys, tmp_path):
    """CLI 对话探针也应走内置条目库，并把用过的条目与出处打出来。"""
    assert _run(["--base-dir", str(tmp_path), "dialogue", "--turns", "2"]) == 0
    out = capsys.readouterr().out
    assert "表型观察 · 对话探针" in out
    assert "盲法审计" in out
    assert "已入库" in out
