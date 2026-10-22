"""Atlas 报告（FR-18）：把全部分析组装成九章的结构化报告（Markdown + JSON）。

九章固定顺序（PRD FR-18）：① 本体 ② 构造效度 ③ 表型景观 ④ 机制异质性 ⑤ 计算亚型
⑥ 四个 controls ⑦ 跨模型收敛矩阵 ⑧ 实测表型出口与维度口径 ⑨ 人类实验建议与冻结 id。

**⑧ 章是实测口径说明**：计分通道清单与对照通道的地位、本 Atlas 不内建
任何「机制 → 维度」映射的显式声明、以及由测量估计出的机制—维度剖面在 ②.4 的指引。

**合成数据告警**：若入参剖面来自离线 mock（`PhenotypeProfile.synthetic`），本报告会输出
显著免责行 —— mock 的表型结构是规则构造的，不是 LLM 涌现的，不能作为科学结论。
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

from .analysis import (
    counterfactual,
    dose_response,
    heterogeneity,
    landscape,
    mechanism_profile,
    specificity,
    subtypes,
)
from .analysis.heterogeneity import ConvergenceResult
from .phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS, PhenotypeProfile, synthetic_count, uncovered_counts
from .llm.registry import ModelProfile
from .ontology.assays import primitives_without_assay
from .ontology.primitives import CANDIDATES, FAMILIES, PRIMITIVES
from .ontology.systems import SYSTEMS

DEFAULT_ENDPOINT = "social_communication_pragmatics"
CONVERGENCE_THRESHOLD = 0.70    # chance-corrected 尺度下的「受损」阈值（完整心智 ≈ 0.92）
CONVERGENCE_GATE = 0.75         # 跨模型收敛判据：效应方向一致的比例

#: 混入离线 mock 剖面时的免责行（写进 notes 与 Markdown 顶部）
SYNTHETIC_DISCLAIMER = (
    "⚠ 本次判定基于**离线 mock（规则式替身）**的剖面：其表型结构是构造出来的、"
    "不是 LLM 的涌现行为，**不构成涌现证据**。真实结论必须用真实 provider 重跑。"
)


@dataclass
class Atlas:
    """Atlas 的全部内容（供 Markdown 与 JSON 两种输出）。"""

    model_version: str
    created_at: str
    endpoint_dim: str
    n_profiles: int
    model_profiles: list[ModelProfile]
    model_ids: list[str]
    dose_fits: list[dose_response.DoseResponseFit]
    specificity_verdicts: list[specificity.SpecificityVerdict]
    generic_verdict: specificity.GenericDegradationVerdict | None
    cells: list[landscape.LandscapeCell]
    convergence: ConvergenceResult | None
    interactions: list[heterogeneity.InteractionResult]
    counterfactuals: list[counterfactual.CounterfactualPrediction]
    experiment_design: dict
    subtypes: list[subtypes.Subtype]
    subtype_scores: dict[int, float]
    convergence_matrix: dict[str, dict[str, str]]
    ontology_problems: list[str]
    assay_problems: list[str]
    mechanism_profile: mechanism_profile.MechanismProfileReport | None = None
    n_synthetic_profiles: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def has_synthetic(self) -> bool:
        return self.n_synthetic_profiles > 0

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "created_at": self.created_at,
            "endpoint_dim": self.endpoint_dim,
            "n_profiles": self.n_profiles,
            "model_profiles": [p.to_dict() for p in self.model_profiles],
            "model_ids": self.model_ids,
            "ontology_problems": self.ontology_problems,
            "assay_problems": self.assay_problems,
            "n_synthetic_profiles": self.n_synthetic_profiles,
            "dose_response": [f.to_dict() for f in self.dose_fits],
            "specificity": [v.to_dict() for v in self.specificity_verdicts],
            "specificity_summary": specificity.summarize(self.specificity_verdicts),
            "dominant_channel_distribution": specificity.dominant_distribution(
                self.specificity_verdicts
            ),
            "generic_degradation": self.generic_verdict.__dict__ if self.generic_verdict else None,
            "landscape": [c.to_dict() for c in self.cells],
            "convergence": self.convergence.to_dict() if self.convergence else None,
            "interactions": [i.__dict__ for i in self.interactions],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "experiment_design": self.experiment_design,
            "subtypes": [s.to_dict() for s in self.subtypes],
            "subtype_scores": {str(k): v for k, v in self.subtype_scores.items()},
            "cross_model_convergence": self.convergence_matrix,
            "mechanism_profile": self.mechanism_profile.to_dict() if self.mechanism_profile else None,
            "notes": self.notes,
        }


def cross_model_convergence(
    profiles: list[PhenotypeProfile],
    endpoint_dim: str,
    *,
    dose: float | None = None,
    drop_threshold: float = 0.15,
) -> dict[str, dict[str, str]]:
    """跨模型收敛矩阵（FR-10 要求 2）：primitive × 模型 → 方向 + 收敛判定。"""
    models = sorted({p.model_id for p in profiles})
    by_model = {m: [p for p in profiles if p.model_id == m] for m in models}
    primitives = sorted(
        {
            next(iter({k: v for k, v in p.theta.items() if v < 0.999}))
            for p in profiles
            if len([k for k, v in p.theta.items() if v < 0.999]) == 1 and p.generic_degradation <= 0
        }
    )
    matrix: dict[str, dict[str, str]] = {}
    for pid in primitives:
        row: dict[str, str] = {}
        impaired_votes = 0
        evaluated = 0
        for model_id in models:
            group = [
                p
                for p in by_model[model_id]
                if p.generic_degradation <= 0
                and {k: v for k, v in p.theta.items() if v < 0.999}.keys() == {pid}
            ]
            baseline = next(
                (
                    p
                    for p in by_model[model_id]
                    if p.generic_degradation <= 0 and not any(v < 0.999 for v in p.theta.values())
                ),
                None,
            )
            if not group or baseline is None:
                row[model_id] = "n/a"
                continue
            probe = min(group, key=lambda p: p.active_levels()[0])
            base_value = baseline.dims.get(endpoint_dim)
            value = probe.dims.get(endpoint_dim)
            if base_value is None or value is None:
                row[model_id] = "n/a"
                continue
            delta = float(value) - float(base_value)
            evaluated += 1
            if delta <= -drop_threshold:
                impaired_votes += 1
                row[model_id] = f"↓({delta:+.2f})"
            elif delta >= drop_threshold:
                row[model_id] = f"↑({delta:+.2f})"
            else:
                row[model_id] = f"→({delta:+.2f})"
        if evaluated == 0:
            row["_verdict"] = "n/a"
        else:
            ratio = impaired_votes / evaluated
            row["_verdict"] = (
                "convergent" if ratio >= CONVERGENCE_GATE else ("partial" if ratio > 0 else "divergent")
            )
        matrix[pid] = row
    return matrix


def build_atlas(
    profiles: list[PhenotypeProfile],
    *,
    model_version: str = "pacs-v0.1",
    endpoint_dim: str = DEFAULT_ENDPOINT,
    model_profiles: list[ModelProfile] | None = None,
    threshold: float = CONVERGENCE_THRESHOLD,
    dose: float | None = None,
) -> Atlas:
    """跑完全部分析，组装 Atlas。"""
    fits = dose_response.build_dose_response(profiles)
    verdicts = specificity.evaluate_specificity(profiles)
    generic = specificity.evaluate_generic_degradation(profiles)
    cells = landscape.landscape_cells(profiles)
    convergence = heterogeneity.find_convergence(
        profiles, endpoint_dim, threshold=threshold, dose=dose
    )
    interactions = heterogeneity.evaluate_interactions(profiles, endpoint_dim, dose=dose)
    counterfacts = counterfactual.build_counterfactuals(convergence) if convergence else []
    design = counterfactual.build_experiment_design(convergence) if convergence else {}
    clones, scores = subtypes.cluster_profiles(
        profiles, baseline_dims=(convergence.baseline_dims if convergence else None)
    )
    matrix = cross_model_convergence(profiles, endpoint_dim, dose=dose)
    mech_profile = mechanism_profile.estimate_mechanism_profile(profiles)

    from .ontology.assays import validate_assays
    from .ontology.primitives import validate_ontology

    notes: list[str] = []
    if synthetic_count(profiles):
        notes.append(SYNTHETIC_DISCLAIMER)
    if primitives_without_assay():
        notes.append(
            "本体缺口：以下 primitive 未派生 assay（观察者条目库未覆盖）→ "
            + "、".join(primitives_without_assay())
        )
    missing_channels = uncovered_counts(profiles)
    if missing_channels:
        # 覆盖偏斜必须显式暴露：没有证据的通道读数缺失，不能被读成「该维度在扰动下不变」。
        worst = sorted(missing_channels.items(), key=lambda kv: -kv[1])
        notes.append(
            "有通道在部分会话中没有被探查到（没有证据支撑，读数记为空）："
            + "、".join(f"{SCORED_CHANNELS.get(ch, ch)}×{n}" for ch, n in worst)
            + "。这些通道不参与对应维度的比较，解释时需注意。"
        )
    if any(p.parse_rate < 0.9 for p in profiles):
        worst = min(profiles, key=lambda p: p.parse_rate)
        notes.append(
            f"存在对话完成率偏低（最低 {worst.parse_rate:.2f}，theta={worst.theta_id}）的被试 —— "
            "该被试的读数只覆盖了部分轮次，证据量低于其余被试，解释时需注意。"
        )
    if not matrix:
        notes.append("跨模型收敛矩阵为空：需对 ≥2 个模型跑同一设计才可评估（FR-10）。")
    if generic is None and profiles:
        # 对照通道（general_cognition）不是表型维度，但分析者会为它单独评分。
        # 不静默：这里显式说明为什么 Control 1 判不了、以及怎么才能判。
        instruments = sorted({p.instrument for p in profiles})
        notes.append(
            "Control 1（通用退化对照）无法评估：本批剖面里没有对照通道的读数。"
            f"测量工具 = {'、'.join(instruments)}；对照通道由分析者 LLM 从同一段对话里"
            "单独评分（观察目标里的「一般认知控制」）。缺少这一通道的剖面"
            "无法评估该对照 —— 如需该对照，请对这批 θ 重新观察一次。"
        )

    return Atlas(
        model_version=model_version,
        created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        endpoint_dim=endpoint_dim,
        n_profiles=sum(1 for p in profiles if p.generic_degradation <= 0),
        model_profiles=list(model_profiles or []),
        model_ids=sorted({p.model_id for p in profiles}),
        dose_fits=fits,
        specificity_verdicts=verdicts,
        generic_verdict=generic,
        cells=cells,
        convergence=convergence,
        interactions=interactions,
        counterfactuals=counterfacts,
        experiment_design=design,
        subtypes=clones,
        subtype_scores=scores,
        convergence_matrix=matrix,
        ontology_problems=validate_ontology(),
        assay_problems=validate_assays(),
        mechanism_profile=mech_profile,
        n_synthetic_profiles=synthetic_count(profiles),
        notes=notes,
    )


def render_atlas(atlas: Atlas) -> str:
    """渲染 Markdown 报告（九章）。"""
    lines: list[str] = []
    lines.append("# 认知扰动图谱（Cognitive Perturbation Atlas）")
    lines.append("")
    lines.append(f"> 模型版本：{atlas.model_version} ｜ 生成时间：{atlas.created_at}")
    lines.append(f"> 被试数（θ 数）：{atlas.n_profiles} ｜ 模型：{', '.join(atlas.model_ids) or '无'}")
    lines.append(f"> 表型终点：{atlas.endpoint_dim}")
    lines.append("")
    lines.append(
        "> **用途声明**：本报告为方法学研究材料，**不用于医学诊断、不替代专业评估**。"
        "仿真提供先验与机制假设，不提供真理。"
    )
    lines.append("")
    if atlas.has_synthetic:
        lines.append(f"> {SYNTHETIC_DISCLAIMER}")
        lines.append("")

    # ---------- ① 本体 ----------
    lines.append("## ① 认知基本单元本体（Level 0）")
    lines.append("")
    lines.append(f"共 {len(PRIMITIVES)} 个 core primitive，分布 {len(FAMILIES)} 个家族；候补 {len(CANDIDATES)} 个。")
    lines.append("")
    lines.append("| 家族 | primitive | 中文 | 定义 |")
    lines.append("|---|---|---|---|")
    for p in PRIMITIVES:
        lines.append(f"| {FAMILIES.get(p.family, p.family)} | `{p.id}` | {p.name_zh} | {p.definition} |")
    lines.append("")
    lines.append("**Level 1 高阶认知（组合涌现，不直接扰动）**")
    lines.append("")
    lines.append("| 构念 | 成员 primitive | 可证伪预期 |")
    lines.append("|---|---|---|")
    for s in SYSTEMS:
        lines.append(
            f"| {s.name_zh} | {' + '.join(f'`{m}`' for m in s.members)} | {s.falsifiable_expectation} |"
        )
    lines.append("")
    lines.append("**五项标准判定（FR-1）**")
    lines.append("")
    missing = primitives_without_assay()
    if atlas.ontology_problems or missing:
        lines.append("- 本体自检问题：" + ("；".join(atlas.ontology_problems) or "无"))
        if missing:
            lines.append("- 未派生 assay 的 primitive：" + "、".join(missing))
    else:
        lines.append(f"- 全部 {len(PRIMITIVES)} 个 primitive 通过本体自检，且均派生独立 assay（含特异性对照）。")
    lines.append("")

    # ---------- ② 构造效度 ----------
    lines.append("## ② 构造效度（FR-9）")
    lines.append("")
    lines.append("### 2.1 扰动特异性")
    lines.append("")
    lines.append(
        "主导维度**由数据给出**（在 6 个表型维度里取下降最多者，对照通道只用于稳定性判据），"
        "不是事先指定的目标维度。集中度 = 次强维度 Δ − 主导维度 Δ，门槛取 "
        "`max(0.05, 2·SE)`（SE 由同 dose 的重复测量估计）。"
    )
    lines.append("")
    if atlas.specificity_verdicts:
        lines.append("| primitive | dose | 主导维度（实测） | Δ主导 | 集中度 | Δ对照 | n | 判定 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for v in atlas.specificity_verdicts:
            lines.append(
                f"| `{v.primitive_id}` | {v.dose:.2f} | {v.dominant_dim} | {v.dominant_delta:+.2f} | "
                f"{v.concentration:+.2f} | {v.control_delta:+.2f} | {v.n_replicates} | **{v.verdict}** |"
            )
        lines.append("")
        counts = specificity.summarize(atlas.specificity_verdicts)
        lines.append("判定汇总：" + "、".join(f"{k}={v}" for k, v in sorted(counts.items())))
        lines.append("")
        dist = specificity.dominant_distribution(atlas.specificity_verdicts)
        lines.append(
            "主导维度分布（证明没有哪个维度天生不被检验）："
            + ("、".join(f"{k}={v}" for k, v in sorted(dist.items())) or "—")
        )
        lines.append("")
        lines.append("```")
        lines.append(specificity.render_table(atlas.specificity_verdicts))
        lines.append("```")
    else:
        lines.append("无单因素扰动剖面，无法评估特异性。")
    lines.append("")

    lines.append("### 2.2 剂量-反应")
    lines.append("")
    lines.append(
        "逐通道拟合：`dominant_dim` 是拟合可用（r² ≥ 0.2）的通道里 |slope| 最大者；"
        "判定只看该通道自身的数据自洽性（点数、方向、r²、单调性），不做机制与维度的预先匹配。"
    )
    lines.append("")
    if atlas.dose_fits:
        lines.append("| primitive | 主导维度（实测） | slope | r² | 单调 | 拟合通道数 | 判定 |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in atlas.dose_fits:
            lines.append(
                f"| `{f.primitive_id}` | {f.dominant_dim or '—'} | {f.slope:+.3f} | {f.r2:.2f} | "
                f"{f.direction} | {f.n_dims_fitted} | **{f.verdict}** |"
            )
        counts = dose_response.summarize(atlas.dose_fits)
        lines.append("")
        lines.append("判定汇总：" + "、".join(f"{k}={v}" for k, v in sorted(counts.items())))
    else:
        lines.append("无足够数据点拟合剂量-反应曲线。")
    lines.append("")

    lines.append("### 2.3 基线验证")
    lines.append("")
    lines.append(
        "完整心智（无扰动）的表型作为全部对比的基线（多次取平均以降低单次抽样的噪声）。"
    )
    lines.append("")

    lines.append("### 2.4 由测量估计的机制—维度剖面")
    lines.append("")
    lines.append(
        "**这是「机制 → 维度」对应关系的唯一来源，而它是测量产物**："
        "逐格 = 该机制单扰动时相对基线的平均 Δ。它不参与任何判据、不进冻结快照、"
        "**禁止回写**为后续运行的先验。"
    )
    lines.append("")
    if atlas.mechanism_profile and atlas.mechanism_profile.rows:
        lines.append("```")
        lines.append(mechanism_profile.render_table(atlas.mechanism_profile))
        lines.append("```")
    else:
        lines.append("无单因素扰动剖面，无法估计机制—维度剖面。")
    lines.append("")

    # ---------- ③ 表型景观 ----------
    lines.append("## ③ 表型景观（FR-12.1）")
    lines.append("")
    if atlas.cells:
        lines.append("```")
        lines.append(landscape.render_all_dimensions(atlas.cells))
        lines.append("```")
    else:
        lines.append("无单因素扰动数据。")
    lines.append("")

    # ---------- ④ 机制异质性 ----------
    lines.append("## ④ 机制异质性（FR-12.2~12.5）")
    lines.append("")
    if atlas.convergence:
        lines.append("```")
        lines.append(heterogeneity.render_convergence(atlas.convergence))
        lines.append("```")
        lines.append("")
        if atlas.interactions:
            lines.append("### 4.1 组合扰动交互效应")
            lines.append("")
            lines.append("| 机制对 | dose | 预期（可加） | 实测 | 交互 | 类型 |")
            lines.append("|---|---|---|---|---|---|")
            for i in atlas.interactions[:14]:
                lines.append(
                    f"| `{i.primitive_a}`+`{i.primitive_b}` | {i.dose:.2f} | {i.expected_additive:.2f} | "
                    f"{i.combined:.2f} | {i.interaction:+.3f} | **{i.kind}** |"
                )
            lines.append("")
    else:
        lines.append("无单因素扰动数据，无法评估机制异质性。")
    lines.append("")

    # ---------- ⑤ 计算亚型 ----------
    lines.append("## ⑤ 计算亚型（FR-13）")
    lines.append("")
    lines.append("```")
    lines.append(subtypes.render_subtypes(atlas.subtypes, atlas.subtype_scores))
    lines.append("```")
    lines.append("")

    # ---------- ⑥ 四个 controls ----------
    lines.append("## ⑥ 四个关键 controls（FR-11）")
    lines.append("")
    lines.append("### Control 1 · Generic degradation（随机通用退化）")
    lines.append("")
    if atlas.generic_verdict:
        g = atlas.generic_verdict
        lines.append(
            f"- 退化水平 {g.level:.2f}；对照通道 = {g.control_value:.2f}；其它维度均值 = {g.mean_other_value:.2f}"
        )
        lines.append(f"- 判定：{'**确认非特异整体下降**' if g.is_nonspecific else '对照通道仍相对保留'}")
        lines.append(f"- 说明：{g.reason}")
    else:
        lines.append("未运行通用退化对照。**该 control 缺失，Atlas 视为不完整。**")
    lines.append("")
    lines.append("### Control 2 · Alternative perturbation（同一表型可由不同 primitive 产生）")
    lines.append("")
    if atlas.convergence and atlas.convergence.n_candidates >= 2:
        lines.append(
            f"- 表型终点「{atlas.endpoint_dim}」下发现 {atlas.convergence.n_candidates} 个候选机制："
            + "、".join(f"`{c.primitive_id}`" for c in atlas.convergence.candidates)
        )
        lines.append(
            f"- 判定：{'**支持机制异质性**' if atlas.convergence.is_heterogeneous else '候选机制当前不可区分'}"
        )
    else:
        lines.append("未发现 ≥2 个候选机制指向同一表型终点 → 当前证据不支持机制异质性。")
    lines.append("")
    lines.append("### Control 3 · LLM architecture（跨模型）")
    lines.append("")
    if atlas.convergence_matrix:
        verdicts: dict[str, int] = {}
        for row in atlas.convergence_matrix.values():
            verdicts[row.get("_verdict", "n/a")] = verdicts.get(row.get("_verdict", "n/a"), 0) + 1
        lines.append("- 判定汇总：" + "、".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
        lines.append(f"- 模型数：{len(atlas.model_ids)} → {'满足 FR-10 的 ≥3' if len(atlas.model_ids) >= 3 else '**不足 3 个**'}")
    else:
        lines.append("单模型或数据不足，无法评估跨模型收敛。**该 control 缺失。**")
    lines.append("")
    lines.append("### Control 4 · Human-independent validation")
    lines.append("")
    lines.append(
        "- 本 Atlas 为 Discovery 阶段产物。**人类独立验证不在本范围**（人类里程碑执行）。"
    )
    lines.append(
        "- 要求：模型冻结后，用**完全未参与模型建立**的数据验证事先产生的预测（含阴性预测）。"
    )
    lines.append("")

    # ---------- ⑦ 跨模型收敛矩阵 ----------
    lines.append("## ⑦ 跨模型收敛矩阵（FR-10）")
    lines.append("")
    if atlas.convergence_matrix:
        model_ids = atlas.model_ids
        lines.append("| primitive | " + " | ".join(model_ids) + " | 判定 |")
        lines.append("|" + "---|" * (len(model_ids) + 2))
        for pid, row in sorted(atlas.convergence_matrix.items()):
            cells = " | ".join(str(row.get(m, "n/a")) for m in model_ids)
            lines.append(f"| `{pid}` | {cells} | **{row.get('_verdict', 'n/a')}** |")
    else:
        lines.append("无跨模型数据。")
    lines.append("")
    if atlas.model_profiles:
        lines.append("**模型档案**")
        lines.append("")
        lines.append("| model_id | 家族 | 架构 | 训练范式 |")
        lines.append("|---|---|---|---|")
        for p in atlas.model_profiles:
            lines.append(f"| {p.model_id} | {p.family} | {p.architecture} | {p.training_regime} |")
        lines.append("")

    # ---------- ⑧ 实测表型出口与维度口径 ----------
    lines.append("## ⑧ 实测表型出口与维度口径（不含预设映射）")
    lines.append("")
    lines.append("**本 Atlas 不内建任何「构造属性 → 测量维度」的映射。** primitive（θ 的构造属性）")
    lines.append("与表型维度之间没有预设对应：维度读数一律由分析者 LLM 从「提问者提问 → 心智作答」")
    lines.append("的对话得出，判定则由数据本身给出（②.1 的主导维度、②.4 的估计剖面）。")
    lines.append("")
    lines.append("计分通道（表型向量 `B` 的键，顺序固定）：")
    lines.append("")
    lines.append("| 通道 id | 标签 | 地位 |")
    lines.append("|---|---|---|")
    for ch, label in SCORED_CHANNELS.items():
        lines.append(
            f"| `{ch}` | {label} | "
            + ("表型维度（进雷达图）" if ch in PHENOTYPE_DIMS else "**对照通道**（不进雷达图，进评分与特异性判据）")
            + " |"
        )
    lines.append("")
    lines.append(
        "- 对照通道是阴性预测（FR-12）与特异性的测量基础：判别机制时**必须排除它**，"
        "否则会用噪声区分机制。"
    )
    lines.append(
        "- 通道**清单**是协议给定的固定词表；通道**读数**只来自分析者。"
        "没有证据的通道记为空读数并在备注里列出，不会被当成「表现典型」。"
    )
    lines.append(
        "- 对外（如与外部模型对齐）需下游自行做维度映射 —— 本项目不再产出任何维度投影或描述集导出。"
    )
    lines.append("")

    # ---------- ⑨ 人类实验建议与冻结 ----------
    lines.append("## ⑨ 人类实验设计建议与冻结（FR-14、FR-15）")
    lines.append("")
    if atlas.experiment_design:
        lines.append("```")
        lines.append(counterfactual.render_experiment_design(atlas.experiment_design, atlas.counterfactuals))
        lines.append("```")
    else:
        lines.append("无候选机制，未产出实验设计建议。")
    lines.append("")
    lines.append("**冻结协议**：冻结对象为 ① primitive 定义 ② 扰动参数 ③ 表型词表与探查登记 ④ 候选机制 ⑤ 人类预测（含阴性预测）。")
    lines.append("冻结后任何修改都会产生新的 freeze_id，旧预测保持不变；验证时须同时检验阴性预测。")
    lines.append("")

    # ---------- 备注 ----------
    if atlas.notes:
        lines.append("## 备注与数据质量")
        lines.append("")
        for note in atlas.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "**用途声明**：本报告为方法学研究材料，不用于医学诊断、不替代专业评估。"
        "仿真提供先验与机制假设，不提供真理；真实人类数据负责逐步校正先验。"
    )
    return "\n".join(lines)


def write_atlas(atlas: Atlas, md_path: Path | str, json_path: Path | str | None = None) -> dict:
    """写出 Atlas（Markdown + 可选 JSON）。"""
    md_out = Path(md_path)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_atlas(atlas), encoding="utf-8")
    result = {"markdown": str(md_out)}
    if json_path:
        json_out = Path(json_path)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(atlas.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["json"] = str(json_out)
    return result
