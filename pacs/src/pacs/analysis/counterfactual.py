"""反事实预测与人类实验反向设计（FR-14）。

`Plan.txt` 第十一 / 十四阶段：不要先做人类实验再找模型能不能解释，而是**让仿真告诉你
最应该测什么**。

- **反事实预测**：`do(M_i ↑) → 恢复模式`。若把机制 M_i 的 primitive 恢复到完整，
  哪些维度应当恢复、哪些**不应**变化；
- **实验反向设计**：对候选机制，给出「最少行为任务集」+ 每个任务的判别目标与预期结果模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..phenotype import PHENOTYPE_DIMS
from .heterogeneity import ConvergenceResult, MechanismCandidate

RECOVERY_TOLERANCE = 0.10   # 恢复量在此以内 → 预测「不应改变」（阴性预测）


@dataclass
class CounterfactualPrediction:
    """一个机制的反事实预测。"""

    mechanism: str          # primitive_id
    observed_dose: float
    intervention: str
    recovered: dict[str, float] = field(default_factory=dict)
    unchanged: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "observed_dose": self.observed_dose,
            "intervention": self.intervention,
            "recovered": {k: round(v, 4) for k, v in self.recovered.items()},
            "unchanged": {k: round(v, 4) for k, v in self.unchanged.items()},
        }


def predict_counterfactual(
    candidate: MechanismCandidate,
    baseline_dims: dict[str, float | None],
) -> CounterfactualPrediction:
    """由「机制扰动的表型」与「完整心智基线」推得 `do(机制 ↑)` 的恢复模式。

    恢复量 = 基线值 − 机制下的值（正 = 该维度应当改善）。约等于 0 的维度即阴性预测
    （「这个机制本来就不该改变它」）。
    """
    recovered: dict[str, float] = {}
    unchanged: dict[str, float] = {}
    for dim in PHENOTYPE_DIMS:
        base = baseline_dims.get(dim)
        observed = candidate.dims.get(dim)
        if base is None or observed is None:
            continue
        gain = float(base) - float(observed)
        if gain > RECOVERY_TOLERANCE:
            recovered[dim] = gain
        else:
            unchanged[dim] = float(observed)
    members = "、".join(candidate.members)
    return CounterfactualPrediction(
        mechanism=candidate.primitive_id,
        observed_dose=candidate.dose,
        intervention=f"do({members} ↑)",
        recovered=recovered,
        unchanged=unchanged,
    )


def build_counterfactuals(result: ConvergenceResult) -> list[CounterfactualPrediction]:
    """对一组候选机制给出各自的反事实预测 —— 应当彼此**不同**（这是机制可区分的核心证据）。"""
    return [predict_counterfactual(c, result.baseline_dims) for c in result.candidates]


@dataclass
class ExperimentTask:
    """建议纳入人类实验的一个任务维度。

    字段名刻意用 `measured_dim` 而不是 `target_dim`：它是「这项任务用来测量哪个表型维度、
    以便区分机制」，与「某机制应当体现在哪个维度」的预设期望无关（后者已被明令禁止）。
    """

    measured_dim: str
    separates: list[str]           # 它能区分哪些机制对
    expected: dict[str, str]       # 机制 → 预期方向描述

    def to_dict(self) -> dict:
        return {"measured_dim": self.measured_dim, "separates": self.separates, "expected": self.expected}


def build_experiment_design(
    result: ConvergenceResult,
    *,
    margin: float = 0.12,
) -> dict:
    """由收敛分析产出「最少人类实验设计」。

    输出：任务清单（每个任务的判别目标 + 各机制的预期方向）、以及不应纳入的任务
    （基线已受损、无区分力的维度）。
    """
    if not result.candidates:
        return {"tasks": [], "notes": ["无候选机制，无法设计判别实验"]}

    tasks: list[ExperimentTask] = []
    for dim in result.minimal_task_set:
        separates: list[str] = []
        expected: dict[str, str] = {}
        for c in result.candidates:
            value = c.feature_delta.get(dim)
            if value is None:
                expected[c.primitive_id] = "无数据"
                continue
            direction = "下降" if value < -margin else ("上升" if value > margin else "基本不变")
            expected[c.primitive_id] = f"{direction}（Δ={value:+.2f}）"
        # 该维度对哪些机制对具有区分力
        for i, a in enumerate(result.candidates):
            for b in result.candidates[i + 1 :]:
                va = a.feature_delta.get(dim)
                vb = b.feature_delta.get(dim)
                if va is not None and vb is not None and abs(va - vb) >= margin:
                    separates.append(f"{a.primitive_id}|{b.primitive_id}")
        tasks.append(ExperimentTask(measured_dim=dim, separates=separates, expected=expected))

    notes: list[str] = []
    if not tasks:
        notes.append("候选机制在当前任务集下不可区分，建议扩充任务或加入新的判别维度。")
    excluded = [
        dim
        for dim in PHENOTYPE_DIMS
        if dim not in result.minimal_task_set and dim != result.endpoint_dim
    ]
    notes.append(
        "无需纳入的任务（无区分力）：" + ("、".join(excluded) if excluded else "（无）")
    )
    notes.append(
        f"机制终点维度「{result.endpoint_dim}」本身不用于判别（各机制在此按构造相同），"
        "判别必须依赖伴随表型。"
    )
    return {
        "endpoint_dim": result.endpoint_dim,
        "n_tasks": len(tasks),
        "tasks": [t.to_dict() for t in tasks],
        "excluded_dims": excluded,
        "notes": notes,
        "power_note": (
            "样本量估计：判别依赖任务间的**模式差异**而非单任务均值，"
            "建议按配对设计的效应量（取候选机制间最小 Δ）与 80% 把握度估算最小样本量。"
        ),
    }


def render_experiment_design(design: dict, counterfactuals: list[CounterfactualPrediction]) -> str:
    lines = ["反事实预测与人类实验反向设计（FR-14）", "=" * 78]
    lines.append(f"表型终点：{design.get('endpoint_dim')}；建议任务数：{design.get('n_tasks', 0)}")
    lines.append("")
    lines.append("① 反事实预测 do(M ↑) —— 恢复模式（机制间应当不同）")
    for cf in counterfactuals:
        recovered = "、".join(f"{k}(+{v:.2f})" for k, v in sorted(cf.recovered.items())) or "（无）"
        lines.append(f"  - {cf.intervention}")
        lines.append(f"      应恢复：{recovered}")
    lines.append("")
    lines.append("② 建议的人类实验任务集")
    for task in design.get("tasks", []):
        lines.append(f"  - 任务维度 {task['measured_dim']}（可区分 {len(task['separates'])} 对机制）")
        for mechanism, expectation in sorted(task["expected"].items()):
            lines.append(f"      {mechanism:<30} 预期：{expectation}")
    lines.append("")
    lines.append("③ 说明")
    for note in design.get("notes", []):
        lines.append(f"  - {note}")
    lines.append(f"  - {design.get('power_note', '')}")
    return "\n".join(lines)
