"""冻结协议（FR-15）。

`Plan.txt` 第十五阶段：人类验证必须「冻结模型」—— Discovery → Freeze → Human phase → Validation。
一旦冻结，**禁止**根据人类结果回头修改模型，以最大限度降低 circular reasoning。

冻结五类对象（PRD FR-15 要求 1）：
1. primitive 定义     2. 扰动参数     3. 表型词表与探查登记
4. 候选机制           5. 人类预测（含**阴性预测**）

**第三类对象是表型词表与探查登记**：计分通道清单与顺序、表型维度/对照通道的划分、
各通道的观察锚点、以及每个 primitive 用哪条探查条目、有几条零敏感度对照、dose 档位 ——
也就是「测量口径」和「用什么探查」，**不含任何「该机制应体现在哪个维度」的断言**。

? **一致性**：`phenotype_vocabulary` 的内容与键名参与 `freeze_id`；修改词表会换 id。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .analysis.counterfactual import CounterfactualPrediction, build_experiment_design
from .analysis.heterogeneity import ConvergenceResult
from .ontology.assays import ASSAYS
from .ontology.primitives import PRIMITIVES
from .ontology.systems import SYSTEMS
from .phenotype import CONTROL_DIM, DIM_ANCHORS, PHENOTYPE_DIMS, SCORED_CHANNELS

_FROZEN_OBJECTS = (
    "primitive_definitions",
    "perturbation_parameters",
    "phenotype_vocabulary",
    "candidate_mechanisms",
    "human_predictions",
)


def digest(obj: object) -> str:
    """对象的稳定摘要（键排序，保证跨运行一致）。"""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _primitive_definitions() -> dict:
    return {
        p.id: {
            "name_zh": p.name_zh,
            "family": p.family,
            "definition": p.definition,
            "computational_role": p.computational_role,
            "impairment_signature": p.impairment_signature,
            "dose_levels": list(p.dose_levels),
        }
        for p in PRIMITIVES
    }


def _phenotype_vocabulary() -> dict:
    """测量口径 + 探查登记（**不含**任何 primitive → 测量维度 的对应）。"""
    return {
        "channels": list(SCORED_CHANNELS),
        "partition": {
            "phenotype_dims": list(PHENOTYPE_DIMS),
            "control_channel": CONTROL_DIM,
        },
        "anchors": {dim: dict(DIM_ANCHORS.get(dim) or {}) for dim in SCORED_CHANNELS},
        "probes": {
            a.primitive_id: {
                "primary_probe": a.primary_probe.key,
                "n_controls": a.n_controls,
                "dose_levels": list(a.dose_levels),
            }
            for a in ASSAYS
        },
        "systems": {
            s.id: {"members": list(s.members), "falsifiable_expectation": s.falsifiable_expectation}
            for s in SYSTEMS
        },
    }


@dataclass
class FreezeSnapshot:
    """一份冻结快照。"""

    freeze_id: str
    created_at: str
    digests: dict[str, str]
    objects: dict[str, object]
    experiment_design: dict
    counterfactuals: list[CounterfactualPrediction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "freeze_id": self.freeze_id,
            "created_at": self.created_at,
            "digests": self.digests,
            "objects": self.objects,
            "experiment_design": self.experiment_design,
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "notes": self.notes,
        }


def build_freeze(
    *,
    perturbations: list[dict],
    convergence: ConvergenceResult | None,
    counterfactuals: list[CounterfactualPrediction] | None = None,
    model_version: str = "pacs-v0.1",
) -> FreezeSnapshot:
    """生成冻结快照。`freeze_id` 由五类对象的摘要共同决定 —— 任一对象变化都会换 id，"""

    candidate_mechanisms = (
        [
            {
                "primitive_id": c.primitive_id,
                "dose": c.dose,
                "endpoint_dim": c.endpoint_dim,
                "endpoint_value": c.endpoint_value,
                "feature_delta": c.feature_delta,
            }
            for c in (convergence.candidates if convergence else [])
        ]
        if convergence
        else []
    )

    predictions = {
        "endpoint_dim": convergence.endpoint_dim if convergence else None,
        "minimal_task_set": list(convergence.minimal_task_set) if convergence else [],
        "pair_separation": {
            k: list(v) for k, v in (convergence.pair_separation.items() if convergence else [])
        },
        "counterfactuals": [c.to_dict() for c in (counterfactuals or [])],
        "negative_predictions": dict(convergence.negative_predictions) if convergence else {},
    }

    objects: dict[str, object] = {
        "primitive_definitions": _primitive_definitions(),
        "perturbation_parameters": {"perturbations": perturbations},
        "phenotype_vocabulary": _phenotype_vocabulary(),
        "candidate_mechanisms": {"candidates": candidate_mechanisms},
        "human_predictions": predictions,
    }

    digests = {name: digest(objects[name]) for name in _FROZEN_OBJECTS}
    combined = digest(digests)
    freeze_id = f"freeze_{combined}"

    import datetime

    experiment_design = build_experiment_design(convergence) if convergence else {}

    notes = [
        "冻结后任何对上述对象的修改都会产生新的 freeze_id；旧预测保持不变（FR-15 要求 1）。",
        "第三类对象是「表型词表与探查登记」：只冻结测量口径与探查设计，"
        "不含任何 primitive → 测量维度 的对应（该对应已按构造效度要求整体删除）。",
        "人类侧接口（Human phenotype → P(M_i|D)）只定义格式、不在此实现（FR-15 要求 2）。",
        "验证时须同时检验阴性预测（「什么应该正常」），不得只看异常项（FR-15 要求 3）。",
    ]
    if convergence and not convergence.is_heterogeneous:
        notes.append("当前候选机制未达「可区分」标准，冻结的预测尚不足以支持机制判别实验。")

    return FreezeSnapshot(
        freeze_id=freeze_id,
        created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        digests=digests,
        objects=objects,
        experiment_design=experiment_design,
        counterfactuals=list(counterfactuals or []),
        notes=notes,
    )


def save_freeze(snapshot: FreezeSnapshot, directory: Path | str) -> Path:
    """写出冻结产物到 `exports/freeze_<id>/`。"""
    out_dir = Path(directory) / snapshot.freeze_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshot.json").write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 预测单独成文，便于交给人类实验方
    (out_dir / "human_predictions.json").write_text(
        json.dumps(snapshot.objects.get("human_predictions", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def verify_freeze(snapshot: FreezeSnapshot) -> list[str]:
    """校验冻结快照自身一致性（摘要是否与对象内容吻合）。"""
    problems: list[str] = []
    for name in _FROZEN_OBJECTS:
        if name not in snapshot.objects:
            problems.append(f"冻结对象缺失：{name}")
            continue
        expected = digest(snapshot.objects[name])
        if snapshot.digests.get(name) != expected:
            problems.append(f"{name} 的摘要与内容不符（快照被篡改？）")
    combined = digest(snapshot.digests)
    if not snapshot.freeze_id.endswith(combined):
        problems.append("freeze_id 与对象摘要不一致")
    return problems


def compare_freeze(a: FreezeSnapshot, b: FreezeSnapshot) -> dict:
    """比较两份冻结，指出哪些对象发生了变化（用于「禁止事后改模型」的审计）。"""
    changed = [name for name in _FROZEN_OBJECTS if a.digests.get(name) != b.digests.get(name)]
    return {
        "identical": a.freeze_id == b.freeze_id,
        "a": a.freeze_id,
        "b": b.freeze_id,
        "changed_objects": changed,
    }
