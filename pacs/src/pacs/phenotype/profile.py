"""表型剖面：一个被试的 `θ → B` 记录，以及基线聚合与对照通道取值。

**表型一律由 LLM 读出**：`B` 来自「提问者 LLM 提问 → 被扰动的心智作答 → 分析者 LLM 逐维
读数」，即 `agents.measure.measure_phenotype`。本模块只负责**装**这份读数（以及为下游分析
提供基线聚合），不含任何规则式测量。

方向约定与 θ 一致：**1.0 = 典型/完好，0.0 = 严重受损**，落在下游分析使用的 0..1 区间里。
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from .. import DEFAULT_AGE_MONTHS
from .dims import CONTROL_DIM, SCORED_CHANNELS


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class PhenotypeProfile:
    """一个虚拟被试的表型：θ → B（由分析者 LLM 从对话里读出）。

    `dims` 的方向：**1.0 = 典型/完好，0.0 = 严重受损**。
    `dims` 的键是**计分通道**（6 个表型维度 + 对照通道）；对照通道不进雷达图，但必须在此，
    否则特异性判定没有「应该不变」的参照（见 `phenotype/dims.py` 的说明）。

    计数口径（对话通道）：
    - `n_items`  计划的观察轮数；
    - `n_parsed` 实际完成的轮数；
    - `parse_rate = n_parsed / n_items` 即**对话完成率**。

    `instrument` 记录这份读数的来源工具。当前只会是 `"dialogue"`；更早落库、
    来源字段不同的剖面仍能读出来，但不会被静默当成同源数据。

    `provider_kind` 记录读数是**谁的**：真实 LLM（`"llm"`）还是离线规则式替身（`"mock"`）。
    mock 路径下的表型结构是**构造出来的**，不是 LLM 涌现的，因此下游分析与报告
    会据此给出免责说明（见 `analysis/mechanism_profile.py` 与 `report.py`）。
    """

    theta_id: str
    model_id: str
    run_id: str
    theta: dict[str, float]
    generic_degradation: float
    dims: dict[str, Optional[float]]
    n_items: int
    n_parsed: int
    #: 心智年龄（月）：**固定常量**，默认即 `DEFAULT_AGE_MONTHS`（见 `pacs.__init__`）。
    #: 字段保留是为了旧剖面仍可读（`from_dict` 照读历史值），新写入一律是常量 ——
    #: 参照线随参数漂移，各次读数就不可比。
    age_months: int = DEFAULT_AGE_MONTHS
    replicate: int = 0          # 同一 θ 的第几次重复测量（参照条件需重复以降低噪声）
    design: str = ""
    note: str = ""
    instrument: str = "dialogue"
    confidence: Optional[float] = None   # 分析者自评置信度
    ts: str = field(default_factory=now_iso)
    #: 产出这份读数的 provider 类别：`"llm"`（真实模型，读数由分析者涌现）或
    #: `"mock"`（离线规则式替身 —— 读数由构造得出，**不构成涌现证据**）。
    provider_kind: str = "llm"
    #: 本轮对话**未被探查**的计分通道（没有证据支撑的维度）。显式记录，使「没问过」
    #: 不会被下游当成「表现典型」——这些通道在 `dims` 里通常是 `None`。
    uncovered_channels: tuple[str, ...] = ()

    @property
    def synthetic(self) -> bool:
        """读数是否由规则式 mock 构造（而非 LLM 涌现）。真实结论必须用非合成的剖面。"""
        return self.provider_kind == "mock"

    @property
    def parse_rate(self) -> float:
        """对话完成率（完成的轮数 / 计划的轮数）。"""
        return self.n_parsed / self.n_items if self.n_items else 0.0

    @property
    def general_cognition(self) -> Optional[float]:
        """对照通道的值 —— 阴性预测的核心指标。"""
        return self.dims.get(CONTROL_DIM)

    def active_levels(self) -> list[float]:
        """被削弱的 primitive 的完整度取值（升序）。"""
        return sorted(v for v in self.theta.values() if v < 0.999)

    def active_primitives(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.theta.items() if v < 0.999))

    def to_dict(self) -> dict:
        return {
            "theta_id": self.theta_id,
            "model_id": self.model_id,
            "run_id": self.run_id,
            "age_months": self.age_months,
            "theta": {k: round(float(v), 4) for k, v in self.theta.items()},
            "generic_degradation": round(float(self.generic_degradation), 4),
            "dims": {k: (None if v is None else round(float(v), 4)) for k, v in self.dims.items()},
            "n_items": self.n_items,
            "n_parsed": self.n_parsed,
            "parse_rate": round(self.parse_rate, 4),
            "replicate": self.replicate,
            "design": self.design,
            "note": self.note,
            "instrument": self.instrument,
            "confidence": (None if self.confidence is None else round(float(self.confidence), 4)),
            "provider_kind": self.provider_kind,
            "synthetic": self.synthetic,
            "uncovered_channels": list(self.uncovered_channels),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhenotypeProfile":
        return cls(
            theta_id=data["theta_id"],
            model_id=data["model_id"],
            run_id=data.get("run_id", ""),
            age_months=int(data.get("age_months", 36)),
            theta={k: float(v) for k, v in (data.get("theta") or {}).items()},
            generic_degradation=float(data.get("generic_degradation", 0.0)),
            dims={k: (None if v is None else float(v)) for k, v in (data.get("dims") or {}).items()},
            n_items=int(data.get("n_items", 0)),
            n_parsed=int(data.get("n_parsed", 0)),
            replicate=int(data.get("replicate", 0)),
            design=data.get("design", ""),
            note=data.get("note", ""),
            instrument=data.get("instrument", "dialogue"),
            confidence=(
                None if data.get("confidence") is None else float(data["confidence"])
            ),
            provider_kind=str(
                data.get("provider_kind")
                # 旧行没有这个字段：只能从 model_id 前缀回推（mock provider 的 model_id
                # 形如 "mock:…"）。默认「llm」会把规则构造的旧剖面冒充成涌现读数。
                or ("mock" if data.get("synthetic")
                    or str(data.get("model_id") or "").startswith("mock") else "llm")
            ),
            uncovered_channels=tuple(data.get("uncovered_channels") or ()),
            ts=data.get("ts", now_iso()),
        )


def aggregate_baseline(
    profiles: list[PhenotypeProfile], model_id: str | None = None
) -> PhenotypeProfile | None:
    """把所有「完整心智」剖面**取平均**，作为比较基线。

    为什么必须平均：单次对话得到的逐维读数是**一次抽样**，噪声大（分析者的给分在同一次
    会话内互相相关，跨会话才独立）。若只用一个基线剖面，这个噪声会进入**所有**下游比较
    （特异性、剂量-反应、通用退化对照），足以把真实效应淹没，甚至让通用退化对照判不出来。
    对 k 个基线取平均可把基线噪声降到约 1/√k。
    """
    intacts = [
        p
        for p in profiles
        if p.generic_degradation <= 0 and not any(v < 0.999 for v in p.theta.values())
    ]
    if model_id:
        intacts = [p for p in intacts if p.model_id == model_id]
    if not intacts:
        return None
    dims: dict[str, Optional[float]] = {}
    for dim in SCORED_CHANNELS:
        values = [p.dims[dim] for p in intacts if p.dims.get(dim) is not None]
        dims[dim] = sum(values) / len(values) if values else None
    first = intacts[0]
    confidences = [p.confidence for p in intacts if p.confidence is not None]
    return PhenotypeProfile(
        theta_id=f"baseline_mean_of_{len(intacts)}",
        model_id=first.model_id,
        run_id=first.run_id,
        age_months=first.age_months,
        theta=dict(first.theta),
        generic_degradation=0.0,
        dims=dims,
        n_items=sum(p.n_items for p in intacts),
        n_parsed=sum(p.n_parsed for p in intacts),
        design="baseline_aggregate",
        note=f"由 {len(intacts)} 个完整心智剖面平均得到",
        instrument=first.instrument,
        confidence=(sum(confidences) / len(confidences)) if confidences else None,
        # 基线里若混了离线 mock 剖面，这一点必须能被下游看出来（否则会静默把构造的
        # 表型当成 LLM 涌现的基线）。只要有一个合成剖面就标 synthetic。
        provider_kind="mock" if any(p.synthetic for p in intacts) else "llm",
    )


def synthetic_count(profiles: list[PhenotypeProfile]) -> int:
    """整批剖面里由规则式 mock 构造的条数（报告免责行用它决定是否告警）。"""
    return sum(1 for p in profiles if p.synthetic)


def uncovered_counts(profiles: list[PhenotypeProfile]) -> dict[str, int]:
    """各计分通道在整批剖面里「没有证据支撑」的次数 —— 覆盖是否均衡的可观测口径。"""
    counts: dict[str, int] = {}
    for p in profiles:
        for ch in p.uncovered_channels:
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def control_channel_value(profile: PhenotypeProfile) -> Optional[float]:
    """对照通道（一般认知）的值。"""
    return profile.dims.get(CONTROL_DIM)


def all_controls_intact(profile: PhenotypeProfile, threshold: float = 0.75) -> bool:
    """对照通道是否保持完好（FR-9.2 特异性的必要条件）。"""
    value = control_channel_value(profile)
    return value is not None and value >= threshold
