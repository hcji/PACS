"""剂量-反应分析（FR-9.3）：`primitive strength → behavior` 应当连续、可重复。

**逐通道**拟合：对每个 primitive，把 6 个表型维度 + 对照通道各自拟合一条 `dose → 读数`
曲线，而不是只拟合一条事先指定的「目标维度」。原因与 `specificity.py` 相同 ——
构造属性与测量维度之间不允许有预设对应，所以「哪个维度随剂量变化」也由数据给出：
`dominant_dim` 取拟合良好的通道里 |slope| 最大者，`per_dim` 保留全部通道的拟合摘要，
读者能看到完整的剂量-反应结构，而不是被挑出来的那一条。

判定只看**该通道自身的数据自洽性**（点数、方向、r²、按剂量聚合后的单调性），
不涉及任何机制与维度的预先匹配。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS
from ..phenotype import PhenotypeProfile
from ..ontology.primitives import PRIMITIVES
from .stats import is_monotonic, linear_fit, mean, stdev

# 判定门槛
PASS_R2 = 0.5
WEAK_R2 = 0.2


@dataclass
class DosePoint:
    dose: float
    value: float


@dataclass
class DoseResponseFit:
    """一个 primitive 的剂量-反应拟合结果。"""

    primitive_id: str
    dominant_dim: str
    points: list[DosePoint]
    slope: float
    r2: float
    monotonic: bool
    direction: str
    verdict: str
    reason: str
    spread: float = 0.0
    per_dim: dict[str, dict] = field(default_factory=dict)   # 通道 → {slope, r2, n_points, direction}
    n_dims_fitted: int = 0

    @property
    def n_points(self) -> int:
        return len(self.points)

    def to_dict(self) -> dict:
        return {
            "primitive_id": self.primitive_id,
            "dominant_dim": self.dominant_dim,
            "n_points": self.n_points,
            "slope": round(float(self.slope), 4),
            "r2": round(float(self.r2), 4),
            "monotonic": bool(self.monotonic),
            "direction": self.direction,
            "verdict": self.verdict,
            "reason": self.reason,
            "spread": round(float(self.spread), 4),
            "n_dims_fitted": self.n_dims_fitted,
            "per_dim": self.per_dim,
        }


def single_perturbation_series(
    profiles: list[PhenotypeProfile],
    primitive_id: str,
    dim: str,
    *,
    model_id: str | None = None,
) -> list[DosePoint]:
    """收集「只扰动该 primitive」的所有 (dose, 该通道读数) 点，并补上完整心智的基线（dose=1）。"""
    pts: list[DosePoint] = []
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        active = {k: v for k, v in p.theta.items() if v < 0.999}
        if p.generic_degradation > 0:
            continue
        if set(active.keys()) != {primitive_id}:
            continue
        value = p.dims.get(dim)
        if value is None:
            continue
        pts.append(DosePoint(dose=float(active[primitive_id]), value=float(value)))

    # 完整心智基线（dose = 1.0）
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        if not p.theta or any(v < 0.999 for v in p.theta.values()):
            continue
        if p.generic_degradation > 0:
            continue
        value = p.dims.get(dim)
        if value is None:
            continue
        pts.append(DosePoint(dose=1.0, value=float(value)))
        break

    pts.sort(key=lambda d: d.dose)
    return pts


def _fit_series(pts: list[DosePoint]) -> dict:
    """拟合一条通道的序列；点数不足时返回 insufficient 标记。"""
    if len(pts) < 3:
        return {"n_points": len(pts), "slope": 0.0, "r2": 0.0, "direction": "unknown",
                "spread": 0.0, "insufficient": True}
    xs = [p.dose for p in pts]
    ys = [p.value for p in pts]
    slope, _intercept, r2 = linear_fit(xs, ys)

    by_dose: dict[float, list[float]] = {}
    for p in pts:
        by_dose.setdefault(p.dose, []).append(p.value)
    ordered = sorted(by_dose)
    agg = [mean(by_dose[d]) for d in ordered]
    _monotonic, direction = is_monotonic(agg)
    spread = (
        mean([stdev(by_dose[d]) for d in ordered if len(by_dose[d]) > 1])
        if any(len(by_dose[d]) > 1 for d in ordered) else 0.0
    )
    return {"n_points": len(pts), "slope": slope, "r2": r2, "direction": direction,
            "spread": spread, "insufficient": False}


def fit_dose_response(
    profiles: list[PhenotypeProfile],
    primitive_id: str,
    *,
    model_id: str | None = None,
    channels: tuple[str, ...] = tuple(SCORED_CHANNELS),
) -> DoseResponseFit:
    """逐通道拟合单个 primitive 的剂量-反应，并给出数据驱动的主导通道。"""
    series = {ch: single_perturbation_series(profiles, primitive_id, ch, model_id=model_id)
              for ch in channels}
    per_dim: dict[str, dict] = {}
    for ch, pts in series.items():
        if not pts:
            continue
        fit = _fit_series(pts)
        per_dim[ch] = {
            "n_points": fit["n_points"],
            "slope": round(float(fit["slope"]), 4),
            "r2": round(float(fit["r2"]), 4),
            "direction": fit["direction"],
        }

    # 主导通道：在表型维度里取「拟合可用且 |slope| 最大」者（对照通道只报告，不参与判定）
    usable = [
        (ch, per_dim[ch]) for ch in PHENOTYPE_DIMS
        if ch in per_dim and per_dim[ch]["n_points"] >= 3
    ]
    if not usable:
        return DoseResponseFit(
            primitive_id=primitive_id,
            dominant_dim="",
            points=[],
            slope=0.0,
            r2=0.0,
            monotonic=False,
            direction="unknown",
            verdict="insufficient",
            reason="所有表型维度的可用点都不足 3 个，无法拟合剂量-反应（需更多 dose 档或重复）",
            per_dim=per_dim,
            n_dims_fitted=len(per_dim),
        )
    dominant_dim, _ = max(usable, key=lambda kv: (abs(kv[1]["slope"]), kv[0]))
    pts = series[dominant_dim]
    fit = _fit_series(pts)
    slope, r2 = float(fit["slope"]), float(fit["r2"])
    by_dose: dict[float, list[float]] = {}
    for p in pts:
        by_dose.setdefault(p.dose, []).append(p.value)
    ordered = sorted(by_dose)
    monotonic, direction = is_monotonic([mean(by_dose[d]) for d in ordered])

    if slope > 0 and r2 >= PASS_R2 and direction in ("increasing", "flat"):
        verdict = "pass"
        reason = (
            f"剂量升高 → 该通道表现改善（slope={slope:+.3f}, r²={r2:.2f}，"
            f"通道「{dominant_dim}」）；重复波动={fit['spread']:.3f}"
        )
    elif slope > 0 and r2 >= WEAK_R2:
        verdict = "weak"
        reason = f"方向正确但拟合偏弱（slope={slope:+.3f}, r²={r2:.2f}）；建议增加重复次数"
    else:
        verdict = "fail"
        reason = (
            f"未观察到连续可重复的剂量-反应（slope={slope:+.3f}, r²={r2:.2f}, "
            f"单调性={direction}，通道「{dominant_dim}」）"
        )
    return DoseResponseFit(
        primitive_id=primitive_id,
        dominant_dim=dominant_dim,
        points=pts,
        slope=slope,
        r2=r2,
        monotonic=monotonic,
        direction=direction,
        verdict=verdict,
        reason=reason,
        spread=float(fit["spread"]),
        per_dim=per_dim,
        n_dims_fitted=len(per_dim),
    )


def build_dose_response(
    profiles: list[PhenotypeProfile],
    *,
    model_id: str | None = None,
    primitives: tuple[str, ...] | None = None,
) -> list[DoseResponseFit]:
    """对全部（或指定）primitive 建立剂量-反应拟合。每个 primitive 一定有一行结果。"""
    ids = primitives or tuple(p.id for p in PRIMITIVES)
    return [fit_dose_response(profiles, pid, model_id=model_id) for pid in ids]


def summarize(fits: list[DoseResponseFit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in fits:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    return counts


def render_table(fits: list[DoseResponseFit]) -> str:
    lines = ["剂量-反应（FR-9.3，逐通道拟合；主导通道由数据给出）", "=" * 96]
    lines.append(
        f"{'primitive':<34}{'主导通道':<36}{'slope':>8}{'r²':>7}{'拟合通道数':>10}  判定"
    )
    lines.append("-" * 96)
    for f in fits:
        lines.append(
            f"{f.primitive_id:<34}{f.dominant_dim or '—':<36}"
            f"{f.slope:>+8.3f}{f.r2:>7.2f}{f.n_dims_fitted:>10}  {f.verdict}"
        )
    return "\n".join(lines)
