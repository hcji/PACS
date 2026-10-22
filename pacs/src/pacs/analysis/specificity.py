"""构造效度与特异性检验（FR-9、FR-11）—— **数据驱动，无预设对应**。

判据（Plan.txt 第六阶段）：
- **特异性**：扰动某个 primitive 应当让**某一个**表型维度显著下降，而**对照通道**
  （general_cognition）基本不变。若出现下降的维度与对照通道一起变差 → 该扰动不合格。
- **通用退化对照（Control 1）**：整体能力下降时，**所有**任务（含对照通道）都应变差。
  它提供了「特异性」的参照系：如果某 primitive 的表现与通用退化无法区分，就不能声称
  观测到了特异机制。

**为什么这里没有「目标维度」**：本项目不允许在构造属性（primitive / θ）与测量维度之间
存在预设对应 —— 测量维度读数只由分析者 LLM 从「提问者提问 → 心智作答」的对话得出
（`agents/measure.py`）。因此「该机制应体现在哪个维度」不由一张表事先指定，而是**由数据
本身给出**：在 6 个表型维度里取下降最多的那个作为 `dominant_dim`，再检验三件事 ——
① 下降是否达到门槛（TARGET_DROP）；② 效应是否**集中**（最强维度显著强于次强维度）；
③ 对照通道是否稳定（CONTROL_TOLERANCE）。

**事后选择主导维度会引入选择偏倚**，因此本模块做了三重对冲：
- 主导维度必须在 **6 个表型维度**里选（对照通道不参与选择，只用于稳定性判据）；
- 判据带**自校准的集中度门槛**（`max(CONCENTRATION_MARGIN, 2·SE)`，SE 由重复测量估计），
  噪声大时门槛自动抬高，避免把「四处都抖一点」误判成集中效应；
- `SpecificityVerdict.profile` 输出**全部计分通道的 Δ**，读者看得到完整效应剖面，
  而不是只看到一个被挑出来的维度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..phenotype import CONTROL_DIM, PHENOTYPE_DIMS
from ..phenotype import PhenotypeProfile, aggregate_baseline, control_channel_value
from .stats import mean, stdev

TARGET_DROP = 0.15       # 主导维度至少下降这么多才算「有效扰动」
# 对照通道的完整性判据：**相对基线**而非绝对下限。原因：单次对话的逐维读数抽样噪声大
# （同一个分析者在一次会话内的给分互相相关），若用绝对阈值（如 0.70）会把纯噪声误判为
# 「对照能力受损」，从而把真实的强效应错判成 fail_control。容差取噪声带的约 1.5 倍。
CONTROL_TOLERANCE = 0.15
MAX_CONTROL_TOLERANCE = 0.28   # 自校准阈值的上限（见 evaluate_generic_degradation）
# 集中度下限：最强维度的下降必须比次强维度多这么多（再有 2·SE 的自校准加持）
CONCENTRATION_MARGIN = 0.05


@dataclass
class SpecificityVerdict:
    """一个 primitive 的特异性检验结果（主导维度由数据给出，不是预设）。"""

    primitive_id: str
    dose: float
    dominant_dim: str            # 下降最多的表型维度（事后选定）
    dominant_delta: float
    second_dim: str
    second_delta: float
    concentration: float         # second_delta − dominant_delta，越大越集中
    control_value: float
    control_delta: float
    profile: dict[str, float] = field(default_factory=dict)   # 全部计分通道的 Δ
    n_replicates: int = 1
    verdict: str = ""            # pass | fail_diffuse | no_effect | fail_control | fail_both | insufficient
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "primitive_id": self.primitive_id,
            "dose": self.dose,
            "dominant_dim": self.dominant_dim,
            "dominant_delta": round(self.dominant_delta, 4),
            "second_dim": self.second_dim,
            "second_delta": round(self.second_delta, 4),
            "concentration": round(self.concentration, 4),
            "control_value": round(self.control_value, 4),
            "control_delta": round(self.control_delta, 4),
            "profile": {k: round(float(v), 4) for k, v in self.profile.items()},
            "n_replicates": self.n_replicates,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def baseline_profile(profiles: list[PhenotypeProfile], model_id: str | None = None) -> PhenotypeProfile | None:
    """完整心智的基线表型：对**所有**完整心智剖面取平均（降低单次抽样的噪声）。"""
    return aggregate_baseline(profiles, model_id)


def single_perturbation_groups(
    profiles: list[PhenotypeProfile], model_id: str | None
) -> dict[str, dict[float, list[PhenotypeProfile]]]:
    """按 primitive → dose → 剖面列表分组，只收「只扰动了一个 primitive」的剖面。

    公开给 `mechanism_profile` 复用：两个模块对「什么是单扰动被试」必须完全一致，
    否则估计出的剖面与特异性判定的口径会悄悄分叉。
    """
    grouped: dict[str, dict[float, list[PhenotypeProfile]]] = {}
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        if p.generic_degradation > 0:
            continue
        active = {k: v for k, v in p.theta.items() if v < 0.999}
        if len(active) != 1:
            continue
        pid = next(iter(active))
        dose = round(float(active[pid]), 6)
        grouped.setdefault(pid, {}).setdefault(dose, []).append(p)
    return grouped


def evaluate_specificity(
    profiles: list[PhenotypeProfile],
    *,
    model_id: str | None = None,
    dose: float | None = None,
) -> list[SpecificityVerdict]:
    """对每个 primitive 在其**最强扰动**（默认最低 dose）上做特异性判定。

    同一 dose 的多次重复会先按通道取均值再判定 —— 单次对话的逐维读数噪声很大，
    不聚合会把噪声当成效应（`aggregate_baseline` 的 docstring 说明了同一件事）。
    """
    base = baseline_profile(profiles, model_id)
    if base is None:
        return []
    base_control = control_channel_value(base)
    base_dims = {dim: base.dims.get(dim) for dim in PHENOTYPE_DIMS}

    verdicts: list[SpecificityVerdict] = []
    for pid, by_dose in sorted(single_perturbation_groups(profiles, model_id).items()):
        if dose is not None:
            probes = [
                p for p in sum(by_dose.values(), [])
                if abs(next(iter(p.active_levels())) - dose) < 1e-6
            ]
        else:
            probes = by_dose[min(by_dose)]
        if not probes:
            continue
        probe_dose = float(next(iter(probes[0].active_levels())))

        # 逐通道跨重复均值 → Δ（相对基线）
        profile: dict[str, float] = {}
        missing: list[str] = []
        for ch in (*PHENOTYPE_DIMS, CONTROL_DIM):
            base_value = base_control if ch == CONTROL_DIM else base_dims.get(ch)
            values = [float(p.dims[ch]) for p in probes if p.dims.get(ch) is not None]
            if base_value is None or not values:
                missing.append(ch)
                continue
            profile[ch] = mean(values) - float(base_value)

        # 表型维度里的主导维度：下降最多者（对照通道不参与选择）
        candidates = {ch: profile[ch] for ch in PHENOTYPE_DIMS if ch in profile}
        if not candidates or CONTROL_DIM not in profile:
            verdicts.append(
                SpecificityVerdict(
                    primitive_id=pid,
                    dose=probe_dose,
                    dominant_dim="",
                    dominant_delta=float("nan"),
                    second_dim="",
                    second_delta=float("nan"),
                    concentration=float("nan"),
                    control_value=float("nan"),
                    control_delta=float("nan"),
                    profile=profile,
                    n_replicates=len(probes),
                    verdict="insufficient",
                    reason="主导维度或对照通道缺少有效读数（未覆盖、解析失败或未评分）："
                           + "、".join(missing or ["—"]),
                )
            )
            continue

        ordered = sorted(candidates.items(), key=lambda kv: (kv[1], kv[0]))
        dominant_dim, dominant_delta = ordered[0]
        second_dim, second_delta = ordered[1] if len(ordered) > 1 else ("", 0.0)
        control_delta = float(profile[CONTROL_DIM])

        # 自校准集中度门槛：跨重复估计主导维度读数的标准误
        margin = CONCENTRATION_MARGIN
        series = [float(p.dims[dominant_dim]) for p in probes if p.dims.get(dominant_dim) is not None]
        if len(series) >= 3:
            se = stdev(series) / (len(series) ** 0.5)
            margin = max(CONCENTRATION_MARGIN, 2.0 * se)
        concentration = second_delta - dominant_delta

        hit = dominant_delta <= -TARGET_DROP
        control_ok = control_delta >= -CONTROL_TOLERANCE
        concentrated = concentration >= margin

        if not control_ok:
            verdict = "fail_control" if hit else "fail_both"
            reason = (
                f"对照通道变化 {control_delta:+.2f}（超出 ±{CONTROL_TOLERANCE} 容差）→ "
                + ("疑似非特异通用退化，不能归因于该 primitive"
                   if hit else "既损害了对照能力，也未在任何表型维度上产生达标下降")
            )
        elif not hit:
            verdict = "no_effect"
            reason = (
                f"最强维度仅变化 {dominant_delta:+.2f}（未达 −{TARGET_DROP}）→ 该 primitive "
                "未能单独产生可达的表型改变（FR-9.2 判为不合格）"
            )
        elif not concentrated:
            verdict = "fail_diffuse"
            reason = (
                f"主导维度「{dominant_dim}」下降 {dominant_delta:+.2f}，但次强维度"
                f"「{second_dim}」也下降 {second_delta:+.2f}（集中度 {concentration:+.2f} < "
                f"门槛 {margin:.2f}）→ 效应分散在多个维度上，不能归因于该单一机制"
            )
        else:
            verdict = "pass"
            reason = (
                f"主导维度「{dominant_dim}」下降 {dominant_delta:+.2f}（≥{TARGET_DROP}），"
                f"集中度 {concentration:+.2f}（门槛 {margin:.2f}）；对照通道变化 {control_delta:+.2f}"
                f"（在 ±{CONTROL_TOLERANCE} 容差内）→ 扰动特异"
            )
        verdicts.append(
            SpecificityVerdict(
                primitive_id=pid,
                dose=probe_dose,
                dominant_dim=dominant_dim,
                dominant_delta=dominant_delta,
                second_dim=second_dim,
                second_delta=second_delta,
                concentration=concentration,
                control_value=float(profile[CONTROL_DIM] + float(base_control)),
                control_delta=control_delta,
                profile=profile,
                n_replicates=len(probes),
                verdict=verdict,
                reason=reason,
            )
        )
    return verdicts


@dataclass
class GenericDegradationVerdict:
    """Control 1 的判定：通用退化应当**同时**拉低所有通道。"""

    level: float
    control_value: float
    mean_other_value: float
    is_nonspecific: bool
    reason: str


def evaluate_generic_degradation(
    profiles: list[PhenotypeProfile], model_id: str | None = None
) -> GenericDegradationVerdict | None:
    """评估 FR-11 Control 1：随机通用退化应当把**所有**通道一起拉低。

    判定：对照通道与其它维度的**平均下降量都超出噪声容差**，才确认这是非特异的整体退化。
    若只有对照通道掉了、其它维度没掉，那更可能是抽样噪声而非通用退化。

    判据是**自校准**的：用重复测量的离散度估计标准误 SE，要求下降量超过 max(最小容差, 2·SE)。
    这样样本噪声大时自动收紧标准，而不是靠一个固定常数硬判。

    ⚠ 通用退化强度须足够：单次对话的逐维读数噪声约 ±0.08–0.13，与较弱的通用退化效应
    同量级，因此建议对照强度 ≥ 0.5。**根本解法是增加重复次数。**
    """
    base = baseline_profile(profiles, model_id)
    if base is None:
        return None
    base_control = control_channel_value(base) or 0.0
    base_others = [
        v for k, v in base.dims.items() if k != CONTROL_DIM and v is not None
    ]
    base_mean_other = sum(base_others) / len(base_others) if base_others else 0.0

    # 对照条件也需重复测量：单个通用退化剖面的噪声足以让判定翻转。
    generic = [
        p
        for p in profiles
        if p.generic_degradation > 0 and (model_id is None or p.model_id == model_id)
    ]
    usable = [
        p
        for p in generic
        if control_channel_value(p) is not None
        and [v for k, v in p.dims.items() if k != CONTROL_DIM and v is not None]
    ]
    if usable:
        controls = [float(control_channel_value(p)) for p in usable]
        means = [
            sum(v for k, v in p.dims.items() if k != CONTROL_DIM and v is not None)
            / len([v for k, v in p.dims.items() if k != CONTROL_DIM and v is not None])
            for p in usable
        ]
        control = sum(controls) / len(controls)
        mean_other = sum(means) / len(means)
        level = sum(p.generic_degradation for p in usable) / len(usable)
        drop_control = base_control - float(control)
        drop_other = base_mean_other - mean_other
        # 自校准阈值：用对照条件的重复离散度估计标准误，要求下降超过 2·SE
        tolerance = CONTROL_TOLERANCE
        if len(controls) >= 3:
            se = stdev(controls) / (len(controls) ** 0.5)
            # 自校准但有上限：阈值随噪声升高，但不得超过 MAX，否则一个离群的重复样本
            # 会把明显的大效应误判为「不显著」。
            tolerance = min(MAX_CONTROL_TOLERANCE, max(CONTROL_TOLERANCE, 2.0 * se))
        nonspecific = drop_control > tolerance and drop_other > CONTROL_TOLERANCE
        return GenericDegradationVerdict(
            level=level,
            control_value=float(control),
            mean_other_value=float(mean_other),
            is_nonspecific=bool(nonspecific),
            reason=(
                f"通用退化 {level:.2f}（{len(usable)} 次重复取平均）："
                f"对照通道 {base_control:.2f} → {control:.2f}"
                f"（下降 {drop_control:+.2f}），其它维度均值 {base_mean_other:.2f} → {mean_other:.2f}"
                f"（下降 {drop_other:+.2f}；判定阈值 {tolerance:.2f} 为自校准）→ "
                + (
                    "确认是非特异的整体下降（这正是 Control 1 期望看到的：它可以与"
                    "「只坏一个维度」的特异扰动相区分）"
                    if nonspecific
                    else "未同时拉低两类通道，尚不足以确认通用退化"
                )
            ),
        )
    return None


def summarize(verdicts: list[SpecificityVerdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


def dominant_distribution(verdicts: list[SpecificityVerdict]) -> dict[str, int]:
    """主导维度在 6 个表型维度上的分布 —— 用来证明「没有哪个维度天生不被检验」。"""
    counts: dict[str, int] = {}
    for v in verdicts:
        if v.dominant_dim:
            counts[v.dominant_dim] = counts.get(v.dominant_dim, 0) + 1
    return counts


def render_table(verdicts: list[SpecificityVerdict]) -> str:
    lines = ["扰动特异性（FR-9.2 / FR-11，数据驱动：无预设目标维度）", "=" * 104]
    lines.append(
        f"{'primitive':<34}{'dose':>6}  {'主导维度':<34}{'Δ主导':>8}{'集中度':>8}{'Δ对照':>8}  判定"
    )
    lines.append("-" * 104)
    for v in verdicts:
        lines.append(
            f"{v.primitive_id:<34}{v.dose:>6.2f}  {v.dominant_dim or '—':<34}"
            f"{v.dominant_delta:>+8.2f}{v.concentration:>+8.2f}{v.control_delta:>+8.2f}  {v.verdict}"
        )
    return "\n".join(lines)
