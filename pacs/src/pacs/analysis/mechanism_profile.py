"""由**测量数据**估计「机制 → 表型维度」剖面（数据驱动的对应关系，唯一来源）。

**这个模块存在的理由**：本项目不允许在构造属性（primitive / θ）与测量维度之间预设对应
关系（`README.md`「扰动与表型之间不允许有任何已知条件的关联」）。但分析又必须知道「哪个
机制在表型上表现成什么样」，否则特异性、异质性、亚型都无从谈起。答案不是翻出一张事先写好
的表，而是**从测量结果里估计出来**：

    profile[mechanism][channel] = 该机制单扰动时的平均 Δ（相对完整心智基线）

它有三个必须守住的属性：

1. **它是测量产物，不是先验**。估计剖面**禁止回写**为后续运行的输入（不得喂给提示词、
   不得参与特异性判据、不得进冻结快照）。想复用它，只能在报告里作为观测结果引用。
2. **它是可被推翻的**。每个格子附重复次数；某机制若在不同重复间不稳定，读者应看得见。
3. **它不挑维度**。全部计分通道（6 个表型维度 + 对照通道）都列出来，不做「主维度」的
   预筛 —— `dominant_channel` 只是对该行数据的一个摘要，不是期望值。

特异性的判定不读本模块：`analysis/specificity.py` 自己从同一批剖面算 Δ 并判 pass/fail，
两者用同一个 `single_perturbation_groups` 分组口径，因此不会悄悄分叉。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..phenotype import CONTROL_DIM, PHENOTYPE_DIMS
from ..phenotype import PhenotypeProfile, SCORED_CHANNELS, channel_label
from .specificity import baseline_profile, single_perturbation_groups
from .stats import mean

#: 写进报告与 API 的硬约定（禁止回写）
NO_WRITEBACK_NOTE = (
    "本剖面是**测量产物**：由「提问者提问 → 心智作答 → 分析者逐维读数」得到，"
    "不是事先设定的对应关系，也**不得回写**为后续运行的先验（不得进提示词、"
    "不得参与特异性判据、不得进冻结快照）。"
)


@dataclass
class MechanismProfileRow:
    """一个 (mechanism, dose) 的实测剖面。"""

    primitive_id: str
    dose: float
    deltas: dict[str, float] = field(default_factory=dict)   # 计分通道 → 平均 Δ
    dominant_channel: str = ""
    concentration: float = 0.0        # 次强维度 Δ − 最强维度 Δ（越大越集中）
    n_replicates: int = 0
    channels_missing: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "primitive_id": self.primitive_id,
            "dose": round(float(self.dose), 4),
            "deltas": {k: round(float(v), 4) for k, v in self.deltas.items()},
            "dominant_channel": self.dominant_channel,
            "concentration": round(float(self.concentration), 4),
            "n_replicates": self.n_replicates,
            "channels_missing": list(self.channels_missing),
        }


@dataclass
class MechanismProfileReport:
    """一整批被试的实测剖面集合。"""

    rows: list[MechanismProfileRow] = field(default_factory=list)
    model_id: str | None = None
    baseline_n: int = 0
    n_profiles: int = 0
    notes: list[str] = field(default_factory=list)

    def strongest_rows(self) -> list[MechanismProfileRow]:
        """每个 primitive 取最强扰动（最低 dose）的一行 —— 报告表格用这个视图。"""
        best: dict[str, MechanismProfileRow] = {}
        for row in self.rows:
            cur = best.get(row.primitive_id)
            if cur is None or row.dose < cur.dose:
                best[row.primitive_id] = row
        return [best[pid] for pid in sorted(best)]

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "model_id": self.model_id,
            "baseline_n": self.baseline_n,
            "n_profiles": self.n_profiles,
            "notes": self.notes,
        }


def estimate_mechanism_profile(
    profiles: list[PhenotypeProfile],
    *,
    model_id: str | None = None,
) -> MechanismProfileReport:
    """从单扰动被试估计机制 → 计分通道 的 Δ 剖面。

    基线与特异性判定共用 `baseline_profile`（多次完整心智剖面取平均），
    分层口径共用 `single_perturbation_groups`。没有基线时返回空报告（附说明），
    而不是拿完整心智自己的读数当基线。
    """
    report = MechanismProfileReport(model_id=model_id, n_profiles=len(profiles))
    base = baseline_profile(profiles, model_id)
    report.notes.append(NO_WRITEBACK_NOTE)
    if base is None:
        report.notes.append("没有可用的完整心智基线（至少需要一次无扰动剖面），无法估计剖面。")
        return report

    base_intacts = [
        p for p in profiles
        if (model_id is None or p.model_id == model_id)
        and p.generic_degradation <= 0
        and p.theta
        and all(v >= 0.999 for v in p.theta.values())
    ]
    report.baseline_n = len(base_intacts)

    for pid, by_dose in sorted(single_perturbation_groups(profiles, model_id).items()):
        for dose in sorted(by_dose):
            probes = by_dose[dose]
            deltas: dict[str, float] = {}
            missing: list[str] = []
            for ch in SCORED_CHANNELS:
                base_value = base.dims.get(ch)
                values = [float(p.dims[ch]) for p in probes if p.dims.get(ch) is not None]
                if base_value is None or not values:
                    missing.append(ch)
                    continue
                deltas[ch] = mean(values) - float(base_value)

            phenotype = {ch: deltas[ch] for ch in PHENOTYPE_DIMS if ch in deltas}
            ordered = sorted(phenotype.items(), key=lambda kv: (kv[1], kv[0]))
            dominant = ordered[0][0] if ordered else ""
            concentration = (ordered[1][1] - ordered[0][1]) if len(ordered) > 1 else 0.0
            report.rows.append(
                MechanismProfileRow(
                    primitive_id=pid,
                    dose=dose,
                    deltas=deltas,
                    dominant_channel=dominant,
                    concentration=concentration,
                    n_replicates=len(probes),
                    channels_missing=tuple(missing),
                )
            )
    if not report.rows:
        report.notes.append("没有「只扰动一个 primitive」的被试，剖面为空。")
    return report


def render_table(report: MechanismProfileReport, *, top_n: int = 3) -> str:
    """每个 primitive 的最强扰动行：主导通道 + 下降最大的前 N 个通道。"""
    lines = [
        "由测量估计的机制—维度剖面（数据驱动；数值 = 相对基线的平均 Δ，负 = 下降）",
        "=" * 104,
        f"{'primitive':<34}{'dose':>6}  {'主导通道':<34}{'Δ主导':>8}{'集中度':>8}{'n':>4}  其它下降通道",
        "-" * 104,
    ]
    for row in report.strongest_rows():
        dominant_delta = row.deltas.get(row.dominant_channel, 0.0)
        others = sorted(
            ((ch, d) for ch, d in row.deltas.items()
             if ch != row.dominant_channel and ch in PHENOTYPE_DIMS and d < 0),
            key=lambda kv: kv[1],
        )[:top_n]
        others_text = "、".join(f"{channel_label(ch)}{d:+.2f}" for ch, d in others) or "—"
        label = channel_label(row.dominant_channel) if row.dominant_channel else "—"
        lines.append(
            f"{row.primitive_id:<34}{row.dose:>6.2f}  {label:<34}"
            f"{dominant_delta:>+8.2f}{row.concentration:>+8.2f}{row.n_replicates:>4}  {others_text}"
        )
    for note in report.notes:
        lines.append("")
        lines.append(f"注：{note}")
    return "\n".join(lines)
