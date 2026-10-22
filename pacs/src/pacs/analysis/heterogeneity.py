"""机制异质性（FR-12.2~12.5）：**同表型不同机制**，以及区分性 / 阴性预测。

这是本项目最核心的结果部分（Plan.txt 第九、十一、十六阶段）：

1. **收敛表型**：找出「哪些不同的 primitive 扰动**都能**产生同一表型终点」；
2. **特征差异**：每个候选机制在**其它维度**上的指纹不同 —— 这正是区分它们的依据；
3. **区分性预测**：给出「机制 A 与机制 B 应该在哪些行为上不同」；
4. **最少任务集**：找出能区分全部候选机制的**最小**任务维度集合（Plan.txt 目标形态：
   只需 3 个任务而不是 30 个）；
5. **阴性预测**：每个机制还必须预测「**什么应该正常**」—— 只预测异常是不够的。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..phenotype import CONTROL_DIM, PHENOTYPE_DIMS
from ..phenotype import PhenotypeProfile, aggregate_baseline

DEFAULT_THRESHOLD = 0.70      # 表型终点被判为「受损」的阈值（chance-corrected 尺度；完整心智 ≈ 0.91）
# 阴性预测容差：必须大于**抽样噪声下限**。每个维度只有 7 道题，chance-corrected 得分下
# 单题翻转即造成约 0.07−0.13 的跳动（3 选项题：一道错 → 该维度掉约 0.12）。容差若设得比
# 噪声还小，就会把纯噪声误报成「该维度也变了」。这里取 0.15，并在报告中说明该限制。
NEGATIVE_TOLERANCE = 0.15
SEPARATION_MARGIN = 0.12      # 两机制在某维度上差异超过此值即视为可分


@dataclass
class MechanismCandidate:
    """一个能产生该表型终点的候选机制。

    机制 = 一组 primitive 的组合（可以是单个，也可以是多成分）。这更贴近真实情形：
    Plan.txt 的核心主张正是「高阶能力的损伤通常源于多个基础过程的组合异常」，
    因此把「机制」限定为单因素扰动会人为缩小候选空间。
    """

    members: tuple[str, ...]
    dose: float
    endpoint_dim: str
    endpoint_value: float
    dims: dict[str, float | None]
    feature_delta: dict[str, float]

    @property
    def primitive_id(self) -> str:
        """单成员时返回 primitive id；多成员时返回 `a+b` 形式（兼容旧调用）。"""
        return self.members[0] if len(self.members) == 1 else "+".join(self.members)

    @property
    def label(self) -> str:
        return f"{self.primitive_id}@{self.dose:.1f}"

    @property
    def is_single(self) -> bool:
        return len(self.members) == 1

    def to_dict(self) -> dict:
        return {
            "members": list(self.members),
            "primitive_id": self.primitive_id,
            "dose": self.dose,
            "endpoint_dim": self.endpoint_dim,
            "endpoint_value": self.endpoint_value,
            "feature_delta": self.feature_delta,
        }


@dataclass
class ConvergenceResult:
    """一个表型终点的机制异质性分析结果。"""

    endpoint_dim: str
    threshold: float
    baseline_dims: dict[str, float | None]
    candidates: list[MechanismCandidate]
    distinguishing_dims: list[tuple[str, float]] = field(default_factory=list)
    minimal_task_set: list[str] = field(default_factory=list)
    negative_predictions: dict[str, dict[str, float]] = field(default_factory=dict)
    pair_separation: dict[str, tuple[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def is_heterogeneous(self) -> bool:
        """是否真的存在机制异质性（≥2 个可区分的候选机制）。"""
        return len(self.candidates) >= 2 and bool(self.minimal_task_set)

    def to_dict(self) -> dict:
        return {
            "endpoint_dim": self.endpoint_dim,
            "threshold": self.threshold,
            "baseline_dims": self.baseline_dims,
            "candidates": [c.to_dict() for c in self.candidates],
            "n_candidates": self.n_candidates,
            "distinguishing_dims": self.distinguishing_dims,
            "minimal_task_set": self.minimal_task_set,
            "negative_predictions": self.negative_predictions,
            "pair_separation": {k: list(v) for k, v in self.pair_separation.items()},
            "is_heterogeneous": self.is_heterogeneous,
            "notes": self.notes,
        }


def _baseline(profiles: list[PhenotypeProfile], model_id: str | None) -> PhenotypeProfile | None:
    """基线 = 所有完整心智剖面的平均（降低单次抽样噪声）。"""
    return aggregate_baseline(profiles, model_id)


def _feature_delta(
    dims: dict[str, float | None], baseline: dict[str, float | None]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for dim in PHENOTYPE_DIMS:
        value = dims.get(dim)
        base = baseline.get(dim)
        if value is None or base is None:
            continue
        out[dim] = round(float(value) - float(base), 4)
    return out


def find_convergence(
    profiles: list[PhenotypeProfile],
    endpoint_dim: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model_id: str | None = None,
    dose: float | None = None,
    include_pairs: bool = True,
    max_candidates: int = 12,
) -> ConvergenceResult | None:
    """找出所有能把 `endpoint_dim` 压到阈值以下的候选机制 = 「同表型不同机制」的候选集。

    候选机制同时考虑：
    - **单因素**扰动（单个 primitive 足以造成该表型终点）；
    - **多因素**组合扰动（≥2 个 primitive 的组合异常造成同一终点）—— 这更贴近真实病理。

    对每个候选集合取**其在该集合下观测到的最差表型**（而非固定取最强 dose），
    以免单次抽样的噪声把某个 dose 的翻转当成「更轻的损伤」。
    """
    if endpoint_dim not in PHENOTYPE_DIMS:
        raise KeyError(f"未知表型维度：{endpoint_dim}")
    baseline = _baseline(profiles, model_id)
    if baseline is None:
        return None

    # 按「激活的 primitive 集合」分组
    grouped: dict[frozenset[str], list[PhenotypeProfile]] = {}
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        if p.generic_degradation > 0:
            continue
        active = frozenset(k for k, v in p.theta.items() if v < 0.999)
        if not active:
            continue
        if len(active) == 1 or (include_pairs and len(active) == 2):
            grouped.setdefault(active, []).append(p)

    candidates: list[MechanismCandidate] = []
    for members, group in grouped.items():
        if dose is not None:
            group = [p for p in group if p.active_levels() and abs(p.active_levels()[0] - dose) < 1e-6]
        if not group:
            continue
        # 取该集合下**最差**的表型观测（含各 dose 与重复）
        probe = min(
            (p for p in group if p.dims.get(endpoint_dim) is not None),
            key=lambda p: float(p.dims[endpoint_dim]),
            default=None,
        )
        if probe is None:
            continue
        value = float(probe.dims[endpoint_dim])
        if value > threshold:
            continue
        observed_dose = probe.active_levels()[0] if probe.active_levels() else 1.0
        candidates.append(
            MechanismCandidate(
                members=tuple(sorted(members)),
                dose=round(observed_dose, 4),
                endpoint_dim=endpoint_dim,
                endpoint_value=round(value, 4),
                dims=dict(probe.dims),
                feature_delta=_feature_delta(probe.dims, baseline.dims),
            )
        )

    # 优先展示单因素（更简约的机制），再按终点损伤程度排序
    candidates.sort(key=lambda c: (len(c.members), c.endpoint_value, c.primitive_id))
    total_found = len(candidates)
    candidates = candidates[:max_candidates]

    result = ConvergenceResult(
        endpoint_dim=endpoint_dim,
        threshold=threshold,
        baseline_dims=dict(baseline.dims),
        candidates=candidates,
    )
    if total_found > len(candidates):
        result.notes.append(
            f"共发现 {total_found} 个候选机制，报告中按简约性（单因素优先）与损伤程度"
            f"列出前 {len(candidates)} 个。"
        )

    if len(candidates) < 2:
        result.notes.append(
            f"仅 {len(candidates)} 个候选机制达到「{endpoint_dim} 受损」阈值 {threshold}；"
            "需要 ≥2 个才能主张机制异质性（C3 标准）。可放宽阈值或扩大扰动 dose 范围。"
        )
        return result

    # ---- 特征差异：在**非终点**维度上，候选机制之间差异越大越能区分 ----
    # 注意：**排除对照通道**（general_cognition）。它按构造不含任何机制信号，
    # 其"候选间差异"纯粹是抽样噪声；若允许它参与判别，就会把噪声当成机制区分依据。
    excluded_from_separation = [endpoint_dim, CONTROL_DIM]
    dist: list[tuple[str, float]] = []
    for dim in PHENOTYPE_DIMS:
        if dim in excluded_from_separation:
            continue
        values = [c.feature_delta.get(dim) for c in candidates]
        clean = [v for v in values if v is not None]
        if len(clean) < 2:
            continue
        spread = max(clean) - min(clean)
        if spread < SEPARATION_MARGIN:
            continue   # 差异未超过噪声/可分阈值，不构成判别依据
        dist.append((dim, round(spread, 4)))
    dist.sort(key=lambda pair: (-pair[1], pair[0]))
    result.distinguishing_dims = dist
    result.notes.append(
        f"判别维度已排除「{endpoint_dim}」（各机制在此按构造相同）与「{CONTROL_DIM}」"
        "（对照组通道，按构造不含机制信号，其差异为噪声）。"
    )

    # ---- 两两可分性：每对机制由哪个维度最能把它们分开 ----
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            best_dim, best_gap = "", 0.0
            for dim, _ in dist:
                va, vb = a.feature_delta.get(dim), b.feature_delta.get(dim)
                if va is None or vb is None:
                    continue
                gap = abs(va - vb)
                if gap > best_gap:
                    best_dim, best_gap = dim, gap
            result.pair_separation[f"{a.primitive_id}|{b.primitive_id}"] = (
                best_dim or endpoint_dim,
                round(best_gap, 4),
            )

    # ---- 最少区分任务集：贪心覆盖所有机制对，直到每对都被某一维度分开 ----
    result.minimal_task_set = minimal_discriminating_task_set(candidates, dist)

    # ---- 阴性预测：每个机制**不应该**改变的维度 ----
    for c in candidates:
        normal = {
            dim: round(float(c.dims.get(dim)), 4)
            for dim in PHENOTYPE_DIMS
            if dim != endpoint_dim
            and c.dims.get(dim) is not None
            and abs(c.feature_delta.get(dim, 0.0)) <= NEGATIVE_TOLERANCE
        }
        result.negative_predictions[c.primitive_id] = normal

    if not result.minimal_task_set:
        result.notes.append(
            "候选机制在所有维度上几乎无法区分（差异小于 margin）→ 这些 primitive 可能只是"
            "同一个潜在过程的不同入口，不能声称机制异质性。"
        )
    return result


def minimal_discriminating_task_set(
    candidates: list[MechanismCandidate],
    distinguishing_dims: list[tuple[str, float]],
    margin: float = SEPARATION_MARGIN,
) -> list[str]:
    """找出能区分全部候选机制的**最小**维度集合（贪心集合覆盖）。

    返回按区分力降序挑选的维度列表 —— 这就是「人类实验只需要测这几个任务」的答案。
    """
    if len(candidates) < 2:
        return []
    pairs = [
        (a.label, b.label)
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
    ]
    remaining = set(pairs)
    chosen: list[str] = []
    for dim, _spread in distinguishing_dims:
        if not remaining:
            break
        separates = {
            (a, b)
            for (a, b) in remaining
            if _gap_on_dim(candidates, a, b, dim) >= margin
        }
        if separates:
            chosen.append(dim)
            remaining -= separates
    return chosen


def _gap_on_dim(
    candidates: list[MechanismCandidate], label_a: str, label_b: str, dim: str
) -> float:
    by_label = {c.label: c for c in candidates}
    va = by_label[label_a].feature_delta.get(dim)
    vb = by_label[label_b].feature_delta.get(dim)
    if va is None or vb is None:
        return 0.0
    return abs(va - vb)


@dataclass
class InteractionResult:
    """组合扰动的交互效应（FR-5 要求 4 / 可加性检验）。"""

    primitive_a: str
    primitive_b: str
    dose: float
    single_a: float
    single_b: float
    combined: float
    baseline: float
    expected_additive: float
    interaction: float
    kind: str


def evaluate_interactions(
    profiles: list[PhenotypeProfile],
    endpoint_dim: str,
    *,
    model_id: str | None = None,
    dose: float | None = None,
) -> list[InteractionResult]:
    """由二阶组合扰动实测交互效应，判定可加 / 超加（协同）/ 次加（拮抗）。

    `interaction = combined − expected_additive`；为负 = 比可加更差 = 超加性损伤。
    """
    baseline = _baseline(profiles, model_id)
    if baseline is None:
        return []
    base_value = baseline.dims.get(endpoint_dim)
    if base_value is None:
        return []

    singles: dict[str, tuple[float, float]] = {}
    pairs: dict[tuple[str, str, float], float] = {}
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        if p.generic_degradation > 0:
            continue
        value = p.dims.get(endpoint_dim)
        if value is None:
            continue
        active = {k: v for k, v in p.theta.items() if v < 0.999}
        if len(active) == 1:
            pid, lvl = next(iter(active.items()))
            singles[pid] = (float(lvl), float(value))
        elif len(active) == 2:
            keys = sorted(active)
            levels = [round(float(active[k]), 4) for k in keys]
            if len(set(levels)) != 1:
                continue   # 只处理等强度的组合，便于与单因素同剂量比较
            pairs[(keys[0], keys[1], levels[0])] = float(value)

    results: list[InteractionResult] = []
    from ..brain.perturb import interaction_sign

    for (a, b, lvl), combined in sorted(pairs.items()):
        if a not in singles or b not in singles:
            continue
        la, va = singles[a]
        lb, vb = singles[b]
        # 只在单因素与组合处于**同一 dose** 时才做可加性比较，否则量纲不可比
        if abs(la - lvl) > 1e-6 or abs(lb - lvl) > 1e-6:
            continue
        expected = va + vb - float(base_value)
        interaction = combined - expected
        results.append(
            InteractionResult(
                primitive_a=a,
                primitive_b=b,
                dose=lvl,
                single_a=round(va, 4),
                single_b=round(vb, 4),
                combined=round(combined, 4),
                baseline=round(float(base_value), 4),
                expected_additive=round(expected, 4),
                interaction=round(interaction, 4),
                kind=interaction_sign(va, vb, combined, float(base_value)),
            )
        )
    return results


def render_convergence(result: ConvergenceResult) -> str:
    """渲染机制异质性结论（报告的核心章节）。"""
    lines = [
        f"收敛表型：{result.endpoint_dim}（阈值 ≤ {result.threshold}）",
        "=" * 78,
    ]
    if not result.candidates:
        lines.append("无候选机制达到阈值。")
        return "\n".join(lines)

    lines.append(f"基线值：{result.baseline_dims.get(result.endpoint_dim)}")
    lines.append(f"候选机制（{result.n_candidates} 个）：")
    for c in result.candidates:
        lines.append(
            f"  - {c.primitive_id:<32} @dose {c.dose:.2f} → {result.endpoint_dim} = {c.endpoint_value:.2f}"
        )
    lines.append("")
    lines.append("特征差异（区分机制的关键维度，按区分力降序）：")
    if result.distinguishing_dims:
        for dim, spread in result.distinguishing_dims:
            lines.append(f"  - {dim:<26} 候选间差异幅度 {spread:.2f}")
    else:
        lines.append("  （无可用维度）")
    lines.append("")
    lines.append(
        "最少区分任务集（人类实验只需测这些）："
        + ("、".join(result.minimal_task_set) if result.minimal_task_set else "（无法区分）")
    )
    lines.append("")
    lines.append("两两可分性（机制对 → 最能分开它们的维度）：")
    for pair, (dim, gap) in sorted(result.pair_separation.items()):
        lines.append(f"  - {pair:<58} → {dim}（差 {gap:.2f}）")
    lines.append("")
    lines.append("阴性预测（这些机制**不应**改变的维度 —— 「什么应该正常」）：")
    for pid, normal in sorted(result.negative_predictions.items()):
        if normal:
            desc = "、".join(f"{d}={v:.2f}" for d, v in normal.items())
            lines.append(f"  - {pid:<32} 应保持正常：{desc}")
        else:
            lines.append(f"  - {pid:<32} （无维度落在容差内 → 该机制影响面较广，需复核特异性）")
    for note in result.notes:
        lines.append("")
        lines.append(f"注：{note}")
    return "\n".join(lines)
