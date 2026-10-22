"""个体级机制拟合与贝叶斯机制比较（FR-14、FR-16）。

Plan.txt 第十七、二十三阶段的落地：
- 不做 `LLM score → ASD/TD` 分类（那会退化成 AI 诊断，对机制没有帮助）；
- 做的是 **`P(M_i | D)`** —— 这个人更符合哪一种计算机制。

做法：每个候选机制有一个**预期的表型向量**（由该机制扰动的被试平均得到）；对一个体的表型
`D`，在各维度上以高斯似然计算 `P(D | M_i)`，再配均匀先验得到后验 `P(M_i | D)`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..phenotype import PHENOTYPE_DIMS
from ..phenotype import PhenotypeProfile
from .stats import gaussian_log_likelihood

DEFAULT_SIGMA = 0.15   # 观测噪声（表型维度的标准差）


@dataclass
class MechanismModel:
    """一个候选机制的预期表型。"""

    primitive_id: str
    dose: float
    expected: dict[str, float]
    n_observations: int = 1

    @property
    def label(self) -> str:
        return f"M[{self.primitive_id}@{self.dose:.1f}]"


@dataclass
class MechanismPosterior:
    """个体级的机制后验。"""

    posterior: dict[str, float]
    log_likelihoods: dict[str, float]
    best: str
    entropy: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "posterior": {k: round(v, 4) for k, v in self.posterior.items()},
            "log_likelihoods": {k: round(v, 3) for k, v in self.log_likelihoods.items()},
            "best": self.best,
            "entropy": round(self.entropy, 4),
            "notes": self.notes,
        }


def build_mechanism_models(
    profiles: list[PhenotypeProfile],
    *,
    model_id: str | None = None,
    dose: float | None = None,
) -> list[MechanismModel]:
    """由单因素扰动剖面构造候选机制模型（每个 primitive 一个）。"""
    grouped: dict[str, list[PhenotypeProfile]] = {}
    for p in profiles:
        if model_id and p.model_id != model_id:
            continue
        if p.generic_degradation > 0:
            continue
        active = {k: v for k, v in p.theta.items() if v < 0.999}
        if len(active) != 1:
            continue
        pid, lvl = next(iter(active.items()))
        if dose is not None and abs(lvl - dose) > 1e-6:
            continue
        # 同名 primitive 的不同 dose 视为不同机制模型
        grouped.setdefault(f"{pid}@{lvl:.2f}", []).append(p)

    models: list[MechanismModel] = []
    for key, group in sorted(grouped.items()):
        pid, lvl_text = key.rsplit("@", 1)
        expected: dict[str, float] = {}
        for dim in PHENOTYPE_DIMS:
            values = [p.dims[dim] for p in group if p.dims.get(dim) is not None]
            if values:
                expected[dim] = sum(values) / len(values)
        models.append(
            MechanismModel(
                primitive_id=pid,
                dose=float(lvl_text),
                expected=expected,
                n_observations=len(group),
            )
        )
    return models


def mechanism_posterior(
    observed: dict[str, float | None],
    models: list[MechanismModel],
    *,
    sigma: float = DEFAULT_SIGMA,
    prior: dict[str, float] | None = None,
) -> MechanismPosterior:
    """计算 `P(M_i | D)`（各维度条件独立的高斯似然 + 均匀/给定先验）。"""
    if not models:
        return MechanismPosterior(posterior={}, log_likelihoods={}, best="", entropy=0.0)

    log_likelihoods: dict[str, float] = {}
    for m in models:
        total = 0.0
        used = 0
        for dim in PHENOTYPE_DIMS:
            value = observed.get(dim)
            if value is None or dim not in m.expected:
                continue
            # 观测数少时按标准误收缩，避免把 n=1 当作零噪声
            local_sigma = max(1e-3, sigma)
            total += gaussian_log_likelihood(float(value), m.expected[dim], local_sigma)
            used += 1
        log_likelihoods[m.label] = total if used else float("-inf")

    # 后验（数值稳定：先减最大值）
    labels = list(log_likelihoods)
    finite = [v for v in log_likelihoods.values() if math.isfinite(v)]
    if not finite:
        uniform = 1.0 / len(labels)
        return MechanismPosterior(
            posterior={label: uniform for label in labels},
            log_likelihoods=log_likelihoods,
            best=labels[0],
            entropy=math.log(len(labels)),
            notes=["所有机制的似然都不可计算，退化为均匀后验"],
        )

    max_ll = max(finite)
    weights: dict[str, float] = {}
    for label in labels:
        ll = log_likelihoods[label]
        if not math.isfinite(ll):
            weights[label] = 0.0
            continue
        w = math.exp(ll - max_ll)
        if prior and label in prior:
            w *= prior[label]
        weights[label] = w
    total = sum(weights.values())
    posterior = (
        {label: w / total for label, w in weights.items()}
        if total > 0
        else {label: 1.0 / len(labels) for label in labels}
    )
    entropy = -sum(p * math.log(p) for p in posterior.values() if p > 0)
    best = max(posterior, key=lambda k: posterior[k])
    notes: list[str] = []
    if entropy > 0.8 * math.log(len(labels)):
        notes.append("后验接近均匀 —— 这些机制在当前任务集下无法区分，需加入更多维度")
    return MechanismPosterior(
        posterior=posterior, log_likelihoods=log_likelihoods, best=best, entropy=entropy, notes=notes
    )


def fit_individual(
    target: PhenotypeProfile,
    models: list[MechanismModel],
    *,
    sigma: float = DEFAULT_SIGMA,
) -> MechanismPosterior:
    """对单个个体做机制拟合。"""
    return mechanism_posterior(target.dims, models, sigma=sigma)


def pair_log_bayes_factor(result: MechanismPosterior, a: str, b: str) -> float:
    """两个机制的 log Bayes factor（正 = 支持 a）。"""
    la = result.log_likelihoods.get(a, float("-inf"))
    lb = result.log_likelihoods.get(b, float("-inf"))
    if not (math.isfinite(la) and math.isfinite(lb)):
        return float("nan")
    return la - lb


def render_posterior(result: MechanismPosterior, title: str = "机制后验") -> str:
    lines = [f"{title}（P(M_i | D)）", "=" * 60]
    if not result.posterior:
        lines.append("无候选机制。")
        return "\n".join(lines)
    for label, prob in sorted(result.posterior.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(prob * 40))
        ll = result.log_likelihoods.get(label, float("nan"))
        lines.append(f"  {label:<34}{prob:>6.3f}  {bar:<40} logL={ll:>9.2f}")
    lines.append("")
    lines.append(f"最符合机制：{result.best}；后验熵 = {result.entropy:.3f}")
    for note in result.notes:
        lines.append(f"注：{note}")
    return "\n".join(lines)
