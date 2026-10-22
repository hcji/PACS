"""计算亚型（FR-13）：让模型自己产生亚型，且**由机制定义而非由症状定义**。

做法：对表型向量 `B` 做 k-means（纯标准库实现），亚型数由轮廓系数自动选择；
每个亚型的报告形式是它的**机制签名**（各 primitive 的平均完整度），而不是症状清单。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..phenotype import PHENOTYPE_DIMS
from ..phenotype import PhenotypeProfile

IMPAIRMENT_THRESHOLD = 0.85   # 平均完整度低于此值 → 该 primitive 视为受损（写入机制签名）


@dataclass
class Subtype:
    subtype_id: str
    member_theta_ids: list[str]
    size: int
    centroid: dict[str, float]
    mechanism_signature: dict[str, float]
    impaired_primitives: list[str]
    deviant_dims: list[str]

    def to_dict(self) -> dict:
        return {
            "subtype_id": self.subtype_id,
            "size": self.size,
            "member_theta_ids": self.member_theta_ids,
            "centroid": {k: round(v, 4) for k, v in self.centroid.items()},
            "mechanism_signature": {k: round(v, 4) for k, v in self.mechanism_signature.items()},
            "impaired_primitives": self.impaired_primitives,
            "deviant_dims": self.deviant_dims,
        }


def _dim_list() -> list[str]:
    return list(PHENOTYPE_DIMS)


def _vector(profile: PhenotypeProfile, dims: list[str], imputation: dict[str, float]) -> list[float]:
    out: list[float] = []
    for dim in dims:
        value = profile.dims.get(dim)
        out.append(imputation[dim] if value is None else float(value))
    return out


def _imputation(profiles: list[PhenotypeProfile], dims: list[str]) -> dict[str, float]:
    """缺失维度的插补值 = 该维度在所有被试上的均值（再退化到 0.5）。"""
    out: dict[str, float] = {}
    for dim in dims:
        values = [p.dims[dim] for p in profiles if p.dims.get(dim) is not None]
        out[dim] = sum(values) / len(values) if values else 0.5
    return out


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans(
    vectors: list[list[float]], k: int, *, seed: int = 42, iterations: int = 100
) -> tuple[list[int], list[list[float]]]:
    """确定性 k-means（k-means++ 初始化，显式 seed）。返回 `(标签, 质心)`。"""
    n = len(vectors)
    if n == 0:
        return [], []
    k = max(1, min(k, n))
    rng = random.Random(seed)

    # k-means++ 初始化
    centroids: list[list[float]] = [list(vectors[rng.randrange(n)])]
    while len(centroids) < k:
        d2 = []
        for v in vectors:
            best = min(_distance(v, c) ** 2 for c in centroids)
            d2.append(best)
        total = sum(d2)
        if total <= 1e-12:
            centroids.append(list(vectors[rng.randrange(n)]))
            continue
        target = rng.random() * total
        cumulative = 0.0
        picked = n - 1
        for idx, value in enumerate(d2):
            cumulative += value
            if cumulative >= target:
                picked = idx
                break
        centroids.append(list(vectors[picked]))

    labels = [0] * n
    for _ in range(iterations):
        changed = False
        for idx, v in enumerate(vectors):
            best_idx = min(range(k), key=lambda c: _distance(v, centroids[c]))
            if labels[idx] != best_idx:
                labels[idx] = best_idx
                changed = True
        for c in range(k):
            members = [vectors[i] for i in range(n) if labels[i] == c]
            if members:
                centroids[c] = [
                    sum(m[d] for m in members) / len(members) for d in range(len(vectors[0]))
                ]
        if not changed:
            break
    return labels, centroids


def silhouette(vectors: list[list[float]], labels: list[int]) -> float:
    """平均轮廓系数 ∈ [-1, 1]。"""
    n = len(vectors)
    if n < 3 or len(set(labels)) < 2:
        return -1.0
    scores: list[float] = []
    for i in range(n):
        same = [j for j in range(n) if j != i and labels[j] == labels[i]]
        if not same:
            scores.append(0.0)
            continue
        a = sum(_distance(vectors[i], vectors[j]) for j in same) / len(same)
        b = float("inf")
        for other in set(labels) - {labels[i]}:
            group = [j for j in range(n) if labels[j] == other]
            if not group:
                continue
            mean_dist = sum(_distance(vectors[i], vectors[j]) for j in group) / len(group)
            b = min(b, mean_dist)
        if b == float("inf"):
            scores.append(0.0)
            continue
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 1e-12 else 0.0)
    return sum(scores) / len(scores)


def choose_k(
    profiles: list[PhenotypeProfile], k_values: tuple[int, ...] = (2, 3, 4, 5), *, seed: int = 42
) -> tuple[int, dict[int, float]]:
    """按轮廓系数选择亚型数（不人工指定 k）。"""
    dims = _dim_list()
    imputation = _imputation(profiles, dims)
    vectors = [_vector(p, dims, imputation) for p in profiles]
    scores: dict[int, float] = {}
    for k in k_values:
        if k >= len(vectors):
            continue
        labels, _ = kmeans(vectors, k, seed=seed)
        scores[k] = round(silhouette(vectors, labels), 4)
    if not scores:
        return 1, {1: 0.0}
    best = max(scores, key=lambda k: (scores[k], -k))
    return best, scores


def cluster_profiles(
    profiles: list[PhenotypeProfile],
    *,
    k: int | None = None,
    seed: int = 42,
    baseline_dims: dict[str, float | None] | None = None,
) -> tuple[list[Subtype], dict[int, float]]:
    """对表型做聚类，返回亚型列表与各 k 的轮廓系数。

    `k=None` 时由 `choose_k` 自动决定。
    """
    usable = [p for p in profiles if p.generic_degradation <= 0]
    if len(usable) < 3:
        return [], {}
    dims = _dim_list()
    imputation = _imputation(usable, dims)
    vectors = [_vector(p, dims, imputation) for p in usable]

    scores: dict[int, float] = {}
    if k is None:
        k, scores = choose_k(usable, seed=seed)
    k = max(1, min(k, len(usable)))
    labels, centroids = kmeans(vectors, k, seed=seed)
    if not scores:
        scores[k] = round(silhouette(vectors, labels), 4)

    # 基线（用于判断哪些维度「偏离」）
    if baseline_dims is None:
        baseline_dims = {dim: imputation[dim] for dim in dims}

    subtypes: list[Subtype] = []
    for c in range(k):
        members = [usable[i] for i in range(len(usable)) if labels[i] == c]
        if not members:
            continue
        # 机制签名：各 primitive 的平均完整度
        primitives = sorted({pid for m in members for pid in m.theta})
        signature = {
            pid: sum(m.theta.get(pid, 1.0) for m in members) / len(members)
            for pid in primitives
        }
        impaired = sorted(pid for pid, level in signature.items() if level < IMPAIRMENT_THRESHOLD)
        centroid = {
            dim: sum(vectors[i][d] for i in range(len(usable)) if labels[i] == c)
            / len(members)
            for d, dim in enumerate(dims)
        }
        deviant = sorted(
            (
                dim
                for dim in dims
                if baseline_dims.get(dim) is not None
                and abs(centroid[dim] - float(baseline_dims[dim])) > 0.10
            ),
            key=lambda dim: -abs(centroid[dim] - float(baseline_dims[dim] or 0.0)),
        )
        subtypes.append(
            Subtype(
                subtype_id=f"S{c + 1}",
                member_theta_ids=sorted(m.theta_id for m in members),
                size=len(members),
                centroid={dim: round(centroid[dim], 4) for dim in dims},
                mechanism_signature=signature,
                impaired_primitives=impaired,
                deviant_dims=deviant,
            )
        )
    subtypes.sort(key=lambda s: (-s.size, s.subtype_id))
    for idx, s in enumerate(subtypes, start=1):
        s.subtype_id = f"S{idx}"
    return subtypes, scores


def render_subtypes(subtypes: list[Subtype], scores: dict[int, float]) -> str:
    lines = ["计算亚型（由机制定义，而非由症状定义）", "=" * 78]
    if not subtypes:
        lines.append("样本不足（<3 个可用被试），无法聚类。")
        return "\n".join(lines)
    if scores:
        lines.append("亚型数选择依据（轮廓系数，越大越好）：")
        for k in sorted(scores):
            lines.append(f"  k={k}: {scores[k]:+.3f}")
        lines.append("")
    for s in subtypes:
        lines.append(f"{s.subtype_id}（{s.size} 个被试）")
        lines.append(
            "  机制签名："
            + (
                "、".join(f"{pid}={lv:.2f}" for pid, lv in sorted(s.mechanism_signature.items()))
                if s.mechanism_signature
                else "（无）"
            )
        )
        lines.append(f"  受损 primitive：{'、'.join(s.impaired_primitives) or '（无）'}")
        lines.append(
            "  偏离维度："
            + (
                "、".join(f"{d}={s.centroid.get(d, float('nan')):.2f}" for d in s.deviant_dims)
                if s.deviant_dims
                else "（无）"
            )
        )
        lines.append("")
    return "\n".join(lines)
