"""纯标准库统计工具（FR-16）。

刻意不依赖 numpy/scipy：所有算法自行实现，关键算法都有已知算例的单元测试。
主报告量是**效应量 + 置信区间**，而不是 p 值（Plan.txt 第二十七阶段）。
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# 基础描述统计
# ---------------------------------------------------------------------------


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    """样本标准差（n − 1）。n < 2 时返回 0.0。"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def variance(xs: Sequence[float]) -> float:
    return stdev(xs) ** 2


def _rank(values: Sequence[float]) -> list[float]:
    """平均秩（处理并列）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return 0.0
    mx = mean([p[0] for p in pairs])
    my = mean([p[1] for p in pairs])
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return cov / math.sqrt(vx * vy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return 0.0
    rx = _rank([p[0] for p in pairs])
    ry = _rank([p[1] for p in pairs])
    return pearson(rx, ry)


# ---------------------------------------------------------------------------
# 效应量
# ---------------------------------------------------------------------------


def cohen_d(treatment: Sequence[float], control: Sequence[float]) -> float:
    """Cohen's d（合并标准差）。方向：treatment 低于 control → d < 0。

    退化情形：两组内部方差都为 0。此时若均值有别，理论上 d 发散；这里返回带符号的
    饱和值 ±10（避免 inf 破坏 JSON 序列化），均值相同才返回 0。
    """
    n1, n2 = len(treatment), len(control)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = variance(treatment), variance(control)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 1e-12:
        diff = mean(treatment) - mean(control)
        if abs(diff) <= 1e-12:
            return 0.0
        return 10.0 if diff > 0 else -10.0
    return (mean(treatment) - mean(control)) / math.sqrt(pooled)


def hedges_g(treatment: Sequence[float], control: Sequence[float]) -> float:
    """小样本校正后的效应量。"""
    n1, n2 = len(treatment), len(control)
    if n1 + n2 < 4:
        return cohen_d(treatment, control)
    d = cohen_d(treatment, control)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return d * correction


def effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "可忽略"
    if a < 0.5:
        return "小"
    if a < 0.8:
        return "中"
    return "大"


# ---------------------------------------------------------------------------
# 重采样：bootstrap 置信区间 / 置换检验
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """百分位 bootstrap 置信区间。确定性（显式 seed）。"""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        single = clean[0] if clean else float("nan")
        return single, single
    rng = random.Random(seed)
    n = len(clean)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo_idx = max(0, min(len(stats) - 1, int(math.floor((alpha / 2) * len(stats)))))
    hi_idx = max(0, min(len(stats) - 1, int(math.ceil((1 - alpha / 2) * len(stats))) - 1))
    return stats[lo_idx], stats[hi_idx]


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    """双侧置换检验的 p 值（用均值差作为统计量）。"""
    a_clean = [v for v in a if v is not None]
    b_clean = [v for v in b if v is not None]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return 1.0
    observed = abs(mean(a_clean) - mean(b_clean))
    pooled = a_clean + b_clean
    n_a = len(a_clean)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        diff = abs(mean(pooled[:n_a]) - mean(pooled[n_a:]))
        if diff >= observed:
            hits += 1
    return (hits + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# 线性拟合（剂量-反应）
# ---------------------------------------------------------------------------


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    """最小二乘直线拟合。返回 `(slope, intercept, r²)`；数据不足时返回 (0, 均值, 0)。"""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 2:
        return 0.0, (mean([p[1] for p in pairs]) if pairs else float("nan")), 0.0
    mx = mean([p[0] for p in pairs])
    my = mean([p[1] for p in pairs])
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    if vx <= 1e-12:
        return 0.0, my, 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    slope = cov / vx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for _, y in pairs)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pairs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, intercept, max(0.0, min(1.0, r2))


def is_monotonic(ys: Sequence[float], tol: float = 0.02) -> tuple[bool, str]:
    """判断序列是否单调（允许 `tol` 的抖动）。返回 `(是否单调, 方向)`。"""
    clean = [y for y in ys if y is not None]
    if len(clean) < 2:
        return False, "unknown"
    diffs = [clean[i + 1] - clean[i] for i in range(len(clean) - 1)]
    significant = [d for d in diffs if abs(d) > tol]
    if not significant:
        return True, "flat"
    if all(d > 0 for d in significant):
        return True, "increasing"
    if all(d < 0 for d in significant):
        return True, "decreasing"
    return False, "non-monotonic"


# ---------------------------------------------------------------------------
# 共轭后验（Normal-Normal）
# ---------------------------------------------------------------------------


def normal_conjugate(
    mu0: float,
    sigma0: float,
    observations: Iterable[float],
    sigma: float,
) -> tuple[float, float]:
    """Normal-Normal 共轭后验。返回 `(后验均值, 后验标准差)`。

    先验 N(mu0, sigma0²)，似然 N(θ, sigma²)。用于把观测逐条并入先验。
    """
    obs = list(observations)
    if not obs or sigma <= 0 or sigma0 <= 0:
        return mu0, sigma0
    precision = 1.0 / (sigma0 ** 2)
    weighted = mu0 * precision
    for x in obs:
        precision += 1.0 / (sigma ** 2)
        weighted += x / (sigma ** 2)
    posterior_var = 1.0 / precision
    return weighted * posterior_var, math.sqrt(posterior_var)


def gaussian_log_likelihood(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        sigma = 1e-6
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))
