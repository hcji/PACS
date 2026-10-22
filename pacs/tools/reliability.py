"""可靠性统计：重复测量之间到底有多一致（Fig.1 系统可靠性实验的度量层）。

**为什么新写这个模块**：仓库里原本**没有任何可靠性统计** —— 全仓库 grep
`ICC|Cronbach|Bland|intraclass|test-retest|coefficient of variation|kappa` 零命中。
`analysis/stats.py` 只有 `mean / stdev / variance / pearson / spearman / bootstrap_ci`
这些积木，`replicate` 字段落了库却从不用于比较。本模块在那些积木之上补齐
ICC(2,1)（含置信区间）、CV、SEM、Bland-Altman、方差分解与结论级一致度。

**零第三方依赖**：项目运行时只有标准库（不引入 numpy/scipy），因此这里自己实现
正则化不完全 beta 函数与 F 分布逆分位 —— ICC 的置信区间要用它们。数值路径用二分法，
确定性、可复现。

**方向约定**：本模块不看读数的方向（1.0 = 完好还是 1.0 = 受损都与它无关），
它只度量「同一目标重复测量之间的接近程度」。ICC 越高越可靠，CV / SEM / 一致性界限越小越可靠。

**核心数据形状**：`matrix` = 行是**目标**（被测量的实体，如某个 θ 条件或某个 θ×维度 单元格），
列是**重复测量次数**（rater / occasion）。所有行等长、无 None。NaN 与缺失值必须在进入本模块
之前剔除（上层 `matrix_from_records` 负责），因为「缺一次测量」和「测出来是 0」是两件事。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

# 复用包内已有的统计积木（pearson/spearman/stdev 等），避免重复造
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pacs.analysis.stats import (  # noqa: E402
    bootstrap_ci,
    mean,
    pearson,
    spearman,
    stdev,
)

Matrix = Sequence[Sequence[float]]
Row = Sequence[float]

#: 一致性判定用的默认容差（读数尺度是 0..1）。
#: 0.15 与包内 `analysis/specificity.py` 的 `TARGET_DROP` / `evidence` 口径同量级 ——
#: 「差 0.15 以上才算看得出区别」是这个项目里已经反复出现的判断尺度。
DEFAULT_TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# 分布工具：不完全 beta + F 分布（ICC 置信区间要用）
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """不完全 beta 函数的连分式展开（Lentz 算法，Numerical Recipes §6.4）。

    为什么用连分式而不是级数：`I_x(a,b)` 在 x 靠近 0 或 1 时级数收敛很慢，
    连分式在两端都稳定。迭代上限与收敛阈值按双精度取，够 ICC 的区间用。
    """
    max_iter = 300
    eps = 3.0e-16
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 beta 函数 `I_x(a, b)`。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    # 用能保证连分式收敛的那一侧（NR 的标准做法）
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_cdf(f: float, df1: float, df2: float) -> float:
    """F 分布累计分布函数 `P(F ≤ f; df1, df2)`。"""
    if f <= 0.0 or df1 <= 0.0 or df2 <= 0.0:
        return 0.0
    return _betai(df1 / 2.0, df2 / 2.0, df1 * f / (df1 * f + df2))


def _gammq(a: float, x: float) -> float:
    """正则化上不完全 gamma 函数 `Q(a, x)`（连分式，Lentz 算法）。"""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammp(a: float, x: float) -> float:
    """正则化下不完全 gamma 函数 `P(a, x)`（级数展开，x < a+1 时收敛快）。"""
    if x <= 0.0:
        return 0.0
    if x >= a + 1.0:
        return 1.0 - _gammq(a, x)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_cdf(x: float, df: float) -> float:
    """卡方分布累计分布函数 `P(X ≤ x; df)`。"""
    if x <= 0.0 or df <= 0.0:
        return 0.0
    return gammp(df / 2.0, x / 2.0)


@lru_cache(maxsize=512)
def chi2_quantile(p: float, df: float, tol: float = 1e-12) -> float:
    """卡方分布逆分位（二分法）。校验值：`chi2_quantile(0.95, 1) ≈ 3.8415`。"""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 必须落在 (0, 1)，收到 {p}")
    if df <= 0:
        return 0.0
    lo, hi = 0.0, max(1.0, df)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return (lo + hi) / 2.0


@lru_cache(maxsize=512)
def f_quantile(p: float, df1: float, df2: float, tol: float = 1e-12) -> float:
    """F 分布逆分位（二分法求解 `f_cdf(f) = p`）。

    没有 SciPy 就得自己求逆。先指数扩张找一个上界，再二分到相对误差小于 tol。
    校验值：`f_quantile(0.95, 5, 10) ≈ 3.3258`、`f_quantile(0.95, 2, 2) = 19.0`。

    带 `lru_cache`：ICC 的区间检验会反复用同一组自由度，缓存后模拟校验才跑得动。
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 必须落在 (0, 1)，收到 {p}")
    if df1 <= 0 or df2 <= 0:
        return float("inf")
    lo, hi = 0.0, 1.0
    while f_cdf(hi, df1, df2) < p:
        hi *= 2.0
        if hi > 1e15:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f_cdf(mid, df1, df2) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 数据造型
# ---------------------------------------------------------------------------


def _shape(matrix: Matrix) -> tuple[int, int]:
    n = len(matrix)
    k = len(matrix[0]) if n else 0
    for row in matrix:
        if len(row) != k:
            raise ValueError("matrix 的每一行必须等长（每个目标都要有同样多次重复测量）")
    return n, k


def _flat(matrix: Matrix) -> list[float]:
    return [float(x) for row in matrix for x in row]


def matrix_from_records(
    records: Sequence[dict],
    value_fn: Callable[[dict], float | None],
    *,
    target_fn: Callable[[dict], str] | None = None,
    replicate_fn: Callable[[dict], int] | None = None,
    target_key: str = "theta_id",
    replicate_key: str = "replicate",
) -> tuple[Matrix, list[str]]:
    """把「逐被试记录」摆成 `目标 × 重复` 矩阵。

    目标 = `target_key` 的取值（默认 θ 条件），列 = 该目标下按 `replicate_key` 排序的测量。
    **只保留测量次数等于最多数的目标** —— 次数不齐的目标如果硬塞进去（补 0 或补均值），
    会把「少测了一次」变成「测出来是某个值」，那是在伪造数据。被丢弃的目标会记在返回值里。

    记录形状千变万化：θ 常常嵌在 `parameters.theta_id`、重复嵌在 `condition.replicate`，
    这时默认的顶层键取不到值，必须用 `target_fn` / `replicate_fn` 显式给取值函数。

    **目标 id 解析为空一律报错，不许静默分组。** 所有记录都解析成空串时，全部目标会被合并
    成一个（或按剩下的字段塌缩），矩阵形状看着「有数」却完全错位 —— 这类错误不会自己浮出来，
    只会给出一个看起来合理的 ICC（实测过：21 个单元格塌成 7 个，ICC 从 0.8 变成 0.29）。
    所以这里选择直接失败。
    """
    if target_fn is None:
        target_fn = lambda record: str(record.get(target_key, ""))  # noqa: E731
    if replicate_fn is None:
        replicate_fn = lambda record: int(record.get(replicate_key, 0) or 0)  # noqa: E731

    grouped: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        value = value_fn(record)
        if value is None:
            continue
        target = str(target_fn(record))
        if not target.strip():
            raise ValueError(
                "目标 id 解析为空：记录形状与 target_key/target_fn 不匹配，会把不同目标混成一组。"
                "请显式传 target_fn/replicate_fn。"
            )
        grouped.setdefault(target, []).append((int(replicate_fn(record)), float(value)))

    if not grouped:
        return [], []

    counts = {target: len(pairs) for target, pairs in grouped.items()}
    full = max(counts.values())
    kept = [t for t in sorted(grouped) if counts[t] == full]
    matrix = [
        [value for _, value in sorted(grouped[target], key=lambda pair: pair[0])]
        for target in kept
    ]
    dropped = [t for t in sorted(grouped) if counts[t] != full]
    return matrix, dropped


def cell_matrix_from_records(
    records: Sequence[dict],
    dims: Sequence[str],
    value_fn: Callable[[dict, str], float | None],
    *,
    target_fn: Callable[[dict], str] | None = None,
    replicate_fn: Callable[[dict], int] | None = None,
    target_key: str = "theta_id",
    replicate_key: str = "replicate",
) -> tuple[Matrix, list[str]]:
    """把记录摆成「(θ × 维度) 单元格 × 重复」矩阵 —— Fig.1 主数字用的形状。

    `value_fn(record, dim)` 取某个记录在某维度上的读数；`target_fn(record)` 返回该记录的 θ 标识
    （**不带维度**，维度由本函数拼在后面）。每个 (θ, 维度) 单元格是一行：目标数从 3 个 θ 涨到
    21 个单元格，ICC 的置信区间才有意义。
    """
    if target_fn is None:
        target_fn = lambda record: str(record.get(target_key, ""))  # noqa: E731
    expanded: list[dict] = []
    for record in records:
        base = str(target_fn(record))
        for dim in dims:
            value = value_fn(record, dim)
            if value is None:
                continue
            expanded.append({"target": f"{base}|{dim}", "value": value, "source": record})
    return matrix_from_records(
        expanded,
        lambda wrapper: wrapper["value"],
        target_fn=lambda wrapper: wrapper["target"],
        replicate_fn=(lambda wrapper: replicate_fn(wrapper["source"])) if replicate_fn else None,
        replicate_key=replicate_key,
    )


# ---------------------------------------------------------------------------
# ICC(2,1)
# ---------------------------------------------------------------------------


@dataclass
class IccResult:
    """一次 ICC 计算的全部产物（含中间量，便于核对与诊断）。"""

    icc: float | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str
    n_targets: int
    n_raters: int
    ms_between_targets: float = 0.0
    ms_between_raters: float = 0.0
    ms_error: float = 0.0
    note: str = ""

    @property
    def reliable(self) -> bool:
        """CI 下界是否高于 0.5 —— 只是给报告一个粗口径，别当成判据。"""
        return self.ci_low is not None and self.ci_low > 0.5

    def to_dict(self) -> dict:
        return {
            "icc": _round(self.icc),
            "ci_low": _round(self.ci_low),
            "ci_high": _round(self.ci_high),
            "ci_method": self.ci_method,
            "n_targets": self.n_targets,
            "n_raters": self.n_raters,
            "ms_between_targets": _round(self.ms_between_targets, 6),
            "ms_between_raters": _round(self.ms_between_raters, 6),
            "ms_error": _round(self.ms_error, 6),
            "note": self.note,
        }


def _round(value, digits: int = 4):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def mean_squares(matrix: Matrix) -> tuple[float, float, float, int, int, int]:
    """双向随机效应方差分析的三条均方（行=目标，列=重复）。

    返回 `(MSR, MSC, MSE, df_row, df_col, df_err)`。手工可核对的最小例：
    3 目标 × 3 重复、行均值 2/5/8、列均值 4/5/6 → MSR=27、MSC=3、MSE=0。
    """
    n, k = _shape(matrix)
    grand = sum(_flat(matrix)) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ss_total = sum((float(x) - grand) ** 2 for row in matrix for x in row)
    ss_row = k * sum((m - grand) ** 2 for m in row_means)
    ss_col = n * sum((m - grand) ** 2 for m in col_means)
    # 浮点误差可能让 ss_err 略小于 0；夹到 0，否则 MSE 会变成负的再开方就出 NaN
    ss_err = max(0.0, ss_total - ss_row - ss_col)
    df_row, df_col = n - 1, k - 1
    df_err = df_row * df_col
    msr = ss_row / df_row if df_row > 0 else 0.0
    msc = ss_col / df_col if df_col > 0 else 0.0
    mse = ss_err / df_err if df_err > 0 else 0.0
    return msr, msc, mse, df_row, df_col, df_err


def icc_point(matrix: Matrix) -> float | None:
    """ICC(2,1) 点估计：双向随机效应、单次测量、绝对一致性。

    公式（Shrout & Fleiss 1979）：
        ICC = (MSR − MSE) / (MSR + (k−1)·MSE + k·(MSC − MSE)/n)

    可能为负 —— 那表示「目标之间的差异还不如重复之间的波动大」，即完全不可靠。
    这里**不夹到 0**，负值本身是要如实报告的信息。
    """
    n, k = _shape(matrix)
    if n < 2 or k < 2:
        return None
    msr, msc, mse, *_ = mean_squares(matrix)
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if abs(denom) < 1e-12:
        return None
    return (msr - mse) / denom


#: 目标数低于此值时**不出** ICC 置信区间（只给点估计）。
#: 依据：模拟标定（见 `tools/可靠性实验.md`）—— jackknife 区间在 21 目标 × 5 重复上实测
#: 覆盖率 0.91–0.93，在 6 目标上降到 0.84–0.89，在 3 目标上宽度接近整个 [0,1] 而失去意义。
#: 低于门槛时报告应改用 `sem_ci`（精确 χ² 区间）陈述精度。
ICC_CI_MIN_TARGETS = 10


def icc21(matrix: Matrix, *, level: float = 0.95) -> IccResult:
    """ICC(2,1) 点估计 + **jackknife** 置信区间。

    **点估计**（Shrout & Fleiss 1979）：
        `ICC = (MSR − MSE) / (MSR + (k−1)·MSE + k·(MSC − MSE)/n)`
    等价写法：`ICC = σ²_target / (σ²_target + σ²_rater + σ²_error)`。

    **区间：对目标（行）留一的 jackknife**。逐次删掉一个目标重算 ICC，得到 n 个伪值，再按
    jackknife 公式给标准误：`SE² = (m−1)/m · Σ(ICC₍ᵢ₎ − mean)²`，区间 = 点估计 ± z·SE。

    **为什么是 jackknife（四种写法都实测过，不要退回去）**：

    | 写法 | 21×5 实测覆盖率 | 问题 |
    |---|---|---|
    | F 上下临界值代回 ICC 表达式 | **0.000** | 两端同时落在点估计同一侧（两个临界值都 > 1） |
    | 对目标重采样的 bootstrap | ≈0 | 复制目标行，而复制行的行内偏差为零 → 人为压低误差、ICC 被系统性抬高 |
    | 方差分量最坏组合（F/χ² + 尾概率） | 0.48–0.58 | 分量强相关，「最坏组合」并不保守；覆盖率随 (n,k) 剧烈漂移 |
    | **对目标留一的 jackknife** | **0.91–0.93** | 不复制行、不假设分量独立，跨形状稳定 |

    jackknife 区间在 21 目标上**略微偏窄**（0.91–0.93 而非 0.95），这一点如实记录，
    目标数不足 `ICC_CI_MIN_TARGETS`
    时不给区间（`ci_low`/`ci_high` 为 None）。
    """
    n, k = _shape(matrix)
    if n < 2 or k < 2:
        return IccResult(None, None, None, "none", n, k, note="目标数或重复数不足 2，无法计算")
    msr, msc, mse, df_row, df_col, df_err = mean_squares(matrix)
    icc = icc_point(matrix)
    if icc is None:
        return IccResult(None, None, None, "none", n, k, note="方差分解退化，无法计算")

    if n < ICC_CI_MIN_TARGETS:
        return IccResult(
            icc=icc, ci_low=None, ci_high=None, ci_method="none",
            n_targets=n, n_raters=k,
            ms_between_targets=msr, ms_between_raters=msc, ms_error=mse,
            note=(f"目标数 {n} < {ICC_CI_MIN_TARGETS}：jackknife 区间在此规模下宽度接近整个 "
                  f"[0,1] 且实测覆盖率降到 0.8 以下，名不副实，故不给区间；"
                  f"精度请用 sem_ci 的精确区间"),
        )

    pseudo: list[float] = []
    for i in range(n):
        value = icc_point(matrix[:i] + matrix[i + 1:])
        if value is not None:
            pseudo.append(value)
    if len(pseudo) < 3:
        return IccResult(
            icc=icc, ci_low=None, ci_high=None, ci_method="none", n_targets=n, n_raters=k,
            ms_between_targets=msr, ms_between_raters=msc, ms_error=mse,
            note="留一伪值不足 3 个，无法给出 jackknife 区间",
        )
    m = len(pseudo)
    center = mean(pseudo)
    variance = (m - 1) / m * sum((value - center) ** 2 for value in pseudo)
    se = math.sqrt(max(0.0, variance))
    z = 1.959963984540054 if abs(level - 0.95) < 1e-9 else _normal_quantile(level)
    lower, upper = icc - z * se, icc + z * se
    if lower > upper:
        lower, upper = upper, lower
    return IccResult(
        icc=icc, ci_low=lower, ci_high=upper, ci_method="jackknife",
        n_targets=n, n_raters=k,
        ms_between_targets=msr, ms_between_raters=msc, ms_error=mse,
        note="jackknife 对目标留一；此规模下实测覆盖率约 0.91–0.93（略偏窄），已如实记录",
    )


# ---------------------------------------------------------------------------
# 离散度：CV / SEM / 方差分解 / Bland-Altman
# ---------------------------------------------------------------------------


def cv(values: Row) -> float | None:
    """变异系数 = 标准差 / 均值（越小越稳）。

    读数在 0..1 之间且可能接近 0，此时 CV 会爆掉（除以一个接近 0 的均值），
    所以均值绝对值小于 `1e-9` 时返回 None 而不是一个天文数字 —— 那种数放进报告只会误导。
    """
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None
    m = mean(clean)
    if abs(m) < 1e-9:
        return None
    return stdev(clean) / abs(m)


def row_cvs(matrix: Matrix) -> list[float | None]:
    """每个目标各自内部的 CV（目标的重复之间有多稳）。"""
    return [cv(row) for row in matrix]


def sem(matrix: Matrix) -> float | None:
    """测量标准误 SEM = √MSE（与读数同单位；越小越可靠）。

    含义：同一个目标重复测量的典型误差幅度。可以直接和 0.15 这个「看得出区别」的尺度对比。
    """
    n, k = _shape(matrix)
    if n < 2 or k < 2:
        return None
    return math.sqrt(max(0.0, mean_squares(matrix)[2]))


def sem_ci(matrix: Matrix, *, level: float = 0.95) -> tuple[float, float] | None:
    """SEM 的置信区间（**精确**的 χ² 区间，任意目标数都成立）。

    为什么 Fig.1 应该报这个而不只是 ICC 区间：`σ²_error` 的区间由
    `df·MSE / χ²(df)` 给出，对正态误差是**精确**的（覆盖率就是名义水平），
    不依赖任何近似组合，小样本下也不会像 ICC 区间那样崩掉。所以「这套测量有多准」
    这个陈述用它最站得住：测量标准误 SEM 及其区间，单位与读数相同，可直接与容差比较。

    区间：`[√(df·MSE/χ²_{1-α/2,df}), √(df·MSE/χ²_{α/2,df})]`。
    """
    n, k = _shape(matrix)
    if n < 2 or k < 2:
        return None
    _, _, mse, _, _, df_err = mean_squares(matrix)
    if df_err <= 0 or mse <= 0:
        return (0.0, 0.0)
    alpha = 1.0 - level
    chi2_hi = chi2_quantile(1.0 - alpha / 2.0, df_err)
    chi2_lo = chi2_quantile(alpha / 2.0, df_err)
    if chi2_hi <= 0 or chi2_lo <= 0:
        return None
    return (
        math.sqrt(df_err * mse / chi2_hi),
        math.sqrt(df_err * mse / chi2_lo),
    )


@dataclass
class VarianceComponents:
    """双向随机效应的方差分解（把总波动拆成「目标间」与「重复内」）。"""

    between_targets: float
    between_raters: float
    error: float

    @property
    def total(self) -> float:
        return self.between_targets + self.between_raters + self.error

    @property
    def target_share(self) -> float | None:
        """目标间方差占总方差的比例 —— 0.8 以上说明主要波动来自真实差异而非噪声。"""
        total = self.total
        return None if total <= 1e-12 else self.between_targets / total

    def to_dict(self) -> dict:
        return {
            "between_targets": _round(self.between_targets, 6),
            "between_raters": _round(self.between_raters, 6),
            "error": _round(self.error, 6),
            "total": _round(self.total, 6),
            "target_share": _round(self.target_share),
        }


def variance_components(matrix: Matrix) -> VarianceComponents | None:
    """把方差拆成目标间 / 重复间 / 误差三项（负的方差分量夹到 0）。

    负分量在矩估计里是可能出现的（样本噪声），标准做法夹到 0 并说明。
    """
    n, k = _shape(matrix)
    if n < 2 or k < 2:
        return None
    msr, msc, mse, *_ = mean_squares(matrix)
    return VarianceComponents(
        between_targets=max(0.0, (msr - mse) / k),
        between_raters=max(0.0, (msc - mse) / n),
        error=max(0.0, mse),
    )


@dataclass
class BlandAltman:
    """两个重复测量之间的一致性（Bland & Altman 1986）。"""

    n_pairs: int
    bias: float
    sd: float
    loa_low: float
    loa_high: float
    level: float

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "bias": _round(self.bias),
            "sd": _round(self.sd),
            "loa_low": _round(self.loa_low),
            "loa_high": _round(self.loa_high),
            "level": self.level,
        }


def bland_altman(a: Row, b: Row, *, level: float = 0.95) -> BlandAltman | None:
    """配对差值的一致性界限：偏倚 = 均值差，界限 = 偏倚 ± 1.96·SD(差)。

    `a` / `b` 是**同一批目标**上的两次测量（长度必须一致）。偏倚接近 0 说明没有系统性偏移；
    界限的宽度说明单次测量的偶然误差有多大。
    """
    if len(a) != len(b):

        raise ValueError("两个测量序列必须等长（要成对比较）")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    if len(diffs) < 2:
        return None
    bias = mean(diffs)
    sd = stdev(diffs)
    z = 1.959963984540054 if level == 0.95 else _normal_quantile(level)
    return BlandAltman(
        n_pairs=len(diffs), bias=bias, sd=sd,
        loa_low=bias - z * sd, loa_high=bias + z * sd, level=level,
    )


def _normal_quantile(level: float) -> float:
    """双侧区间的正态临界值（只用 0.95 附近，用有理逼近足够）。"""
    p = 0.5 + level / 2.0
    # Acklam 逆正态逼近
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# 剖面形状的一致性（与「水平」的一致性互补）
# ---------------------------------------------------------------------------


@dataclass
class ProfileAgreement:
    """两个剖面（一组维度上的读数）的接近程度。"""

    n_dims: int
    pearson: float
    spearman: float
    mean_abs_diff: float
    max_abs_diff: float
    within_tolerance: float

    def to_dict(self) -> dict:
        return {
            "n_dims": self.n_dims,
            "pearson": _round(self.pearson),
            "spearman": _round(self.spearman),
            "mean_abs_diff": _round(self.mean_abs_diff),
            "max_abs_diff": _round(self.max_abs_diff),
            "within_tolerance": _round(self.within_tolerance),
        }


def profile_agreement(a: Row, b: Row, *, tolerance: float = DEFAULT_TOLERANCE) -> ProfileAgreement | None:
    """两个剖面的相关性 + 逐维差异。

    为什么要和 ICC 并列报：ICC 只看「水平能不能分开」，剖面相关看「形状稳不稳」——
    一个仪器可能水平可靠（整体高低分得开）但形状不可靠（每次哪一维最高都在变）。
    两者都稳定才是真的可靠。
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    diffs = [abs(x - y) for x, y in pairs]
    return ProfileAgreement(
        n_dims=len(pairs),
        pearson=pearson(xs, ys),
        spearman=spearman(xs, ys),
        mean_abs_diff=mean(diffs),
        max_abs_diff=max(diffs),
        within_tolerance=sum(1 for d in diffs if d <= tolerance) / len(diffs),
    )


def pairwise_agreement(rows: Sequence[Row], *, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """一组剖面之间的**所有配对**平均一致性（同一条件下的多次重复）。"""
    profiles = [profile_agreement(rows[i], rows[j], tolerance=tolerance)
                for i in range(len(rows)) for j in range(i + 1, len(rows))]
    profiles = [p for p in profiles if p is not None]
    if not profiles:
        return {"n_pairs": 0}
    return {
        "n_pairs": len(profiles),
        "pearson": mean([p.pearson for p in profiles]),
        "spearman": mean([p.spearman for p in profiles]),
        "mean_abs_diff": mean([p.mean_abs_diff for p in profiles]),
        "max_abs_diff": max(p.max_abs_diff for p in profiles),
        "within_tolerance": mean([p.within_tolerance for p in profiles]),
        "tolerance": tolerance,
    }


def rank_stability(rows: Sequence[Row], dims: Sequence[str] | None = None) -> dict:
    """维度**排序**的稳定度：每次重复里各维度的高低顺序是否一致。

    读数「整体都在 0.7 附近」但每次排序大乱，与「排序一致但整体水平漂移」是两种不同的
    不可靠。做法：对每一对重复，算它们维度排序的 Spearman ρ，再取平均；同时给出众数排序。
    """
    clean = [list(row) for row in rows]
    if len(clean) < 2:
        return {"n_pairs": 0}
    rhos: list[float] = []
    orders: list[tuple[int, ...]] = []
    for i in range(len(clean)):
        orders.append(_order_key(clean[i]))
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            rho = spearman(clean[i], clean[j])
            if rho is not None:
                rhos.append(rho)
    if not rhos:
        return {"n_pairs": 0}
    mode_order = max(set(orders), key=orders.count)
    result = {
        "n_pairs": len(rhos),
        "mean_spearman": _round(mean(rhos)),
        "min_spearman": _round(min(rhos)),
        "identical_order_rate": _round(orders.count(mode_order) / len(orders)),
    }
    if dims:
        result["modal_order"] = [dims[i] for i in mode_order]
    return result


def _order_key(row: Row) -> tuple[int, ...]:
    """把一行读数变成「从高到低的维度下标」元组（并列按原下标，保证确定性）。"""
    return tuple(sorted(range(len(row)), key=lambda i: (-float(row[i]), i)))


def agreement_within(a: Row, b: Row, *, tolerance: float = DEFAULT_TOLERANCE) -> float | None:
    """两个序列里 |Δ| ≤ tolerance 的比例。"""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(1 for x, y in pairs if abs(x - y) <= tolerance) / len(pairs)


# ---------------------------------------------------------------------------
# 类别一致度（结论级：asd_likeness 的档位）
# ---------------------------------------------------------------------------

#: 档位的序（用于加权 kappa 与「差几档」的判断）。
LIKENESS_ORDER = ("none", "mild", "moderate", "marked")


def cohen_kappa(a: Sequence[str], b: Sequence[str], *, weighted: bool = False) -> float | None:
    """Cohen's κ（两个评定者对同一批单位给出的类别标签）。

    `weighted=True` 时用线性权重 `1 − |i−j|/(C−1)`（档位是有序的，差一档和差三档不该同罚）。
    κ = (po − pe) / (1 − pe)；|pe| 接近 1（几乎全落在同一类）时返回 None，避免 0/0。
    """
    if len(a) != len(b):
        raise ValueError("两个标注序列必须等长")
    pairs = [(str(x), str(y)) for x, y in zip(a, b)]
    if not pairs:
        return None
    categories = sorted({x for pair in pairs for x in pair})
    if len(categories) < 2:
        # 所有单位都在同一类：一致但 κ 无定义（pe = 1）
        return None
    index = {c: i for i, c in enumerate(categories)}
    n = len(pairs)
    observed = [[0.0] * len(categories) for _ in categories]
    for x, y in pairs:
        observed[index[x]][index[y]] += 1
    po = sum(observed[i][i] for i in range(len(categories))) / n
    row_marg = [sum(observed[i]) / n for i in range(len(categories))]
    col_marg = [sum(observed[i][j] for i in range(len(categories))) / n
                for j in range(len(categories))]
    pe = sum(row_marg[i] * col_marg[i] for i in range(len(categories)))
    if abs(1.0 - pe) < 1e-12:
        return None
    if not weighted:
        return (po - pe) / (1.0 - pe)
    order = {c: LIKENESS_ORDER.index(c) for c in categories if c in LIKENESS_ORDER}
    if len(order) != len(categories):
        return (po - pe) / (1.0 - pe)
    span = len(LIKENESS_ORDER) - 1
    po_w = 0.0
    pe_w = 0.0
    for i in range(len(categories)):
        for j in range(len(categories)):
            weight = 1.0 - abs(order[categories[i]] - order[categories[j]]) / span
            po_w += weight * observed[i][j] / n
            pe_w += weight * row_marg[i] * col_marg[j]
    if abs(1.0 - pe_w) < 1e-12:
        return None
    return (po_w - pe_w) / (1.0 - pe_w)


def categorical_agreement(rows: Sequence[Sequence[str]]) -> dict:
    """一组重复的类别标注之间的一致度。

    `rows` = 每个单位一行，行内是该单位在各次重复下的标签。做法：对每一对重复算一次
    Cohen's κ（含线性加权），再取平均 —— 比把它们拼成一张大表更贴近「重复测量」的语义。
    同时给出「全部重复完全同一档」的比例（完全一致率），这个数最直观。
    """
    units = [list(row) for row in rows if row and len(row) == len(rows[0])]
    if not units or len(units[0]) < 2:
        return {"n_units": len(units), "n_raters": 0}
    n_raters = len(units[0])
    kappas: list[float] = []
    weighted: list[float] = []
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            a = [u[i] for u in units]
            b = [u[j] for u in units]
            k = cohen_kappa(a, b)
            kw = cohen_kappa(a, b, weighted=True)
            if k is not None:
                kappas.append(k)
            if kw is not None:
                weighted.append(kw)
    unanimous = sum(1 for u in units if len(set(u)) == 1) / len(units)
    return {
        "n_units": len(units),
        "n_raters": n_raters,
        "n_pairs": len(kappas),
        "kappa": _round(mean(kappas)) if kappas else None,
        "kappa_weighted": _round(mean(weighted)) if weighted else None,
        "unanimous_rate": _round(unanimous),
    }


# ---------------------------------------------------------------------------
# 结论文本的一致度（词汇层近似，不是语义相似度）
# ---------------------------------------------------------------------------


def _ngrams(text: str, n: int) -> set[str]:
    """字符 n-gram（中文没有空格分词，字符 n-gram 是便宜且语言无关的做法）。"""
    compact = "".join(str(text or "").split())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def conclusion_similarity(texts: Sequence[str]) -> dict:
    """结论文本之间的一致度 —— **词汇层近似，不是语义相似度**。

    用字符一元组与二元组的 Jaccard 平均。它衡量的是「用词与句式有多像」，
    两个意思相同但措辞完全不同的结论会被判为低相似。作为结论级可靠性的**辅助**指标，
    主指标应看 `categorical_agreement` 的档位一致度（那是结构化的、可辩护的）。
    另外报告结论字数的离散度（协议目标 350 字）。
    """
    cleaned = [str(t or "") for t in texts if str(t or "").strip()]
    pairs_uni: list[float] = []
    pairs_bi: list[float] = []
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            pairs_uni.append(jaccard(_ngrams(cleaned[i], 1), _ngrams(cleaned[j], 1)))
            pairs_bi.append(jaccard(_ngrams(cleaned[i], 2), _ngrams(cleaned[j], 2)))
    lengths = [len(t) for t in cleaned]
    out = {
        "n_texts": len(cleaned),
        "n_pairs": len(pairs_uni),
        "jaccard_unigram": _round(mean(pairs_uni)) if pairs_uni else None,
        "jaccard_bigram": _round(mean(pairs_bi)) if pairs_bi else None,
        "chars_mean": _round(mean(lengths), 1) if lengths else None,
        "chars_cv": _round(cv(lengths)) if len(lengths) > 1 else None,
        "metric_note": "词汇层近似（字符 n-gram Jaccard），不是语义相似度",
    }
    return out


# ---------------------------------------------------------------------------
# 打包：把一批重复测量算成一张指标表
# ---------------------------------------------------------------------------


def reliability_report(
    matrix: Matrix,
    *,
    level: float = 0.95,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 42,
) -> dict:
    """一个条件的全套可靠性指标（Fig.1 每个格子/面板一行）。

    `matrix` 行=目标、列=重复。返回可直接进 JSON 的字典。

    ICC 只有点估计（区间要看目标数够不够，见 `icc21`）；**精度陈述用 SEM 及其精确区间**，
    它在任意目标数下都成立。
    """
    n, k = _shape(matrix)
    rows = [list(row) for row in matrix]
    icc = icc21(rows, level=level)
    components = variance_components(rows)
    cvs = [c for c in row_cvs(rows) if c is not None]
    sem_value = sem(rows)
    sem_interval = sem_ci(rows, level=level)
    # 剖面相关 / 排序稳定度都要按**次（列）**算，不能按**目标（行）**算：
    # 次（occasion）向量 = 该次测量在全部目标上的取值，两条次向量的相关就是
    # 「整体剖面能不能复现」；拿两行（两个目标）去相关，比的是两个目标本身，不是可靠性。
    occasions = [[row[j] for row in rows] for j in range(k)] if rows else []
    # Bland-Altman 也按次配对：第 1 次测量 vs 其余各次，单位是「同一个目标的两次测量」
    ba_list = [
        bland_altman(occasions[0], occasions[j], level=level)
        for j in range(1, len(occasions))
    ]
    ba_list = [ba for ba in ba_list if ba is not None]
    return {
        "n_targets": n,
        "n_raters": k,
        "icc": icc.to_dict(),
        "variance_components": components.to_dict() if components else None,
        "sem": _round(sem_value),
        "sem_ci_low": _round(sem_interval[0]) if sem_interval else None,
        "sem_ci_high": _round(sem_interval[1]) if sem_interval else None,
        "cv_mean": _round(mean(cvs)) if cvs else None,
        "cv_max": _round(max(cvs)) if cvs else None,
        "bland_altman": ([ba.to_dict() for ba in ba_list] or None),
        "bland_altman_max_width": (
            _round(max(ba.loa_high - ba.loa_low for ba in ba_list)) if ba_list else None
        ),
        "profile_agreement": pairwise_agreement(occasions, tolerance=tolerance),
        "rank_stability": rank_stability(occasions),
        "tolerance": tolerance,
    }
