"""Fig.1 可视化 · 数据层。

只做一件事：把落盘的 result_1_robustness.xlsx（一行一被试）读成「四轴 × 心智设定 ×
7 个计分通道」的矩阵，再算出每个面板要画的数字。统计口径全部复用实验侧的
``tools/reliability.py``（纯标准库），不另写一套公式 —— 图里的数与报告里的数必须同源。

口径备忘（引用图里任何数字时都要带上）：
  读数 1.0 = 与同龄典型发展无异，0.0 = 严重偏离；ICC 越高越可靠，SEM / CV 越小越可靠。
  (θ × 维度) 单元格为目标、重复次为测量 → cell 矩阵（27~28 个目标）。
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "results" / "data" / "result_1_robustness.xlsx"
# 同一轮实验的完整版导出（多 8 列：condition_id / axis / 均值 / asd_level / confidence / 字数）。
# 读数与 result_1_robustness.xlsx 逐格相同（已核对），只多出「档位判定」这一列 —— κ 要用它。
XLSX_FULL = ROOT / "pacs" / "data" / "runs" / "fig1_main" / "tables" / "subjects_table.xlsx"
PIPELINE_PANELS = ROOT / "pacs" / "data" / "runs" / "fig1_main" / "reliability" / "fig1_panels.json"


def _load_reliability():
    spec = importlib.util.spec_from_file_location(
        "_fig1_rel", ROOT / "pacs" / "tools" / "reliability.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fig1_rel"] = mod
    spec.loader.exec_module(mod)
    return mod


rel = _load_reliability()

# ---------------------------------------------------------------- 常量与标签

DIMS: list[tuple[str, str, str]] = [
    ("social_interaction_motivation", "S1", "社会互动与社会动机"),
    ("social_cognition", "S2", "社会认知"),
    ("social_communication_pragmatics", "S3", "社会沟通与语用"),
    ("cognitive_flexibility_adaptation", "S4", "认知灵活性与适应"),
    ("repetitive_restricted_behavior", "S5", "重复性与限制性行为"),
    ("novelty_uncertainty_sensory_behavior", "S6", "新颖性、不确定性与感觉相关行为"),
    ("general_cognition", "G", "一般认知控制（对照通道）"),
]
DIM_SHORT = {k: s for k, s, _ in DIMS}
DIM_CN = {k: c for k, _, c in DIMS}
DIM_KEYS = [k for k, _, _ in DIMS]

# 心智设定：按完整度升序排（M1 削弱最重 → M4 完整），与参考图一致
MIND_ORDER = [
    ("belief_updating 0.30", "M1", "0.30"),
    ("belief_updating 0.40", "M2", "0.40"),
    ("belief_updating 0.80", "M3", "0.80"),
    ("intact (no perturbation)", "M4", "1.00"),
]
MIND_SHORT = {label: s for label, s, _ in MIND_ORDER}
MIND_FULL = {label: c for label, _, c in MIND_ORDER}
MIND_LABELS = [label for label, _, _ in MIND_ORDER]

REFERENCE_CONDITION = "axis1_same_brain"

# 臂（condition_id）→ 展示用的组名、换了什么、颜色键
ARM_SPECS = [
    dict(
        condition="axis2_brain_pro",
        group="Brain model",
        change="模拟大脑换成 deepseek-v4-pro",
        color="brain",
        cross_model=True,
    ),
    dict(
        condition="axis3_analyst_pro",
        group="Analyst model",
        change="分析者换成 deepseek-v4-pro",
        color="analyst",
        cross_model=False,
    ),
    dict(
        condition="axis3_examiner_pro",
        group="Examiner model",
        change="提问者换成 deepseek-v4-pro",
        color="examiner",
        cross_model=False,
    ),
    dict(
        condition="axis3_probe_seed7",
        group="Probe seed",
        change="题序 seed 42 → 7（第 7 位起）",
        color="seed",
        cross_model=False,
    ),
]
LENGTH_SPECS = [
    dict(
        condition="axis4_turns_25",
        group="50 vs 25 turns",
        change="硬固定 25 轮（低于 7×5 配额）",
        color="len25",
    ),
    dict(
        condition="axis4_turns_100",
        group="50 vs 100 turns",
        change="硬固定 100 轮",
        color="len100",
    ),
]

# 调色：与结果图的语义色板一致（定性配色，蓝=脑、青=分析者、橙=提问者、紫=题序、
# 绿系=完整度梯度、灰=对照通道）
PALETTE = {
    "brain": "#2E6FD6",
    "analyst": "#18A08C",
    "examiner": "#F08A24",
    "seed": "#8A5BD6",
    "len25": "#5B8DB8",
    "len100": "#1F4E79",
    "reference": "#3D3D3D",
    "dim": {
        "social_interaction_motivation": "#2E6FD6",
        "social_cognition": "#18A08C",
        "social_communication_pragmatics": "#F08A24",
        "cognitive_flexibility_adaptation": "#D64545",
        "repetitive_restricted_behavior": "#8A5BD6",
        "novelty_uncertainty_sensory_behavior": "#E8C41C",
        "general_cognition": "#8A8F98",
    },
    "mind": {label: c for label, c in zip(MIND_LABELS, ["#2166AC", "#67A9CF", "#EF8A62", "#B2182B"])},
    "panel_bg": "#F2F7FB",
    "panel_edge": "#A9C4DC",
    "ink": "#1A1A1A",
    "muted": "#5A6472",
    "grid": "#D8DEE6",
}


# ---------------------------------------------------------------- 读表


@dataclass
class Subject:
    row_id: str
    condition: str
    mind: str
    completeness: float
    replicate: int
    turns_run: int
    brain_model: str
    examiner_model: str
    analyst_model: str
    dims: dict[str, float | None]
    summary: str = ""
    level: str | None = None

    @property
    def has_reading(self) -> bool:
        return any(v is not None for v in self.dims.values())


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_subjects(xlsx: Path = XLSX, full: Path = XLSX_FULL) -> list[Subject]:
    ws = openpyxl.load_workbook(xlsx, data_only=True)["by_subject"]
    rows = list(ws.iter_rows(values_only=True))
    hdr, body = rows[0], rows[1:]
    # 表头带中文维度名，按前缀匹配定位列
    col = {name: i for i, name in enumerate(hdr) if name}

    def find(prefix: str) -> int:
        for name, i in col.items():
            if str(name).startswith(prefix):
                return i
        raise KeyError(prefix)

    dim_cols = {k: find(k) for k in DIM_KEYS}
    i_id, i_mind = col["row_id"], col["mind_setting"]
    i_comp, i_rep, i_turns = col["completeness"], col["replicate"], col["turns_run"]
    i_brain, i_exam, i_ana = col["brain_model"], col["examiner_model"], col["analyst_model"]
    i_sum = col["analyst_summary"]

    subjects = []
    for r in body:
        rid = str(r[i_id])
        subjects.append(
            Subject(
                row_id=rid,
                condition=rid.split("#")[0],
                mind=str(r[i_mind]),
                completeness=_num(r[i_comp]) or 0.0,
                replicate=int(r[i_rep]),
                turns_run=int(r[i_turns]),
                brain_model=str(r[i_brain]),
                examiner_model=str(r[i_exam]),
                analyst_model=str(r[i_ana]),
                dims={k: _num(r[i]) for k, i in dim_cols.items()},
                summary=str(r[i_sum] or ""),
            )
        )

    # 档位判定（none/mild/moderate/marked）来自同轮实验的完整导出；逐格读数已核对一致
    if full.exists():
        ws2 = openpyxl.load_workbook(full, data_only=True)["by_subject"]
        rows2 = list(ws2.iter_rows(values_only=True))
        hdr2 = rows2[0]
        i2_id, i2_lvl = hdr2.index("row_id"), hdr2.index("asd_level")
        levels = {str(r[i2_id]): (str(r[i2_lvl]) if r[i2_lvl] else None) for r in rows2[1:]}
        for s in subjects:
            s.level = levels.get(s.row_id)

    return subjects


# ---------------------------------------------------------------- 矩阵与指标


def by_condition(subjects: list[Subject]) -> dict[str, list[Subject]]:
    out: dict[str, list[Subject]] = {}
    for s in subjects:
        out.setdefault(s.condition, []).append(s)
    return out


def cell_matrix(subjects: list[Subject], *, minds: list[str] | None = None) -> tuple[list[str], list[str], list[list[float]]]:
    """目标 = (心智设定 × 通道)，测量 = 重复次。

    与实验侧 `matrix_from_records` 同一口径：**只保留测量次数达到最多的目标**。少测一次的目标
    硬塞进去就得补值，那是在伪造数据，所以宁可丢掉（本项目实际丢了 1 格：100 轮档的分析者
    对某一维合法地给了 null）。返回的矩阵保证是完整矩形。
    """
    minds = minds or [m for m in MIND_LABELS if any(s.mind == m for s in subjects)]
    reps = sorted({s.replicate for s in subjects})
    index = {(s.mind, s.replicate): s for s in subjects}
    collected: dict[str, list[float]] = {}
    for mind in minds:
        for dim in DIM_KEYS:
            row = []
            for rep in reps:
                s = index.get((mind, rep))
                v = s.dims.get(dim) if s else None
                if v is not None:
                    row.append(v)
            if row:
                collected[f"{MIND_SHORT[mind]}|{DIM_SHORT[dim]}"] = row
    if not collected:
        return [], minds, []
    full = max(len(v) for v in collected.values())
    targets = [t for t in collected if len(collected[t]) == full]
    key = {}
    for mind in minds:
        for dim in DIM_KEYS:
            key[f"{MIND_SHORT[mind]}|{DIM_SHORT[dim]}"] = (minds.index(mind), DIM_KEYS.index(dim))
    targets.sort(key=lambda t: key[t])
    return targets, minds, [collected[t] for t in targets]


def cell_means(subjects: list[Subject]) -> dict[tuple[str, str], float]:
    """(心智设定, 通道) → 重复次均值；缺格的心智设定/通道不出现。"""
    acc: dict[tuple[str, str], list[float]] = {}
    for s in subjects:
        for dim, v in s.dims.items():
            if v is not None:
                acc.setdefault((s.mind, dim), []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def cell_sd(subjects: list[Subject]) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for s in subjects:
        for dim, v in s.dims.items():
            if v is not None:
                acc.setdefault((s.mind, dim), []).append(v)
    out = {}
    for k, v in acc.items():
        if len(v) > 1:
            out[k] = rel.stdev(v)
    return out


@dataclass
class ArmMetrics:
    condition: str
    group: str
    change: str
    color: str
    n_subjects: int
    n_targets: int
    n_replicates: int
    icc: float | None
    icc_lo: float | None
    icc_hi: float | None
    sem: float | None
    sem_lo: float | None
    sem_hi: float | None
    cv_mean: float | None
    within_profile_pearson: float | None
    within_profile_spearman: float | None
    within_tolerance: float | None
    mean_abs_diff_within: float | None
    rank_stability: float | None
    kappa: float | None
    weighted_kappa: float | None
    unanimous: float | None
    # 与参考臂的一致性
    r_vs_ref: float | None = None
    r_vs_ref_lo: float | None = None
    r_vs_ref_hi: float | None = None
    rho_vs_ref: float | None = None
    mad_vs_ref: float | None = None
    max_abs_diff_vs_ref: float | None = None
    tol_vs_ref: float | None = None
    bias_vs_ref: float | None = None
    loa_low: float | None = None
    loa_high: float | None = None
    turns_mean: float | None = None
    within_pair_pearson: list[float] = field(default_factory=list)
    within_pair_spearman: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    import math

    z = math.atanh(max(min(r, 0.999999), -0.999999))
    se = 1.0 / math.sqrt(max(n - 3, 1))
    half = rel._normal_quantile(level) * se
    return math.tanh(z - half), math.tanh(z + half)


def _likeness_agreement(subj: list[Subject]) -> dict:
    """档位一致度：单位 = 心智设定，评定者 = 重复次（与实验侧 categorical_agreement 同口径）。"""
    rows = []
    for mind in MIND_LABELS:
        cell = [s for s in subj if s.mind == mind]
        if not cell or any(s.level is None for s in cell):
            continue
        cell.sort(key=lambda s: s.replicate)
        rows.append([s.level for s in cell])
    if not rows or len({len(r) for r in rows}) != 1:
        return {}
    cat = rel.categorical_agreement(rows)
    return {
        "kappa": cat.get("kappa"),
        "weighted_kappa": cat.get("kappa_weighted"),
        "unanimous": cat.get("unanimous_rate"),
    }


def arm_metrics(subj: list[Subject], spec: dict) -> ArmMetrics:
    targets, _, matrix = cell_matrix(subj)  # 目标 × 重复：ICC / SEM / 逐目标 CV 的形状
    profiles = [list(c) for c in zip(*matrix)]  # 次 × 目标：剖面相关 / 排序稳定度的形状
    icc = rel.icc21(matrix)
    s = rel.sem(matrix)
    sci = rel.sem_ci(matrix)
    cvs = [c for c in rel.row_cvs(matrix) if c is not None]
    pair = rel.pairwise_agreement(profiles)
    ranks = rel.rank_stability(profiles)
    pair_pearson, pair_spearman = [], []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            pa = rel.profile_agreement(profiles[i], profiles[j])
            if pa is not None and pa.pearson is not None:
                pair_pearson.append(pa.pearson)
                pair_spearman.append(pa.spearman)
    cat = _likeness_agreement(subj)
    turns = [x.turns_run for x in subj]
    return ArmMetrics(
        condition=spec["condition"],
        group=spec["group"],
        change=spec["change"],
        color=spec["color"],
        n_subjects=len(subj),
        n_targets=len(targets),
        n_replicates=len({x.replicate for x in subj}),
        icc=icc.icc,
        icc_lo=icc.ci_low,
        icc_hi=icc.ci_high,
        sem=s,
        sem_lo=sci[0] if sci else None,
        sem_hi=sci[1] if sci else None,
        cv_mean=(sum(cvs) / len(cvs)) if cvs else None,
        within_profile_pearson=pair.get("pearson"),
        within_profile_spearman=pair.get("spearman"),
        within_tolerance=pair.get("within_tolerance"),
        mean_abs_diff_within=pair.get("mean_abs_diff"),
        rank_stability=ranks.get("mean_spearman"),
        kappa=cat.get("kappa"),
        weighted_kappa=cat.get("weighted_kappa"),
        unanimous=cat.get("unanimous"),
        turns_mean=sum(turns) / len(turns) if turns else None,
        within_pair_pearson=pair_pearson,
        within_pair_spearman=pair_spearman,
    )


def compare_to_reference(base: list[Subject], other: list[Subject], m: ArmMetrics) -> ArmMetrics:
    """把 other 臂的 (心智设定 × 通道) 格均值与参考臂逐格配对。"""
    a, b = cell_means(base), cell_means(other)
    keys = [k for k in a if k in b]
    keys.sort(key=lambda k: (MIND_LABELS.index(k[0]), DIM_KEYS.index(k[1])))
    va, vb = [a[k] for k in keys], [b[k] for k in keys]
    r = rel.pearson(va, vb)
    rho = rel.spearman(va, vb)
    pa = rel.profile_agreement(va, vb)
    ba = rel.bland_altman(va, vb)
    m.r_vs_ref = r
    if r is not None:
        m.r_vs_ref_lo, m.r_vs_ref_hi = _fisher_ci(r, len(keys))
    m.rho_vs_ref = rho
    m.mad_vs_ref = pa.mean_abs_diff if pa else None
    m.max_abs_diff_vs_ref = pa.max_abs_diff if pa else None
    m.tol_vs_ref = pa.within_tolerance if pa else None
    m.bias_vs_ref = ba.bias if ba else None
    m.loa_low = ba.loa_low if ba else None
    m.loa_high = ba.loa_high if ba else None
    m.extra["n_cells"] = len(keys)
    m.extra["cells"] = keys
    return m


# ---------------------------------------------------------------- 受影响维度判定

#: 「受影响」的判定口径：在**同一次重复内**，某表型通道的读数比对照通道低 τ 以上。
#: 为什么以对照通道为参照：对照通道是设计上「应该被豁免」的那一条（阴性预测的测量基础），
#: 拿它作同道参照可以把「整体水平整体下移」消掉，剩下的才是**选择性**——哪几维真的被打了。
#: 为什么 τ = 0.20：0.15 是文档里「看得出区别」的尺度，取比它大一档，避免边界值来回翻。
AFFECTED_TAU = 0.20
PHENOTYPE_KEYS = [k for k, _, _ in DIMS if k != "general_cognition"]
CONTROL_KEY = "general_cognition"


def affected_flags(profile: dict, tau: float = AFFECTED_TAU) -> set[str]:
    """一次测量的「受影响通道集合」。对照通道缺失或该通道无读数时不计入。"""
    ctrl = profile.get(CONTROL_KEY)
    if ctrl is None:
        return set()
    return {
        k for k in PHENOTYPE_KEYS
        if profile.get(k) is not None and (ctrl - profile[k]) >= tau
    }


def _jaccard(a: set, b: set) -> float:
    """两个集合的 Jaccard；**两个都为空时记 1**（都判「没有维度受影响」是一次一致判断）。"""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def affected_metrics(subj: list[Subject], tau: float = AFFECTED_TAU) -> dict:
    """每个心智设定上，「受影响维度判定」在重复间稳不稳。

    只关心**被影响的那几维**是否被稳定指认出来，不关心不受影响维度之间的排序 ——
    后者本来就没什么可复现的（它们的读数都在典型范围内来回抖）。

    返回每个设定的四个分布：
      jaccard       —— 两两重复之间受影响集合的 Jaccard（C(5,2)=10 个值）
      flag_agree    —— 两两重复之间、6 个通道上「受影响 / 不受影响」判定一致的比例
      detect_rate   —— 6 个通道各自的被检出比例（值 = 该通道被几次重复标为受影响）
      rank_rho      —— 只在受影响通道上算的 Spearman（两两重复；少于 3 个通道时
                       退化为「缺口最大的 3 个通道」，保证秩相关有定义）
    """
    from itertools import combinations

    out = {}
    for mind in MIND_LABELS:
        rows = sorted([s for s in subj if s.mind == mind], key=lambda s: s.replicate)
        if not rows:
            continue
        flags = {s.replicate: affected_flags(s.dims, tau) for s in rows}
        reps = [s.replicate for s in rows]

        jac, agree, rho = [], [], []
        for a_rep, b_rep in combinations(reps, 2):
            A, B = flags[a_rep], flags[b_rep]
            jac.append(_jaccard(A, B))
            agree.append(
                sum(1 for k in PHENOTYPE_KEYS if (k in A) == (k in B)) / len(PHENOTYPE_KEYS)
            )
            # 只在受影响通道上比排序；通道太少（<3）就退回「缺口最大的 3 个」
            subset = sorted(A | B)
            sa = next(s for s in rows if s.replicate == a_rep)
            sb = next(s for s in rows if s.replicate == b_rep)
            if len(subset) < 3:
                deficits = {}
                for k in PHENOTYPE_KEYS:
                    va, vb = sa.dims.get(k), sb.dims.get(k)
                    ca, cb = sa.dims.get(CONTROL_KEY), sb.dims.get(CONTROL_KEY)
                    if None in (va, vb, ca, cb):
                        continue
                    deficits[k] = ((ca - va) + (cb - vb)) / 2
                subset = [k for k, _ in sorted(deficits.items(), key=lambda kv: -kv[1])][:3]
            if len(subset) >= 3:
                xa = [sa.dims.get(k) for k in subset]
                xb = [sb.dims.get(k) for k in subset]
                if all(v is not None for v in xa + xb):
                    r = rel.spearman(xa, xb)
                    if r is not None:
                        rho.append(r)

        out[mind] = dict(
            mind=mind,
            label=MIND_SHORT[mind],
            completeness=MIND_FULL[mind],
            n_replicates=len(reps),
            tau=tau,
            jaccard=jac,
            flag_agree=agree,
            detect_rate={k: sum(1 for f in flags.values() if k in f) / len(reps) for k in PHENOTYPE_KEYS},
            rank_rho=rho,
            flags={MIND_SHORT[mind]: sorted(flags[r]) for r in reps},
            union=sorted(set().union(*flags.values())) if flags else [],
        )
    return out


def rank_metrics(subj: list[Subject], tau: float = AFFECTED_TAU) -> dict:
    """受影响维度的**名次**口径：逐通道平均名次 + 每次重复的 top-2 通道。

    名次定义：在同一次重复内，把 6 个表型通道按**读数升序**排名，名次 1 = 读数最低 = 最受影响。
    （等价于按「比对照通道的缺口」降序排 —— 同一次重复内对照通道是个常数，减它不改变顺序。）
    并列用**平均名次**（两个并列第一各记 1.5），避免人为决定谁先谁后。

    top-2 = 读数最小的两个通道；出现精确并列时按 `DIMS` 里的先后顺序定夺（保证可复现）。
    ``top2[rep]`` **按名次排好序**：第 0 个是 top-1、第 1 个是 top-2（图上靠这个分色）。
    """
    out = {}
    for mind in MIND_LABELS:
        rows = sorted([s for s in subj if s.mind == mind], key=lambda s: s.replicate)
        if not rows:
            continue
        per_rep_rank: list[dict] = []
        top2: dict[int, list[str]] = {}
        for s in rows:
            vals = {k: s.dims.get(k) for k in PHENOTYPE_KEYS}
            if any(v is None for v in vals.values()):
                continue
            ordered = sorted(PHENOTYPE_KEYS, key=lambda k: (vals[k], PHENOTYPE_KEYS.index(k)))
            # 平均名次：先按升序给序号，再把名次相同的取平均
            ranks: dict[str, float] = {}
            i = 0
            while i < len(ordered):
                j = i
                while j + 1 < len(ordered) and vals[ordered[j + 1]] == vals[ordered[i]]:
                    j += 1
                avg = (i + j) / 2 + 1  # 1-based 平均名次
                for t in range(i, j + 1):
                    ranks[ordered[t]] = avg
                i = j + 1
            per_rep_rank.append(ranks)
            top2[s.replicate] = ordered[:2]
        out[mind] = dict(
            mind=mind,
            label=MIND_SHORT[mind],
            theta=float(MIND_FULL[mind]),
            avg_rank={k: sum(r[k] for r in per_rep_rank) / len(per_rep_rank) for k in PHENOTYPE_KEYS},
            top2={r: top2[r] for r in sorted(top2)},
            top2_counts={
                k: sum(1 for r in top2 if k in top2[r]) / len(top2) if top2 else None
                for k in PHENOTYPE_KEYS
            },
            n_replicates=len(per_rep_rank),
        )
    return out


# ---------------------------------------------------------------- 面板 b：同一脑重复


def per_mind_reliability(subj: list[Subject]) -> dict:
    """每个心智设定各自的（目标 = 7 个通道、测量 = 5 次重复）可靠性。

    ICC 用 7 通道为目标；jackknife(留一通道) 给出分布，因为 4 个 θ 的区间口径不适用于
    这一层（目标数 < 10，报告侧同样拒绝给区间）。CV 逐通道 → 7 个值。
    排序稳定度 = 所有重复配对（C(5,2)=10）的 Spearman。
    """
    out = {}
    for mind in MIND_LABELS:
        rows = [s for s in subj if s.mind == mind]
        if not rows:
            continue
        reps = sorted({s.replicate for s in rows})
        index = {s.replicate: s for s in rows}
        matrix = [[index[rep].dims[d] for rep in reps] for d in DIM_KEYS]
        matrix = [row for row in matrix if all(v is not None for v in row)]

        icc = rel.icc21(matrix)
        # jackknife 留一通道：ICC 与 SEM 都按同一把尺子给分布
        icc_jk, sem_jk = [], []
        for i in range(len(matrix)):
            sub = matrix[:i] + matrix[i + 1:]
            v = rel.icc_point(sub)
            if v is not None:
                icc_jk.append(v)
            sv = rel.sem(sub)
            if sv is not None:
                sem_jk.append(sv)
        cvs = [c for c in rel.row_cvs(matrix) if c is not None]
        sd_channels = [rel.stdev([x for x in row if x is not None]) for row in matrix]
        sd_channels = [v for v in sd_channels if v is not None]
        profiles = [list(c) for c in zip(*matrix)]  # 次 × 通道
        rank = rel.rank_stability(profiles)
        # 逐配对 Spearman（与 rank_stability 同口径，但要拿到分布而不是均值）
        pair_rhos = []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                x = [index[reps[i]].dims[d] for d in DIM_KEYS]
                y = [index[reps[j]].dims[d] for d in DIM_KEYS]
                if all(v is not None for v in x) and all(v is not None for v in y):
                    rho = rel.spearman(x, y)
                    if rho is not None:
                        pair_rhos.append(rho)
        out[mind] = dict(
            mind=mind,
            label=MIND_SHORT[mind],
            completeness=MIND_FULL[mind],
            icc=icc.icc,
            icc_jk=icc_jk,
            sem=rel.sem(matrix),
            sem_jk=sem_jk,
            cv=cvs,
            cv_mean=sum(cvs) / len(cvs) if cvs else None,
            sd_channels=sd_channels,
            rank_mean=rank.get("mean_spearman"),
            rank_pairs=pair_rhos,
            # 档位 κ 不在这里给：这一层的单位只有 1 个心智设定，单个单位的 κ 无定义。
            # 结论级 κ（单位 = 4 个心智设定，评定者 = 重复次）在 arm_metrics 里算。
            n_subjects=len(rows),
            matrix=matrix,
            reps=reps,
            target_labels=[DIM_SHORT[d] for d in DIM_KEYS][: len(matrix)],
        )
    return out


# ---------------------------------------------------------------- 自检


def profile_ranges(subs: list[Subject]) -> dict[str, tuple[float, float, float, int]]:
    """每个通道在读到的全部原始读数上的取值全距（min / median / max / n）。

    一个条件下同一通道有多条读数：4 档 θ × 每次重复各一条（k=5 的臂 20 条、k=3 的臂 12 条）。
    这里不做任何聚合或截尾 —— 全距就是最小到最大，中位数只是画在线上做参照点。
    """
    out = {}
    for key in DIM_KEYS:
        vals = sorted(s.dims[key] for s in subs if s.dims.get(key) is not None)
        if not vals:
            continue
        out[key] = (vals[0], float(statistics.median(vals)), vals[-1], len(vals))
    return out


def per_mind_ranges(subs: list[Subject]) -> dict[str, dict[str, tuple[float, float, float, int]]]:
    """按心智设定拆开的取值全距：{mind: {channel: (min, median, max, n)}}。

    拆到 (条件 × 心智设定 × 通道) 之后每格只剩 k 条读数（k = 该臂的重复次数），
    min–max 在 k=3 时会很脆 —— 面板 C 的图注里必须写明这一点。
    """
    out = {}
    for mind in MIND_LABELS:
        rows = [s for s in subs if s.mind == mind]
        if rows:
            out[mind] = profile_ranges(rows)
    return out


def selfcheck(arms: dict[str, ArmMetrics]) -> list[str]:
    """把本脚本从 xlsx 算出来的数与实验侧落盘的 fig1_panels.json 逐项比对。"""
    if not PIPELINE_PANELS.exists():
        return ["未找到 fig1_panels.json，跳过一致性核对"]
    doc = json.loads(PIPELINE_PANELS.read_text(encoding="utf-8"))
    msgs, worst = [], 0.0
    for ax in ("axis1", "axis2", "axis3", "axis4"):
        for row in doc["panels"][ax]["rows"]:
            c = row["condition"]
            if c not in arms:
                continue
            m = arms[c]
            pairs = [
                ("icc", m.icc, row["icc"]),
                ("sem", m.sem, row["sem"]),
                ("cv", m.cv_mean, row["cv_mean"]),
                ("pearson", m.within_profile_pearson, row["pearson"]),
                ("spearman", m.within_profile_spearman, row["spearman"]),
                ("tol", m.within_tolerance, row["within_tolerance"]),
                ("kappa", m.kappa, row["likeness_kappa"]),
            ]
            for name, got, want in pairs:
                if got is None or want is None:
                    continue
                d = abs(got - want)
                worst = max(worst, d)
                flag = "OK " if d < 5e-4 else ("≈  " if d < 5e-3 else "!! ")
                if flag != "OK ":
                    msgs.append(f"{flag}{c}.{name}: xlsx={got:.4f} pipeline={want:.4f} Δ={d:.4f}")
    msgs.insert(0, f"与 fig1_panels.json 的最大偏差 = {worst:.2e}")
    return msgs


# ---------------------------------------------------------------- 组装

DISPLAY_ORDER = [s["condition"] for s in ARM_SPECS] + [s["condition"] for s in LENGTH_SPECS] + [REFERENCE_CONDITION]


def build() -> dict:
    subs = load_subjects()
    groups = by_condition(subs)
    ref = groups[REFERENCE_CONDITION]

    m_ref = arm_metrics(ref, dict(condition=REFERENCE_CONDITION, group="Same brain", change="同参数重跑 5 次（参考臂）", color="reference"))
    arms = {}
    for spec in ARM_SPECS + LENGTH_SPECS:
        subj = groups.get(spec["condition"])
        if not subj:
            continue
        m = arm_metrics(subj, spec)
        arms[spec["condition"]] = compare_to_reference(ref, subj, m)
    arms[REFERENCE_CONDITION] = m_ref

    heat = {}
    for mind in MIND_LABELS:
        rows = sorted([s for s in ref if s.mind == mind], key=lambda s: s.replicate)
        if rows:
            heat[mind] = [[rows[i].dims[d] for i in range(len(rows))] for d in DIM_KEYS]

    return dict(
        subjects=subs,
        groups=groups,
        reference=ref,
        arms=arms,
        arm_order=DISPLAY_ORDER,
        heat=heat,
        per_mind=per_mind_reliability(ref),
        affected=affected_metrics(ref),
        ranks=rank_metrics(ref),
        ranges={c: profile_ranges(g) for c, g in groups.items()},
        ranges_by_mind={c: per_mind_ranges(g) for c, g in groups.items()},
        checks=selfcheck(arms),
    )


if __name__ == "__main__":
    d = build()
    print("\n".join(d["checks"]))
    print(f"\n被试 {len(d['subjects'])}；参考臂 {len(d['reference'])}")
    for c in d["arm_order"]:
        m = d["arms"][c]
        print(
            f"{c:22s} n={m.n_subjects:3d} tgt={m.n_targets:2d} k={m.n_replicates} "
            f"ICC={m.icc if m.icc is None else round(m.icc,4)} "
            f"SEM={m.sem if m.sem is None else round(m.sem,4)} "
            f"CV={m.cv_mean if m.cv_mean is None else round(m.cv_mean,4)} "
            f"r_ref={m.r_vs_ref if m.r_vs_ref is None else round(m.r_vs_ref,4)} "
            f"tol={m.tol_vs_ref if m.tol_vs_ref is None else round(m.tol_vs_ref,3)} "
            f"k={m.kappa if m.kappa is None else round(m.kappa,3)}"
        )
    print()
    for mind, v in d["per_mind"].items():
        print(
            f"{MIND_SHORT[mind]} {v['completeness']}  ICC={v['icc']:.4f} "
            f"jk=[{min(v['icc_jk']):.3f},{max(v['icc_jk']):.3f}] "
            f"CV={v['cv_mean']:.4f} rank={v['rank_mean']:.4f} kappa={v.get('kappa')}"
        )
