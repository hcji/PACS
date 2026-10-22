"""Fig.2 绘图：把实验 2（单维度扰动批）的落盘结果画成一张多面板图。

数据源以 `results/data/result_2_univariate.xlsx`（实验自己落盘的那张表）为主，
panel E 的检验另取同一个模型的 intact 重复（见下）。

  第一行  A 表型扰动热图（31 primitives × 6 表型维度，两个剂量并排）｜ B 表型效应曼哈顿图
  第二行  E primitive–phenotype 二部关联网络 × 2（两档剂量）+ 各自右侧的显著个数条形图

## 口径（全部照抄实验侧的判据，不另立一套）

- `baseline`：5 份完整心智剖面的均值（由 xlsx 的 `*_mean − *_delta` 反解，逐行核对唯一）。
- `Δ`：本图一律用**涌现方向** `Δ = baseline − perturbed`，**正值 = 更像 ASD**。
  xlsx 里的 `*_delta` 是 `perturbed − baseline`（372 格中 371 格为负），照画会得到一张单色图；
  翻转符号后与参考图「正值 = 表型增强」的读法一致。
- `TARGET_DROP = 0.15`：主导维度至少下降这么多才算「有效扰动」（= 判定列的口径，仍用于
  verdict / 受影响维度数；panel E 的**边**已改用统计检验，不再用这个门槛）。
- `CONTROL_TOLERANCE = 0.15`：对照通道 general_cognition 的容差带。
- `集中度 = 次强维度 Δ − 主导维度 Δ`，门槛 `max(0.05, 2·SE)`，SE 由 3 次重复逐次读数估计。
  这三个常数直接取自 `pacs/analysis/specificity.py`。
- **panel E 的边 = 两样本 t 检验**（`TEST_ALPHA`，BH-FDR 校正）：每个 primitive × 每个
  表型维度，把该 primitive 的 **3 次重复**与参照组 `result_1_robustness.xlsx` 里
  `axis1_same_brain` × intact 的 **5 次重复**一起做 t 检验（等方差版，与两组单因素 ANOVA
  等价，F = t²，脚本逐格核对）；参照组的逐维度均值与上面的 `baseline` 常数逐位一致
  （`load()` 里有断言），所以参照就是这个实验自己的基线，只是从「一个均值」还原成了
  「5 个观测」。条形图的第二根柱子是**同一套 p 值收严到 `TEST_ALPHA_STRICT`** 的计数
  （同一检验、更严阈值），读的是「结论在更严阈值下还剩多少」；Welch 与 Behrens–Fisher
  两版计数仍逐格落盘在 metrics.json 里供口径核对，只是不再上柱。

脚本每次都把这样重算出的判定与 xlsx 里 `specificity_verdict` 列**逐行比对**，不一致就报错 ——
图里的判定与实验报告里的判定必须是同一个判定。

**主剂量取 0.4**（primitive 完整度 0.4，即较强的那一档）：该档效应最清晰、剖面最完整。
0.8 档不丢 —— 它同样有网络 + 条形图，也进 metrics.json（`primitives_dose_0.8`）。

运行：``python results/scripts/fig2_visualize.py``
输出：``results/figure/figure2_univariate_phenotypes.{png,pdf}`` + ``figure2_metrics.json``
"""

from __future__ import annotations

import json
import math
import sys
import textwrap
import warnings
from pathlib import Path
from statistics import stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway, ttest_ind

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "results" / "data" / "result_2_univariate.xlsx"
OUT_DIR = ROOT / "results" / "figure"
STEM = "figure2_univariate_phenotypes"

# ---------------------------------------------------------------- 常量与判据

#: 6 个表型维度（顺序 = 热图列序 = 网络右端序）；general_cognition 是对照通道，单列附在热图右侧
DIMS = [
    "social_interaction_motivation",
    "social_cognition",
    "social_communication_pragmatics",
    "cognitive_flexibility_adaptation",
    "repetitive_restricted_behavior",
    "novelty_uncertainty_sensory_behavior",
]
CONTROL = "general_cognition"

#: 实验侧判据（`pacs/src/pacs/analysis/specificity.py`），原样引用
TARGET_DROP = 0.15
CONTROL_TOLERANCE = 0.15
CONCENTRATION_MARGIN = 0.05

PRIMARY_DOSE = 0.4            # 主剂量 = primitive 完整度 0.4（较强的一档）
SECONDARY_DOSE = 0.8

#: panel F 另取一张表：**剂量阶梯**（5 个 primitive 各跑 5 个 θ 档、每档 5 次重复）。
#: 与 panel A–E 的 result_2 不是同一批运行：result_2 每格 3 次重复、只有 0.4 / 0.8 两档，
#: result_4 把 5 个 primitive 补成 0.2–0.9 的完整阶梯、每格 5 次重复（其中 0.4 / 0.8 两档里
#: 更早批次跑的那 3 次重复是**复用**的，没有重跑 —— error 结构不同源，见该表自带的说明）。
XLSX4 = ROOT / "results" / "data" / "result_4_does_response.xlsx"
#: 阶梯的档位（升序）。横轴就按这一串画：0.8 与 0.9 之间比别处窄，是数据本身如此，不重排。
DOSE_LADDER = [0.2, 0.4, 0.6, 0.8, 0.9]
#: panel F 的 5 张子图（= result_4 的行序 = 阶梯实验选中的 5 个 primitive）。
LADDER_PRIMITIVES = [
    "agent_state_updating",
    "belief_updating",
    "rule_updating",
    "self_other_distinction",
    "social_reward_weighting",
]

#: panel E 的边判据是**统计检验**（不是 Δ 门槛）：每个 primitive × 每个表型维度，
#: 拿该 primitive 的 3 次重复读数与**参照组的重复读数**做两样本 t 检验，得 p 值；
#: 再在「同一剂量 × 同一维度」这一族 31 个检验内做 BH-FDR 校正，q < TEST_ALPHA 算显著。
#:
#: 两组（3 次扰动 vs 5 次 intact）时，等方差 t 检验与单因素 ANOVA 是同一个检验（F = t²，
#: p 值逐位相同，load() 里对这种等价性下了断言），所以这里直接按 t 检验写、按 t 检验报。
#: 主判据（画边、节点数字）用**等方差（合并方差）**那一版、等级 q < `TEST_ALPHA`；
#: 条形图并排给出**同一检验、更严等级** `TEST_ALPHA_STRICT` 下的计数 —— 读的是「结论在更严
#: 阈值下还剩多少」，不是换一套检验。本数据的「扰动组方差 / 参照组方差」中位数约 5 倍
#: （205/372 个格子超过 4 倍），等方差假定在这里并不成立，所以细读计数时要知道：浅色柱虽然
#: 挂着更严的名义等级，实际第一类错误率并不是 1%（见 `TEST_ALPHA_STRICT` 处的说明）。
TEST_ALPHA = 0.05
#: 条形图第二根柱子（浅色）的等级：与实心柱**同一套** p 值（合并方差 t 检验 + 同族 BH-FDR），
#: 只把判显著的等级从 0.05 收到 0.01，因此浅色柱恒为实心柱的子集。
#: 这是**名义**等级，不是保证：零假设模拟（n=3 vs 5，本数据构型）给出的实际第一类错误率是
#: 名义 0.05 时约 11%、名义 0.01 时约 3.6%（方差比 1× 时 1.0%、20× 时 7.0%），而本数据逐格
#: 方差比的四分位区间是 1.2×–17×，所以一个统一等级没法把整族都校准到 5% —— 这条在
#: 这条口径写进 `figure2_metrics.json`，别把 0.01 读成 1% 的保证。
TEST_ALPHA_STRICT = 0.01
#: 参照组 = 同一模型下 axis1 的 intact 重复（result_1）。用它有两个硬理由：一是它每次读数
#: 都在，组内方差可用（xlsx 的 summary 只落了一个 baseline 均值，没法做检验）；二是它的
#: 逐维度均值与 summary 反解出的 baseline 常数**逐位相同**（load() 里有断言），也就是
#: 实验侧本来就是拿这批读数当基线的，不是本图另找的参照。
REF_XLSX = ROOT / "results" / "data" / "result_1_robustness.xlsx"
REF_ROW_PREFIX = "axis1_same_brain"
REF_SETTING = "intact (no perturbation)"
TEST_MIN_REPLICATES = 3

CONTROL_EN = "General cognition (control)"

#: 表型维度英文：热图列头（1 行，竖排） / 网络右端与图例（2 行） / 小条形图（2 行）
DIM_1LINE = {
    "social_interaction_motivation": "Social interaction & motivation",
    "social_cognition": "Social cognition",
    "social_communication_pragmatics": "Social communication & pragmatics",
    "cognitive_flexibility_adaptation": "Cognitive flexibility & adaptation",
    "repetitive_restricted_behavior": "Repetitive/restricted behavior",
    "novelty_uncertainty_sensory_behavior": "Novelty/uncertainty & sensory behavior",
}
DIM_2L = {
    "social_interaction_motivation": "Social interaction\n& motivation",
    "social_cognition": "Social cognition",
    "social_communication_pragmatics": "Social communication\n& pragmatics",
    "cognitive_flexibility_adaptation": "Cognitive flexibility\n& adaptation",
    "repetitive_restricted_behavior": "Repetitive/\nrestricted behavior",
    "novelty_uncertainty_sensory_behavior": "Novelty/uncertainty\n& sensory behavior",
}

#: 31 个 primitive 的英文名（xlsx 只落 primitive_id / primitive_zh）
EN = {
    "agent_state_updating": "Agent-state updating",
    "ambiguity_tolerance": "Ambiguity tolerance",
    "attention_persistence": "Attention persistence",
    "attention_shifting": "Attention shifting",
    "belief_updating": "Belief updating",
    "confidence_calibration": "Confidence calibration",
    "evidence_weighting": "Evidence weighting",
    "familiarity_valuation": "Familiarity valuation",
    "feature_binding": "Feature binding",
    "goal_maintenance": "Goal maintenance",
    "information_relevance_weighting": "Info.-relevance weighting",
    "information_seeking_under_uncertainty": "Info. seeking (uncertainty)",
    "information_valuation": "Information valuation",
    "inhibition": "Inhibition",
    "local_global_weighting": "Local–global weighting",
    "novelty_valuation": "Novelty valuation",
    "other_state_representation": "Other-state representation",
    "perspective_transformation": "Perspective taking",
    "prediction_error_sensitivity": "Prediction-error sensitivity",
    "prior_strength": "Prior strength",
    "relational_representation": "Relational representation",
    "representation_maintenance": "Representation maintenance",
    "representation_updating": "Representation updating",
    "rule_updating": "Rule updating",
    "salience_assignment": "Salience assignment",
    "selective_attention": "Selective attention",
    "self_other_distinction": "Self–other distinction",
    "signal_noise_discrimination": "Signal–noise discrimination",
    "social_reward_weighting": "Social reward weighting",
    "switching_cost": "Switching cost",
    "uncertainty_estimation": "Uncertainty estimation",
}

# ---------------------------------------------------------------- 调色
# 维度配色与 Fig.1 完全一致（同一批表型维度在两图里同色），来源 `fig1_visualize.py`
NATURE = {
    "ink": "#1A1A1A",
    "muted": "#5F6B7A",
    "grid": "#E1E5EA",
    "panel_edge": "#C7D0D9",
    "dim": {
        "social_interaction_motivation": "#3A6EA5",
        "social_cognition": "#2E8B7F",
        "social_communication_pragmatics": "#D9822B",
        "cognitive_flexibility_adaptation": "#BF4E4A",
        "repetitive_restricted_behavior": "#8A6BBF",
        "novelty_uncertainty_sensory_behavior": "#C9A227",
        "general_cognition": "#9AA1A9",
    },
    "narrow": "#2E8B7F",
    "distributed": "#8A6BBF",
    "warn": "#B5413A",
}
C = NATURE
INK, MUTED, GRID, PE = C["ink"], C["muted"], C["grid"], C["panel_edge"]
DIMC = C["dim"]


def _truncate(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
    """截取一段现有色带（避免最浅端贴近白底而看不清）。逐字照抄 `fig1_visualize.py`。"""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_{lo}_{hi}", [base(v) for v in np.linspace(lo, hi, n)]
    )


#: 与 Fig.1 panel B 同一色带、同一截取区间：偏离重（0）→ 深，典型（1）→ 浅。
#: panel A 现在直接画 phenotype dimension 读数（0–1，越大越典型），与这条色带的语义一致。
HEAT_CMAP = _truncate("YlGnBu_r", 0.03, 0.82)

# 字号：**正文只有一个字号**。原来是 7.2 / 6.6 / 6.2 / 5.5 四档，最小那档在 17 in 图幅上只有
# 0.32 pt/图宽，缩到期刊栏宽就快读不出来了。现在四档并成 `FS_BODY = 7.2`（比原来最小的那档
# 大 31%、比刻度那档大 9%），粗细仍按 Fig.1 的分工：刻度加粗，坐标轴标题与图例常规。
# 图题是唯一一处更大的字号（17 pt）——「统一」指正文各档，标题不并入正文。
FS_FIG = 17.0      # 图题（全图仅一处）
FS_BODY = 7.2      # 正文：坐标轴标题 / 刻度 / 图内注记，全部同一档
FS_DOSE = 10.0     # 剂量标签 θ = 0.40 / 0.80：比正文大一号、加粗（它是读图时的入口，不是注记）
#: panel F 那份**共用图例**：比正文那档再大一档（8.6 pt = 正文的 1.19 倍）。
#: 这是全图唯一一处「图例比正文大」的地方，与 §0 那条「正文只有一个字号」的规矩是例外关系 ——
#: 那 6 个维度名是 panel F 唯一的编码说明（面板上没有标题、没有别的标注），放大是为了让它
#: 在缩到期刊栏宽之后还读得出来。别的面板的图例仍是 FS_BODY（7.2 pt），没有跟着一起放大。
FS_F_LEGEND = 8.6
FS_SMALL = FS_BODY
FS_TINY = FS_BODY
FS_TICK = FS_BODY

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
    }
)


def unify_axis_ticks(fig):
    """全图刻度标签统一字号 + 加粗。必须在 ``fig.canvas.draw()`` 之后调用。

    与 Fig.1 完全同一条规矩：**全图只有一个刻度字号**（``FS_TICK`` = 6.6 pt，加粗），
    数值轴和分类轴都不例外。31 个 primitive 名与竖向列头靠放大图幅、预留槽宽来容纳，
    而不是把字缩小 —— 小字正是本图要避免的。
    """
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for lab in axis.get_ticklabels():
                lab.set_fontsize(FS_TICK)
                lab.set_fontweight("bold")


# =================================================================== 数据层


def _dose_series(subject: pd.DataFrame, col: str) -> dict[tuple[str, float], list[float]]:
    """逐被试读数 → ``{(primitive, dose): [读数…]}``。"""
    out: dict[tuple[str, float], list[float]] = {}
    for setting, group in subject.groupby("mind_setting"):
        parts = str(setting).split()
        if len(parts) != 2 or parts[0] == "generic_degradation":
            continue                                  # 通用退化对照不是 primitive 扰动
        out[(parts[0], round(float(parts[1]), 4))] = [
            float(v) for v in group[col].tolist() if v is not None and not pd.isna(v)
        ]
    return out


def _shares(em: dict, i: int) -> list[float]:
    """一个 primitive 的受损量在 6 个表型维度上的**占比** p（∑p = 1）。

    占比是 panel C 的纵轴（C = max p）与 panel D 的横轴（D_eff = exp H）共用的底料，
    两者都只看「受损量怎么分布」，不引入任何阈值。
    """
    vals = [em[dim][i] for dim in DIMS]
    tot = sum(vals)
    return [v / tot for v in vals]


def _d_eff(p: list[float]) -> float:
    """有效表型维度数 D_eff = exp(H)，H 为占比 p 的香农熵 —— 1 = 只坏一个维度，6 = 六个维度坏得一样多。"""
    return math.exp(-sum(q * math.log(q) for q in p if q > 0))


def _t_p(perturbed: list[float], reference: list[float], equal_var: bool = True) -> float:
    """两样本 t 检验的 p 值：该 primitive 的 3 次扰动重复 vs 参照组的重复读数。

    「考虑三次重复」就是在这里落地的：不是拿 3 次的均值去跟 baseline 常数比（那样组内
    方差为零、任何微小差异都无从判断），而是把 3 次读数当成 3 个观测，与参照组的观测一起
    进标准误。`equal_var=True` 是合并方差那一版（自由度 (3−1)+(5−1) = 6，与两组单因素
    ANOVA 完全等价，load() 里有断言核对）；`equal_var=False` 是 Welch 版，自由度由
    Welch–Satterthwaite 公式降到 2.4 上下 —— 本数据扰动组的方差远大于参照组，
    两个版本给出的显著个数相差很大（141 vs 61），所以 Welch 一版逐格落盘供口径核对；
    但**图上的第二根柱子用的是同一套合并方差 p 值收严到 0.01**，不是 Welch。
    重复数不足就返回 1.0（画出来就是「没有显著差异」），不让缺数据变成伪显著。
    """
    if len(perturbed) < TEST_MIN_REPLICATES or len(reference) < TEST_MIN_REPLICATES:
        return 1.0
    # 有 3/372 个格子的 3 次扰动重复完全相同（如 confidence_calibration@0.8 的 [0.85, 0.85, 0.85]），
    # scipy 会为「矩计算精度损失」发一条 RuntimeWarning；那时扰动组方差为 0，合并方差全部
    # 来自参照组，p 值仍然有限、仍然可解释（已逐格核对：无穷 p 0 个、p < 1e-300 的 0 个）。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Precision loss occurred in moment.*")
        p = float(ttest_ind(perturbed, reference, equal_var=equal_var).pvalue)
    return 1.0 if math.isnan(p) else p


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg：一族 p 值 → 同一顺序的 q 值（单调、截尾在 1）。

    每个剂量要做 31 primitive × 6 维度 = 186 次检验，不校正就把「显著」当结论会偏松，
    所以按「同一剂量 × 同一维度」的 31 个检验成一族做 FDR —— 每一族正好对应条形图里
    那一根柱子（「这个维度上有多少个 primitive 显著」），族的大小与要报告的结论对齐。
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for k in range(n - 1, -1, -1):                     # 从最大 p 往回扫，保证单调
        prev = min(prev, p[order[k]] * n / (k + 1))
        q[order[k]] = prev
    return q


def _intact_reference() -> dict[str, list[float]]:
    """参照组 = 同一模型下 intact 的**逐次读数**（`result_1` 里 `axis1_same_brain` × intact 的 5 行）。

    两处用它：panel E 的两样本 t 检验（需要组内方差），以及 panel F 的「完整心智」那一点
    （x = 1 − θ = 0 处的均值与 SEM）。两处用的是同一批 5 份完整心智剖面，不是各取一套。
    """
    ref = pd.read_excel(REF_XLSX, "by_subject")
    ref = ref[
        ref["row_id"].astype(str).str.startswith(REF_ROW_PREFIX)
        & (ref["mind_setting"] == REF_SETTING)
    ]
    ref = ref.rename(
        columns={
            col: dim
            for col in ref.columns
            for dim in DIMS + [CONTROL]
            if col.startswith(dim + " [")
        }
    )
    return {
        dim: [float(v) for v in ref[dim].tolist() if v is not None and not pd.isna(v)]
        for dim in DIMS + [CONTROL]
    }


def load() -> dict:
    summ = pd.read_excel(XLSX, "summary_by_cell")
    subject = pd.read_excel(XLSX, "by_subject")
    rename = {}
    for col in subject.columns:                        # 逐被试表的列名带中文后缀，剥掉
        for dim in DIMS + [CONTROL]:
            if col.startswith(dim + " ["):
                rename[col] = dim
    subject = subject.rename(columns=rename)

    # baseline = 逐行 *_mean − *_delta，应当恒定；不恒定就说明口径理解错了
    baseline = {}
    for dim in DIMS + [CONTROL]:
        vals = (summ[f"{dim}_mean"] - summ[f"{dim}_delta"]).round(6).unique()
        assert len(vals) == 1, f"{dim} 的 baseline 不唯一：{vals}"
        baseline[dim] = float(vals[0])

    # 参照组（panel E 的 t 检验用）：intact 重复的**逐次读数**，不是一个均值。
    # 下面这条断言是这次改判据的关键依据 —— 参照组均值必须与上面的 baseline 常数一致，
    # 否则「用哪批数据当参照」就变了口径，宁可报错也不能画出去。
    reference = _intact_reference()
    for dim, vals in reference.items():
        assert len(vals) >= TEST_MIN_REPLICATES, f"参照组 {dim} 只有 {len(vals)} 次重复"
        ref_mean = sum(vals) / len(vals)
        assert abs(ref_mean - baseline[dim]) < 1e-9, (
            f"参照组 {dim} 的均值 {ref_mean:.6f} ≠ baseline {baseline[dim]:.6f}"
        )

    per_dose, given = {}, {}
    for dose in (PRIMARY_DOSE, SECONDARY_DOSE):
        block = summ[summ["dose"] == dose].set_index("primitive_id")
        em = {dim: (-block[f"{dim}_delta"]).astype(float).tolist() for dim in DIMS}
        em_ctrl = (-block[f"{CONTROL}_delta"]).astype(float).tolist()
        reps = {dim: _dose_series(subject, dim) for dim in DIMS}
        given[float(dose)] = {pid: str(block.loc[pid, "specificity_verdict"]) for pid in block.index}

        # panel E 的边判据：逐 (primitive, 维度) 的 t 检验 p → 按维度成一族做 BH-FDR。
        # 三个版本都算：合并方差（主判据，画边、节点数字、以及收严到 0.01 的那根浅色柱）
        # 与 Welch（不再上柱，只逐格落盘供口径核对）。
        # q 存在行里（metrics 也要落盘），显著与否 = q < α 且方向为「受损」（Δ > 0）。
        pval = {
            dim: np.array(
                [_t_p(reps[dim].get((pid, float(dose)), []), reference[dim], True)
                 for pid in block.index],
                dtype=float,
            )
            for dim in DIMS
        }
        pval_welch = {
            dim: np.array(
                [_t_p(reps[dim].get((pid, float(dose)), []), reference[dim], False)
                 for pid in block.index],
                dtype=float,
            )
            for dim in DIMS
        }
        qval = {dim: _bh_fdr(pval[dim]) for dim in DIMS}
        qval_welch = {dim: _bh_fdr(pval_welch[dim]) for dim in DIMS}

        # 口径自检：两组时等方差 t 检验 = 单因素 ANOVA，F = t²。逐格核对，相等才继续。
        for dim in DIMS:
            for i, pid in enumerate(block.index):
                series = reps[dim].get((pid, float(dose)), [])
                if len(series) < TEST_MIN_REPLICATES:
                    continue
                f_p = float(f_oneway(series, reference[dim]).pvalue)
                assert abs(f_p - pval[dim][i]) < 1e-9, (
                    f"{pid}@{dose}×{dim}：t 检验 p {pval[dim][i]:.3e} ≠ ANOVA F 检验 p {f_p:.3e}"
                )

        rows = []
        for i, pid in enumerate(block.index):
            order = sorted(range(len(DIMS)), key=lambda j: (-em[DIMS[j]][i], DIMS[j]))
            dom_j, sec_j = order[0], order[1]
            dom, sec = em[DIMS[dom_j]][i], em[DIMS[sec_j]][i]
            series = reps[DIMS[dom_j]].get((pid, float(dose)), [])
            se = stdev(series) / math.sqrt(len(series)) if len(series) >= 3 else 0.0
            rows.append(
                {
                    "primitive_id": pid,
                    "en": EN.get(pid, pid),
                    "zh": str(block.loc[pid, "primitive_zh"]),
                    #: 直接读 xlsx 的逐通道读数（0–1，越大越典型）—— panel A 画的就是它，不做任何翻转
                    "reading": [float(block.loc[pid, f"{dim}_mean"]) for dim in DIMS],
                    "emergence": [em[dim][i] for dim in DIMS],
                    "control": em_ctrl[i],
                    "control_reading": float(block.loc[pid, f"{CONTROL}_mean"]),
                    "max_effect": max(em[dim][i] for dim in DIMS),
                    "dominant_dim": DIMS[dom_j],
                    "dominant_effect": dom,
                    "second_dim": DIMS[sec_j],
                    "second_effect": sec,
                    "concentration": dom - sec,
                    "margin": max(CONCENTRATION_MARGIN, 2.0 * se),
                    "hit": dom >= TARGET_DROP,
                    "control_ok": em_ctrl[i] <= CONTROL_TOLERANCE,
                    "n_affected": sum(1 for dim in DIMS if em[dim][i] >= TARGET_DROP),
                    #: 受损量在六个维度上的占比 p（和为 1）。下面两个连续量都由它算，
                    #: 与实验侧的 `concentration`（差值，带门槛）无关，见 `figure2_metrics.json`。
                    "share": _shares(em, i),
                    "conc_share": max(_shares(em, i)),
                    "d_eff": _d_eff(_shares(em, i)),
                    #: t 检验结果（panel E）：逐维度 q 值与「显著」标记。显著 = q < α 且 Δ > 0；
                    #: 两个剂量下都没有出现「显著但方向相反（Δ < 0）」的格子，这一条是守卫。
                    #: 画边与节点数字用合并方差那一版（`sig`）；`sig_strict` 是**同一套 q 值**把等级
                    #: 收到 TEST_ALPHA_STRICT 的结果（条形图浅色柱），恒为 `sig` 的子集；Welch 一版
                    #: 仍逐格落盘供口径核对，但不再上柱。
                    "p_t": [float(pval[dim][i]) for dim in DIMS],
                    "p_welch": [float(pval_welch[dim][i]) for dim in DIMS],
                    "q_t": [float(qval[dim][i]) for dim in DIMS],
                    "q_welch": [float(qval_welch[dim][i]) for dim in DIMS],
                    "sig": [bool(qval[dim][i] < TEST_ALPHA and em[dim][i] > 0) for dim in DIMS],
                    "sig_strict": [
                        bool(qval[dim][i] < TEST_ALPHA_STRICT and em[dim][i] > 0) for dim in DIMS
                    ],
                    "sig_welch": [
                        bool(qval_welch[dim][i] < TEST_ALPHA and em[dim][i] > 0) for dim in DIMS
                    ],
                    "n_sig": sum(1 for dim in DIMS if qval[dim][i] < TEST_ALPHA and em[dim][i] > 0),
                    "n_sig_strict": sum(
                        1 for dim in DIMS
                        if qval[dim][i] < TEST_ALPHA_STRICT and em[dim][i] > 0
                    ),
                    "n_sig_welch": sum(
                        1 for dim in DIMS if qval_welch[dim][i] < TEST_ALPHA and em[dim][i] > 0
                    ),
                    "n_sig_uncorrected": sum(
                        1 for dim in DIMS if pval[dim][i] < TEST_ALPHA and em[dim][i] > 0
                    ),
                }
            )
        per_dose[float(dose)] = rows

    def verdict(row: dict) -> str:
        """与 `pacs/analysis/specificity.py` 同一套分支。"""
        if not row["control_ok"]:
            return "fail_control" if row["hit"] else "fail_both"
        if not row["hit"]:
            return "no_effect"
        return "pass" if row["concentration"] >= row["margin"] else "fail_diffuse"

    for dose, rows in per_dose.items():
        for row in rows:
            row["verdict"] = verdict(row)
            if row["verdict"] != given[dose][row["primitive_id"]]:
                raise AssertionError(
                    f"{row['primitive_id']}@{dose}: 重算判定 {row['verdict']} "
                    f"≠ xlsx 落盘 {given[dose][row['primitive_id']]}"
                )

    rows = per_dose[PRIMARY_DOSE]
    dom_count = {dim: 0 for dim in DIMS}
    for row in rows:
        dom_count[row["dominant_dim"]] += 1
    hist = {k: 0 for k in range(len(DIMS) + 1)}
    for row in rows:
        hist[row["n_affected"]] += 1
    vcount: dict[str, int] = {}
    for row in rows:
        vcount[row["verdict"]] = vcount.get(row["verdict"], 0) + 1

    # panel E 条形图的数字：每个剂量 × 每个维度有多少个 primitive「显著」。
    # 实心柱 = 主判据（合并方差 t 检验，画边与节点数字）；浅色柱 = 同一套 p 值收严到
    # TEST_ALPHA_STRICT（恒为实心柱的子集）。Welch 版与不做 FDR 校正的计数也落盘
    # （metrics.json 里都写，选哪个口径有据可查），只是不再上柱。
    sig_count = {
        float(dose): {dim: sum(1 for r in rws if r["sig"][j]) for j, dim in enumerate(DIMS)}
        for dose, rws in per_dose.items()
    }
    sig_count_strict = {
        float(dose): {dim: sum(1 for r in rws if r["sig_strict"][j]) for j, dim in enumerate(DIMS)}
        for dose, rws in per_dose.items()
    }
    sig_count_welch = {
        float(dose): {dim: sum(1 for r in rws if r["sig_welch"][j]) for j, dim in enumerate(DIMS)}
        for dose, rws in per_dose.items()
    }
    sig_count_p = {
        float(dose): {
            dim: sum(1 for r in rws if r["p_t"][j] < TEST_ALPHA and r["emergence"][j] > 0)
            for j, dim in enumerate(DIMS)
        }
        for dose, rws in per_dose.items()
    }
    n_no_sig = {
        float(dose): sum(1 for r in rws if not any(r["sig"])) for dose, rws in per_dose.items()
    }
    n_no_sig_strict = {
        float(dose): sum(1 for r in rws if not any(r["sig_strict"]))
        for dose, rws in per_dose.items()
    }
    n_no_sig_welch = {
        float(dose): sum(1 for r in rws if not any(r["sig_welch"]))
        for dose, rws in per_dose.items()
    }
    n_tests = len(rows) * len(DIMS)

    by8 = {r["primitive_id"]: r for r in per_dose[SECONDARY_DOSE]}
    stronger = sum(1 for r in rows if r["max_effect"] > by8[r["primitive_id"]]["max_effect"])

    # panel E 的行序与分块**两张网络共用**，一律按主剂量 0.4 的主导维度分组、组内按 0.4 的
    # 最大效应降序。理由与 panel B 的颜色规矩一样：两张网络并排是为了逐行比两个剂量，
    # 各档各自分组会让同一个 primitive 在两张图里落在不同的行上，逐行对读就失效了。
    e_order = [
        r["primitive_id"]
        for dim in DIMS
        for r in sorted([q for q in rows if q["dominant_dim"] == dim],
                        key=lambda q: -q["max_effect"])
    ]
    block_dim = {r["primitive_id"]: r["dominant_dim"] for r in rows}

    # panel A 的行序：对 0.4 档的 6 维读数做层次聚类（平均连接、欧氏距离），
    # 两个档位用**同一份行序**，这样左右两张热图逐行可比。
    mat4 = np.array([r["reading"] for r in rows], dtype=float)
    tree = linkage(pdist(mat4, metric="euclidean"), method="average")
    leaves = leaves_list(tree)
    if mat4[leaves[0]].mean() > mat4[leaves[-1]].mean():
        leaves = leaves[::-1]                       # 让受影响最重的一批落在顶部
    cluster_order = [rows[int(i)]["primitive_id"] for i in leaves]

    return {
        "baseline": baseline,
        "reference": reference,
        "rows": rows,
        "rows_by_dose": per_dose,
        "cluster_order": cluster_order,
        "e_order": e_order,
        "block_dim": block_dim,
        "dom_count": dom_count,
        "hist": hist,
        "verdict_count": vcount,
        "sig_count": sig_count,
        "sig_count_strict": sig_count_strict,
        "sig_count_welch": sig_count_welch,
        "sig_count_uncorrected": sig_count_p,
        "n_no_sig": n_no_sig,
        "n_no_sig_strict": n_no_sig_strict,
        "n_no_sig_welch": n_no_sig_welch,
        "n_tests": n_tests,
        "n_stronger_at_0.4": stronger,
        "n_subjects": int(len(subject)),
        "n_given_verdicts_checked": sum(len(v) for v in given.values()),
    }


# ---------------------------------------------------------------- panel F 的数据层


def _sem(values: list[float]) -> float:
    """样本均值的标准误（ddof=1）。重复数 < 2 时返回 0（这种格子不该出现，load_dose 里有断言）。"""
    vals = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def load_dose(baseline_ref: dict[str, float]) -> dict:
    """panel F 的数据：`result_4_does_response.xlsx` 的剂量阶梯（5 primitive × 5 θ × 5 重复）。

    - **画的是均值读数**：`summary_by_cell` 的 `{dim}_mean`，即该 (primitive, θ, 通道) 上
      5 次重复的均值 —— 0–1，**越大越典型**，与 panel A 同一口径，不翻转、不做差分。
    - **误差棒 = ±1 SEM**（`by_subject` 逐被试读数，ddof=1），不是 SD：5 张子图要跨 primitive
      比「同一 θ 下剖面的差异」，SD 会把「重复之间的离散度」与「维度之间的差异」混在一起，
      SEM 才是「这个均值估得准不准」的那把尺。
    - **逐格核对**：`by_subject` 按 (primitive, θ) 聚出来的均值必须与 `summary_by_cell` 的
      `{dim}_mean` 一致（容差 1e-4，即该列的舍入位数）。图里画的均值与表里落盘的均值
      必须是同一个数，不一致就报错而不是悄悄画一张对不上的图。
    - **完整心智那一点**（横轴的 1 − θ = 0）：本表的 `*_mean − *_delta` 反解出的 baseline 是一份
      恒定剖面（与 result_2 那份逐位相同 —— 传进来的 `baseline_ref` 逐个核对，误差 < 1e-9），
      它的均值与 SEM 取自**参照组那 5 份完整心智的逐次读数**（`_intact_reference()`，
      与 panel E 的检验是同一批）；也就是说 x = 0 处不是阶梯实验的某个格子，是完整心智本身。
    """
    summ = pd.read_excel(XLSX4, "summary_by_cell")
    subject = pd.read_excel(XLSX4, "by_subject")
    subject = subject.rename(
        columns={col: dim for col in subject.columns for dim in DIMS + [CONTROL]
                 if col.startswith(dim + " [")}
    )

    assert {round(float(d), 6) for d in summ["dose"]} == {round(d, 6) for d in DOSE_LADDER}, (
        f"剂量档位与 DOSE_LADDER 不一致：{sorted(set(summ['dose']))}"
    )
    assert set(summ["primitive_id"]) == set(LADDER_PRIMITIVES), (
        f"primitive 集合与 LADDER_PRIMITIVES 不一致：{sorted(set(summ['primitive_id']))}"
    )

    # 完整心智剖面：先由本表反解，再与 result_2 那份 baseline 逐位核对 —— 两份表用的是
    # 同一批 5 份完整心智剖面（阶梯表自己也是这么写的），对不上就说明基线换了、不能画。
    intact_profile = {}
    for dim in DIMS + [CONTROL]:
        vals = (summ[f"{dim}_mean"] - summ[f"{dim}_delta"]).round(6).unique()
        assert len(vals) == 1, f"result_4 的 {dim} 反解出的完整心智值不唯一：{vals}"
        intact_profile[dim] = float(vals[0])
        assert abs(intact_profile[dim] - baseline_ref[dim]) < 1e-9, (
            f"{dim}：result_4 反解出的完整心智 {intact_profile[dim]:.6f} ≠ "
            f"result_2 的 baseline {baseline_ref[dim]:.6f}"
        )
    reference = _intact_reference()
    intact = {
        dim: {"mean": intact_profile[dim], "sem": _sem(reference[dim]), "n": len(reference[dim])}
        for dim in DIMS + [CONTROL]
    }

    cells: dict[str, dict[float, dict]] = {}
    n_rep_min, n_checked = None, 0
    for pid in LADDER_PRIMITIVES:
        cells[pid] = {}
        for dose in DOSE_LADDER:
            sel = summ[(summ["primitive_id"] == pid) & (summ["dose"].round(6) == round(dose, 6))]
            assert len(sel) == 1, f"{pid} @ θ={dose} 在 summary_by_cell 里不是唯一一行"
            row = sel.iloc[0]
            block = subject[subject["mind_setting"] == f"{pid} {dose:.2f}"]
            assert len(block) == int(row["n_replicates"]) >= 3, (
                f"{pid} @ θ={dose}：by_subject 有 {len(block)} 行，"
                f"summary 写的是 {int(row['n_replicates'])} 次重复"
            )
            n_rep_min = len(block) if n_rep_min is None else min(n_rep_min, len(block))
            chans = {}
            for dim in DIMS + [CONTROL]:
                reps = [float(v) for v in block[dim].tolist()]
                mean_sub = sum(reps) / len(reps)
                mean_sum = float(row[f"{dim}_mean"])
                assert abs(mean_sub - mean_sum) <= 1e-4, (
                    f"{pid} @ θ={dose} 的 {dim}：by_subject 均值 {mean_sub:.6f} ≠ "
                    f"summary 的 {mean_sum:.6f}"
                )
                n_checked += 1
                chans[dim] = {"mean": mean_sum, "sem": _sem(reps), "n": len(reps)}
            cells[pid][float(dose)] = {
                "channels": chans,
                "n_replicates": len(block),
                "dominant_dim": str(row["dominant_dim"]),
            }

    return {
        "cells": cells,
        "primitive_order": list(LADDER_PRIMITIVES),
        "doses": list(DOSE_LADDER),
        "n_replicates_per_cell": n_rep_min,
        "n_means_checked": n_checked,
        "n_subjects": int(len(subject)),
        "intact": intact,
        "baseline_check": {dim: round(intact_profile[dim], 6) for dim in DIMS + [CONTROL]},
    }


# =================================================================== 绘图层


def heat_order(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: -r["max_effect"])


def check_no_gray_text(fig) -> list[tuple[str, str]]:
    """版式自检：图里**不许再出现灰色文字**。

    本图不使用灰色文字解释 —— 编码的含义改由坐标轴标题与图例（黑字）承担。
    灰字很容易在后续改动里悄悄回来（顺手给某个注记套一个 ``color=MUTED``），
    所以在 draw 之后把所有 Text 实体的颜色过一遍：低彩度 + 中等明度即判为灰。
    纯黑（near-black）与纯白不算灰，因此黑色正文、圆内白字都不会被误报。
    """
    def is_gray(color) -> bool:
        try:                                   # get_color() 可能是名字串、可能是 RGBA 元组
            r, g, b = to_rgba(color)[:3]
        except (ValueError, TypeError):
            return False
        chroma = max(r, g, b) - min(r, g, b)
        value = max(r, g, b)
        return chroma < 0.18 and 0.25 < value < 0.88

    found = []
    for ax in fig.axes:
        items = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis in (ax.xaxis, ax.yaxis):
            items += list(axis.get_ticklabels())
        if ax.get_legend() is not None:
            items += ax.get_legend().get_texts()
            items.append(ax.get_legend().get_title())
        for t in items:
            if t.get_text().strip() and is_gray(t.get_color()):
                found.append((t.get_text().replace("\n", " ")[:40], str(t.get_color())))
    # panel E 的图例挂在 figure 上（两张网络 + 两张条形图共用一份），不在 fig.axes[...].legends 里
    for leg in fig.legends:
        for t in list(leg.get_texts()) + [leg.get_title()]:
            if t.get_text().strip() and is_gray(t.get_color()):
                found.append((t.get_text().replace("\n", " ")[:40], str(t.get_color())))
    for t in fig.texts:
        if t.get_text().strip() and is_gray(t.get_color()):
            found.append((t.get_text().replace("\n", " ")[:40], str(t.get_color())))
    return found


def check_text_overlaps(fig) -> list[tuple[str, str]]:
    """版式自检：图级标题 / 面板标题 / 脚注 / 坐标轴标题 / **刻度标签**互相压字就报出来。

    这张图靠肉眼盯会漏，而且是连着漏了三次：第一版漏「panel A 副标题压在 panel B 副标题上」，
    第二版漏「panel B 的 xlabel 顶到 panel E 的标题」（那是 ``ax.set_xlabel`` 不是 ``fig.text``），
    第三版才意识到**刻度标签也不在** ``fig.texts`` 里 —— 而本图最长的文字恰恰是
    panel A / panel B 那些竖排的维度名（最长一句 33 字符）。所以刻度标签一并收进来。
    在 ``fig.canvas.draw()`` 之后直接量渲染包围盒。只报告、不阻断。
    """
    renderer = fig.canvas.get_renderer()
    boxes = []
    for t in fig.texts:
        if t.get_text().strip():
            boxes.append((t.get_text().replace("\n", " ")[:46], t.get_window_extent(renderer), None))
    # figure 级图例（panel E 两张网络 + 两张条形图共用一份）的文字也要进来量
    for leg in fig.legends:
        for t in list(leg.get_texts()) + [leg.get_title()]:
            if t.get_text().strip():
                boxes.append((t.get_text().replace("\n", " ")[:46],
                              t.get_window_extent(renderer), None))
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts):
            if t.get_text().strip():
                boxes.append((t.get_text().replace("\n", " ")[:46], t.get_window_extent(renderer),
                              None))
        if not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_ticklabels():
                if t.get_text().strip():
                    boxes.append((t.get_text().replace("\n", " ")[:46],
                                  t.get_window_extent(renderer), axis))
    clashes = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # 只有「同一条轴上的两个相邻刻度」才豁免（0.0 / 0.2 / 0.4 本来就贴在一起）。
            # 轴内文字（panel B 的标注、panel C 的代表名）之间**不豁免** —— 它们叠在一起就是错。
            if boxes[i][2] is not None and boxes[i][2] is boxes[j][2]:
                continue
            hit = Bbox.intersection(boxes[i][1], boxes[j][1])
            if hit is not None and hit.width > 2.0 and hit.height > 2.0:
                clashes.append((boxes[i][0], boxes[j][0]))
    return clashes


def panel_a(fig, ax4, ax8, cb_ax, data):
    """表型扰动图：0.4 与 0.8 两个档位并排两张热图，同一份行序。

    - 格子值 = **phenotype dimension 的实测读数**（0–1，越大越典型），不翻转、不做差分。
    - 行序 = 对 0.4 档的 6 维读数做层次聚类（平均连接 / 欧氏距离）得到的叶序，上下都反向对齐成
      「受影响最重的在顶部」。0.8 档沿用同一行序，因此左右可逐行直接比。
    - 色带与 Fig.1 panel B 完全相同（`YlGnBu_r` 截取 0.03–0.82）：偏重 → 深，典型 → 浅。
    """
    rows = [r for pid in data["cluster_order"] for r in data["rows"] if r["primitive_id"] == pid]
    by8 = {r["primitive_id"]: r for r in data["rows_by_dose"][SECONDARY_DOSE]}
    n = len(rows)
    mat4 = np.array([r["reading"] for r in rows], dtype=float)
    mat8 = np.array([by8[r["primitive_id"]]["reading"] for r in rows], dtype=float)

    norm = Normalize(0.0, 1.0)                      # 与 Fig.1 panel B 同一口径：读数本身就是 0–1
    for ax, mat in ((ax4, mat4), (ax8, mat8)):
        ax.imshow(mat, aspect="auto", cmap=HEAT_CMAP, norm=norm,
                  interpolation="nearest", extent=(0, len(DIMS), n, 0))
        ax.set_xticks(np.arange(len(DIMS)) + 0.5)
        ax.set_xticklabels([DIM_1LINE[d] for d in DIMS], rotation=90,
                           fontsize=FS_TICK, fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_yticks([])
        ax.set_ylim(n, 0)
        ax.set_xticks(np.arange(0, len(DIMS) + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, n + 1, 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    # 行标签只挂在左侧那张上（两张图行序相同）；按主导维度上色
    ax4.set_yticks(np.arange(n) + 0.5)
    ax4.set_yticklabels([r["en"] for r in rows], fontsize=FS_TINY)
    ax4.tick_params(axis="y", length=0, pad=3)
    for lab, r in zip(ax4.get_yticklabels(), rows):
        lab.set_color(DIMC[r["dominant_dim"]])

    sm = ScalarMappable(norm=norm, cmap=HEAT_CMAP)
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(PE)
    cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(length=2, pad=2)
    # 色标说明改到色标**上方**横排（原来是右侧竖排）：正文统一放到 7.2 pt 后，右侧那一列
    # 竖排说明与 panel B 的纵轴刻度名挤在同一条 0.36–0.40 的窄带里（三样东西抢一条带子），
    # 挪到上方就只剩它自己；色标刻度名仍在右侧。
    cb.ax.set_title("Phenotype score\n(0 = most ASD-like, 1 = typical)", fontsize=FS_TINY,
                    linespacing=1.3, pad=4)


def panel_b(fig, ax4, ax8, data):
    """曼哈顿图：横轴 = 6 个 phenotype dimension（一列一个维度），纵轴 = 受损量 Δ（正 = 更像 ASD）。

    列内 31 个点 = 31 个 primitive；点色 = 该列维度。**每列在这列最高点之上依次列出该维度
    Δ 最大的 3 个 primitive**（顺序即名次，名字 + Δ 数值），板上对应的那个点套一个加粗环、
    环旁标同名次数字 —— 环/数字与列顶那三行文字互为索引，所以不需要引线。

    两档上下排（完整度 0.4 在上、0.8 在下），**共用同一纵轴尺度与同一套维度列**，
    于是维度名只在下面那张出现一次；列内中位数横杠与 Δ ≥ 0.15 门槛虚线两张共用，
    图例只在上面那张给一次。上下排而不是左右排，是为了把 6 列摊到整幅图宽上 ——
    每列三行标注最宽要 ~1.3 in，列宽不够就一定会和邻列的标注连成一片。
    """
    JITTER = 0.30          # 列内横向抖动幅度（1 个单位 = 1 列）
    LANE = 0.26            # 被标注的三个点在列内的固定 x：名次 1/2/3 = 左/中/右
    LANE_CLEAR = 0.055     # 其余点与这三条名次道的最小距离，保证环里只有被标注的那个点
    HEAD_PT = 36.0         # 面板顶部让给列顶三行标注的高度（环 4.3 + 空档 + 两行行距 + 半行字）
    panels = ((ax4, PRIMARY_DOSE, data["rows"]),
              (ax8, SECONDARY_DOSE, data["rows_by_dose"][SECONDARY_DOSE]))
    vmax = max(r["emergence"][k] for _, _, rows in panels for r in rows for k in range(len(DIMS)))
    # 标注的行距/空档按面板实际高度换算，这样改版式（面板变高变矮）不用回头调这几个数
    h_pt = LAYOUT["B8"][3] * fig.get_figheight() * 72.0
    ymax = vmax / (1.0 - HEAD_PT / h_pt)
    ROW_DY = 9.0 * (ymax / h_pt)          # 三行标注的行距（数据单位）
    TOP_GAP = 13.0 * (ymax / h_pt)        # 第一行标注与列内最高点之间的距离
    # 名次数字（半宽/半高）、环（半径）、散点（半径）的碰撞盒，都用数据单位：x 是列宽单位，
    # 1 列 = 面板宽度/6，y 是 Δ 单位。数值按 dpi 200 下实测的点径/字宽折算。
    RANK_HALF = (0.020, 0.017)
    RING_HALF = (0.036, 0.022)   # 环 = s 54 的散点：直径 8.3 pt，一列宽 1.59 in
    DOT_HALF = (0.016, 0.009)
    # 数据单位 → pt 的换算（横向按列宽，纵向按 Δ 轴），用来判断数字离哪个环更近
    pt_per_col = LAYOUT["B8"][2] * fig.get_figwidth() * 72.0 / len(DIMS)
    pt_per_dy = h_pt / ymax
    rng = np.random.default_rng(20260917)
    #: 标注文字的颜色按 primitive 的**主剂量（0.4）主导维度**取 —— 与 panel A 的行标签同一套颜色。
    #: 不能按各档自己的主导维度取：同一个 primitive 在两个剂量下的主导维度会变（本图里有 3 个），
    #: 那样同一个原语会在上下两张面板里变成两种颜色，「一原语一颜色」这条编码就断了。
    label_col = {r["primitive_id"]: DIMC[r["dominant_dim"]] for r in data["rows"]}
    #: (dose, 维度名) -> [(名次, primitive 英文名, Δ, 列内 x, y), …]，供 metrics 落盘
    top3: dict[tuple, list[tuple]] = {}
    data["b_top3"] = top3

    for ax, dose, rows in panels:
        for k, dim in enumerate(DIMS):
            vals = [float(r["emergence"][k]) for r in rows]
            col = DIMC[dim]
            # 这一列 Δ 最大的 3 个 primitive = 本列被标注的那三个
            order = sorted(range(len(rows)), key=lambda i: -vals[i])[:3]
            # 它们的位置**固定**成列内左/中/右三条名次道：抖动是随机的，两个 Δ 只差 0.01 的点
            # 会抖到 20 px 以内，两个环就糊成一团、名次数字也没处放。
            lane = {i: k + LANE * (rank - 1) for rank, i in enumerate(order)}
            xs = np.empty(len(rows))
            for i in range(len(rows)):
                if i in lane:
                    xs[i] = lane[i]
                    continue
                while True:                       # 其余点避开这三条道，免得有散点落进环里
                    x = k + rng.uniform(-JITTER, JITTER)
                    if all(abs(x - lane[j]) > LANE_CLEAR for j in lane):
                        break
                xs[i] = x
            ax.scatter(xs, vals, s=10, facecolor=col, edgecolor="white", linewidth=0.35,
                       zorder=4)
            ax.plot([k - 0.22, k + 0.22], [float(np.median(vals))] * 2, color=INK,
                    linewidth=1.5, zorder=6, solid_capstyle="butt")

            # 每列在这列最高点之上依次列出该维度 Δ 最大的 3 个 primitive（顺序即名次）。
            # 文字整段落在本列的横向范围里，所以不会碰到邻列的标注，也不会压到本列的点。
            top3[(dose, dim)] = []
            placed: list[tuple[float, float]] = []      # 已经放下的名次数字落点
            for rank, i in enumerate(order, 1):
                r = rows[i]
                # 标注文字与 panel A 的 primitive 行标签同规格：同字号、同加粗、同配色 ——
                # 按该 primitive 的主剂量主导维度上色，而不是按它所在的列。
                # **不再写行首的名次数字**：正文统一放大到 7.2 pt 后，
                # "2  Prediction-error sensitivity  0.64" 这类最长的一行要 1.65 in，
                # 而一列只有 1.59 in，三行会与邻列的三行挤成一片。名次本来就由
                # 「列表自上而下 = 第 1/2/3 名」和板上环旁的那个数字共同承担，行首的
                # 数字是重复的一份 —— 去掉它，最长一行降到 1.5 in，列内放得下。
                ax.text(k, vals[order[0]] + TOP_GAP + (3 - rank) * ROW_DY,
                        f"{r['en']} {vals[i]:.2f}",
                        fontsize=FS_TICK, fontweight="bold", color=label_col[r["primitive_id"]],
                        ha="center", va="center", zorder=8)
                px, py = float(lane[i]), vals[i]
                top3[(dose, dim)].append((rank, r["en"], round(vals[i], 4), px, py))
                ax.scatter([px], [py], s=54, facecolor="none", edgecolor=INK, linewidth=1.0,
                           zorder=6)

                # 名次数字：候选落点依次试，要求既不压到已放的数字，也不落进任何一个名次环、
                # 任何一个散点里。
                def free(qx: float, qy: float) -> bool:
                    def apart(a, b, half):
                        return abs(qx - a) > half[0] + RANK_HALF[0] or abs(qy - b) > half[1] + RANK_HALF[1]
                    if not all(apart(a, b, (0.0, 0.0)) for a, b in placed):
                        return False
                    if not all(apart(lane[j], vals[j], RING_HALF) for j in lane):
                        return False
                    return all(apart(x, y, DOT_HALF) for x, y in zip(xs, vals))

                for ox, oy in ((0.085, 0.038), (-0.085, 0.038), (0.085, -0.045), (-0.085, -0.045),
                               (0.115, 0.038), (-0.115, 0.038), (0.085, 0.100), (-0.085, 0.100),
                               (0.155, 0.038), (-0.155, 0.038), (0.085, 0.162), (-0.085, 0.162)):
                    if free(px + ox, py + oy):
                        break
                # 数字必须离自己的环最近 —— 否则读者会把它认到隔壁名次道上。
                own = math.hypot(ox * pt_per_col, oy * pt_per_dy)
                for j in lane:
                    if abs(lane[j] - px) < 1e-9:
                        continue
                    other = math.hypot((px + ox - lane[j]) * pt_per_col, (py + oy - vals[j]) * pt_per_dy)
                    if other < own:
                        raise AssertionError(
                            f"panel B：{dim} 列的名次数字 {rank} 离别的环更近（{other:.1f} < {own:.1f} pt）")
                placed.append((px + ox, py + oy))
                ax.text(px + ox, py + oy, str(rank), fontsize=FS_TINY, fontweight="bold",
                        color=INK, ha="center", va="center", zorder=8)

        ax.set_xticks(range(len(DIMS)))
        ax.set_xlim(-0.5, len(DIMS) - 0.5)
        ax.set_ylim(0.0, ymax)
        # 不画 Δ ≥ 0.15 的门槛虚线：横向参考只剩两张面板各自的 y 刻度。
        # 那个 0.15 仍然活在数据层（verdict / 受影响维度数的判定口径），只是不再画进图里；
        # panel E 的边判据本来就是另一套（t 检验 + BH-FDR），两条口径不同，图上都不标反而更干净。
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # 两档共用一套维度列：名字只在下面那张给一次，上面那张连刻度线都不画
    ax4.tick_params(axis="x", length=0, labelbottom=False)
    ax8.set_xticklabels([DIM_2L[d] for d in DIMS], rotation=90,
                        fontsize=FS_TICK, fontweight="bold")
    # 纵轴标题也是两张共用一个：画在两块面板的垂直中点上，而不是挂在上面那一张上
    # （挂在上面那张时，读下面那张就得回头找纵轴标题）。刻度数字仍然两张各自保留。
    fig.text(LAYOUT["B8"][0] - 0.024,
             (LAYOUT["B4"][1] + LAYOUT["B4"][3] + LAYOUT["B8"][1]) / 2.0,
             "Phenotype impairment   Δ = baseline − perturbed",
             fontsize=FS_BODY, fontweight="bold", color=INK, rotation=90,
             ha="center", va="center")
    ax4.legend(
        handles=[
            Line2D([], [], color=INK, linewidth=1.5, label="median"),
            Line2D([], [], color="none", marker="o", markersize=5.5, markerfacecolor="none",
                   markeredgecolor=INK, markeredgewidth=1.0,
                   label="top 3 of the dimension  (numbered)"),
        ],
        loc="upper left", frameon=False, fontsize=FS_SMALL, handlelength=1.2,
        handletextpad=0.45, labelspacing=0.4, borderaxespad=0.2, alignment="left",
    )


def panel_e(fig, ax, ax_bar, rows, dose, sig_count, sig_count_strict, bar_xmax, e_order, block_dim):
    """二部网络（**一个剂量一张**）：左 31 primitives（按主导维度分块）→ 右 6 表型维度。

    **行序与分块一律用主剂量 0.4 的主导维度**（`e_order` / `block_dim`），不用本档自己的：
    两张网络并排就是为了逐行比两个剂量，各档各自分组会让同一个 primitive 在两张图里落到
    不同的行上（0.8 档的分块也会散掉），逐行对读就成了空话。这与 panel B「颜色一律取主剂量
    主导维度」是同一条规矩：一个 primitive 在整张图里只有一个位置、一种颜色。

    **边判据 = t 检验，不再是 Δ 门槛**：逐 (primitive, 表型维度) 做一次两样本 t 检验
    （该 primitive 的 3 次重复 vs 参照组的 intact 重复，见 `load()`），在同一剂量 × 同一
    维度的 31 个检验内做 BH-FDR，q < `TEST_ALPHA` 且方向为受损（Δ > 0）才连线。线的粗细
    仍然是 Δ（效应量）—— 「显著」与「多大」分成两条编码，混在一起会读不出来。

    网络右侧挂一张同剂量的条形图：**每个表型维度上有多少个 primitive 显著**，也就是该
    右端节点的入边数 —— 节点里的数字与实心柱子高度是同一个数，节点面积也按它缩放。
    每行两根柱子：实心 = 主判据（合并方差 t 检验 + BH-FDR，q < `TEST_ALPHA`），浅色 =
    **同一套 q 值收严到 `TEST_ALPHA_STRICT`**，所以浅色柱恒为实心柱的子集，两柱之差读的是
    「结论在更严阈值下还剩多少」（存活率），不是换一套检验。

    节点位置由分块决定（块大小 = 主剂量下以该维度为主导的原语数），「显著个数」是入边数，
    两者是不同的量，已在图注里分开写清。
    """
    by_pid = {r["primitive_id"]: r for r in rows}
    grouped = [by_pid[pid] for pid in e_order]
    span = {dim: sum(1 for pid in e_order if block_dim[pid] == dim) for dim in DIMS}
    y_left = {pid: (len(e_order) - 1 - i) for i, pid in enumerate(e_order)}

    # 右端节点的纵向位置**落在左端行格的整数行上**（取每块的中点行），不取半个行距的分数位：
    # 这样左端 31 个节点与右端 6 个节点共用同一条行格子线，左右节点的高度看得出是对齐的；
    # 取分数位时右端节点会悬在两条左端行之间，读起来像错位。
    # 块间**不留空**（原来是 0.9 行）：留空会让六块累计下移，最下面那个节点（新颖性/感觉）
    # 落到 −3 行 —— 低于左端最后一行，就不再「和某个 primitive 同高」了；分块本来就由
    # 左端名字的颜色和右端节点的颜色看得出来，不需要靠空档。
    gap, cursor, y_right = 0.0, len(e_order) - 1, {}
    for dim in DIMS:
        y_right[dim] = float(round(cursor - (span[dim] - 1) / 2.0))
        cursor -= span[dim] + gap

    x_l, x_r = 0.0, 1.0
    # 上下留白收到「刚好装得下墨迹」：下边只留 0.9 行（最低那个右端节点在 2 行、半径约 1.6 行，
    # 最低左端节点在 0 行、半径 0.3 行），上边留 2.4 行（最高右端节点在 28 行 + 1.6 行半径，
    # 左端最高在 30 行）。下面留 4.6 行 —— 那一截原本给图例让位，不设图例后就成了
    # 一条空白，网络看上去比旁边的条形图矮；现在墨迹铺满坐标区，两者可见高度才一致。
    y_lo = min(min(y_left.values()), min(y_right.values())) - 0.9
    y_hi = max(max(y_left.values()), max(y_right.values())) + 2.4

    n_none = 0
    for r in grouped:
        y = y_left[r["primitive_id"]]
        dim_c = DIMC[block_dim[r["primitive_id"]]]        # 该 primitive 的颜色：主剂量主导维度
        drew = False
        for j, dim in enumerate(DIMS):
            if not r["sig"][j]:
                continue                                     # 该维度上不显著 → 不连线
            drew = True
            val = r["emergence"][j]                          # 线宽仍用效应量 Δ
            ax.add_patch(FancyArrowPatch(
                (x_l, y), (x_r, y_right[dim]), connectionstyle="arc3,rad=0.06", arrowstyle="-",
                color=DIMC[dim], linewidth=0.22 + 3.0 * (val / 0.71),
                alpha=0.13 + 0.52 * (val / 0.71), zorder=2))
        n_none += int(not drew)
        # 节点描边**永远**用该 primitive 的颜色（= panel A 行标签的颜色），不再因为「一个显著
        # 维度都没有」而变灰 —— 灰圈与 panel A 的配色规矩冲突。「它没有边」直接看得到（节点上
        # 没有线），计数写进图注文本，不再占图例一条。
        ax.scatter([x_l], [y], s=12, facecolor="white", edgecolor=dim_c,
                   linewidth=0.9, zorder=5)
        # 左端 primitive 名也按主导维度上色，与 panel A 的行标签**同一套颜色**（同一个 primitive
        # 在 A 与 E 里颜色相同），不再是清一色黑；**加粗**与 panel A 的行标签、右端那两行维度名
        # 一致 —— 这一列原来是全图唯一一处常规字重的人名（右端、条形图刻度名、panel A 行标签
        # 都是加粗），补齐。左端留的那 0.62 个单位是按加粗后的实测宽度核过的（见下）。
        ax.text(x_l - 0.028, y, r["en"], fontsize=FS_TINY, fontweight="bold",
                color=dim_c, ha="right", va="center", zorder=6)

    for dim in DIMS:
        y, n = y_right[dim], sig_count[dim]
        # n = 0 的维度照样画出一个小节点、里面写 0 —— 「这个维度上没有一个 primitive 显著」
        # 本身就是要看的信息，不能悄悄略过。节点面积 ∝ n（但留一个可见的下限）。
        ax.scatter([x_r], [y], s=45 + 11.0 * n, facecolor=DIMC[dim], edgecolor="white",
                   linewidth=1.0, zorder=6)
        ax.text(x_r, y, str(n), fontsize=FS_TINY + 0.2, color="white", fontweight="bold",
                ha="center", va="center", zorder=7)
        ax.text(x_r + 0.035, y, DIM_2L[dim], fontsize=FS_TINY, color=INK, ha="left",
                va="center", fontweight="bold", zorder=6, linespacing=1.25)

    # 左右两条留白带按统一字号（7.2 pt）后的实测宽度重排：左边 0.62 个单位放得下最长的
    # primitive 名（"Info. seeking (uncertainty)" ≈ 1.3 in），右边 0.50 个单位放得下右端
    # 两行维度名（"Social communication" ≈ 1.0 in）。扇面剩 1.0 个单位 ≈ 2.4 in。
    ax.set_xlim(-0.62, 1.50)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")

    # 右旁条形图：每个表型维度上显著的 primitive 个数（= 右端节点的入边数）。
    # 每行两根：实心 = 主判据（合并方差 t 检验，与画边同一版），浅色 = 同一套 q 值收严到
    # TEST_ALPHA_STRICT（存活率那一根）。
    ypos = np.arange(len(DIMS))[::-1]
    counts = [sig_count[dim] for dim in DIMS]
    counts_s = [sig_count_strict[dim] for dim in DIMS]
    ax_bar.barh(ypos + 0.19, counts, height=0.34, color=[DIMC[d] for d in DIMS],
                edgecolor="white", linewidth=0.7, zorder=3)
    ax_bar.barh(ypos - 0.19, counts_s, height=0.34, color=[DIMC[d] for d in DIMS], alpha=0.38,
                edgecolor="white", linewidth=0.7, zorder=3)
    for y, c in zip(ypos + 0.19, counts):
        ax_bar.text(c + 0.45, y, str(c), va="center", ha="left", fontsize=FS_SMALL,
                    fontweight="bold", color=INK)
    for y, c in zip(ypos - 0.19, counts_s):
        ax_bar.text(c + 0.45, y, str(c), va="center", ha="left", fontsize=FS_SMALL,
                    fontweight="bold", color=INK)
    ax_bar.set_yticks(ypos)
    # 用与 panel A 列头 / panel E 右端节点**完全同一套**维度名，不用缩写；
    # 而且**按维度上色**（不是黑色）：这样 panel E 内部就自带一份「维度名 ↔ 颜色」的对照，
    # 读者（与验收者）可以拿它去核对左端 primitive 名的颜色 —— 那些名字的颜色 = 该 primitive
    # 主剂量主导维度的颜色，也就是 panel A 行标签的颜色。
    ax_bar.set_yticklabels([DIM_2L[d] for d in DIMS], fontsize=FS_TICK, fontweight="bold",
                           linespacing=1.2)
    for lab, dim in zip(ax_bar.get_yticklabels(), DIMS):
        lab.set_color(DIMC[dim])
    ax_bar.set_xlim(0, bar_xmax)          # 两档共用同一条横轴，柱长才能跨剂量比
    ax_bar.set_xticks([0, 10, 20, 30])
    ax_bar.set_xlabel("Number of primitives", fontsize=FS_BODY, fontweight="bold", labelpad=1.5)
    # 条形图**不设子图标题**：柱子是什么由横轴标题与
    # 下方那张共用图例里的两条柱形说明承担。
    # 条形图**不画横向网格线**：柱长靠共用的那条横轴与
    # 刻度数字读，下面也没有分块分隔线。
    ax_bar.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.tick_params(axis="y", length=0, pad=3)
    ax_bar.tick_params(axis="x", pad=1.5)
    return n_none                     # 供两张网络共用的那份图例写「无显著维度 n = ?」


def panel_f(fig, axes, data):
    """panel F：5 个 primitive 的剂量–反应曲线（一档一张），5 张共用一份图例、一条横轴标题。

    - 每张子图：横轴 = **1 − θ**（θ = primitive 完整度，所以 1 − θ = 削弱强度，越大越弱），
      纵轴 = 该表型维度的**平均读数**（5 次重复的均值，0–1，**越大越典型**）——
      与 panel A 同一口径，不翻转、不做差分。所以曲线随 1 − θ 上升而下降。
    - **横轴的 0 是完整心智**（θ = 1）：5 条曲线的起点因此是同一条基线剖面，
      不是阶梯实验的某个格子 —— 值由本表 `*_mean − *_delta` 反解（与 result_2 的 baseline 逐位一致），
      误差棒用**参照组那 5 份完整心智的逐次读数**（与 panel E 的检验同一批）。
      这样「从完整到最弱」是一条完整的曲线，而不是从 0.9 档开始的一段。
    - 每张画 **6 条线**（6 个表型维度）。**对照通道 general_cognition 不画**；
      它的数值仍逐格留在 `figure2_metrics.json` 的 `panel_f.per_primitive` 里，需要时能查。
    - 误差棒 = **±1 SEM**（5 次重复，ddof=1），不是 SD：这里要比的是
      「同一档位下剖面的差异」，SD 会把重复之间的离散度混进来。
    - **5 张共用同一纵轴尺度**（0–1，与 panel A 的色带同一口径）：各定各的会让「哪一张掉得深」
      被各自的坐标轴吃掉，跨 primitive 就没法比。刻度数字 5 张各自保留，纵轴标题只写最左一张。
    - 横轴是**连续**量：刻度落在实测档位上（0 / 0.1 / 0.2 / 0.4 / 0.6 / 0.8），**不重排成等距** ——
      0.8 与 0.9 两档挨得近是数据本身如此，重排会把「θ = 0.9 只比 0.8 高一点」画成「高了一整档」。
    - **不画面板标题、也不画面板字母**（与全图一致）。primitive 的身份由**轴内注记**
      承担：每张左上角写它的英文名，按该 primitive **主剂量 0.4 的主导维度**上色 —— 与
      panel A 的行标签、panel B 的列顶标注、panel E 的左端名字是同一套规矩（一个 primitive
      在整张图里只有一种颜色）。名字写在坐标区**里面**（写在外面就变成面板标题了）。
    - **图例在一整行之下、5 张共用一份**（figure 级，6 项一排）：6 条编码在 5 张图里含义相同，
      一张图一份就是 5 份重复。**横轴标题也共享一条**（`fig.text`，居中于整行之下）：
      5 张的横轴刻度完全相同，重复 5 遍没有信息量。
    """
    prim = data["dose_ladder"]["primitive_order"]
    axes = list(axes)
    assert len(axes) == len(prim), f"panel F 给了 {len(axes)} 张子图，但有 {len(prim)} 个 primitive"
    #: 颜色取 result_2 主剂量 0.4 的主导维度（不是 result_4 自己的）：整张图里一个 primitive
    #: 只有一种颜色，panel A/B/E 画的都是这个 5 个 primitive 在 0.4 档的主导维度。
    dom2 = {r["primitive_id"]: r["dominant_dim"] for r in data["rows"]}
    #: 横轴 = 1 − θ，**按 1 − θ 升序排**（θ 降序）：最左是完整心智（0），往右削弱越强。
    tx = [1.0 - d for d in DOSE_LADDER][::-1]          # 0.1 / 0.2 / 0.4 / 0.6 / 0.8
    xs = np.asarray([0.0] + tx, dtype=float)           # 0 = 完整心智
    intact = data["dose_ladder"]["intact"]
    #: 线的粗细：这一段比 A–E 里的任何一条线都粗（要求「增加线条 width」）。横轴只有 6 个点、
    #: 每张也只有 6 条线，1.7 pt 不会糊成一片，反而让「同一档位下哪条线在哪」看得清。
    LW, MS, ELW, CAP = 1.7, 2.7, 0.75, 1.4

    for k, (ax, pid) in enumerate(zip(axes, prim)):
        assert pid in dom2, f"panel F 的 {pid} 不在 result_2 的 31 个 primitive 里，取不到它的颜色"
        cells = data["dose_ladder"]["cells"][pid]
        for dim in DIMS:
            means = [intact[dim]["mean"]] + [cells[d]["channels"][dim]["mean"] for d in DOSE_LADDER[::-1]]
            sems = [intact[dim]["sem"]] + [cells[d]["channels"][dim]["sem"] for d in DOSE_LADDER[::-1]]
            ax.errorbar(xs, np.asarray(means, dtype=float), yerr=np.asarray(sems, dtype=float),
                        color=DIMC[dim], linewidth=LW, marker="o", markersize=MS,
                        markeredgewidth=0.0, capsize=CAP, elinewidth=ELW, zorder=3)
        ax.set_xlim(-0.045, xs[-1] + 0.045)
        ax.set_ylim(0.0, 1.0)                      # 与 panel A 的读数口径同一条 0–1
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_xticks(xs)
        ax.set_xticklabels(["0"] + [f"{v:.1f}" for v in tx])   # 0 = 完整心智
        if k == 0:                                 # 纵轴标题只写最左一张（5 张同一尺度）
            ax.set_ylabel("Phenotype score  (mean of 5; 1 = typical)",
                          fontsize=FS_BODY, fontweight="bold", labelpad=2.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="both", pad=1.5)
        ax.text(0.02, 0.985, EN[pid], transform=ax.transAxes, fontsize=FS_BODY,
                fontweight="bold", color=DIMC[dom2[pid]], ha="left", va="top", zorder=9)

    # 共享的横轴标题：画在整行的横轴之下、图例之上（5 张的刻度一样，写 5 遍是重复）
    fig.text(F_LEGEND_X, F_XLABEL_IN / FIG_H,
             "1 − θ   (0 = intact mind; higher = stronger perturbation)",
             fontsize=FS_BODY, fontweight="bold", color=INK, ha="center", va="center")

    # 共用图例：一行 6 项，居中于整行之下。**字号比正文大一档**（FS_F_LEGEND = 8.6 pt，
    # 放大：这 6 个维度名是 panel F 唯一的编码说明，面板上没有标题、没有别的标注）；其余各项
    # 与 panel B 那份逐项相同（无边框、handlelength 1.2 / handletextpad 0.45 /
    # columnspacing 0.95 / labelspacing 0.4、左对齐 —— 手柄长度是字号单位，会跟着一起放大）。
    handles = [Line2D([], [], color=DIMC[d], linewidth=LW, marker="o", markersize=MS,
                      markeredgewidth=0.0, label=DIM_1LINE[d]) for d in DIMS]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(F_LEGEND_X, F_LEGEND_TOP_IN / FIG_H),
               ncol=len(handles), frameon=False, fontsize=FS_F_LEGEND, handlelength=1.2,
               handletextpad=0.45, columnspacing=0.95, labelspacing=0.4,
               borderaxespad=0.0, alignment="left")
    return len(handles)




# 版式：所有面板位置集中在这里，改版式只动这张表
#
# **两种坐标**：`_LAYOUT_BASE` 里的数字是**加 panel F 之前那版 15.5 in 图幅**上的图分数
# （A–E 全部照旧，一个数没动），下面那三个函数把它换算到新图幅的图分数上 —— 所以
# panel F 只是把画布**加高**，原有面板的英寸几何（行宽、行高、字号相对大小、间距）逐位不变。
# 这比「压缩 A–E 腾地方」安全得多：第二行那两张 31 行的网络是按行高排过版的，动高度就会压字。
FIG_W, FIG_H_BASE, FIG_H = 17.0, 15.5, 18.2
#: 底部新增的一条带子（in）：panel F 的 5 张子图 + 上面那行共用图例都在这里。
BAND_IN = FIG_H - FIG_H_BASE


def _y(y: float) -> float:
    """旧图幅的图分数 y → 新图幅的图分数 y（英寸位置原样保留，只是离底边多了 BAND_IN）。"""
    return (y * FIG_H_BASE + BAND_IN) / FIG_H


def _h(h: float) -> float:
    """旧图幅的高度（图分数）→ 新图幅的高度（图分数），英寸不变。"""
    return h * FIG_H_BASE / FIG_H


def _box(x: float, y_in: float, w: float, h_in: float) -> list[float]:
    """按「左 / 下 / 宽 / 高」给 panel F 排位置，单位一律 **in**（自下往上量）。"""
    return [x, y_in / FIG_H, w, h_in / FIG_H]


LAYOUT = {
    # A 的两张热图各收窄 0.004 图宽（每列热图格 0.35 in，仍然够），把这条窄缝让给右边的色标，
    # 色标再让给 B —— B 的六列因此各宽 1.69 in，放得下 7.2 pt 下最长的列顶标注（1.61 in）；
    # 不这么挤一遍，B 那三行标注会和邻列的三行贴成一片（那是字号统一后唯一的硬约束）。
    "A4":   [0.095, 0.655, 0.122, 0.262],
    "A8":   [0.229, 0.655, 0.122, 0.262],
    "A_cb": [0.355, 0.655, 0.0090, 0.262],
    # B 改成曼哈顿图后**上下排**（而不是左右排）：6 列摊到 10.1 in 宽上，每列才放得下
    # 三行「primitive 名 + Δ」的标注而不与邻列连成一片。两张共用同一套维度列，
    # 维度名只在下面那张竖排一次 —— 所以下面那张的底边要抬到第二行面板标题之上。
    "B4":   [0.404, 0.760, 0.591, 0.152],
    "B8":   [0.404, 0.588, 0.591, 0.152],
    # 第二行：两张网络并排 + 各自右侧一张条形图（原 panel C/D 已删）。
    # 网络 5.1 in、条形图绘图区 1.2 in，宽度是这样定的：条形图的 y 刻度名（六个维度名，
    # 与网络右端节点旁那一份同名）必须落在「网络坐标区右边界」与「条形图绘图区」之间的
    # 空档里 —— 空档按 FS_TICK 下最长那行维度名（"Cognitive flexibility"）的实测宽度留，
    # 网络坐标区里右端节点名又要放得下，两个约束一起把 E 的宽度压到 0.300。
    # 纵向：底部不设脚注框，让出来的高度给了第二行，
    # 于是 5.5 in（原来是 4.25 in）—— 31 行左端标签的间距、扇面、两根柱子都更松。
    # **四张图上下边界完全一致**（网络与条形图同一个 y0、同一个高度）：对齐，
    # 于是网络的行格与条形图的六个维度行在同一段竖直范围里，左右对读不用再对高低。
    # 纵向：网络的墨迹铺满坐标区（见 panel_e 里的留白说明），条形图的坐标区则**抬高一个
    # 「刻度名 + 横轴标题」的高度**（实测 0.026 图高）——这样两件东西的**可见高度**相同：
    # 网络的墨迹 = 0.085–0.464，条形图的墨迹（含下方那两行字）= 0.085–0.464。
    # 若按坐标框对齐（都从 0.085 起），条形图因为下面挂着两行字，看上去会比网络高一截。
    "E4":     [0.070, 0.056, 0.300, 0.416],
    "E4_bar": [0.444, 0.081, 0.070, 0.383],
    "E8":     [0.548, 0.056, 0.300, 0.416],
    "E8_bar": [0.920, 0.081, 0.070, 0.383],
}
#: A–E 的图分数是「旧图幅」上的，逐个换算到新图幅（英寸几何不变）
LAYOUT = {k: [v[0], _y(v[1]), v[2], _h(v[3])] for k, v in LAYOUT.items()}

# ---------------------------------------------------------------- 第三行：panel F
#: 5 张子图横向铺满整行，左右边界与第二行一致（左 = E 网络的 0.070，右 = 条形图的 0.990）。
F_ROW_X0, F_ROW_X1 = 0.070, 0.990
F_GAP_IN = 0.37                                     # 两张子图之间的空档（in）
F_W_IN = ((F_ROW_X1 - F_ROW_X0) * FIG_W - 4 * F_GAP_IN) / 5.0
F_PITCH = (F_W_IN + F_GAP_IN) / FIG_W
F_Y_IN, F_H_IN = 0.72, 2.78                         # 子图下缘 / 高度（in，自底量）
for _i in range(5):
    LAYOUT[f"F{_i + 1}"] = _box(F_ROW_X0 + _i * F_PITCH, F_Y_IN, F_W_IN / FIG_W, F_H_IN)
#: 共享横轴标题与共用图例都在**整行之下**（要求「图例放到下方」）：
#:   子图底 0.72 in → 刻度名（最低 0.564 in） → 横轴标题（中心 0.472 in） → 图例（顶边 0.38 in、
#:   高 0.17 in） → 图底留白 0.21 in。三样东西之间的净空各 0.045 in（实测，不是估的）。
#: 横轴标题居中于整行（5 张刻度相同，写 5 遍是重复）；图例**一行 6 项**，居中于整行。
F_XLABEL_IN = 0.472
F_LEGEND_TOP_IN = 0.38
F_LEGEND_X = (F_ROW_X0 + F_ROW_X1) / 2.0
#: 面板标题（字母 + 标题同一行）。本图**没有**面板副标题 —— 说明性灰字已全部去掉，
#: 编码的含义改为写进坐标轴标题与图例（见各 panel 函数）。
    #: 本图不画 panel C/D，字母**保持不动**：网络仍是 panel E（一档一张，和 panel A/B 一样一个字母两张图），
#: 所以字母序列是 A B E，中间的空号是有意的，没有重排成 A B C。
#: **不画面板标题与剂量标签**（字母 A/B/E、标题文字、"primitive integrity 0.4/0.8"
#: 都不再画）。图里只剩坐标轴标题、刻度与图例承担编码含义；面板身份与剂量归属写进论文图注
#: （正文 §「图里不写」一节有交代）。所以这里不再有 TITLE_Y / DOSE_LABEL_Y 这套表。
#:
#: 剂量归属**只**在以下三处可查，图上无从辨认：本文档、`figure2_metrics.json` 的
#: `layout_note`、以及脚本里 E4/E8 的布局表（左 = 0.4，右 = 0.8，与 panel A/B 的左右一致）。


def draw(fig, data):
    ax_a4 = fig.add_axes(LAYOUT["A4"])
    ax_a8 = fig.add_axes(LAYOUT["A8"])
    cb_a = fig.add_axes(LAYOUT["A_cb"])
    panel_a(fig, ax_a4, ax_a8, cb_a, data)

    ax_b4 = fig.add_axes(LAYOUT["B4"])
    ax_b8 = fig.add_axes(LAYOUT["B8"])
    panel_b(fig, ax_b4, ax_b8, data)

    # 第二行：两张网络 + 各自右侧的条形图，各对应一个剂量（原 panel C/D 已删）
    ax_e4 = fig.add_axes(LAYOUT["E4"])
    ax_e4_bar = fig.add_axes(LAYOUT["E4_bar"])
    ax_e8 = fig.add_axes(LAYOUT["E8"])
    ax_e8_bar = fig.add_axes(LAYOUT["E8_bar"])
    data["e_stats"] = {}
    # 两张条形图共用一条横轴（取两档、两根柱子里最长的柱子定上限）—— 各定各的上限会让
    # 同样高的柱子在不同面板之间长得不一样长，比较就成了假比较。
    bar_xmax = max(
        max(data[c][float(dose)][dim] for c in ("sig_count", "sig_count_strict"))
        for dose in (PRIMARY_DOSE, SECONDARY_DOSE)
        for dim in DIMS
    ) * 1.24 + 1.0
    for ax_e, ax_bar, dose in ((ax_e4, ax_e4_bar, PRIMARY_DOSE),
                               (ax_e8, ax_e8_bar, SECONDARY_DOSE)):
        rows = data["rows"] if dose == PRIMARY_DOSE else data["rows_by_dose"][dose]
        data["e_stats"][dose] = {
            "significant_per_dimension": data["sig_count"][float(dose)],
            "significant_per_dimension_strict": data["sig_count_strict"][float(dose)],
            "significant_per_dimension_welch": data["sig_count_welch"][float(dose)],
            "primitives_without_any_significant_dimension": data["n_no_sig"][float(dose)],
            "primitives_without_any_significant_dimension_strict":
                data["n_no_sig_strict"][float(dose)],
            "dominant_count_per_dimension": data["dom_count"] if dose == PRIMARY_DOSE else {
                dim: sum(1 for r in rows if r["dominant_dim"] == dim) for dim in DIMS
            },
        }
        data["e_stats"][dose]["n_no_edge"] = panel_e(
            fig, ax_e, ax_bar, rows, dose, data["sig_count"][float(dose)],
            data["sig_count_strict"][float(dose)], bar_xmax, data["e_order"], data["block_dim"])

    # panel E 不设图例（边判据、线宽 = Δ 与
    # 两根柱子各是哪一版检验）—— 这几条编码含义现在只在图注文本里
    # （`figure2_metrics.json` → `caption_text_not_shown`），图上不再有图例。
    # 注意：因此本图**唯一**的编码说明在坐标轴标题（Δ 的定义、柱长是什么）与图注文本里；
    # 两根柱子的区别（实心 = q < 0.05，浅色 = 同一套 q 值收到 0.01）也只能从图注文本读到。

    # 第三行：panel F —— 5 个 primitive 的剂量–反应曲线（一档一张），共用一份图例。
    # 数据源是另一张表（result_4 的剂量阶梯，5 档 × 5 次重复，见 load_dose()），
    # 与 A–E 的 result_2 不是同一批运行，图上不再另作区分（都在图注里写明）。
    ax_f = [fig.add_axes(LAYOUT[f"F{i + 1}"]) for i in range(len(LADDER_PRIMITIVES))]
    data["f_legend_entries"] = panel_f(fig, ax_f, data)

    # 剂量标签改写成 θ：θ = primitive 完整度（保留原语组件的比例），只标剂量、不标面板字母与
    # 面板标题。六处：A 的两张热图各一处、B 的上下两张各一处、
    # E 的两组「网络 + 条形图」各一处（居中于整组，不只居中于网络）。
    # 这里的高度值都是**基础图幅**上的分数：本函数末尾统一交给 _y() 换算到实际图幅。
    dose_labels = [(ax_a4, PRIMARY_DOSE, _y(0.9225)), (ax_a8, SECONDARY_DOSE, _y(0.9225)),
                   (ax_b4, PRIMARY_DOSE, LAYOUT["B4"][1] + LAYOUT["B4"][3] + _h(0.011)),
                   (ax_b8, SECONDARY_DOSE, LAYOUT["B8"][1] + LAYOUT["B8"][3] + _h(0.011))]
    for ax, dose, y in dose_labels:
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, y, f"θ = {dose:.2f}",
                 fontsize=FS_DOSE, fontweight="bold", color=INK, ha="center", va="center")
    for ax_e, ax_bar, dose in ((ax_e4, ax_e4_bar, PRIMARY_DOSE), (ax_e8, ax_e8_bar, SECONDARY_DOSE)):
        net, bar = ax_e.get_position(), ax_bar.get_position()
        fig.text((net.x0 + bar.x1) / 2, net.y1 + _h(0.017), f"θ = {dose:.2f}",
                 fontsize=FS_DOSE, fontweight="bold", color=INK, ha="center", va="center")

    fig.text(0.028, _y(0.978),
             "Figure 2 |  Univariate cognitive-primitive perturbations reveal structured and "
             "distributed emergence of ASD phenotypes",
             fontsize=FS_FIG, fontweight="bold", color=INK, va="center", ha="left")
    # 面板标题与面板字母不画：面板身份写进论文图注；剂量用 θ 标在每张/每组之上。


def caption_text(data) -> str:
    """图的文字说明（**不出现在图上**）。

    这段文字现在只落进 `figure2_metrics.json`（`caption_text_not_shown`），供论文图注改写时
    直接取用；图面上只剩坐标轴标题、刻度与 panel E 的共用图例。数字全部由图里的数据现算。
    """
    vc = data["verdict_count"]
    get = lambda k: vc.get(k, 0)  # noqa: E731
    hist = data["hist"]
    n_tot = sum(hist.values())
    n_ctrl = sum(1 for r in data["rows"] if not r["control_ok"])
    d_effs = sorted(r["d_eff"] for r in data["rows"])
    concs = sorted(r["conc_share"] for r in data["rows"])
    med_deff = d_effs[len(d_effs) // 2]
    med_c = concs[len(concs) // 2]
    sig4 = sum(data["sig_count"][PRIMARY_DOSE].values())
    sig8 = sum(data["sig_count"][SECONDARY_DOSE].values())
    st4 = sum(data["sig_count_strict"][PRIMARY_DOSE].values())
    st8 = sum(data["sig_count_strict"][SECONDARY_DOSE].values())
    lad = data["dose_ladder"]
    #: panel F 的话要数字支撑：逐 primitive 的六维平均下降幅度（完整心智 → 最弱那一档）；
    #: 以及「最低的一条线是不是该 primitive 自己的维度」（用主剂量 0.4 的主导维度比）。
    drops = [sum(lad["cells"][pid][DOSE_LADDER[0]]["channels"][d]["mean"]
                 - lad["intact"][d]["mean"] for d in DIMS) / len(DIMS)
             for pid in lad["primitive_order"]]
    n_low_is_dom = sum(
        1 for pid in lad["primitive_order"]
        if min(DIMS, key=lambda d: lad["cells"][pid][DOSE_LADDER[0]]["channels"][d]["mean"])
        == lad["cells"][pid][PRIMARY_DOSE]["dominant_dim"])
    text = (
        "A single cognitive-primitive perturbation does not buy a selective phenotype.  "
        "Throughout, θ is the primitive's integrity (the share of its components left intact) and "
        "Δ = baseline − perturbed measures how far the phenotype profile moved towards the ASD-like "
        "end; panel E's networks draw an edge where the two-sample t-test is significant and give its "
        "width to Δ, while the paired bar chart reports, per phenotype dimension, how many primitives "
        "are significant under the pooled-variance test (solid bar, q < 0.05) and how many of those "
        "survive a stricter threshold on the same test (faded bar, q < 0.01).  "
        f"Across the six phenotype dimensions the median primitive has an effective dimensionality "
        f"D_eff = {med_deff:.2f} of {len(DIMS)} (1 = only one dimension is hit, {len(DIMS)} = all six equally) "
        f"and a phenotypic concentration C = {med_c:.2f} — its worst dimension carries only about "
        f"{100 * med_c:.0f}% of the total impairment, where a fully selective perturbation would carry 100%.  "
        f"Under the experiment's own criteria only {get('pass')} keep the general-cognition control inside ±0.15 while passing the concentration test "
        f"({get('fail_diffuse')} fail as diffuse, {get('no_effect')} show no effect, {get('fail_control') + get('fail_both')} break the control channel).  "
        f"A per-dimension t-test against the intact reference (each primitive's 3 perturbed repeats vs 5 intact repeats, "
        f"BH-FDR q < {TEST_ALPHA:g}) calls {sig4} of {data['n_tests']} primitive × dimension cells significant at "
        f"integrity 0.4 and {sig8} at 0.8, and at 0.4 only {data['n_no_sig'][PRIMARY_DOSE]} of {n_tot} primitives escape without a single "
        f"significant dimension; tightening the same test to q < {TEST_ALPHA_STRICT:g} leaves {st4} and {st8} of those cells (faded bar), "
        f"so roughly two thirds of the counts survive the stricter threshold.  These counts are the least assumption-free "
        f"part of the figure: the perturbed repeats are the noisier ones (median variance ratio perturbed/reference ≈ 5) and the "
        f"equal-variance test is anti-conservative in this 3-repeats-vs-5 design (realised type-I error ≈ 11% at a nominal 5%), so the "
        f"reading above rests on the effect sizes (Δ, D_eff, C) rather than on the significance counts alone.  "
        f"Effects are dose-dependent for {data['n_stronger_at_0.4']} of {n_tot} primitives, which worsen as integrity falls from 0.8 to 0.4, "
        f"and at the stronger dose the control channel is itself depressed for {n_ctrl} of {n_tot} primitives — so the map reads as a distributed, "
        "partly non-specific shift of the whole phenotype profile rather than as a set of clean one-primitive → one-dimension links.  "
        f"Panel F follows five of those primitives down the whole dose ladder (θ = "
        f"{', '.join(f'{d:g}' for d in DOSE_LADDER)}; {lad['n_replicates_per_cell']} repeats per cell, "
        "a separate run from panels A–E).  Each subplot plots the MEAN phenotype score of the six "
        "phenotype dimensions against 1 − θ (0 = the intact mind, 0.8 = θ 0.2, the weakest cell; "
        "0–1, higher = more typical; error bars = ±1 SEM), all five subplots on one shared 0–1 scale, "
        "with one shared x title and one shared legend below the row.  The leftmost point of every "
        "curve is the intact profile itself, taken from the same five intact minds the rest of the "
        f"figure uses, so the curves start from one common baseline.  Taking a primitive apart costs "
        f"the profile {min(-d for d in drops):.2f}–{max(-d for d in drops):.2f} on average over the six dimensions by the "
        f"weakest dose, and at that dose the lowest curve is that primitive's own dimension in "
        f"{n_low_is_dom} of {len(lad['primitive_order'])} subplots — the damage is aimed at the right "
        "dimension.  It is still not selective: the other five dimensions fall with it, by different "
        "amounts in every subplot, and the six curves cross rather than separating, which is the same "
        "distributed profile the heat map and the networks show, read along the dose axis instead of "
        "across primitives."
    )
    return textwrap.fill(text, 232)


def _panel_f_metrics(data) -> dict:
    """panel F 落盘的每一个数：逐 primitive × 逐档（含完整心智那一点）的均值 / SEM / 最低通道。

    图上的 5 × 6 条曲线的 30 个点 + 5 × 6 个完整心智点、以及对应的误差棒都能在这里查到；
    颜色用的主导维度取 **result_2 主剂量 0.4** 那一份（与 A/B/E 同一套规矩），
    与 result_4 自己算出的 `dominant_dim` 可能不同（例如 agent_state_updating 在
    result_4 的几个档位上主导维度是 social_cognition）—— 这就是两处并存的口径，
    图上用的是前者的颜色、曲线的最低点由数据自己决定。
    """
    lad = data["dose_ladder"]
    dom2 = {r["primitive_id"]: r["dominant_dim"] for r in data["rows"]}
    per = {}
    for pid in lad["primitive_order"]:
        cells = lad["cells"][pid]
        by_dose, lowest = {}, {}
        for dose in DOSE_LADDER:
            ch = cells[dose]["channels"]
            by_dose[str(dose)] = {
                dim: {"mean": round(ch[dim]["mean"], 4), "sem": round(ch[dim]["sem"], 4),
                      "n_replicates": ch[dim]["n"]}
                for dim in DIMS + [CONTROL]
            }
            lowest[str(dose)] = min(DIMS, key=lambda d: ch[d]["mean"])
        rise = [cells[DOSE_LADDER[-1]]["channels"][d]["mean"]
                - cells[DOSE_LADDER[0]]["channels"][d]["mean"] for d in DIMS]
        ctrl = [cells[d]["channels"][CONTROL]["mean"] for d in DOSE_LADDER]
        per[pid] = {
            "primitive": EN[pid],
            "colour_dim_at_primary_dose_result_2": dom2.get(pid),
            "dominant_dim_at_primary_dose_result_4": cells[PRIMARY_DOSE]["dominant_dim"],
            "mean_score_by_dose": by_dose,
            "lowest_channel_by_dose": lowest,
            "mean_rise_over_six_dimensions_0.2_to_0.9": round(sum(rise) / len(DIMS), 4),
            "control_score_by_dose": [round(v, 4) for v in ctrl],
            "control_min": round(min(ctrl), 4),
            "control_range": round(max(ctrl) - min(ctrl), 4),
        }
    return {
        "source": str(XLSX4.relative_to(ROOT)).replace("\\", "/"),
        "sheets": ["summary_by_cell", "by_subject"],
        "primitives": lad["primitive_order"],
        "doses": DOSE_LADDER,
        "n_replicates_per_cell": lad["n_replicates_per_cell"],
        "x": "1 − θ (θ = the primitive's integrity), i.e. perturbation strength: the leftmost point "
             f"is 0 (intact mind) and the ladder's measurements sit at "
             f"{[round(1 - d, 2) for d in DOSE_LADDER[::-1]]}; ticks are placed at the measured "
             "values, NOT re-spaced evenly (0.8/0.9 are close together because the ladder put them "
             "there), so the horizontal gaps carry the true distances",
        "y": "mean phenotype score of the cell (0-1, higher = more typical), i.e. summary_by_cell's "
             "{dim}_mean; NOT sign-flipped into Δ — so a curve falling to the right means the "
             "profile degrades as the primitive is taken apart",
        "error_bars": "±1 SEM (ddof=1). Ladder cells: the 5 replicates in by_subject. Intact point: "
                      "the 5 intact minds of the reference group (the same 5 used by panel E's test)",
        "intact_point": {
            "x": 0.0,
            "meaning": "the intact mind, not a ladder cell: θ = 1 is not in the run grid",
            "values": {dim: {"mean": round(lad["intact"][dim]["mean"], 4),
                             "sem": round(lad["intact"][dim]["sem"], 4),
                             "n_replicates": lad["intact"][dim]["n"]} for dim in DIMS + [CONTROL]},
            "cross_check": "the profile is first re-derived inside result_4 itself as "
                           "{dim}_mean − {dim}_delta (constant per dimension) and then asserted "
                           "equal to result_2's baseline constants to 1e-9, so both tables are "
                           "shown to sit on the same 5 intact minds",
        },
        "shared_y_scale": [0.0, 1.0],
        "channels_per_subplot": DIMS,
        "control_channel": "general_cognition is NOT drawn (removed at the user's request); its "
                           "per-cell values are still kept in per_primitive.mean_score_by_dose and "
                           "control_score_by_dose for reference",
        "line_width_pt": 1.7,
        "line_width_note": "thicker than anything else in the figure (other panels' lines: panel B's "
                           "median rule 1.5 pt, panel E's edges 0.22-3.0 pt scaled by Δ)",
        "legend": f"ONE figure-level legend BELOW the five subplots, {len(DIMS)} entries in a single "
                  "row; identical treats to the panel-B legend (borderless, handlelength 1.2, "
                  "handletextpad 0.45, columnspacing 0.95, labelspacing 0.4) EXCEPT the font size, "
                  f"which is raised to {FS_F_LEGEND} pt (the figure's body size is {FS_BODY} pt) "
                  "because these six dimension names are panel F's only encoding key; at that size "
                  "the row measures 11.7 in, still inside the 17 in canvas",
        "x_title": "ONE shared x title for the five subplots, centred on the row below the tick "
                   "labels and above the legend: '1 − θ   (0 = intact mind; higher = stronger "
                   "perturbation)'; the y title is written on the leftmost subplot only, both in "
                   "the figure's axis-title style (7.2 pt bold, like panel B's and panel E's)",
        "panel_letters_titles": "none: the subplots carry no title and no panel letter, exactly "
                                "like the rest of the figure; a primitive is identified by the "
                                "in-panel annotation (its name, set inside the axes at the top "
                                "left and coloured with that primitive's panel-A colour)",
        "mean_cross_check": f"{lad['n_means_checked']}/{lad['n_means_checked']} cell × channel means "
                            "recomputed from by_subject match summary_by_cell within 1e-4",
        "per_primitive": per,
    }


def metrics(data) -> dict:
    def clean(rows):
        return [
            {
                "primitive_id": r["primitive_id"],
                "primitive": r["en"],
                "primitive_zh": r["zh"],
                "reading": {d: round(v, 4) for d, v in zip(DIMS, r["reading"])},
                "emergence": {d: round(v, 4) for d, v in zip(DIMS, r["emergence"])},
                "control_emergence": round(r["control"], 4),
                "dominant_dim": r["dominant_dim"],
                "dominant_effect": round(r["dominant_effect"], 4),
                "second_dim": r["second_dim"],
                "second_effect": round(r["second_effect"], 4),
                "selectivity_gap": round(r["concentration"], 4),
                "concentration_C": round(r["conc_share"], 4),
                "d_eff": round(r["d_eff"], 4),
                "selectivity_margin": round(r["margin"], 4),
                "n_affected": r["n_affected"],
                "max_effect": round(r["max_effect"], 4),
                "t_q_by_dimension": {d: round(v, 6) for d, v in zip(DIMS, r["q_t"])},
                "welch_q_by_dimension": {d: round(v, 6) for d, v in zip(DIMS, r["q_welch"])},
                "n_significant_dimensions": r["n_sig"],
                "n_significant_dimensions_welch": r["n_sig_welch"],
                "n_significant_dimensions_without_fdr": r["n_sig_uncorrected"],
                "verdict": r["verdict"],
            }
            for r in heat_order(rows)
        ]

    return {
        "source": str(XLSX.relative_to(ROOT)).replace("\\", "/"),
        "criteria": {
            "target_drop": TARGET_DROP,
            "control_tolerance": CONTROL_TOLERANCE,
            "concentration_margin": CONCENTRATION_MARGIN,
            "source_of_criteria": "pacs/src/pacs/analysis/specificity.py",
            "self_check": f"{data['n_given_verdicts_checked']}/{data['n_given_verdicts_checked']} "
                          "verdicts recomputed from the xlsx match the stored specificity_verdict column",
        },
        "primary_dose": PRIMARY_DOSE,
        "secondary_dose": SECONDARY_DOSE,
        "baseline": {k: round(v, 4) for k, v in data["baseline"].items()},
        "panel_a": {
            "value": "mean phenotype reading per cell (0-1, higher = more typical); not sign-flipped",
            "colour_map": "YlGnBu_r truncated to 0.03-0.82, identical to Figure 1 panel B",
            "row_order": "hierarchical clustering (average linkage, Euclidean) on the 0.4 level",
            "cluster_order": data["cluster_order"],
            "doses_shown": [PRIMARY_DOSE, SECONDARY_DOSE],
            "note": "the general-cognition control column is not shown in panel A",
        },
        "panel_f": _panel_f_metrics(data),
        "n_affected_histogram_not_drawn": {
            "note": "experiment-side classification at Δ ≥ 0.15: how many of the six phenotype "
                    "dimensions each primitive pushed past the threshold. Not drawn anywhere any "
                    "more (the panel that histogrammed it was removed), kept here for reference",
            "counts": {str(k): v for k, v in data["hist"].items()},
        },
        "dominant_dim_count": data["dom_count"],
        "verdict_count_dose_0.4": data["verdict_count"],
        "n_primitives_worse_at_0.4_than_0.8": data["n_stronger_at_0.4"],
        "panel_b": {
            "chart": "Manhattan-style plot: one column per phenotype dimension, one dot per primitive",
            "x": "6 phenotype dimensions (one column each); the two dose panels share the same columns",
            "y": "impairment Δ = baseline − perturbed (positive = more ASD-like)",
            "shared_y_scale_across_doses": True,
            "dots_per_column": "31 primitives; horizontal position inside a column carries no value "
                               "(random jitter), except that the 3 annotated primitives sit at fixed "
                               "left/centre/right lanes so their rings stay apart and their rank "
                               "numerals stay unambiguous",
            "labelled": "the 3 primitives with the largest Δ in each dimension, printed above that "
                        "column and marked on the plot with a ring + rank numeral",
            "label_colour": "each printed primitive name is coloured by that primitive's dominant "
                            "dimension at the PRIMARY dose (0.4), the same colour its row label has "
                            "in panel A, so a primitive keeps one colour across both dose panels",
            "per_dose": {
                str(dose): {
                    "top3_per_dimension": {
                        dim: [
                            {"rank": rank, "primitive": name, "delta": delta}
                            for rank, name, delta, _, _ in data["b_top3"][(dose, dim)]
                        ]
                        for dim in DIMS
                    },
                    "per_dimension": {
                        dim: {
                            "n": len(rows),
                            "mean": round(float(np.mean([r["emergence"][j] for r in rows])), 4),
                            "median": round(float(np.median([r["emergence"][j] for r in rows])), 4),
                            "q1": round(float(np.percentile([r["emergence"][j] for r in rows], 25)), 4),
                            "q3": round(float(np.percentile([r["emergence"][j] for r in rows], 75)), 4),
                            "min": round(min(r["emergence"][j] for r in rows), 4),
                            "max": round(max(r["emergence"][j] for r in rows), 4),
                        }
                        for j, dim in enumerate(DIMS)
                    },
                }
                for dose, rows in ((PRIMARY_DOSE, data["rows"]),
                                   (SECONDARY_DOSE, data["rows_by_dose"][SECONDARY_DOSE]))
            },
        },
        "panel_e": {
            "chart": "bipartite primitive-phenotype network, one panel per dose; "
                     "left = 31 primitives grouped by dominant dimension, right = 6 phenotype "
                     "dimensions; the companion bar chart to the right of each network gives, per "
                     "dimension, the number of primitives with a significant effect — one solid bar "
                     "for the criterion the edges are drawn from (pooled test, q < alpha), one faded "
                     "bar for the SAME test and p-values at the stricter threshold q < "
                     "alpha_strict (a survival count, always a subset of the solid bar)",
            "edge_criterion": {
                "test": "two-sample t-test (scipy.stats.ttest_ind) per primitive x phenotype dimension, "
                        "2 groups: that primitive's 3 perturbed repeats vs the reference group's "
                        "repeats (n = 5); equal_var=True (pooled variance) for the edges and the node "
                        "numbers, which is exactly the 2-group one-way ANOVA (F = t², df = 6) — "
                        "asserted cell by cell against scipy.stats.f_oneway in load()",
                "strict_threshold_variant": "the faded bar uses the same pooled p-values and the same "
                                            "BH-FDR q-values, only the significance level is "
                                            "tightened from alpha = 0.05 to alpha_strict = 0.01, so it "
                                            "reads as 'how much survives a stricter threshold'. It is "
                                            "a NOMINAL level, not a guarantee: this test's realised "
                                            "type-I error rate in this 3-vs-5 design is about 11% at a "
                                            "nominal 5% and about 3.6% at a nominal 1% (null "
                                            "simulations; per-cell variance ratios span 1.2x-17x, so "
                                            "no single level calibrates the whole family)",
                "welch_variant_not_drawn": "equal_var=False is still computed cell by cell and kept in "
                                           "the metrics below for cross-checking the criterion, but it "
                                           "is no longer drawn",
                "reference_group": f"{REF_XLSX.name}, rows with row_id starting "
                                   f"'{REF_ROW_PREFIX}' and mind_setting '{REF_SETTING}'",
                "reference_self_check": "the reference group's per-dimension means reproduce this "
                                        "figure's baseline constants to 1e-16 (asserted in load()), "
                                        "so the reference is the experiment's own baseline, not a "
                                        "second opinion imported by this figure",
                "multiple_comparison": "Benjamini-Hochberg FDR within each (dose, phenotype "
                                       "dimension) family of 31 tests — one family per bar; the same "
                                       "q-values are then compared against both thresholds",
                "alpha": TEST_ALPHA,
                "alpha_strict": TEST_ALPHA_STRICT,
                "direction": "an edge also requires Δ = baseline − perturbed > 0; no cell at "
                             "either dose is significant in the opposite direction",
                "edge_width": "still Δ (effect size) — width is not a p-value",
                "n_tests_per_dose": data["n_tests"],
            },
            "dose_0.4": {
                "significant_primitives_per_dimension_pooled": data["sig_count"][PRIMARY_DOSE],
                "significant_primitives_per_dimension_strict": data["sig_count_strict"][PRIMARY_DOSE],
                "significant_primitives_per_dimension_welch": data["sig_count_welch"][PRIMARY_DOSE],
                "primitives_with_no_significant_dimension_pooled": data["n_no_sig"][PRIMARY_DOSE],
                "primitives_with_no_significant_dimension_strict": data["n_no_sig_strict"][PRIMARY_DOSE],
                "primitives_with_no_significant_dimension_welch": data["n_no_sig_welch"][PRIMARY_DOSE],
                "pooled_counts_without_fdr": data["sig_count_uncorrected"][PRIMARY_DOSE],
                "primitives_per_dominant_dimension": data["e_stats"][PRIMARY_DOSE][
                    "dominant_count_per_dimension"],
            },
            "dose_0.8": {
                "significant_primitives_per_dimension_pooled": data["sig_count"][SECONDARY_DOSE],
                "significant_primitives_per_dimension_strict": data["sig_count_strict"][SECONDARY_DOSE],
                "significant_primitives_per_dimension_welch": data["sig_count_welch"][SECONDARY_DOSE],
                "primitives_with_no_significant_dimension_pooled": data["n_no_sig"][SECONDARY_DOSE],
                "primitives_with_no_significant_dimension_strict": data["n_no_sig_strict"][SECONDARY_DOSE],
                "primitives_with_no_significant_dimension_welch": data["n_no_sig_welch"][SECONDARY_DOSE],
                "pooled_counts_without_fdr": data["sig_count_uncorrected"][SECONDARY_DOSE],
                "primitives_per_dominant_dimension": data["e_stats"][SECONDARY_DOSE][
                    "dominant_count_per_dimension"],
            },
            "note": "both networks share ONE row order and one block structure (primitive order "
                    "and left-hand grouping both taken from the primary dose 0.4, same rule as the "
                    "panel-B colours), so a primitive sits on the same row in both doses and the "
                    "right-hand nodes line up; the number printed inside a node and the solid bar "
                    "are the SIGNIFICANT count (in-degree) under the pooled test, while the block "
                    "size is the primary-dose dominant count — two different quantities, both "
                    "stated in the caption; the faded bar is the same in-degree at the stricter "
                    "threshold",
        },
        "continuous_summary_not_drawn": {
            "note": "the D_eff / concentration figures quoted in the caption text below; panels C and "
                    "D that used to draw them were removed at the user's request, the numbers remain "
                    "valid",
            "median_d_eff_0.4": round(float(np.median([r["d_eff"] for r in data["rows"]])), 4),
            "range_d_eff_0.4": [round(min(r["d_eff"] for r in data["rows"]), 4),
                                round(max(r["d_eff"] for r in data["rows"]), 4)],
            "median_concentration_C_0.4": round(
                float(np.median([r["conc_share"] for r in data["rows"]])), 4),
        },
        "layout_note": {
            "panel_letters_titles": "NOT drawn in the image: the user asked for all panel titles and "
                                    "panel letters to be removed (this includes panel F: no 'F' and "
                                    "no subplot titles; a primitive is named by the in-panel "
                                    "annotation instead)",
            "dose_labels": "the dose IS labelled in the figure, as θ = 0.40 / θ = 0.80 (θ = the "
                           "primitive's integrity): once above each dose's panel in row 1 and once "
                           "above each dose's network + bar-chart group in row 2; the left/right "
                           "order is the same everywhere (left = 0.40); panel F labels the x axis "
                           "as 1 − θ instead (0 = intact mind), with ONE shared title for its five "
                           "subplots",
            "legend": "panel E has NO legend (removed at the user's request): the edge criterion, "
                      "edge width = Δ and the meaning of the solid vs faded bars are now stated in "
                      "the caption text (caption_text_not_shown) and in the axis titles only; "
                      "panel F has exactly ONE legend for its five subplots, placed BELOW the row "
                      "(same style as the panel-B legend and as Figure 1: borderless, 7.2 pt, one "
                      "row",
            "rows": "row 1 = heat map (2 doses) + Manhattan (2 doses); row 2 = network + its bar "
                    "chart, one pair per dose; row 3 = panel F, five dose-response subplots "
                    "(one per laddered primitive) with a shared x title and a shared legend below "
                    "them",
            "canvas": "17.0 x 18.2 in (was 17.0 x 15.5 in before panel F). The canvas was made "
                      "TALLER rather than the upper panels shorter: every A-E box keeps the exact "
                      "inch geometry it had on the 15.5 in canvas (see _y()/_h()), so nothing that "
                      "was already laid out had to be re-tuned. The extra 2.7 in at the bottom hold "
                      "panel F (2.78 in of axes, bottom edge 0.72 in above the figure bottom), its "
                      "shared x title and its legend below the row; the space between row 2's lowest "
                      "ink and the top of panel F's axes is 0.18 in",
            "footer_text_box": "removed at the user's request; its wording is kept in "
                               "caption_text_not_shown below",
            "typography": "all body text (axis titles, tick labels, in-plot annotations, primitive "
                          "and dimension names) is one size, 7.2 pt; the θ dose labels are "
                          "10 pt bold; the 5.5/6.2/6.6 "
                          "steps of the earlier ladder are gone; the figure title (17 pt) is the "
                          "only larger text. TWO exceptions, both deliberate: panel F's shared "
                          "legend is 8.6 pt (FS_F_LEGEND) because those six dimension names are "
                          "panel F's only encoding key, and every primitive/dimension NAME in the "
                          "figure is bold — including the 31 left-hand names in panel E's networks, "
                          "which used to be the only names left on regular weight",
            "grid_lines": "panel E draws none: no grid in the bar charts and no block separator "
                          "rules in the networks; panel B no longer draws the Δ = 0.15 "
                          "threshold line either (it remains a data-layer criterion only)",
            "panel_e_geometry": "the network and its bar chart have the SAME VISIBLE vertical extent "
                                "(within 0.03 in): the network's ink fills its axes box, while the "
                                "bar chart's axes box is raised by the height of its tick labels + "
                                "x title so that its ink (bars + those two lines of text) ends at "
                                "the same levels; the right-hand dimension nodes sit on the "
                                "left-hand row grid (row indices 28/22/19/14/6/2 out of 0-30, each "
                                "inside its own block)",
            "primitive_colour": "each primitive's name and node outline use the colour of its "
                                "dominant dimension at the primary dose - the same colour its row "
                                "label carries in panel A; the left-hand names are BOLD, like every "
                                "other primitive/dimension name in the figure (they used to be the "
                                "only regular-weight names), and the bolded longest name (1.40 in) "
                                "still fits the 0.62-unit left gutter; the grey 'no significant "
                                "dimension' outline was dropped for that reason",
        },
        "primitives": clean(data["rows"]),
        "primitives_dose_0.8": clean(data["rows_by_dose"][SECONDARY_DOSE]),
    }


def save(fig, path: Path) -> str:
    """先写临时文件再原子替换 —— 目标被 PDF 阅读器（WPS / 资源管理器预览窗格）占用时，
    整次运行不该白跑：新图落到 ``*_pending`` 下并明确报出来，关掉阅读器后重跑即可归位。
    """
    tmp = path.with_name(path.stem + "_pending" + path.suffix)
    for stale in path.parent.glob(path.stem + "_pending*"):
        stale.unlink(missing_ok=True)
    fig.savefig(tmp, dpi=200 if path.suffix == ".png" else None, facecolor="white")
    try:
        import os

        os.replace(tmp, path)
        return "ok"
    except PermissionError:
        return f"目标被占用 → 新图在 {tmp.name}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    data["dose_ladder"] = load_dose(data["baseline"])   # panel F 的数据源（另一张表，见常量区）

    # 图幅：面板 F 是**加高**画布加出来的（A–E 的英寸几何一个没动，见 LAYOUT 上面的换算函数）
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw(fig, data)
    caption = caption_text(data)          # 不再上图：文字说明只落进 metrics.json
    fig.canvas.draw()
    fig.canvas.draw()
    unify_axis_ticks(fig)
    clashes = check_text_overlaps(fig)
    grays = check_no_gray_text(fig)

    results = {name: save(fig, OUT_DIR / f"{STEM}{name}") for name in (".png", ".pdf")}
    plt.close(fig)

    m = metrics(data)
    m["caption_text_not_shown"] = caption
    (OUT_DIR / "figure2_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ {results['.png']:<24} {STEM}.png  ({(OUT_DIR / f'{STEM}.png').stat().st_size // 1024} KB)")
    print(f"✓ {results['.pdf']:<24} {STEM}.pdf")
    print(f"✓ {'ok':<24} figure2_metrics.json")
    print(f"  自检：{data['n_given_verdicts_checked']} 个判定与 xlsx specificity_verdict 列逐行一致")
    if grays:
        print(f"  ⚠ 灰字自检：仍有 {len(grays)} 处灰色文字")
        for txt, col in grays:
            print(f"      「{txt}」{col}")
    else:
        print("  灰字自检：全图无灰色文字（编码含义由坐标轴标题、图注文本与 panel B 的图例承担）")
    if clashes:
        print(f"  ⚠ 版式自检：{len(clashes)} 处图级文字重叠")
        for a, b in clashes:
            print(f"      「{a}」╳「{b}」")
    else:
        print("  版式自检：图级标题/脚注无重叠")
    print(f"  baseline       : { {k: round(v, 3) for k, v in data['baseline'].items()} }")
    mat = np.array([r["reading"] for r in data["rows"]], dtype=float)
    mat8 = np.array([r["reading"] for r in data["rows_by_dose"][SECONDARY_DOSE]], dtype=float)
    print(f"  热图读数范围    : {min(mat.min(), mat8.min()):.3f} … {max(mat.max(), mat8.max()):.3f}（0–1，越大越典型）")
    print(f"  聚类行序首/末   : {data['cluster_order'][0]} → {data['cluster_order'][-1]}")
    print(f"  受影响维度分布  : {data['hist']}")
    print(f"  主导维度分布    : {data['dom_count']}")
    lad = data["dose_ladder"]
    print(f"  panel F        : {lad['n_subjects']} 名被试 × {len(lad['primitive_order'])} primitive"
          f" × {len(DOSE_LADDER)} 档 θ {DOSE_LADDER} × {lad['n_replicates_per_cell']} 次重复"
          f"（逐格均值与 xlsx 核对 {lad['n_means_checked']}/{lad['n_means_checked']}）"
          f" + 完整心智点 x = 1 − θ = 0（参照组 5 份，与 baseline 逐位一致）")
    for pid in lad["primitive_order"]:
        cells = lad["cells"][pid]
        lo = min(DIMS, key=lambda x: cells[DOSE_LADDER[0]]["channels"][x]["mean"])
        intact = [lad["intact"][d]["mean"] for d in DIMS]
        drop = sum(cells[DOSE_LADDER[0]]["channels"][d]["mean"] - lad["intact"][d]["mean"]
                   for d in DIMS) / len(DIMS)
        print(f"    {EN[pid]:<26} 最低通道 @1−θ=0.8 = {lo:<38}"
              f" 六维均值 完整 {sum(intact) / len(DIMS):.3f} → 最弱 {sum(intact) / len(DIMS) + drop:.3f}"
              f"（{drop:+.3f}）")
    print(f"  @0.4 判定       : {data['verdict_count']}")
    print(f"  0.4 比 0.8 更差 : {data['n_stronger_at_0.4']}/31")
    ref_n = {k: len(v) for k, v in data["reference"].items()}
    print(f"  参照组重复数    : {ref_n}（逐维度均值 = baseline 常数，已在 load() 断言）")
    for dose in (PRIMARY_DOSE, SECONDARY_DOSE):
        counts = data["sig_count"][float(dose)]
        strict = data["sig_count_strict"][float(dose)]
        wel = data["sig_count_welch"][float(dose)]
        raw = data["sig_count_uncorrected"][float(dose)]
        print(f"  t 检验显著 @{dose:g} : 合并方差 q<{TEST_ALPHA:g} {sum(counts.values())}/{data['n_tests']} 格"
              f"（不做 FDR 为 {sum(raw.values())}）  逐维度 {list(counts.values())}"
              f"  无边 primitive {data['n_no_sig'][float(dose)]}/31")
        print(f"                同一检验 q<{TEST_ALPHA_STRICT:g} {sum(strict.values())}/{data['n_tests']} 格"
              f"  逐维度 {list(strict.values())}  无边 primitive {data['n_no_sig_strict'][float(dose)]}/31"
              f"   ← 条形图浅色柱")
        print(f"                Welch（不上柱，仅落盘） {sum(wel.values())}/{data['n_tests']} 格"
              f"  逐维度 {list(wel.values())}  无边 primitive {data['n_no_sig_welch'][float(dose)]}/31")
    return 0


if __name__ == "__main__":
    sys.exit(main())
