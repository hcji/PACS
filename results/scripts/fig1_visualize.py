"""Fig.1 绘图：把 fig1_viz_data 算出的面板数据画成一张五面板的多面板图。

版式（上排 2 块、下排 3 块）：
  第一行  A 智能体设计 ｜ B 同一脑 test–retest
  第二行  C 跨模型/组件取值全距 ｜ D 轮次长度 ｜ E 题目顺序
下排三块拼起来的横向跨度与上排 A+B 严格相等（都是 0.988 个图宽）。

全图面板底色为白，文字只用英文。运行：``python results/scripts/fig1_visualize.py``
输出：``results/figure/figure1_pacs_reliability.{png,pdf}`` + ``figure1_metrics.json``。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig1_viz_data as V  # noqa: E402

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
    }
)

# ---------------------------------------------------------------- 调色（Nature 风格：低饱和、可印刷）

NATURE = {
    "ink": "#1A1A1A",
    "muted": "#5F6B7A",
    "grid": "#E1E5EA",
    "panel_bg": "#FFFFFF",
    "panel_edge": "#C7D0D9",
    "brain": "#3A6EA5",
    "analyst": "#2E8B7F",
    "examiner": "#D9822B",
    "seed": "#8A6BBF",
    "len25": "#7FA8C9",
    "len100": "#2B4C6F",
    "reference": "#4A4A4A",
    "dim": {
        "social_interaction_motivation": "#3A6EA5",
        "social_cognition": "#2E8B7F",
        "social_communication_pragmatics": "#D9822B",
        "cognitive_flexibility_adaptation": "#BF4E4A",
        "repetitive_restricted_behavior": "#8A6BBF",
        "novelty_uncertainty_sensory_behavior": "#C9A227",
        "general_cognition": "#9AA1A9",
    },
    "mind": {label: c for label, c in zip(V.MIND_LABELS, ["#2C7BB6", "#8CC5E3", "#EFA05C", "#AF3B3B"])},
}
C = NATURE
INK, MUTED, GRID = C["ink"], C["muted"], C["grid"]
PB, PE = C["panel_bg"], C["panel_edge"]

FS_FIG = 17.0      # 图题（全图仅一处）
FS_TITLE = 13.0    # 面板标题
FS_HEAD = 8.6      # 卡片 / 子图标题、分组标签
FS_BODY = 7.2      # 正文标签、坐标轴标题
FS_SMALL = 6.2     # 次级标签
FS_TINY = 5.5      # 密集注释、热图格内数值
FS_TICK = 6.6      # 全图刻度标签只用这一个字号（并且加粗）

plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK})


def unify_axis_ticks(fig):
    """全图刻度标签统一字号 + 加粗。

    必须在 ``fig.canvas.draw()`` 之后调用：刻度 Text 实体是首次绘制时才建出来的。
    """
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for lab in axis.get_ticklabels():
                lab.set_fontsize(FS_TICK)
                lab.set_fontweight("bold")

#: 热图只画 6 个表型通道；对照通道 general_cognition 不进热图（它仍进 ICC/SEM 等统计）
HEAT_DIMS = V.DIMS[:6]

#: panel B 右侧 strip 的名次配色：颜色只说「第几名」，不说通道（通道由横轴位置给出）
TOP1_RED = "#BF4E4A"
TOP2_BLUE = "#3A6EA5"
RANK_OTHER = "#DCE2E8"

#: 通道英文短名（长名见 fig1_viz_data.py）
DIM_EN = {
    "social_interaction_motivation": "Social interaction",
    "social_cognition": "Social cognition",
    "social_communication_pragmatics": "Communication",
    "cognitive_flexibility_adaptation": "Flexibility",
    "repetitive_restricted_behavior": "Repetitive behavior",
    "novelty_uncertainty_sensory_behavior": "Novelty & sensory",
    "general_cognition": "General cognition (control)",
}


def _truncate(name: str, lo: float, hi: float, n: int = 256) -> LinearSegmentedColormap:
    """截取一段现有色带（避免最浅端贴近白底而看不清）。"""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_{lo}_{hi}", [base(v) for v in np.linspace(lo, hi, n)]
    )


#: Nature 风格测序色带：偏离重（0）→ 深，典型（1）→ 浅。YlGnBu_r 去掉了最浅的一小段。
CMAP = _truncate("YlGnBu_r", 0.03, 0.82)

# 面板外框（figure fraction）
#: 上排 A+B 与下排 C+D+E 各自首尾相接，两排的横向跨度同为 0.988（首尾 x 都是 0.006 / 0.994），
#: 面板之间留 0.014 的缝。
RECT = {
    "A": [0.006, 0.498, 0.530, 0.452],
    "B": [0.548, 0.498, 0.446, 0.452],
    "C": [0.006, 0.045, 0.393, 0.432],
    "D": [0.413, 0.045, 0.300, 0.432],
    "E": [0.727, 0.045, 0.267, 0.432],
}

FIG_W, FIG_H = 13.2, 10.2

#: 是否给四个面板画外框。置 False 时面板靠留白与标题分隔（当前采用）。
SHOW_PANEL_BORDER = False


# ---------------------------------------------------------------- 通用小工具


#: 各面板的标题文字。当前不画面板标题与字母键；把 ``DRAW_PANEL_TITLES`` 打开即可恢复。
PANEL_TITLES = {
    "A": "Agent design: the PACS artificial mind",
    "B": "Test–retest reliability (same brain)",
    "C": "Model-component robustness",
    "D": "Observation-length robustness",
    "E": "Question-order robustness",
}

#: 是否画面板标题（字母键 + 标题文字）。当前为 False。
DRAW_PANEL_TITLES = False


def frame(fig, key: str, title_en: str = ""):
    """建一个面板画布：白底、无边框。返回 frame 轴（坐标 0..1）。

    面板之间只靠留白分隔（``SHOW_PANEL_BORDER`` 打开可恢复描边）。
    ``title_en`` 保留在签名里是为了让调用处仍写明这块是什么；是否真的画由
    ``DRAW_PANEL_TITLES`` 决定，当前不画。
    """
    ax = fig.add_axes(RECT[key])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if SHOW_PANEL_BORDER:
        ax.add_patch(
            FancyBboxPatch(
                (0.002, 0.002),
                0.996,
                0.996,
                boxstyle="round,pad=0,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor=PB,
                edgecolor=PE,
                linewidth=1.0,
                zorder=0,
                clip_on=False,
            )
        )
    if DRAW_PANEL_TITLES:
        # 标题的缩进按「物理英寸」给，否则面板越宽、序号与标题之间的空隙越大
        ax.text(0.000, 0.978, key + ".", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_TITLE, fontweight="bold", color=INK, zorder=5)
        ax.text(0.235 / (RECT[key][2] * FIG_W), 0.978, title_en or PANEL_TITLES.get(key, ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_TITLE,
                fontweight="bold", color=INK, zorder=5)
    return ax


def circle_in(ax, cx, cy, r_in, *, fc, ec="none", lw=0.0, z=5):
    """按「物理英寸」画正圆：axes 坐标下 x/y 的物理尺度不同，直接给半径会画成椭圆。"""
    pos = ax.get_position()
    rx = r_in / (pos.width * FIG_W)
    ry = r_in / (pos.height * FIG_H)
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ry, transform=ax.transAxes,
                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False))


def panel_rect(key: str, x, y, w, h):
    """面板局部坐标（0..1）× → figure 坐标 rect。"""
    L, B, W, H = RECT[key]
    return [L + x * W, B + y * H, w * W, h * H]


def card(ax, x, y, w, h, *, fc="white", ec=PE, lw=0.9, r=0.03, alpha=1.0, z=1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            transform=ax.transAxes,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
            zorder=z,
            clip_on=False,
        )
    )


def arrow(ax, p, q, *, color="#7A8797", lw=1.4, style="-|>", ms=7, z=4):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            transform=ax.transAxes,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            zorder=z,
            clip_on=False,
        )
    )


def _icon_transform(ax, cx, cy, s):
    """图标局部坐标系（原点在中心、半径 0.5）→ axes 坐标，并按物理长宽比校正。"""
    pos = ax.get_position()
    sx = (s * (pos.height * FIG_H)) / (pos.width * FIG_W)
    return Affine2D().scale(sx, s).translate(cx, cy) + ax.transAxes


def icon_brain(ax, cx, cy, s, color):
    tr = _icon_transform(ax, cx, cy, s)
    ax.add_patch(Ellipse((0, 0), 0.92, 1.0, transform=tr, fill=False, ec=color, lw=1.5, zorder=6, clip_on=False))
    for dx, dy, w, h in ((-0.20, 0.16, 0.30, 0.10), (0.02, -0.14, 0.30, 0.10), (0.18, 0.20, 0.24, 0.09)):
        ax.add_patch(Rectangle((dx, dy), w, h, transform=tr, fill=False, ec=color, lw=1.1, zorder=6, clip_on=False))
    ax.add_patch(Circle((0, 0), 0.10, transform=tr, fc=color, ec="none", zorder=7, clip_on=False))


def icon_bubbles(ax, cx, cy, s, color):
    tr = _icon_transform(ax, cx, cy, s)
    ax.add_patch(FancyBboxPatch((-0.46, -0.06), 0.72, 0.50, boxstyle="round,pad=0,rounding_size=0.14",
                                transform=tr, fill=False, ec=color, lw=1.4, zorder=6, clip_on=False))
    ax.add_patch(FancyBboxPatch((-0.06, -0.46), 0.54, 0.44, boxstyle="round,pad=0,rounding_size=0.14",
                                transform=tr, fc="white", ec=color, lw=1.4, zorder=7, clip_on=False))


def icon_doc(ax, cx, cy, s, color):
    tr = _icon_transform(ax, cx, cy, s)
    ax.add_patch(FancyBboxPatch((-0.36, -0.48), 0.72, 0.96, boxstyle="round,pad=0,rounding_size=0.08",
                                transform=tr, fc="white", ec=color, lw=1.4, zorder=6, clip_on=False))
    for i, dy in enumerate((0.22, 0.02, -0.18, -0.34)):
        w = 0.44 if i < 3 else 0.24
        ax.add_patch(Rectangle((-0.24, dy), w, 0.07, transform=tr, fc=color, ec="none", zorder=7, clip_on=False))


# ---------------------------------------------------------------- A agent design


def _chip(ax, cx, cy, text, *, ec, fc, color, w=0.086, h=0.023):
    """角色卡顶部的小圆角标签（θ in prompt / no θ）。"""
    card(ax, cx - w / 2, cy - h / 2, w, h, fc=fc, ec=ec, lw=0.9, r=0.012, z=5)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS_TINY, color=color,
            transform=ax.transAxes, zorder=6)


#: θ 向量里挑出来展示的几条 primitive（英文名取自 `ontology/primitives.py` 的 name_en）。
#: 本实验只拧 `belief_updating`，所以只有它是 0.80，其余全部 1.00 ——
#: 这与真实落盘的 θ 向量一致（见 pacs/data/runs/fig1_main/.../subjects/*.json 的 profile.theta）。
A_THETA_ROWS = [
    ("feature binding", 1.00, "B"),
    ("attention shifting", 1.00, "C"),
    ("prior strength", 1.00, "D"),
    ("belief updating", 0.80, "D"),
    ("novelty valuation", 1.00, "F"),
    ("inhibition", 1.00, "G"),
]


def panel_a(ax, d):
    """A · 智能体设计图：θ 扰动 → PACS 三角色 → 表型读数，下挂三件真实产物。

    版式取自参考图：上排三块（认知基元 / PACS 三角色 / 表型剖面），
    下排三张卡展示**这条链上真实的三件产物**（大脑提示词 / 提问者-大脑对话 / 分析者反馈）。

    上排是示意，下排是实货，三件产物都取自 `fig1_main` 的实际落盘（θ = belief_updating 0.80 号被试
    `002_theta_f2cf3d0fa8a5_r0`）：提示词片段摘自 `build_brain_prompt_steps()` 的原文步骤
    （persona / injection / consequences / conduct），对话是 transcript 里的原文（译成英文），
    读数是 `analysis.readings` 与 `analysis.asd_likeness` 的真实值，Summary 是分析者写的那段总结。
    """
    THETA = "#5B3FA8"
    TOP_B, TOP_T = 0.470, 0.925          # 上排三块
    BOT_B, BOT_T = 0.020, 0.432          # 下排三张产物卡

    # ---------------- 上排左：θ 向量 ----------------
    card(ax, 0.008, TOP_B, 0.170, TOP_T - TOP_B, fc="#F6F3FC", ec="#C9BCE6", lw=1.1)
    ax.text(0.093, TOP_T - 0.046, "Cognitive\nprimitives", ha="center", va="center",
            fontsize=FS_HEAD, fontweight="bold", color=THETA, transform=ax.transAxes,
            zorder=4, linespacing=1.35)
    ax.text(0.093, TOP_T - 0.100, "31 processes · 8 families", ha="center", va="center",
            fontsize=FS_SMALL, color=INK, transform=ax.transAxes, zorder=4)

    for i, (name, val, _fam) in enumerate(A_THETA_ROWS):
        y = TOP_T - 0.144 - i * 0.029
        hit = val < 1.0
        if hit:   # 被拧的那一条底色标出来
            card(ax, 0.016, y - 0.014, 0.154, 0.028, fc="#E7DFF7", ec="none", lw=0.0, r=0.008, z=3)
        ax.text(0.022, y, name, ha="left", va="center", fontsize=FS_SMALL,
                fontweight="bold" if hit else "normal", color=THETA if hit else INK,
                transform=ax.transAxes, zorder=5)
        ax.text(0.160, y, f"{val:.2f}", ha="right", va="center", fontsize=FS_SMALL,
                fontweight="bold", color=THETA if hit else INK, transform=ax.transAxes, zorder=5)
    ax.text(0.093, TOP_T - 0.144 - 6 * 0.029 - 0.006, "… the other 25 at 1.00",
            ha="center", va="center", fontsize=FS_SMALL, color=INK, transform=ax.transAxes, zorder=4)

    ax.text(0.093, TOP_B + 0.074, "θ = (P1 … P31)", ha="center", va="center",
            fontsize=FS_BODY, fontweight="bold", color=INK, transform=ax.transAxes, zorder=4)
    ax.text(0.093, TOP_B + 0.040, "one primitive dialled down,\nthe other 30 left intact",
            ha="center", va="center", fontsize=FS_SMALL, color=INK, transform=ax.transAxes,
            zorder=4, linespacing=1.45)

    # ---------------- 上排中：PACS 三个角色 ----------------
    card(ax, 0.196, TOP_B, 0.486, TOP_T - TOP_B, fc="#F5F9FD", ec="#B9D2EA", lw=1.1)
    ax.text(0.439, TOP_T - 0.034, "PACS", ha="center", va="center", fontsize=FS_TITLE,
            fontweight="bold", color="#2E6EA6", transform=ax.transAxes, zorder=4)
    ax.text(0.439, TOP_T - 0.064, "three roles, three-way blinding",
            ha="center", va="center", fontsize=FS_SMALL, color=INK,
            transform=ax.transAxes, zorder=4)

    roles = [
        (icon_brain, C["brain"], "Artificial Brain", "the perturbed mind",
         ["its own completeness", "the questions asked"],
         ["the other two roles", "their judgments"], True),
        (icon_bubbles, C["examiner"], "Examiner", "a separate LLM",
         ["observation targets", "item bank, transcript"],
         ["θ, primitive names", "completeness values"], False),
        (icon_doc, C["analyst"], "Analyst", "a separate LLM",
         ["the whole dialogue", "anchors and coverage"],
         ["θ and the internal", "construction"], False),
    ]
    cw, cgap, cx0 = 0.142, 0.026, 0.203
    c_bot, c_top = TOP_B + 0.018, TOP_T - 0.082
    for i, (icon, color, name, tag, sees, blind, has_theta) in enumerate(roles):
        x = cx0 + i * (cw + cgap)
        card(ax, x, c_bot, cw, c_top - c_bot, fc="white", ec=color, lw=1.4, r=0.026, z=3)
        _chip(ax, x + cw / 2, c_top - 0.024, "θ in prompt" if has_theta else "no θ",
              ec=THETA if has_theta else "#9AA3AD",
              fc="#F4F0FB" if has_theta else "#F2F4F6",
              color=THETA if has_theta else INK)
        icon(ax, x + cw / 2, c_top - 0.086, 0.042, color)
        ax.text(x + cw / 2, c_top - 0.132, name, ha="center", va="center", fontsize=FS_BODY,
                fontweight="bold", color=color, transform=ax.transAxes, zorder=5)
        ax.text(x + cw / 2, c_top - 0.152, tag, ha="center", va="center", fontsize=FS_SMALL,
                color=INK, transform=ax.transAxes, zorder=5)
        ax.text(x + 0.012, c_top - 0.182, "sees", ha="left", va="center", fontsize=FS_SMALL,
                fontweight="bold", color=color, transform=ax.transAxes, zorder=5)
        for j, line in enumerate(sees):
            ax.text(x + 0.012, c_top - 0.206 - j * 0.028, line, ha="left", va="center",
                    fontsize=FS_SMALL, color=INK, transform=ax.transAxes, zorder=5)
        ax.text(x + 0.012, c_top - 0.270, "cannot see", ha="left", va="center", fontsize=FS_SMALL,
                fontweight="bold", color="#B24A4A", transform=ax.transAxes, zorder=5)
        for j, line in enumerate(blind):
            ax.text(x + 0.012, c_top - 0.294 - j * 0.028, line, ha="left", va="center",
                    fontsize=FS_SMALL, color=INK, transform=ax.transAxes, zorder=5)
        if i < 2:
            arrow(ax, (x + cw + 0.002, c_top - 0.086), (x + cw + cgap - 0.002, c_top - 0.086),
                  color="#5A7A96", lw=1.5, ms=7)

    fl = c_top - 0.086
    arrow(ax, (0.180, fl), (0.196, fl), color="#7A8797", lw=1.6, ms=8)
    arrow(ax, (0.682, fl), (0.706, fl), color="#7A8797", lw=1.6, ms=8)

    # ---------------- 上排右：表型剖面（6 维 + 1 段总结） ----------------
    card(ax, 0.706, TOP_B, 0.288, TOP_T - TOP_B, fc="#F3F9F5", ec="#B7D8C6", lw=1.1)
    ax.text(0.850, TOP_T - 0.046, "Phenotype\nprofile", ha="center", va="center",
            fontsize=FS_HEAD, fontweight="bold", color="#1F6B4A", transform=ax.transAxes,
            zorder=4, linespacing=1.35)
    ax.text(0.850, TOP_T - 0.100, "6 dimensions, each scored 0–1",
            ha="center", va="center", fontsize=FS_SMALL, color=INK,
            transform=ax.transAxes, zorder=4)
    for i, (key, short, _cn) in enumerate(HEAT_DIMS):
        y = TOP_T - 0.144 - i * 0.033
        ax.add_patch(Circle((0.722, y), 0.0085, transform=ax.transAxes, fc=C["dim"][key],
                            ec="none", zorder=5, clip_on=False))
        ax.text(0.738, y, short, ha="left", va="center", fontsize=FS_BODY,
                fontweight="bold", color=INK, transform=ax.transAxes, zorder=5)
        ax.text(0.774, y, DIM_EN[key], ha="left", va="center", fontsize=FS_BODY, color=INK,
                transform=ax.transAxes, zorder=5)
    # 单独一项：分析者对整段对话的总结，它不是表型维度
    ys = TOP_T - 0.144 - 6 * 0.033 - 0.028
    ax.plot([0.722, 0.984], [ys, ys], color="#B7D8C6", lw=0.9, transform=ax.transAxes,
            zorder=4, clip_on=False)
    ax.add_patch(Rectangle((0.714, ys - 0.030), 0.016, 0.016, transform=ax.transAxes,
                           fc="#1F6B4A", ec="none", zorder=5, clip_on=False))
    ax.text(0.738, ys - 0.022, "Summary description", ha="left", va="center", fontsize=FS_BODY,
            fontweight="bold", color="#1F6B4A", transform=ax.transAxes, zorder=5)
    ax.text(0.738, ys - 0.044, "the analyst's summary of the whole\ndialogue — not a dimension",
            ha="left", va="top", fontsize=FS_SMALL, color=INK, transform=ax.transAxes,
            zorder=5, linespacing=1.45)

    # ---------------- 下排：三件真实产物 ----------------
    cards = [
        ("Artificial brain prompt", THETA, [
            "Become a human mind, not an assistant.",
            "You are this mind, from the inside.",
            "",
            "PARAMETER:  belief updating",
            "BASELINE:  1.00    CURRENT:  0.80",
            "the other 30 processes stay at 1.00",
            "",
            "take the consequences as felt, not as",
            "symptoms to report; answer in ≤ 100",
            "characters, and never mention",
            "parameters, models or prompts",
        ]),
        ("Examiner–brain dialogue", C["examiner"], [
            "Examiner:  The bear parks his car, then",
            "falls asleep, and I move it while he",
            "sleeps. Where will he look for it?",
            "Brain:  In the old spot — he does not",
            "know the car was moved.",
            "",
            "Examiner:  The bear wants the whole car",
            "park moved onto the rug. Is that OK?",
            "Brain:  Move it? It is fine where it",
            "is, the cars are parked. Let us stay.",
        ]),
        ("Analyst feedback", C["analyst"], [
            "readings:  S1 0.90   S2 0.88",
            "                 S3 0.82   S4 0.58",
            "                 S5 0.62   S6 0.82",
            "",
            "ASD-likeness:   mild",
            "",
            "Summary:  “insistence on sameness",
            "and rule switching stand out; social",
            "interaction, social cognition and",
            "pragmatics are relatively preserved.”",
        ]),
    ]
    cw2, cgap2, x2 = 0.316, 0.018, 0.008
    for i, (title, color, body) in enumerate(cards):
        x = x2 + i * (cw2 + cgap2)
        card(ax, x, BOT_B, cw2, BOT_T - BOT_B, fc="white", ec="#C9BCE6", lw=1.1, r=0.022, z=3)
        ax.text(x + 0.014, BOT_T - 0.042, title, ha="left", va="center", fontsize=FS_HEAD,
                fontweight="bold", color=color, transform=ax.transAxes, zorder=4)
        for j, line in enumerate(body):
            ax.text(x + 0.014, BOT_T - 0.076 - j * 0.032, line, ha="left", va="center",
                    fontsize=FS_BODY, color=INK, transform=ax.transAxes, zorder=4)


# ---------------------------------------------------------------- B same-brain test–retest


def panel_b(fig, ax, d):
    """B · Left: mind setting × phenotype heatmap with values. Right: is the *identified
    affected set* stable across replicates (not the ranking of all channels)."""
    heat, reps = d["heat"], sorted({s.replicate for s in d["reference"]})
    n_group, gap = len(HEAT_DIMS), 1.0
    unit = n_group + gap
    n_y = unit * len(V.MIND_LABELS) - gap

    hax = fig.add_axes(panel_rect("B", 0.132, 0.150, 0.415, 0.650))
    norm = Normalize(0.0, 1.0)
    for gi, mind in enumerate(V.MIND_LABELS):
        block = [heat[mind][i] for i in range(n_group)]  # 只取 6 个表型通道
        arr = np.array([[np.nan if v is None else v for v in row] for row in block], dtype=float)
        y0 = gi * unit
        hax.pcolormesh(
            np.arange(len(reps) + 1),
            np.arange(y0, y0 + n_group + 1),
            arr,
            cmap=CMAP,
            norm=norm,
            edgecolors="white",
            linewidth=0.8,
        )
        for ri in range(arr.shape[0]):
            for ci in range(arr.shape[1]):
                v = arr[ri, ci]
                if np.isnan(v):
                    continue
                rgba = CMAP(norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                hax.text(ci + 0.5, y0 + ri + 0.5, f"{v:.2f}", ha="center", va="center",
                         fontsize=FS_TINY, fontweight="bold",
                         color="#FFFFFF" if lum < 0.55 else "#12263A", zorder=5)
        hax.text(-0.295, y0 + n_group / 2, V.MIND_SHORT[mind], transform=hax.get_yaxis_transform(),
                 ha="right", va="center", fontsize=FS_HEAD, fontweight="bold", color=C["mind"][mind])
        hax.text(-0.140, y0 + n_group / 2, f"θ={V.MIND_FULL[mind]}", transform=hax.get_yaxis_transform(),
                 ha="right", va="center", fontsize=FS_BODY, color=MUTED)
        for ri, (key, short, _cn) in enumerate(HEAT_DIMS):
            hax.text(-0.022, y0 + ri + 0.5, short, transform=hax.get_yaxis_transform(),
                     ha="right", va="center", fontsize=FS_SMALL, color=INK)
    hax.set_xlim(0, len(reps))
    hax.set_ylim(0, n_y)
    hax.invert_yaxis()
    hax.set_xticks(np.arange(len(reps)) + 0.5)
    hax.set_xticklabels([f"Rep{i + 1}" for i in range(len(reps))])
    hax.xaxis.set_ticks_position("top")
    hax.set_yticks([])
    for sp in hax.spines.values():
        sp.set_visible(False)

    cax = fig.add_axes(panel_rect("B", 0.578, 0.150, 0.016, 0.650))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.ax.set_title("Phenotype\nscore\n0 – 1", fontsize=FS_SMALL, pad=3, color=MUTED, linespacing=1.35)
    cb.set_ticks([])
    cb.outline.set_linewidth(0.6)

    # --- 右侧：受影响维度的名次 ---
    # 四张右图同宽同高、等间距：4 × RH + 3 × RGAP = 0.650 = 左侧热图的厚度，
    # 于是右列的顶边与热图顶边齐平、底边与热图底边齐平（累计高度一致）。
    rk = d["ranks"]
    RX, RW = 0.752, 0.236
    RH, RGAP = 0.137, 0.034
    TOP = 0.150 + 0.650

    def rrow(i: int) -> float:
        """第 i 张右图的底边（面板局部坐标，i 从 0 起）。"""
        return TOP - i * (RH + RGAP) - RH

    # ① 平均名次 × θ（名次 1 = 最受影响，所以纵轴翻转）
    # θ 等距摆位：真值 0.30/0.40 只差 0.10，按线性轴画两个刻度标签会叠在一起。
    ax1 = fig.add_axes(panel_rect("B", RX, rrow(0), RW, RH))
    xs = list(range(len(V.MIND_LABELS)))
    ax1.set_xlim(-0.30, 3.50)
    ax1.set_ylim(6.7, 0.5)
    for key, short, _cn in HEAT_DIMS:
        ys = [rk[m]["avg_rank"][key] for m in V.MIND_LABELS]
        ax1.plot(xs, ys, color=C["dim"][key], lw=1.4, marker="o", ms=3.4,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=4, clip_on=False)
    # 右端直接标线：按最小间距把标签摊开，避免并列的线（θ=0.3 时 S4/S5 同为 1.5）叠在一起
    placed: list[float] = []
    last = V.MIND_LABELS[-1]
    for key, short, _cn in sorted(HEAT_DIMS, key=lambda k: rk[last]["avg_rank"][k[0]]):
        y_end = rk[last]["avg_rank"][key]
        y_lab = y_end if not placed else max(y_end, placed[-1] + 0.66)
        placed.append(y_lab)
        ax1.plot([xs[-1], xs[-1] + 0.16], [y_end, y_lab], color=C["dim"][key], lw=0.7, zorder=3)
        ax1.text(xs[-1] + 0.22, y_lab, short, color=C["dim"][key], fontsize=FS_TINY,
                 ha="left", va="center", zorder=6)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([V.MIND_FULL[m] for m in V.MIND_LABELS])
    ax1.set_yticks([1, 2, 3, 4, 5, 6])
    ax1.set_yticklabels([str(v) for v in range(1, 7)])
    ax1.tick_params(length=2.2, pad=1.5)
    ax1.set_ylabel("Average rank", fontsize=FS_BODY, fontweight="bold", color=INK, labelpad=2)
    ax1.grid(color=GRID, lw=0.5, alpha=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax1.spines[sp].set_color(PE)

    # ②③④ top-2 受影响维度 × 重复（strip）：行 = 重复，列 = 通道，实心点 = 该次重复判出的 top-2
    strips = [m for m in V.MIND_LABELS[:3] if m in rk]
    for si, mind in enumerate(strips):
        sx = fig.add_axes(panel_rect("B", RX, rrow(si + 1), RW, RH))
        sx.set_xlim(0.4, 6.6)
        sx.set_ylim(5.6, 0.4)
        reps = sorted(rk[mind]["top2"])
        for ri, rep in enumerate(reps):
            top1, top2 = rk[mind]["top2"][rep][0], rk[mind]["top2"][rep][1]
            for ci, (key, _short, _cn) in enumerate(HEAT_DIMS):
                # 点的颜色只说名次，不说通道：top-1 红、top-2 蓝、其余灰
                fc, size = (TOP1_RED, 30) if key == top1 else (
                    (TOP2_BLUE, 30) if key == top2 else (RANK_OTHER, 6))
                sx.scatter([ci + 1], [ri + 1], s=size, c=[fc],
                           edgecolors="white" if size > 10 else "none", linewidths=0.6, zorder=4)
            sx.axhline(ri + 0.5, color=GRID, lw=0.5, alpha=0.8, zorder=1)
        for ci in range(len(HEAT_DIMS) - 1):
            sx.axvline(ci + 1.5, color=GRID, lw=0.5, alpha=0.8, zorder=1)
        sx.set_xticks([i + 1 for i in range(len(HEAT_DIMS))])
        sx.set_xticklabels([s for _k, s, _c in HEAT_DIMS] if si == len(strips) - 1 else [])
        sx.set_yticks([r + 1 for r in range(len(reps))])
        sx.set_yticklabels([str(r + 1) for r in range(len(reps))])
        sx.tick_params(length=0, pad=1.5)
        for sp in sx.spines.values():
            sp.set_visible(False)
        # 左侧描一条心智设定色的竖线：它就是下面图例的对应项（子图本身不带标题）
        sx.spines["left"].set_visible(True)
        sx.spines["left"].set_color(C["mind"][mind])
        sx.spines["left"].set_linewidth(1.8)

    # 三张 strip 共用一条纵轴标题：挂在三张图的整段中线上，不做三份重复
    mid = (rrow(1) + rrow(len(strips))) / 2
    ax.text(0.722, mid, "Replicate index", transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=FS_BODY, fontweight="bold", color=INK)

    # 图例摆在第 4 张图下方：第一行说点名次配色，第二行说清这 3 张图分别是哪档心智设定。
    # matplotlib 是按列填的，所以要按 [第一行1, 第二行1, 第一行2, ...] 的顺序传。
    rank_handles = [
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc=TOP1_RED, mec="white", mew=0.6,
               label="Top-1 affected"),
        Line2D([], [], marker="o", ls="none", ms=4.4, mfc=TOP2_BLUE, mec="white", mew=0.6,
               label="Top-2 affected"),
        Line2D([], [], marker="o", ls="none", ms=2.6, mfc=RANK_OTHER, mec="none",
               label="other channel"),
    ]
    set_handles = [
        Line2D([], [], color=C["mind"][m], lw=2.4, solid_capstyle="butt",
               label=f"{V.MIND_SHORT[m]} (θ={V.MIND_FULL[m]})")
        for m in strips
    ]
    interleaved = [h for pair in zip(rank_handles, set_handles) for h in pair]
    ax.legend(
        handles=interleaved, loc="center right", bbox_to_anchor=(RX + RW, 0.078),
        frameon=False, fontsize=FS_SMALL, ncol=3, handlelength=1.4, handletextpad=0.45,
        columnspacing=0.95, labelspacing=0.35, title="Top-2 affected channels per replicate",
        title_fontsize=FS_SMALL, alignment="left",
    )


# ---------------------------------------------------------------- C model-component robustness


#: panel C 的两行（＝两条换件臂）。本图不画换提问者臂；换题序臂在 panel E 单独成图。
C_ROWS = ["axis2_brain_pro", "axis3_analyst_pro"]

#: 一格之内 4 条线的横向偏移（格宽 = 1），顺序与 MIND_LABELS 一致
C_DX = (-0.30, -0.10, 0.10, 0.30)

#: 行标签直接写模型名（不再用 A/B 指代）。角色顺序固定为 Brain → Examiner → Analyst，
#: 换掉的那个角色自然落在自己的位置上。配置与已落盘的 `condition.json` 逐条核对过。
#: 模型全名：DS Flash v4 = `deepseek-flash`，DS Pro v4 = `deepseek-v4-pro`。
C_ROW_LABELS = {
    "axis2_brain_pro": "Brain = DS Pro v4 · Analyst = DS Flash v4",
    "axis3_analyst_pro": "Brain = DS Flash v4 · Analyst = DS Pro v4",
}


def panel_c(fig, ax, d):
    """C · 每个「换了哪个模型」条件下，各表型通道在 4 档心智设定上的取值全距。

    一行一个条件；行内 7 个通道，每个通道下并排 4 条竖线 —— **线的颜色 = 心智设定**
    （M1→M4 从左到右，剂量由重到轻），两端是该通道在**该 (条件, 心智设定)** 下的最小/最大值，
    白边圆点 = 中位数。同参数重跑参照臂不作为单独曲线画出：它的读数在面板 B 的
    热图里就是「同脑」那一列，数值另见 `figure1_metrics.json` 的 `profile_value_ranges`。
    """
    rbm = d["ranges_by_mind"]
    group = {s["condition"]: s["group"] for s in V.ARM_SPECS}

    X0, XW = 0.042, 0.658          # 画区（面板局部坐标）
    # 行高给足，但行间要留缝：行标题不能与上一行垂下来的 0.0 刻度、也不能互相挤在一起
    AX_TOP, AX_H, PITCH = 0.884, 0.342, 0.392
    n_dim = len(V.DIMS)

    for i, cond in enumerate(C_ROWS):
        y0 = AX_TOP - i * PITCH - AX_H
        sax = fig.add_axes(panel_rect("C", X0, y0, XW, AX_H))
        sax.set_xlim(0.42, n_dim + 0.58)
        sax.set_ylim(-0.05, 1.05)
        for j, (key, _short, _cn) in enumerate(V.DIMS):
            for mind, dx in zip(V.MIND_LABELS, C_DX):
                r = rbm.get(cond, {}).get(mind, {}).get(key)
                if r is None:
                    continue
                lo, med, hi, _n = r
                sax.plot([j + 1 + dx] * 2, [lo, hi], color=C["mind"][mind], lw=2.2,
                         solid_capstyle="round", zorder=4)
                sax.plot([j + 1 + dx], [med], marker="o", ms=2.4, ls="none",
                         mfc=C["mind"][mind], mec="white", mew=0.6, zorder=5)
        sax.set_xticks(range(1, n_dim + 1))
        sax.set_xticklabels([s for _k, s, _c in V.DIMS] if i == len(C_ROWS) - 1 else [])
        # 通道标签一律黑：这个面板的颜色已经改指心智设定，再给标签上通道色会两套色系打架
        for tick in sax.get_xticklabels():
            tick.set_color(INK)
        # 网格线按 0.25 走，但只标 0.0 / 0.5 / 1.0 三个（行与行之间留白本来就窄）
        sax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        sax.set_yticklabels(["0.0", "", "0.5", "", "1.0"])
        sax.tick_params(length=2.2, pad=1.5)
        sax.grid(axis="y", color=GRID, lw=0.5, alpha=0.85, zorder=0)
        sax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            sax.spines[sp].set_visible(False)
        sax.spines["bottom"].set_color(PE)

        # 行标签用 A/B 记法写清三个角色各是谁（n / k 由本图注给出）
        ax.text(X0, y0 + AX_H + 0.012, C_ROW_LABELS[cond], transform=ax.transAxes,
                ha="left", va="baseline", fontsize=FS_HEAD, fontweight="bold", color=INK)

    # 图例改成读心智设定（4 档 θ），不再列通道 —— 通道由横轴位置给出
    handles = [
        Line2D([], [], color=C["mind"][mind], lw=2.4, solid_capstyle="round",
               label=f"{V.MIND_SHORT[mind]} (θ={V.MIND_FULL[mind]})")
        for mind in V.MIND_LABELS
    ]
    handles.append(Line2D([], [], marker="o", ls="none", ms=3.4, mfc="#5A6472", mec="white",
                          mew=0.7, label="median reading"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.725, AX_TOP + 0.028),
              frameon=False, fontsize=FS_SMALL, handlelength=1.2, handletextpad=0.45,
              labelspacing=0.40, title="Mind setting", title_fontsize=FS_SMALL, alignment="left")



# ---------------------------------------------------------------- D observation length


def _cell_scatter(fig, rect, d, cond, ylab, subtitle):
    """一个「50 轮参照读数 vs 该条件读数」的单元格散点（面板 D / E 共用同一套画法）。

    只画 6 个表型通道（对照通道 G 不上图），所以每张 24 个点 = 4 档心智设定 × 6 通道。
    ``R²`` 就在**这 24 个被画出来的点**上现算（带截距的最小二乘，R² = r²）——
    它不等于 `figure1_metrics.json` 里的 `r_vs_ref`（那个含 G、是 28 个格），引用时要分清。
    """
    sax = fig.add_axes(rect)
    ref = V.cell_means(d["reference"])
    other = V.cell_means(d["groups"][cond])
    keys = sorted([k for k in ref if k in other and k[1] in V.PHENOTYPE_KEYS],
                  key=lambda k: (V.MIND_LABELS.index(k[0]), V.DIM_KEYS.index(k[1])))
    xs = np.array([ref[k] for k in keys])
    ys = np.array([other[k] for k in keys])
    for k, xv, yv in zip(keys, xs, ys):
        sax.scatter([xv], [yv], s=26, c=[C["dim"][k[1]]], alpha=0.88,
                    edgecolors="white", linewidths=0.5, zorder=4)
    lim = (0.05, 1.02)
    sax.plot(lim, lim, color="#9AA3AD", ls=(0, (4, 3)), lw=1.0, zorder=2)
    r2 = float(np.corrcoef(xs, ys)[0, 1]) ** 2
    box = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2)
    sax.text(0.05, 0.955, f"R² = {r2:.2f}", transform=sax.transAxes,
             fontsize=FS_HEAD, fontweight="bold", color=INK, va="top", zorder=6, bbox=box)
    sax.set_xlim(*lim)
    sax.set_ylim(*lim)
    sax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sax.tick_params(length=2.4)
    sax.set_xlabel("50-turn reference score", fontsize=FS_BODY, fontweight="bold",
                   color=INK, labelpad=2)
    sax.set_ylabel(ylab, fontsize=FS_BODY, fontweight="bold", color=INK, labelpad=2)
    sax.set_title(subtitle, fontsize=FS_HEAD, fontweight="bold", color=INK, pad=5, loc="left")
    for sp in ("top", "right"):
        sax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        sax.spines[sp].set_color(PE)
    return sax


#: 三张散点同宽同高。宽度按**物理英寸**给再换算成局部比例 —— D 与 E 的面板宽度不同，
#: 只写局部比例会让 E 那张悄悄比 D 的窄。
_SC_IN, _SC_LEFT_IN, _SC_GAP_IN = 1.70, 0.23, 0.24
#: 高度对齐「面板 C 两行叠起来」的跨度（C 的行从 0.150 到 0.884）
_SC_Y, _SC_H = 0.150, 0.734


def _scatter_rect(panel: str, x_in: float):
    w = RECT[panel][2] * FIG_W
    return panel_rect(panel, x_in / w, _SC_Y, _SC_IN / w, _SC_H)


def panel_d(fig, ax, d):
    """D · 50 vs 25、50 vs 100 轮：单元格读数对 50 轮参照读数。"""
    for i, (cond, sub, color, ylab) in enumerate([
        ("axis4_turns_25", "50 vs 25 turns", C["len25"], "25-turn score"),
        ("axis4_turns_100", "50 vs 100 turns", C["len100"], "100-turn score"),
    ]):
        _cell_scatter(fig, _scatter_rect("D", _SC_LEFT_IN + i * (_SC_IN + _SC_GAP_IN)),
                      d, cond, ylab, sub)


def panel_e(fig, ax, d):
    """E · 题目顺序改变（题序 seed 42 → 7，前 6 位题序不变）：同样的单元格散点。

    这一臂的对照口径与面板 D 完全一致（都是对 50 轮参照的单元格均值），所以三张散点可以直接比：
    r 越靠近 1、点越贴着对角线，说明该组件改动越不改变读数。
    """
    _cell_scatter(fig, _scatter_rect("E", 0.21),
                  d, "axis3_probe_seed7", "seed-7 score", "seed 42 vs seed 7")

    # 通道色标：三张散点共用，摆在面板 E 的右侧（整排散点之后）。
    # 样式跟面板 C 那条图例对齐：挂在本面板的 frame 轴上、同字号同间距、文字一律黑，颜色只由色块承担。
    # 对照通道 G 不画进散点，所以这里也不列它。
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", ms=5.0, mfc=C["dim"][key],
                   mec="white", mew=0.6, label=f"{short}   {DIM_EN[key]}")
            for key, short, _cn in V.DIMS if key in V.PHENOTYPE_KEYS
        ],
        loc="upper left", bbox_to_anchor=(2.06 / (RECT["E"][2] * FIG_W), 0.906),
        frameon=False, fontsize=FS_SMALL, handlelength=1.2, handletextpad=0.45,
        labelspacing=0.40, title="Channels", title_fontsize=FS_SMALL, alignment="left",
    )



# ---------------------------------------------------------------- 组装


def build_figure(outdir: Path):
    d = V.build()
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    fig.text(0.006, 0.9740,
             "Figure 1  |  Reproducibility and Robustness of the PACS System",
             fontsize=FS_FIG, fontweight="bold", color=INK, va="center")

    ax_a = frame(fig, "A", "Agent design: the PACS artificial mind")
    ax_b = frame(fig, "B", "Test–retest reliability (same brain)")
    ax_c = frame(fig, "C", "Model-component robustness")
    ax_d = frame(fig, "D", "Observation-length robustness")
    ax_e = frame(fig, "E", "Question-order robustness")

    panel_a(ax_a, d)
    panel_b(fig, ax_b, d)
    panel_c(fig, ax_c, d)
    panel_d(fig, ax_d, d)
    panel_e(fig, ax_e, d)

    # 全图刻度统一字号 + 加粗：先 draw 一次把刻度实体建出来，再改才落得实
    fig.canvas.draw()
    unify_axis_ticks(fig)

    fig.text(
        0.006, 0.0075,
        "Source: results/data/result_1_robustness.xlsx — 116 simulated subjects, 7 conditions, 4 mind settings, "
        "36-month mind, 7 scored channels. Reading 1.0 = age-typical, 0.0 = severely deviant;",
        fontsize=FS_TINY, color=MUTED, va="bottom",
    )
    fig.text(
        0.006, 0.0015,
        "higher ICC, lower SEM/CV = more reliable. Simulated agents, not real children: methodological research "
        "material, not for medical diagnosis or as a substitute for professional assessment.",
        fontsize=FS_TINY, color=MUTED, va="bottom",
    )
    return d, fig


def write_artifacts(d, fig, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / "figure1_pacs_reliability.png"
    pdf = outdir / "figure1_pacs_reliability.pdf"
    fig.savefig(png, dpi=200)
    # 矢量版常被 PDF 阅读器占着句柄。先原地重试几次（阅读器释放句柄往往只要一瞬间），
    # 仍写不进去才换名字写 —— 别把整次出图判失败，也别留下一个旧内容的同名文件。
    for attempt in range(3):
        try:
            fig.savefig(pdf)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.8)
            else:
                alt = outdir / "figure1_pacs_reliability.vector.pdf"
                print(f"  ⚠ {pdf.name} 被其他程序占用（PDF 阅读器？），已重试 3 次；"
                      f"矢量版改写 {alt.name} —— 请关掉阅读器后重跑，把正名文件刷新")
                fig.savefig(alt)
                pdf = alt

    metrics = {
        "reference_arm": {
            k: getattr(d["arms"][V.REFERENCE_CONDITION], k)
            for k in (
                "n_subjects", "n_targets", "n_replicates", "icc", "icc_lo", "icc_hi",
                "sem", "sem_lo", "sem_hi", "cv_mean", "within_profile_pearson",
                "within_profile_spearman", "within_tolerance", "rank_stability",
                "kappa", "weighted_kappa", "unanimous", "turns_mean",
            )
        },
        "arms": {
            c: {k: getattr(m, k) for k in (
                "group", "change", "n_subjects", "n_targets", "n_replicates", "icc", "icc_lo", "icc_hi",
                "sem", "sem_lo", "sem_hi", "cv_mean", "within_profile_pearson", "within_tolerance",
                "rank_stability", "kappa", "unanimous", "r_vs_ref", "r_vs_ref_lo", "r_vs_ref_hi",
                "rho_vs_ref", "mad_vs_ref", "max_abs_diff_vs_ref", "tol_vs_ref", "bias_vs_ref",
                "loa_low", "loa_high", "turns_mean",
            )}
            for c, m in d["arms"].items()
        },
        "per_mind_setting": {
            V.MIND_SHORT[m]: {k: v for k, v in vals.items() if k not in ("matrix", "reps")}
            for m, vals in d["per_mind"].items()
        },
        "affected_dimension_identification": {
            "tau": V.AFFECTED_TAU,
            "rule": "a phenotype channel is flagged in a replicate when its reading falls "
                    ">= tau below the control channel in that same replicate; "
                    "Jaccard(empty, empty) = 1",
            "plotted": False,
            "by_mind_setting": {
                V.MIND_SHORT[m]: {k: v for k, v in a.items() if k != "flags"} | {"flags": a["flags"]}
                for m, a in d["affected"].items()
            },
        },
        "affected_rank": {
            "rule": "per replicate, the 6 phenotype channels ranked by reading ascending "
                    "(= by deficit-from-control descending); rank 1 = most affected. "
                    "Ties get the average rank. Top-2 = the two smallest readings, "
                    "ties broken by DIM order. top2_by_replicate is rank-ordered: "
                    "index 0 = top-1 (red on the figure), index 1 = top-2 (blue).",
            "by_mind_setting": {
                V.MIND_SHORT[m]: {
                    "theta": r["theta"],
                    "avg_rank": r["avg_rank"],
                    "top2_by_replicate": r["top2"],
                    "top2_counts": r["top2_counts"],
                }
                for m, r in d["ranks"].items()
            },
        },
        "profile_value_ranges_by_mind": {
            "rule": "per (condition × mind setting × channel): min / median / max over every raw "
                    "reading in that cell (k = the arm's replicate count), no trimming. "
                    "This is what panel C plots.",
            "by_condition": {
                c: {
                    V.MIND_SHORT[mind]: {
                        k: {"min": v[0], "median": v[1], "max": v[2], "n_readings": v[3]}
                        for k, v in rows.items()
                    }
                    for mind, rows in by_mind.items()
                }
                for c, by_mind in d["ranges_by_mind"].items()
            },
        },
        "profile_value_ranges": {
            "rule": "per (condition × channel): min / median / max over every raw reading that "
                    "channel took in that condition (4 θ levels × replicates, k=5 或 3), no trimming. "
                    "The grey line drawn next to each channel is the same statistic for the shared "
                    "same-brain reference arm.",
            "by_condition": {
                c: {k: {"min": v[0], "median": v[1], "max": v[2], "n_readings": v[3]}
                    for k, v in rows.items()}
                for c, rows in d["ranges"].items()
            },
        },
        "consistency_check_vs_pipeline": d["checks"],
    }
    (outdir / "figure1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return png, pdf


if __name__ == "__main__":
    OUT = Path(__file__).resolve().parents[1] / "figure"
    data, figure = build_figure(OUT)
    print("\n".join(data["checks"]))
    png, pdf = write_artifacts(data, figure, OUT)
    print(f"\n写出：\n  {png}\n  {pdf}")
