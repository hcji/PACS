"""Fig. 4：认知基本单元（cognitive primitive）的**扰动组合效应**——从"能不能重复"到"组合之后会怎样"。

## 五个面板（a–e）与它们各自回答的一个问题

- **Panel a · Stability across replicates**：465 个双变量组合的交互效应在 3 次重复测量里稳不稳。
  上：**横轴 = 该组合 3 次重复的测量 SD，纵轴 = 分布密度**（高斯核、Silverman 带宽，按 3/3 同号与
  2/3 同号两组**堆叠**，所以两块面积 = 两组占比）；下：同号率随 |效应| 的变化（噪声底基本不随强度变，
  所以只有大效应才压得住它）。本图不画「结论层面的稳定性」一张（那张的第三个柱子上
  挂着 aggravated 的格子计数），数值仍留在 metrics 的 `panel_a.conclusion_stability` 里。
- **Panel b · Aggravation of an already-present phenotype**：坡度图的**两端是「Main cognitive（一个
  已经满足受损判据的单因素）」与「该因素 + 它的 secondary 伙伴（实测的 pair）」**——比的是
  "再加一个损伤之后实测会不会更重"，**不是**「两个单因素加和（可加预期 A + B − baseline）」与
  pair 的比较；可加预期只在 panel e 里出现。
  入组条件沿用 Fig.2：某个单因素在该维度上**已经受损**（`Δ ≥ 0.15` 且 `q < 0.05`）。
  核心问题：**第二个 cognitive perturbation 是否进一步加重原本已经出现的表型？**
  每个表型一个小斜坡图（paired/slope），两端是 **Main cognitive（左）** 与 **Its secondary
  cognitive（右）**。**扇面**的画法（见 §本图新引入的定义 3）：**每一个满足判据的单因素都是一条线**
  ——左端 = 它自己的 Δ，右端 = **它 + 它最坏的那个 partner**（同一维度内每个 Main 只留变化量最大的
  那个伙伴）。**除最显著的那一条外全部灰**（颜色只留给重点），最显著的那条加粗、
  按方向着色，并在右下角写出"谁 + 加谁 + 变化量"；**不再标 "Δ x → y"**（端点由坐标轴直接读出）。不再统计 better / worse 的格子数
  （要看比例回到 `figure4_metrics.json` 的 `panel_b`）。
- **Panel c · Emergence of a new phenotype**：核心问题：**两个 perturbation 联合后，能否产生任一
  单因素都没有出现的新表型？** 上：每个维度上有**多少对**满足"任何单个 cognitive 都不显著、而
  pair 显著（`Δ ≥ 0.15` 且 `q < 0.05`）"；下：每个**新涌现表型**（= 每个维度）里最显著的
  那一对（y 轴 = 维度名，条子 = 该格的 Δ，右边写 pair 名）。
- **Panel d · Interaction landscape**：31 个 primitive × 6 个表型维度的「受累伙伴数」矩阵。
  **不回答"谁最强"**，回答「谁和别的 cognitive 损伤组合起来最可能波及更多维度」——
  格子 = 该 primitive 有多少个伙伴在该维度上受损。右侧两条横条（行和 / 新表型格子数）本图不画，数字仍留在 metrics 的 `panel_d.totals` 与 `new_phenotype_partners` 里。
- **Panel e · Representative cognitive interactions**：**两个代表例** —— 一个**协同**（真实叠加超过
  可加预期）、一个**缓冲**（真实叠加低于可加预期）。版式：**第二行最右列**上下两张柱子图
  （逐维度画 单因素 A / 单因素 B / 可加预期 / 真实叠加 四根柱子，真实叠加那根按方向着色：
  红 = 比可加更差、蓝 = 更好），每张图上方只留一行角色标签 + 一行 pair 名；
  **右列的 Observations / Possible cause 文字不出现在图上**（正文仍由 `_e_case_texts()`
  现算并写进 metrics 的 `panel_e`，硬门 7 照旧验那两条断言）。
  **四根柱子都带 error bar**（= 该量在 3 次重复上的 SD，ddof = 1；可加预期没有自己的测量值，
  逐重复算成 `drop_A + drop_B`，误差棒是这个和的 SD），硬门 8 用逐重复读数独立对账（容差 5e-4）。
  **注意柱子的 SD 与结论的 SD 不是一个量级**：同一 repeat 里所有条件的读数会一起上下浮动
  （示例 pair 的三个量 SD 0.03~0.13），交互效应是差值，这个整体浮动被减掉了 —— 协同例的三次
  `A + B − 组合` 是 0.9517 / 0.9550 / 0.9517，所以 6 通道聚合的交互效应 SD 只有 0.002。
  逐维度的交互效应 SD 仍有 0.04~0.23，所以正文里那句"3 次重复一致"标的是**6 通道聚合**的 SD
  （选 pair 用的也是它），不是柱子上那根须；读图要在**同一维度内**比较真实叠加与可加预期
  （四根柱子共享同一个 repeat 的整体浮动），不要拿不同维度的绝对高度互比。
  **图例已化简**成两行三列（四根柱子 + 轴下方两个记号；SD 的含义交给图注），右对齐压在
  panel 右上角；原来 panel 底部那条七格图例与代号对照表都不在那里了。

## 口径（全图统一，且**沿用上游判据，不另立一套**）

表型读数 0–1、**越大越典型**；`Δ = 基线 − 读数`，**Δ 越大 = 越受损**。数据表里

    interaction = combined − expected_additive        （逐维度算）

`pacs/brain/perturb.py::interaction_sign` 以**损伤量**为尺度：`combined < expected` → superadditive
（协同）；`combined > expected` → subadditive（缓冲）。本图统一画

    interaction strength = expected_additive − combined = −interaction

**正 = 协同（红）；负 = 缓冲（蓝）**。全图配色语义（**图上不再写**，只在这里与图注里定义）：

| 颜色 | 含义 | 出现在 |
|---|---|---|
| 红 `#D1443F` | 比可加预期更差（superadditive / 协同 / 加重） | b 的坡线与变化量、e 的真实叠加柱与角色标签 |
| 蓝 `#3D86C6` | 比可加预期更好（subadditive / 缓冲 / 减轻） | b 的坡线与变化量、e 的真实叠加柱与角色标签 |
| 紫 `#7B52AB` | 新表型（两个单因素都没占到、组合占到了） | c 的率条与实例条、d 的紫柱、e 的角色标签 |
| 灰 `#9AA1A9` / `#5F6469` | 中性：单因素本身、可加预期、行和、unchanged | a、b、d、e |

panel a 的分布图用绿/橙（3/3 与 2/3 同号），只表示"重复一致性"，由自己的图例说明，不占用红/蓝。
窄轴上的两字代号（`Mo So Pr Fl RR Se`）由**整图唯一一份** `CODE_KEY` 对照表定义，印在**图底脚注**
（两行；d / e 的 6 列同序，所以一份就够，见 §版式约定）。

"受影响"判据**逐字沿用 Fig.2**：`f2._t_p(perturbed_reps, f2._intact_reference(), equal_var=True)`
的两样本 t 检验 + 同档位同维度族内 BH-FDR（单因素族 = 该档位 31 个 primitive；组合族 = 该维度下
465 对），且 `Δ ≥ f2.TARGET_DROP (0.15)` 且 `q < f2.TEST_ALPHA (0.05)`。

**全部单因素与组合都在 θ = 0.8 这一档上比**：组合表只有 0.8，单因素也取 0.8（31 个 primitive ×
3 次重复），差值、阈值、可加预期、新表型判定全部是同一个档位内的量。档位无关的只有一个——
t 检验的参照组（`result_1` 的完整心智读数，无扰动）。

所有均值只在 **6 个 ASD 表型维度**上算，`general_cognition`（对照通道）不进入任何均值：它按构造对任何
扰动都会掉，混进来会把"哪个表型受损"稀释成"扰动有多重"。

## 反解出的逐维度基线（本图 Δ 的零点）

`baseline_d = interaction_<dim> − combined_dim + single_A_dim + single_B_dim` 逐对反解取中位数，
465 对上的离散只有 **0.0002**，六维均值恰好等于 6 通道基线 0.8957：

    SIM 0.9280 · SC 0.9140 · SCP 0.9160 · CFA 0.8920 · RRB 0.8540 · NUS 0.8700

硬门 4 再对一遍：单因素表里 `{dim}_mean + (-{dim}_delta) = baseline_d` 必须逐条成立。

## 本图新引入的定义（上游没有，写在这里以免被误读）

1. **重复内交互效应**：逐 replicate 用同一 replicate 的单因素 6 通道读数重算
   `effect_rep = (single_A6 + single_B6 − baseline6) − combined6`；它的 3 次均值与表内
   `interaction_phenotype_mean` 一致（实测最大偏差 0.0091，硬门 5）。逐 replicate **同号**只说明
   方向稳健，**不等于**逐 replicate 都显著。
2. **加重 / 持平 / 减轻**：以**主因素**（见下）为参照，`Δ_comb − Δ_main` 记为变化量，
   方向阈值 **±0.03**（远小于受影响判据 0.15，只用来分方向）。
3. **panel b 的扇面（fan）**：panel b 的单位不是 (pair, 维度) 格子，而是**取向** ——
   左端 = **Main cognitive**：这一对里满足受影响判据（`Δ ≥ 0.15` 且 `q < 0.05`）的那个单因素；
   右端 = 它 + 另一个因素（**Its secondary cognitive**，即实测的 pair）。
   **所有满足判据的单因素都纳入**，每个维度画成一张扇面：
   该维度上每一个符合条件的 Main cognitive 各一条线，每条只取**它自己最坏的那个 partner**
   （同一 Main 配 30 个伙伴里变化量最大的那个）；线色 = 该线方向，只有最显著的那一条加粗并标名称。
   六维的扇面大小分别是 9 / 4 / 9 / 13 / 12 / 14 条，合计 **61 条**（池子 = 1517 格 → 1830 个取向；
   1830 个取向按 Main 去重后就是这 61 条）。定义在 `_orientations()`，硬门 10 逐条复核
   「每条线的 Main 都满足判据 / 变化量 = 两端之差 / 该维度的每个 Main 都在扇面里 /
   每个 Main 用的确实是它自己最坏的那条 / 标名的那条是扇面里最大的」。
4. **结论稳定性**：逐 replicate 直接套同一阈值（`Δ_rep ≥ 0.15` 即该 replicate 也"受影响"），
   不逐 replicate 做检验（n = 1 时检验无意义）。这条量（三类结论的"3/3 成立率"）**不出现在图上**；数字仍在 `panel_a.conclusion_stability` 里。
   **逐重复读数是逐维度的**（`combo_dim_r`）；若改用 6 通道均值逐重复读数，对"某一维的格子"
   来说口径不对 —— 逐维度口径下是 affected 56.7% / new 32.3% / worse 19.8%
   （6 通道均值口径为 38.0% / 44.4% / 14.0%）。panel a 的两张图仍然用 6 通道均值口径
   （它讲的就是聚合层面的稳定性）。
5. **面板 e 的 error bar**：柱子 = 3 次重复的均值，须 = 同一个量在这 3 次重复上的 SD（ddof = 1）。
   可加预期是外推量，没有自己的测量值，逐重复按 `drop_A + drop_B` 算、再取这 3 个值的 SD。
   须很长（最大 0.25）不等于"结论不稳"：同一 repeat 里各条件同向浮动，差值把这块浮动减掉了，
   所以"真实叠加 vs 可加预期"比单看一根柱子稳得多（见 panel e 一节的数）。

## 口径取舍（都是数据决定的，改任何一处都会换掉画出来的例子）

1. **panel b 的池子不含"组合也必须受影响"这一条**：1517 格 = 只要有一个单因素已经受损就进。
   本图不统计三档比例，但这条口径仍然决定"每个维度的最坏取向"是从哪个池子里挑出来的。
2. **"最坏的一格"不等于"稳定的一格"**：172 个加重格里只有 19.8% 在 3 次重复里都保持加重
   （组合受影响 56.7%、新表型 32.3%；口径见定义 4），所以 panel b 画出的那一条坡线只能读作
   "该维度上加重最明显的一条"；panel e 的协同例子另加了「pair 级 3 次同号 + 6 通道聚合 sd < 0.05」，
   画出来的那对 sd 只有 0.002。
3. **取向规则会放大最坏值**：一个格子的两条合法坡线里只画变化量最大的那条（见 §取向），
   等于在 1830 个取向里挑最极端的一个，比按格子挑（1517 个）更容易挑到高值 —— 这是"报最坏情况"
   的必然代价，读 panel b 的 +0.18 ~ +0.40 时要记住它是**上界**，不是该维度的典型值。
4. **"重复一致"是聚合层面的结论，不是逐维度的**：选代表 pair、判"3 次同号"、panel a 的两张图
   全用**6 通道均值**口径（同一 repeat 的整体浮动在差值里被减掉，所以聚合交互效应可以稳定到 0.002）；
   逐维度的交互效应 SD 有 0.04~0.23，图上四根柱子的 error bar（各自量的 SD，最大 0.25）比这个
   更大。**这不是矛盾**：柱子的须量的是"这个条件在这 3 次里整体浮动多少"，不是"真实 vs 可加
   这个差稳不稳"。要在同一维度内做比较，不要跨维度比绝对高度。
5. **panel c 的下半张按 pair 去重**：99 格里只有 74 对，同一对可以在两个维度上都算新表型；
   只画每对最强的那一格（"一对可以造出多于一个新表型"这件事写在图注里，图上不再写）。

## 可加预期会越界（panel e 读图须知）

可加预期是读数的线性外推：`expected = A + B − baseline`，两个单因素都掉到 0.2~0.4 时它可以算到负数。
本图不截断、不裁剪（截断会把交互效应改掉），只把 y 轴按数据取上限；所以 6 个代表 pair 都特意选了
「可加预期仍在量程内」的（`Δadd ≤ 0.70`），最强的那些缓冲 pair（可加预期算到 −0.08）留给 panel b 交代。

## 硬门（不通过就不落盘）

1. 465 对 = C(31, 2)，每对 3 个 replicate、6 个维度齐；
2. 表内 `interaction_phenotype_mean` = 6 个 `interaction_<dim>` 的均值（容差 1e-4）；
3. 逐维度链路自洽：`combined_dim − (A + B − baseline_d)` 重算出表内 `interaction_<dim>`（容差 5e-4）；
4. 单因素 `mean + (−delta)` = 逐维度基线（容差 1e-4）；
5. 重复内交互效应的 3 次均值 = 表内 `interaction_phenotype_mean`（容差 0.02）；
6. panel d 矩阵列和 = 2 × 该维度上 affected 的组合数；
7. panel e 的 2 对代表 pair（一个协同、一个缓冲）必须「3 次重复同号」、6 维齐全、不是同一对，
   且 `_e_case_texts()` 写进 metrics 的 Observations 两条断言成立（协同例至少有一维新表型且至少有一维
   超过可加预期；缓冲例 6 维全部低于可加预期且没有一维还在受损水平）；
8. panel e 画上去的每个数 = 表内 `interaction_<dim>` 取负（容差 1e-9，provenance 门）；
   每根柱子的**均值与 error bar 还必须能由逐重复读数独立重算**（均值 = 3 次重复均值、须 = 这 3 个
   值的 SD，容差 5e-4 —— 单因素汇总表的 `{d}_mean` 是四舍五入过的，与 by_subject 三行均值最多差
   3.3e-05，可加预期要把两个偏差相加所以到 6.7e-05；组合表逐位一致）；
9. 已知答案门：最强协同必须是 `prior_strength + relational_representation`、最强缓冲必须是
   `representation_updating + rule_updating` —— 符号约定一旦翻掉这两个"最"立刻换人。
10. panel b 的取向门：每维画出的那条坡线，Main 必须满足受影响判据、两端不能是同一个因素、
   变化量必须等于两个端点读数之差（容差 1e-9）、且必须是该维度所有取向里变化量最大的那条。

另有三条排版自检（图会照画，但会打印出来）：`check_text_clashes`（只量文字，不算引线）、
`check_inside_frames`（figure 级文字与坐标轴标题必须落在某个面板框内 —— 这个查得出"字跑出面板"，
前者查不出）与 `check_text_covered`（某段文字被**另一个**坐标轴的不透明白底压住超过 25%：两个子图
上下排得太近时就查得出"上面那张的 x 轴标题被下面那张的白底吃掉"）。

## 复用

`importlib` 加载 `fig2_visualize`，不复制调色板 / 字号 / 判据 / 落盘 / 排版自检：
`_t_p` / `_bh_fdr` / `_intact_reference` / `TARGET_DROP` / `TEST_ALPHA` / `EN` / `NATURE` / `INK` /
`PE` / `DIM_1LINE` / `DIM_2L` / `FS_BODY` / `FS_DOSE` / `_truncate` / `save()`（原子替换）/
`check_no_gray_text()`。**字号与刻度不照抄**：f2 的正文档是 7.2 pt、`unify_axis_ticks()` 把刻度
写死成 7.2 pt 加粗，本图整体放大到正文 9.0 pt / 标题 12.0 pt（见 §版式约定），所以自己实现了一份
`unify_axis_ticks()`（同一条规矩：刻度不单独设档，跟着正文档走）。

## 用法

```bash
python results/scripts/fig4_interaction_stability.py            # 画图 + 落盘 png/pdf/metrics.json
python results/scripts/fig4_interaction_stability.py --dry-run  # 只打印选定 / 计数 / 自检，不画图
```

## 版式约定

背景全白；**面板不画外框**，**也不印面板字母与标题**（字母与标题写进图注）；
面板之间只靠横向留白（第一行 0.52 in、第二行三栏等距 0.38 in）与 0.36 in 纵向留白分隔，
边界仍由 `panel_frames()` 定义 —— 排版自检照旧按它判"字有没有跑出面板"；
口径、判据、颜色语义、代号对照一律写进 `figure4_metrics.json`，图上只保留坐标轴标签、图例、
图底那行 `CODE_KEY` 脚注与数据标注；图上文字一律英文（中文只在源码注释里）；
图例手动排，不依赖自动尺寸；字形只用 Arial 覆盖得到的（`· × Δ ≥ ≤ → − –` 都在；`∝` / `∥` 不在，
用了会渲染成空框）。

**栅格（2 行，行间隙 0.36 in）**：第一行 = a（5.90 in）+ b（9.90 in），第二行 = c（5.10 in）+
d（5.70 in）+ e（4.76 in）——d 只剩 31 × 6 矩阵，右侧腾出的位置给 e 的上下两张柱子图；三栏
等距留白（0.38 in），e 的框与整行右边缘对齐。行高 6.04 / 5.69 in（无标题时从 6.60 / 6.25
各减 0.56，各面板的内容尺寸不变）。图底 0.70 in 的脚注带只放整图唯一一份 `CODE_KEY`（两行）。

**表型维度名与字体字号整幅对齐 Fig.2**（同一篇论文里两张图读起来得是同一套字）：

- 维度名逐字取自 `fig2_visualize.py` 的 `DIM_1LINE`（一行全名，宽槽用）与 `DIM_2L`（Fig.2 自己
  的两行写法，窄槽用）—— 见 `DIM_1LINE` / `DIM_2L` 两个常量，本图不再自造缩写。只有 panel d/e
  的 6 列热条与 panel e 的横轴刻度换成两字代号（一列 0.4 in，全名铺不下）。代号由整图唯一一份
  `CODE_KEY` 对照表定义，印在图底脚注带（两行、每行三个维度；两处的 6 列同序，
  所以一份就够），
  表里写的仍是 Fig.2 的全名。
- **字体字号照 Fig.2 的表，但正文整档放大**：Arial/Helvetica；Fig.2 是「正文一档 + 粗体标题一档」
  （`FS_BODY = 7.2 pt` / `FS_DOSE = 10 pt`，比值 1.39），本图把正文乘 1.25 ——
  `FS = 9.0 pt`（坐标轴标题、图内注记、图例、**刻度**全用它，刻度不再单独设档）。
  面板字母与标题已从图上删除，所以本图只有这一档字；`FS_FIG` / `FS_F_LEGEND` 两档也不出现。
- 放大字号的两条技术注意：① 建图前把 `axes.labelsize / xtick.labelsize / ytick.labelsize /
  legend.fontsize` 一起覆盖成 `FS`（f2 的 rcParams 写的是 7.2 pt，坐标轴标题的位置与刻度预留的槽宽
  都是按它算的）；② `unify_axis_ticks()` 之后要**再 draw 一次**，否则坐标轴标题还停在按 7.2 pt
  刻度算出来的位置上，x 轴标题会压在刻度上（9 pt 之前不会暴露）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    """按显式路径加载同级脚本（与 `fig3_dialogue_evidence.py` 同一套做法）。

    必须先登记进 `sys.modules`：`@dataclass` 在装饰时要回查 `sys.modules[cls.__module__]`。
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


f2 = _load("fig2_visualize", "results/scripts/fig2_visualize.py")

if str(ROOT / "pacs" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "pacs" / "src"))
from pacs.ontology.primitives import PRIMITIVES  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "results" / "figure"
STEM = "figure4_interaction_stability"
METRICS_NAME = "figure4_metrics.json"

PAIR_XLSX = ROOT / "results" / "data" / "result_3_multivariable.xlsx"
UNI_XLSX = ROOT / "results" / "data" / "result_2_univariate.xlsx"

#: 全图唯一的档位：θ = 0.8（primitive 完整度 0.8）。
#: 组合实验（result_3）**只有 0.8 这一档**（465 对 × 3 次重复），所以"单因素 → 组合"只能在 0.8 上
#: 做同档比较：单因素读数取 result_2 summary_by_cell 的 dose == 0.8（31 个 primitive），
#: 逐 replicate 的单因素取 mind_setting 以 " 0.80" 结尾的 93 行。**0.4 档在本图里一次都没用**
#: （0.4 是 Fig.2 的主剂量，本图不复用它，也不把两档混在一起）。
#: 唯一的档位无关量是 t 检验的参照组（result_1 的完整心智读数，没有扰动、因此没有档位）。
PAIR_DOSE = 0.80

PHENO6 = [
    "social_interaction_motivation",
    "social_cognition",
    "social_communication_pragmatics",
    "cognitive_flexibility_adaptation",
    "repetitive_restricted_behavior",
    "novelty_uncertainty_sensory_behavior",
]
CONTROL = "general_cognition"

CODE = {          # 轴上用的两字代号（6 组 × 4 根柱子只有这么宽）
    "social_interaction_motivation": "Mo",
    "social_cognition": "So",
    "social_communication_pragmatics": "Pr",
    "cognitive_flexibility_adaptation": "Fl",
    "repetitive_restricted_behavior": "RR",
    "novelty_uncertainty_sensory_behavior": "Se",
}
#: 维度名与 Fig.2 **逐字一致**：两张表直接从 `fig2_visualize.py` 取，本图不另立缩写。
#: `DIM_1LINE` = 一行全名（宽槽用）；`DIM_2L` = Fig.2 自己的两行写法（窄槽用，
#: Fig.2 的热图列头、网络右端、小条形图用的就是这两张表）。
DIM_1LINE = dict(f2.DIM_1LINE)
DIM_2L = dict(f2.DIM_2L)

#: 两字代号只在「6 列铺不进 2.4 in」这种槽里用（panel d 的矩阵列头、panel e 的横轴刻度），
#: 名字本身仍用 Fig.2 的一行全名：代号不是另一套命名，只是同一批名字的缩写。
#: 不画说明行后，这份对照表就是**图上唯一一段非数据文字**，印在
#: **panel e 框内的最底部**（两行，每行三个维度；panel d / e 的 6 列同序，所以一条就够）。
CODE_KEY = ("column code:  " + "  ·  ".join(f"{CODE[d]} = {DIM_1LINE[d]}" for d in PHENO6[:3])
            + "\n" + "  ·  ".join(f"{CODE[d]} = {DIM_1LINE[d]}" for d in PHENO6[3:]))

# 全图配色语义（与 panel a/d/f 的"交互强度"同号）：
#   红 = 比可加预期更差（superadditive / 协同 / 加重）
#   蓝 = 比可加预期更好（subadditive / 缓冲 / 减轻）
#   紫 = 新表型（neither single factor reached the criterion）
#   灰 = 中性（单因素本身、可加预期、unchanged）
C_SYN = "#D1443F"
C_BUF = "#3D86C6"
C_NEW = "#7B52AB"      # 新表型（单开一色，避免和上面的红/蓝语义打架）
C_ADD = "#ABAAA7"
C_WEAK = "#DCE1E6"
C_STAB3 = "#2E8B57"    # 3/3 同号
C_STAB2 = "#E8A33D"    # 2/3 同号
BAR_C = {"a": "#D2A96A", "b": "#8C9BA8", "add": "#9AA1A9"}   # 两根单因素用中性色，不占用红/蓝

#: panel e 的角色标签用色：与全图红/蓝/紫/灰的语义同号（角色名由 `_pick()` 给定）
ROLE_C = {"synergistic": C_SYN, "buffering": C_BUF}   # panel e 只剩这两个角色（红 = 协同、蓝 = 缓冲）


AGG_TOL = 0.03          # 加重 / 减轻 的方向阈值（见 docstring §新定义 2）
DADD_MAX = 0.70         # panel e 只挑"可加预期仍在量程内"的 pair

CMAP = f2._truncate("RdBu_r", 0.06, 0.94)
CMAP_CNT = f2._truncate("YlOrRd", 0.04, 0.82)
#: 字号**结构与 Fig.2 完全一致**（同一套字、同一套分工），只把整档放大到清晰可读：
#: Fig.2 是「正文一档 + 粗体标题一档」，正文 FS_BODY = 7.2 pt、标题 FS_DOSE = 10 pt（比值 1.39）。
#: 本图把这两档整体乘 1.25 —— 正文 9.0 pt、标题 12.0 pt —— 坐标轴标题、图内注记、图例、
#: **刻度**全部用同一档（刻度不再单独设档，也不再沿用 f2.FS_TICK 的 7.2 pt）。
FS = 9.0                        # 正文：坐标轴标题 / 图内注记 / 图例 / 刻度，全图就这一档
#: 面板字母 / 标题从图上删除，所以不再有 12 pt 的标题档（图注里补字母与标题）。

#: 覆盖 f2 留下的 rcParams 字号（那一份是按 Fig.2 的 7.2 pt 写的）。必须在建图之前设好：
#: 坐标轴标题的位置、刻度标签预留的槽宽都是按这个值算出来的。
plt.rcParams.update({"axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "axes.titlesize": FS})

# ---------------------------------------------------------------------------
# 图幅与面板几何（英寸）。两行：稳定性 / 加重 + 新表型 + 景观 + 代表交互
# ---------------------------------------------------------------------------

#: 图幅：**宽度不动**（17.0 in）—— 字号放大、图幅不放大，字相对图幅就变大，缩到期刊栏宽才真的更清晰。
#: 高度按放大后的字号重排。原来第三行（e 通栏、左图右文）已取消：e 的两张柱子图移到第二行
#: 最右列；面板字母与标题写进图注，图高从 18.83 in 缩到 13.13 in，
#: 底部 0.70 in 只留 `CODE_KEY` 脚注。
FIG_W = 17.0
M_L, M_R, M_T, M_B = 0.34, 0.34, 0.34, 0.70
#: 面板字母与标题不画：标题带 0.56 in 从两行行高里扣掉，
#: 各面板的内容尺寸（子图高、矩阵行高、柱子图高）保持不变。
ROW1_H, ROW2_H, ROW_GAP = 6.04, 5.69, 0.36
R1_W = {"a": 5.90, "b": 9.90}              # 第一行：a + b
R2_W = {"c": 5.10, "d": 5.70, "e": 4.76}   # 第二行：c + d + e（e = 该行最右列）

ROW2_Y = M_B
ROW1_Y = ROW2_Y + ROW2_H + ROW_GAP
FIG_H = ROW1_Y + ROW1_H + M_T              # 0.70 + 5.69 + 0.36 + 6.04 + 0.34 = 13.13

#: `CODE_KEY` 脚注的基线：落在最低一行面板下方的边距带里（顶边仍低于 ROW2_Y）。
CODE_KEY_Y = 0.30

#: 图里没有面板字母 / 标题了：内容距面板顶边只留 0.12 in。
TOP_ONLY = 0.12
#: panel b 的维度名写在小图上方（两行约 0.325 in + 0.03 × 子图高），顶部要专门让出来。
TOP_LABEL = 0.46


def rect(x_in: float, y_in: float, w_in: float, h_in: float) -> list[float]:
    return [x_in / FIG_W, y_in / FIG_H, w_in / FIG_W, h_in / FIG_H]


def col_x(widths: dict[str, float]) -> dict[str, float]:
    """一行里各面板的左边界（英寸）：整行铺满内容宽度，间隙等分。"""
    slack = FIG_W - M_L - M_R - sum(widths.values())
    gap = slack / (len(widths) - 1)
    out, cur = {}, M_L
    for k, w in widths.items():
        out[k] = cur
        cur += w + gap
    return out


# ---------------------------------------------------------------------------
# 数据层
# ---------------------------------------------------------------------------


def _channels(df: pd.DataFrame) -> pd.DataFrame:
    """把 `xxx [中文]` 的列名压成 `xxx`（表头里的中文只作人读的对照）。"""
    return df.rename(columns={c: c.split(" [")[0] for c in df.columns})


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def load() -> dict:
    pair = _channels(pd.read_excel(PAIR_XLSX, sheet_name="interaction_by_pair"))
    sub3 = _channels(pd.read_excel(PAIR_XLSX, sheet_name="by_subject"))
    cell = _channels(pd.read_excel(UNI_XLSX, sheet_name="summary_by_cell"))
    sub2 = _channels(pd.read_excel(UNI_XLSX, sheet_name="by_subject"))

    # 组合表里 mind_setting = "A + B 0.80"；表内列顺序与 mind_setting 顺序可能不一致，
    # 所以所有查表都走「两个 primitive 排序后的键」。
    sub3["pair"] = sub3.mind_setting.str.replace(r"\s+0\.\d+$", "", regex=True)
    sub3["ph6"] = sub3[PHENO6].astype(float).mean(axis=1)
    pair["kkey"] = [_key(str(a), str(b)) for a, b in zip(pair.primitive_a, pair.primitive_b)]
    prow = {r.kkey: r for _, r in pair.iterrows()}

    ids = [p.id for p in PRIMITIVES]
    name = {p.id: p.name_en for p in PRIMITIVES}
    fam = {p.id: p.family for p in PRIMITIVES}
    order = sorted(ids, key=lambda p: (fam[p], f2.EN.get(p, p)))

    # ---- 单因素：逐维度读数 / 6 通道读数 / 逐 replicate 的 6 通道读数 ---------
    u8 = cell[cell.dose == PAIR_DOSE].set_index("primitive_id")
    sdim = {(p, d): float(u8.loc[p, f"{d}_mean"]) for p in ids for d in PHENO6}
    sdrop = {(p, d): -float(u8.loc[p, f"{d}_delta"]) for p in ids for d in PHENO6}
    s6 = {p: float(np.mean([sdim[(p, d)] for d in PHENO6])) for p in ids}

    singles = sub2[(sub2.mind_setting.str.endswith(f" {PAIR_DOSE:.2f}"))
                   & ~sub2.mind_setting.str.startswith("generic")]
    s6r, sdim_r = {}, {}
    for ms, g in singles.groupby("mind_setting"):
        p = ms.rsplit(" ", 1)[0]
        for _, row in g.iterrows():
            rep = int(row.replicate)
            s6r[(p, rep)] = float(np.mean([float(row[d]) for d in PHENO6]))
            for d in PHENO6:
                sdim_r[(p, d, rep)] = float(row[d])
    sgroups = list(singles.groupby("mind_setting"))

    # ---- 反解逐维度基线：interaction_<dim> = combined − (A + B − baseline_d) ---
    # `combo_r` = 逐重复的**6 通道均值**（panel a 的重复内交互效应用它，与表内
    # interaction_phenotype_mean 同一口径）；`combo_dim_r` = 逐重复**逐维度**的读数
    # （panel e 的 error bar 与"某格在 3 次重复里都成立"这类判断必须用逐维度的）。
    combo, combo_r, combo_dim_r = {}, {}, {}
    for k, g in sub3.groupby("pair"):
        kk = _key(*k.split(" + "))
        combo[kk] = {d: float(np.mean([float(v) for v in g[d] if not pd.isna(v)])) for d in PHENO6}
        for _, r in g.iterrows():
            combo_r[(kk, int(r.replicate))] = float(r.ph6)
            for d in PHENO6:
                combo_dim_r[(kk, d, int(r.replicate))] = float(r[d])
    base_d, base_dev = {}, {}
    for d in PHENO6:
        imp = np.array([float(r[f"interaction_{d}"]) - combo[r.kkey][d]
                        + sdim[(str(r.primitive_a), d)] + sdim[(str(r.primitive_b), d)]
                        for _, r in pair.iterrows()])
        base_d[d] = float(np.median(imp))
        base_dev[d] = float(np.ptp(imp))
    base6 = float(np.mean(list(base_d.values())))

    # ---- 单因素是否"已受影响"：逐字沿用 Fig.2 判据（t 检验 + 族内 BH-FDR）----
    ref = f2._intact_reference()
    aff_s, q_s = {}, {}
    for d in PHENO6:
        pv = np.array([f2._t_p([float(v) for v in g[d] if not pd.isna(v)], ref[d], True)
                       for _, g in sgroups])
        qv = f2._bh_fdr(pv)
        for (ms, _), q in zip(sgroups, qv):        # 逐字沿用 Fig.2 的族 = 该档位 31 个 primitive
            p = ms.rsplit(" ", 1)[0]
            q_s[(p, d)] = float(q)
            aff_s[(p, d)] = bool(sdrop[(p, d)] >= f2.TARGET_DROP and q < f2.TEST_ALPHA)

    # ---- 组合是否"受影响"：同一判据，族 = 该维度下的 465 对 ------------------
    kkeys = sorted(combo)
    aff_c, q_c, drop_c, drop_c_rep = {}, {}, {}, {}
    for d in PHENO6:
        pv = np.array([f2._t_p([float(v) for v in sub3[sub3["pair"].map(lambda s: _key(*s.split(" + "))) == kk][d]
                                if not pd.isna(v)], ref[d], True) for kk in kkeys])
        qv = f2._bh_fdr(pv)
        for kk, q in zip(kkeys, qv):
            q_c[(kk, d)] = float(q)
            drop_c[(kk, d)] = base_d[d] - combo[kk][d]
            aff_c[(kk, d)] = bool(drop_c[(kk, d)] >= f2.TARGET_DROP and q < f2.TEST_ALPHA)
            drop_c_rep[(kk, d)] = [base_d[d] - combo_dim_r[(kk, d, r)] for r in (0, 1, 2)]

    # ---- panel a：逐 replicate 的交互强度与方向一致性 ------------------------
    st = []
    for a, b in kkeys:
        reps = [(s6r[(a, r)] + s6r[(b, r)] - base6) - combo_r[((a, b), r)] for r in (0, 1, 2)]
        st.append(dict(a=a, b=b, kkey=(a, b), mean=float(np.mean(reps)),
                       sd=float(np.std(reps, ddof=1)), reps=reps,
                       sign3=len({int(np.sign(v)) for v in reps}) == 1))
    stab = pd.DataFrame(st)
    s3map = dict(zip(stab.kkey, stab.sign3))
    sdmap = dict(zip(stab.kkey, stab.sd))

    # ---- panel b/c：(pair, 维度) 格子 ----------------------------------------
    rows = []
    for kk in kkeys:
        a, b = kk
        for d in PHENO6:
            hit = [p for p in (a, b) if aff_s[(p, d)]]
            ref_single = None if not hit else min(sdim[(p, d)] for p in hit)
            rec = dict(a=a, b=b, kkey=kk, dim=d, n_hit=len(hit), single=ref_single,
                       drop_a=base_d[d] - sdim[(a, d)], drop_b=base_d[d] - sdim[(b, d)],
                       drop_comb=drop_c[(kk, d)], comb=combo[kk][d],
                       drop_single=None if not hit else base_d[d] - ref_single,
                       aff_c=aff_c[(kk, d)], q_c=q_c[(kk, d)], s3=bool(s3map[kk]),
                       sd=float(sdmap[kk]), reps=drop_c_rep[(kk, d)])
            rec["add"] = sdim[(a, d)] + sdim[(b, d)] - base_d[d]
            rec["drop_add"] = base_d[d] - rec["add"]
            rec["one_rep"] = None if not hit else [max(base_d[d] - sdim_r[(p, d, r)] for p in hit)
                                                   for r in (0, 1, 2)]
            rec["change"] = None if not hit else rec["drop_comb"] - rec["drop_single"]
            rec["strength"] = -float(prow[kk][f"interaction_{d}"])
            # error bar 用的逐重复读数：同一 replicate 的单因素 / 组合读数对齐相减，
            # 可加预期逐重复算成 drop_A + drop_B（外推量没有自己的测量值）。
            da_r = [base_d[d] - sdim_r[(a, d, r)] for r in (0, 1, 2)]
            db_r = [base_d[d] - sdim_r[(b, d, r)] for r in (0, 1, 2)]
            for tag, vals in (("a", da_r), ("b", db_r),
                              ("add", [x + y for x, y in zip(da_r, db_r)]),
                              ("comb", drop_c_rep[(kk, d)])):
                rec[f"sd_{tag}"] = float(np.std(vals, ddof=1))
                rec[f"mu_{tag}"] = float(np.mean(vals))
            rows.append(rec)
    it = pd.DataFrame(rows)
    it["pair"] = [f"{a} + {b}" for a, b in it.kkey]

    taken = it[it.n_hit > 0].copy()
    taken["kind"] = np.where(taken.change > AGG_TOL, "worse",
                             np.where(taken.change < -AGG_TOL, "better", "same"))
    new = it[(it.n_hit == 0) & it.aff_c]

    stable = {
        "affected_3of3": float(np.mean([all(v >= f2.TARGET_DROP for v in r)
                                        for r in it[it.aff_c].reps])),
        "new_3of3": float(np.mean([all(v >= f2.TARGET_DROP for v in r) for r in new.reps])),
        "worse_3of3": float(np.mean([all(r.reps[i] > r.one_rep[i] for i in range(3))
                                     for _, r in taken[taken.kind == "worse"].iterrows()])),
    }

    # ---- panel d：primitive × 维度 的"受累伙伴数" ---------------------------
    cnt = {(p, d): 0 for p in ids for d in PHENO6}
    for kk in kkeys:
        for d in PHENO6:
            if aff_c[(kk, d)]:
                cnt[(kk[0], d)] += 1
                cnt[(kk[1], d)] += 1
    newcnt = {p: 0 for p in ids}
    for _, r in new.iterrows():
        newcnt[r.a] += 1
        newcnt[r.b] += 1
    d_order = sorted(ids, key=lambda p: -sum(cnt[(p, d)] for d in PHENO6))
    cnt_mat = np.array([[cnt[(p, d)] for d in PHENO6] for p in d_order], float)

    return dict(ids=ids, order=order, d_order=d_order, name=name, fam=fam, pair=pair, prow=prow,
                items=it, taken=taken, new=new, stab=stab, combo=combo, combo_r=combo_r,
                sdim=sdim, sdrop=sdrop, s6=s6, s6r=s6r, base_d=base_d, base6=base6,
                base_dev=base_dev, aff_s=aff_s, aff_c=aff_c, q_s=q_s, q_c=q_c, drop_c=drop_c,
                sdim_r=sdim_r, combo_dim_r=combo_dim_r,
                stable=stable, cnt_mat=cnt_mat, newcnt=newcnt, e_selected=None)


# ---------------------------------------------------------------------------
# 选定代表 pair（数据驱动）
# ---------------------------------------------------------------------------


def _pick(data: dict) -> None:
    """panel e 的两个代表 pair —— 全部要求「3 次重复同号」：代表例要么稳，要么不配当代表。"""
    it, stab = data["items"], data["stab"]
    s3 = dict(zip(stab.kkey, stab.sign3))
    cand = []
    for kk, g in it.groupby("kkey"):
        if not s3[kk] or len(g) != 6:
            continue
        s = g.set_index("dim").loc[PHENO6]
        hit = s.n_hit > 0
        cand.append(dict(
            kkey=kk, pair=f"{kk[0]} + {kk[1]}",
            worst=float(s.change[hit].max()) if hit.any() else 0.0,
            best=float(s.change[hit].min()) if hit.any() else 0.0,
            nnew=int((~hit & s.aff_c).sum()),
            syn=float(s.strength.max()), buf=float(s.strength.min()),
            dadd_max=float(s.drop_add.max()), sd=float(s.sd.iloc[0])))
    C = pd.DataFrame(cand)
    C["mixed"] = (C.worst > 0.05) & (C.best < -0.20)

    picks, used, used_p = [], set(), set()

    def take(role: str, col: str, mask: pd.Series | None = None, asc: bool = False):
        """在「满足该角色条件、且不与已选 pair 共用 primitive」的池子里取 col 最极端的一对。

        `asc=True` 用于「越负越强」的量（缓冲强度 `buf`、同一对内的最大缓解 `best`）。
        优先不共用 primitive（31 个 primitive 里挑 6 对代表，共用会让人以为是同一件事的重复）；
        池子空时退回全集，此时同一个 primitive 会出现在两个子图里。
        """
        pool = C[(~C.kkey.isin(used)) & (mask if mask is not None else True)]
        fresh = pool[~pool.kkey.map(lambda kk: bool(set(kk) & used_p))]
        row = (fresh if len(fresh) else pool).sort_values(col, ascending=asc).iloc[0]
        picks.append(dict(kkey=tuple(row.kkey), pair=row.pair, role=role, _worst=float(row.worst),
                          _nnew=int(row.nnew), _buf=float(row.buf), _dadd=float(row.dadd_max),
                          _sd=float(row.sd)))
        used.add(tuple(row.kkey))
        used_p.update(row.kkey)

    # 两个代表例：一个**协同**（真实叠加超过可加预期），一个**缓冲**（真实叠加低于可加预期）。
    take("synergistic", "worst",
         (C.worst > 0.15) & (C.syn > 0) & (C.sd < 0.05))     # 协同里加重最明显的那对
    take("buffering", "buf", (C.dadd_max < DADD_MAX) & (C.buf < -0.30) & (C.sd < 0.15), asc=True)
    data["e_selected"] = picks

    # panel b 的取向表（定义见 `_orientations`）：
    #   b_fan[d]   = 该维度上**每个符合条件的 Main cognitive** 各留一条线 —— 只留它最坏的那个 partner
    #   b_picks[d] = 这些线里最显著（变化量最大）的那一条，图上标它的名称
    ori = _orientations(data)
    data["ori"] = ori
    data["b_fan"] = {d: ori[ori.dim == d].loc[ori[ori.dim == d].groupby("main").change.idxmax()]
                     for d in PHENO6}
    data["b_picks"] = {d: fan.loc[fan.change.idxmax()] for d, fan in data["b_fan"].items()}


def _orientations(data: dict) -> pd.DataFrame:
    """把 (pair, 维度) 格子摊成**取向**（orientation）：每个满足受影响判据的单因素都能当 Main。

    panel b 的横轴两端是「Main cognitive（左）」与「Its secondary cognitive（右）」：
    左端 = 这一对里**满足判据**（`Δ ≥ 0.15` 且 `q < 0.05`，即 `aff_s`）的那个单因素，
    右端 = 它 + 另一个因素（实测的 pair）。一个格子里两个单因素都满足判据时，这个格子有**两条
    合法坡线**（A 当主、B 当次；或 B 当主、A 当次），两条都算数 —— 判定口径：**所有符合条件的
    都纳入 Main cognitive**。panel b 报的是变化量更大的那一条（见 docstring 的"最坏一格"口径）。

    变化量 `= Δ(pair) − Δ(Main cognitive)`。参照取两个受影响单因素里 Δ 更大的那个；
    另一个取向对应的变化量更小，因此这里的数字是这条坡线的**下界**。
    """
    rows = []
    for t in data["taken"].itertuples():
        for main_p in (t.a, t.b):
            if not data["aff_s"][(main_p, t.dim)]:
                continue                       # 不满足判据的单因素不能当 Main
            drop_main = float(data["base_d"][t.dim] - data["sdim"][(main_p, t.dim)])
            rows.append(dict(dim=t.dim, pair=t.pair, main=main_p,
                             secondary=t.b if main_p == t.a else t.a,
                             n_hit=int(t.n_hit), drop_main=drop_main,
                             drop_comb=float(t.drop_comb),
                             change=float(t.drop_comb) - drop_main,
                             q=float(t.q_c), sd=float(t.sd), s3=bool(t.s3)))
    return pd.DataFrame(rows)


def _pair_of(data: dict, kk) -> pd.DataFrame:
    return data["items"][data["items"].kkey == kk].set_index("dim").loc[PHENO6]


# ---------------------------------------------------------------------------
# 硬门
# ---------------------------------------------------------------------------


def gate(data: dict) -> tuple[list[str], dict]:
    bad = []
    it, pair = data["items"], data["pair"]
    if len(pair) != 465 or len(data["combo"]) != 465 or len(it) != 465 * 6:
        bad.append(f"规模不对：{len(pair)} 对 / {len(data['combo'])} 对 / {len(it)} 格")
    if not np.isfinite(data["stab"].sd).all():
        bad.append("有 pair 的重复间 SD 不是有限数")

    dev0 = float((pair[[f"interaction_{d}" for d in PHENO6]].mean(axis=1)
                  - pair.interaction_phenotype_mean).abs().max())
    if dev0 > 1e-4:
        bad.append(f"interaction_phenotype_mean 与 6 维均值不符：{dev0:.2e}")

    dev1 = 0.0
    for _, r in pair.iterrows():
        kk = r.kkey
        for d in PHENO6:
            add = data["sdim"][(str(r.primitive_a), d)] + data["sdim"][(str(r.primitive_b), d)] \
                - data["base_d"][d]
            dev1 = max(dev1, abs((data["combo"][kk][d] - add) - float(r[f"interaction_{d}"])))
    if dev1 > 5e-4:
        bad.append(f"逐维度链路重算不符：{dev1:.2e}")

    dev2 = max(abs((data["sdim"][(p, d)] + data["sdrop"][(p, d)]) - data["base_d"][d])
               for p in data["ids"] for d in PHENO6)
    if dev2 > 1e-4:
        bad.append(f"单因素 mean + (−delta) 与反解基线不符：{dev2:.2e}")

    # 重复内交互强度画的是 strength = additive − combined = −interaction_phenotype_mean，
    # 所以这里要反号比对（差一个负号是最容易悄悄发生、又最难在图上发现的错误）。
    dev3 = float(max(abs(-float(r["mean"])
                         - float(data["prow"][r.kkey].interaction_phenotype_mean))
                     for _, r in data["stab"].iterrows()))
    if dev3 > 0.02:
        bad.append(f"重复内交互强度的 3 次均值与表内值不符：{dev3:.3f}")

    # 已知答案：符号约定一旦翻掉，这两个「最」就会换人。
    top_syn = data["stab"].loc[data["stab"]["mean"].idxmax()]
    top_buf = data["stab"].loc[data["stab"]["mean"].idxmin()]
    if top_syn.kkey != ("prior_strength", "relational_representation"):
        bad.append(f"最强协同换人了：{top_syn.kkey} {top_syn['mean']:+.4f}")
    if top_buf.kkey != ("representation_updating", "rule_updating"):
        bad.append(f"最强缓冲换人了：{top_buf.kkey} {top_buf['mean']:+.4f}")

    colsum = data["cnt_mat"].sum(axis=0)
    for i, d in enumerate(PHENO6):
        want = 2 * int(sum(1 for kk in data["combo"] if data["aff_c"][(kk, d)]))
        if int(colsum[i]) != want:
            bad.append(f"panel d 矩阵列和 {d}: {int(colsum[i])} != {want}")

    # panel b 的扇面口径（所有满足判据的单因素都能当 Main cognitive）
    ori = data["ori"]
    for d, fan in data["b_fan"].items():
        for _, g in fan.iterrows():
            if not data["aff_s"][(g.main, d)]:
                bad.append(f"panel b 画的 Main 不满足判据：{d} / {g.main}")
            if g.main == g.secondary:
                bad.append(f"panel b 的取向两端是同一个因素：{d} / {g.main}")
            if abs(float(g.change) - (float(g.drop_comb) - float(g.drop_main))) > 1e-9:
                bad.append(f"panel b 取向的变化量与两端不符：{d} / {g.main}")
        # 扇面必须覆盖该维度**每一个**符合条件的 Main cognitive，且每个 Main 用自己最坏的那条
        want_mains = {p for p in set(fan.main) | set(fan.secondary) if data["aff_s"][(p, d)]}
        if set(fan.main) != want_mains:
            bad.append(f"panel b 的 {d} 扇面缺 Main cognitive：{sorted(want_mains - set(fan.main))}")
        w = ori[ori.dim == d].groupby("main").change.max()
        gp = fan.set_index("main").change
        if float((w.reindex(gp.index) - gp).abs().max()) > 1e-9:
            bad.append(f"panel b 的 {d} 有 Main cognitive 没取自己最坏的那条")
    for d, r in data["b_picks"].items():
        if abs(float(r.change) - float(data["b_fan"][d].change.max())) > 1e-9:
            bad.append(f"panel b 的 {d} 标名的那条不是扇面里最大的")
        if float(r.change) < float(ori[ori.dim == d].change.max()) - 1e-9:
            bad.append(f"panel b 的 {d} 没取到变化量最大的取向")
    dev5 = float(max((abs(float(r.change) - (float(r.drop_comb) - float(r.drop_main)))
                      for r in data["b_picks"].values()), default=0.0))
    info_drawn = int(sum(len(f) for f in data["b_fan"].values()))   # 扇面画出来的线数

    s3 = dict(zip(data["stab"].kkey, data["stab"].sign3))
    e_keys = [q["kkey"] for q in data["e_selected"]]
    if len(set(e_keys)) != 2:
        bad.append(f"panel e 没选出 2 对互不相同的组合：{e_keys}")
    if [q["role"] for q in data["e_selected"]] != ["synergistic", "buffering"]:
        bad.append("panel e 的两个例子不是「一个协同 + 一个缓冲」")
    for kk in e_keys:
        if not s3[kk]:
            bad.append(f"代表 pair {kk} 不是 3 次重复同号")
        if len(_pair_of(data, kk)) != 6:
            bad.append(f"代表 pair {kk} 维度不全")
    # 右列正文是照数据现写的，这里再独立验一遍它写出去的那两条断言（改了选法就会当场发现）
    s_syn, s_buf = _pair_of(data, e_keys[0]), _pair_of(data, e_keys[1])
    if float(s_syn.strength.max()) <= 0:
        bad.append("panel e 的协同例没有任何一维超过可加预期")
    if not any(int(s_syn.loc[d, "n_hit"]) == 0 and bool(s_syn.loc[d, "aff_c"]) for d in PHENO6):
        bad.append("panel e 的协同例没有新表型格子，正文那句「neither single factor affects at all」写不出来")
    if int((s_buf.drop_comb < s_buf.drop_add).sum()) != len(PHENO6):
        bad.append("panel e 的缓冲例不是 6 维全部低于可加预期，正文那句「in all 6 dimensions」写不出来")
    if float(s_buf.drop_comb.max()) > f2.TARGET_DROP:
        bad.append("panel e 的缓冲例还有维度处在受损水平，正文那句「near the intact baseline」写不出来")
    # panel e 画上去的数值必须**逐维等于表里那一列**（provenance：不是脚本自己另算的一套）
    dev4 = 0.0
    for kk in e_keys:
        s = _pair_of(data, kk)
        for d in PHENO6:
            dev4 = max(dev4, abs(float(s.loc[d, "strength"])
                                 + float(data["prow"][kk][f"interaction_{d}"])))
    if dev4 > 1e-9:
        bad.append(f"panel e 的数值与表内 interaction_<dim> 不一致：{dev4:.2e}")
    # error bar 与柱子均值都必须能由**逐重复读数**独立重算出来（均值等于柱子、SD 等于误差棒）。
    # 容差 5e-4 而不是 1e-9：单因素汇总表的 {d}_mean 是四舍五入过的，与 by_subject 三行均值
    # 最多差 3.3e-05（可加预期要把 A、B 两个偏差相加，所以这里到 6.7e-05）；组合表完全一致。
    dev5e = 0.0
    for kk in e_keys:
        a, b = kk
        s = _pair_of(data, kk)
        for d in PHENO6:
            base = data["base_d"][d]
            da = [base - data["sdim_r"][(a, d, r)] for r in (0, 1, 2)]
            db = [base - data["sdim_r"][(b, d, r)] for r in (0, 1, 2)]
            for tag, vals in (("a", da), ("b", db), ("add", [x + y for x, y in zip(da, db)]),
                              ("comb", [base - data["combo_dim_r"][(kk, d, r)]
                                        for r in (0, 1, 2)])):
                dev5e = max(dev5e, abs(float(np.mean(vals)) - float(s.loc[d, f"drop_{tag}"])),
                            abs(float(np.std(vals, ddof=1)) - float(s.loc[d, f"sd_{tag}"])))
    if dev5e > 5e-4:
        bad.append(f"panel e 的 error bar / 柱子均值与逐重复读数不符：{dev5e:.2e}")

    info = dict(six_dim_mean_dev=dev0, chain_dev=dev1, single_base_dev=dev2, mean_chain_dev=dev3,
                ef_provenance_dev=dev4, e_err_dev=dev5e, orientation_dev=dev5,
                panel_b_slopes_drawn=info_drawn,
                strongest_synergy=[top_syn.kkey, round(float(top_syn["mean"]), 4)],
                strongest_buffering=[top_buf.kkey, round(float(top_buf["mean"]), 4)],
                base_d={k: round(v, 4) for k, v in data["base_d"].items()},
                base_d_spread={k: round(v, 4) for k, v in data["base_dev"].items()},
                colsum=[int(v) for v in colsum])
    return bad, info


# ---------------------------------------------------------------------------
# 绘图小工具
# ---------------------------------------------------------------------------


def thin_axis(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        if s in keep:
            ax.spines[s].set_linewidth(0.8)
            ax.spines[s].set_color(f2.INK)
        else:
            ax.spines[s].set_visible(False)


def unify_axis_ticks(fig):
    """全图刻度统一为 `FS` 加粗 —— 与 `f2.unify_axis_ticks` 同一条规矩（刻度不单独设档），
    但字号跟着本图放大后的正文档走（f2 那一版写死 7.2 pt，会把刻度拉回小一号）。"""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for lab in axis.get_ticklabels():
                lab.set_fontsize(FS)
                lab.set_fontweight("bold")


def plain_ax(fig, rect_):
    ax = fig.add_axes(rect_)
    ax.set_facecolor("white")
    return ax


def check_text_clashes(fig) -> list[tuple[str, str]]:
    """排版自检：与 `fig2_visualize.check_text_overlaps` 同一套判据，但**只量文字、不量引线**。

    `Annotation.get_window_extent()` 返回「文字 ∪ 引线」的并集，量引线扫过的文字会误报。
    这里对 Annotation 取底类 `Text` 的包围盒，只留文字本身。其余规则（同一条轴上的相邻刻度豁免、
    重叠 > 2 px 才报）逐字照抄。
    """
    renderer = fig.canvas.get_renderer()
    boxes: list[tuple[str, Bbox, object]] = []

    def add(t, axis=None, dense=False):
        if not t.get_text().strip():
            return
        bb = (Text.get_window_extent(t, renderer) if isinstance(t, Annotation)
              else t.get_window_extent(renderer))
        boxes.append((t.get_text().replace("\n", " ")[:46], bb, axis if dense else None))

    for t in fig.texts:
        add(t)
    for leg in fig.legends:
        for t in list(leg.get_texts()) + [leg.get_title()]:
            add(t)
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts):
            add(t)
        if not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            labs = list(axis.get_ticklabels())
            # 同轴相邻刻度的豁免只给"密集数值轴"（>= 6 个刻度）；只有两根刻度时互压是真问题
            for t in labs:
                add(t, axis, dense=len(labs) >= 6)
    clashes = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][2] is not None and boxes[i][2] is boxes[j][2]:
                continue
            hit = Bbox.intersection(boxes[i][1], boxes[j][1])
            if hit is not None and hit.width > 2.0 and hit.height > 2.0:
                clashes.append((boxes[i][0], boxes[j][0]))
    return clashes


def panel_frames() -> list[tuple[float, float, float, float]]:
    """五个面板的框（英寸）：x, y, w, h —— y 从底边算：第一行 a+b，第二行 c+d+e。"""
    out = [(col_x(R1_W)[k], ROW1_Y, R1_W[k], ROW1_H) for k in R1_W]
    out += [(col_x(R2_W)[k], ROW2_Y, R2_W[k], ROW2_H) for k in R2_W]
    return out


def check_text_covered(fig) -> list[tuple[str, str, float]]:
    """排版自检：文字有没有被**别的坐标轴的白底**压住。

    `check_text_clashes` 只比文字与文字，`check_inside_frames` 只看文字与面板框 —— 两者都查不出
    「上面那张图的 x 轴标题被下面那张图的白底盖掉一半」：那是一个 zorder 更高的 axes patch 压在
    文字上。panel c 的上下两张子图要特别注意（上面那张的 x 轴标题容易被盖掉半截）。

    判据：对每个**不透明底纹**的 axes，收集所有**不属于它**的文字（其它 axes 的标题 / 刻度 /
    注记，以及 figure 级文字），凡是包围盒与它相交、且相交面积超过该文字面积的 25%，就算被压。
    25% 是留量：刻度标签贴着坐标轴边缘、偶尔压过 1~2 px 是正常排版，整行字被盖掉一半才是缺陷。
    """
    r = fig.canvas.get_renderer()
    texts: list[tuple[str, Bbox, object]] = []

    def own(t):
        for ax in fig.axes:
            if any(t is x for x in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts)) \
                    or any(t is x for x in (ax.get_xticklabels() + ax.get_yticklabels())):
                return ax
        return None

    for t in fig.texts:
        if t.get_text().strip():
            texts.append((t.get_text().replace("\n", " ")[:40],
                          t.get_window_extent(r), None))
    for ax in fig.axes:
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title, *ax.texts,
                  *ax.get_xticklabels(), *ax.get_yticklabels()):
            if t.get_text().strip():
                texts.append((t.get_text().replace("\n", " ")[:40], t.get_window_extent(r), ax))

    bad = []
    for ax in fig.axes:
        fc = ax.get_facecolor()
        if not ax.axison or len(fc) == 4 and fc[3] == 0:
            continue                                   # 透明底纹的 axes 不遮字
        for label, bb, owner in texts:
            if owner is ax:
                continue
            inter = Bbox.intersection(bb, ax.get_window_extent(r))
            if inter is None:
                continue
            covered = (inter.width * inter.height) / max(bb.width * bb.height, 1e-9)
            if covered > 0.25:
                bad.append((label, f"axes@{round(ax.get_position().x0 * FIG_W, 2)},"
                                   f"{round(ax.get_position().y0 * FIG_H, 2)}",
                            round(covered, 2)))
    return bad


def check_inside_frames(fig) -> list[tuple[str, float, float]]:
    """排版自检：figure 级文字必须落在某个面板框内。

    `check_text_clashes` 只查互相压字，查不出「文字跑出面板框」——panel d 的 x 轴标题与
    panel c 的上图 x 轴标题都写到过框外。这里按包围盒判，容差 0.02 in。
    唯一的例外是图底脚注带：`CODE_KEY` 整段在最低一行面板的下方（y 顶边 ≤ ROW2_Y），
    它是图注式的 footnote，不属于任何面板。
    """
    r = fig.canvas.get_renderer()
    bad = []
    texts = list(fig.texts)
    for ax in fig.axes:                       # 坐标轴标题不在 fig.texts 里，漏过一次
        texts += [ax.xaxis.label, ax.yaxis.label]
    for t in texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r).transformed(fig.transFigure.inverted())
        if bb.y1 * FIG_H <= ROW2_Y + 0.02:
            continue                          # 图底脚注（CODE_KEY），不要求落在面板内
        # 容差必须**按英寸**加：以前写成 `bb.x1 <= (x + w) / FIG_W + 0.02`，等于给了
        # 0.02 × 17 in = 0.34 in 的余量（比想给的宽 17 倍）—— panel c 的 x 轴标题越出框 0.18 in
        # 也一直没报出来。改成先换算成英寸再比。
        ok = any(bb.x0 * FIG_W >= x - 0.02 and bb.x1 * FIG_W <= x + w + 0.02
                 and bb.y0 * FIG_H >= y - 0.02 and bb.y1 * FIG_H <= y + h + 0.02
                 for x, y, w, h in panel_frames())
        if not ok:
            bad.append((t.get_text().replace("\n", " ")[:44], round(bb.x1 * FIG_W, 2),
                        round(bb.y0 * FIG_H, 2)))
    return bad


def _count_color(v: float, vmax: float) -> str:
    return "white" if v > 0.62 * vmax else f2.INK


def _sign_c(diff: float, tol: float = 0.02) -> str:
    """按「真实组合相对可加预期」的方向取色 —— 与 panel a/d/f 的红/蓝语义严格同号。"""
    return C_SYN if diff > tol else (C_BUF if diff < -tol else C_ADD)


# ---------------------------------------------------------------------------
# Panel a · Stability across the three replicates
# ---------------------------------------------------------------------------


def _kde(vals, grid: np.ndarray, n_tot: int, bw: float | None = None) -> np.ndarray:
    """高斯核密度（Silverman 带宽），按 n_tot 归一化 —— 于是各组的曲线可以**直接堆叠**。

    堆叠要求每一组的面积 = 该组占比，所以分母用总样本数而不是组内样本数（与 Fig.2 的
    交互效应分布图同一套带宽规则）。
    """
    v = np.asarray(vals, float)
    if bw is None:
        sd = float(np.std(v, ddof=1))
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        bw = 0.9 * min(sd, iqr / 1.34) * len(v) ** (-0.2) if len(v) > 1 else 0.02
    bw = max(bw, 1e-3)
    z = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (n_tot * bw * np.sqrt(2 * np.pi))


def panel_a(fig, data: dict):
    x0, y0 = col_x(R1_W)["a"], ROW1_Y
    lab, w = 0.62, R1_W["a"] - 0.62 - 0.10
    stab = data["stab"]
    s3 = stab.sign3.to_numpy()
    mean, sd = stab["mean"].to_numpy(), stab.sd.to_numpy()
    # 两个子图：上 = 测量 SD 的分布，下 = 同号率随 |效应| 的变化。原来下排是并排两张，
    # 右边那张（三类结论的 3/3 成立率）不画；面板标题与说明行也不画，
    # 省下来的高度全部给了上面那张分布图。
    bot_h, bot_y, gap = 1.93, 0.90, 0.72     # 9 pt 下：3 行刻度（0.51 in）+ x 轴标题要 0.90 in
    sc_h = ROW1_H - TOP_ONLY - bot_h - gap - bot_y

    # ---- 上：测量 SD 的分布密度（按 3/3、2/3 两组堆叠） ----------------------
    ax = plain_ax(fig, rect(x0 + lab, y0 + bot_y + bot_h + gap, w, sc_h))
    med = float(np.median(sd))
    grid = np.linspace(0.0, 0.42, 320)
    d3 = _kde(sd[s3], grid, len(sd))
    d2 = _kde(sd[~s3], grid, len(sd))
    ax.fill_between(grid, 0, d3, color=C_STAB3, alpha=0.85, linewidth=0, zorder=3)
    ax.fill_between(grid, d3, d3 + d2, color=C_STAB2, alpha=0.85, linewidth=0, zorder=3)
    ax.plot(grid, d3 + d2, color=f2.INK, lw=0.8, zorder=4)
    ax.axvline(med, color=f2.INK, lw=0.8, ls=(0, (4, 2.5)), zorder=5)
    ax.text(med + 0.012, float((d3 + d2).max()) * 0.96, f"median SD = {med:.3f}", fontsize=FS,
            ha="left", va="top", color=f2.INK)
    ax.set_xlim(0, 0.42)
    ax.set_ylim(0, float((d3 + d2).max()) * 1.22)
    ax.set_xlabel("SD of the interaction effect across the 3 replicates", labelpad=2)
    ax.set_ylabel("density", labelpad=3)
    ax.set_yticks([])
    thin_axis(ax)
    ax.legend(handles=[Patch(facecolor=C_STAB3, edgecolor="none",
                             label=f"3/3 same direction  ({int(s3.sum())})"),
                       Patch(facecolor=C_STAB2, edgecolor="none",
                             label=f"2/3 same direction  ({int((~s3).sum())})")],
              loc="upper right", fontsize=FS, frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.3)

    # ---- 下：同号率 vs |强度|（铺满整个面板宽度） -----------------
    ax2 = plain_ax(fig, rect(x0 + lab, y0 + bot_y, w, bot_h))
    edges = [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 1)]
    rate, ns = [], []
    for lo, hi in edges:
        m = (np.abs(mean) >= lo) & (np.abs(mean) < hi)
        ns.append(int(m.sum()))
        rate.append(100 * float(s3[m].mean()) if m.sum() else 0.0)
    xs = np.arange(len(edges))
    ax2.bar(xs, rate, width=0.70, color="#9AA1A9", edgecolor=f2.INK, linewidth=0.5, zorder=3)
    ax2.axhline(100 * float(s3.mean()), color=f2.INK, lw=0.8, ls=(0, (4, 2.5)), zorder=2)
    for x, (r, n) in enumerate(zip(rate, ns)):
        ax2.text(x, r + 5, f"{r:.0f}%", fontsize=FS, ha="center", va="bottom", color=f2.INK)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{lo:g}–\n{hi:g}\nn={n}" if hi < 1 else f">\n{lo:g}\nn={n}"
                         for (lo, hi), n in zip(edges, ns)], linespacing=1.35)
    ax2.set_xlim(-0.6, len(edges) - 0.4)
    ax2.set_ylim(0, 126)
    ax2.set_yticks([0, 50, 100])
    ax2.set_xlabel("| interaction effect |", labelpad=1.5)
    ax2.set_ylabel("3/3 same direction (%)", labelpad=3)
    thin_axis(ax2)


# ---------------------------------------------------------------------------
# Panel b · Is the phenotype already present made worse?
# ---------------------------------------------------------------------------


def _printed_pair(drop_main: float, drop_comb: float) -> tuple[float, float, float]:
    """图上印的取整规矩：两个端点各印两位小数，变化量按**印出来的两个数相减**。

    否则 "0.26 → 0.49" 配未取整的 +0.24 会被读成算错（0.2573 / 0.4940 的真实差是 0.2367）。
    """
    ds, dc = round(float(drop_main), 2), round(float(drop_comb), 2)
    return ds, dc, round(dc - ds, 2)


def panel_b(fig, data: dict):
    x0, y0 = col_x(R1_W)["b"], ROW1_Y
    # 每个维度一张扇面：**每个符合条件的 Main cognitive 一条线**（左端 = 它的 Δ，右端 = 它 +
    # secondary partner 的 Δ），每个 Main cognitive 只留它最坏的那个 partner（变化量最大）。
    # 线色 = 该线自己的方向（红 = 更差、蓝 = 更好、灰 = 持平）；最显著的那一条加粗并标名称。
    fans, picks = data["b_fan"], data["b_picks"]

    lab, top = 0.62, TOP_LABEL  # 顶部只给小图上方的两行维度名留位置
    cw = (R1_W["b"] - lab - 0.14 - 2 * 0.26) / 3
    # y 轴上限罩住扇面里**所有**画出来的线（不只最坏那条），否则细线会被裁掉
    ymax = max(1.10 * float(pd.concat(list(fans.values()))[["drop_main", "drop_comb"]]
                             .to_numpy().max()), 0.1)
    bot = 0.50                  # 下排 x 刻度（两行 0.326 in）的落脚处
    ROW_GAP_B = 1.00            # 上排刻度（0.33 in）与下排头顶的维度名（0.36 in）都要塞进这条缝
    ch = (ROW1_H - top - bot - ROW_GAP_B) / 2
    ytop = y0 + ROW1_H - top - ch
    NAME = data["name"]
    for i, d in enumerate(PHENO6):
        col, row = i % 3, i // 3
        ax = plain_ax(fig, rect(x0 + lab + col * (cw + 0.26), ytop - row * (ch + ROW_GAP_B), cw, ch))
        worst = picks[d]
        # 扇面：每个符合条件的 Main cognitive 一条线，每个只留它最坏的那个 partner。
        # **除最显著的那条以外一律灰**：背景只回答"有哪些 Main、各自被推到哪"，
        # 方向与幅度由那条加粗的彩色线承担，颜色语义因此在图上只有一处。
        # 两端各点一个小圆点，读者一眼能看出左侧是哪些 Main、右侧被推到哪。
        # **除最显著那条以外一律灰**：颜色语义在每张小图里只有一处，不会有第二条彩色线
        # 让人误以为也是"重点"；灰色背景只回答"有哪些 Main、各自被推到哪"。
        fan = fans[d].sort_values("drop_main", ascending=False)
        for _, g in fan.iterrows():
            ax.plot([0, 1], [g.drop_main, g.drop_comb], color=C_ADD, lw=1.3, alpha=0.9, zorder=3,
                    marker="o", markersize=2.4, markeredgewidth=0, solid_capstyle="round")
        chg_w = float(worst.change)
        agg = chg_w > AGG_TOL
        rel = chg_w < -AGG_TOL
        c_w = C_SYN if agg else (C_BUF if rel else C_ADD)      # 最坏那条线的颜色 = 方向
        ctag = C_SYN if agg else (C_BUF if rel else f2.INK)    # 方向词的顏色（灰字全图禁用）
        ax.plot([0, 1], [worst.drop_main, worst.drop_comb], color=c_w, lw=2.6, alpha=1.0,
                zorder=5, solid_capstyle="round")
        ax.scatter([0, 1], [worst.drop_main, worst.drop_comb], s=22, color=c_w, zorder=6,
                   linewidths=0)
        ax.set_xlim(-0.06, 1.06)        # 紧贴两个端点：坡线两边不再留空白，横轴标签也放得开
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Main\ncognitive", "Its secondary\ncognitive"], linespacing=1.30)
        tls = ax.get_xticklabels()
        tls[0].set_horizontalalignment("left")
        tls[-1].set_horizontalalignment("right")
        if col == 0:
            ax.set_yticks([0, 0.2, 0.4, 0.6])
            ax.set_ylabel("Δ impairment", labelpad=2)
        else:
            ax.set_yticklabels([])
        thin_axis(ax)
        # 维度名写在小图上方（Fig.2 的两行写法）：压在坐标轴里会盖住坡线
        ax.text(0.0, 1.03, DIM_2L[d], transform=ax.transAxes, fontsize=FS, fontweight="bold",
                color=f2.INK, ha="left", va="bottom", linespacing=1.30)
        # 不标 "Δ x → y"：最显著那条线的两个端点由坐标轴直接读出，
        # 下方只留"谁 + 加谁 + 变化量"。
        ds, dc, num = _printed_pair(worst.drop_main, worst.drop_comb)
        # 右下：**只标最显著的那一对**（最坏那条坡线的 Main + its secondary + 方向与变化量）。
        # 名字与方向词分两个 text 画（一个 text 只能一个颜色），名字压在上面、方向词贴底边；
        # 三行排而不是挤成两行，是为了每一行都不超过小图宽度 —— 旧的"名字 + 数字"一行会横向
        # 越出坐标轴、压到左边的 y 轴脊线上。背景那 3~13 条线不标名字。
        lh = (FS * 1.30 / 72.0) / ch        # 一行字占多少 axes 高度
        ax.text(0.98, 0.04 + lh, f"{NAME[worst.main]}\n+ {NAME[worst.secondary]}",
                transform=ax.transAxes, fontsize=FS, ha="right", va="bottom", linespacing=1.30,
                fontweight="bold", color=f2.INK, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
        ax.text(0.98, 0.04, (f"aggravated {num:+.2f}" if agg else
                             f"relieved {num:+.2f}" if rel else
                             f"unchanged {num:+.2f}"),
                transform=ax.transAxes, fontsize=FS, ha="right", va="bottom",
                fontweight="bold", color=ctag, zorder=6.1)


# ---------------------------------------------------------------------------
# Panel c · Can a pair create a phenotype neither single produced?
# ---------------------------------------------------------------------------


def panel_c(fig, data: dict):
    x0, y0 = col_x(R2_W)["c"], ROW2_Y
    it, new = data["items"], data["new"]

    # 每维的 pair 数：**该维度上没有任何单个 cognitive 显著、而 pair 显著**
    # （Δ ≥ 0.15 且 q < 0.05）的对数。一个 (pair, 维度) 格 = 一对，所以这就是格子数。
    rows = []
    for d in PHENO6:
        el = int(((it.dim == d) & (it.n_hit > 0)).sum())
        em = int(((it.dim == d) & (it.n_hit == 0) & it.aff_c).sum())
        rows.append(dict(dim=d, elig=465 - el, pairs=em, rate=100 * em / max(465 - el, 1)))
    R = pd.DataFrame(rows).sort_values("pairs", ascending=False)

    lab, top = 1.80, TOP_ONLY
    # 6.25 in 高：0.68 标题 + 2.32 上图 + 0.47 缝 + 2.32 下图 + 0.46 底边。
    # 两个图都是 6 行两行维度名（0.326 in），行距 2.32/6.2 = 0.374 in 够；
    # 缝要放下上图 x 刻度 + x 轴标题（合计 0.41 in），底边同理。
    b1_h = 2.32
    xmax = float(R.pairs.max()) * 1.30
    ax = plain_ax(fig, rect(x0 + lab, y0 + ROW2_H - top - b1_h, R2_W["c"] - lab - 0.12, b1_h))
    ys = np.arange(len(R))[::-1]
    ax.barh(ys, R.pairs, height=0.62, color=C_NEW, edgecolor=f2.INK, linewidth=0.5, zorder=3)
    for y, r in zip(ys, R.itertuples()):
        inside = r.pairs > 0.45 * xmax
        ax.text(r.pairs - 0.35 if inside else r.pairs + 0.35, y, f"{r.pairs}",
                fontsize=FS, ha="right" if inside else "left", va="center",
                color="white" if inside else f2.INK, fontweight="bold" if inside else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([DIM_2L[d] for d in R.dim], fontsize=FS, linespacing=1.30)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(R) - 0.4)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("the pair is significant, no single is", labelpad=1.5)
    thin_axis(ax, keep=("bottom",))

    # ---- 下图：每个**新涌现表型**（维度）里最显著的那一对 ------------------------
    # y 轴 = 维度名（Fig.2 的两行写法），条子 = 该维度上 Δ 最大的新表型格子，
    # 条子右边写这一对的名称；一对没有新表型的维度不出现（本数据六维都有）。
    best = (new.sort_values("drop_comb", ascending=False).groupby("dim", sort=False).head(1)
            .set_index("dim").reindex(PHENO6).dropna(subset=["drop_comb"], how="any")
            .sort_values("drop_comb", ascending=False))
    XLIM = max(float(best.drop_comb.max()) * 1.95, 0.55)
    # 底边 0.46 in 要同时放下 x 刻度（0.17）和 x 轴标题（0.24），否则标题会越出面板框底
    ax2 = plain_ax(fig, rect(x0 + lab, y0 + 0.46, R2_W["c"] - lab - 0.12, 2.32))
    ys = np.arange(len(best))[::-1]
    ax2.barh(ys, best.drop_comb, height=0.62, color=C_NEW, edgecolor=f2.INK, linewidth=0.5)
    for y, r in zip(ys, best.itertuples()):
        ax2.text(r.drop_comb - 0.02 * XLIM, y, f"Δ {r.drop_comb:.2f}", fontsize=FS,
                 ha="right", va="center", color="white", fontweight="bold")
        ax2.text(r.drop_comb + 0.02 * XLIM, y, f"{data['name'][r.a]}\n+ {data['name'][r.b]}",
                 fontsize=FS, ha="left", va="center", color=f2.INK, linespacing=1.15)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([DIM_2L[d] for d in best.index], fontsize=FS, linespacing=1.30)
    ax2.set_xlim(0, XLIM)
    ax2.set_ylim(-0.6, len(best) - 0.4)
    ax2.set_xticks([0, 0.2, 0.4])
    ax2.set_xlabel("Δ impairment of the new phenotype", labelpad=1.5)
    thin_axis(ax2, keep=("bottom",))


# ---------------------------------------------------------------------------
# Panel d · Interaction landscape: which primitive spreads the furthest
# ---------------------------------------------------------------------------


def panel_d(fig, data: dict):
    x0, y0 = col_x(R2_W)["d"], ROW2_Y
    order, mat = data["d_order"], data["cnt_mat"]

    # 第二行 5.69 in 高：31 行 × 0.160 = 4.96，剩下的给 x 轴标题。
    # 左槽 2.70 in 是 9 pt 下最长那个 primitive 名（"information-seeking under uncertainty"）要的宽度。
    # 本图不画右侧两条横条（行和 / 新表型格子数），这一栏只剩矩阵。
    lab, top, rowh = 2.70, TOP_ONLY, 0.160
    matw = R2_W["d"] - lab - 0.10
    mtop = y0 + ROW2_H - top - rowh * 31
    axm = plain_ax(fig, rect(x0 + lab, mtop, matw, rowh * 31))
    vmax = float(mat.max())
    axm.imshow(mat, cmap=CMAP_CNT, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(31):
        for j in range(6):
            axm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white",
                                    linewidth=0.35))
            axm.text(j, i, f"{int(mat[i, j])}", fontsize=FS, ha="center", va="center",
                     color=_count_color(mat[i, j], vmax))
    axm.set_xticks(np.arange(6))
    axm.set_xticklabels([CODE[d] for d in PHENO6])
    axm.set_yticks(np.arange(31))
    axm.set_yticklabels([data["name"][p] for p in order], fontsize=FS)
    axm.set_xlim(-0.5, 5.5)
    axm.set_ylim(30.5, -0.5)
    axm.set_xlabel("phenotype dimension", labelpad=3)
    thin_axis(axm, keep=())
    axm.tick_params(length=0, pad=2)


# ---------------------------------------------------------------------------
# Panel e · Representative interactions: single, additive, real
# ---------------------------------------------------------------------------


#: panel e 的图例块（化简版）：两行三列，右对齐压在 panel 右上角；宽 4.02 in 与柱子图同宽。
E_LEG_W, E_LEG_H = 4.02, 0.38
E_LEG_DY = 0.12        # 图例顶边距面板顶边（面板标题已删，图例直接顶到上边）
E_HEAD_GAP = 0.14      # 图例底边与第一张图「角色标签」之间的缝
#: panel e 两张柱子图的垂直预算（in）：角色标签 0.36 + x 刻度 0.20 + 两张图之间 0.26。
E_HEAD_H, E_TICK_H, E_GAP = 0.36, 0.20, 0.26


def _e_num(x: float, nd: int = 2) -> str:
    """panel e 正文里的数字：小数位固定，负号用 U+2212（与刻度、全图其他面板一致）。"""
    return f"{x:.{nd}f}".replace("-", "\u2212")


def _e_legend(fig, x_right: float, y_top: float):
    """panel e 化简后的图例：两行三列（四根柱子 + 轴下方两个记号），右对齐到 x_right。

    SD 的含义交给图注（误差棒是通用记号），于是七格缩成六格、一行变两行，压进 panel 右上角。
    """
    axl = plain_ax(fig, rect(x_right - E_LEG_W, y_top - E_LEG_H, E_LEG_W, E_LEG_H))
    axl.set_xlim(0, E_LEG_W)
    axl.set_ylim(0, E_LEG_H)
    axl.set_axis_off()
    cols = (0.0, 1.42, 2.72)          # 三列的左边界（in）
    rows = (0.215, 0.035)             # 两行的底边（in）
    sw, sh = 0.15, 0.09               # 色块尺寸（in）

    def label(x, y, s):
        axl.text(x, y + sh / 2, s, fontsize=FS, ha="left", va="center", color=f2.INK)

    for x, c, lb, hatch in ((cols[0], BAR_C["a"], "single A", None),
                            (cols[1], BAR_C["b"], "single B", None),
                            (cols[2], "white", "additive A+B", "////")):
        axl.add_patch(Rectangle((x, rows[0]), sw, sh, facecolor=c,
                                edgecolor=C_ADD if hatch else f2.INK,
                                linewidth=0.5, hatch=hatch))
        label(x + sw + 0.06, rows[0], lb)
    # 真实叠加按方向着色：左半蓝（比可加轻）、右半红（比可加重）
    x = cols[0]
    axl.add_patch(Rectangle((x, rows[1]), sw / 2, sh, facecolor=C_BUF,
                            edgecolor=f2.INK, linewidth=0.4))
    axl.add_patch(Rectangle((x + sw / 2, rows[1]), sw / 2, sh, facecolor=C_SYN,
                            edgecolor=f2.INK, linewidth=0.4))
    label(x + sw + 0.06, rows[1], "real combination")
    axl.plot([cols[1] + sw / 2], [rows[1] + sh / 2], marker="o", ms=3.0, color=f2.INK)
    label(cols[1] + sw + 0.06, rows[1], "affected")
    axl.plot([cols[2] + sw / 2], [rows[1] + sh / 2], marker="*", ms=6.0, color=C_SYN)
    label(cols[2] + sw + 0.06, rows[1], "new phenotype")


def _e_case_texts(data: dict, q: dict) -> tuple[str, str, str]:
    """panel e 右列的三段文字：标题行 / Observations / 可能原因。

    图里出现的每个数字都在这里**现算**（与柱子同一列 `drop_a / drop_b / drop_add / drop_comb`），
    不写死，所以正文与柱子不可能对不上（硬门 7 另外再验两条"正文里的断言"）。
    """
    kk, role = q["kkey"], q["role"]
    s = _pair_of(data, kk)
    sd = float(s.sd.iloc[0])
    head = f"{role.capitalize()} example  \u2013  {data['name'][kk[0]]} + {data['name'][kk[1]]}"
    if role == "synergistic":
        over = sorted((d for d in PHENO6
                       if float(s.loc[d, "drop_comb"]) - float(s.loc[d, "drop_add"]) > 0.03),
                      key=lambda d: -(float(s.loc[d, "drop_comb"]) - float(s.loc[d, "drop_add"])))
        parts = " and ".join(
            f"{DIM_1LINE[d]} (additive {_e_num(float(s.loc[d, 'drop_add']))} \u2192 real "
            f"{_e_num(float(s.loc[d, 'drop_comb']))})" for d in over[:2])
        newd = [d for d in PHENO6 if int(s.loc[d, "n_hit"]) == 0 and bool(s.loc[d, "aff_c"])]
        obs = (f"Observations. The pair overshoots the additive expectation in {parts}, and it reaches "
               f"\u0394 {_e_num(float(s.loc[newd[0], 'drop_comb']))} in {DIM_1LINE[newd[0]]} \u2013 a dimension "
               f"neither single factor affects at all (a new phenotype). All 3 replicates agree at the "
               f"6-dimension aggregate level (SD {sd:.3f}); the error bars give the per-dimension "
               f"replicate SD.")
        cause = ("Possible cause of synergy. The two lesions sit at the two ends of one loop: information "
                 "seeking is what turns uncertainty into a probe (ask, explore, test), while switching cost "
                 "is what prices leaving the current task set. Either lesion alone leaves the loop runnable "
                 "\u2013 a system that stops probing can still switch when pushed, and a system that switches "
                 "cheaply can still act on a probe it generates. Losing both closes the loop: nothing probes "
                 "and nothing releases the current set, so the same single-factor damage compounds instead of "
                 "adding up, and it reaches a dimension neither lesion touched.")
    else:
        gapd = min(PHENO6, key=lambda d: float(s.loc[d, "drop_comb"]) - float(s.loc[d, "drop_add"]))
        gap = float(s.loc[gapd, "drop_comb"]) - float(s.loc[gapd, "drop_add"])
        obs = (f"Observations. In all {len(PHENO6)} dimensions the real combination stays below the "
               f"additive expectation \u2013 by up to {_e_num(gap)} in {DIM_1LINE[gapd]} (additive "
               f"{_e_num(float(s.loc[gapd, 'drop_add']))} \u2192 real "
               f"{_e_num(float(s.loc[gapd, 'drop_comb']))}) \u2013 and the pair reads at or near the intact "
               f"baseline everywhere (largest residual \u0394 {_e_num(float(s.drop_comb.max()))}). No "
               f"dimension emerges that neither single factor had reached. All 3 replicates agree at the "
               f"6-dimension aggregate level (SD {sd:.3f}); the error bars give the per-dimension "
               f"replicate SD.")
        cause = ("Possible cause of buffering. Both lesions cut the same upstream gate: valuation decides "
                 "whether an information gap is worth pursuing at all, and selective attention decides which "
                 "of the parallel channels is let through. Cut both and the stream is severed before it "
                 "reaches the dimensions the other lesion would have disturbed \u2013 the second lesion removes "
                 "the route along which the first one\u2019s damage would be expressed. The pair therefore "
                 "saturates instead of adding, and the additive expectation (a linear extrapolation) "
                 "over-predicts: it assumes a pathway that is already off.")
    return head, obs, cause


def panel_e(fig, data: dict):
    """第二行最右列：上下两张柱子图（一个协同例、一个缓冲例），各自冠一行彩色角色标签。

    文字解释（Observations / Possible cause）不出现在图上，正文仍由 `_e_case_texts()`
    现算并写进 metrics（硬门 7 照旧验那两条断言）；图例化简成两行三列压在 panel 右上角，
    不再占 panel 底部；`CODE_KEY` 已移到图底脚注（见 `draw()`）。
    """
    x0, y0 = col_x(R2_W)["e"], ROW2_Y
    W, H = R2_W["e"], ROW2_H
    sel = data["e_selected"]

    sub = {q["kkey"]: _pair_of(data, q["kkey"]) for q in sel}
    # y 轴范围要罩住**误差棒的两端**，不然最大那根柱子的上帽会被裁掉、最矮那根的下帽会顶到轴下
    lo, hi = 0.0, 0.0
    for s in sub.values():
        for d in PHENO6:
            for c, e in (("drop_a", "sd_a"), ("drop_b", "sd_b"),
                         ("drop_add", "sd_add"), ("drop_comb", "sd_comb")):
                lo = min(lo, float(s.loc[d, c]) - float(s.loc[d, e]))
                hi = max(hi, float(s.loc[d, c]) + float(s.loc[d, e]))
    ylim = (lo - 0.03, hi + 0.05)

    lab = 0.62                     # 左槽：y 轴标题 + 刻度
    cw = W - lab - 0.10            # 柱子图宽（6 组 × 4 根）
    # 垂直预算：图例顶 0.12 + 图例 0.38 + 缝 0.14、每块「角色标签 0.36 + 图 + x 刻度 0.20」、
    # 两张图之间 0.26、底边 0.10
    top = y0 + H - E_LEG_DY - E_LEG_H - E_HEAD_GAP
    plot_h = (top - y0 - 0.10 - 2 * E_TICK_H - E_GAP - 2 * E_HEAD_H) / 2
    _e_legend(fig, x0 + W - 0.12, y0 + H - E_LEG_DY)

    for i, q in enumerate(sel):
        kk, role = q["kkey"], q["role"]
        head_y = top - i * (E_HEAD_H + plot_h + E_TICK_H + E_GAP)
        ax = plain_ax(fig, rect(x0 + lab, head_y - E_HEAD_H - plot_h, cw, plot_h))
        s = sub[kk]
        wd = 0.20
        for j, d in enumerate(PHENO6):
            r = s.loc[d]
            if r.n_hit > 0:
                ax.plot([j], [ylim[0] * 0.42], marker="o", ms=2.8, color=f2.INK, zorder=5)
            if r.n_hit == 0 and bool(r.aff_c):
                ax.plot([j], [ylim[0] * 0.42], marker="*", ms=5.5, color=C_SYN, zorder=5)
            for off, val, err, colr, hatch in [
                    (-1.5 * wd, float(r.drop_a), float(r.sd_a), BAR_C["a"], None),
                    (-0.5 * wd, float(r.drop_b), float(r.sd_b), BAR_C["b"], None),
                    (0.5 * wd, float(r.drop_add), float(r.sd_add), "white", "////"),
                    (1.5 * wd, float(r.drop_comb), float(r.sd_comb),
                     _sign_c(float(r.drop_comb) - float(r.drop_add)), None)]:
                ax.bar(j + off, val, width=wd, color=colr, edgecolor=C_ADD if hatch else f2.INK,
                       linewidth=0.6 if hatch else 0.35, hatch=hatch, zorder=3)
                # error bar = 该量在 3 次重复上的 SD（ddof = 1）：重复一致性是这张图的判据之一
                # （选 pair 就要求 3 次同号），柱子的高低没有误差棒是读不出来的。
                ax.errorbar(j + off, val, yerr=err, fmt="none", ecolor=f2.INK, elinewidth=0.6,
                            capsize=1.3, capthick=0.6, zorder=5)
        ax.set_xlim(-0.62, 5.62)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels([CODE[d] for d in PHENO6])
        ax.axhline(0, color=f2.INK, lw=0.8, zorder=2)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_ylabel("\u0394 impairment", labelpad=2)
        thin_axis(ax)
        # 角色标签 + pair 名：原来是右列文字块的第一行，现在分两行压在图上（文字段已删）
        fig.text((x0 + lab) / FIG_W, head_y / FIG_H, f"{role.capitalize()} example",
                 fontsize=FS, fontweight="bold", color=ROLE_C[role], ha="left", va="top")
        fig.text((x0 + lab) / FIG_W, (head_y - 0.16) / FIG_H,
                 f"{data['name'][kk[0]]} + {data['name'][kk[1]]}",
                 fontsize=FS, color=f2.INK, ha="left", va="top")


# ---------------------------------------------------------------------------
# 组装 / 落盘
# ---------------------------------------------------------------------------


def draw(data: dict):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
    fig.patch.set_facecolor("white")
    for f in (panel_a, panel_b, panel_c, panel_d, panel_e):
        f(fig, data)
    # 整图唯一的代号对照表（两行）：从 panel e 框内移到图底脚注带，panel d / e 的 6 列同序，
    # 所以仍然只写这一份。
    fig.text((M_L + 0.06) / FIG_W, CODE_KEY_Y / FIG_H, CODE_KEY, fontsize=FS, color=f2.INK,
             ha="left", va="bottom", linespacing=1.30)
    return fig


def metrics(data: dict, info: dict) -> dict:
    it, taken, new, stab = data["items"], data["taken"], data["new"], data["stab"]
    ori_by_cell = int(len(data["taken"]) + (data["taken"].n_hit == 2).sum())
    out = {
        "figure": STEM,
        "provenance": {
            "pair_table": str(PAIR_XLSX.relative_to(ROOT)),
            "single_table": str(UNI_XLSX.relative_to(ROOT)),
            "dose": PAIR_DOSE,
            "single_factor_dose": PAIR_DOSE,
            "pair_dose": PAIR_DOSE,
            "dose_scope": "every single-factor and every pair quantity in this figure is the theta = 0.8 "
                          "cell; the 0.4 dose of result_2 is not used anywhere in this figure",
            "affected_criterion": "f2._t_p(perturbed, intact_reference, equal_var=True) + BH-FDR "
                                  "within dose x dimension family; drop >= TARGET_DROP (0.15) and "
                                  "q < TEST_ALPHA (0.05)",
            "baseline": {"per_dimension": data["base_d"], "per_dimension_spread": data["base_dev"],
                         "mean_of_six": data["base6"]},
            "phenotype_dimension_names": "verbatim from fig2_visualize.py DIM_1LINE (one line) and "
                                         "DIM_2L (two lines): "
                                         + " / ".join(DIM_1LINE[d].replace("\n", " ") for d in PHENO6),
            "column_codes": CODE_KEY.replace("\n", " "),
            "typography": "fig2 body scale x 1.25: Arial/Helvetica sans-serif; one body size FS = 9.0 pt "
                          "for every axis label, annotation, legend and tick label (ticks bold, as in "
                          "fig2); panel letters and panel titles are not drawn on the figure (they "
                          "belong in the caption); fig2 itself uses 7.2 pt body and 10 pt bold headings, "
                          "so the body scale is multiplied by 1.25 and the figure width is unchanged "
                          "(17 in) to make the type larger relative to the figure",
            "definitions": {
                "interaction_strength": "expected_additive − combined = −interaction_<dim>; "
                                        "positive = synergistic, negative = buffering",
                "replicate_strength": "(single_A6 + single_B6 − baseline6) − combined6, "
                                      "computed inside each replicate",
                "aggravation_change": "Δ(pair) − Δ(the heavier single factor of that pair); "
                                      "±0.03 direction threshold; panel b never uses the additive "
                                      "expectation A + B",
                "panel_b_reference": "the more impaired single factor of the pair, not the additive "
                                     "expectation single A + single B (used in panel e only)",
                "conclusion_stability": "same threshold applied per replicate, no per-replicate test",
                "error_bars": "every bar in panel e carries the SD of that same quantity across the "
                              "3 replicates (ddof = 1); the additive expectation has no measurement "
                              "of its own, so its per-replicate value is drop_A + drop_B and the "
                              "error bar is the SD of that sum",
            },
        },
        "panel_a": {
            "n_pairs": int(len(stab)),
            "same_direction_3of3": int(stab.sign3.sum()),
            "same_direction_3of3_pct": float(100 * stab.sign3.mean()),
            "sd_median": float(stab.sd.median()), "sd_mean": float(stab.sd.mean()),
            "sd_max": float(stab.sd.max()),
            "sd_quantiles": {str(q): float(np.percentile(stab.sd, q)) for q in (10, 25, 50, 75, 90)},
            "sd_class_counts": {"3of3": int(stab.sign3.sum()), "2of3": int((~stab.sign3).sum())},
            "corr_abs_strength_vs_sd": float(np.corrcoef(np.abs(stab["mean"]), stab.sd)[0, 1]),
            "agreement_by_abs_strength": {
                "0.02-0.05": [int(stab["mean"].abs().between(.02, .05).sum()),
                              float(100 * stab.sign3[stab["mean"].abs().between(.02, .05)].mean())],
                "0.05-0.10": [int(stab["mean"].abs().between(.05, .10).sum()),
                              float(100 * stab.sign3[stab["mean"].abs().between(.05, .10)].mean())],
                ">=0.10": [int((stab["mean"].abs() >= .10).sum()),
                           float(100 * stab.sign3[stab["mean"].abs() >= .10].mean())]},
            "conclusion_stability": dict(data["stable"], _note="no longer drawn: the panel a "
                                                         "chart that showed these three rates was "
                                                         "removed on request (its third bar carried "
                                                         "an aggravated-cell count)"),
            "agreement_bins": [
                dict(lo=lo, hi=hi, n=int(((np.abs(stab["mean"]) >= lo)
                                          & (np.abs(stab["mean"]) < hi)).sum()),
                     same_3of3_pct=float(100 * stab.sign3[(np.abs(stab["mean"]) >= lo)
                                                          & (np.abs(stab["mean"]) < hi)].mean()))
                for lo, hi in [(0, .02), (.02, .05), (.05, .10), (.10, .20), (.20, 1)]],
        },
        "panel_b": {
            "displayed": "per dimension a fan of slopes: one slope per Main cognitive factor (every "
                         "single factor of that dimension that already meets Δ >= 0.15 and q < 0.05), "
                         "each drawn against ITS OWN worst partner (largest change), so the left end "
                         "spreads over the qualifying factors and the right end shows where each is "
                         "pushed. Exactly one slope is bold and labelled: the largest change among "
                         "them, named as 'Main cognitive + its secondary partner'",
            "counts_dropped": "better / worse / unchanged cell tallies are deliberately not on the "
                              "figure any more; the per-dimension pool size is kept below as "
                              "provenance for the selection",
            "n_cells": int(len(taken)),
            "n_orientations": int(ori_by_cell),
            "both_singles_affected": int((taken.n_hit == 2).sum()),
            "per_dimension": {d: dict(n=int((taken.dim == d).sum()),
                                      n_orientations=int((data["ori"].dim == d).sum()))
                              for d in PHENO6},
            "fan": {d: [dict(main=str(g.main), secondary=str(g.secondary),
                            drop_main=float(g.drop_main), drop_comb=float(g.drop_comb),
                            change=float(g.change))
                       for _, g in data["b_fan"][d].sort_values("drop_main", ascending=False)
                       .iterrows()]
                    for d in PHENO6},
            "selected": [dict(dim=d, main=str(r.main), secondary=str(r.secondary),
                             pair=str(r.main) + " + " + str(r.secondary),
                             drop_main=float(r.drop_main), drop_comb=float(r.drop_comb),
                             change=float(r.change), n_hit=int(r.n_hit), q=float(r.q),
                             sd=float(r.sd), sign3=bool(r.s3))
                         for d, r in data["b_picks"].items()],
            "top_aggravated": [dict(pair=r.pair, dim=r.dim, change=float(r.change),
                                    drop_single=float(r.drop_single), drop_comb=float(r.drop_comb))
                               for _, r in taken.nlargest(5, "change").iterrows()],
            "top_relieved": [dict(pair=r.pair, dim=r.dim, change=float(r.change),
                                  drop_single=float(r.drop_single), drop_comb=float(r.drop_comb))
                             for _, r in taken.nsmallest(5, "change").iterrows()],
        },
        "panel_c": {
            "n_cells": int(len(new)), "n_pairs": int(new.pair.nunique()),
            "per_dimension": {d: dict(pairs=int((new.dim == d).sum()),
                                      eligible=int(((it.dim == d) & (it.n_hit == 0)).sum()),
                                      rate_pct=float(100 * (new.dim == d).sum()
                                                     / max(int(((it.dim == d) & (it.n_hit == 0)).sum()), 1)))
                              for d in PHENO6},
            "strongest_per_dimension": [dict(dim=r.Index, pair=str(r.pair), drop_comb=float(r.drop_comb),
                                             a=str(r.a), b=str(r.b))
                                        for r in new.sort_values("drop_comb", ascending=False)
                                        .groupby("dim", sort=False).head(1)
                                        .set_index("dim").reindex(PHENO6).dropna(
                                            subset=["drop_comb"], how="any")
                                        .sort_values("drop_comb", ascending=False)
                                        .itertuples()],
            "top": [dict(pair=r.pair, dim=r.dim, drop_comb=float(r.drop_comb), q=float(r.q_c),
                         drop_a=float(r.drop_a), drop_b=float(r.drop_b))
                    for _, r in new.sort_values("drop_comb", ascending=False)
                    .drop_duplicates("kkey").head(9).iterrows()],
            "n_cells_per_pair_max": int(new.groupby("kkey").size().max()),
        },
        "panel_d": {
            "drawn": "matrix only; the two right-hand bar charts (row sum, new-phenotype cells) "
                     "were removed on request, so totals and new_phenotype_partners are kept here "
                     "as provenance but are not drawn",
            "dimension_order": PHENO6,
            "family_of_primitive": {p: data["fam"][p] for p in data["ids"]},
            "row_order": data["d_order"],
            "totals": {p: int(data["cnt_mat"][i].sum()) for i, p in enumerate(data["d_order"])},
            "matrix": {p: [int(v) for v in data["cnt_mat"][i]] for i, p in enumerate(data["d_order"])},
            "new_phenotype_partners": {p: int(data["newcnt"][p]) for p in data["ids"]},
        },
        "panel_e": [], "gates": info,
    }
    for q in data["e_selected"]:
        kk, s = q["kkey"], _pair_of(data, q["kkey"])
        head, obs, cause = _e_case_texts(data, q)
        out["panel_e"].append(dict(
            pair=q["pair"], role=q["role"], sd=float(s.sd.iloc[0]), sign3=bool(s.s3.iloc[0]),
            heading=head, observations=obs, possible_cause=cause, text_on_figure=False,
            per_dimension={d: dict(single_A=float(s.loc[d, "drop_a"]),
                                   single_B=float(s.loc[d, "drop_b"]),
                                   additive_expectation=float(s.loc[d, "drop_add"]),
                                   real_combination=float(s.loc[d, "drop_comb"]),
                                   strength=float(s.loc[d, "strength"]),
                                   error_bar_sd=dict(single_A=float(s.loc[d, "sd_a"]),
                                                     single_B=float(s.loc[d, "sd_b"]),
                                                     additive_expectation=float(s.loc[d, "sd_add"]),
                                                     real_combination=float(s.loc[d, "sd_comb"])),
                                   already_affected=bool(s.loc[d, "n_hit"] > 0),
                                   new_phenotype=bool(s.loc[d, "n_hit"] == 0 and s.loc[d, "aff_c"]),
                                   change_vs_worst_single=(None if s.loc[d, "n_hit"] == 0
                                                           else float(s.loc[d, "change"])))
                              for d in PHENO6}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = load()
    _pick(data)
    bad, info = gate(data)

    print(f"465 对：3/3 同号 {int(data['stab'].sign3.sum())} "
          f"({100 * data['stab'].sign3.mean():.1f}%)，SD 中位 {data['stab'].sd.median():.4f}")
    print(f"单因素 affected 格子 {int(sum(data['aff_s'].values()))}；"
          f"组合 affected 格子 {int(sum(data['aff_c'].values()))}")
    t = data["taken"]
    print(f"panel b：每个维度一张扇面 = 该维度**每个符合条件的 Main cognitive** 一条线"
          f"（每个只留它最坏的那个 partner，共 {sum(len(f) for f in data['b_fan'].values())} 条），"
          f"只给最显著的一对标名称")
    for d in PHENO6:
        r, n = data["b_picks"][d], len(data["b_fan"][d])
        print(f"  panel b · {d:<38} {n:>2} Main cognitive   worst: {r.main} + {r.secondary}"
              f"  Δ {r.drop_main:.2f} → {r.drop_comb:.2f}  {r.change:+.3f}")
    print(f"panel c：新表型 {len(data['new'])} 格，涉及 {data['new'].pair.nunique()} 对")
    print("结论稳定性（已不上图，只留 metrics）：" + "，".join(
        f"{k} {100 * v:.1f}%" for k, v in data["stable"].items()))
    print(f"panel d：最多 {data['d_order'][0]} = {int(data['cnt_mat'][0].sum())}，"
          f"最少 {data['d_order'][-1]} = {int(data['cnt_mat'][-1].sum())}")
    for q in data["e_selected"]:
        s = _pair_of(data, q["kkey"])
        print(f"  panel e · {q['role']:<18} {q['pair']:<58}"
              f" worst Δ{q['_worst']:+.3f}  new {q['_nnew']}  buf {q['_buf']:+.3f}  "
              f"Δadd≤{q['_dadd']:.2f}  sd {q['_sd']:.3f}")
    print(f"硬门：{'全部通过' if not bad else '失败'}  ·  逐维度链路偏差 {info['chain_dev']:.2e}  ·  "
          f"重复内均值偏差 {info['mean_chain_dev']:.4f}  ·  单因素基线偏差 {info['single_base_dev']:.2e}  ·  "
          f"panel e provenance 偏差 {info['ef_provenance_dev']:.1e}  ·  "
          f"panel e error bar 偏差 {info['e_err_dev']:.1e}")
    for b in bad:
        print("  !", b)
    if args.dry_run or bad:
        return 1 if bad else 0

    fig = draw(data)
    fig.canvas.draw()
    unify_axis_ticks(fig)
    # 刻度放大后必须再排一次版：坐标轴标题的位置是**按刻度字号**留出来的（f2 的 rcParams
    # 还写着 7.2 pt），不重排的话 x 轴标题会压在刻度上 —— 这一条只有 9 pt 之后才暴露出来。
    fig.canvas.draw()
    clashes = check_text_clashes(fig)
    outside = check_inside_frames(fig)
    covered = check_text_covered(fig)
    grays = f2.check_no_gray_text(fig)
    for a, b in clashes:
        print(f"  ! 文字相撞：{a!r} × {b!r}")
    for a, x1, y0 in outside:
        print(f"  ! 文字跑出面板框：{a!r} → 右边界 {x1} in, 下边界 {y0} in")
    for a, b, frac in covered:
        print(f"  ! 文字被坐标轴底纹压住：{a!r} 被 {b} 盖了 {100 * frac:.0f}%")
    for a, b in grays:
        print(f"  ! 灰色文字：{a!r} {b}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res_png = f2.save(fig, OUT_DIR / f"{STEM}.png")
    res_pdf = f2.save(fig, OUT_DIR / f"{STEM}.pdf")
    plt.close(fig)
    (OUT_DIR / METRICS_NAME).write_text(
        json.dumps(metrics(data, info), indent=1, ensure_ascii=False), encoding="utf-8")
    n_bad = len(clashes) + len(covered) + len(grays)
    print(f"落盘：{STEM}.png [{res_png}] / .pdf [{res_pdf}] / {METRICS_NAME}"
          + (f"（文字相撞 {len(clashes)} 处，被底纹压住 {len(covered)} 处，灰字 {len(grays)} 处）"
             if n_bad else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
