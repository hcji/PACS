"""把多档合并运行目录导出成发布表（`results/dose_supplement/`）。

## 它和 `export_result_tables.py` 的关系

`export_result_tables.py` 只认死路径：读 `pacs/data/runs/scale_single` 与 `scale_pair`，
并**覆盖写** `result_2_univariate.xlsx` / `result_3_multivariable.xlsx`。本脚本要读的是另一个
目录、写的是另一个文件，所以不能直接调它 —— 但表版式与字段口径必须与它**逐项一致**
（否则同一段下游代码读三个文件时类型/列名对不上）。

做法是 `import export_result_tables as ert`，复用它的 `load_records` / `subject_row` /
`sheet_by_subject` / `sheet_dose_and_specificity` / `write_rows`，只换输入目录与输出路径。
**既有脚本一个字都不改**。

## 为什么 `--doses` 必须与合并目录的聚合一致

特异性判定是按 `--doses` 逐档求值的（`scale_single.aggregate` 里
`specificities = {dose: sp.evaluate_specificity(combined, dose=dose) for dose in doses}`）。
合并目录的聚合如果按默认档位跑，报告里就会出现「数据是 0.2/0.6/0.9、判定按 0.4/0.8 算」的
静默错配。所以本脚本在导出前会先核对 `scale_analysis.json` 里实际存在的档位与 `--doses`
是否逐档一致，**对不上就报错退出**，不出一份数字与档位错位的表。

## 用法

```bash
# 四档版本（0.2/0.4/0.6/0.8）
python export_dose_supplement.py

# 五档版本（加 0.9）—— 与四档版本并存，不互相覆盖
python export_dose_supplement.py \
    --merged-run "…/runs/scale_single_dose_merged_5dose" \
    --doses 0.2,0.4,0.6,0.8,0.9 --xlsx-name dose_supplement_univariate_5dose.xlsx
```

## 三张表

- `by_subject`        5 primitive × 档位数 × 3 重复（其中 0.4/0.8 两档是结果 2 的既有数据，
                      是复制来的、**未重跑**；`row_id` 见下方「⚠」）
- `summary_by_cell`   5 primitive × 档位数：逐维 mean/delta + 主导维度 + 特异性与剂量-反应判定
                      （剂量-反应对该 primitive 的所有档 + 1.0 基线锚点拟合，所以同一
                      primitive 的各行给出同一组拟合值）
- `dose_ladder`       5 primitive × 7 维度：把各档 delta 横排，用来直接看阶梯形状
                      （最重档落在哪一档、是否单调）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
PACS = _ROOT / "pacs"
for _path in (str(PACS / "src"), str(PACS / "tools"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import export_result_tables as ert  # noqa: E402  （复用字段口径与版式，不触发它的写盘）

from openpyxl.utils import get_column_letter  # noqa: E402
from pacs.phenotype import channel_label  # noqa: E402

DEFAULT_MERGED_RUN = PACS / "data" / "runs" / "scale_single_dose_merged"
DEFAULT_OUT_DIR = _ROOT / "results" / "dose_supplement"
DEFAULT_XLSX_NAME = "dose_supplement_univariate_4dose.xlsx"

#: 合并批次实际跑的 5 个 primitive
PRIMITIVES: tuple[str, ...] = (
    "agent_state_updating",
    "belief_updating",
    "rule_updating",
    "self_other_distinction",
    "social_reward_weighting",
)

#: 默认档位，**降序**给出（横排 delta 时读数应随档位下降而下降）
DEFAULT_DOSES: tuple[float, ...] = (0.8, 0.6, 0.4, 0.2)
#: 默认每（primitive × 档）的重复次数
DEFAULT_REPLICATES = 3

#: 结果 2 里在 0.4/0.8 两档各有这么多次重复 —— 「可回指结果 2 的行数」的期望值由此而来
RESULT_2_DOSES: tuple[float, ...] = (0.4, 0.8)
RESULT_2_REPLICATES = 3

#: 结果 2 里也有的那些行 —— 用来核对「复用数据确实对得上」，不是新跑出来的
RESULT_2 = _ROOT / "results" / "data" / "result_2_univariate.xlsx"

#: `result_2_univariate.xlsx` 的 `by_subject` 版式 = 本脚本的 16 列省略三个模型列后的 13 列。
#: 省略后**列宽随之左移三位**（F/G/H 拿到原 I/J/K 的宽度，M 拿到原 P 的宽度）。
#: 要「与 result_2 同格式」就得照抄省略后的结果，而不是照抄那个脚本里的常量。
RESULT_2_DROP_COLUMNS = ("brain_model", "examiner_model", "analyst_model")
RESULT_2_HEADERS = [h for h in ert.HEADERS if h not in RESULT_2_DROP_COLUMNS]
RESULT_2_WIDTHS = {
    "A": 29.5555555555556, "B": 31.6666666666667, "C": 14.0, "D": 11.0,
    "F": 26.0, "G": 25.0, "H": 26.0, "M": 46.0,
}

#: `result_2` 的 `summary_by_cell` 实测只给这些列设了列宽（H–T、W 是默认宽）。
#: 既然要求「与 result_2 同格式」，就照它实测的来。
RESULT_2_SUMMARY_WIDTHS = {
    "A": 34.0, "B": 16.0, "C": 8.0, "D": 12.0, "E": 14.0, "F": 12.0, "G": 14.0,
    "U": 32.0, "V": 14.0, "X": 12.0, "Y": 22.0, "Z": 20.0,
    "AA": 16.0, "AB": 10.0, "AC": 80.0,
}
#: `result_2` 的 `by_subject` 冻结在 A44（前 43 行被钉住）。
#: 默认用 A2（可用 `--freeze-panes A44` 复现 result_2）。
RESULT_2_FREEZE = "A44"


def apply_result_2_summary_widths(ws) -> None:
    """把 `summary_by_cell` 的列宽调成 `result_2` 里实测的那一套（其余列清掉显式列宽）。

    `openpyxl` 不接受把 `width` 设成 `None`，要清掉显式列宽只能删掉该列的 `ColumnDimension`。
    """
    for i in range(1, ws.max_column + 1):
        letter = get_column_letter(i)
        if letter in RESULT_2_SUMMARY_WIDTHS:
            ws.column_dimensions[letter].width = RESULT_2_SUMMARY_WIDTHS[letter]
        elif letter in ws.column_dimensions:
            del ws.column_dimensions[letter]


# ---------------------------------------------------------------------------


def dose_text(doses: tuple[float, ...]) -> str:
    """档位串（与命令行 `--doses` 同格式，升序）。"""
    return ",".join(f"{d:g}" for d in sorted(doses))


def load_analysis(run_dir: Path, doses: tuple[float, ...]) -> dict:
    """读合并目录的聚合结果，并核对档位与 `doses` 逐档一致（对不上直接失败）。"""
    path = run_dir / "analysis" / "scale_analysis.json"
    if not path.is_file():
        raise SystemExit(
            f"✗ 找不到聚合结果：{path}\n"
            "  先跑 `python tools/scale_single.py --aggregate-only "
            f"--doses {dose_text(doses)} --no-generic --out <合并目录>`。"
        )
    result = json.loads(path.read_text(encoding="utf-8"))

    present = sorted({round(float(c["dose"]), 4) for c in result["cells"]})
    if present != sorted(doses):
        raise SystemExit(
            f"✗ 聚合结果里的档位是 {present}，本脚本要的是 {sorted(doses)}。\n"
            f"  这通常意味着合并目录的聚合没带 `--doses {dose_text(doses)}`："
            "特异性判定会按默认档位算，表里的数字与档位错位。"
        )
    spec_doses = sorted({round(float(k), 4) for k in result["specificity"]})
    if spec_doses != sorted(doses):
        raise SystemExit(
            f"✗ specificity 的档位是 {spec_doses}，不是 {sorted(doses)} —— "
            f"聚合时漏了 `--doses {dose_text(doses)}`。"
        )
    return result


def sheet_dose_ladder(wb, result: dict, doses: tuple[float, ...]) -> int:
    """逐 primitive × 维度的各档 delta 横排 —— 看剂量阶梯的形状。

    `worst_dose` 是 delta 最负的那一档（不需要任何容差阈值）；`monotone_steps` 是在
    档位从高到低这个顺序上 delta **严格递降**的相邻对数（0 到 len(doses)-1）。
    两个都是无参数描述量：noisy 的 3 次重复下严格单调很少成立，所以只报计数，
    形状判断留给读者看着各档 delta 自己做。
    """
    cells: dict[tuple[str, float], dict] = {}
    for cell in result["cells"]:
        cells[(cell["primitive_id"], round(float(cell["dose"]), 4))] = cell

    headers = [
        "primitive_id", "primitive_zh", "dimension", "dimension_zh",
        *[f"delta_{dose:g}" for dose in doses],
        "worst_dose", "range", "monotone_steps",
    ]
    rows = []
    for pid in sorted({c["primitive_id"] for c in result["cells"]}):
        zh = cells[(pid, doses[0])]["primitive_zh"]
        for dim in ert.DIMS:
            deltas: list[object] = []
            for dose in doses:
                channel = (cells.get((pid, dose)) or {}).get("channels", {}).get(dim) or {}
                deltas.append(channel.get("delta", ""))
            # 各档必须都有值才算得出最重档与极差；缺一档就只留 delta 空着，
            # 不去猜（缺值与「没变化」在剂量阶梯上是两种完全不同的读法）。
            if not all(isinstance(d, (int, float)) for d in deltas):
                rows.append([pid, zh, dim, channel_label(dim), *deltas, "", "", ""])
                continue
            desc = [float(d) for d in deltas]
            worst = doses[min(range(len(doses)), key=lambda i: desc[i])]
            steps = sum(1 for a, b in zip(desc, desc[1:]) if b < a)
            rows.append([
                pid, zh, dim, channel_label(dim), *deltas,
                round(worst, 4), round(max(desc) - min(desc), 4), steps,
            ])

    widths = {"A": 32, "B": 16, "C": 34, "D": 22}
    for offset in range(len(doses)):
        widths[get_column_letter(5 + offset)] = 12
    for offset, width in enumerate([12, 10, 14]):
        widths[get_column_letter(5 + len(doses) + offset)] = width

    ws = wb.create_sheet("dose_ladder")
    ert.write_rows(ws, headers, rows, widths)
    return len(rows)


# ---------------------------------------------------------------------------


def dose_of(record: dict) -> float:
    """该被试被削弱的档位（单因素设计下唯一）。"""
    active = (record.get("parameters") or {}).get("active") or {}
    return round(float(next(iter(active.values()))), 4) if active else 0.0


def source_row_id(record: dict) -> str:
    """该记录在**它自己那个批次**里的 `row_id`（结果 1/2/3 的口径，不含剂量）。

    只用来回指来源批次，**不能**当本文件的唯一键用（见 `row_id`）。
    """
    return ert.row_id(record)


def row_id(record: dict) -> str:
    """本文件的唯一键：`<条件>#<剂量 2 位>#<被试序号 3 位>_r<重复>`。

    为什么必须把剂量嵌进来：结果 1/2/3 的 `row_id` 是 `<条件>#<被试序号>_r<重复>`，
    而「被试序号」是**该批次剂量表里的位次**，不是剂量本身。两个批次的剂量表不同
    （既有批次 001 = 0.8 / 002 = 0.4；补充批次 001 = 0.2 / 002 = 0.6），所以合并后
    `single_belief_updating#001_r0` 同时指 0.8 和 0.2 两份不同的测量 —— 照抄旧格式会
    让 60 行只剩 30 个不同的 `row_id`，恰好每个撞两次。

    数据本身不含糊（`mind_setting` / `completeness` 两列都带着剂量），但一个自称是行
    标识的列不能重复，所以这里显式嵌入剂量。这也是本文件与结果 2/3 唯一的**取值**差异；
    列名、列序、列数仍逐项一致。
    """
    condition = record["condition"]
    return (f"{condition['condition_id']}#{dose_of(record):.2f}"
            f"#{int(record.get('subject_index') or 0):03d}"
            f"_r{int(condition.get('replicate') or 0)}")


def order_for_reading(records: list[dict]) -> list[dict]:
    """按（条件, 剂量降序, 重复）排 —— 这是结果 2 那张表行序在 4 档下的推广。

    结果 2 的行序是（条件, 被试序号, 重复），而它两个档的被试序号恰好是 001 = 0.8、
    002 = 0.4，所以那个序就是「剂量降序」。合并目录里 0.8/0.4 与 0.2/0.6 来自两个批次、
    被试序号各自从 001 起，照抄旧排序会把同一个 primitive 的四个档打散成六行一组交错，
    看不出阶梯。所以显式按剂量降序排，与结果 2 的读法保持一致。
    """
    return sorted(records, key=lambda r: (
        str(r["condition"]["condition_id"]),
        -dose_of(r),
        int((r["condition"].get("replicate") or 0)),
    ))


def sheet_by_subject(wb, records: list[dict], *, layout: str = "full") -> None:
    """逐被试表。

    `layout="full"` 写 16 列（含三方模型名）；`layout="result2"` 写 `result_2` 那样的 13 列
    （删掉三个模型列，列宽随之左移三位）。两种情况都只有 `row_id` 是本文件的唯一键
    （`<条件>#<剂量>#<序号>_r<重复>`），其余列名、列序、字体、冻结都与来源一致。
    """
    drop = layout == "result2"
    rows = []
    for record in records:
        row = ert.subject_row(record)
        if drop:
            del row[5:8]                     # brain_model / examiner_model / analyst_model
        row[0] = row_id(record)
        rows.append(row)
    headers = RESULT_2_HEADERS if drop else ert.HEADERS
    widths = RESULT_2_WIDTHS if drop else ert.COLUMN_WIDTHS
    ws = wb.create_sheet(ert.SHEET)
    ert.write_rows(ws, headers, rows, widths)


def traceability(records: list[dict]) -> tuple[int, int, int]:
    """返回（可回指结果 2 的行数, 本次新跑的行数, 重复 row_id 数）。

    回指按（来源批次 row_id, 剂量）成对匹配：只用 row_id 会漏判 —— 新跑的 0.2 档与
    既有的 0.8 档在各自批次里的 row_id 本来就相同（都是 `#001`），单看 id 会把新数据
    算成复用数据。
    """
    import openpyxl

    wb = openpyxl.load_workbook(RESULT_2, read_only=True, data_only=True)
    try:
        ws = wb["by_subject"]
        existing = {
            (str(row[0]), str(row[2]))
            for row in ws.iter_rows(min_row=2, values_only=True) if row[0]
        }
    finally:
        wb.close()

    reused = sum(1 for r in records if (source_row_id(r), ert.completeness(r)) in existing)
    ids = [row_id(r) for r in records]
    return reused, len(records) - reused, len(ids) - len(set(ids))


def write_readme(
    result: dict,
    records: list[dict],
    reused: int,
    fresh: int,
    *,
    doses: tuple[float, ...],
    replicates: int,
    merged_run: Path,
    out_dir: Path,
    xlsx_name: str,
    sources: list[str],
) -> Path:
    desc = " / ".join(f"{d:g}" for d in sorted(doses, reverse=True))
    lines: list[str] = []
    lines.append(f"# 剂量阶梯（{desc}）· 合并视图")
    lines.append("")
    lines.append("## 这份表是什么")
    lines.append("")
    lines.append(f"把 {len(doses)} 个档位合到同一张剂量-反应曲线与同一份逐档特异性表里的视图。")
    lines.append("其中一部分档位是**复用数据**，本批次没有重跑（逐档来源见下）。")
    lines.append("")
    lines.append(f"- primitive（5 个）：{'、'.join(PRIMITIVES)}")
    lines.append(f"- 档位：{desc}")
    lines.append(f"- 重复：每（primitive × 档）{replicates} 次")
    lines.append("- 协议：50 轮（配额不足补至 100）/ seed 42 / 题池 100 @ seed 42 / "
                 "text_limit 100 / max_tokens 8000 / temperature 0.3 / 三方同为 deepseek-flash")
    lines.append("- 完整心智基线：复用 `pacs/data/runs/fig1_main/axis1/axis1_same_brain` 的 5 份"
                 "（协议 / 三方模型 / 题池逐位一致，每次开跑前都校验），不重跑")
    lines.append("- 通用退化对照：**未纳入**（`--no-generic`）")
    lines.append("- 心智年龄：固定 36 个月")
    lines.append("")
    lines.append("### 来源批次")
    lines.append("")
    for source in sources:
        lines.append(f"- {source}")
    lines.append("")
    lines.append(f"- 行数构成：`by_subject` {len(records)} 行 —— **{reused} 行**可回指结果 2"
                 f"（0.4 / 0.8 两档里更早批次跑的那 {RESULT_2_REPLICATES} 次重复，**未重跑**），"
                 f"其余 **{fresh} 行**为本轮新增或重跑")
    lines.append(f"- 档位覆盖：`summary_by_cell` {len(result['cells'])} 行 = "
                 f"{len({c['primitive_id'] for c in result['cells']})} primitive × {len(doses)} 档")
    lines.append("")
    lines.append("### ⚠ `row_id` 的格式与结果 2/3 不同（列名与列序仍一致）")
    lines.append("")
    lines.append("本文件的 `row_id` 是 `<条件>#<剂量>#<被试序号>_r<重复>`（如 "
                 "`belief_updating#0.20#001_r0`），比结果 2/3 多一段剂量。")
    lines.append("")
    lines.append("原因是结果 2/3 的「被试序号」是**该批次剂量表里的位次**、不是剂量本身：")
    lines.append("不同批次的同一位次对应不同剂量。照抄旧格式会让多档合并后大量行 `row_id` 相同")
    lines.append("（同一个 `#001_r0` 同时指数个不同剂量的测量）。一个自称是行标识的列不能重复，")
    lines.append("所以这里嵌入了剂量。数据本身不带歧义：`mind_setting`（`belief_updating 0.20`）")
    lines.append("与 `completeness`（`0.2`）两列都写着剂量。")
    lines.append("")
    lines.append("## 三张表怎么读")
    lines.append("")
    lines.append("| 表 | 一行是什么 | 用途 |")
    lines.append("|---|---|---|")
    lines.append("| `by_subject` | 一个被试 | 逐被试 7 维读数 + 分析者结论全文 |")
    lines.append("| `summary_by_cell` | 一个（primitive × 档） | 逐维 mean/delta、主导维度、"
                 "特异性判定、剂量-反应斜率与 r² |")
    lines.append("| `dose_ladder` | 一个（primitive × 维度） | 各档 delta 横排 + 最重档 + 极差 + "
                 "严格递降步数，看阶梯形状 |")
    lines.append("")
    lines.append("`delta` 一律 = 该格均值 − 完整心智基线均值；负值 = 比同龄典型发展更差。")
    lines.append("`completeness` 就是剂量本身（该认知过程的**保留程度**）：1.0 = 完整，0.2 = 几乎不可用。")
    lines.append("")
    lines.append("## 产物出处")
    lines.append("")
    lines.append("| 层 | 路径 |")
    lines.append("|---|---|")
    lines.append(f"| 合并运行目录（{len(doses)} 档原始记录） | `{merged_run}` |")
    lines.append(f"| └ 装配出处（逐份记录来源、θ 指纹） | `{merged_run / 'PROVENANCE.md'}` |")
    lines.append(f"| └ 聚合报告（逐格数值 + 结论） | "
                 f"`{merged_run / 'analysis' / 'scale_analysis.md'}` |")
    lines.append("| 各来源批次原始运行 | 见上方「来源批次」 |")
    lines.append(f"| 本表 | `{out_dir / xlsx_name}` |")
    lines.append("")
    lines.append("逐被试的**完整对话过程**在运行目录的 `cells/<条件>/subjects/*.json` 的 "
                 "`dialogue.turns` 里（逐轮原文），本表只带分析者结论（`analyst_summary` 列）。")
    lines.append("")
    lines.append("## 必须一并引用的局限")
    lines.append("")
    lines.append("1. **复用档位与新跑档位不是同一批测量**：同源、协议逐项一致、题池逐位相同，"
                 "但**误差结构不同源** —— 阶梯上复用档的重复间离散度不能与新跑的档位直接比。")
    lines.append("2. **0.2 档接近「该认知过程几乎不可用」**：可能触发整体性塌陷，使特异性判定")
    lines.append("   集体落到 `fail_control` / `fail_diffuse`。**这是剂量过高时该设计的已知行为，")
    lines.append("   不是测量失败**，结论里必须写清而不能当成异常丢弃。")
    lines.append("3. 每档 3 次重复是特异性自校准门槛的下限（`specificity.py` 要求 ≥3 才用重复")
    lines.append("   估计标准误）：门槛本身会随重复噪声在档位之间摆动（实测同一 primitive 的四档")
    lines.append("   门槛从 0.05 摆到 0.32），所以**逐档判定之间的差异主要反映门槛在动，"
                 "不足以当作「某档特异、另一档不特异」的证据**。")
    lines.append("4. 真实 provider 不接收 seed、temperature 固定 0.3：同一 θ 的重复之间测到的是")
    lines.append("   服务端采样噪声 + 分析者给分噪声，不是「同一仪器同一样本」的复现。")
    lines.append("5. 50 轮只比「7 通道 × 5 问 = 35」的配额多 15 轮余量，仍可能有补问（实际轮数 >50）。")
    lines.append("6. 单因素设计**不识别交互**：本表不给任何交互/协同结论。")
    lines.append("7. 仿真产物（虚拟被试），非真实儿童数据；仅用于方法学研究，不用于诊断。")
    lines.append("")
    lines.append("用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。")
    lines.append("")

    # 说明文件随 xlsx 命名（`…_5dose.xlsx` → `…_5dose.md`），这样同一目录里可以同时放
    # 四档版与五档版，各自的说明不会互相覆盖。
    path = out_dir / f"{Path(xlsx_name).stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def verify_written(path: Path) -> dict:
    """读回落盘文件核对 —— 报的是**产物本身**，不是内存里的意图。"""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        counts: dict[str, int] = {}
        for name in wb.sheetnames:
            ws = wb[name]
            counts[name] = sum(
                1 for row in ws.iter_rows(min_row=2, values_only=True)
                if any(c is not None and c != "" for c in row)
            )
        ws = wb["by_subject"]
        ids = [r[0] for r in ws.iter_rows(min_row=2, values_only=True) if r[0]]
        counts["distinct_row_id"] = len(set(ids))
        counts["by_subject_cols"] = ws.max_column
        counts["distinct_mind_setting"] = len({
            r[1] for r in ws.iter_rows(min_row=2, values_only=True) if r[1]
        })
    finally:
        wb.close()
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_dose_supplement",
        description="把多档合并运行目录导出成发布表",
        epilog=(
            "示例：\n"
            "  python export_dose_supplement.py\n"
            "  python export_dose_supplement.py \\\n"
            '    --merged-run "…/runs/scale_single_dose_merged_5dose" \\\n'
            "    --doses 0.2,0.4,0.6,0.8,0.9 \\\n"
            "    --xlsx-name dose_supplement_univariate_5dose.xlsx\n"
        ),
    )
    parser.add_argument("--merged-run", default=str(DEFAULT_MERGED_RUN),
                        help=f"合并运行目录（默认 {DEFAULT_MERGED_RUN.name}）")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"发布表输出目录（默认 results/dose_supplement）")
    parser.add_argument("--doses",
                        default=",".join(f"{d:g}" for d in DEFAULT_DOSES),
                        help="档位，逗号分隔；必须与合并目录聚合时的 --doses 一致")
    parser.add_argument("--xlsx-name", default=DEFAULT_XLSX_NAME, dest="xlsx_name",
                        help=f"输出文件名（默认 {DEFAULT_XLSX_NAME}）")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                        help=f"每（primitive × 档）的重复次数（默认 {DEFAULT_REPLICATES}）")
    parser.add_argument("--by-subject-layout", choices=("full", "result2"), default="full",
                        dest="by_subject_layout",
                        help="by_subject 版式：full = 16 列含三方模型名；"
                             "result2 = 13 列（删掉三个模型列，列宽左移三位，与 "
                             "result_2_univariate.xlsx 逐项一致）")
    parser.add_argument("--no-ladder", action="store_true", dest="no_ladder",
                        help="不写 dose_ladder 表（result_2 只有 by_subject + summary_by_cell 两张表）")
    parser.add_argument("--freeze-panes", default="A2", dest="freeze_panes",
                        help="by_subject 的冻结窗格（默认 A2；传 A44 可逐项复现 result_2，"
                             "那会把前 43 行钉住）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doses = tuple(float(c) for c in str(args.doses).replace("，", ",").split(",") if c.strip())
    replicates = int(args.replicates)
    layout = args.by_subject_layout
    ladder = not args.no_ladder
    merged_run = Path(args.merged_run)
    if not merged_run.is_absolute():
        merged_run = (_ROOT / merged_run).resolve()
    out_dir = Path(args.out_dir)
    xlsx = out_dir / args.xlsx_name

    result = load_analysis(merged_run, doses)
    records = order_for_reading(ert.load_records(merged_run))
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = ert.new_workbook()
    sheet_by_subject(wb, records, layout=layout)
    ert.sheet_dose_and_specificity(wb, merged_run)
    if ladder:
        sheet_dose_ladder(wb, result, doses)
    if layout == "result2":
        apply_result_2_summary_widths(wb["summary_by_cell"])
    if args.freeze_panes != "A2":
        wb[ert.SHEET].freeze_panes = args.freeze_panes
    wb.save(xlsx)

    reused, fresh, duplicates = traceability(records)
    sources = sorted({
        str(Path(r["_path"]).parents[2].relative_to(_ROOT)) for r in records
        if r.get("_path")
    })
    readme = write_readme(result, records, reused, fresh, doses=doses,
                          replicates=replicates, merged_run=merged_run, out_dir=out_dir,
                          xlsx_name=args.xlsx_name, sources=sources)
    counts = verify_written(xlsx)

    expected_subjects = len(PRIMITIVES) * len(doses) * replicates
    expected_reused = len(PRIMITIVES) * len(RESULT_2_DOSES) * RESULT_2_REPLICATES
    expected_cells = len(PRIMITIVES) * len(doses)
    expected_ladder = len(PRIMITIVES) * len(ert.DIMS)

    print(f"✓ {xlsx}")
    print(f"  by_subject      : {counts['by_subject']} 行（期望 {expected_subjects}"
          f" = 5 × {len(doses)} 档 × {replicates} 重复）"
          f"　不同 row_id {counts['distinct_row_id']} 个"
          f"　可回指结果 2 的 {reused} 行 / 新跑 {fresh} 行")
    print(f"  summary_by_cell : {counts['summary_by_cell']} 行（期望 {expected_cells}）")
    if ladder:
        print(f"  dose_ladder     : {counts['dose_ladder']} 行（期望 {expected_ladder}）")
    else:
        print(f"  工作表          : {wb.sheetnames}（未写 dose_ladder，与 result_2 的两表版式一致）")
    print(f"  不同 mind_setting: {counts['distinct_mind_setting']} 种"
          f"（期望 {expected_cells} = 5 primitive × {len(doses)} 档）")
    print(f"  {readme}")

    checks = [
        ("by_subject 行数", counts["by_subject"], expected_subjects),
        ("by_subject 不同 row_id 数", counts["distinct_row_id"], expected_subjects),
        ("summary_by_cell 行数", counts["summary_by_cell"], expected_cells),
        ("不同 mind_setting 数", counts["distinct_mind_setting"], expected_cells),
        ("可回指结果 2 的行数", reused, expected_reused),
        ("新跑行数", fresh, expected_subjects - expected_reused),
        ("重复 row_id 个数", duplicates, 0),
    ]
    if ladder:
        checks.append(("dose_ladder 行数", counts["dose_ladder"], expected_ladder))
    if layout == "result2":
        checks.append(("by_subject 列数", counts["by_subject_cols"], len(RESULT_2_HEADERS)))
        checks.append(("工作表数", len(wb.sheetnames), 2))
    ok = True
    for label, got, want in checks:
        if got != want:
            print(f"✗ {label}：{got} != {want}")
            ok = False
    print("✓ 全部校验通过" if ok else "✗ 有校验未通过")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
