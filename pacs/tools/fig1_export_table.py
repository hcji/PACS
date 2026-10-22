"""把逐被试产物导出成一张好读的表：一行一被试，列是心智设定 / 6 个观测维度 / 分析者总结 / 元信息。

**为什么要单独导一份**：`flat/all_subjects.csv` 是给统计和绘图用的（列名是内部字段名、7 个维度
不分观测通道与对照通道、心智设定只给中文 `theta_describe`）。这份导出面向人读：
列名用英文、把 6 个观测维度与 1 个对照通道分开摆、心智设定写成「intact / belief_updating 0.80」
这种一眼能对上的形式，并附一份列说明。

**分析者总结保持原文（中文）**：那是模型当时写下的原始数据，翻译等于改写数据。要英文版请
另跑一遍翻译，不要把译文和原文混在同一列里。

产物：`<root>/tables/subjects_table.{csv,xlsx}`（xlsx 有 4 个 sheet：逐被试 / 按心智设定汇总 /
按条件×心智设定 / 列说明）。stdlib + 可选 openpyxl（没装就只出 CSV）。

运行：`PYTHONPATH=src python tools/fig1_export_table.py`
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fig1_reliability as fig1  # noqa: E402  （借 DIMS / SCORED_CHANNELS，保持口径一致）

#: 6 个「观测维度」= 除对照通道外的全部打分通道。对照通道单独一列，不混在里面。
CONTROL = "general_cognition"
OBSERVED = tuple(d for d in fig1.DIMS if d != CONTROL)

META_COLUMNS = (
    "row_id", "mind_setting", "completeness", "condition_id", "axis", "replicate",
    "turns_run", "brain_model", "examiner_model", "analyst_model",
)

COLUMN_NOTES: tuple[tuple[str, str], ...] = (
    ("row_id", "<condition>#<sequence>_r<replicate>：这条记录在磁盘上的位置标签"),
    ("mind_setting", "心智设定。intact = 无扰动；其余是单个原语被降到该完整度"),
    ("completeness", "完整度：1.00 = 完整，越小削弱越重。**不是损伤度**"),
    ("condition_id", "七个条件之一（见可靠性报告的四条轴）"),
    ("axis", "所属轴：① 同一脑重复 ② 不同 LLM ③ 不同提问者/分析者/题序 ④ 不同轮次"),
    ("replicate", "该 (条件, 心智设定) 下的第几次重复，从 0 起"),
    ("turns_run", "实际跑完的对话轮数（50/100 协议下配额满足即收工，故可能大于设定值）"),
    ("brain_model", "扮演被试（心智）的模型"),
    ("examiner_model", "扮演提问者（医生）的模型"),
    ("analyst_model", "扮演分析者的模型。提问者与分析者同模型时才叫「同一个医生」"),
    *((dim, f"[观测维度] {fig1.SCORED_CHANNELS[dim]}；1.0 = 与同龄典型发展无异，0.0 = 严重偏离")
      for dim in OBSERVED),
    (CONTROL, f"[对照通道] {fig1.SCORED_CHANNELS[CONTROL]}；"
              f"不参与孤独症样表型判定，用于看特异性（是否「哪里都掉」）"),
    ("mean_6_observed", "6 个观测维度的均值"),
    ("mean_all_7", "含对照通道的 7 维均值（分析者自己也报了这个数）"),
    ("asd_present", "分析者的档位判定：是否表现出孤独症样表型特征"),
    ("asd_level", "档位：none / mild / moderate / marked"),
    ("analyst_confidence", "分析者自报的置信度"),
    ("analyst_summary", "**分析者的总结原文（中文）**。原始数据，未翻译"),
    ("analyst_summary_chars", "总结字数"),
    ("n_subjects", "[汇总表] 该格并入的被试数"),
    ("verdict_none/mild/moderate/marked", "[汇总表] 分析者档位判定的计数"),
    ("*_mean / *_sd", "[汇总表] 该格内跨被试的均值 / 标准差"),
    ("sheet by_mind_setting",
     "把**全部 7 个条件**合在一起按心智设定汇总 —— 这些条件的脑/提问者/分析者/轮数协议不同，"
     "合起来读会把那些轴的效应和 θ 效应叠加。要干净地看剂量-反应，请看 by_condition_x_mind "
     "里 condition_id = axis1_same_brain 的那 4 行（参考臂：唯一变量是 θ）"),
)


def mind_setting(parameters: dict) -> str:
    """把 θ 写成一眼能对上的英文标签。"""
    if parameters.get("is_intact"):
        return "intact (no perturbation)"
    active = parameters.get("active") or {}
    if len(active) == 1:
        (primitive, dose), = active.items()
        return f"{primitive} {float(dose):.2f}"
    return "+".join(f"{k} {float(v):.2f}" for k, v in sorted(active.items())) or "unknown"


def completeness(parameters: dict) -> float | None:
    if parameters.get("is_intact"):
        return 1.0
    values = [float(v) for v in (parameters.get("active") or {}).values()]
    return min(values) if values else None


def subject_row(record: dict, *, index: int) -> dict:
    parameters = record.get("parameters") or {}
    condition = record.get("condition") or {}
    analysis = record.get("analysis") or {}
    dialogue = record.get("dialogue") or {}
    readings = {d: analysis.get("readings", {}).get(d) for d in fig1.DIMS}
    observed = [v for d, v in readings.items() if d in OBSERVED and v is not None]
    every = [v for v in readings.values() if v is not None]
    likeness = analysis.get("asd_likeness") or {}
    row = {
        "row_id": f"{condition.get('condition_id')}#{index:03d}_r{condition.get('replicate')}",
        "mind_setting": mind_setting(parameters),
        "completeness": completeness(parameters),
        "condition_id": condition.get("condition_id"),
        "axis": condition.get("axis"),
        "replicate": condition.get("replicate"),
        "turns_run": dialogue.get("n_examiner_turns"),
        "brain_model": condition.get("brain_model"),
        "examiner_model": condition.get("examiner_model"),
        "analyst_model": condition.get("analyst_model"),
    }
    for dim in fig1.DIMS:
        row[dim] = readings.get(dim)
    row["mean_6_observed"] = (sum(observed) / len(observed)) if observed else None
    row["mean_all_7"] = (sum(every) / len(every)) if every else None
    row["asd_present"] = likeness.get("present")
    row["asd_level"] = likeness.get("level")
    row["analyst_confidence"] = analysis.get("confidence")
    row["analyst_summary"] = analysis.get("conclusion") or ""
    row["analyst_summary_chars"] = analysis.get("conclusion_chars")
    return row


def load_rows(root: Path) -> list[dict]:
    """按 (条件, 序号) 稳定排序读入，序号取文件名里的三位序号。"""
    rows: list[dict] = []
    for path in sorted(root.glob("*/*/subjects/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            index = int(path.name.split("_", 1)[0])
        except ValueError:
            index = 0
        rows.append(subject_row(record, index=index))
    rows.sort(key=lambda r: (str(r["axis"]), str(r["condition_id"]), str(r["mind_setting"]),
                             int(r["replicate"] or 0)))
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def aggregate(rows: list[dict], *, by_condition: bool) -> list[dict]:
    """按「心智设定」或「条件 × 心智设定」汇总：每维均值/标准差、总体均值、档位分布。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = ((row["condition_id"], row["mind_setting"]) if by_condition
               else (row["mind_setting"],))
        groups[key].append(row)

    out: list[dict] = []
    for key in sorted(groups, key=lambda k: (k[0], -(completeness_key(k[-1])))):
        members = groups[key]
        item: dict = {}
        if by_condition:
            item["condition_id"] = key[0]
        item["mind_setting"] = key[-1]
        item["completeness"] = next((r["completeness"] for r in members
                                     if r["completeness"] is not None), None)
        item["n_subjects"] = len(members)
        for dim in fig1.DIMS:
            values = [r[dim] for r in members if r[dim] is not None]
            item[f"{dim}_mean"] = _mean(values)
            item[f"{dim}_sd"] = _sd(values)
        for name, field in (("mean_6_observed", "mean_6_observed"),
                            ("mean_all_7", "mean_all_7")):
            values = [r[field] for r in members if r[field] is not None]
            item[f"{name}_mean"] = _mean(values)
            item[f"{name}_sd"] = _sd(values)
        levels = Counter(str(r["asd_level"]) for r in members)
        for level in ("none", "mild", "moderate", "marked"):
            item[f"verdict_{level}"] = levels.get(level, 0)
        item["n_turns_mean"] = _mean([float(r["turns_run"]) for r in members
                                      if r["turns_run"] is not None])
        out.append(item)
    return out


def completeness_key(label: str) -> float:
    """把心智设定标签还原成完整度，用于把梯子按「完整 → 削弱」排序。"""
    if label.startswith("intact"):
        return 1.0
    parts = label.split()
    try:
        return float(parts[-1])
    except (IndexError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：Excel 直接双击打开中文才不乱码
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})


def _cell(value):
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return "" if value is None else value


def write_xlsx(path: Path, sheets: list[tuple[str, list[dict], list[str]]]) -> bool:
    """写 xlsx。openpyxl 不在就返回 False（只出 CSV，不让导出整体失败）。"""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        return False
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows, columns in sheets:
        sheet = workbook.create_sheet(name[:31])
        sheet.append([column_label(c) for c in columns])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for row in rows:
            sheet.append([_cell(row.get(c)) for c in columns])
        sheet.freeze_panes = "A2"
        for index, column in enumerate(columns, start=1):
            width = 46 if column == "analyst_summary" else max(
                10, min(26, max(len(str(column_label(column))), 8) + 2))
            sheet.column_dimensions[get_column_letter(index)].width = width
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return True


def column_label(column: str) -> str:
    """列名尽量自解释：维度列带上中文标签，方便对照。"""
    if column in fig1.SCORED_CHANNELS:
        return f"{column} [{fig1.SCORED_CHANNELS[column]}]"
    return column


def main() -> int:
    parser = argparse.ArgumentParser(prog="fig1_export_table")
    parser.add_argument("--root", default=str(_ROOT / "data" / "runs" / "fig1_main"))
    parser.add_argument("--out", default=None, help="缺省写到 <root>/tables/")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "tables"

    rows = load_rows(root)
    if not rows:
        raise SystemExit(f"✗ 没有读到被试记录：{root}")

    subject_columns = list(META_COLUMNS) + list(fig1.DIMS) + [
        "mean_6_observed", "mean_all_7", "asd_present", "asd_level",
        "analyst_confidence", "analyst_summary", "analyst_summary_chars",
    ]
    by_mind = aggregate(rows, by_condition=False)
    mind_columns = (["mind_setting", "completeness", "n_subjects"]
                    + [f"{d}_mean" for d in fig1.DIMS] + [f"{d}_sd" for d in fig1.DIMS]
                    + ["mean_6_observed_mean", "mean_all_7_mean",
                       "verdict_none", "verdict_mild", "verdict_moderate", "verdict_marked",
                       "n_turns_mean"])
    by_cond = aggregate(rows, by_condition=True)
    cond_columns = ["condition_id"] + mind_columns
    notes = [{"column": column_label(c), "meaning": meaning} for c, meaning in COLUMN_NOTES]

    write_csv(out_dir / "subjects_table.csv", rows, subject_columns)
    write_csv(out_dir / "by_mind_setting.csv", by_mind, mind_columns)
    write_csv(out_dir / "by_condition_x_mind.csv", by_cond, cond_columns)
    wrote_xlsx = write_xlsx(out_dir / "subjects_table.xlsx", [
        ("by_subject", rows, subject_columns),
        ("by_mind_setting", by_mind, mind_columns),
        ("by_condition_x_mind", by_cond, cond_columns),
        ("column_notes", notes, ["column", "meaning"]),
    ])

    print(f"被试行数：{len(rows)}　心智设定 {len(by_mind)} 档　条件×设定 {len(by_cond)} 行")
    print(f"已写入 {out_dir / 'subjects_table.csv'}")
    print(f"已写入 {out_dir / 'by_mind_setting.csv'}")
    print(f"已写入 {out_dir / 'by_condition_x_mind.csv'}")
    print(f"已写入 {out_dir / 'subjects_table.xlsx'}" if wrote_xlsx
          else "（openpyxl 不在，跳过 xlsx）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
