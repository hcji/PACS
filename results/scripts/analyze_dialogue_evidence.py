"""从 result_2 / result_4 回读**已记录的对话探查过程**，做三个问题的个案分析。

## 这个脚本要回答什么

1. **受损的心智在某一维度上，如何区别于正常心智。**
   同一个计分通道，完整心智的逐字证据 vs 受损心智的逐字证据，并排。
2. **同一维度得分降低、但机制不同时，行为表现是否有差异。**
   两轴独立证据：①从原始对话算的客观转写标记；②分析者已记录的观察文本（含套话剥离）。
3. **同一机制轻度只坏一个表型、严重时坏多个，原因是什么。**
   四条解释逐条判定：E1 非特异通用退化 / E2 集中度坍缩 / E3 证据共用 / E4 真实级联。

## 范围：个案，不是全表

三问各只做 2–3 个代表性个案，**个案由 Fig. 2 的既有产物选定**（不手工挑）：

- Q1 取自 `results/data/figure2_link_candidates.csv` 的 tier A（五项筛选全过）；
- Q2 取自 `figure2_metrics.json → panel_b.per_dose["0.4"].top3_per_dimension`，
  并要求组内 Δ 配对（|Δ₁−Δ₂| ≤ `--match-tolerance`）以排除「只是轻重不同」；
- Q3 取自 `figure2_metrics.json → panel_f.per_primitive` 的 5 个可做剂量阶梯的 primitive，
  按「最能区分四条解释」选 3 例。

选定规则写在 `select_*` 函数里、逐条打印理由，落在 `case_selection` 表上。

## 两条必须随数据一起读的口径

**① 逐题配对做不到。** 提问者是 LLM 即时生成措辞，`probe_key` 只是题池槽位；实测同一个
`probe_key` 下的问题文本在各次运行间**零重合**（连两次完整心智重跑之间也是 0）。所以配对只
落在**通道（`focus`）层面**；`probe_key` 恰好相同时做**槽位级**配对，并明确标注「同槽位、
非同一句」。措辞不同的两次提问只能算「同一探查目标的两次取样」，不能算同一道题。

**② 完整心智参照只有 10 份，且是同一个 θ。**
数值 Δ 严格沿用官方 5 份（`fig1_main/axis1/axis1_same_brain`，与 `figure2_metrics.json` 的
`baseline` 逐位相同）；行为参照扩到 `fig1_realcheck/axis1` 同 θ 的 5 份。
这 10 份是**同一个完整心智 θ 的 10 段独立会话**，不是 10 个个体。

## 它与既有文件的关系

只读 `result_2` / `result_4` / 批次 `flat/all_subjects.csv` / 逐被试 JSON / Fig. 2 产物；
只写 `results/data/` 下四个新文件。脚本首尾各算一次 `result_1..4` 的 sha256 并断言相等 ——
防止任何一次误覆盖。版式复用 `export_result_tables.py` 的 `write_rows` / `new_workbook`
（宋体 11、表头行高 57.6、冻结 A2、`id [中文]` 双语表头），与 `export_dose_supplement.py`
同一套做法。

## 红线

本项目禁止「构造属性 → 测量维度」的预设对应（`pacs/tests/test_no_preset_mapping.py`）。
本脚本遵守同一条纪律：

- 「哪个机制在哪个维度上掉得多」一律**从 result_2/result_4 的测量值读出**，脚本内没有任何
  机制 → 维度的对应表；
- `results/data/prior_hypotheses.json` 里的机制—维度配对只用于**个案排序的偏好**
  （挑与已有理论相关的个案来讲），**不参与任何数值计算与结论**，且在报告里显式标注；
- 本产物是**事后解释层**，不是测量：不得回写为后续运行的先验（同
  `pacs/analysis/mechanism_profile.py` 的 `NO_WRITEBACK_NOTE`）。

## 用法

```bash
python analyze_dialogue_evidence.py                    # 全离线，零网络调用
python analyze_dialogue_evidence.py --dry-run          # 只打印选定与闸门，不写盘
python analyze_dialogue_evidence.py --llm-contrast-plan-only   # 列出「必要时」值得判读的配对
python analyze_dialogue_evidence.py --llm-contrast     # 对已记录对话做匿名化对照判读
```
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

_ROOT = Path(__file__).resolve().parents[2]
PACS = _ROOT / "pacs"
for _path in (str(PACS / "src"), str(PACS / "tools"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import export_result_tables as ert  # noqa: E402  （复用字段口径与版式，不触发它的写盘）

from pacs.phenotype import (  # noqa: E402
    CONTROL_DIM,
    PHENOTYPE_DIMS,
    SCORED_CHANNELS,
    channel_label,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

RESULT_2 = _ROOT / "results" / "data" / "result_2_univariate.xlsx"
RESULT_4 = _ROOT / "results" / "data" / "result_4_does_response.xlsx"
#: 首尾各算一次 sha256 —— 这四份是已发布产物，本脚本绝不许碰它们
PROTECTED = (
    _ROOT / "results" / "data" / "result_1_robustness.xlsx",
    RESULT_2,
    _ROOT / "results" / "data" / "result_3_multivariable.xlsx",
    RESULT_4,
)

RUNS = PACS / "data" / "runs"
SINGLE_RUN = RUNS / "scale_single"
DOSE_RUN = RUNS / "scale_single_dose_merged_5dose_5rep"

FIGURE2_METRICS = _ROOT / "results" / "figure" / "figure2_metrics.json"
LINK_CANDIDATES = _ROOT / "results" / "data" / "figure2_link_candidates.csv"
PRIOR_HYPOTHESES = _ROOT / "results" / "data" / "prior_hypotheses.json"

OUT_DIR = _ROOT / "results" / "data"
XLSX_NAME = "dialogue_evidence_casebook.xlsx"
MD_NAME = "dialogue_evidence_casebook.md"
JSON_NAME = "dialogue_evidence_metrics.json"
LLM_JSON_NAME = "dialogue_evidence_llm_contrast.json"

#: 完整心智参照：官方 5 份（数值 Δ 用）+ 同 θ 同协议的另 5 份（行为参照用）
OFFICIAL_BASELINE_DIR = RUNS / "fig1_main" / "axis1" / "axis1_same_brain" / "subjects"
REFERENCE_DIRS = (
    OFFICIAL_BASELINE_DIR,
    RUNS / "fig1_realcheck" / "axis1" / "axis1_same_brain" / "subjects",
)

CHANNELS = list(SCORED_CHANNELS)          # 7 个计分通道（6 表型 + 对照）
PHENO = list(PHENOTYPE_DIMS)              # 6 个表型维度
#: 「这个维度算被影响了」的读数下降门槛 —— 与 `pacs/analysis/specificity.py` 的 TARGET_DROP 同值
HIT = 0.15
HIT_STRICT = 0.30
#: 任务文本 → 提问轮 的匹配门槛与置信边际（见 `attribute_evidence`）
MATCH_MIN = 0.34
MATCH_MARGIN = 1.5
#: 参照记录里同通道的提问轮数下限 —— 低于它就不给该通道的行为底样（样本太少没有意义）
REF_MIN_TURNS = 3

ROW_ID_2 = re.compile(r"^(?P<cond>.+?)#(?P<idx>\d{3})_r(?P<rep>\d+)$")
ROW_ID_4 = re.compile(r"^(?P<cond>.+?)#(?P<dose>\d+\.\d+)#(?P<idx>\d{3})_r(?P<rep>\d+)$")

#: `probe_key` 形态：`tools<量表条目序号>:<该条目下的第几问>`
PROBE_KEY = re.compile(r"^tools(\d+):(\d+)$")

NO_WRITEBACK = (
    "本产物是**事后解释层**，不是测量：由「已记录的对话 + 已记录的逐维读数」再读一遍得到，"
    "不得回写为后续运行的先验（不得进提示词、不得参与特异性判据、不得进冻结快照）。"
)

LIMITATIONS = [
    "1. 每格 3 次重复是特异性自校准门槛的下限，重复间噪声不可忽略。",
    "2. 真实 provider 不接收 seed、temperature 固定 0.3，测得的是服务端采样噪声。",
    "3. 50 轮配额余量有限，部分通道的问题数刚够配额。",
    "4. 基线是**复用数据**（来自 Fig.1 主运行），需连路径与指纹一起引用。",
    "5. 单因素设计不识别交互 —— 一个通道的变化无法归因到单一机制的独立效应。",
    "6. 仿真产物，不是真实儿童数据。",
    "7. 0.2 档已接近「该认知过程几乎不可用」，可能触发通用性整体退化，这不是测量错误。",
    "8. 剂量阶梯里 0.4/0.8 两档是 result_2 的**复用数据**（同一次测量，非独立重复）。",
    "9. 完整心智参照是**同一个完整心智 θ 的 10 段独立会话**，不是 10 个不同个体。",
    "10. **不存在逐题配对**：提问措辞由提问者 LLM 每次重新生成，跨运行对比只到通道 / 题池槽位"
    "粒度；`probe_key` 相同的两次提问也不是同一句话。",
    "11. 分析者观察文本是 LLM 散文、**不是受控测量工具**；轴一（转写标记）与轴二（记录文本）"
    "须同时给出，两轴一致方可作强结论。",
    "12. 每格 3 次重复下的置换检验为**探索性**，不得当作确证性检验报告。",
]


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(paths) -> dict[str, dict]:
    """产物指纹：大小 + sha256。用于「跑完确认没被覆盖」。"""
    out: dict[str, dict] = {}
    for path in paths:
        path = Path(path)
        out[path.name] = {
            "path": str(path),
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256(path) if path.is_file() else "",
        }
    return out


def clean_text(value) -> str:
    return " ".join(str(value or "").split())


def bigrams(text) -> set[str]:
    """字符 2-gram 集合（去空白）。中文短句上比词袋稳，且不需要分词器。"""
    joined = "".join(ch for ch in str(text or "") if not ch.isspace())
    if len(joined) < 2:
        return {joined} if joined else set()
    return {joined[i:i + 2] for i in range(len(joined) - 1)}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def coverage(needle: set, haystack: set) -> float:
    """`needle` 有多少比例能在 `haystack` 里找到 —— 长问句里找短任务名要用这个，不用 Jaccard。"""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def norm_answer(text) -> str:
    return re.sub(r"[\s，。、！？：；「」『』（）()…—\-~·,.!?:;\"']", "", str(text or ""))


def mean_or_blank(values) -> object:
    present = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(present) / len(present), 4) if present else ""


def safe_stdev(values) -> float:
    present = [float(v) for v in values if isinstance(v, (int, float))]
    return statistics.stdev(present) if len(present) > 1 else 0.0


# ---------------------------------------------------------------------------
# 读表 / 读记录
# ---------------------------------------------------------------------------


def read_sheet(path: Path, sheet: str) -> list[dict]:
    """读一张 xlsx 表为 list[dict]（表头在第 1 行）。只读模式，不改盘。"""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook[sheet].iter_rows(values_only=True)
        headers = ["" if h is None else str(h) for h in next(rows)]
        return [dict(zip(headers, raw)) for raw in rows]
    finally:
        workbook.close()


def dim_of_header(header: str) -> str | None:
    """`social_cognition [社会认知]` → `social_cognition`；不是通道列则返回 None。"""
    for channel in CHANNELS:
        if header.startswith(channel + " ["):
            return channel
    return None


def as_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def dose_of(record: dict) -> float:
    active = (record.get("parameters") or {}).get("active") or {}
    if active:
        return round(float(next(iter(active.values()))), 4)
    generic = float((record.get("parameters") or {}).get("generic_degradation") or 0.0)
    return round(generic, 4)


def primitive_of(record: dict) -> str:
    active = (record.get("parameters") or {}).get("active") or {}
    return sorted(active)[0] if active else ""


def dose_from_active(text) -> float | None:
    """`all_subjects.csv` 的 `theta_active` 形如 `belief_updating=0.6` → 0.6。"""
    text = str(text or "")
    if "=" not in text:
        return None
    try:
        return round(float(text.rsplit("=", 1)[-1]), 4)
    except ValueError:
        return None


def build_join(run_dir: Path) -> tuple[dict, dict]:
    """`(条件, 被试序号, 重复[, 剂量]) → {subject_json_path, session_id, …}`。

    用批次自带的 `flat/all_subjects.csv`：它带 `session_id` / `provider_kind` / `synthetic` /
    `subject_json_path` 四样 xlsx 里没有的东西，是 xlsx 行回指到逐被试 JSON 的唯一桥。

    **剂量必须进键。** 5 档批次的剂量表把同一个 `被试序号 × 重复` 分给了不同的档
    （实测 50 个三元组里有 40 个在档位间碰撞），只按三元组查会静默取到另一个剂量的记录。
    result_4 的 `row_id` 之所以要把剂量嵌进来，正是为了消歧；这里必须同样处理。
    对三元组唯一的情况保留一份三元组索引，作为 `theta_active` 解析不出剂量时的退路。
    """
    path = run_dir / "flat" / "all_subjects.csv"
    if not path.is_file():
        raise SystemExit(f"✗ 找不到 {path} —— 缺了它就无法把 row_id 回指到逐被试 JSON。")
    by4: dict[tuple, dict] = {}
    by3: dict[tuple, dict] = {}
    seen3: Counter = Counter()
    payloads: list[tuple[tuple, dict]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            triple = (str(row["condition_id"]), int(row["subject_index"]), int(row["replicate"]))
            payload = {
                "subject_json_path": row["subject_json_path"],
                "session_id": row["session_id"],
                "provider_kind": row.get("provider_kind", ""),
                "synthetic": row.get("synthetic", ""),
                "theta_id": row.get("theta_id", ""),
                "theta_describe": row.get("theta_describe", ""),
                "n_turns_run": row.get("n_turns_run", ""),
            }
            seen3[triple] += 1
            payloads.append((triple, payload))
            dose = dose_from_active(row.get("theta_active"))
            if dose is not None:
                by4[triple + (dose,)] = payload
    for triple, payload in payloads:
        if seen3[triple] == 1:
            by3[triple] = payload
    return by4, by3


def resolve_json(raw: str, run_dir: Path) -> Path:
    """把 `subject_json_path` 解析成真实路径 —— 两个批次的存法不同（相对 pacs/ vs 绝对）。"""
    text = str(raw).replace("\\", "/")
    candidates = [Path(text)]
    if not Path(text).is_absolute():
        candidates += [PACS / text, _ROOT / text, run_dir.parent / text]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


@dataclass
class Subject:
    """一个被试行：来自哪张表 + 它自己的逐被试 JSON。"""

    row_id: str
    source_batch: str                     # result_2 / result_4
    condition_id: str
    primitive_id: str
    dose: float
    replicate: int
    subject_index: int
    readings: dict[str, float | None]
    json_path: Path
    session_id: str
    provider_kind: str
    synthetic: str
    theta_id: str = ""
    content_sha: str = ""                 # 逐被试 JSON 的内容指纹，用来判「同一段会话」
    reused: bool = False                  # 与另一张表里的某一行是同一段会话
    reused_from: str = ""                 # 复用源的 row_id
    record: dict = field(default_factory=dict)
    pairs: list[dict] = field(default_factory=list)

    @property
    def is_control_cell(self) -> bool:
        return not self.primitive_id

    def value(self, channel: str):
        return self.readings.get(channel)


def load_subject_rows(path: Path, run_dir: Path, batch: str, counters: dict) -> list[Subject]:
    """读一张发布表的第一张表，逐行回指到逐被试 JSON，并逐份核对。

    `row_id` 的剂量段（result_4 有、result_2 没有）是回指的**必要**组成部分：
    5 档批次里同一个 `被试序号 × 重复` 会被分给不同档，只按三元组查会取到另一个剂量的记录。
    """
    raw_rows = read_sheet(path, "by_subject")
    by4, by3 = build_join(run_dir)
    pattern = ROW_ID_4 if batch == "result_4" else ROW_ID_2
    subjects: list[Subject] = []
    for raw in raw_rows:
        row_id = str(raw.get("row_id") or "")
        match = pattern.match(row_id)
        counters["rows_total"] += 1
        if not match:
            counters["row_id_unparsed"] += 1
            raise SystemExit(f"✗ {batch} 的 row_id 解析不了：{row_id!r}（格式与预期不符）")
        condition_id = match.group("cond")
        index = int(match.group("idx"))
        replicate = int(match.group("rep"))
        triple = (condition_id, index, replicate)
        dose_in_id = round(float(match.group("dose")), 4) if "dose" in match.groupdict() else None

        joined = None
        if dose_in_id is not None:
            joined = by4.get(triple + (dose_in_id,))
        if joined is None:
            joined = by3.get(triple)
        if joined is None:
            counters["join_missing"] += 1
            raise SystemExit(
                f"✗ {batch} 的 {row_id} 在 all_subjects.csv 里找不到对应行"
                f"（按（条件, 被试序号, 重复{', 剂量' if dose_in_id is not None else ''}）查）。"
            )
        counters["join_ok"] += 1

        json_path = resolve_json(joined["subject_json_path"], run_dir)
        if not json_path.is_file():
            counters["json_missing"] += 1
            raise SystemExit(f"✗ {row_id} 指向的 JSON 不存在：{json_path}")
        raw_bytes = json_path.read_bytes()
        record = json.loads(raw_bytes.decode("utf-8"))

        counters["json_ok"] += 1
        if str(record.get("session_id")) != str(joined["session_id"]):
            counters["session_mismatch"] += 1
            raise SystemExit(
                f"✗ {row_id} 的 session_id 对不上："
                f"csv={joined['session_id']} json={record.get('session_id')}"
            )

        # 回指到的那份记录的剂量必须与 row_id 自报的剂量一致 —— 这正是上面那个键存在的理由
        actual_dose = dose_of(record)
        if dose_in_id is not None and abs(actual_dose - dose_in_id) > 1e-9:
            counters["dose_mismatch"] += 1
            raise SystemExit(
                f"✗ {row_id} 的剂量段是 {dose_in_id:g}，但回指到的记录实际是 "
                f"{actual_dose:g}（{json_path.name}）—— 回指取错了记录。"
            )

        readings: dict[str, float | None] = {}
        for header, value in raw.items():
            channel = dim_of_header(header)
            if channel:
                readings[channel] = as_float(value)

        subjects.append(
            Subject(
                row_id=row_id,
                source_batch=batch,
                condition_id=condition_id,
                primitive_id=primitive_of(record),
                dose=actual_dose,
                replicate=replicate,
                subject_index=index,
                readings=readings,
                json_path=json_path,
                session_id=str(record.get("session_id")),
                provider_kind=str(joined.get("provider_kind") or ""),
                synthetic=str(joined.get("synthetic") or ""),
                theta_id=str(joined.get("theta_id") or ""),
                content_sha=hashlib.sha256(raw_bytes).hexdigest(),
                record=record,
            )
        )
    return subjects


def mark_reused(reference_rows: list[Subject], candidates: list[Subject]) -> int:
    """把 candidates 里与 reference_rows **是同一段会话**的行标出来。

    判据是 `(θ 指纹, 剂量, 重复, JSON 内容哈希)` 四样全同 —— 只比文件名不够（两个批次
    的目录不同），只比 session_id 也不够（要确认内容真的一样）。这样「复用数据不是独立重复」
    才落在数据上，而不是靠一句声明。
    """
    index = {
        (row.theta_id, round(row.dose, 4), row.replicate, row.content_sha): row.row_id
        for row in reference_rows
    }
    marked = 0
    for row in candidates:
        source = index.get((row.theta_id, round(row.dose, 4), row.replicate, row.content_sha))
        if source:
            row.reused = True
            row.reused_from = source
            marked += 1
    return marked


def load_reference_records() -> tuple[list[dict], list[dict]]:
    """返回 (官方 5 份, 全部 10 份参照)；只收真实 provider 的完整心智记录。"""
    official: list[dict] = []
    reference: list[dict] = []
    for directory in REFERENCE_DIRS:
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            params = record.get("parameters") or {}
            profile = record.get("profile") or {}
            if not (params.get("is_intact") and not params.get("active")):
                continue
            if float(params.get("generic_degradation") or 0.0) > 0:
                continue
            if str(profile.get("provider_kind")) != "llm" or profile.get("synthetic"):
                continue
            record["_path"] = str(path)
            reference.append(record)
            if directory == OFFICIAL_BASELINE_DIR:
                official.append(record)

    return official, reference


# ---------------------------------------------------------------------------
# 阶段 1 · 证据归属仪器（三问共用的量化基础）
# ---------------------------------------------------------------------------


def pair_turns(record: dict) -> list[dict]:
    """把逐轮对话配成 (提问轮 → 紧随其后的作答轮)。题/答是交替的，但允许连续提问。"""
    pairs: list[dict] = []
    pending: dict | None = None
    for turn in (record.get("dialogue") or {}).get("turns") or []:
        role = turn.get("role")
        if role == "examiner":
            pending = turn
        elif role == "brain" and pending is not None:
            answer = str(turn.get("content") or "")
            pairs.append({
                "focus": str(pending.get("focus") or ""),
                "probe_key": str(pending.get("probe_key") or ""),
                "question": str(pending.get("question") or pending.get("content") or ""),
                "answer": answer,
                "answer_chars": int(turn.get("chars") or len(answer)),
                "over_limit": bool(turn.get("over_limit")),
            })
            pending = None
    return pairs


def pairs_for_channel(pairs: list[dict], channel: str) -> list[dict]:
    return [p for p in pairs if p["focus"] == channel]


def channel_counts(record: dict) -> dict[str, int]:
    """该会话逐通道的提问轮数（以提问轮自报的 `focus` 为准）。"""
    counts: dict[str, int] = {channel: 0 for channel in CHANNELS}
    for turn in (record.get("dialogue") or {}).get("turns") or []:
        if turn.get("role") != "examiner":
            continue
        focus = str(turn.get("focus") or "")
        if focus in counts:
            counts[focus] += 1
    return counts


def task_evidence(record: dict, channel: str) -> list[dict]:
    """该通道下分析者记录的逐条任务测量。"""
    entry = ((record.get("analysis") or {}).get("dimensions") or {}).get(channel) or {}
    out: list[dict] = []
    for item in entry.get("task_measurements") or []:
        if isinstance(item, dict):
            out.append({
                "task": clean_text(item.get("task")),
                "measurement": clean_text(item.get("measurement")),
            })
        else:
            out.append({"task": clean_text(item), "measurement": ""})
    return out


def observations(record: dict, channel: str) -> list[str]:
    entry = ((record.get("analysis") or {}).get("dimensions") or {}).get(channel) or {}
    raw = entry.get("observations") or []
    if isinstance(raw, str):
        return [clean_text(part) for part in raw.split("；") if clean_text(part)]
    return [clean_text(item) for item in raw if clean_text(item)]


def attribute_evidence(record: dict, pairs: list[dict], *,
                       match_min: float = MATCH_MIN,
                       margin: float = MATCH_MARGIN) -> dict:
    """把分析者的每条证据反查回**它实际来自哪一问**。

    做法：拿 `task_measurements[].task` 的字符 2-gram 去覆盖匹配本次会话每个提问轮的
    问题文本（用覆盖率而不是 Jaccard —— 任务名是分析者对长问句的压缩，Jaccard 会被
    长问句稀释到很低）。置信判据是**两档**的，因为实测「最佳匹配得分不高、但次佳为 0」
    是常见且可靠的情形（分析者的任务名是压缩标签，绝对覆盖率天然偏低）：

    - 严格档：最佳 ≥ 0.34 且 ≥ 1.5 × 次佳；
    - 宽松档：最佳 ≥ 0.20 且 ≥ 2.0 × 次佳（次佳为 0 时只要最佳达标即可）。

    两档都不满足就记 `unattributed` —— **不硬塞**。

    返回 `{'rows': {维度 → [归属]}, 'scores': [...], 'unattributed': n, 'matched': n}`。
    """
    questions = [(index, bigrams(pair["question"]), pair) for index, pair in enumerate(pairs)]
    rows: dict[str, list[dict]] = {}
    scores: list[float] = []
    matched = 0
    unattributed = 0

    for channel in CHANNELS:
        entries: list[dict] = []
        for item in task_evidence(record, channel):
            needle = bigrams(item["task"])
            scored = sorted(
                ((coverage(needle, question_grams), pair) for _, question_grams, pair in questions),
                key=lambda pair_: pair_[0],
                reverse=True,
            )
            best = scored[0] if scored else (0.0, None)
            second = scored[1] if len(scored) > 1 else (0.0, None)
            scores.append(best[0])
            second_score = second[0] if second[1] is not None else 0.0
            confident = best[1] is not None and (
                (best[0] >= match_min and best[0] >= margin * second_score)
                or (best[0] >= 0.20 and best[0] >= 2.0 * second_score)
            )
            if confident:
                matched += 1
                entries.append({
                    "task": item["task"],
                    "measurement": item["measurement"],
                    "score": round(best[0], 4),
                    "matched_focus": best[1]["focus"] or "unattributed",
                    "matched_probe_key": best[1]["probe_key"],
                    "matched_question": best[1]["question"],
                    "matched_answer": best[1]["answer"],
                })
            else:
                unattributed += 1
                entries.append({
                    "task": item["task"],
                    "measurement": item["measurement"],
                    "score": round(best[0], 4),
                    "matched_focus": "unattributed",
                    "matched_probe_key": "",
                    "matched_question": "",
                    "matched_answer": "",
                })
        rows[channel] = entries

    return {
        "rows": rows,
        "scores": scores,
        "matched": matched,
        "unattributed": unattributed,
        "attribution_rate": round(matched / (matched + unattributed), 4) if (matched + unattributed) else 0.0,
        "score_median": round(statistics.median(scores), 4) if scores else 0.0,
    }


def own_channel_share(attribution: dict, channel: str) -> float:
    """该通道的证据里，有多少比例确实来自**它自己**的提问轮。"""
    entries = attribution["rows"].get(channel) or []
    if not entries:
        return 0.0
    own = sum(1 for e in entries if e["matched_focus"] == channel)
    return round(own / len(entries), 4)


def from_channel_share(attribution: dict, channel: str, source: str) -> float:
    """该通道的证据里，有多少比例来自 `source` 通道的提问轮（E3 用的就是它）。"""
    entries = attribution["rows"].get(channel) or []
    if not entries:
        return 0.0
    hits = sum(1 for e in entries if e["matched_focus"] == source)
    return round(hits / len(entries), 4)


# ---------------------------------------------------------------------------
# 阶段 1b · 客观转写标记（轴一）
# ---------------------------------------------------------------------------


def dialogue_markers(pairs: list[dict]) -> dict:
    """限定在某个通道的 Q→A 对上算的行为标记 —— 只用原始对话，不碰分析者的散文。

    `echo_*` 是与提问用词的重合（模仿 / 回声）；`self_repeat_*` 是与自己前 3 轮的重合
    （perseveration）；两者都用字符 2-gram Jaccard。
    """
    if not pairs:
        return {
            "n_pairs": 0, "answer_chars_mean": 0.0, "answer_chars_median": 0.0,
            "q_back_rate": 0.0, "echo_jaccard_mean": 0.0, "self_repeat_max_mean": 0.0,
            "dup_answer_rate": 0.0, "short_or_unknown_rate": 0.0, "over_limit_rate": 0.0,
        }
    answers = [p["answer"] for p in pairs]
    chars = [p["answer_chars"] for p in pairs]
    total = len(pairs)

    q_back = sum(
        1 for a in answers
        if "？" in a or "?" in a or a.rstrip().endswith(("吗", "呢"))
    )
    echo = [jaccard(bigrams(p["answer"]), bigrams(p["question"])) for p in pairs]
    self_repeat: list[float] = []
    for index, answer in enumerate(answers):
        previous = answers[max(0, index - 3):index]
        self_repeat.append(
            max(jaccard(bigrams(answer), bigrams(prev)) for prev in previous) if previous else 0.0
        )
    normalized = [norm_answer(a) for a in answers]
    duplicates = total - len({n for n in normalized if n})
    short = sum(
        1 for p in pairs
        if p["answer_chars"] <= 4
        or any(word in p["answer"] for word in ("不知道", "不晓得", "不清楚", "忘了"))
    )
    over = sum(1 for p in pairs if p["over_limit"])

    return {
        "n_pairs": total,
        "answer_chars_mean": round(statistics.fmean(chars), 2),
        "answer_chars_median": round(statistics.median(chars), 2),
        "q_back_rate": round(q_back / total, 4),
        "echo_jaccard_mean": round(statistics.fmean(echo), 4),
        "self_repeat_max_mean": round(statistics.fmean(self_repeat), 4),
        "dup_answer_rate": round(duplicates / total, 4),
        "short_or_unknown_rate": round(short / total, 4),
        "over_limit_rate": round(over / total, 4),
    }


MARKER_KEYS = (
    "answer_chars_mean", "q_back_rate", "echo_jaccard_mean",
    "self_repeat_max_mean", "dup_answer_rate", "short_or_unknown_rate", "over_limit_rate",
)


# ---------------------------------------------------------------------------
# 阶段 1c · 套话停用表（从语料自动导出，不手工写死）
# ---------------------------------------------------------------------------


def split_fragments(texts) -> list[str]:
    """把观察文本切成短语片段 —— 套话通常落在片段粒度而不是整句。"""
    out: list[str] = []
    for text in texts:
        for part in re.split(r"[，。；、！？：,;!?]", str(text or "")):
            part = clean_text(part)
            if len(part) >= 3:
                out.append(part)
    return out


def load_corpus_fragments(subjects: list[Subject], reference: list[dict]) -> tuple[dict, int]:
    """`片段 → 包含它的机制集合`，外加语料里的机制总数。

    **分母用「机制（primitive）数」而不是「会话数」。** 套话的风险是「在多个机制之间重复、
    于是把机制差别抹平」；一个片段在同一个 primitive 的 5 次重复里都出现，不足以说明它是套话。
    """
    index: dict[str, set] = defaultdict(set)
    mechanisms = {"__reference__"}
    for subject in subjects:
        label = subject.primitive_id or subject.condition_id
        mechanisms.add(label)
        for channel in CHANNELS:
            for fragment in split_fragments(observations(subject.record, channel)):
                index[fragment].add(label)
    for record in reference:
        for channel in CHANNELS:
            for fragment in split_fragments(observations(record, channel)):
                index[fragment].add("__reference__")
    return index, len(mechanisms)


def boilerplate_set(index: dict[str, set], n_mechanisms: int, rate: float) -> set[str]:
    """跨机制出现率 ≥ `rate` 的片段算套话（且至少落在 3 个机制里，避免小语料误判）。

    阈值是参数、不是拍脑袋的常量：报告里会同时给出**实测最高覆盖率**，这样「表很薄」
    与「表是空的」能被区分开 —— 前者是发现，后者是配置错误。
    """
    if n_mechanisms <= 0:
        return set()
    cutoff = max(3.0, rate * n_mechanisms)
    return {fragment for fragment, labels in index.items() if len(labels) >= cutoff}


def strip_boilerplate(texts, boilerplate: set[str]) -> list[str]:
    kept, _ = _split_and_filter(texts, boilerplate)
    return kept


def _split_and_filter(texts, boilerplate: set[str]) -> tuple[list[str], int]:
    """切成片段并剔除套话，同时**分别**数清保留与剔除的条数。

    两个数必须分开数：`observations` 是整句、切完之后条数会变多，用「前后长度相减」算
    剔除量会得到负数（整句 3 条 → 片段 6 条，其中剔除 1 条，相减得 -2）。所以要切一次、
    在同一个循环里计数。
    """
    kept: list[str] = []
    dropped = 0
    for text in texts:
        for part in re.split(r"[，。；、！？：,;!?]", str(text or "")):
            part = clean_text(part)
            if len(part) < 3:
                continue
            if part in boilerplate:
                dropped += 1
            else:
                kept.append(part)
    return kept, dropped


def evidence_text(record: dict, channel: str) -> str:
    """该通道的逐字证据（观察 + 任务测量）拼成一段可读文本。"""
    parts = list(observations(record, channel))
    for item in task_evidence(record, channel):
        parts.append(f"【任务】{item['task']} → 【实际表现】{item['measurement']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 阶段 1d · 置换检验（探索性）
# ---------------------------------------------------------------------------


def fingerprint_vector(marker: dict) -> list[float]:
    return [float(marker.get(key) or 0.0) for key in MARKER_KEYS]


def standardize(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        return []
    width = len(vectors[0])
    out = [[0.0] * width for _ in vectors]
    for column in range(width):
        values = [v[column] for v in vectors]
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        center = statistics.fmean(values)
        for row, vector in enumerate(vectors):
            out[row][column] = (vector[column] - center) / spread if spread > 1e-9 else 0.0
    return out


def _distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def separation_ratio(groups: dict[str, list[list[float]]]) -> float:
    """组间距离 / 组内距离。组 = 机制，组内点 = 各次重复的标记指纹。"""
    centroids: dict[str, list[float]] = {}
    within: list[float] = []
    for label, members in groups.items():
        if not members:
            continue
        width = len(members[0])
        centroids[label] = [
            statistics.fmean(member[column] for member in members) for column in range(width)
        ]
        within.extend(_distance(member, centroids[label]) for member in members)
    if len(centroids) < 2:
        return 0.0
    labels = sorted(centroids)
    between = [
        _distance(centroids[labels[i]], centroids[labels[j]])
        for i in range(len(labels)) for j in range(i + 1, len(labels))
    ]
    within_mean = statistics.fmean(within) if within else 0.0
    return round(statistics.fmean(between) / within_mean, 4) if within_mean > 1e-9 else 0.0


def permutation_p(groups: dict[str, list[list[float]]], observed: float, *,
                  draws: int = 2000, seed: int = 20260919) -> float:
    """机制标签可交换的零假设下，组间/组内比值的经验 p 值。**探索性**，n 很小。"""
    labels: list[str] = []
    points: list[list[float]] = []
    sizes: list[int] = []
    for label in sorted(groups):
        members = groups[label]
        sizes.append(len(members))
        labels.extend([label] * len(members))
        points.extend(members)
    if len(points) < 4 or len(set(labels)) < 2:
        return float("nan")
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        regrouped: dict[str, list[list[float]]] = defaultdict(list)
        for label, point in zip(shuffled, points):
            regrouped[label].append(point)
        if separation_ratio(regrouped) >= observed:
            hits += 1
    return round((hits + 1) / (draws + 1), 4)


# ---------------------------------------------------------------------------
# 个案选定（全部由 Fig. 2 产物驱动）
# ---------------------------------------------------------------------------


def read_prior_hypotheses(link_rows: list[dict]) -> dict[str, set[str]]:
    """**只用于个案排序偏好**的先验假设：`primitive_id → {维度}`。

    输入是 `results/data/prior_hypotheses.json` 里的机制—维度配对；只保留在当前
    `figure2_link_candidates.csv` 里真实存在的 id，避免写错的配对静默生效。
    这些假设**不参与任何数值计算**。
    """
    valid_primitives = {row["primitive_id"] for row in link_rows}
    valid_dimensions = {row["dimension"] for row in link_rows}

    out: dict[str, set[str]] = defaultdict(set)
    if not PRIOR_HYPOTHESES.is_file():
        return out
    payload = json.loads(PRIOR_HYPOTHESES.read_text(encoding="utf-8"))
    for item in payload.get("pairs") or []:
        primitive = str(item.get("primitive_id") or "")
        dimension = str(item.get("dimension") or "")
        if primitive in valid_primitives and dimension in valid_dimensions:
            out[primitive].add(dimension)
    return out


def select_q1_cases(link_rows: list[dict], hypotheses: dict[str, set[str]], *,
                    limit: int, tier: str) -> tuple[list[dict], dict]:
    """Q1：tier A 里选，跨通道且尽量不重复 primitive。"""
    pool = [r for r in link_rows if str(r.get("tier")) == tier]
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_dim[row["dimension"]].append(row)

    def rank(row: dict):
        return (
            str(row.get("verdict")) == "pass",
            row["primitive_id"] in hypotheses,
            float(row["delta"]),
        )

    best_per_dim: dict[str, dict] = {}
    for dimension, rows in by_dim.items():
        best_per_dim[dimension] = max(rows, key=rank)

    ordered = sorted(best_per_dim.items(), key=lambda kv: rank(kv[1]), reverse=True)
    selected: list[dict] = []
    used: set[str] = set()
    for dimension, row in ordered:
        if len(selected) >= limit:
            break
        if row["primitive_id"] in used:
            continue
        selected.append(row)
        used.add(row["primitive_id"])
    # 凑不满就允许复用 primitive（仍不重复通道）
    for dimension, row in ordered:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)

    availability = {
        "tier": tier,
        "n_rows_in_pool": len(pool),
        "per_dimension": {channel: len(by_dim.get(channel, [])) for channel in PHENO},
        "dimensions_without_case": [
            channel for channel in PHENO if channel not in {r["dimension"] for r in selected}
        ],
    }
    return selected[:limit], availability


def select_q2_cases(link_rows: list[dict], metrics: dict, hypotheses: dict[str, set[str]], *,
                    limit: int, tolerance: float, dose: float) -> list[dict]:
    """Q2：Fig.2 panel B 的「每通道最大 3 个」∪ 研究者假设，取组内 Δ 配对的一组。"""
    top3 = (metrics.get("panel_b", {}).get("per_dose", {}).get(str(dose), {})
            .get("top3_per_dimension", {}) or {})
    names = {row["primitive"]: row["primitive_id"] for row in link_rows}
    delta_at = {
        (row["primitive_id"], row["dimension"]): float(row["delta"])
        for row in link_rows if abs(float(row["dose"]) - dose) < 1e-9
    }

    cases: list[dict] = []
    for channel in PHENO:
        members: dict[str, float] = {}
        for item in top3.get(channel, []) or []:
            primitive = names.get(item.get("primitive")) or item.get("primitive_id") or ""
            if primitive:
                members[primitive] = float(item["delta"])
        for primitive, dims in hypotheses.items():
            if channel in dims:
                value = delta_at.get((primitive, channel))
                if value is not None:
                    members[primitive] = value
        if len(members) < 2:
            continue

        # 组内最大 Δ 配对的子集：对每个起点取能容纳的最大成员集，再按假设命中数定胜负
        ordered = sorted(members.items(), key=lambda kv: kv[1], reverse=True)
        candidates: list[list[tuple[str, float]]] = []
        for index in range(len(ordered)):
            group = [ordered[index]]
            for other in ordered[index + 1:]:
                if ordered[index][1] - other[1] <= tolerance:
                    group.append(other)
            if len(group) >= 2:
                candidates.append(group)
        if not candidates:
            continue
        size = max(len(g) for g in candidates)
        finals = [g for g in candidates if len(g) == size]
        finals.sort(
            key=lambda g: (
                sum(1 for name, _ in g if channel in hypotheses.get(name, set())),
                statistics.fmean(delta for _, delta in g),
            ),
            reverse=True,
        )
        group = finals[0]
        dropped = [
            (name, delta) for name, delta in ordered
            if name not in {n for n, _ in group}
            and name in hypotheses and channel in hypotheses[name]
        ]
        cases.append({
            "dimension": channel,
            "dose": dose,
            "members": [name for name, _ in group],
            "member_deltas": {name: delta for name, delta in group},
            "span": round(max(d for _, d in group) - min(d for _, d in group), 4),
            "n_hypotheses": sum(1 for name, _ in group if channel in hypotheses.get(name, set())),
            "mean_delta": round(statistics.fmean(d for _, d in group), 4),
            "hypothesis_dropped_for_gap": dropped,
        })

    cases.sort(key=lambda c: (c["n_hypotheses"], c["mean_delta"]), reverse=True)
    return cases[:limit]


def dose_ladders(summary_rows: list[dict]) -> dict[str, list[dict]]:
    """按 primitive 归拢各档的逐维 delta / 对照 delta / 主导维度。"""
    ladders: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        primitive = str(row.get("primitive_id") or "")
        deltas = {channel: as_float(row.get(f"{channel}_delta")) for channel in CHANNELS}
        means = {channel: as_float(row.get(f"{channel}_mean")) for channel in CHANNELS}
        phenotype = {
            channel: deltas[channel] for channel in PHENO
            if deltas.get(channel) is not None
        }
        lowest = min(phenotype.items(), key=lambda kv: (kv[1], kv[0]))[0] if phenotype else ""
        ordered = sorted(phenotype.items(), key=lambda kv: (kv[1], kv[0]))
        concentration = (
            round(ordered[1][1] - ordered[0][1], 4) if len(ordered) > 1 else 0.0
        )
        ladders[primitive].append({
            "dose": float(row["dose"]),
            "n_replicates": int(row.get("n_replicates") or 0),
            "deltas": deltas,
            "means": means,
            "dominant_dim": str(row.get("dominant_dim") or ""),
            "lowest_dim": lowest,
            "concentration": concentration,
            "control_delta": deltas.get(CONTROL_DIM),
            "control_mean": means.get(CONTROL_DIM),
            # 「只看幅度」的口径（Q3 原来的读法）
            "n_affected": sum(1 for channel in PHENO if (deltas.get(channel) or 0) <= -HIT),
            "n_affected_strict": sum(
                1 for channel in PHENO if (deltas.get(channel) or 0) <= -HIT_STRICT
            ),
            "mean_delta": mean_or_blank([deltas.get(channel) for channel in PHENO]),
            "specificity_verdict": str(row.get("specificity_verdict") or ""),
        })
    for rows in ladders.values():
        rows.sort(key=lambda item: item["dose"], reverse=True)
    return ladders


def select_q3_cases(ladders: dict[str, list[dict]], metrics: dict, *, limit: int) -> list[dict]:
    """Q3：在 5 个可做阶梯的 primitive 里，按「最能区分四条解释」各挑一个 signature。"""
    candidates = list((metrics.get("panel_f", {}).get("per_primitive") or {}).keys())
    rows: list[dict] = []
    for primitive in candidates:
        steps = ladders.get(primitive) or []
        if len(steps) < 3:
            continue
        switches = sum(
            1 for a, b in zip(steps, steps[1:])
            if (a["dominant_dim"] or a["lowest_dim"])
            and (a["dominant_dim"] or a["lowest_dim"]) != (b["dominant_dim"] or b["lowest_dim"])
        )
        rows.append({
            "primitive_id": primitive,
            "n_doses": len(steps),
            "lowest_channel_switches": switches,
            "max_concentration": round(max(step["concentration"] for step in steps), 4),
            "control_drop": round(
                max((step["control_delta"] or 0.0) for step in steps) * -1, 4
            ),
            "worst_control_delta": round(
                min((step["control_delta"] or 0.0) for step in steps), 4
            ),
            "n_affected_at_lightest": steps[0]["n_affected"],
            "n_affected_at_heaviest": steps[-1]["n_affected"],
        })
    if not rows:
        return []

    chosen: list[dict] = []
    rules = [
        ("s1 · 主导通道跨档会换（受影响集合本身在变）", lambda r: r["lowest_channel_switches"]),
        ("s2 · 剖面在某档最集中（集中度随剂量的轨迹）", lambda r: r["max_concentration"]),
        ("s3 · 最重档对照通道掉得最多（非特异通用退化）", lambda r: -r["worst_control_delta"]),
    ]
    for label, key in rules:
        if len(chosen) >= limit:
            break
        pool = [r for r in rows if r["primitive_id"] not in {c["primitive_id"] for c in chosen}]
        if not pool:
            break
        pick = max(pool, key=key)
        pick = dict(pick)
        pick["signature"] = label
        pick["signature_value"] = round(float(key(pick)), 4)
        chosen.append(pick)
    return chosen


# ---------------------------------------------------------------------------
# 阶段 2 · Q1
# ---------------------------------------------------------------------------


def probe_slot_contrast(subject_pairs: list[dict], reference_records: list[dict], *,
                        max_slots: int = 3) -> tuple[list[dict], int]:
    """`probe_key` 槽位恰好相同的配对 —— **同槽位、非同一句**，必须照此措辞。"""
    subject_slots: dict[str, dict] = {}
    for pair in subject_pairs:
        if pair["probe_key"] and pair["probe_key"] not in subject_slots:
            subject_slots[pair["probe_key"]] = pair
    reference_slots: dict[str, dict] = {}
    for record in reference_records:
        for pair in pair_turns(record):
            if pair["probe_key"] and pair["probe_key"] not in reference_slots:
                reference_slots[pair["probe_key"]] = pair

    shared = sorted(set(subject_slots) & set(reference_slots))
    out: list[dict] = []
    for key in shared[:max_slots]:
        impaired = subject_slots[key]
        intact = reference_slots[key]
        out.append({
            "probe_key": key,
            "intact_question": intact["question"],
            "intact_answer": intact["answer"],
            "impaired_question": impaired["question"],
            "impaired_answer": impaired["answer"],
            "wording_identical": intact["question"] == impaired["question"],
        })
    return out, len(shared)


def build_q1_rows(baseline: dict, reference: list[dict], cases: list[dict],
                  subject_index: dict[tuple[str, str, float], list[Subject]],
                  attributions: dict[str, dict], summary_r2: list[dict], **_) -> list[dict]:
    """Q1 的数字分三层报，不能混成一个「Δ」：

    - `published_*`：发布表（`summary_by_cell`）里那个数，也就是 Fig.2 用的数；
    - `delta_recomputed_result2`：只用 **result_2 的行**重算的 Δ —— 用来证明「row_id 回指取对了人」，
      应与 published 逐位相同；
    - `pooled_*`：把**两个批次同剂量的全部会话**合起来算的均值。5 档批次在 0.4/0.8 除复用外还补跑了
      新的重复，所以池子里的会话可能比发布表多。多出来的样本更有信息量，但它与发布表的 n 不同，
      **不能拿它与发布表的 Δ 对账** —— 否则会把「样本更多」误读成「数字对不上」。
    """
    published = {(str(r.get("primitive_id")), float(r.get("dose"))): r for r in summary_r2}
    rows: list[dict] = []
    for rank, case in enumerate(cases, start=1):
        dimension = case["dimension"]
        primitive = case["primitive_id"]
        dose = float(case["dose"])
        subjects = list(subject_index.get((primitive, dimension, dose), [])) or \
            list(subject_index.get((primitive, "", dose), []))
        if not subjects:
            continue
        r2_only = [s for s in subjects if s.source_batch == "result_2"]
        cell = published.get((primitive, dose), {})
        perturbation = [s.value(dimension) for s in subjects if s.value(dimension) is not None]
        pooled_mean = mean_or_blank(perturbation)
        r2_values = [s.value(dimension) for s in r2_only if s.value(dimension) is not None]
        published_mean = as_float(cell.get(f"{dimension}_mean"))
        published_delta = as_float(cell.get(f"{dimension}_delta"))

        reference_pairs = [
            pair for record in reference for pair in pairs_for_channel(pair_turns(record), dimension)
        ]
        reference_marker = dialogue_markers(reference_pairs)
        reference_readings = [
            (record.get("analysis") or {}).get("readings", {}).get(dimension)
            for record in reference
        ]
        reference_readings = [float(v) for v in reference_readings if v is not None]

        impaired_pairs: list[dict] = []
        for subject in subjects:
            impaired_pairs.extend(pairs_for_channel(subject.pairs, dimension))
        impaired_marker = dialogue_markers(impaired_pairs)

        slots, n_shared = probe_slot_contrast(impaired_pairs, reference)
        attribution = attributions.get(subjects[0].session_id, {})
        own_share = own_channel_share(attribution, dimension) if attribution else ""

        evidence_intact = "\n---\n".join(
            evidence_text(record, dimension) for record in reference[:2]
        )
        evidence_impaired = "\n---\n".join(
            evidence_text(subject.record, dimension) for subject in subjects
        )

        rows.append({
            "case": f"Q1-{rank}",
            "dimension": dimension,
            "dimension_zh": channel_label(dimension),
            "primitive_id": primitive,
            "dose": dose,
            "source": f"figure2_link_candidates.csv tier {case.get('tier')}",
            "selection_reason": (
                f"tier {case.get('tier')}；verdict={case.get('verdict')}；"
                f"Δ={case.get('delta')}；q_pooled={case.get('q_pooled')}"
                + ("；命中先验假设" if case.get("from_hypothesis") else "")
            ),
            "baseline_reading": baseline.get(dimension, ""),
            "published_mean": "" if published_mean is None else published_mean,
            "published_delta": "" if published_delta is None else published_delta,
            "n_published": int(cell.get("n_replicates") or 0),
            "delta_recomputed_result2": (
                round(statistics.fmean(r2_values) - float(baseline[dimension]), 4)
                if r2_values and dimension in baseline else ""
            ),
            "pooled_mean": pooled_mean,
            "delta_pooled": round(float(pooled_mean) - float(baseline[dimension]), 4)
            if isinstance(pooled_mean, float) and dimension in baseline else "",
            "n_pooled": len(subjects),
            "n_result2": len(r2_only),
            "reference_reading_mean": mean_or_blank(reference_readings),
            "rep_range": round(max(perturbation) - min(perturbation), 4) if len(perturbation) > 1 else "",
            "evidence_axis1_intact": json.dumps(reference_marker, ensure_ascii=False),
            "evidence_axis1_impaired": json.dumps(impaired_marker, ensure_ascii=False),
            "axis1_verdict": describe_axis1(reference_marker, impaired_marker),
            "probe_key_shared": n_shared,
            "probe_key_wording_identical": (
                sum(1 for s in slots if s["wording_identical"]) if slots else 0
            ),
            "slot_contrast_note": (
                "同 probe_key 槽位、措辞由提问者即时生成 —— 不是同一句话" if n_shared else
                "本通道与参照没有共同的 probe_key 槽位，只能做通道级对比"
            ),
            "own_channel_evidence_share": own_share,
            "evidence_intact": evidence_intact,
            "evidence_impaired": evidence_impaired,
            "slot_contrast": json.dumps(slots, ensure_ascii=False),
        })
    return rows


def describe_axis1(intact: dict, impaired: dict) -> str:
    """把轴一的两组标记压成一句可读的判定 —— 只在差异方向明确时才下断言。"""
    parts: list[str] = []
    if impaired["n_pairs"] == 0 or intact["n_pairs"] == 0:
        return "样本不足，不作判定"
    shorter = impaired["answer_chars_mean"] - intact["answer_chars_mean"]
    if abs(shorter) >= max(3.0, 0.25 * intact["answer_chars_mean"]):
        parts.append(f"应答长度{'缩短' if shorter < 0 else '变长'} {abs(shorter):.1f} 字")
    for key, label in (
        ("dup_answer_rate", "逐字重复应答占比"),
        ("self_repeat_max_mean", "自我重复"),
        ("short_or_unknown_rate", "极短/未作答占比"),
        ("q_back_rate", "反问率"),
    ):
        gap = impaired[key] - intact[key]
        if abs(gap) >= 0.08:
            parts.append(f"{label}{'升' if gap > 0 else '降'} {abs(gap):.2f}")
    return "；".join(parts) if parts else "轴一未读出方向明确的差异"


# ---------------------------------------------------------------------------
# 阶段 3 · Q2
# ---------------------------------------------------------------------------


def build_q2_rows(cases: list[dict], subject_index: dict[tuple[str, str, float], list[Subject]],
                  boilerplate: set[str], *, tolerance: float, seed: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    group_stats: list[dict] = []

    for rank, case in enumerate(cases, start=1):
        dimension = case["dimension"]
        dose = float(case["dose"])
        members = case["members"]

        fingerprints: dict[str, list[list[float]]] = {}
        member_evidence: dict[str, list[str]] = {}
        member_markers: dict[str, dict] = {}
        member_deltas: dict[str, float] = {}
        for primitive in members:
            subjects = subject_index.get((primitive, dimension, dose), []) or \
                subject_index.get((primitive, "", dose), [])
            if not subjects:
                continue
            fingerprints[primitive] = []
            member_evidence[primitive] = []
            for subject in subjects:
                pairs = pairs_for_channel(subject.pairs, dimension)
                fingerprints[primitive].append(fingerprint_vector(dialogue_markers(pairs)))
                member_evidence[primitive].extend(observations(subject.record, dimension))
            member_markers[primitive] = dialogue_markers(
                [p for s in subjects for p in pairs_for_channel(s.pairs, dimension)]
            )
            # Δ 一律用发布表里那个数（`member_deltas` 取自 link_candidates，与 Fig.2 同源），
            # 不在这里自己重算 —— 否则本文件与 Fig.2 的数字可能悄悄分叉。
            member_deltas[primitive] = case["member_deltas"].get(primitive, "")

        if len(fingerprints) < 2:
            continue

        pooled = [vector for vectors in fingerprints.values() for vector in vectors]
        scaled = standardize(pooled)
        groups: dict[str, list[list[float]]] = {}
        cursor = 0
        for primitive in sorted(fingerprints):
            size = len(fingerprints[primitive])
            groups[primitive] = scaled[cursor:cursor + size]
            cursor += size

        ratio = separation_ratio(groups)
        p_value = permutation_p(groups, ratio, seed=seed)

        # 轴二：原始 / 剔除套话后的文本重合 + 各自独有短语
        pair_similarity: list[dict] = []
        labels = sorted(member_evidence)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                left, right = labels[i], labels[j]
                raw = jaccard(
                    bigrams("".join(member_evidence[left])),
                    bigrams("".join(member_evidence[right])),
                )
                strip_left = strip_boilerplate(member_evidence[left], boilerplate)
                strip_right = strip_boilerplate(member_evidence[right], boilerplate)
                _, dropped = _split_and_filter(
                    member_evidence[left] + member_evidence[right], boilerplate)
                stripped = jaccard(
                    bigrams("".join(strip_left)), bigrams("".join(strip_right))
                )
                only_left = sorted({f for f in strip_left if f not in set(strip_right)})
                only_right = sorted({f for f in strip_right if f not in set(strip_left)})
                pair_similarity.append({
                    "pair": [left, right],
                    "raw_similarity": round(raw, 4),
                    "stripped_similarity": round(stripped, 4),
                    "boilerplate_removed": dropped,
                    "distinctive": {left: only_left[:6], right: only_right[:6]},
                })

        axis1 = "机制可区分" if ratio >= 1.5 else ("机制难以区分" if ratio < 1.0 else "介于两者之间")
        axis2 = "文本可区分" if all(
            item["stripped_similarity"] < 0.5 for item in pair_similarity
        ) else "文本高度重合"
        group_stats.append({
            "case": f"Q2-{rank}",
            "dimension": dimension,
            "dimension_zh": channel_label(dimension),
            "dose": dose,
            "members": members,
            "delta_span": case["span"],
            "match_tolerance": tolerance,
            "separation_ratio": ratio,
            "permutation_p": p_value,
            "axis1_verdict": axis1,
            "axis2_verdict": axis2,
            "two_axis_agreement": (
                "两轴一致：可区分" if axis1 == "机制可区分" and axis2 == "文本可区分"
                else "两轴一致：不可区分" if axis1 == "机制难以区分" and axis2 == "文本高度重合"
                else "两轴不一致 → 建议用 --llm-contrast 判读"
            ),
            "hypothesis_dropped_for_gap": case["hypothesis_dropped_for_gap"],
        })

        for primitive in members:
            if primitive not in fingerprints:
                continue
            markers = member_markers[primitive]
            others = [item for item in pair_similarity if primitive in item["pair"]]
            distinctive: list[str] = []
            for item in others:
                distinctive.extend(item["distinctive"].get(primitive, []))
            rows.append({
                "case": f"Q2-{rank}",
                "dimension": dimension,
                "dimension_zh": channel_label(dimension),
                "dose": dose,
                "primitive_id": primitive,
                "delta_published": case["member_deltas"].get(primitive, ""),
                "n_replicates": len(fingerprints[primitive]),
                "axis1_separation_ratio": ratio,
                "axis1_permutation_p": p_value,
                "axis1_verdict": axis1,
                "axis2_verdict": axis2,
                "text_raw_similarity_mean": mean_or_blank(
                    [item["raw_similarity"] for item in others]
                ),
                "text_stripped_similarity_mean": mean_or_blank(
                    [item["stripped_similarity"] for item in others]
                ),
                "distinctive_phrases": "；".join(list(dict.fromkeys(distinctive))[:8]),
                "markers": json.dumps(markers, ensure_ascii=False),
                "observations": "\n".join(
                    dict.fromkeys(member_evidence[primitive])
                ),
            })
    return rows, group_stats


# ---------------------------------------------------------------------------
# 阶段 4 · Q3
# ---------------------------------------------------------------------------


def premise_verdict(rows: list[dict]) -> str:
    """Q3 的前提（「轻度只坏一个表型、严重影响多个」）是否成立。

    判据用**幅度**：`Δ ≤ −0.15` 记一个受影响维度，`Δ ≤ −0.30` 记一个深度受影响维度。
    比较最轻档与最重档：两个计数都增长才算「成立」；只有一个增长算「部分成立」；
    都不增长算「不成立」。显著性检验已在 Fig.2 的候选表中单独报告，这里不重复叠加。
    """
    if len(rows) < 2:
        return "档位不足，不作判定"
    lightest, heaviest = rows[0], rows[-1]
    light = lightest["n_affected_015"]
    heavy = heaviest["n_affected_015"]
    light_deep = lightest["n_affected_030"]
    heavy_deep = heaviest["n_affected_030"]

    if heavy > light and heavy_deep > light_deep:
        return (f"前提成立：受影响维度数随削弱加深从 {light} 升到 {heavy}"
                f"（Δ≤−0.30 的从 {light_deep} 升到 {heavy_deep}）")
    if heavy > light:
        return (f"前提**部分成立**：受影响维度数从 {light} 升到 {heavy}，"
                f"但 Δ≤−0.30 的没有增加（{light_deep}→{heavy_deep}）")
    if heavy_deep > light_deep:
        return (f"前提**部分成立**：受影响维度数在轻度档就已到 {light}（六维里 {light} 个），"
                f"随削弱加深基本不变（→{heavy}）；变的是**深度**：Δ≤−0.30 的从 {light_deep} "
                f"升到 {heavy_deep}。所以这条机制的表现是「同一个宽集合整体下移」，"
                "不是「从 1 个扩到多个」")
    return (f"前提**不成立**：受影响维度数（{light}→{heavy}）与深度受影响维度数"
            f"（{light_deep}→{heavy_deep}）都不随削弱加深而增加。")


def build_q3_rows(case: dict, ladder: list[dict],
                  subject_index: dict[tuple[str, str, float], list[Subject]],
                  attributions: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for step in ladder:
        dose = step["dose"]
        dominant = step["dominant_dim"] or step["lowest_dim"]
        subjects = [
            s for s in subject_index.get((case["primitive_id"], dominant, dose), [])
        ] or [s for s in subject_index.get((case["primitive_id"], "", dose), [])]

        spillover = [
            channel for channel in PHENO
            if channel != dominant and (step["deltas"].get(channel) or 0) <= -HIT
        ]
        shared: list[float] = []
        own: list[float] = []
        for subject in subjects:
            attribution = attributions.get(subject.session_id)
            if not attribution:
                continue
            for channel in spillover:
                shared.append(from_channel_share(attribution, channel, dominant))
                own.append(own_channel_share(attribution, channel))

        probed = 0
        if subjects:
            counts = channel_counts(subjects[0].record)
            probed = sum(1 for channel in CHANNELS if counts.get(channel))

        control_delta = step["control_delta"]
        rows.append({
            "case": case["signature"].split(" ·")[0],
            "primitive_id": case["primitive_id"],
            "dose": dose,
            "n_replicates": step["n_replicates"],
            "n_affected_015": step["n_affected"],
            "n_affected_030": step["n_affected_strict"],
            "dominant_dim": dominant,
            "dominant_dim_zh": channel_label(dominant),
            "second_dim": next(
                (channel for channel, _ in sorted(
                    ((c, step["deltas"].get(c) or 0) for c in PHENO if c != dominant),
                    key=lambda kv: (kv[1], kv[0]),
                )), ""),
            "concentration": step["concentration"],
            "mean_delta_six_dims": step["mean_delta"],
            "control_delta": control_delta,
            "n_channels_probed": probed,
            "specificity_verdict": step["specificity_verdict"],
            **{f"delta_{channel}": step["deltas"].get(channel, "") for channel in PHENO},
            "E1_control_tolerance_exceeded": (
                "是" if isinstance(control_delta, float) and control_delta <= -HIT else "否"
            ),
            "E2_concentration": step["concentration"],
            "E3_spillover_evidence_from_dominant": mean_or_blank(shared),
            "E4_own_channel_evidence_share": mean_or_blank(own),
            "spillover_dims": "；".join(channel_label(c) for c in spillover) or "无",
            "reading": explain_dose_step(step, dominant, shared, own),
        })
    return rows


def explain_dose_step(step: dict, dominant: str, shared: list[float], own: list[float]) -> str:
    """把四种解释的判定压成一句（只在有信号时下断言）。"""
    parts: list[str] = []
    control = step["control_delta"]
    if isinstance(control, float) and control <= -HIT:
        parts.append(f"E1 成立：对照通道也掉 {control:+.2f}，该档的分散不能全归因于这个机制")
    else:
        parts.append("E1 未触发：对照通道在容差内")
    parts.append(f"E2 集中度 {step['concentration']:+.2f}")
    if shared:
        share = statistics.fmean(shared)
        if share >= 0.34:
            parts.append(f"E3 有信号：溢出通道的证据有 {share:.2f} 来自主导通道的问句（测量层溢出）")
        else:
            parts.append(f"E3 未触发：溢出通道的证据只有 {share:.2f} 来自主导通道")
    else:
        parts.append("E3 无法判定：该档没有可归属的溢出通道证据")
    if own:
        share = statistics.fmean(own)
        parts.append(f"E4 候选：溢出通道自有证据占比 {share:.2f}")
    return "；".join(parts)


# ---------------------------------------------------------------------------
# 阶段 5 · 可选 LLM 判读（只总结已记录的对话，不对心智做探查）
# ---------------------------------------------------------------------------


CONTRAST_SYSTEM = (
    "你是在读两份**已经记录在案的访谈转写**。两份转写来自两个不同的受访者，"
    "问的是同一个观察目标。请只依据转写本身作答：\n"
    "- 两位受访者在行为表现上**是否可以被区分开**？\n"
    "- 如果能，共同点是什么、各自独有的表现是什么？\n"
    "- 差别主要是**种类上的**（不同的行为模式）还是**程度上的**（同一种模式轻一点重一点）？\n"
    "不要推测受访者的诊断、不要提及任何机制名称、不要给建议。"
    "只输出一个 JSON 对象，键为：distinguishable（yes/no/partial）、shared_features（字符串数组）、"
    "unique_to_A（字符串数组）、unique_to_B（字符串数组）、"
    "kind_vs_degree（kind/degree/mixed）、confidence（0-1 的小数）、note（一句话）。"
)


def transcript_excerpt(pairs: list[dict], limit: int = 8) -> str:
    lines = []
    for pair in pairs[:limit]:
        lines.append(f"问：{pair['question']}\n答：{pair['answer']}")
    return "\n".join(lines)


def run_llm_contrast(plan: list[dict], *, allow_mock: bool, temperature: float,
                     counters: dict) -> list[dict]:
    """只发 `role="analyst"`，消息体只由**已记录 JSON** 拼成 —— 不探查心智。"""
    try:
        from pacs.config import Settings
        from pacs.llm.parse import parse_analyst_json
        from pacs.llm.provider import ChatMessage, get_provider
    except ImportError as error:                    # pragma: no cover
        raise SystemExit(f"✗ 无法导入 pacs 的 LLM 层：{error}")

    settings = Settings.load(base_dir=PACS)
    provider = get_provider(settings.llm)
    if str(getattr(provider, "name", "")) == "mock" and not allow_mock:
        raise SystemExit(
            "✗ config.json 里选中的是 mock provider。真判读请配置真实 provider，"
            "或显式加 --allow-mock（产物会被标成 mock）。"
        )
    model_id = f"{provider.name}:{provider.model}"
    out: list[dict] = []
    for index, item in enumerate(plan, start=1):
        user = (
            f"观察目标：{item['dimension_zh']}\n\n"
            f"【受访者A 的转写】\n{item['transcript_a']}\n\n"
            f"【受访者B 的转写】\n{item['transcript_b']}"
        )
        counters["llm_calls"] += 1
        text = provider.chat(
            [ChatMessage(role="user", content=CONTRAST_SYSTEM + "\n\n" + user)],
            role="analyst", temperature=temperature,
        )
        payload, error = parse_analyst_json(text)
        out.append({
            "comparison": item["comparison"],
            "dimension": item["dimension"],
            "anonymized": True,
            "layer": "post_hoc_interpretation",
            "model_id": model_id,
            "provider_kind": "mock" if str(getattr(provider, "name", "")) == "mock" else "llm",
            "parse_error": error,
            "judgement": payload,
            "raw_response": text,
            "prompt": user,
            "revealed_after_judgement": item.get("reveal") or {},
            "no_writeback": NO_WRITEBACK,
        })
    return out


def build_contrast_plan(groups: list[dict], subject_index, *, tolerance: float,
                        max_pairs: int) -> list[dict]:
    """列出「必要时」值得判读的配对：两轴结论不一致、或轴一落在难以区分的区间。"""
    plan: list[dict] = []
    for group in groups:
        if group["two_axis_agreement"].startswith("两轴一致：可区分"):
            continue
        dimension = group["dimension"]
        dose = float(group["dose"])
        members = [m for m in group["members"]]
        if len(members) < 2:
            continue
        left = subject_index.get((members[0], dimension, dose), []) or \
            subject_index.get((members[0], "", dose), [])
        right = subject_index.get((members[1], dimension, dose), []) or \
            subject_index.get((members[1], "", dose), [])
        if not left or not right:
            continue
        plan.append({
            "comparison": f"{group['case']} · {dimension} @ dose {dose:g}",
            "dimension": dimension,
            "dimension_zh": channel_label(dimension),
            "reason": group["two_axis_agreement"],
            "delta_span": group["delta_span"],
            "transcript_a": transcript_excerpt(
                [p for s in left for p in pairs_for_channel(s.pairs, dimension)]
            ),
            "transcript_b": transcript_excerpt(
                [p for s in right for p in pairs_for_channel(s.pairs, dimension)]
            ),
            "reveal": {"A": members[0], "B": members[1]},
        })
        if len(plan) >= max_pairs:
            return plan
    return plan


# ---------------------------------------------------------------------------
# 阶段 6 · 报告
# ---------------------------------------------------------------------------


def render_report(context: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# 对话探查证据个案分析（对话探查过程回读）")
    add("")
    add("本文件由 `results/scripts/analyze_dialogue_evidence.py` 生成，数据源是 "
        "`result_2_univariate.xlsx` 与 `result_4_does_response.xlsx` 以及它们各自回指的"
        "**逐被试 JSON**（`dialogue.turns` 逐轮原样保存，`analysis.dimensions` 逐维保存）。")
    add("")
    add(f"> {NO_WRITEBACK}")
    add("")

    add("## 0. 数据、口径与闸门")
    add("")
    add("### 0.1 来源与回指")
    add("")
    add("| 批次 | 发布表 | 行数 | 逐被试 JSON | 去重的复用行 |")
    add("|---|---|---|---|---|")
    for item in context["sources"]:
        add(f"| {item['batch']} | `{item['xlsx']}` | {item['rows']} | "
            f"`{item['run_dir']}` | {item['reused']} |")
    add("")
    add("闸门（全过才继续）："
        f"row_id 解析 {context['gate']['row_id_unparsed']} 失败；"
        f"回指成功 {context['gate']['join_ok']}；"
        f"session_id 不符 {context['gate']['session_mismatch']}；"
        f"JSON 缺失 {context['gate']['json_missing']}。")
    add("")

    add("### 0.2 完整心智参照")
    add("")
    add(f"- **数值 Δ** 沿用官方 {context['reference']['n_official']} 份"
        f"（`{context['reference']['official_dir']}`），与 `figure2_metrics.json` 的 "
        "`baseline` 逐位相同（脚本内断言）。")
    add(f"- **行为参照**扩到 {context['reference']['n_reference']} 份"
        "（另加 `fig1_realcheck/axis1` 同 θ 同协议的记录）。")
    add("")
    add(f"**回指对账**：只用 result_2 的行从逐被试 JSON 重算的 Δ 与发布表的 Δ 逐位相同"
        f"（{context['delta_check']['n']} 例，最大偏差 "
        f"{context['delta_check']['worst_gap']}）—— 这证明 `row_id` 回指取到的是发布表用的那批被试。")
    add("")
    add("| 通道 | 官方基线读数 | 参照读数均值 |")
    add("|---|---|---|")
    for channel in CHANNELS:
        add(f"| {channel_label(channel)} (`{channel}`) | {context['baseline'].get(channel, '')} | "
            f"{context['reference']['reading_mean'].get(channel, '')} |")
    add("")
    add("### 0.3 个案选定")
    add("")
    add("个案**由 Fig. 2 的既有产物选定**，不手工挑；每条都写明选自哪个产物、命中哪条规则。")
    add("")
    add("| 个案 | 问题 | 选自 | 命中规则 |")
    add("|---|---|---|---|")
    for item in context["case_selection"]:
        add(f"| {item['case']} | {item['question']} | {item['source']} | {item['rule']} |")
    add("")

    add("### 0.4 两个必须一起读的口径")
    add("")
    add("**① 不存在逐题配对。** 提问者是 LLM 即时生成措辞，`probe_key` 只是题池槽位。"
        "实测同一个 `probe_key` 下的问题文本在各次运行间**零重合**。因此：")
    add("")
    add("| 个案通道 | 与参照共享的 `probe_key` 槽位数 | 其中措辞相同的 |")
    add("|---|---|---|")
    for row in context["q1_rows"]:
        add(f"| {row['dimension_zh']} | {row['probe_key_shared']} | "
            f"{row['probe_key_wording_identical']} |")
    add("")
    add("**② 完整心智参照是同一个 θ 的 10 段独立会话**，不是 10 个个体。")
    add("")
    add("### 0.4b 两轴的判定门槛（写死在这里，避免读者猜）")
    add("")
    add("**轴一**（客观转写标记，7 项：应答长度均值、中位、反问率、复述提问用词、自我重复、"
        "逐字重复占比、极短/未作答占比）把各机制的重复指纹标准化后，算「组间距离 / 组内距离」：")
    add("")
    add("| 比值 | 轴一判定 |")
    add("|---|---|")
    add("| ≥ 1.5 | 机制可区分 |")
    add("| 1.0 – 1.5 | 介于两者之间 |")
    add("| < 1.0 | 机制难以区分 |")
    add("")
    add("**轴二**：剔除套话后，机制两两之间的字符 2-gram Jaccard 全部 < 0.5 记为「文本可区分」，"
        "否则「文本高度重合」。")
    add("")
    add("**合取规则**：两轴都判可区分才写「两轴一致：可区分」；两轴都判不可区分才写"
        "「两轴一致：不可区分」；其余一律写「两轴不一致 → 建议用 `--llm-contrast` 判读」，"
        "**不强行给结论**。")
    add("")
    add("### 0.4c 产物构成")
    add("")
    add(f"`{XLSX_NAME}` 共 5 张表（比计划里的 4 张多一张 `q2_group_stats`：组级的"
        "「组间/组内距离比、置换检验 p、两轴判定」是每组一条的事实，重复贴在每个成员行上会难读，"
        "所以单独成表；`q2_cases` 只放逐机制的行）：")
    add("")
    add("| 表 | 内容 |")
    add("|---|---|")
    add("| `case_selection` | 每个个案选自 Fig. 2 的哪个产物、命中哪条规则；附 Q1 的逐通道候选数与未覆盖通道 |")
    add("| `q1_cases` | 逐个案：三层 Δ（发布表 / 只用 result_2 重算 / 池化）、两轴标记、逐字证据并排 |")
    add("| `q2_cases` | 逐（个案 × 机制）：Δ、轴一指纹、两轴判定、共有与独有短语、该通道的全部观察记录 |")
    add("| `q2_group_stats` | 逐个案一行：Δ 极差、距离比、置换检验 p、两轴判定与合取结论 |")
    add("| `q3_cases` | 逐（个案 × 档位）：逐维 Δ、受影响维度数、集中度、对照 Δ、E1–E4 判定 |")
    add("")
    add("### 0.5 证据归属仪器的匹配质量")
    add("")
    add(f"把分析者的 `task_measurements[].task` 反查回具体提问轮"
        f"（覆盖率 ≥ {MATCH_MIN}、且最佳至少是次佳的 {MATCH_MARGIN} 倍）："
        f"归属成功 {context['attribution']['matched']} 条、"
        f"无法归属 {context['attribution']['unattributed']} 条"
        f"（归属率 {context['attribution']['attribution_rate']}，"
        f"匹配得分中位 {context['attribution']['score_median']}）。"
        "**无法归属的一律记 `unattributed`，不硬塞** —— 这是本产物最主要的误差来源之一。")
    add("")

    # ---------------- Q1 ----------------
    add("---")
    add("")
    add("## 1. 受损的心智在某一维度上，如何区别于正常心智")
    add("")
    add("### 1.1 结论")
    add("")
    for row in context["q1_rows"]:
        add(f"- **{row['case']} · {row['dimension_zh']}（{row['primitive_id']} @ {row['dose']:g}）**："
            f"该通道读数从基线 {row['baseline_reading']} 降到发布表均值 {row['published_mean']}"
            f"（发布表 Δ = {row['published_delta']}，n={row['n_published']}；"
            f"只用 result_2 的行重算 Δ = {row['delta_recomputed_result2']}）。"
            f"合并两个批次后该档共 {row['n_pooled']} 段会话"
            f"（其中 result_2 有 {row['n_result2']} 段），池化均值 {row['pooled_mean']}"
            f"（池化 Δ = {row['delta_pooled']}），重复间极差 {row['rep_range']}。"
            f"轴一读数：{row['axis1_verdict']}。")
    add("")
    add("### 1.2 支撑数字（轴一 · 客观转写标记）")
    add("")
    add(f"标记只在该通道的 Q→A 对上计算。参照侧合并 {context['reference']['n_reference']} 份会话。")
    add("")
    for row in context["q1_rows"]:
        add(f"**{row['case']} · {row['dimension_zh']}**")
        add("")
        add("| 侧 | 对数 | 应答长度均值 | 中位 | 反问率 | 复述提问用词 | 自我重复 | 逐字重复 | 极短/未作答 |")
        add("|---|---|---|---|---|---|---|---|---|")
        for label, payload in (("完整心智", row["evidence_axis1_intact"]),
                               ("受损心智", row["evidence_axis1_impaired"])):
            marker = json.loads(payload)
            add(f"| {label} | {marker['n_pairs']} | {marker['answer_chars_mean']} | "
                f"{marker['answer_chars_median']} | {marker['q_back_rate']} | "
                f"{marker['echo_jaccard_mean']} | {marker['self_repeat_max_mean']} | "
                f"{marker['dup_answer_rate']} | {marker['short_or_unknown_rate']} |")
        add("")
    add("### 1.3 逐字证据（轴二）")
    add("")
    for row in context["q1_rows"]:
        add(f"**{row['case']} · {row['dimension_zh']} —— 完整心智侧（参照记录）**")
        add("")
        add("```text")
        add(row["evidence_intact"] or "（无记录）")
        add("```")
        add("")
        add(f"**{row['case']} · {row['dimension_zh']} —— 受损心智侧（{row['primitive_id']} "
            f"@ {row['dose']:g}；池化 {row['n_pooled']} 段会话，其中 result_2 有 "
            f"{row['n_result2']} 段）**")
        add("")
        add("```text")
        add(row["evidence_impaired"] or "（无记录）")
        add("```")
        add("")
        slots = json.loads(row["slot_contrast"])
        if slots:
            add(f"**同 `probe_key` 槽位的逐字对照**（{row['slot_contrast_note']}）")
            add("")
            for slot in slots:
                add(f"- 槽位 `{slot['probe_key']}`（措辞相同："
                    f"{'是' if slot['wording_identical'] else '否'}）")
                add(f"  - 完整心智问：{slot['intact_question']}")
                add(f"  - 完整心智答：{slot['intact_answer']}")
                add(f"  - 受损心智问：{slot['impaired_question']}")
                add(f"  - 受损心智答：{slot['impaired_answer']}")
            add("")
        else:
            add(f"> {row['slot_contrast_note']}。")
            add("")
        add(f"该通道读数所用证据中，来自**它自己通道**的提问轮的比例："
            f"{row['own_channel_evidence_share']}。")
        add("")

    # ---------------- Q2 ----------------
    add("---")
    add("")
    add("## 2. 同一维度得分降低、但机制不同时，行为表现是否有差异")
    add("")
    add("组内成员按 Δ 配对（|Δ₁−Δ₂| ≤ "
        f"{context['params']['match_tolerance']}），以排除「只是轻重不同」。")
    add("")
    add("### 2.1 结论")
    add("")
    for group in context["q2_groups"]:
        dropped = group["hypothesis_dropped_for_gap"]
        add(f"- **{group['case']} · {group['dimension_zh']} @ {group['dose']:g}**："
            f"成员 {'、'.join(group['members'])}（Δ 极差 {group['delta_span']}）。"
            f"轴一 {group['axis1_verdict']}（组间/组内距离比 {group['separation_ratio']}，"
            f"置换检验 p = {group['permutation_p']}，**探索性**）；"
            f"轴二 {group['axis2_verdict']}。→ **{group['two_axis_agreement']}**")
        if dropped:
            add(f"  - 注：{ '、'.join(f'{name}(Δ={delta})' for name, delta in dropped) } "
                f"因 Δ 与组内极差超出容差被排除，不能算「同等程度」的对照。")
    add("")
    add("### 2.2 支撑数字")
    add("")
    add("| 个案 | 维度 | 机制 | Δ | 逐字重复 | 自我重复 | 应答长度 | 反问率 | 剔除套话后文本重合 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in context["q2_rows"]:
        marker = json.loads(row["markers"])
        add(f"| {row['case']} | {row['dimension_zh']} | {row['primitive_id']} | "
            f"{row['delta_published']} | {marker['dup_answer_rate']} | "
            f"{marker['self_repeat_max_mean']} | {marker['answer_chars_mean']} | "
            f"{marker['q_back_rate']} | {row['text_stripped_similarity_mean']} |")
    add("")
    add("| 个案 | 维度 | 机制对 | 原始文本重合 | 剔除套话后 | 被剥掉的套话片段数 |")
    add("|---|---|---|---|---|---|")
    for item in context["q2_pairs"]:
        add(f"| {item['case']} | {item['dimension_zh']} | {' vs '.join(item['pair'])} | "
            f"{item['raw_similarity']} | {item['stripped_similarity']} | "
            f"{item['boilerplate_removed']} |")
    add("")
    add(f"套话停用表从语料自动导出：判据是**跨机制**出现率 ≥ "
        f"{context['boilerplate']['rate_threshold']}（且至少落在 3 个机制里），"
        f"命中 {context['boilerplate']['n_fragments']} 条片段（语料共 "
        f"{context['boilerplate']['n_mechanisms']} 个机制）。不手工写死。")
    add("")
    add("实测覆盖率最高的片段（这正是「套话会把机制差别抹平」的量化边界）：")
    add("")
    add("| 片段 | 出现在多少个机制 | 占比 |")
    add("|---|---|---|")
    for item in context["boilerplate"]["widest"]:
        add(f"| 「{item['fragment']}」 | {item['n_mechanisms']} | {item['share']} |")
    add("")
    add("**没有任何片段是普遍套话** —— 覆盖率最高的也只落在 "
        f"{context['boilerplate']['widest'][0]['n_mechanisms']} 个机制上。所以「机制差别被套话抹平」"
        "在这批数据里是**有限度**的：套话确实重复，但不是所有机制共用的同一句。")
    add("")
    add("### 2.3 逐字证据（轴二）")
    add("")
    for row in context["q2_rows"]:
        add(f"**{row['case']} · {row['dimension_zh']} · `{row['primitive_id']}`**"
            f"（独有短语：{row['distinctive_phrases'] or '无'}）")
        add("")
        add("```text")
        add(row["observations"] or "（无记录）")
        add("```")
        add("")

    # ---------------- Q3 ----------------
    add("---")
    add("")
    add("## 3. 同一机制轻度只坏一个表型、严重影响多个，原因是什么")
    add("")
    add("四条解释逐条判定：**E1** 非特异通用退化（对照通道一起塌）／"
        "**E2** 集中度坍缩（不再集中在一个维度）／"
        "**E3** 证据共用（溢出通道的读数靠主导通道的问句挣来，属测量层溢出）／"
        "**E4** 真实级联（溢出通道有自己的证据、对照通道稳定）。")
    add("")
    add("### 3.1 结论")
    add("")
    for case in context["q3_cases"]:
        rows = [r for r in context["q3_rows"] if r["primitive_id"] == case["primitive_id"]]
        if not rows:
            continue
        add(f"- **{case['signature']} —— `{case['primitive_id']}`**"
            f"（这条 signature 的挑选用值：{case['signature_value']}）")
        add("")
        add(f"  **前提检查**：{case.get('premise', '（未判定）')}")
        add("")
        add("| 档位 | 受影响维度数 (Δ≤−0.15) | Δ≤−0.30 | 主导维度 | 集中度 | 对照 Δ | 被探查通道数 |")
        add("|---|---|---|---|---|---|---|")
        for row in rows:
            add(f"| {row['dose']:g} | {row['n_affected_015']} | {row['n_affected_030']} | "
                f"{row['dominant_dim_zh']} | {row['concentration']:+.2f} | "
                f"{row['control_delta']} | {row['n_channels_probed']} |")
        add("")
        for row in rows:
            add(f"  - @ {row['dose']:g}：{row['reading']}")
        add("")
    add("### 3.2 支撑数字（逐维 Δ）")
    add("")
    add("| 个案 | 机制 | 档位 | " + " | ".join(channel_label(c) for c in PHENO)
        + " | 对照通道 | E3 溢出证据来自主导通道 | E4 溢出通道自有证据占比 |")
    add("|---|---|---|" + "---|" * len(PHENO) + "---|---|---|")
    for row in context["q3_rows"]:
        adds = " | ".join(str(row.get(f"delta_{channel}", "")) for channel in PHENO)
        add(f"| {row['case']} | {row['primitive_id']} | {row['dose']:g} | {adds} | "
            f"{row['control_delta']} | {row['E3_spillover_evidence_from_dominant']} | "
            f"{row['E4_own_channel_evidence_share']} |")
    add("")
    add("> 覆盖度对照：上表「被探查通道数」是**该会话实际被问到的通道数**。"
        "如果它随剂量变化，「受影响维度变多」就可能只是提问覆盖变了，而不是表型真的扩散。"
        "本次各档的被探查通道数见上，全部为 " +
        f"{sorted({r['n_channels_probed'] for r in context['q3_rows']})}。")
    add("")

    # ---------------- 局限 ----------------
    add("---")
    add("")
    add("## 4. 必须随产物一并引用的局限")
    add("")
    for item in LIMITATIONS:
        add(f"- {item}")
    add("")
    add("### 4.1 本产物特有的读法限制")
    add("")
    add("1. **每条结论都必须标注它来自哪一轴。** 轴一（转写标记）是客观但粗的；"
        "轴二（分析者记录）细但它是 LLM 散文。两轴一致才可作强结论，不一致时以轴一为准并降级措辞。")
    add("2. **E3/E4 依赖证据归属仪器的匹配质量**（见 0.5）。归属不上的证据被记为 "
        "`unattributed`，会让 E3 的分子偏小 —— 即 E3 的判定偏保守。")
    add("3. `results/data/prior_hypotheses.json` 里的机制—维度配对**只用于个案排序偏好**，"
        "不参与任何数值计算；报告中标为「命中先验假设」的地方仅表示该个案与已有理论相关。")
    add("")

    add("## 5. 自检清单")
    add("")
    for item in context["self_check"]:
        add(f"- {item}")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_dialogue_evidence",
        description="从 result_2 / result_4 的表回读已记录的对话探查过程，做三个问题的个案分析。",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--q1-limit", type=int, default=3, help="Q1 个案数（2–3）")
    parser.add_argument("--q2-limit", type=int, default=3, help="Q2 个案数（2–3）")
    parser.add_argument("--q3-limit", type=int, default=3, help="Q3 个案数（2–3）")
    parser.add_argument("--q1-tier", default="A", choices=("A", "B"),
                        help="Q1 从 figure2_link_candidates.csv 的哪一档里选个案")
    parser.add_argument("--match-tolerance", type=float, default=0.05,
                        help="Q2 的 Δ 配对容差")
    parser.add_argument("--boilerplate-rate", type=float, default=0.20,
                        help="片段跨机制出现率达到多少算套话（实测语料最高约 0.42，"
                             "所以 0.60 会让这张表变空；0.20 是能真正起作用的档）")
    parser.add_argument("--match-min", type=float, default=MATCH_MIN,
                        help="证据归属的覆盖率门槛")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--llm-contrast", action="store_true",
                        help="对已记录对话做匿名化对照判读（默认关闭；不对心智做探查）")
    parser.add_argument("--llm-contrast-plan-only", action="store_true",
                        help="只列出值得判读的配对，不调用模型")
    parser.add_argument("--llm-max-pairs", type=int, default=4)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只打印选定与闸门，不写盘")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    counters = {"rows_total": 0, "row_id_unparsed": 0, "join_ok": 0, "join_missing": 0,
                "json_ok": 0, "json_missing": 0, "session_mismatch": 0, "dose_mismatch": 0,
                "reused_in_result_4": 0, "llm_calls": 0}

    before = fingerprint(PROTECTED)

    print("=== 阶段 0 · 闸门 ===")
    subjects_r2 = load_subject_rows(RESULT_2, SINGLE_RUN, "result_2", counters)
    subjects_r4 = load_subject_rows(RESULT_4, DOSE_RUN, "result_4", counters)
    print(f"  result_2: {len(subjects_r2)} 行 → 逐被试 JSON 全部回指、session_id 与剂量段全部相符")
    print(f"  result_4: {len(subjects_r4)} 行 → 逐被试 JSON 全部回指、session_id 与剂量段全部相符")
    counters["reused_in_result_4"] = mark_reused(subjects_r2, subjects_r4)
    reused_doses = sorted({f"{s.dose:g}" for s in subjects_r4 if s.reused})
    print(f"  去重：result_4 里有 {counters['reused_in_result_4']} 行与 result_2 是**同一段会话**"
          f"（θ 指纹+剂量+重复+内容哈希四样全同；档位 {reused_doses}）"
          "，不当作独立重复")

    official, reference = load_reference_records()
    print(f"  完整心智参照：官方 {len(official)} 份 + 补充 {len(reference) - len(official)} 份 "
          f"= {len(reference)} 份")
    if len(official) != 5:
        raise SystemExit(f"✗ 官方基线应为 5 份，实际 {len(official)} 份 —— 闸门不通过。")

    metrics = json.loads(FIGURE2_METRICS.read_text(encoding="utf-8"))
    baseline = {k: float(v) for k, v in metrics["baseline"].items()}
    computed = {}
    for channel in CHANNELS:
        values = [
            float((record.get("analysis") or {}).get("readings", {}).get(channel))
            for record in official
            if (record.get("analysis") or {}).get("readings", {}).get(channel) is not None
        ]
        computed[channel] = round(statistics.fmean(values), 3) if values else None
    mismatch = {c: (baseline.get(c), computed.get(c)) for c in CHANNELS
                if computed.get(c) is None or abs(computed[c] - baseline.get(c, -9)) > 1e-9}
    if mismatch:
        raise SystemExit(f"✗ 官方 5 份基线的读数均值与 figure2_metrics.json 的 baseline 不一致：{mismatch}")
    print(f"  ✓ 官方 5 份基线均值与 figure2_metrics.json 的 baseline 逐位相同（{len(CHANNELS)} 个通道）")

    with LINK_CANDIDATES.open(encoding="utf-8-sig", newline="") as handle:
        link_rows = list(csv.DictReader(handle))
    hypotheses = read_prior_hypotheses(link_rows)
    print(f"  先验假设（仅用于个案排序偏好）："
          f"{ {k: sorted(v) for k, v in hypotheses.items()} }")

    print("=== 阶段 1 · 证据归属仪器 ===")
    all_subjects = subjects_r2 + subjects_r4
    # 逐被试池去重：同一段会话只保留一份。5 档批次是**逐字节复制** result_2 的 0.4/0.8 单元格
    # 装配出来的，所以这两档的会话在两张表里各出现一次；不合并就会把同一段对话当成两个样本
    # （后果：证据块重复贴、重复间极差算错、置换检验的 n 虚高）。保留先出现的那份（result_2）。
    pool: list[Subject] = []
    seen_content: set[str] = set()
    for subject in all_subjects:
        if subject.content_sha in seen_content:
            continue
        seen_content.add(subject.content_sha)
        pool.append(subject)
    print(f"  逐被试池去重：{len(all_subjects)} 行 → {len(pool)} 段不同的会话"
          f"（合并掉 {len(all_subjects) - len(pool)} 行重复）")

    fragments, n_mechanisms = load_corpus_fragments(pool, reference)
    boilerplate = boilerplate_set(fragments, n_mechanisms, args.boilerplate_rate)
    widest = sorted(fragments.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
    print(f"  套话停用表：{len(boilerplate)} 条片段"
          f"（判据：跨机制出现率 ≥ {args.boilerplate_rate} 且至少落在 3 个机制里；"
          f"语料 {n_mechanisms} 个机制）")
    print(f"    实测覆盖率最高的片段："
          + "、".join(f"「{f}」{len(l)}/{n_mechanisms}" for f, l in widest)
          + " —— 没有任何片段是普遍套话，套话只在低比例上重复")

    for subject in pool:
        subject.record["_pairs"] = pair_turns(subject.record)
        subject.pairs = subject.record["_pairs"]
    for record in reference:
        record["_pairs"] = pair_turns(record)

    # 下标：按 (primitive, dose) 与 (primitive, channel, dose) 都能取
    subject_index: dict[tuple[str, str, float], list[Subject]] = defaultdict(list)
    for subject in pool:
        subject_index[(subject.primitive_id, "", subject.dose)].append(subject)
        for channel in CHANNELS:
            subject_index[(subject.primitive_id, channel, subject.dose)].append(subject)

    # 只对个案涉及的会话算归属（归属是逐个案的仪器，不是全表产物）
    q1_cases, q1_availability = select_q1_cases(
        link_rows, hypotheses, limit=args.q1_limit, tier=args.q1_tier)
    q2_cases = select_q2_cases(
        link_rows, metrics, hypotheses, limit=args.q2_limit,
        tolerance=args.match_tolerance, dose=float(metrics.get("primary_dose") or 0.4))
    summary_r4 = read_sheet(RESULT_4, "summary_by_cell")
    ladders = dose_ladders(summary_r4)
    q3_cases = select_q3_cases(ladders, metrics, limit=args.q3_limit)

    print("  个案选定：")
    for row in q1_cases:
        print(f"    Q1 · {row['dimension']} ← {row['primitive_id']} "
              f"(tier {row.get('tier')}, Δ={row.get('delta')}, verdict={row.get('verdict')})")
    for case in q2_cases:
        print(f"    Q2 · {case['dimension']} @ {case['dose']:g} ← {case['members']} "
              f"(Δ 极差 {case['span']})")
    for case in q3_cases:
        print(f"    Q3 · {case['signature']} ← {case['primitive_id']} "
              f"(用值 {case['signature_value']})")
    if q1_availability["dimensions_without_case"]:
        print(f"    注：tier {args.q1_tier} 里没有个案的通道 —— "
              f"{[channel_label(c) for c in q1_availability['dimensions_without_case']]}"
              f"（tier 池逐通道行数 {q1_availability['per_dimension']}）")

    # 个案涉及的会话才做归属
    needed: set[str] = set()
    for row in q1_cases:
        for subject in pool:
            if subject.primitive_id == row["primitive_id"] and abs(subject.dose - float(row["dose"])) < 1e-9:
                needed.add(subject.session_id)
    for case in q2_cases:
        for subject in pool:
            if subject.primitive_id in case["members"] and abs(subject.dose - float(case["dose"])) < 1e-9:
                needed.add(subject.session_id)
    for case in q3_cases:
        for subject in pool:
            if subject.primitive_id == case["primitive_id"]:
                needed.add(subject.session_id)

    attributions: dict[str, dict] = {}
    for subject in pool:
        if subject.session_id in needed:
            attributions[subject.session_id] = attribute_evidence(
                subject.record, subject.pairs, match_min=args.match_min)
    agg_matched = sum(a["matched"] for a in attributions.values())
    agg_unmatched = sum(a["unattributed"] for a in attributions.values())
    agg_scores = [s for a in attributions.values() for s in a["scores"]]
    attribution_summary = {
        "n_sessions": len(attributions),
        "matched": agg_matched,
        "unattributed": agg_unmatched,
        "attribution_rate": round(agg_matched / (agg_matched + agg_unmatched), 4)
        if (agg_matched + agg_unmatched) else 0.0,
        "score_median": round(statistics.median(agg_scores), 4) if agg_scores else 0.0,
    }
    print(f"  证据归属：{attribution_summary['matched']} 条归属成功、"
          f"{attribution_summary['unattributed']} 条无法归属"
          f"（归属率 {attribution_summary['attribution_rate']}）")

    print("=== 阶段 2–4 · 三问的个案 ===")
    summary_r2 = read_sheet(RESULT_2, "summary_by_cell")
    q1_rows = build_q1_rows(baseline, reference, [
        dict(row, from_hypothesis=row["primitive_id"] in hypotheses)
        for row in q1_cases
    ], subject_index, attributions, summary_r2)
    q2_rows, q2_groups = build_q2_rows(
        q2_cases, subject_index, boilerplate,
        tolerance=args.match_tolerance, seed=args.seed)
    q3_rows: list[dict] = []
    for case in q3_cases:
        q3_rows.extend(build_q3_rows(case, ladders.get(case["primitive_id"]) or [],
                                     subject_index, attributions))
    for case in q3_cases:
        case["premise"] = premise_verdict(
            [r for r in q3_rows if r["primitive_id"] == case["primitive_id"]])
    print(f"  Q1 {len(q1_rows)} 例、Q2 {len(q2_rows)} 行（{len(q2_groups)} 组）、"
          f"Q3 {len(q3_rows)} 行（{len(q3_cases)} 例）")

    # 反向校验：本脚本从逐被试 JSON 重算的 Δ，必须与发布表里的 Δ 对得上 ——
    # 这一条同时验证了「row_id → JSON 的回指取对了人」。对不上就报出来，不静默放过。
    delta_checks = []
    for row in q1_rows:
        published = as_float(row.get("published_delta"))
        computed = as_float(row.get("delta_recomputed_result2"))
        if published is None or computed is None:
            continue
        delta_checks.append({
            "case": row["case"],
            "published": published,
            "recomputed_result2": computed,
            "gap": round(abs(published - computed), 4),
        })
    worst_gap = max((c["gap"] for c in delta_checks), default=0.0)
    print(f"  Δ 反向校验（发布表 Δ vs 只用 result_2 的行重算）：{len(delta_checks)} 例，"
          f"最大偏差 {worst_gap}"
          + ("（≤0.005，row_id 回指取对了人）" if worst_gap <= 0.005 else " ⚠ 偏差偏大，需排查"))

    contrast_plan = build_contrast_plan(
        q2_groups, subject_index, tolerance=args.match_tolerance, max_pairs=args.llm_max_pairs)
    llm_result: list[dict] = []
    if args.llm_contrast_plan_only:
        plan_path = Path(args.out_dir) / "dialogue_evidence_llm_contrast_plan.json"
        if not args.dry_run:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps({
                "note": "「必要时」的落地：只列两轴结论不一致或轴一落在难以区分区间的配对。",
                "layer": "post_hoc_interpretation",
                "no_writeback": NO_WRITEBACK,
                "pairs": [{k: v for k, v in item.items() if k != "transcript_a" and k != "transcript_b"}
                          for item in contrast_plan],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ 判读计划 → {plan_path}")
        else:
            print(f"  [dry-run] 判读计划 {len(contrast_plan)} 对")
    if args.llm_contrast:
        print("=== 阶段 5 · LLM 判读（只读已记录的对话，不探查心智）===")
        llm_result = run_llm_contrast(
            contrast_plan, allow_mock=args.allow_mock, temperature=args.llm_temperature,
            counters=counters)
        print(f"  完成 {len(llm_result)} 次判读（模型 {llm_result[0]['model_id'] if llm_result else '—'}）")

    # ---------------- 写盘 ----------------
    out_dir = Path(args.out_dir)
    case_selection = []
    for rank, row in enumerate(q1_cases, start=1):
        case_selection.append({
            "case": f"Q1-{rank}",
            "question": "Q1 受损心智在某一维度上如何区别于正常心智",
            "source": "figure2_link_candidates.csv",
            "rule": f"tier {row.get('tier')}；各通道至多 1 例、跨通道、优先 verdict=pass、"
                    f"优先命中先验假设 → {row['dimension']} ← {row['primitive_id']}",
        })
    for rank, case in enumerate(q2_cases, start=1):
        case_selection.append({
            "case": f"Q2-{rank}",
            "question": "Q2 同维度不同机制的行为是否有差异",
            "source": "figure2_metrics.json → panel_b.per_dose['0.4'].top3_per_dimension",
            "rule": f"组内最大 Δ 配对子集（|Δ差| ≤ {args.match_tolerance}），"
                    f"平局优先命中先验假设 → {case['dimension']} "
                    f"← {case['members']}（Δ 极差 {case['span']}）",
        })
    for case in q3_cases:
        case_selection.append({
            "case": case["signature"].split(" ·")[0],
            "question": "Q3 同机制轻度一个表型、严重多个表型的原因",
            "source": "figure2_metrics.json → panel_f.per_primitive",
            "rule": f"{case['signature']} → {case['primitive_id']}（用值 {case['signature_value']}）",
        })

    q1_rule = dict(q1_availability)
    for row, item in zip(q1_rows, case_selection[:len(q1_rows)]):
        item["rule"] += (f"；该通道 tier {args.q1_tier} 候选 {q1_rule['per_dimension'].get(row['dimension'], 0)} 条")

    # 机制对的文本重合（原始 / 剔除套话后）—— 逐个案重新取一次证据文本，
    # 与 `q2_cases` 表上的按机制汇总互不依赖。
    q2_pair_detail: list[dict] = []

    context = {
        "params": {
            "match_tolerance": args.match_tolerance,
            "boilerplate_rate": args.boilerplate_rate,
            "match_min": args.match_min,
            "q1_tier": args.q1_tier,
            "seed": args.seed,
        },
        "baseline": baseline,
        "reference": {
            "n_official": len(official),
            "n_reference": len(reference),
            "official_dir": str(OFFICIAL_BASELINE_DIR.relative_to(_ROOT)),
            "reading_mean": {
                channel: mean_or_blank([
                    (record.get("analysis") or {}).get("readings", {}).get(channel)
                    for record in reference
                ]) for channel in CHANNELS
            },
        },
        "gate": counters,
        "sources": [
            {"batch": "result_2", "xlsx": str(RESULT_2.relative_to(_ROOT)),
             "rows": len(subjects_r2), "run_dir": str(SINGLE_RUN.relative_to(_ROOT)),
             "reused": "—（源表）"},
            {"batch": "result_4", "xlsx": str(RESULT_4.relative_to(_ROOT)),
             "rows": len(subjects_r4), "run_dir": str(DOSE_RUN.relative_to(_ROOT)),
             "reused": counters["reused_in_result_4"]},
        ],
        "case_selection": case_selection,
        "attribution": attribution_summary,
        "delta_check": {
            "n": len(delta_checks),
            "worst_gap": worst_gap,
            "rows": delta_checks,
        },
        "q1_rows": q1_rows,
        "q2_rows": q2_rows,
        "q2_groups": [
            {k: v for k, v in group.items() if not k.startswith("_")} for group in q2_groups
        ],
        "q2_pairs": [],
        "q3_cases": q3_cases,
        "q3_rows": q3_rows,
        "boilerplate_sample": sorted(boilerplate)[:12],
        "boilerplate": {
            "n_fragments": len(boilerplate),
            "rate_threshold": args.boilerplate_rate,
            "n_mechanisms": n_mechanisms,
            "widest": [{"fragment": f, "n_mechanisms": len(l), "share": round(len(l) / n_mechanisms, 3)}
                       for f, l in widest],
        },
    }

    # 让报告与 JSON 拿到真正的机制对文本重合数：从 q2_rows 上的独有短语倒推不足，
    # 所以这里补一次成对的相似度计算（数据量很小）。
    q2_pair_detail: list[dict] = []
    for group in q2_groups:
        dimension = group["dimension"]
        dose = float(group["dose"])
        texts: dict[str, list[str]] = {}
        for primitive in group["members"]:
            subjects = subject_index.get((primitive, dimension, dose), []) or \
                subject_index.get((primitive, "", dose), [])
            collected: list[str] = []
            for subject in subjects:
                collected.extend(observations(subject.record, dimension))
            if collected:
                texts[primitive] = collected
        labels = sorted(texts)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                left, right = labels[i], labels[j]
                raw = jaccard(bigrams("".join(texts[left])), bigrams("".join(texts[right])))
                strip_left = strip_boilerplate(texts[left], boilerplate)
                strip_right = strip_boilerplate(texts[right], boilerplate)
                _, dropped = _split_and_filter(texts[left] + texts[right], boilerplate)
                stripped = jaccard(bigrams("".join(strip_left)), bigrams("".join(strip_right)))
                q2_pair_detail.append({
                    "case": group["case"],
                    "dimension_zh": group["dimension_zh"],
                    "pair": [left, right],
                    "raw_similarity": round(raw, 4),
                    "stripped_similarity": round(stripped, 4),
                    "boilerplate_removed": dropped,
                })
    context["q2_pairs"] = q2_pair_detail

    after = fingerprint(PROTECTED)
    unchanged = before == after
    context["self_check"] = [
        f"闸门：row_id 解析失败 {counters['row_id_unparsed']}；回指成功 {counters['join_ok']}；"
        f"session_id 不符 {counters['session_mismatch']}；"
        f"剂量段与回指记录不符 {counters['dose_mismatch']}；JSON 缺失 {counters['json_missing']}。",
        f"去重：result_4 里识别出 {counters['reused_in_result_4']} 行与 result_2 是同一段会话"
        f"（判据 = θ 指纹 + 剂量 + 重复 + JSON 内容哈希四样全同）。",
        f"参照：官方 {len(official)} 份、合计 {len(reference)} 份；"
        f"官方均值与 figure2_metrics.json 的 baseline 逐位相同。",
        f"Δ 反向校验（发布表 Δ vs 只用 result_2 的行重算）：{len(delta_checks)} 例，"
        f"最大偏差 {worst_gap}"
        + ("（≤0.005，row_id 回指取对了人）" if worst_gap <= 0.005 else " ⚠ 偏差偏大，需排查"),
        f"证据归属：成功 {attribution_summary['matched']}、无法归属 "
        f"{attribution_summary['unattributed']}（归属率 {attribution_summary['attribution_rate']}）。",
        f"网络调用计数：{counters['llm_calls']}"
        + ("（未开 --llm-contrast，全程离线）" if not args.llm_contrast else "。"),
        f"result_1..4 的 sha256 与大小在运行前后完全不变："
        + ("是 ✓" if unchanged else "否 ✗ —— 有文件被改动，必须排查"),
        f"个案数：Q1 {len(q1_rows)}、Q2 {len(q2_groups)} 组、Q3 {len(q3_cases)} 例。",
    ]

    if args.dry_run:
        print("=== [dry-run] 不写盘 ===")
        for line in context["self_check"]:
            print("  " + line)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_workbook(out_dir / XLSX_NAME, context, q1_availability)
    (out_dir / MD_NAME).write_text(render_report(context), encoding="utf-8")
    payload = dict(context)
    payload["layer"] = "post_hoc_interpretation"
    payload["no_writeback"] = NO_WRITEBACK
    payload["limitations"] = LIMITATIONS
    payload["llm_contrast_run"] = bool(args.llm_contrast)
    (out_dir / JSON_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if llm_result:
        (out_dir / LLM_JSON_NAME).write_text(
            json.dumps({"layer": "post_hoc_interpretation", "no_writeback": NO_WRITEBACK,
                        "judgements": llm_result}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 阶段 6 · 产物 ===")
    print(f"  ✓ {out_dir / XLSX_NAME}")
    print(f"  ✓ {out_dir / MD_NAME}")
    print(f"  ✓ {out_dir / JSON_NAME}")
    if llm_result:
        print(f"  ✓ {out_dir / LLM_JSON_NAME}")
    if not unchanged:
        print("  ✗ 既有 result 文件被改动 —— 必须排查！")
        return 2
    for line in context["self_check"]:
        print("  " + line)
    return 0


def write_workbook(path: Path, context: dict, availability: dict) -> None:
    workbook = ert.new_workbook()

    selection_headers = ["case", "question", "source", "rule"]
    selection_rows = [
        [item["case"], item["question"], item["source"], item["rule"]]
        for item in context["case_selection"]
    ]
    for channel, count in availability["per_dimension"].items():
        selection_rows.append([
            f"coverage · {channel}", "Q1 该通道在 tier 池里的候选行数", availability["tier"], str(count),
        ])
    selection_rows.append([
        "coverage · 无个案的通道", "Q1 选定后仍未覆盖的通道",
        availability["tier"],
        "、".join(channel_label(c) for c in availability["dimensions_without_case"]) or "无",
    ])
    ert.write_rows(workbook.create_sheet("case_selection"), selection_headers, selection_rows,
                   {"A": 26, "B": 34, "C": 30, "D": 96})

    q1_headers = [
        "case", "dimension", "dimension_zh", "primitive_id", "dose", "source", "selection_reason",
        "baseline_reading", "published_mean", "published_delta", "n_published",
        "delta_recomputed_result2", "pooled_mean", "delta_pooled", "n_pooled", "n_result2",
        "reference_reading_mean", "rep_range", "axis1_verdict",
        "evidence_axis1_intact", "evidence_axis1_impaired",
        "probe_key_shared", "probe_key_wording_identical", "slot_contrast_note",
        "own_channel_evidence_share", "evidence_intact", "evidence_impaired", "slot_contrast",
    ]
    ert.write_rows(workbook.create_sheet("q1_cases"), q1_headers,
                   [[row.get(h, "") for h in q1_headers] for row in context["q1_rows"]],
                   {"A": 8, "B": 32, "C": 18, "D": 24, "E": 7, "F": 34, "G": 46, "H": 14,
                    "I": 14, "J": 14, "K": 10, "L": 16, "M": 12, "N": 12, "O": 10, "P": 10,
                    "Q": 14, "R": 12, "S": 44, "T": 60, "U": 60, "V": 12, "W": 14, "X": 40,
                    "Y": 16, "Z": 70, "AA": 70, "AB": 60})

    q2_headers = [
        "case", "dimension", "dimension_zh", "dose", "primitive_id", "delta_published",
        "n_replicates", "axis1_separation_ratio", "axis1_permutation_p", "axis1_verdict",
        "axis2_verdict", "text_raw_similarity_mean", "text_stripped_similarity_mean",
        "distinctive_phrases", "markers", "observations",
    ]
    ert.write_rows(workbook.create_sheet("q2_cases"), q2_headers,
                   [[row.get(h, "") for h in q2_headers] for row in context["q2_rows"]],
                   {"A": 8, "B": 32, "C": 18, "D": 7, "E": 24, "F": 14, "G": 10, "H": 14,
                    "I": 12, "J": 30, "K": 30, "L": 14, "M": 14, "N": 60, "O": 60, "P": 80})

    q3_headers = [
        "case", "primitive_id", "dose", "n_replicates", "n_affected_015", "n_affected_030",
        "dominant_dim", "dominant_dim_zh", "second_dim", "concentration", "mean_delta_six_dims",
        "control_delta", "n_channels_probed", "specificity_verdict",
        *[f"delta_{channel}" for channel in PHENO],
        "E1_control_tolerance_exceeded", "E2_concentration",
        "E3_spillover_evidence_from_dominant", "E4_own_channel_evidence_share",
        "spillover_dims", "reading",
    ]
    ert.write_rows(workbook.create_sheet("q3_cases"), q3_headers,
                   [[row.get(h, "") for h in q3_headers] for row in context["q3_rows"]],
                   {"A": 8, "B": 24, "C": 7, "D": 10, "E": 12, "F": 12, "G": 32, "H": 18,
                    "I": 30, "J": 12, "K": 14, "L": 12, "M": 12, "N": 14,
                    **{get_column_letter(15 + i): 14 for i in range(len(PHENO))},
                    **{get_column_letter(15 + len(PHENO) + i): w
                       for i, w in enumerate([16, 14, 18, 18, 40, 76])}})

    group_headers = [
        "case", "dimension", "dimension_zh", "dose", "members", "delta_span", "match_tolerance",
        "separation_ratio", "permutation_p", "axis1_verdict", "axis2_verdict",
        "two_axis_agreement", "hypothesis_dropped_for_gap",
    ]
    group_rows = []
    for group in context["q2_groups"]:
        group_rows.append([
            group["case"], group["dimension"], group["dimension_zh"], group["dose"],
            " + ".join(group["members"]), group["delta_span"], group["match_tolerance"],
            group["separation_ratio"], group["permutation_p"], group["axis1_verdict"],
            group["axis2_verdict"], group["two_axis_agreement"],
            "；".join(f"{name}(Δ={delta})" for name, delta in group["hypothesis_dropped_for_gap"]) or "无",
        ])
    ert.write_rows(workbook.create_sheet("q2_group_stats"), group_headers, group_rows,
                   {"A": 8, "B": 32, "C": 18, "D": 7, "E": 44, "F": 12, "G": 14, "H": 14,
                    "I": 12, "J": 26, "K": 26, "L": 34, "M": 40})

    workbook.save(path)


if __name__ == "__main__":
    sys.exit(main())
