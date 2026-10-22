"""回答解析：从 LLM 的自由文本里抽出提问者的问题与分析者的读数。

真实模型的输出不会永远守格式，因此解析采取「多级回退 + 明确标记是否解析成功」：
- `parse_examiner_question` —— 提问者的下一个问题（解析不出就重试一次，仍不行则终止该轮）；
- `parse_analyst_json`      —— 分析者的表型读数（格式失败必须暴露，不能静默变成「一切正常」）。

心智自由作答，没有选项编号可解析；表型的判读统一由分析者 LLM 负责。
"""

from __future__ import annotations

import json
import re

from ..phenotype.conclusion import (
    CONCLUSION_CHARS_MAX,
    CONCLUSION_CHARS_MIN,
    CONCLUSION_CHARS_TARGET,
    CONCLUSION_MAX_CHARS,
    diagnosis_wording,
    normalize_asd_likeness,
)

_QUOTED_RE = re.compile(r"[「『\"']([^」』\"']{1,40})[」』\"']")

#: 读数 JSON 里的**结构性字段**：它们不是测量通道。正则兜底若不排除它们，截断的输出里
#: 一句 `"confidence": 0.8` 就会被读成「有一个叫 confidence 的维度得了 80 分」——
#: 一条凭空造出来的读数（比报「没有读数」糟得多，因为它看起来像数据）。
_STRUCTURAL_KEYS = frozenset({
    "confidence", "score", "deviation", "deviation_text", "dims", "dimensions", "items",
    "mean_reading", "conclusion", "conclusion_chars", "conclusion_target_chars",
    "conclusion_notes", "asd_likeness", "present", "level", "basis", "narrative", "notes",
    "observations", "task_measurements", "task", "measurement", "result", "error", "ok",
    "raw", "session_id", "model_id", "theta_id", "brain_id", "ts", "n_turns",
})


def has_self_state_ok(text: str) -> bool:
    """大脑初始化是否成功（`SELF_STATE: OK` 标记）。"""
    return bool(re.search(r"SELF_STATE\s*[:：]\s*OK", text or "", re.IGNORECASE))


# ---------------------------------------------------------------------------
# 表型观察的对话探针：提问者 / 分析者
# ---------------------------------------------------------------------------

_QUESTION_RE = re.compile(r"QUESTION\s*[:：]\s*(.+)", re.IGNORECASE)
_FOCUS_RE = re.compile(r"FOCUS\s*[:：]\s*(.+)", re.IGNORECASE)
#: 通道 id 的形状：小写字母开头，后面允许小写字母/数字/下划线
_CHANNEL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_examiner_question(text: str) -> str:
    """抽出提问者的问题文本。

    回退顺序：`QUESTION:` 行 → 去掉常见前缀后的首段 → 原文。宁可拿到稍脏的问题，
    也不返回空串（空问题会让整个对话停摆）。
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    match = _QUESTION_RE.search(raw)
    if match:
        question = match.group(1).strip()
        if question:
            return question
    cleaned = re.sub(r"^\s*(?:问题|场景|提问)\s*[:：]\s*", "", raw)
    # 反复剥掉「好的，」「那么，」这类引导语并清掉残留的开头标点。
    # 只剥**带逗号**的引导语：裸剥单字会把「那天下雨了会怎样？」「好孩子会怎么做？」这类
    # 真正的问句开头也吃掉 —— 兜底解析宁可留下一个「那么」，也不能篡改问题本身。
    for _ in range(4):
        before = cleaned
        cleaned = re.sub(r"^(?:好的|好吧|嗯|呃|那么|是这样|这样)\s*[，,、]\s*", "", cleaned)
        cleaned = re.sub(r"^[，,、。；;：:\s]+", "", cleaned)
        if cleaned == before:
            break
    for block in re.split(r"\n\s*\n", cleaned):
        line = block.strip()
        if line:
            return line
    return cleaned.strip()


def parse_examiner_focus(text: str) -> str:
    """抽出提问者自报的 `FOCUS:` 方面 id（没有就返回空串）。

    为什么让提问者自报：题库从「题目来源」降为「参考材料」之后，提问者会**自己出题**，
    于是「这一轮在问哪个方面」不再能从题库条目反查出来。没有它，自行出题的那些轮次
    在逐轮归属里是空的 —— mock 分析者会看不出选择性受损，真实分析者则失去逐轮线索。

    只接受**像通道 id 的**取值（小写字母/下划线/数字），并且调用方会再拿它去对照
    已知通道集合：模型偶尔会把中文名或整句话写进来，那种一律当作「没报」。
    """
    match = _FOCUS_RE.search(text or "")
    if not match:
        return ""
    raw = match.group(1).strip().strip("`*_ ")
    raw = raw.split()[0].rstrip("，,。.；;、）)") if raw else ""
    return raw if _CHANNEL_ID_RE.match(raw or "") else ""


def parse_examiner_turn(text: str) -> tuple[str, str]:
    """一次提问者输出 → `(问题, 自报方面)`；两者都做过兜底。"""
    return parse_examiner_question(text), parse_examiner_focus(text)


def presentation_of(text: str, fallback: str = "") -> str:
    """从提问者的输出里取出**完整呈现**（情境描述 + 处理方式提问），去掉格式标记。

    为什么不能只用解析出的问题：情境式探查里，儿童回应的是整个情境呈现；若只把
    `QUESTION:` 那一行存下来，对话记录与后续分析都会丢掉「当时设定了什么场景」。
    """
    raw = (text or "").strip()
    if not raw:
        return fallback
    kept = [
        line for line in raw.splitlines()
        if not _QUESTION_RE.match(line.strip()) and not _FOCUS_RE.match(line.strip())
    ]
    cleaned = "\n".join(line.rstrip() for line in kept).strip()
    return cleaned or fallback


def parse_analyst_json(
    text: str, *, allowed: "set[str] | tuple[str, ...]" = ()
) -> tuple[dict | None, str]:
    """解析分析者的严格 JSON 输出，返回 `(payload, 错误说明)`。

    回退顺序（每一级都必须**只产出真实读数**）：
    1. 整段 JSON → 文本里第一段 `{...}` → 代码块里的 JSON；
    2. **补齐被截断的 JSON**（模型写一半就被 `max_tokens` 截断是常见故障）：按括号栈把结尾补上，
       救回已经写出来的那些维度；
    3. 逐维度正则兜底 —— 但**只认真正的测量通道**：`allowed` 是调用方给的通道清单（观察目标
       ＋上传量表的条目 id），不在清单里的键一律丢掉，结构性字段（`confidence` / `score` /
       `deviation` …）永远不算维度。

    第 3 条保证读数只含测量通道：截断的 JSON 里可能含 `"confidence": 0.8`；
    若把 `confidence` 当成维度，界面上就会出现一条叫「confidence」的轴、平均分 80，
    而这份「读数」跟任何测量通道都无关。宁可报「没有可用读数」，也不能造一个。

    分析者的输出是表型读数，格式失败必须显式暴露（错误说明会写进反馈记录），
    不能静默变成「一切正常」。
    """
    raw = (text or "").strip()
    if not raw:
        return None, "分析者返回空内容"

    candidates: list[str] = [raw]
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    block = _JSON_BLOCK_RE.search(raw)
    if block:
        candidates.append(block.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return _normalize_feedback(data), ""

    # 被截断的 JSON：补齐括号后往往能救回前面已经写完的维度（真实读数，比正则可靠）
    truncated = _repair_truncated_json(raw)
    if truncated is not None:
        payload = _normalize_feedback(truncated)
        if payload.get("dims"):
            payload["notes"] = _join_notes(
                (payload.get("notes") or ""),
                "分析者输出在写完之前就结束了（JSON 不完整），已按写出来的部分读出；"
                "这次读数可能不完整 —— 请调大输出额度（max_tokens）或换用非推理模型后重新观察。",
            )
            return payload, "分析者输出被截断（JSON 不完整，已救回写出的部分）"

    dims = {}
    for key, value in re.findall(
        r'"?([A-Za-z_][A-Za-z0-9_]*)"?\s*[:=]\s*(0?\.\d+|[01](?:\.0+)?)', raw
    ):
        if key in _STRUCTURAL_KEYS:
            continue          # confidence / score / deviation 是字段，不是维度
        if allowed and key not in allowed:
            continue          # 调用方给了通道清单：清单外的键不认
        dims[key] = round(min(1.0, max(0.0, float(value))), 4)
    if dims:
        return (
            {
                "dims": dims,
                "dim_details": {},
                "confidence": None,
                "evidence": [],
                "narrative": "",
                "notes": "仅用正则从非 JSON 文本中救回部分维度读数，格式不合约定。",
                # 正则兜底救不回结论：它是一整段自然语言，靠正则拼不出来。留空键，
                # 让界面显示「未给出结论」而不是把兜底文本当成结论。
                "conclusion": "",
                "conclusion_chars": 0,
                "asd_likeness": {"present": None, "level": "unknown", "basis": []},
            },
            "分析者输出不是合法 JSON（已用正则兜底）",
        )
    if truncated is not None:
        return None, "分析者输出被截断（JSON 不完整），且没有救回任何维度读数"
    return None, "分析者输出无法解析为 JSON，也未能救回任何维度读数"


def _join_notes(existing: str, extra: str) -> str:
    """把两条说明并成一段（解析问题写在前面，模型自己的 notes 跟在后面）。"""
    return " ".join(part.strip() for part in (existing, extra) if part and part.strip())


def reading_dims(dims: object) -> dict:
    """从一组 dims 里挑出**像测量通道的键**，丢掉结构字段。

    用在读数离开反馈记录的最后一道关口上：结构字段（如 `confidence`）永远不算维度，
    否则界面上会多出一条凭空造出来的读数。上传量表的条目 id（`q1` 之类）
    不是结构字段，照常保留。
    """
    if not isinstance(dims, dict):
        return {}
    return {str(key): value for key, value in dims.items() if str(key) not in _STRUCTURAL_KEYS}


def _repair_truncated_json(text: str) -> dict | None:
    """把被截断的 JSON 补成可解析的对象；补不出来（不是截断、或结构本身就坏）返回 None。

    做法：扫一遍文本（跳过字符串内部），记下每个「值已经完整、可以安全截断」的位置与当时的
    括号栈；从最后往前试，截到该位置、去掉尾逗号、按栈把括号补齐再解析。截断通常发生在某个
    值写完之前，因此最近的几个安全点就能救回来。

    只在**括号确实没闭合**时才认「被截断」：括号都闭合却解析失败，说明是别处的语法错，
    不归这一级管（交给正则兜底，报错也更如实）。
    """
    start = text.find("{")
    if start < 0:
        return None
    chunk = text[start:]
    stack: list[str] = []
    in_string = False
    escaped = False
    points: list[tuple[int, tuple[str, ...]]] = []
    for index, char in enumerate(chunk):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return None                       # 括号不配对：不是「只是被截断」
            points.append((index + 1, tuple(stack)))
        elif char == "," and stack:
            points.append((index, tuple(stack)))
    if not stack:
        return None                               # 括号闭合 → 不是截断
    for cut, open_at_cut in reversed(points[-40:]):
        head = chunk[:cut].rstrip().rstrip(",")
        try:
            data = json.loads(head + "".join(reversed(open_at_cut)))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _score_to_typicality(value) -> float | None:
    """把 `score` 归一化成 [0, 1] 的**典型性**（1 = 与同龄典型发展无异）。

    协议只规定「behavioral score: 0–100」，没说两端各代表什么。本项目内部约定典型性方向，
    并在分析者提示词里写死。这里做的是**单位换算**：>1 视作 0–100 分制，≤1 视作已经是 0..1
    的比例（真实模型偶尔会给 0.85 表示 85 分）。两种都给不出数值时返回 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number > 1.0:
        number = number / 100.0
    return round(min(1.0, max(0.0, number)), 4)


def _normalize_dimension(entry) -> dict:
    """一维的行为记录：score / confidence / observations / task_measurements / deviation。"""
    if not isinstance(entry, dict):
        return {}
    out: dict = {}
    score = entry.get("score")
    typicality = _score_to_typicality(score)
    if typicality is not None:
        raw_score = float(score)
        out["score"] = round(raw_score if raw_score > 1.0 else raw_score * 100.0, 1)
        out["typicality"] = typicality
    confidence = entry.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        out["confidence"] = round(min(1.0, max(0.0, float(confidence))), 2)
    observations = entry.get("observations")
    if isinstance(observations, (list, tuple)):
        out["observations"] = [str(o).strip() for o in observations if str(o).strip()]
    measurements = entry.get("task_measurements")
    if isinstance(measurements, (list, tuple)):
        cleaned = []
        for item in measurements:
            if isinstance(item, dict):
                task = str(item.get("task") or "").strip()
                measurement = str(item.get("measurement") or item.get("result") or "").strip()
                if task or measurement:
                    cleaned.append({"task": task, "measurement": measurement})
            elif str(item).strip():
                cleaned.append({"task": "", "measurement": str(item).strip()})
        # 显式给了空列表 ≠ 没给这个字段：前者是「报告了，但没有任务级测量」，后者是缺项。
        # 两者必须区分得开，否则界面无法说明「为什么这一维没有测量记录」。
        out["task_measurements"] = cleaned
    deviation = entry.get("deviation")
    if isinstance(deviation, (int, float)) and not isinstance(deviation, bool):
        out["deviation"] = round(float(deviation), 1)
    elif isinstance(deviation, str) and deviation.strip():
        out["deviation_text"] = deviation.strip()
    return out


def _normalize_conclusion(data: dict) -> dict:
    """归一化探查结论：文本 + 字数 + 表型印象 + 两条越界提示。

    - 结论超长（跑飞）→ 截到 `CONCLUSION_MAX_CHARS`，并说明「已截断」；
    - 字数越出约定区间 → **不截断、不改写**，只在 `conclusion_notes` 里记一行；
    - 结论里出现诊断式用词 → 记一行告警（文本原样保留，由界面以警示色显示）。
      结论是**表型印象**，不是临床诊断 —— 这一条是口径红线，宁可显式告警也不静默接受。
    """
    text = str(data.get("conclusion") or "").strip()
    notes: list[str] = []
    if len(text) > CONCLUSION_MAX_CHARS:
        text = text[:CONCLUSION_MAX_CHARS]
        notes.append(f"结论超过 {CONCLUSION_MAX_CHARS} 字，已截断")
    chars = len(text)
    if text and not (CONCLUSION_CHARS_MIN <= chars <= CONCLUSION_CHARS_MAX):
        notes.append(f"结论 {chars} 字，超出约 {CONCLUSION_CHARS_TARGET} 字（{CONCLUSION_CHARS_MIN}–{CONCLUSION_CHARS_MAX}）的口径")
    hits = diagnosis_wording(text)
    if hits:
        notes.append("结论出现诊断式用词（" + "、".join(hits) + "）：应为表型印象而非临床诊断")
    return {
        "conclusion": text,
        "conclusion_chars": chars,
        "conclusion_target_chars": CONCLUSION_CHARS_TARGET,
        "conclusion_notes": notes,
        "asd_likeness": normalize_asd_likeness(data.get("asd_likeness")),
    }


def _normalize_feedback(data: dict) -> dict:
    """把分析者的 JSON 归一化成稳定结构（缺字段补空，dims 一律是 [0,1] 的典型性）。

    接受两种形态：
    - **协议阶段 10**：`dimensions` —— 每维一个记录（score 0–100、confidence、observations、
      task_measurements、deviation）。`score` 换算成 [0,1] 典型性进 `dims`，整条记录进
      `dim_details` 供界面逐维展示。
    - **扁平形态**：`dims` —— 直接是 {维度: 0..1}。仍然接受（上传量表的逐题读数）。
    """
    dims: dict[str, float] = {}
    dim_details: dict[str, dict] = {}

    dimensions_raw = data.get("dimensions") if isinstance(data.get("dimensions"), dict) else {}
    for key, entry in dimensions_raw.items():
        detail = _normalize_dimension(entry)
        if "typicality" in detail:
            dims[str(key)] = detail["typicality"]
        if detail:
            dim_details[str(key)] = detail

    # 旧的扁平 dims（以及上传量表的逐题读数）仍然照收
    dims_raw = data.get("dims") if isinstance(data.get("dims"), dict) else {}
    for key, value in dims_raw.items():
        try:
            dims.setdefault(str(key), round(min(1.0, max(0.0, float(value))), 4))
        except (TypeError, ValueError):
            continue

    # 上传量表：条目读数走 items（0..1），同样并进 dims 以保持既有下游不变
    items_raw = data.get("items") if isinstance(data.get("items"), dict) else {}
    for key, value in items_raw.items():
        try:
            dims[str(key)] = round(min(1.0, max(0.0, float(value))), 4)
        except (TypeError, ValueError):
            continue

    evidence = []
    for entry in data.get("evidence") or []:
        if isinstance(entry, dict):
            evidence.append({
                "dim": str(entry.get("dim", "")),
                "quote": str(entry.get("quote", "")),
                "reading": str(entry.get("reading", "")),
            })
        elif isinstance(entry, str):
            evidence.append({"dim": "", "quote": entry, "reading": ""})

    confidence = data.get("confidence")
    try:
        confidence = None if confidence is None else round(min(1.0, max(0.0, float(confidence))), 2)
    except (TypeError, ValueError):
        confidence = None
    if confidence is None:
        per_dim = [d["confidence"] for d in dim_details.values() if "confidence" in d]
        if per_dim:
            confidence = round(sum(per_dim) / len(per_dim), 2)

    return {
        "dims": dims,
        "dim_details": dim_details,
        "confidence": confidence,
        "evidence": evidence,
        "narrative": str(data.get("narrative") or ""),
        "notes": str(data.get("notes") or ""),
        **_normalize_conclusion(data),
    }
