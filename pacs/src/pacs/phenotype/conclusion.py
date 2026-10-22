"""探查结论：字数口径、表型印象档位，以及结论文本的两条红线检查。

分析者在逐维读数之外还要写一段**病例报告式结论**（是否表现出孤独症样表型特征 / 程度 /
各方面能力的相对表现）。这里放的是这段结论的**唯一口径**：字数区间、程度档位与中文对照、
以及「不许写成临床诊断」的用词检查 —— 提示词构造、输出解析、界面显示三处都读这里，
否则界面上的「约 350 字」与实际约束会各说各话。

**这不是诊断工具。** 结论是**行为学表型印象**：只描述这段对话里出现的表现与程度，
不构成临床诊断，也不对真实个体作任何判断（被试是虚拟心智，见 `SIMULATION_NOTICE`）。
"""

from __future__ import annotations

#: 目标字数与允许区间（字符数）。提示词按这个写，解析后越出区间只记一行说明（不截断）。
CONCLUSION_CHARS_TARGET = 350
CONCLUSION_CHARS_MIN = 300
CONCLUSION_CHARS_MAX = 400

#: 结论的硬截断长度：只防「模型跑飞写出几千字」，不是质量口径。
CONCLUSION_MAX_CHARS = 2000

#: 表型印象的程度档位（结构化字段 `asd_likeness.level` 的合法取值）。
ASD_LIKENESS_LEVELS = ("none", "mild", "moderate", "marked")

#: 档位的中文对照（提示词与界面共用，避免两处各写一套词）。
ASD_LIKENESS_LEVEL_ZH = {
    "none": "无",
    "mild": "轻度",
    "moderate": "中度",
    "marked": "明显",
}

#: 中文档位 → 档位 id（模型偶尔直接用中文作答）。
_LEVEL_BY_ZH = {zh: key for key, zh in ASD_LIKENESS_LEVEL_ZH.items()}
#: 常见同义写法 → 档位 id。
_LEVEL_ALIASES = {
    "absent": "none",
    "no": "none",
    "negative": "none",
    "normal": "none",
    "typical": "none",
    "slight": "mild",
    "low": "mild",
    "medium": "moderate",
    "moderate_high": "marked",
    "severe": "marked",
    "high": "marked",
    "pronounced": "marked",
    "significant": "marked",
}

#: **红线用词**：结论里出现这些词就说明它写成了临床诊断（本项目不允许）。
#: 只在结论里检查，不改写文本 —— 处理方式是告警，由人去看。
DIAGNOSIS_WORDS = ("确诊", "诊断", "患病", "患者", "病症", "疾病", "临床")

#: 结论文本里**允许**出现的「非诊断」自我声明用词（与 `DIAGNOSIS_WORDS` 一起用于消歧：
#: 「这不是诊断」这种自我否认也会命中关键词，但它恰恰是我们要的口径，不该被当成违规）。
_DISCLAIMER_HINTS = ("不是诊断", "非诊断", "不构成诊断", "不下诊断", "不是临床诊断", "非临床")


def normalize_level(value) -> str:
    """把模型给的程度写成合法档位；认不出的一律 `unknown`（不猜一个中间档）。"""
    text = str(value or "").strip().lower().replace(" ", "").replace("-", "_")
    if not text:
        return "unknown"
    if text in ASD_LIKENESS_LEVELS:
        return text
    if text in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[text]
    raw = str(value or "").strip()
    if raw in _LEVEL_BY_ZH:
        return _LEVEL_BY_ZH[raw]
    for zh, key in _LEVEL_BY_ZH.items():
        if zh in raw:
            return key
    return "unknown"


def normalize_asd_likeness(raw) -> dict:
    """归一化 `asd_likeness`；缺失或认不出时给 `unknown` 而不是替模型下结论。

    `present` 必须是真正的布尔判断：模型写 `"是"/"否"` 之类的字符串也认下来，
    但写不出可判定的值时留 `None` —— 界面据此显示「未给出」，绝不默认成「未表现出」。
    """
    if not isinstance(raw, dict):
        raw = {}
    present_raw = raw.get("present")
    present: bool | None = None
    if isinstance(present_raw, bool):
        present = present_raw
    elif isinstance(present_raw, (int, float)):
        present = bool(present_raw)
    else:
        text = str(present_raw or "").strip().lower()
        if text in ("true", "yes", "y", "1", "是", "有"):
            present = True
        elif text in ("false", "no", "n", "0", "否", "没有", "无"):
            present = False

    level = normalize_level(raw.get("level"))
    if level == "unknown" and present is False:
        # 「未表现出」但没给程度 → 程度就是「无」，这不是猜，是同一件事的另一种写法
        level = "none"
    if present is None and level in ASD_LIKENESS_LEVELS:
        present = level != "none"

    basis: list[str] = []
    for item in raw.get("basis") or ():
        text = str(item or "").strip()
        if text:
            basis.append(text[:200])
        if len(basis) >= 6:
            break
    return {"present": present, "level": level, "basis": basis}


def diagnosis_wording(text: str) -> list[str]:
    """返回结论里命中的**诊断式用词**（去重、保序）。空列表 = 没有违规用词。

    只看「像不像在写诊断」：出现「确诊」「诊断」等词且**没有**同时给出非诊断声明时才算命中。
    这一步是提示而非改写：文本照原样保留，由界面以警示色显示。
    """
    body = str(text or "")
    if not body:
        return []
    if any(hint in body for hint in _DISCLAIMER_HINTS):
        return []
    hits: list[str] = []
    for word in DIAGNOSIS_WORDS:
        if word in body and word not in hits:
            hits.append(word)
    return hits
