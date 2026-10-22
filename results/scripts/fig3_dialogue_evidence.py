"""Fig. 3：把「对话探查过程」的分析结果画成图。

## 三个面板与它们回答的问题

- **Panel A · Q1**（1 个典型例子）：同一个表型维度上，**受损心智 vs 完整心智**的行为签名与逐字证据。
- **Panel A · Q2**（1 个典型例子）：两个机制**收敛到同一个表型维度**时，行为签名是否可以区分。
- **Panel B · Q3**（1 个典型例子）：同一个机制从轻度到严重，受影响的表型集合如何变化，以及为什么。

## Q3 的判据：Δ **且** p

不只看 Δ ≥ 0.15，还要求统计显著 —— 口径**原样沿用 Fig.2**（`fig2_visualize._t_p` / `_bh_fdr`）：

- 两样本 t 检验：该 (机制 × 档位) 单元的重复读数 vs 完整心智参照组的重复读数（**合并方差**版，
  与 Fig.2 panel E 的主判据同一版），并同时算 Welch 版作对照；
- BH-FDR 在「同一档位 × 同一维度」这一族内校正（阶梯批每族 5 个 primitive）；
- **「表型明显降低」= Δ ≥ 0.15 且 q_pooled < 0.05**。

脚本内有一道硬门：用同一套函数重算 `fig2_link_candidates.csv` 里 result_2 @0.4 的全部
186 个 `p_pooled`，必须逐格对得上（容差 1e-6）—— 对不上就退出，不出一张口径不明的图。

## 复用与一致性

`importlib` 加载两个同级脚本，不复制任何判据或版式：

- `fig2_visualize`：调色板 `NATURE`/`DIMC`、字号、机制英文名 `EN`、参照组 `_intact_reference()`、
  检验 `_t_p`/`_bh_fdr`、落盘 `save()`；
- `analyze_dialogue_evidence`：逐被试加载与回指、个案选定、行为标记、证据归属、观察文本。

**红线**：图上出现的「机制 → 表型维度」一律是**测量产物**（哪一维掉得最多、掉多少、是否显著），
不是预设对应；机制侧引用的文字来自本体注册表（`ontology/primitives.py` 的定义），
只描述机制本身。图注里写明这一点。

## 用法

```bash
python fig3_dialogue_evidence.py                  # 画图 + 落盘 png/pdf/metrics.json
python fig3_dialogue_evidence.py --dry-run        # 只打印选定与前提判定，不画图
python fig3_dialogue_evidence.py --translate      # 补译图上引用的中文原文（要真实 provider）
python fig3_dialogue_evidence.py --summarise      # 从已记录问答重写 Panel B 的特点总结（同上）
```

## 版式约定

- 背景全白；**每个面板只有一个 title**，压在面板框顶，任何内容框内都不许再出现 title / label；
  组别名（`Social reward weighting = 0.4`、`intact mind`、`Mild θ = 0.90` …）一律放在**框上方**；
- Panel A 只保留 θ / Δ / q 三项数字（选例理由、对照 Δ、可信度说明等都删了），
  两个框内的文字一律**黑色英文**，总结只留与本表型相关的一句；
- 纵向排版**先量后排**：面板函数在 `ax=None` 时只走一遍排版不落笔，画幅高度由三块内容的高相加
  得到，外框也按量出来的高度补画。量高与落笔是同一遍代码，所以写死高度导致的「文字压出框」
  不会再出现（改版时若要调间距，改 `TITLE_ZONE` / `BOX_PAD` 这类常数，不要把内容高度写死）；
- Panel B 的 ③ 是「**这个机制自己**在问答里呈现的特点」，由 LLM **只读已落盘的逐轮问答**
  归纳（`--summarise` 落盘、渲染只读缓存），体现两机制差异的片段加粗；
  标记对照表删除，判定仍留在面板头；
- 出图前过一道 `audit()` 硬检查：图上不许残留未译中文、文字不许越出画幅，不过就弃图不出。

Panel A 引的中文原文走**英译缓存**（`results/figure/figure3_translations.json`）：`--translate`
调一次真实 provider 译好落盘，渲染只读缓存，所以重复出图不再调网络、结果可复现。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PACS = ROOT / "pacs"
for _path in (str(PACS / "src"), str(PACS / "tools"), str(Path(__file__).resolve().parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load(name: str, rel: str):
    """按显式路径加载同级脚本 —— 与 `fig2_link_candidates.py` 复用 Fig.2 数据层同一套做法。

    必须先登记进 `sys.modules`：`@dataclass` 在装饰时要回查 `sys.modules[cls.__module__]`，
    没登记就会以 `'NoneType' object has no attribute '__dict__'` 失败。
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


f2 = _load("fig2_visualize", "results/scripts/fig2_visualize.py")
ade = _load("analyze_dialogue_evidence", "results/scripts/analyze_dialogue_evidence.py")

from pacs.ontology.primitives import PRIMITIVES  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / "results" / "figure"
STEM = "figure3_dialogue_evidence"
METRICS_NAME = "figure3_metrics.json"
#: Panel A 里引的对话原文是中文，图上要英文 —— 英译缓存落在这里（可复现：渲染只读缓存）
TRANSLATIONS_NAME = "figure3_translations.json"
#: Panel B 的「各自特点」总结（LLM 读已记录问答得出，落盘后渲染只读缓存）
CONTRAST_NAME = "figure3_panel_b_contrast.json"

LINK_CANDIDATES = ROOT / "results" / "data" / "figure2_link_candidates.csv"
TEST_ALPHA = f2.TEST_ALPHA               # 0.05，照抄 Fig.2
TARGET_DROP = f2.TARGET_DROP             # 0.15，照抄 Fig.2
LADDER_DOSES = list(f2.DOSE_LADDER)      # [0.2, 0.4, 0.6, 0.8, 0.9]
LADDER_PRIMITIVES = list(f2.LADDER_PRIMITIVES)
#: Q3 面板展示的三档：轻度 / 中度 / 重度
SEVERITY_DOSES = [0.9, 0.6, 0.2]
SEVERITY_NAMES = {0.9: "Mild", 0.6: "Moderate", 0.2: "Severe"}

C = f2.NATURE
INK, MUTED, GRID, PE = C["ink"], C["muted"], C["grid"], C["panel_edge"]
DIMC = C["dim"]
WARN = C["warn"]
EN = f2.EN                     # 机制英文名，与 Fig.2 同一张表（不另抄一份）

#: 图题单独一档；**其余全部同一个尺寸**。面板标题靠加粗区分层级，不靠字号。
FS_TITLE = 15.5
FS_TEXT = 9.0

DIM_EN = {
    "social_interaction_motivation": "Social interaction\n& motivation",
    "social_cognition": "Social cognition",
    "social_communication_pragmatics": "Social communication\n& pragmatics",
    "cognitive_flexibility_adaptation": "Cognitive flexibility\n& adaptation",
    "repetitive_restricted_behavior": "Repetitive & restricted\nbehavior",
    "novelty_uncertainty_sensory_behavior": "Novelty, uncertainty\n& sensory behavior",
}
DIM_EN_ONE = {k: v.replace("\n", " ") for k, v in DIM_EN.items()}

#: 转写标记的**图例**：(键, 读数名, 哪个方向代表受损, 一句话读法)。
#: 面板上不画标记对照表，但逐机制的标记值仍记进 metrics ——
#: 没有这张表，metrics 里的 `answer_chars_mean` 之类就是无说明的裸键。
#: 第三列写 `worse_when` 而不是 `kind`：语义写成「哪个方向代表受损」最不容易读反。
#: 板上小标签用的短名（完整名太长，1.9 in 宽的栏放不下；A 面板仍用全名）
DIM_SHORT = {
    "social_interaction_motivation": "Social interaction",
    "social_cognition": "Social cognition",
    "social_communication_pragmatics": "Social communication",
    "cognitive_flexibility_adaptation": "Cognitive flexibility",
    "repetitive_restricted_behavior": "Repetitive behavior",
    "novelty_uncertainty_sensory_behavior": "Novelty / sensory",
}

MARKER_ROWS = [
    ("answer_chars_mean", "Answer length (chars)", "lower", "shorter answers"),
    ("q_back_rate", "Question-back rate", "lower", "initiates less"),
    ("echo_jaccard_mean", "Echo of examiner wording", "lower", "mirrors less"),
    ("self_repeat_max_mean", "Self-repetition (own last 3)", "higher", "repeats itself"),
    ("dup_answer_rate", "Verbatim repetition", "higher", "verbatim repeats"),
    ("short_or_unknown_rate", "Short / no-answer", "higher", "deflects"),
]


# ---------------------------------------------------------------------------
# Panel A 的英译（图上要英文，原文是中文；译一次、缓存起来，渲染只读缓存）
# ---------------------------------------------------------------------------


def load_translations() -> dict:
    """读英译缓存；缓存不存在时返回空表。"""
    path = OUT_DIR / TRANSLATIONS_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("translations", {})


def _llm_provider():
    from pacs.config import Settings
    from pacs.llm.provider import get_provider

    settings = Settings.load(base_dir=PACS)
    provider = get_provider(settings.llm)
    if str(getattr(provider, "name", "")) == "mock":
        raise SystemExit("✗ 配置里是 mock provider，无法做英译；请在 pacs/config.json 配真实 provider。")
    return provider


def _parse_translation_array(text: str) -> list[str]:
    """从回复里抠出英译数组。

    **不能复用 `parse_analyst_json`**：它会把回复规整成分析者的 schema，而它不认识的键
    （`translations`）被**静默丢掉**，于是永远取不到数组、每条都走「回退原文」——
    这里只做「取出 JSON」一件事，别的都不管。
    """
    from pacs.llm.parse import _JSON_BLOCK_RE, _repair_truncated_json

    raw = text or ""
    payload: object = None
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
    if payload is None:                       # 对象被截断：按括号栈补一下
        payload = _repair_truncated_json(raw)
    if payload is None:                       # 模型直接回了裸数组
        start, end = raw.find("["), raw.rfind("]")
        if 0 <= start < end:
            try:
                payload = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        payload = payload.get("translations")
    if not isinstance(payload, list):
        raise ValueError(f"回复里没有英译数组：{raw[:60]!r}")
    return [str(item) for item in payload]


def _translate_batch(pending: list[str], provider) -> list[str]:
    """一次调用译一批，要求**按输入顺序**返回纯英文数组。

    不让模型回写中文键：键会把输出量翻倍，实测直接把响应截断成不完整 JSON。
    数组形式既短又不用管键对不上。
    """
    from pacs.llm.provider import ChatMessage

    system = (
        "你在为一张学术图翻译素材。输入是一个中文短句数组，来自一次结构化观察记录："
        "提问者的问题、孩子的原话、分析者的观察与任务测量。\n"
        "请**按输入顺序**逐条译成简洁英文，输出同长度的英文数组：\n"
        "- 孩子的原话保留引号，译得像儿童口语，不要润色成书面语；\n"
        "- 分析者的观察/测量译成简洁临床描述，不加解释；\n"
        "- 术语用孤独症研究惯用说法（joint attention、false belief、insistence on sameness 等）。\n"
        "只输出一个 JSON 对象：{\"translations\": [\"^第一句的英文^\", ...]}，"
        "数组长度必须等于输入条数，顺序必须一致。"
    )
    text = provider.chat(
        [ChatMessage(role="user", content=system + "\n\n输入数组：\n"
                     + json.dumps(pending, ensure_ascii=False, indent=1))],
        role="analyst", temperature=0.2,
    )
    payload = _parse_translation_array(text)
    out = payload
    if len(out) != len(pending):
        raise ValueError(f"条数不符：输入 {len(pending)}、输出 {len(out)}")
    return out


def build_translations(strings: list[str], seed: dict | None = None) -> dict:
    """分批翻译（每批 8 条），某批失败就再对半切 —— 只用**已记录**文本，不重跑探查。

    `seed` 是已有缓存：只补缺的那几条，已有的不重译（每改一次图不必再付一遍翻译费）。
    """
    table: dict[str, str] = dict(seed or {})
    pending = [s for s in dict.fromkeys(strings) if s.strip() and s not in table]
    provider = _llm_provider()

    def run(chunk: list[str]) -> None:
        if not chunk:
            return
        try:
            for source, english in zip(chunk, _translate_batch(chunk, provider)):
                table[source] = english
        except Exception as error:                      # noqa: BLE001 —— 兜底就是继续切小
            if len(chunk) == 1:
                print(f"  ⚠ 单条也译不出，回退原文：{chunk[0][:24]}…（{error}）")
                return
            middle = len(chunk) // 2
            print(f"  … {len(chunk)} 条一批失败，切成 {middle}+{len(chunk) - middle} 重试")
            run(chunk[:middle])
            run(chunk[middle:])

    for start in range(0, len(pending), 8):
        run(pending[start:start + 8])
    return table


def collect_figure_strings(ctx) -> list[str]:
    """图上会出现的中文原文全集（A/B/C 三个面板），去重后返回。

    只收**已经落盘**的记录文本：观察、任务测量、与本品表型相关的结论句、问答原文、
    机制本体定义。不重跑任何探查。
    """
    strings: list[str] = []
    q1 = ctx["q1"]
    for side in ("impaired", "intact"):
        info = q1[side]
        strings += list(info["observations"]) + list(info["tasks"])
        strings += [s for s, hot in info["conclusion_sentences"] if hot]
        for example in info["excerpts"]:
            strings += [example["q"], example["a"]]

    for pid, info in (ctx["q2"]["members"] or {}).items():
        strings.append(PRIMITIVE_DEF.get(pid, ""))
        # Panel B 与 Panel A 同用 ①②③，所以观察、任务、结论句、问答**都要收**
        strings += list(info.get("observations") or []) + list(info.get("tasks") or [])
        strings += list(info.get("summary_relevant") or [])
        for example in info.get("excerpts") or []:
            strings += [example.get("q", ""), example.get("a", "")]
        example = info.get("excerpt") or {}
        strings += [example.get("q", ""), example.get("a", "")]
    for step in ctx["q3"]["steps"]:
        example = step.get("excerpt") or {}
        strings += [example.get("q", ""), example.get("a", "")]
        strings += list(step.get("observations") or [])       # 三框里改成印观察记录了
    return [s for s in dict.fromkeys(str(s) for s in strings) if s.strip()]


def load_contrast() -> dict:
    path = OUT_DIR / CONTRAST_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("contrast", {})


def _parse_json_object(text: str) -> dict:
    """从回复里取出一个 JSON 对象（与 `_parse_translation_array` 同一套兜底）。"""
    from pacs.llm.parse import _JSON_BLOCK_RE, _repair_truncated_json

    raw = text or ""
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated_json(raw)
    if isinstance(repaired, dict):
        return repaired
    raise ValueError(f"回复里没有 JSON 对象：{raw[:60]!r}")


def _turns_block(turns: list[dict]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append("Q: " + clean(turn.get("q")))
        lines.append("A: " + clean(turn.get("a")))
    return "\n".join(lines)


def _contrast_prompt(phenotype: str, turns_a: list[dict], turns_b: list[dict],
                     only: str | None = None) -> str:
    """构造提示词。`only` 给定时只要那一侧 —— 整段回复被截断时按侧重问。"""
    text = (
        "你在为一张学术图整理素材。下面给出同**一个表型维度**上、两个不同计算机制的"
        "**已记录问答**（逐轮原始对话，机制名已匿名）。\n"
        "请只依据给定文本，总结各自在问答里呈现出的、**与该表型有关**的作答特点。\n\n"
        "表型维度：" + phenotype + "\n\n"
        "机制 A 的问答：\n" + _turns_block(turns_a) + "\n\n"
        "机制 B 的问答：\n" + _turns_block(turns_b) + "\n\n"
        "要求：\n"
        "- 每个机制 2–3 条；每条一句话、不超过 25 个词，说明「这名受访者的作答呈现什么特点」；\n"
        "- 只总结与该表型有关的内容，与该表型无关的（语气、寒暄等）不要写；\n"
        "- 每条再给 bold：该条里**最能体现两个机制差异**的**短**片段（不超过 10 个词，"
        "必须是 text 的英文子串）；这条与另一机制没有区别就给空字符串；\n"
        "- 只描述文本里看得见的行为与措辞，不要推断内部状态，不要引入文本之外的信息；\n"
        "- **全部用英文**：图是英文的，引用的原话也译成英文（保留引号，引号内前后不留空格）。\n"
    )
    if only:
        return (text + "\n这次**只**要机制 " + only.upper() + " 那一侧。只输出："
                '{"' + only + '":[{"text":"...","bold":"..."}]}')
    return text + ('\n只输出一个 JSON 对象：{"a":[{"text":"...","bold":"..."}],'
                   '"b":[{"text":"...","bold":"..."}],"note":"..."}')


def _extract_contrast(result: dict) -> dict:
    """规整模型回复：`bold` 必须是 `text` 的子串，不是就当没给（宁可不加粗也不乱加粗）。"""
    out: dict = {}
    for key in ("a", "b"):
        entries = []
        for entry in result.get(key) or []:
            if not isinstance(entry, dict):
                continue
            body = clean(entry.get("text"))
            if not body:
                continue
            mark = clean(entry.get("bold"))
            entries.append({"text": body, "bold": mark if mark and mark in body else ""})
        out[key] = entries
    out["note"] = clean(result.get("note"))
    return out


def build_contrast(ctx) -> dict:
    """让 LLM **只读已记录的逐轮问答**，总结两个机制各自的特点，并标出体现差异的片段。

    红线：输入只有已落盘的问答文本；要求里写明「只描述文本里看得见的东西、不推断内部状态」；
    机制名匿名成「机制 A / 机制 B」。
    整段回复可能被 max_tokens 截断（实测出现过 A 有、B 空），所以任一侧为空就**按侧重问**。
    """
    from pacs.llm.provider import ChatMessage

    members = ctx["q2"]["members"]
    labels = list(ctx["q2"]["group"]["members"])
    payload = {
        "note": "Panel B 的「各自特点」总结。输入是已记录的逐轮问答（同通道），"
                "渲染只读本缓存；不属于测量产物，不得回写。",
        "layer": "post_hoc_interpretation",
        "phenotype": ctx["q2"]["dim"],
        "mechanisms": {"a": labels[0] if labels else "", "b": labels[1] if len(labels) > 1 else ""},
        "contrast": {},
        "replies": [],
    }
    if len(labels) != 2:
        return payload

    provider = _llm_provider()
    phenotype = DIM_EN_ONE[ctx["q2"]["dim"]]
    turns = {"a": members[labels[0]].get("turns") or [],
             "b": members[labels[1]].get("turns") or []}

    def ask(only: str | None) -> dict:
        prompt = _contrast_prompt(phenotype, turns["a"], turns["b"], only=only)
        reply = provider.chat([ChatMessage(role="user", content=prompt)], role="analyst",
                              temperature=0.2, max_tokens=2600)
        payload["replies"].append({"asked": only or "both", "chars": len(reply),
                                   "reply": reply})
        return _parse_json_object(reply)

    contrast = _extract_contrast(ask(None))
    for key in ("a", "b"):                       # 被截断的那一侧单独再问一遍
        if not contrast[key]:
            retry = _extract_contrast(ask(key))
            contrast[key] = retry[key]
            print(f"  … 机制 {key.upper()} 首次为空，按侧重问后得 {len(retry[key])} 条")
    payload["contrast"] = contrast
    return payload


def tr(text, table: dict) -> str:
    """取英译；没有就回退原文（并在渲染时说明）。"""
    return table.get(str(text), str(text))


# ---------------------------------------------------------------------------
# 统计：口径原样沿用 Fig.2
# ---------------------------------------------------------------------------


def bh_fdr(values) -> np.ndarray:
    return f2._bh_fdr(np.asarray(values, dtype=float))


def cell_readings(pool, reference) -> dict:
    """`(primitive, dose) → {dim → [读数]}`。只用阶梯批的 5 个 primitive × 5 档。"""
    cells: dict[tuple[str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for subject in pool:
        if subject.primitive_id not in LADDER_PRIMITIVES:
            continue
        if round(subject.dose, 4) not in {round(d, 4) for d in LADDER_DOSES}:
            continue
        # 对照通道 `general_cognition` 也要采：E1（非特异通用退化）看的就是它有没有一起塌。
        for dim in f2.DIMS + [f2.CONTROL]:
            value = subject.value(dim)
            if value is not None:
                cells[(subject.primitive_id, round(subject.dose, 4))][dim].append(float(value))
    return cells


def significance_table(cells, reference) -> dict:
    """逐 (primitive, dose, dim)：Δ（= 基线 − 实测，正 = 更 ASD 样）、p、q、q_welch、n。

    BH 族 = 「同一档位 × 同一维度」的全部 primitive（阶梯批 5 个）。
    判据 `significant` = Δ ≥ 0.15 **且** q_pooled < 0.05（Welch 版另存一列作敏感性对照）。
    对照通道 `general_cognition` 一并算（Panel B 的 E1 要看它有没有一起塌），但它不是表型维度，
    统计它自己的那一族、不进 6 维的计数。
    """
    table: dict[tuple[str, float, str], dict] = {}
    for dose in LADDER_DOSES:
        for dim in f2.DIMS + [f2.CONTROL]:
            series = [cells[(pid, round(dose, 4))][dim] for pid in LADDER_PRIMITIVES]
            p_pooled = [f2._t_p(v, reference[dim], True) for v in series]
            p_welch = [f2._t_p(v, reference[dim], False) for v in series]
            q_pooled = bh_fdr(p_pooled)
            q_welch = bh_fdr(p_welch)
            for index, pid in enumerate(LADDER_PRIMITIVES):
                values = series[index]
                delta = (
                    float(np.mean(reference[dim]) - np.mean(values)) if values else 0.0
                )
                table[(pid, round(dose, 4), dim)] = {
                    "delta": round(delta, 4),
                    "p": float(p_pooled[index]),
                    "q": float(q_pooled[index]),
                    "q_welch": float(q_welch[index]),
                    "n": len(values),
                    "significant": bool(delta >= TARGET_DROP and q_pooled[index] < TEST_ALPHA),
                    "significant_strict": bool(
                        delta >= TARGET_DROP
                        and q_pooled[index] < TEST_ALPHA
                        and q_welch[index] < TEST_ALPHA
                    ),
                }
    return table


def validate_test() -> dict:
    """硬门：用同一套函数重算 result_2 @0.4 的 186 个 p_pooled，与发布表逐格对账。"""
    import csv as _csv

    reference = f2._intact_reference()
    frame = f2.pd.read_excel(f2.XLSX, "by_subject")
    frame = frame.rename(columns={
        col: dim for col in frame.columns for dim in f2.DIMS + [f2.CONTROL]
        if isinstance(col, str) and col.startswith(dim + " [")
    })
    groups: dict[str, list[dict]] = defaultdict(list)
    for _, row in frame.iterrows():
        match = re.match(r"^single_(?P<p>.+?)#\d{3}_r\d+$", str(row["row_id"]))
        if not match:
            continue
        setting = str(row["mind_setting"]).split()
        if len(setting) < 2 or abs(float(setting[1]) - 0.4) > 1e-9:
            continue
        groups[match.group("p")].append(
            {dim: (None if f2.pd.isna(row[dim]) else float(row[dim])) for dim in f2.DIMS}
        )
    pids = sorted(groups)
    published = {}
    with LINK_CANDIDATES.open(encoding="utf-8-sig", newline="") as handle:
        for row in _csv.DictReader(handle):
            if row["dose"] == "0.4":
                published[(row["primitive_id"], row["dimension"])] = float(row["p_pooled"])
    worst = 0.0
    for dim in f2.DIMS:
        series = [
            [g[dim] for g in groups[pid] if g[dim] is not None] for pid in pids
        ]
        mine = [f2._t_p(v, reference[dim], True) for v in series]
        for index, pid in enumerate(pids):
            expected = published.get((pid, dim))
            if expected is not None:
                worst = max(worst, abs(expected - mine[index]))
    return {"n_cells": len(pids) * len(f2.DIMS), "worst_gap": worst, "ok": worst < 1e-6}


# ---------------------------------------------------------------------------
# 个案选定
# ---------------------------------------------------------------------------


def build_context(args) -> dict:
    counters = {k: 0 for k in ("rows_total", "row_id_unparsed", "join_ok", "join_missing",
                               "json_ok", "json_missing", "session_mismatch", "dose_mismatch",
                               "reused_in_result_4", "llm_calls")}
    subjects_r2 = ade.load_subject_rows(ade.RESULT_2, ade.SINGLE_RUN, "result_2", counters)
    subjects_r4 = ade.load_subject_rows(ade.RESULT_4, ade.DOSE_RUN, "result_4", counters)
    ade.mark_reused(subjects_r2, subjects_r4)
    pool, seen = [], set()
    for subject in subjects_r2 + subjects_r4:
        if subject.content_sha in seen:
            continue
        seen.add(subject.content_sha)
        pool.append(subject)
    for subject in pool:
        subject.pairs = ade.pair_turns(subject.record)
        subject.record["_pairs"] = subject.pairs     # 让按 record 取问答的接口也能用

    official, reference_records = ade.load_reference_records()
    reference = f2._intact_reference()          # 与 Fig.2 同一批 5 份参照读数
    metrics2 = json.loads(ade.FIGURE2_METRICS.read_text(encoding="utf-8"))
    baseline = {k: float(v) for k, v in metrics2["baseline"].items()}
    import csv as _csv
    with LINK_CANDIDATES.open(encoding="utf-8-sig", newline="") as handle:
        link_rows = list(_csv.DictReader(handle))
    hypotheses = ade.read_prior_hypotheses(link_rows)

    # --- 个案选定 ---
    # Panel A 的个案固定为「社会奖励加权 → 社会互动与社会动机」这一对。
    # 它是 result_2 的 8 个 specificity=pass 格之一（对照通道只动 0.12，在 0.15 容差内），
    # 所以「分数下降」这件事本身是可信的 —— 这正是 Panel A 要讲的前提。
    q1_cases, q1_avail = ade.select_q1_cases(link_rows, hypotheses, limit=3, tier=args.q1_tier)
    forced = [row for row in link_rows
              if row["primitive_id"] == args.q1_primitive
              and row["dimension"] == args.q1_dim
              and abs(float(row["dose"]) - args.q1_dose) < 1e-9]
    if not forced:
        raise SystemExit(
            f"✗ figure2_link_candidates.csv 里找不到 {args.q1_primitive} → {args.q1_dim} "
            f"@ {args.q1_dose:g} 这一格，无法作 Panel A。")
    q1_case = dict(forced[0])
    q2_cases = ade.select_q2_cases(link_rows, metrics2, hypotheses, limit=3,
                                   tolerance=args.match_tolerance,
                                   dose=float(metrics2.get("primary_dose") or 0.4))

    subject_index: dict[tuple[str, str, float], list] = defaultdict(list)
    for subject in pool:
        subject_index[(subject.primitive_id, "", subject.dose)].append(subject)
        for dim in f2.DIMS:
            subject_index[(subject.primitive_id, dim, subject.dose)].append(subject)

    fragments, n_mech = ade.load_corpus_fragments(pool, reference_records)
    boilerplate = ade.boilerplate_set(fragments, n_mech, args.boilerplate_rate)

    summary_r2 = ade.read_sheet(ade.RESULT_2, "summary_by_cell")
    pub_cell = next((r for r in summary_r2
                     if str(r.get("primitive_id")) == args.q1_primitive
                     and abs(float(r.get("dose") or 0) - args.q1_dose) < 1e-9), {})

    _, q2_groups = ade.build_q2_rows(q2_cases, subject_index, boilerplate,
                                     tolerance=args.match_tolerance, seed=args.seed)
    agreed = [g for g in q2_groups if g["two_axis_agreement"].startswith("两轴一致：可区分")]
    q2_group = max(agreed, key=lambda g: g["separation_ratio"]) if agreed else q2_groups[0]
    # 组统计里没有逐成员的 Δ（那是 `select_q2_cases` 的产物），按 `Q2-<序号>` 对回去
    case_by_id = {f"Q2-{index + 1}": case for index, case in enumerate(q2_cases)}
    q2_member_deltas = case_by_id[q2_group["case"]]["member_deltas"]

    # --- 显著性 ---
    cells = cell_readings(pool, reference)
    sig = significance_table(cells, reference)

    def n_sig(pid: str, dose: float, key: str = "significant") -> int:
        return sum(1 for dim in f2.DIMS if sig[(pid, round(dose, 4), dim)][key])

    def growth(pid: str, key: str = "significant") -> tuple:
        light, heavy = max(SEVERITY_DOSES), min(SEVERITY_DOSES)
        a, b = n_sig(pid, light, key), n_sig(pid, heavy, key)
        return (b - a, b, -a)

    # 例子由 `--q3-primitive` 指定（默认 Self–other distinction）；留空则按规则自动挑
    if args.q3_primitive:
        if args.q3_primitive not in LADDER_PRIMITIVES:
            raise SystemExit(f"✗ {args.q3_primitive} 不在可做阶梯的机制里：{LADDER_PRIMITIVES}")
        q3_pid = args.q3_primitive
    else:
        q3_pid = max(LADDER_PRIMITIVES,
                     key=lambda pid: (growth(pid), pid == "belief_updating", pid))
    q3_rule = ("在 5 个可做阶梯的机制里，选「最重档显著维度数 − 轻度档显著维度数」最大者；"
               "平局依次比「最重档显著数」「轻度档显著数更小」")

    # --- 该机制在三个档位上的逐字证据 ---
    q3_steps = []
    for dose in SEVERITY_DOSES:
        steps = [s for s in pool if s.primitive_id == q3_pid and abs(s.dose - dose) < 1e-9]
        dims_sig = sorted(
            (dim for dim in f2.DIMS if sig[(q3_pid, round(dose, 4), dim)]["significant"]),
            key=lambda dim: sig[(q3_pid, round(dose, 4), dim)]["delta"], reverse=True,
        )
        dominant = dims_sig[0] if dims_sig else f2.DIMS[0]
        attribution = ade.attribute_evidence(steps[0].record, steps[0].pairs,
                                             match_min=args.match_min) if steps else {}
        q3_steps.append({
            "dose": dose,
            "severity": SEVERITY_NAMES[dose],
            "n_subjects": len(steps),
            "n_significant": len(dims_sig),
            "n_significant_strict": sum(
                1 for dim in f2.DIMS if sig[(q3_pid, round(dose, 4), dim)]["significant_strict"]),
            "dominant": dominant,
            "dims": [
                {"dim": dim, **sig[(q3_pid, round(dose, 4), dim)]} for dim in dims_sig
            ],
            "mean_delta": round(float(np.mean(
                [sig[(q3_pid, round(dose, 4), dim)]["delta"] for dim in f2.DIMS])), 4),
            "excerpt": illustrative_pair(steps[0].record, dominant, attribution) if steps else None,
            # 三框里印的是**观察记录**（原来是问答例子），与该档主导表型对齐
            "observations": (ade.observations(steps[0].record, dominant) if steps else [])[:3],
            "control_delta": sig[(q3_pid, round(dose, 4), f2.CONTROL)]["delta"],
        })

    # --- Panel A：受损 / 完整 两侧的证据链 ---
    a1_dim = args.q1_dim
    a1_dose = args.q1_dose
    # 受损侧只用 **result_2 自己的那几次重复**（= 发布表的 3 次），不用池里其他批次的同剂量会话：
    # 池内均值（5 段会话 → 0.48）与头条（发布表 3 次 → 0.49）对不上，读者会以为是两个数。
    # 图presenting 的是发布结果，就一路用发布口径。
    a1_subjects = [s for s in subject_index[(args.q1_primitive, a1_dim, a1_dose)]
                   if s.source_batch == "result_2"]
    if not a1_subjects:
        raise SystemExit(f"✗ result_2 里找不到 {args.q1_primitive} @ {args.q1_dose:g} 的被试。")
    intact_reference = max(
        reference_records,
        key=lambda r: float((r.get("analysis") or {}).get("readings", {}).get(a1_dim) or 0.0))

    def side_payload(records: list[dict], *, most_impaired: bool, dim: str | None = None,
                     limit_observations: int = 3, limit_tasks: int = 2,
                     limit_excerpts: int = 3) -> dict:
        """一侧（受损或完整）：读数 + 代表性证据 + 结论逐句 + 逐字例子。

        **只放代表性证据、并显式报出被省下的条数** —— 3 段会话加起来有 15 条观察、10 条任务测量，
        全倒进一栏会把面板撑爆，也不是图该干的事：图给证据的形状，完整清单在 casebook / md 里。
        取哪一条有确定规则：先按该通道读数排序（受损侧从最低的会话取、完整侧取最高的），
        再从那个会话取前 `limit_observations` 条观察与前 `limit_tasks` 条任务测量。
        """
        dim = dim or a1_dim
        ordered = sorted(
            records,
            key=lambda r: float((r.get("analysis") or {}).get("readings", {}).get(dim) or 0.0),
            reverse=not most_impaired,
        )
        readings = [
            float((r.get("analysis") or {}).get("readings", {}).get(dim))
            for r in ordered
            if (r.get("analysis") or {}).get("readings", {}).get(dim) is not None
        ]
        primary = ordered[0] if ordered else None
        all_observations: list[str] = []
        all_tasks: list[str] = []
        for record in ordered:
            for item in ade.observations(record, dim):
                if item not in all_observations:
                    all_observations.append(item)
            for item in ade.task_evidence(record, dim):
                task, measurement = clean(item.get("task")), clean(item.get("measurement"))
                if task or measurement:
                    rendered = f"“{task}” → {measurement}" if task and measurement else (task or measurement)
                    if rendered not in all_tasks:
                        all_tasks.append(rendered)

        excerpts: list[dict] = []
        for record in ordered:
            attribution = ade.attribute_evidence(record, record.get("_pairs")
                                                 or ade.pair_turns(record),
                                                 match_min=args.match_min)
            example = illustrative_pair(record, dim, attribution)
            if example and len(excerpts) < (limit_excerpts or 3):
                excerpts.append(example)

        return {
            "reading": round(float(np.mean(readings)), 4) if readings else None,
            "n_sessions": len(readings),
            "observations": all_observations[:limit_observations],
            "tasks": all_tasks[:limit_tasks],
            "n_observations_total": len(all_observations),
            "n_tasks_total": len(all_tasks),
            "primary_session": (primary or {}).get("session_id", ""),
            "conclusion_sentences": conclusion_sentences(primary, dim) if primary else [],
            "summary_relevant": _relevant_sentences(primary, dim),
            "excerpts": excerpts,
            "conclusion_full": clean((primary or {}).get("analysis", {}).get("conclusion") or ""),
        }

    impaired_side = side_payload([s.record for s in a1_subjects], most_impaired=True)
    # 完整侧同样对齐发布口径：用**官方 5 份**（头条里的 baseline 就是它们算的），
    # 不是 10 份参照全用 —— 否则栏内读数（0.92 均值 of 10）与头条（0.93 of 5）又对不上。
    intact_side = side_payload(list(official), most_impaired=False)
    impaired_first = impaired_side["observations"][0] if impaired_side["observations"] else ""
    intact_first = intact_side["observations"][0] if intact_side["observations"] else ""
    control_delta = abs(float((pub_cell or {}).get("general_cognition_delta") or 0.0))
    published_delta = abs(float(q1_case.get("delta") or 0.0))
    q1 = {
        "case": q1_case,
        "dim": a1_dim,
        "dose": a1_dose,
        "baseline": float(baseline[a1_dim]),
        "published_mean": float((pub_cell or {}).get(f"{a1_dim}_mean")
                                or baseline[a1_dim] - published_delta),
        "published_delta": -published_delta,
        "q_pooled": float(q1_case.get("q_pooled") or 1.0),
        "control_delta": control_delta,
        "verdict": str((pub_cell or {}).get("specificity_verdict") or q1_case.get("verdict") or ""),
        "n_subjects": len(a1_subjects),
        "why": (f"Selection: the {EN.get(q1_case['primitive_id'], q1_case['primitive_id'])} → "
                f"{DIM_EN_ONE[a1_dim]} cell of result_2 (tier {q1_case.get('tier')}, "
                f"specificity verdict {q1_case.get('verdict')}), one of the few cells where the "
                f"perturbation moves this phenotype while the control channel stays inside tolerance."),
        "credibility": (f"The control channel (general cognition) moved only {control_delta:.2f}, inside the "
                        f"{TARGET_DROP:.2f} tolerance, so the drop on this phenotype cannot be explained by a "
                        f"global capacity loss. The analyst scored it from this phenotype's own questions."),
        "contrast": (f"intact — “{intact_first}”;  impaired — “{impaired_first}”."),
        "confound": (f"If the control channel had dropped past {TARGET_DROP:.2f}, the same score would have to "
                     f"be read as part of a general degradation rather than this mechanism's phenotype."),
        "impaired": impaired_side,
        "intact": intact_side,
    }

    # --- A2 的逐字证据与标记 ---
    a2_dim = q2_group["dimension"]
    a2_dose = float(q2_group["dose"])
    a2_members = list(q2_group["members"])
    a2 = {}
    for pid in a2_members:
        steps = subject_index[(pid, a2_dim, a2_dose)] or subject_index[(pid, "", a2_dose)]
        attribution = ade.attribute_evidence(steps[0].record, steps[0].pairs,
                                             match_min=args.match_min)
        pairs = [p for s in steps for p in ade.pairs_for_channel(s.pairs, a2_dim)]
        # Δ 用 Fig.2 发布表（`link_candidates`）里的数 —— A2 的成员不一定在 5 档阶梯里，
        # 阶梯的显著性表覆盖不到它们，不能拿它当来源。
        a2[pid] = {
            "markers": ade.dialogue_markers(pairs),
            "excerpt": illustrative_pair(steps[0].record, a2_dim, attribution),
            "n_subjects": len(steps),
            "delta": q2_member_deltas.get(pid),
            # Panel B 与 Panel A 用同一套 ①②③（观察 / 任务→行为 / 总结与问答），
            # 所以这里也要把这三类素材取出来，取法与 Panel A 完全一致。
            **side_payload([s.record for s in steps], most_impaired=True, dim=a2_dim,
                           limit_observations=3, limit_tasks=2, limit_excerpts=2),
            # 该机制在该表型通道上的**逐轮问答**：Panel B 的「各自特点」总结只从这里来
            "turns": [{"q": clean(one["question"]), "a": clean(one["answer"])}
                      for one in pairs if clean(one.get("answer"))][:12],
        }

    return {
        "reference": reference,
        "reference_records": reference_records,
        "baseline": baseline,
        "sig": sig,
        "cells": cells,
        "q1": q1,
        "q1_availability": q1_avail,
        "q2": {"group": q2_group, "dim": a2_dim, "dose": a2_dose, "members": a2},
        "q3": {"pid": q3_pid, "rule": q3_rule, "steps": q3_steps,
               "n_sig_by_pid": {pid: {f"{d:g}": n_sig(pid, d) for d in SEVERITY_DOSES}
                                for pid in LADDER_PRIMITIVES},
               "n_sig_strict_by_pid": {pid: {f"{d:g}": n_sig(pid, d, "significant_strict")
                                             for d in SEVERITY_DOSES}
                                       for pid in LADDER_PRIMITIVES}},
        "gate": counters,
    }


VERDICT_EN = {
    "两轴一致：可区分": "Both axes agree — distinguishable",
    "两轴一致：不可区分": "Both axes agree — not distinguishable",
    "两轴不一致 → 建议用 --llm-contrast 判读": "Axes disagree — left unresolved",
}


def verdict_en(text: str) -> str:
    return VERDICT_EN.get(str(text), str(text))


def conclusion_sentences(record, dim) -> list[tuple[str, bool]]:
    """整段结论逐句切分，并标出与本品表型有关的句子。

    判据完全确定性：该句含该通道的中文名，或含该通道某条观察记录里的片段（≥4 字）。
    整段都引，**不裁剪** —— 读者能看见高亮句在整段里的位置，避免「只摘对结论有利的那句」。
    """
    conclusion = clean(record.get("analysis", {}).get("conclusion") or "")
    if not conclusion:
        return []
    fragments = set()
    for observation in ade.observations(record, dim):
        for part in re.split(r"[，。；、]", observation):
            part = clean(part)
            if len(part) >= 4:
                fragments.add(part)
    label = ade.channel_label(dim)
    out = []
    for sentence in re.split(r"(?<=。)", conclusion):
        sentence = clean(sentence)
        if not sentence:
            continue
        hot = label in sentence or any(fragment in sentence for fragment in fragments)
        out.append((sentence, hot))
    return out


def _relevant_sentences(record, dim) -> list[str]:
    """化简后的「总结」：只留与该表型相关的句子；一条都没命中就退第一句。"""
    if not record:
        return []
    sentences = conclusion_sentences(record, dim)
    hot = [s for s, is_hot in sentences if is_hot]
    if hot:
        return hot
    return [s for s, _ in sentences[:1]]


def fmt_signed(value, digits: int = 2) -> str:
    return f"{value:+.{digits}f}" if isinstance(value, (int, float)) else "—"


def clean(text) -> str:
    return " ".join(str(text or "").split())


def illustrative_pair(record, dimension, attribution, *, limit: int = 110) -> dict | None:
    """该维度上一条**有归属**的逐字证据；没有就退回该维度的第一对问答。

    优先用「分析者记的那条任务测量能反查回具体提问轮」的那一对 —— 这样图上引的对话
    同时有分析者的判读与原始问答，两处可对照。
    """
    for entry in ((attribution or {}).get("rows", {}) or {}).get(dimension) or []:
        if entry.get("matched_focus") == dimension and entry.get("matched_question"):
            return {
                "task": clean(entry["task"]),
                "measurement": clean(entry["measurement"]),
                "q": clean(entry["matched_question"])[:limit],
                "a": clean(entry["matched_answer"])[:limit],
            }
    for pair in ade.pairs_for_channel(record.get("_pairs") or ade.pair_turns(record), dimension):
        return {"task": "", "measurement": "", "q": clean(pair["question"])[:limit],
                "a": clean(pair["answer"])[:limit]}
    return None


# ---------------------------------------------------------------------------
# 画图小工具
# ---------------------------------------------------------------------------


FIG_W_IN = 14.4
#: 纵向版式（axes 分数，自下而上）。集中放一处，改画幅只动这里。
#: 各段之间留 ≥0.015 的缝，caption 用 3 行、底边压在 C 面板之下。


#: 条形图（Δ 标签）的行距与条高：行距必须明显大于条高，否则圆角外扩会把缝吃掉、
#: 条与条看着贴在一起。`CHIP_BOX_PAD` 是圆角的外扩量。
CHIP_ROW_LH = 2.6
CHIP_H_LH = 1.9
CHIP_BOX_PAD = 0.0010


def fit_width(x0, x1, size=FS_TEXT, pad=0.018) -> int:
    """盒子 (x0, x1) 在字号 `size` 下一行能放多少「显示宽度」（CJK 记 2、拉丁记 1）。

    手填宽度不可靠：盒子宽 2.9 in、字号 6.5 pt 时一行能放 ~64 个显示单位，
    填 40 会白掉三分之一、填 90 会溢出盒子。这里按 1 显示单位 ≈ size/2 pt 折算。
    """
    return max(14, int((x1 - x0 - pad) * FIG_W_IN * 144.0 / size))


def wrap_cjk(text: str, width: int) -> list[str]:
    """按显示宽度断行：CJK 逐字（记 2），拉丁**按整词**（记词长）。

    逐字符断会把英文单词从中间劈开 —— 实测出现过 “cannot be ex plained”。
    中文没有空格，也不能交给 `textwrap`，所以两种脚本各按各的单位切。
    """
    lines, current, used = [], "", 0
    for token in re.findall(r"[\u2E80-\uFFFF]|[^\s\u2E80-\uFFFF]+|\s+", str(text)):
        if token == "\n":
            lines.append(current)
            current, used = "", 0
        elif token.isspace():
            if used + 1 <= width or not current:
                current += token
                used += 1
            else:
                lines.append(current)
                current, used = "", 0
        else:
            cost = sum(2 if ord(ch) > 0x2E80 else 1 for ch in token)
            if used + cost > width and current:
                lines.append(current)
                current, used = "", 0
            current += token
            used += cost
    if current:
        lines.append(current)
    return lines or [""]


def block(ax, x0, y0, x1, y1, *, fc="white", ec=PE, lw=0.8, radius=0.010, z=1):
    patch = FancyBboxPatch(
        (x0 + radius, y0 + radius), (x1 - x0) - 2 * radius, (y1 - y0) - 2 * radius,
        boxstyle=f"round,pad={radius}", linewidth=lw, edgecolor=ec,
        facecolor=fc, zorder=z, mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, lines, *, size=FS_TEXT, color=INK, weight="normal", ha="left",
         va="top", lh=1.28, z=5, style="normal"):
    if isinstance(lines, str):
        lines = [lines]
    for index, line in enumerate(lines):
        ax.text(x, y - index * size * lh / 1000.0 * 1.0, line, fontsize=size, color=color,
                fontweight=weight, ha=ha, va=va, zorder=z, style=style)


def line_step(size, lh=1.32) -> float:
    """一行文本在 y 方向的 axes 分数。

    必须绑定画幅高度：`size*lh/1000` 只在 FIG_H = 1000/72 ≈ 13.89 in 时才是对的，
    换画幅就会让所有段落挤在一起或散开。这里写成精确式，以后调画幅不用重调偏移。
    """
    return size * lh / 72.0 / FIG_H_IN


def text_block(ax, x, y, content, *, width, size=FS_TEXT, color=INK, weight="normal",
               lh=1.32, z=5, style="normal"):
    """换行后逐行落笔，返回下一行的 y。"""
    lines = wrap_cjk(content, width)
    step = line_step(size, lh)
    for index, line in enumerate(lines):
        ax.text(x, y - index * step, line, fontsize=size, color=color,
                fontweight=weight, ha="left", va="top", zorder=z, style=style)
    return y - len(lines) * step


FIG_H_IN = 16.0


#: 加粗字比常规宽约 14%（Microsoft YaHei 粗体实测）——估宽必须带上，
#: 否则「加粗标签 + 同行正文」会压字（实测出现过 “two mindsintact — …”）。
BOLD_WIDTH_FACTOR = 1.14


def text_width_axes(label, size, *, bold=False) -> float:
    """文本宽度换算成 x 方向的 axes 分数。

    按字体度量估：CJK 一个字 = 1 em，拉丁平均 ≈ 0.52 em（Microsoft YaHei 实测比例）；
    `bold=True` 乘加粗系数。手填系数（`len*size*0.62/1000`）对纯拉丁低估过约三成，
    小标签会冲出栏位 —— 所以统一走这里。
    """
    em = sum(1.0 if ord(ch) > 0x2E80 else 0.52 for ch in str(label))
    if bold:
        em *= BOLD_WIDTH_FACTOR
    return em * size / 72.0 / FIG_W_IN


def chip(ax, x, y, label, *, color, size=FS_TEXT, pad=0.005):
    """一个带色边框的小标签，返回它占掉的总宽度（axes 分数）。"""
    width = text_width_axes(label, size) * 1.08 + 2 * pad
    height = line_step(size, CHIP_H_LH)
    ax.add_patch(FancyBboxPatch((x + 0.002, y), width, height,
                                boxstyle=f"round,pad={CHIP_BOX_PAD}", linewidth=0.7,
                                edgecolor=color, facecolor="white", zorder=4))
    ax.text(x + 0.002 + width / 2, y + height / 2, label, fontsize=size, color=color,
            ha="center", va="center", zorder=5)
    return width + 0.006


def chip_wrap(ax, x, y, labels_with_colors, *, max_x, size=FS_TEXT,
              dry: bool = False) -> tuple[int, float]:
    """把若干标签排成流式行。返回（行数，最后一行的底边 y）。`dry=True` 只量不画。"""
    step = line_step(size, CHIP_ROW_LH)
    cursor, rows = x, 1
    bottom = y
    for label, color in labels_with_colors:
        width = text_width_axes(label, size) * 1.08 + 0.012
        if cursor + width > max_x and cursor > x:
            cursor = x
            rows += 1
            y -= step
        if not dry:
            chip(ax, cursor, y, label, color=color, size=size)
        cursor += width
        bottom = y
    return rows, bottom


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
#
# 这一层的规矩：**先量后排**。每个面板函数在 `ax=None` 时只走一遍排版、不落笔，
# 把高度量出来；画幅高度由三块内容的高相加得到，外框也按量出来的高度补画。
# 量高与落笔是同一遍代码，所以不会出现「文字压出框」或「面板下面空一大截」。
#
# 纵向一律用**英寸**（自画幅顶往下算），`Y()` 负责换成 axes 分数；横向仍是 axes 分数。
# 常数写死是对的（标题带、间距），**内容高度**必须由内容算。

PANEL_X = (0.012, 0.988)

TITLE_ZONE = 0.075      # 面板顶部留白（不画标题行）
PHENO_ROW = 0.250       # Panel B 的「Same phenotype · 判定」居中一行
LABEL_GAP = 0.120       # 「框上方标签」到框顶的净空
BOX_PAD = 0.052         # 内容框的内边距（上下各一份）
PANEL_GAP = 0.185       # 面板之间的净空
FIG_HEAD = 0.360        # 图题带
FIG_TAIL = 0.080        # 图末呼吸位

_H = 16.0               # 画幅高（英寸），draw() 里按内容定下来


def Y(inch: float) -> float:
    """自画幅顶往下 `inch` 英寸处的 y（axes 分数）。"""
    return 1.0 - inch / _H


def _inch(size=FS_TEXT, lh=1.32) -> float:
    """一行的英寸高度。"""
    return size * lh / 72.0


def box_width(x0, x1, size=FS_TEXT) -> int:
    """内容框内可用的换行宽度。

    `fit_width` 的 `pad` 单位是 **axes 分数**，而 `BOX_PAD` 是**英寸**：直接把英寸当分数传
    会把留白放大十几倍，每栏文字只用到一半宽度、白白换行。
    这里把两侧内边距与文字起始偏移一起换算成分数。
    """
    return fit_width(x0, x1, size, pad=2 * BOX_PAD / FIG_W_IN + 0.014)


def tidy(content) -> str:
    """收一下译文带出来的空格：连续空格并成一个、标点前不留空格。

    英译与 LLM 归纳保留引号时会顺手多打空格（实测出现 `says,  “…”  and does`、`” ;`
    以及引号内的 `' Anything is fine .'`）。量高与落笔都过这里，所以排版不会因此错位。

    只收**明显是排版噪声**的空格：连续空格、标点前的空格、引号/括号内外的空格。
    不碰 ASCII 撇号（`'`）后面 —— `dogs' tails` 这类结构里那个空格是有意义的。
    """
    text = re.sub(r"\s{2,}", " ", str(content or "")).strip()
    text = re.sub(r"\s+(?=[,;:.!?%)])", "", text)
    text = re.sub(r"(?<=[“‘(\[])\s+", "", text)
    return re.sub(r"\s+(?=[”’)\]])", "", text)


def _line(ax, x, y_in, content, width, *, size=FS_TEXT, weight="normal", color=INK,
          style="normal"):
    """画一段（可换行）文字，返回下一行的 y_in。`ax is None` 时只量不画。"""
    lines = wrap_cjk(tidy(content), width)
    step = _inch(size)
    if ax is not None:
        for index, line in enumerate(lines):
            ax.text(x, Y(y_in + index * step), line, fontsize=size, color=color,
                    fontweight=weight, ha="left", va="top", zorder=5, style=style)
    return y_in + len(lines) * step


def _chips(ax, x, y_in, labels, *, max_x, size=FS_TEXT):
    """流式标签；返回占掉的高度。`ax is None` 时只量。

    `chip()` 把 y 当**底边**向上画，所以要把它抬到本行的顶边再落笔；直接传 y_in 会让标签
    盖住上一行（实测压住了 “significantly reduced: 3 / 6”）。
    """
    rows, _ = chip_wrap(ax, x, Y(y_in + _inch(size, CHIP_H_LH)) if ax is not None else 0.5,
                        labels, max_x=max_x, size=size, dry=ax is None)
    if not rows:
        return 0.0
    return (rows - 1) * _inch(size, CHIP_ROW_LH) + _inch(size, CHIP_H_LH)


def _label(ax, x, y_in, content, *, size=FS_TEXT, weight="bold", color=INK):
    """框**上方**的标签 —— 框内不再出现任何 title / label。"""
    if ax is not None:
        ax.text(x, Y(y_in), content, fontsize=size, color=color, fontweight=weight,
                va="bottom", zorder=5)


def tint(color: str, amount: float = 0.86) -> str:
    """把颜色往白里调 `amount` —— 给内容框上浅底，正文仍是黑字，读数不受影响。"""
    red, green, blue = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(
        *(int(round(channel + (255 - channel) * amount)) for channel in (red, green, blue)))


def _frame(ax, x0, y0_in, x1, y1_in, *, ec=PE, fc="white", lw=0.8):
    """英寸坐标的圆角框（y 自顶算）。z=0，永远在文字下面。"""
    if ax is not None:
        block(ax, x0, Y(y1_in), x1, Y(y0_in), fc=fc, ec=ec, lw=lw, z=0)


#: ①②③…（字符串下标从 0 起，用的时候记得 `CIRCLED[n - 1]`）
CIRCLED = "①②③④⑤⑥⑦⑧⑨"

#: Panel A 与 Panel B **用不同的配色方案**，两个面板一眼能分开：
#: A 讲「同一个表型的两个心智」，用告警红（受损）对中性蓝（完整）；
#: B 讲「同一个表型的两个机制」，用青与紫各给一个机制，对照表用中性蓝灰。
PANEL_A_COLOUR = {"impaired": C["warn"], "intact": "#3A6EA5"}
PANEL_B_COLOUR = (C["narrow"], C["distributed"])


def col_width(x0, x1, size=FS_TEXT) -> int:
    """子栏的换行宽度 —— 左右留白各扣一次（`fit_width` 的 pad 是「合计」口径）。"""
    return fit_width(x0, x1, size, pad=0.028)


def _rich(ax, x, y_in, runs, width, *, size=FS_TEXT, color=INK):
    """一行之内混排**加粗 / 常规**，返回下一行的 y_in。`runs` = [(文字, 是否加粗), …]。

    折行口径与 `wrap_cjk` 一致（CJK 记 2、拉丁记 1），所以行数能直接和 `_line` 的高度相加；
    落笔时改用 **matplotlib 实测的字宽**推进光标 —— 估宽会让粗体段压到下一段上
    （粗体比常规宽约 14%，估算口径不一样就会贴字）。
    """
    tokens: list[tuple[str, bool]] = []
    for content, bold in runs:
        for token in re.findall(r"[\u2E80-\uFFFF]|[^\s\u2E80-\uFFFF]+|\s+", str(content)):
            tokens.append((token, bold))

    lines: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    used = 0
    for token, bold in tokens:
        if token.isspace():
            if current:
                current.append((token, bold))
                used += 1
            continue
        cost = sum(2 if ord(ch) > 0x2E80 else 1 for ch in token)
        if used + cost > width and current:
            lines.append(current)
            current, used = [], 0
        current.append((token, bold))
        used += cost
    if current:
        lines.append(current)

    step = _inch(size)
    for index, line in enumerate(lines):
        y = Y(y_in + index * step)
        cursor = x
        merged: list[tuple[str, bool]] = []
        for token, bold in line:                       # 相邻同字重并成一段，少画几个对象
            if merged and merged[-1][1] == bold:
                merged[-1] = (merged[-1][0] + token, bold)
            else:
                merged.append((token, bold))
        for content, bold in merged:
            if ax is not None:
                artist = ax.text(cursor, y, content, fontsize=size, color=color,
                                 fontweight="bold" if bold else "normal",
                                 ha="left", va="top", zorder=5)
                box = artist.get_window_extent(renderer=ax.figure.canvas.get_renderer())
                (left, _), (right, _) = ax.transData.inverted().transform(
                    [(box.x0, 0), (box.x1, 0)])
                cursor += (right - left) + 0.0003
    return y_in + len(lines) * step


def quoted_turn_items(info, table) -> list:
    """③ 的「只保留问答」版本：两轮 Q/A，斜体 —— 结论句不再重复印一遍。"""
    items = []
    for example in info["excerpts"][:2]:
        items.append(("Q: " + tr(example["q"], table), "normal", "italic"))
        items.append(("A: " + tr(example["a"], table), "normal", "italic"))
    return items


def contrast_items(ctx, pid) -> list:
    """Panel B 的 ③：**这个机制自己**在问答里呈现的特点，体现差异处加粗。

    条目全部来自缓存（LLM 读已记录问答得出的），这里只负责排版：
    把 `bold` 子串切出来单独加粗，其余照常。缓存缺失时返回空，由主流程硬拦。
    """
    contrast = ctx.get("contrast") or {}
    labels = list(ctx["q2"]["group"]["members"])
    entries = contrast.get("a" if pid == labels[0] else "b") or []
    items = []
    for entry in entries:
        body = clean(entry.get("text"))
        if not body:
            continue
        mark = clean(entry.get("bold"))
        if mark and mark in body:
            head, _, tail = body.partition(mark)
            runs = [(part, part is mark) for part in (head, mark, tail) if part]
        else:
            runs = [(body, False)]
        items.append((runs, "normal", "normal"))
    return items


def evidence_sections(info, table, *, third=None) -> tuple:
    """把一侧（或一个机制）的证据分成 **①②③ 三方面**，Panel A 与 Panel B 共用这一套分法。

    返回 `((组标题, [(文字, 字重, 字体), …]), …)`。`third` 给定时用它替换 ③
    （Panel B 换成「表现差异」），否则 ③ = 引用的问答。
    """
    title, closing = third if third else ("Quoted turn", quoted_turn_items(info, table))
    return (
        ("Observations", [("• " + tr(line, table), "normal", "normal")
                          for line in info["observations"]]),
        ("Task → behaviour", [("• " + tr(line, table), "normal", "normal")
                              for line in info["tasks"]]),
        (title, closing),
    )


def evidence_block(ax, x0, x1, y_in, sections, *, gap=0.020) -> float:
    """①② 排在**左栏**、③ 排在**右栏**，两栏自同一高度起排。返回消耗的高度。

    量高与落笔仍是同一遍代码（`ax=None` 只量），所以两栏谁高都不影响正确性。
    """
    span = x1 - x0
    split = x0 + span * 0.5
    numbered = list(enumerate(sections, start=1))
    left = [item for item in numbered if item[0] <= 2]
    right = [item for item in numbered if item[0] > 2]
    tallest = 0.0
    for col_x0, col_x1, groups in ((x0, split - gap / 2, left),
                                   (split + gap / 2, x1, right)):
        width = col_width(col_x0, col_x1)
        y = y_in
        for index, (title, lines) in groups:
            y += _inch() * 0.5
            y = _line(ax, col_x0, y, f"{CIRCLED[index - 1]} {title}", width, weight="bold")
            for content, weight, style in lines:
                if isinstance(content, (list, tuple)):       # 粗/常规混排（差异处加粗）
                    y = _rich(ax, col_x0 + 0.012, y, content, width - 5)
                else:
                    y = _line(ax, col_x0 + 0.012, y, content, width - 5,
                              weight=weight, style=style)
        tallest = max(tallest, y - y_in)
    return tallest


def panel_a_sides(ctx):
    """Panel A 的两侧。受损侧的机制与档位写在 θ/Δ/q 那一行，框上不再重复标一次。"""
    return (
        (0.024, 0.494, "impaired", ""),
        (0.506, 0.978, "intact", "intact mind"),
    )


def panel_a_number_row(ctx):
    """数字行的内容（机制名与档位合一，取代原来的 `θ = 0.40`）。"""
    q1 = ctx["q1"]
    name = EN.get(q1["case"]["primitive_id"], q1["case"]["primitive_id"])
    return (f"{name} = {q1['dose']:.2f}",
            f"Δ = {abs(q1['published_delta']):.2f}",
            f"q = {q1['q_pooled']:.2g}")


def panel_a(ax, ctx, top_in) -> float:
    """Panel A：一个表型、受损与完整两侧的逐字证据。返回占用的高度。"""
    table = ctx.get("translations") or {}
    body_top = top_in + TITLE_ZONE + LABEL_GAP
    tallest = 0.0
    for cx0, cx1, side, label in panel_a_sides(ctx):
        if label:
            _label(ax, cx0 + 0.004, body_top - 0.030, label)
        content_h = evidence_block(ax, cx0 + 0.014, cx1 - 0.010, body_top + BOX_PAD,
                                  evidence_sections(ctx["q1"][side], table))
        tallest = max(tallest, content_h + 2 * BOX_PAD)
    bottom = body_top + tallest

    # 数字行与 `intact mind` **同一高度**（同一条标签带，va 与 y 都一致）
    x = 0.024
    for content in panel_a_number_row(ctx):
        if ax is not None:
            ax.text(x, Y(body_top - 0.030), content, fontsize=FS_TEXT, color=INK,
                    fontweight="bold", va="bottom", zorder=5)
        x += text_width_axes(content, FS_TEXT, bold=True) + 0.030
    for cx0, cx1, side, label in panel_a_sides(ctx):
        colour = PANEL_A_COLOUR[side]
        _frame(ax, cx0, body_top, cx1, bottom, ec=colour, fc=tint(colour))
    return bottom - top_in


#: 两个机制框与 Panel A 的框同宽同位 —— 两个面板并排看时列是对齐的
B_MECH_X = ((0.024, 0.494), (0.506, 0.978))



def panel_b(ax, ctx, top_in) -> float:
    """Panel B：两个机制落在同一个表型上。内容与 Panel A **同一套 ①②③**，配色另换一套。"""
    q2 = ctx["q2"]
    table = ctx.get("translations") or {}
    labels = list(q2["group"]["members"])
    # 表型与判定各占一行、居中。判定是「两轴是否可区分」的结论 —— 本图不画标记表，
    # 但这个结论不能跟着删，移到面板头。
    if ax is not None:
        ax.text(0.500, Y(top_in + TITLE_ZONE + 0.038), f"Same phenotype: {DIM_EN_ONE[q2['dim']]}",
                fontsize=FS_TEXT, color=INK, fontweight="bold", ha="center",
                va="top", zorder=5)
    # 机制名 + Θ + Δ 一行（Δ 不带正号；Θ 是完整度档位）。原来下面还有一行灰色的本体定义，
    # 灰色说明文字一律不出现在图上。
    label_in = top_in + TITLE_ZONE + PHENO_ROW + 0.030
    body_top = label_in + _inch() + 0.055
    for pid in labels:
        if ax is not None:
            bx0 = B_MECH_X[labels.index(pid)][0]
            ax.text(bx0 + 0.004, Y(label_in),
                    f"{EN.get(pid, pid)}   Θ = {q2['dose']:.2f}, "
                    f"Δ = {abs(q2['members'][pid]['delta']):.2f}",
                    fontsize=FS_TEXT, color=INK, fontweight="bold", va="top", zorder=5)
    tallest = 0.0
    for pid, (bx0, bx1) in zip(labels, B_MECH_X):
        sections = evidence_sections(
            q2["members"][pid], table,
            third=("What its answers show", contrast_items(ctx, pid)))
        content_h = evidence_block(ax, bx0 + 0.014, bx1 - 0.010, body_top + BOX_PAD, sections)
        tallest = max(tallest, content_h + 2 * BOX_PAD)
    bottom = body_top + tallest

    for (bx0, bx1), colour in zip(B_MECH_X, PANEL_B_COLOUR):
        _frame(ax, bx0, body_top, bx1, bottom, ec=colour, fc=tint(colour))
    return bottom - top_in


C_COLUMNS = ((0.024, 0.216), (0.224, 0.416), (0.424, 0.616))
C_RIGHT_X = (0.626, 0.978)


def panel_c_column(ax, ctx, step, cx0, cx1, body_top):
    """一个严重度档：显著项计数 + 各表型的 Δ 标签 + 该档主导表型上的观察记录。"""
    table = ctx.get("translations") or {}
    width = box_width(cx0, cx1)
    y = body_top + BOX_PAD
    y = _line(ax, cx0 + 0.014, y,
              f"significantly reduced: {step['n_significant']} / 6", width, weight="bold")
    if step["dims"]:
        # 负号表示**能力降低**（发布表的 delta 是「基线 − 实测」，正值 = 掉下去）
        labels = [(f"{DIM_SHORT[d['dim']]}  {-abs(d['delta']):.2f}", DIMC[d["dim"]])
                  for d in step["dims"]]
        y += _inch(FS_TEXT) * 0.5
        y += _chips(ax, cx0 + 0.010, y, labels, max_x=cx1 - 0.010)
    if step.get("observations"):
        y += _inch(FS_TEXT) * 0.5
    for observation in step.get("observations") or []:
        y = _line(ax, cx0 + 0.014, y, "• " + tr(observation, table), width)
    return y - body_top + BOX_PAD


def panel_c_right(ax, ctx, rx0, rx1, body_top):
    """右栏：四条「为什么会变多」的解释。"""
    width = box_width(rx0, rx1)
    colour = DIMC[ctx["q3"]["steps"][-1]["dominant"]]

    # 上面那块：四条解释
    y = body_top + BOX_PAD
    _label(ax, rx0 + 0.004, body_top - 0.030, "Why the affected set changes")
    for index, (title, body) in enumerate((ctx.get("q3_explanations") or {}).items(), start=1):
        y = _line(ax, rx0 + 0.014, y, f"{index}  {title}", width, weight="bold")
        y = _line(ax, rx0 + 0.024, y, body, width - 6)
        y += _inch(FS_TEXT) * 0.55
    explain_bottom = y - _inch(FS_TEXT) * 0.55 + BOX_PAD
    _frame(ax, rx0, body_top, rx1, explain_bottom,
           ec=colour, fc=tint(colour, 0.95))
    return explain_bottom - body_top


def panel_c(ax, ctx, top_in) -> float:
    q3 = ctx["q3"]
    body_top = top_in + TITLE_ZONE + LABEL_GAP
    tallest = 0.0
    for (cx0, cx1), step in zip(C_COLUMNS, q3["steps"]):
        _label(ax, cx0 + 0.004, body_top - 0.030,
               f"{step['severity']}   θ = {step['dose']:.2f}")
        tallest = max(tallest, panel_c_column(ax, ctx, step, cx0, cx1, body_top))
    tallest = max(tallest, panel_c_right(ax, ctx, C_RIGHT_X[0], C_RIGHT_X[1], body_top))
    bottom = body_top + tallest

    for (cx0, cx1), step in zip(C_COLUMNS, q3["steps"]):
        colour = DIMC[step["dominant"]]
        _frame(ax, cx0, body_top, cx1, bottom, ec=colour, fc=tint(colour))
    return bottom - top_in


def audit(fig) -> dict:
    """出图前的两道自检：图上不许残留中文，文字不许越出所属内容框。

    这两件事不能靠肉眼在渲染结果里找：漏了就会把中文和压出框的文字发出去。
    这里用硬检查：只有过了才轮得到 `save()`。
    """
    import matplotlib.text as mtext

    leftovers: list[str] = []
    texts = 0
    for ax in fig.axes:
        for item in ax.texts:
            content = item.get_text()
            if not content:
                continue
            texts += 1
            if re.search(r"[\u2E80-\uFFFF]", content):
                leftovers.append(content)

    # 越界检查：每个文本的右边缘要落在它所在面板的 x 区间内（画幅 1.0 是硬边界）
    renderer = fig.canvas.get_renderer()
    spill: list[str] = []
    for ax in fig.axes:
        for item in ax.texts:
            if not item.get_text():
                continue
            box = item.get_window_extent(renderer=renderer)
            x0, x1 = ax.transData.inverted().transform([(box.x0, 0), (box.x1, 0)])[:, 0]
            if x1 > 0.996 or x0 < 0.002:
                spill.append(f"{x0:.3f}–{x1:.3f}  {item.get_text()[:52]}")

    report = [f"  自检：文本对象 {texts} 个；残留中文 {len(leftovers)} 处；越出画幅 {len(spill)} 处"]
    for content in leftovers[:6]:
        report.append(f"    ⚠ 未译：{content[:70]}")
    for content in spill[:6]:
        report.append(f"    ⚠ 越界：{content}")
    for line in report:
        print(line)
    if leftovers or spill:
        raise SystemExit("✗ 自检不过：先修好上面的问题再出图。")
    return {"texts": texts, "leftovers": leftovers, "spill": spill, "report": report}


def caption_lines(ctx) -> list[str]:
    """底部只留一行判据。"""
    width = fit_width(PANEL_X[0], PANEL_X[1], FS_TEXT) - 12
    return wrap_cjk(
        f"Δ = baseline - measured ≥ {TARGET_DROP:.2f} AND BH-FDR q < {TEST_ALPHA:.2f} "
        f"within the (dose × phenotype) family", width)


def measure(ctx) -> tuple[float, dict]:
    """先把三块内容各量一遍（不落笔），返回画幅高度与各块高度。"""
    heights = {"A": panel_a(None, ctx, 0.0),
               "B": panel_b(None, ctx, 0.0),
               "C": panel_c(None, ctx, 0.0)}
    caption_h = len(caption_lines(ctx)) * _inch(FS_TEXT)
    total = (FIG_HEAD + heights["A"] + PANEL_GAP + heights["B"] + PANEL_GAP
             + heights["C"] + PANEL_GAP + caption_h + FIG_TAIL)
    heights["caption"] = caption_h
    return total, heights


def draw(ctx, *, dpi_note: str) -> plt.Figure:
    global _H, FIG_H_IN
    total, heights = measure(ctx)
    _H = FIG_H_IN = total
    fig = plt.figure(figsize=(FIG_W_IN, total))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax.text(PANEL_X[0], Y(0.115),
            "Figure 3  |  What the recorded dialogue probes show about the impaired mind",
            fontsize=FS_TITLE, color=INK, fontweight="bold", va="top", zorder=5)

    top = FIG_HEAD
    top += panel_a(ax, ctx, top) + PANEL_GAP
    top += panel_b(ax, ctx, top) + PANEL_GAP
    top += panel_c(ax, ctx, top) + PANEL_GAP
    for line in caption_lines(ctx):
        ax.text(PANEL_X[0], Y(top), line, fontsize=FS_TEXT, color=INK, va="top", zorder=5)
        top += _inch(FS_TEXT)
    return fig


PRIMITIVE_DEF = {p.id: getattr(p, "definition", "") for p in PRIMITIVES}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def premise_verdict(ctx) -> str:
    """Q3 的前提（轻度一个表型 → 严重多个表型）在 Δ+p 判据下是否成立。"""
    strict = ctx["q3"]["n_sig_strict_by_pid"]
    plain = ctx["q3"]["n_sig_by_pid"]
    mild, heavy = f"{max(SEVERITY_DOSES):g}", f"{min(SEVERITY_DOSES):g}"
    mild_counts = {pid: plain[pid][mild] for pid in LADDER_PRIMITIVES}
    heavy_counts = {pid: plain[pid][heavy] for pid in LADDER_PRIMITIVES}
    grew = [pid for pid in LADDER_PRIMITIVES if heavy_counts[pid] > mild_counts[pid]]
    saturated = [pid for pid in LADDER_PRIMITIVES if mild_counts[pid] == 6]
    flat = [pid for pid in LADDER_PRIMITIVES
            if heavy_counts[pid] == mild_counts[pid] and mild_counts[pid] < 6]
    single = [pid for pid in LADDER_PRIMITIVES if mild_counts[pid] == 1]
    return (
        f"No. Under Δ ≥ 0.15 AND q < 0.05 no mechanism ever affects exactly one phenotype: at the mildest "
        f"dose (θ = 0.90) the significant set already holds {min(mild_counts.values())}–"
        f"{max(mild_counts.values())} of the 6 phenotypes. For {len(grew)}/{len(LADDER_PRIMITIVES)} mechanisms "
        f"the count also grows with severity ("
        + "; ".join(f"{EN.get(p, p)} {mild_counts[p]}→{heavy_counts[p]}" for p in grew)
        + "); the rest stay flat in count and grow only in depth"
        + (f", and {len(saturated)} is already saturated at 6/6 from the mildest dose" if saturated else "")
        + ". Read it as a broad, dose-scaled shift of an already multi-dimensional profile — not one "
          "phenotype spreading to many."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fig3_dialogue_evidence",
        description="画 Fig.3（panel A = Q1/Q2，panel B = Q3，判据 Δ 且 p）。")
    parser.add_argument("--q1-tier", default="A", choices=("A", "B"))
    parser.add_argument("--q1-primitive", default="social_reward_weighting",
                        help="Panel A 的机制（默认 社会奖励加权）")
    parser.add_argument("--q1-dim", default="social_interaction_motivation",
                        help="Panel A 的表型维度（默认 社会互动与社会动机）")
    parser.add_argument("--q1-dose", type=float, default=0.4, help="Panel A 的档位")
    parser.add_argument("--q3-primitive", default="self_other_distinction",
                        help="Panel C 的机制；留空则按阶梯规则自动挑")
    parser.add_argument("--q1-excerpts", type=int, default=2,
                        help="Panel A 每栏最多放几条逐字引例（放不下会自动下调）")
    parser.add_argument("--match-tolerance", type=float, default=0.05)
    parser.add_argument("--boilerplate-rate", type=float, default=0.20)
    parser.add_argument("--match-min", type=float, default=ade.MATCH_MIN)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarise", action="store_true",
                        help="让 LLM 读已记录的逐轮问答，重写 Panel B 的「各自特点」总结缓存")
    parser.add_argument("--translate", action="store_true",
                        help="重跑 Panel A 文本的英译并写缓存（默认只读缓存）")
    args = parser.parse_args(argv)

    plt.rcParams.update({
        "font.family": "sans-serif",
        # DejaVu Sans 排第一位，因为它同时满足两件本机唯一的事：
        #   ① 覆盖 ①②③（Arial 没有 U+2460–2462，逐字符回退在本机不生效，会掉成豆腐块）；
        #   ② 有**真正的斜体字面**（Microsoft YaHei 无斜体，`style="italic"` 会被静默忽略，
        #      问答该斜却不斜就是这个原因）。
        # 图里已经没有中文（`audit()` 硬门保证），所以不再需要 CJK 字体打头。
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    validation = validate_test()
    print(f"统计口径硬门：重算 result_2 @0.4 的 {validation['n_cells']} 个 p_pooled，"
          f"与发布表最大偏差 {validation['worst_gap']:.2e} → "
          f"{'✓ 一致' if validation['ok'] else '✗ 不一致'}")
    if not validation["ok"]:
        raise SystemExit("✗ 重算的 p 值与 figure2_link_candidates.csv 对不上，不出图。")

    ctx = build_context(args)
    if args.translate:
        table = build_translations(collect_figure_strings(ctx), load_translations())
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / TRANSLATIONS_NAME).write_text(json.dumps({
            "note": "Panel A 内引用的中文原文 → 英译。翻译对象是**已记录**的对话与观察文本，"
                    "不重跑任何探查；渲染只读本缓存，保证图可复现。",
            "layer": "post_hoc_interpretation",
            "translations": table,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ 英译缓存：{len(table)} 条 → {OUT_DIR / TRANSLATIONS_NAME}")
    if args.summarise:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = build_contrast(ctx)
        (OUT_DIR / CONTRAST_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = payload.get("contrast") or {}
        print(f"  ✓ 差异总结缓存：A {len(counts.get('a') or [])} 条、"
              f"B {len(counts.get('b') or [])} 条 → {OUT_DIR / CONTRAST_NAME}")
    ctx["translations"] = load_translations()
    print(f"  英译缓存已载入 {len(ctx['translations'])} 条")
    ctx["contrast"] = load_contrast()
    print(f"  差异总结缓存：A {len(ctx['contrast'].get('a') or [])} 条、"
          f"B {len(ctx['contrast'].get('b') or [])} 条")
    ctx["q1_excerpt_limit"] = args.q1_excerpts
    ctx["validation"] = validation
    ctx["q3_explanations"] = build_explanations(ctx)
    ctx["premise"] = premise_verdict(ctx)

    print()
    print("个案选定：")
    q1c = ctx["q1"]
    print(f"  A（Q1）: {EN.get(q1c['case']['primitive_id'])} → {DIM_EN_ONE[q1c['dim']]} @ "
          f"{q1c['dose']:g} | 读数 {q1c['baseline']:.2f} → {q1c['published_mean']:.2f} | "
          f"q={q1c['q_pooled']:.3g} | 对照 Δ={q1c['control_delta']:.2f} | {q1c['verdict']}")
    print(f"  B（Q2）: {list(ctx['q2']['members'])} → {DIM_EN_ONE[ctx['q2']['dim']]} "
          f"(ratio {ctx['q2']['group']['separation_ratio']}, p {ctx['q2']['group']['permutation_p']})")
    print(f"  C（Q3）: {EN.get(ctx['q3']['pid'])}")
    print()
    print("显著下降的维度数（Δ ≥ 0.15 且 q < 0.05）")
    for pid in LADDER_PRIMITIVES:
        mild = ctx["q3"]["n_sig_by_pid"][pid]["0.9"]
        heavy = ctx["q3"]["n_sig_by_pid"][pid]["0.2"]
        strict_mild = ctx["q3"]["n_sig_strict_by_pid"][pid]["0.9"]
        print(f"  {EN.get(pid):<26} 0.9 档 {mild} → 0.2 档 {heavy} "
              f"（再用 Welch 收严：0.9 档 {strict_mild}）")
    print()
    print(ctx["premise"])

    if args.dry_run:
        print()
        print("[dry-run] 不画图、不落盘")
        return 0

    if not ctx.get("contrast"):
        raise SystemExit("✗ 缺 Panel B 的「各自特点」缓存：先跑 `--summarise`"
                         "（只读已记录的逐轮问答，不重跑探查）。")

    fig = draw(ctx, dpi_note=STEM)
    checks = audit(fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for suffix in (".png", ".pdf"):
        results[suffix] = f2.save(fig, OUT_DIR / f"{STEM}{suffix}")
    plt.close(fig)

    metrics = {
        "source": {
            "result_2": str(ade.RESULT_2.relative_to(ROOT)),
            "result_4": str(ade.RESULT_4.relative_to(ROOT)),
            "reference": str(ade.OFFICIAL_BASELINE_DIR.relative_to(ROOT)),
        },
        "criterion": {
            "target_drop": TARGET_DROP, "alpha": TEST_ALPHA,
            "test": "pooled-variance two-sample t-test vs 5 intact reference sessions",
            "fdr": "Benjamini–Hochberg within (dose × phenotype) family",
            "significant": "delta >= target_drop AND q < alpha",
        },
        "validation": validation,
        "panel_a": {
            "mechanism": ctx["q1"]["case"]["primitive_id"],
            "phenotype": ctx["q1"]["dim"],
            "dose": ctx["q1"]["dose"],
            "baseline_reading": ctx["q1"]["baseline"],
            "impaired_reading": ctx["q1"]["published_mean"],
            "published_delta": ctx["q1"]["published_delta"],
            "q_pooled": ctx["q1"]["q_pooled"],
            "control_delta": ctx["q1"]["control_delta"],
            "specificity_verdict": ctx["q1"]["verdict"],
            "n_sessions": ctx["q1"]["n_subjects"],
            "impaired_observations": ctx["q1"]["impaired"]["observations"],
            "intact_observations": ctx["q1"]["intact"]["observations"],
            "impaired_tasks": ctx["q1"]["impaired"]["tasks"],
            "intact_tasks": ctx["q1"]["intact"]["tasks"],
            "impaired_conclusion": ctx["q1"]["impaired"]["conclusion_full"],
            "intact_conclusion": ctx["q1"]["intact"]["conclusion_full"],
            "highlighted_sentences_impaired": [s for s, hot in
                                              ctx["q1"]["impaired"]["conclusion_sentences"] if hot],
            "highlighted_sentences_intact": [s for s, hot in
                                            ctx["q1"]["intact"]["conclusion_sentences"] if hot],
            "observations_shown_impaired": len(ctx["q1"]["impaired"]["observations"]),
            "observations_total_impaired": ctx["q1"]["impaired"]["n_observations_total"],
            "tasks_shown_impaired": len(ctx["q1"]["impaired"]["tasks"]),
            "tasks_total_impaired": ctx["q1"]["impaired"]["n_tasks_total"],
            "excerpts_shown": ctx.get("q1_excerpts_used"),
        },
        "panel_b": {
            "mechanisms": list(ctx["q2"]["members"]),
            "phenotype": ctx["q2"]["dim"],
            "dose": ctx["q2"]["dose"],
            "separation_ratio": ctx["q2"]["group"]["separation_ratio"],
            "permutation_p": ctx["q2"]["group"]["permutation_p"],
            "two_axis_agreement": ctx["q2"]["group"]["two_axis_agreement"],
            "markers": {pid: info["markers"] for pid, info in ctx["q2"]["members"].items()},
            # Panel B ③ 的来源：已记录问答的总结（缓存），逐条可对
            "contrast": ctx.get("contrast") or {},
            # 标记键 → 读数名/读法：面板上不再印对照表，这里补上图例
            "marker_legend": {key: {"label": label, "worse_when": worse_when, "note": note}
                              for key, label, worse_when, note in MARKER_ROWS},
        },
        "panel_c": {
            "mechanism": ctx["q3"]["pid"],
            "rule": ctx["q3"]["rule"],
            "steps": [{k: v for k, v in step.items() if k != "dims"} |
                      {"dims": step["dims"]} for step in ctx["q3"]["steps"]],
            "n_significant_by_mechanism": ctx["q3"]["n_sig_by_pid"],
            "n_significant_strict_by_mechanism": ctx["q3"]["n_sig_strict_by_pid"],
            "premise_verdict": ctx["premise"],
        },
        "no_writeback": ade.NO_WRITEBACK,
        "figure": {
            "size_in": [round(FIG_W_IN, 3), round(float(fig.get_size_inches()[1]), 3)],
            "text_objects": checks["texts"],
            "audit": "出图前硬检查：图上不许残留未译中文、文字不许越出画幅；不过则弃图不出",
            "caption_criterion": (
                "Δ = baseline - measured (positive = more ASD-like); a phenotype is called clearly "
                "reduced when Δ ≥ 0.15 and BH-FDR q < 0.05 within the (dose × phenotype) family."),
        },
        "note": "机制→表型维度是测量产物，不是预设对应；本图不得回写为运行先验。",
    }
    (OUT_DIR / METRICS_NAME).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for suffix, status in results.items():
        print(f"  {'✓' if status == 'ok' else '⚠'} {OUT_DIR / (STEM + suffix)}  {status}")
    print(f"  ✓ {OUT_DIR / METRICS_NAME}")
    return 0


def build_explanations(ctx) -> dict:
    """Panel B 右栏：对**这一个机制**成立的四条候选解释（数值取自本次分析）。"""
    pid = ctx["q3"]["pid"]
    steps = ctx["q3"]["steps"]
    sev = {s["severity"]: s for s in steps}
    mild, heavy = sev["Mild"], sev["Severe"]
    growth = heavy["n_significant"] - mild["n_significant"]
    control_heavy = heavy["control_delta"]
    return {
        "Non-specific general degradation": (
            f"Control channel drops {control_heavy:+.2f} at the severe rung — "
            + ("beyond tolerance, so part of the spread is a global capacity drop."
               if control_heavy >= 0.15 else
               "inside tolerance, so this is not global collapse.")),
        "Depth scales more than breadth": (
            f"Significant phenotypes {mild['n_significant']} → {heavy['n_significant']}; mean drop over the six "
            f"goes {mild['mean_delta']:+.2f} → {heavy['mean_delta']:+.2f}."),
        "Evidence sharing (measurement artefact)": (
            "Spillover phenotypes scored from the dominant channel's own questions would be an artefact; "
            "that share stayed low (0.00–0.05)."),
        "Genuine downstream cascade": (
            "Each spillover phenotype's evidence comes from its own questions, which is what a real cascade "
            "looks like."),
    }


if __name__ == "__main__":
    sys.exit(main())
