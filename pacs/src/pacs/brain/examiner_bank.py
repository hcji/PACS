"""观察者内置条目库的运行时访问层（读取随包分发的 `data/examiner_bank.json`）。

条目库由 `tools/build_examiner_bank.py` 从 `resources/asdkb/`（AsdKB 孤独症知识库精选）
生成，**运行时不再依赖 resources 目录**。库内含四个层次：

- `items`（441 条）：20 套筛查量表的逐题条目（14 套有逐题，6 套仅元信息），
  每条带施测方式、维度归属与判定依据、可扰动基础过程映射、派生症状与标准关联；
  其中 50 条附**儿童面向探查语句**（人工重写），其余 391 条只作观察要点。
- `symptoms`（65 条）、`criteria`（43 条）、`treatments`（28 条）：术语、标准与干预参考层。

选条目时只取有探查语句的条目；其余条目进观察提纲（给分析者当评分锚点）。
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from .. import DEFAULT_AGE_MONTHS

BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "examiner_bank.json"

#: 施测方式 → 中文（界面与提示词共用）
ADMIN_ZH = {
    "caregiver_report": "知情者报告",
    "clinician_rating": "临床评定",
    "self_report": "自评（成人版）",
    "observation": "主试者现场任务",
    "split": "分段（知情者报告 + 现场任务）",
}


@lru_cache(maxsize=1)
def load_bank() -> dict:
    """载入随包条目库（进程内缓存一次）。文件缺失时给出明确指引，而不是静默返回空。"""
    if not BANK_PATH.exists():
        raise FileNotFoundError(
            f"内置条目库不存在：{BANK_PATH}\n"
            "请用 `python tools/build_examiner_bank.py` 从 resources/asdkb 重新生成。"
        )
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))


def _all_items() -> list[dict]:
    return load_bank()["items"]


def item_by_key(key: str) -> dict | None:
    return next((i for i in _all_items() if i["key"] == key), None)


def items_by_keys(keys: list[str]) -> list[dict]:
    """按给定顺序取回条目（用于用户在界面上勾选指定条目）。"""
    index = {i["key"]: i for i in _all_items()}
    return [index[k] for k in keys if k in index]


#: 通道 signature 取该通道亲和度最高的前 N 个 primitive（N = SIGNATURE_TOP）。
SIGNATURE_TOP = 10


def _heaviest_item_profile(channel: str) -> dict[str, float]:
    """该通道里**敏感度总权重最高**的那条条目的 primitive 权重（按键名稳定排序取最大）。"""
    best: tuple[float, str, dict] | None = None
    for item in _all_items():
        if channel not in (item.get("dims") or {}):
            continue
        prims = {pid: float(w) for pid, w in (item.get("primitives") or {}).items()}
        if not prims:
            continue
        candidate = (round(sum(prims.values()), 4), str(item.get("key") or ""), prims)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else {}


def channel_signature(channel: str, *, top: int = SIGNATURE_TOP) -> tuple[str, ...]:
    """某通道的「基础过程画像」：问这个通道时，离线 mock 假定会**被考察到**的 primitive。

    用途：提问者**自己出题**时没有题库条目可查，但这一轮问的是哪个通道是知道的
    （提问者自报 `FOCUS`）。离线 mock 需要一份「这个通道大体在考察什么」的画像，
    才能让选择性受损照旧显现在对的通道上。

    构成 = 两部分之并集：
    1. 该通道**条目数亲和度**最高的前 `top` 个 primitive（该通道有 82–285 条条目，
       逐个统计「这条条目是否声明对该 primitive 敏感」，得到「这个通道的题普遍会问到什么」）；
    2. **载荷最重的那条条目**的 primitive（个别 primitive 在条目里很少见、却是某条高载荷
       条目的核心，例如 `attention_shifting` —— 只按亲和度排序会把它漏掉）。

    **已知局限（不静默）**：条目库里少数 primitive（如 `selective_attention`、
    `familiarity_valuation`、`confidence_calibration`）在任何通道的 signature 里都不出现，
    因此「全部轮次都由提问者自造」的离线跑法下，单独扰动这些 primitive 不会改变读数。
    这是 mock 的近似（真实模型自行出题时考察什么，离线无从得知），不是项目的机制主张；
    `tests/test_question_generation.py` 把这个局限钉成了断言，条目库变化时会立刻暴露。
    """
    counts: dict[str, int] = {}
    for item in _all_items():
        if channel not in (item.get("dims") or {}):
            continue
        for pid in (item.get("primitives") or {}):
            counts[pid] = counts.get(pid, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ordered = [pid for pid, _ in ranked[:top]]
    for pid in _heaviest_item_profile(channel):
        if pid not in ordered:
            ordered.append(pid)
    return tuple(ordered)


def dims_map() -> dict:
    return load_bank()["dims"]


def mapping_rules() -> dict:
    return load_bank()["mapping_rules"]


def symptoms() -> list[dict]:
    """65 条症状（含分类与维度归属）。"""
    return load_bank()["symptoms"]


def criteria() -> list[dict]:
    """43 条 DSM/ICD 行为标准条目。"""
    return load_bank()["criteria"]


def treatments() -> list[dict]:
    """28 条循证干预（EBP）—— 参考层，不是题目。"""
    return load_bank()["treatments"]


def source_meta() -> dict:
    return load_bank()["meta"]


def bank_stats() -> dict:
    """给界面的汇总与筛选项（facets）。直接取自生成时写入的统计，保证与文件自洽。"""
    bank = load_bank()
    stats = dict(bank["stats"])
    stats["dims"] = bank["dims"]
    stats["sources"] = [bank["meta"]["source"]["name"]]
    return stats


def age_band(item: dict) -> tuple[int | None, int | None]:
    """条目适用月龄区间；`age` 字段被污染的条目返回 `(None, None)`。"""
    if item.get("age_field_broken"):
        return None, None
    return item.get("age_months_min"), item.get("age_months_max")


def age_fit(item: dict, age_months: int = DEFAULT_AGE_MONTHS, slack: int = 6) -> str:
    """该条目对给定月龄是否适配。

    - `in_range`：量表适用区间覆盖该月龄；
    - `below`：量表针对更小年龄（幼儿筛查工具对 3 岁仍具观察价值，但排在年龄匹配条目之后）；
    - `above`：量表针对更大年龄；`unknown`：源数据年龄字段缺失或已被污染。
    """
    low, high = age_band(item)
    if low is None and high is None:
        return "unknown"
    if low is not None and age_months < low - slack:
        return "above"
    if high is not None and age_months > high + 6:
        return "below"
    return "in_range"


def list_items(
    *,
    dim: str = "",
    primitive: str = "",
    admin: str = "",
    probe_mode: str = "",
    source_id: str = "",
    probe_ready: bool = False,
    age: int | None = None,
    query: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """按条件浏览条目库（界面用）。返回 `{total, items}`。"""
    items = _all_items()
    if dim:
        items = [i for i in items if dim in (i.get("dims") or {})]
    if primitive:
        items = [i for i in items if primitive in (i.get("primitives") or {})]
    if admin:
        items = [i for i in items if i["admin"] == admin]
    if probe_mode:
        items = [i for i in items if i.get("probe_mode") == probe_mode]
    if source_id:
        items = [i for i in items if i["source_id"] == source_id]
    if probe_ready:
        items = [i for i in items if i.get("probe_zh")]
    if age is not None:
        items = [i for i in items if age_fit(i, age) in ("in_range", "below")]
    if query:
        needle = query.strip()
        items = [
            i for i in items
            if needle in i.get("text_zh", "")
            or needle in (i.get("probe_zh") or "")
            or needle in i.get("source_name", "")
        ]
    total = len(items)
    return {"total": total, "items": items[offset: offset + max(0, limit)]}


#: 探查方式：auto 用条目的首选形态；scenario 强制情境式；question 强制直接提问。
PROBE_MODES = ("auto", "scenario", "question")


def probe_text(item: dict, mode: str = "auto") -> str:
    """取该条目的探查呈现文本。

    - `auto`：条目首选形态（`probe_zh`）；
    - `scenario`：优先情境式（`scenario_zh`），无情境的条目回退到首选形态；
    - `question`：优先人工问句（`question_zh`），无问句的条目回退到首选形态。
    """
    if mode == "scenario":
        return item.get("scenario_zh") or item.get("probe_zh") or ""
    if mode == "question":
        return item.get("question_zh") or item.get("probe_zh") or ""
    return item.get("probe_zh") or ""


def dialogue_probes(
    *,
    age_months: int = DEFAULT_AGE_MONTHS,
    goals: tuple[str, ...] = (),
    n: int = 6,
    seed: int = 42,
    keys: list[str] | None = None,
    mode: str = "auto",
) -> list[dict]:
    """挑出用于对话探查的条目。

    `keys` 给定时按用户指定顺序取（界面勾选）；否则按年龄适配 + 目标覆盖自动挑：
    先让每个目标维度至少有一条（同一次挑选中**优先换量表**，避免 6 轮全落在同一套量表上），
    再按确定性随机补齐。目标数已经用满 `n` 时没有补齐空间，此时选题完全由目标与排序决定。
    """
    if keys:
        return items_by_keys(keys)

    if mode not in PROBE_MODES:
        raise ValueError(f"未知探查方式：{mode}（可选：{'/'.join(PROBE_MODES)}）")
    fits = {"in_range": 0, "below": 1, "unknown": 2, "above": 3}

    def form_rank(item: dict) -> int:
        """指定探查方式时，优先挑真正具备该形态的条目（缺形态的只作兜底）。"""
        if mode == "scenario":
            return 0 if item.get("scenario") else 1
        if mode == "question":
            return 0 if item.get("question_zh") else 1
        return 0

    items = [i for i in _all_items()
             if probe_text(i, mode) and age_fit(i, age_months) != "above"]
    items.sort(key=lambda i: (fits[age_fit(i, age_months)], form_rank(i), i["source_id"], i["key"]))

    chosen: list[dict] = []
    chosen_keys: set[str] = set()
    used_scales: set[str] = set()
    for goal in goals:
        pool = [i for i in items
                if i["key"] not in chosen_keys and goal in (i.get("dims") or {})]
        # 换量表优先：同一次选题里，先用还没出现过的量表覆盖该目标
        hit = next((i for i in pool if i["source_id"] not in used_scales), None)
        if hit is None and pool:
            hit = pool[0]
        if hit is not None:
            chosen.append(hit)
            chosen_keys.add(hit["key"])
            used_scales.add(hit["source_id"])

    rest = [i for i in items if i["key"] not in chosen_keys]
    rng = random.Random(seed)
    rng.shuffle(rest)
    for item in rest:
        if len(chosen) >= n:
            break
        chosen.append(item)
        chosen_keys.add(item["key"])
    return chosen[:n]


def observation_rubric(
    *,
    age_months: int = DEFAULT_AGE_MONTHS,
    goals: tuple[str, ...] = (),
    limit: int = 12,
    seed: int = 42,
) -> list[dict]:
    """分析者的观察提纲：不可直接询问儿童的条目（知情者报告 / 临床评定）。

    它们给分析者提供「该看什么」的锚点，使打分不至于只看对话是否流畅。
    """
    items = [
        i for i in _all_items()
        if i["admin"] in ("caregiver_report", "clinician_rating", "observation")
        and age_fit(i, age_months) in ("in_range", "below", "unknown")
    ]
    if goals:
        preferred = [i for i in items if set(goals) & set(i.get("dims") or {})]
        items = preferred or items
    # 临床评定条目最抽象、最能锚定评分，排在前面
    items.sort(key=lambda i: (i["admin"] != "clinician_rating", i["source_id"], i["key"]))
    seen: set[str] = set()
    out: list[dict] = []
    rng = random.Random(seed)
    for item in items:
        if item["key"] in seen:
            continue
        seen.add(item["key"])
        out.append(item)
        if len(out) >= limit:
            break
    _ = rng
    return out


def render_probe_plan(items: list[dict], mode: str = "auto") -> str:
    """把探查条目渲染成给提问者的「计划探查的方面」清单（不照读，供其灵活措辞）。

    情境式条目渲染成一行 `情境：… ｜ 提问：…`，并写明施测方式：先用孩子能听懂的话
    把情境讲清楚，再问「这种情况你会怎么处理」，然后按回答追问。
    """
    dims = dims_map()
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        labels = "、".join(dims.get(d, d) for d in (item.get("dims") or {})) or "—"
        text = probe_text(item, mode)
        scenario = item.get("scenario") or {}
        is_scenario = item.get("probe_mode") == "scenario" or (mode == "scenario" and scenario)
        if is_scenario:
            lines.append(f"{index}. [{labels}] {text.replace(chr(10), ' ｜ ')}")
            if scenario.get("observe"):
                lines.append(f"   观察目标：{scenario['observe']}")
            lines.append("   （先把这个情境用孩子能听懂的话讲清楚，再问「这种情况你会怎么处理」，"
                         "按孩子的回答追问；不要照读）")
        else:
            lines.append(f"{index}. [{labels}] {text}")
            if item.get("admin") != "observation":
                lines.append("   （该条目原为知情者报告条目；这里由你现场观察/自然提问来探查）")
    return "\n".join(lines)


def render_rubric(items: list[dict]) -> str:
    """把观察提纲渲染成给分析者的评分锚点（含该条目的观察要点）。"""
    dims = dims_map()
    lines: list[str] = []
    for item in items:
        labels = "、".join(dims.get(d, d) for d in (item.get("dims") or {})) or "—"
        lines.append(
            f"- [{labels}|{ADMIN_ZH.get(item['admin'], item['admin'])}] "
            f"{item.get('text_zh')}（{item.get('source_name')}）"
        )
        scenario = item.get("scenario") or {}
        if scenario.get("observe"):
            lines.append(f"    观察要点：{scenario['observe']}")
    return "\n".join(lines)
