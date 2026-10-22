"""用户上传的量表（界面「量表上传」入口）。

**上传的量表一律由分析者 LLM 逐条打分**（`llm`）：量表条目会作为「请依据同一段对话逐条给出
该条目上的表现读数」写进分析者的输入（见 `brain/architecture.build_analyst_prompt`）。
这与表型的读取通道一致 —— 全部读数的来源都是 LLM 对回应序列的解读。

本模块只有 `llm` 一种模式：上传量表的每个条目都由分析者 LLM 逐条打分。
题目里的 `key` / `primitive_sensitivity` 只作出处信息保留，不参与打分；
`parse_item_set` 会在 `notices` 里显式说明这一点
（「禁止静默缺失」：口径变了就必须让人看见，而不是悄悄算出另一套数）。

为什么不做「缺权重就默认给个权重」或「照旧按正确率算」：那会让统计口径凭空捏造，
把上传量表伪装成与内置量表等价的客观测量 —— 宁可显式降级为 LLM 打分并标注出来。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .phenotype import SCORED_CHANNELS

SCORING_LLM = "llm"


def _as_options(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)):
        return tuple(str(o) for o in raw)
    return ()


def parse_item_set(payload: dict, *, name: str = "") -> tuple[str, dict, list[str]]:
    """校验并归一化一份上传的量表，返回 `(set_id, normalized, problems)`。

    `problems` 非空即视为校验失败 —— 宁可拒绝，也不静默丢弃条目：
    题目被悄悄丢掉的量表会得到一个看似正常、实则错误的表型。
    """
    problems: list[str] = []
    notices: list[str] = []
    if not isinstance(payload, dict):
        return "", {}, ["上传内容必须是一个 JSON 对象"]

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return "", {}, ["缺少 items 数组，或数组为空"]

    set_name = str(payload.get("name") or name or "").strip()
    if not set_name:
        problems.append("缺少量表名称（name）")

    items: list[dict] = []
    seen_ids: set[str] = set()
    saw_forced_choice_field = False

    for idx, raw in enumerate(raw_items):
        where = f"第 {idx + 1} 条"
        if not isinstance(raw, dict):
            problems.append(f"{where}：不是对象")
            continue
        item_id = str(raw.get("id") or "").strip()
        if not item_id:
            problems.append(f"{where}：缺少 id")
            continue
        if item_id in seen_ids:
            problems.append(f"{where}：id 重复（{item_id}）")
            continue
        seen_ids.add(item_id)

        dim = str(raw.get("dim") or raw.get("battery_dim") or "").strip()
        if not dim:
            problems.append(f"{where}：缺少 dim（维度）")
            continue

        prompt = str(raw.get("prompt") or raw.get("text") or "").strip()
        if not prompt:
            problems.append(f"{where}：缺少 prompt（题干）")
            continue

        raw_options = raw.get("options")
        if raw_options is not None:
            options = _as_options(raw_options)
            if len(options) < 2:
                problems.append(f"{where}：options 至少要有 2 个选项")
                continue
        else:
            options = ("（开放作答）",)

        # 以下三项是旧定量管线的字段：不再参与打分，但保留在条目里作出处信息。
        legacy: dict = {}
        if raw.get("key") is not None:
            try:
                key = int(raw["key"])
            except (TypeError, ValueError):
                problems.append(f"{where}：key 不是整数")
                continue
            if not (0 <= key < len(options)):
                problems.append(f"{where}：key={key} 超出选项范围 0..{len(options) - 1}")
                continue
            legacy["key"] = key
            saw_forced_choice_field = True
        sensitivity_raw = raw.get("primitive_sensitivity") or raw.get("sensitivity") or {}
        if isinstance(sensitivity_raw, dict):
            sensitivity: dict[str, float] = {}
            for pid, weight in sensitivity_raw.items():
                try:
                    sensitivity[str(pid)] = float(weight)
                except (TypeError, ValueError):
                    problems.append(f"{where}：敏感度 {pid} 不是数值")
            if sensitivity:
                legacy["primitive_sensitivity"] = sensitivity
                saw_forced_choice_field = True

        items.append({
            "id": item_id,
            "dim": dim,
            "level1_target": str(raw.get("level1_target") or dim),
            "prompt": prompt,
            "options": list(options),
            "note": str(raw.get("note") or f"上传量表 {set_name}"),
            "legacy": legacy,
        })

    if not items:
        problems.append("没有一条可用条目")
        return "", {}, problems

    if saw_forced_choice_field:
        notices.append(
            "题目里带有 key / primitive_sensitivity 字段：这些字段只作出处信息保留，"
            "本量表按**分析者 LLM 逐条打分**处理。"
        )
    custom_dims = sorted({i["dim"] for i in items if i["dim"] not in SCORED_CHANNELS})
    if custom_dims:
        notices.append(
            "含自定义维度（" + "、".join(custom_dims) + "）：逐条读数照常给出并落库，"
            "但下游的机制分析只定义在内置维度与对照通道上，自定义维度不参与那些分析。"
        )

    set_id = "set_" + hashlib.sha256(
        json.dumps([set_name, [i["id"] for i in items]], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]

    normalized = {
        "set_id": set_id,
        "name": set_name,
        "scoring_mode": SCORING_LLM,
        "mode_reason": "本版本的表型全部由 LLM 读出：量表条目写进分析者的输入，由它依据同一段对话逐条打分",
        "notices": notices,
        "dims": _dims_of(items),
        "items": items,
    }
    return set_id, normalized, problems


def _dims_of(items: list[dict]) -> list[dict]:
    """量表覆盖的维度（内置维度给中文名，自定义维度原样保留并标注）。"""
    seen: list[str] = []
    for item in items:
        if item["dim"] not in seen:
            seen.append(item["dim"])
    return [
        {"dim": dim, "label": SCORED_CHANNELS.get(dim, f"{dim}（自定义维度）"),
         "builtin": dim in SCORED_CHANNELS}
        for dim in seen
    ]
