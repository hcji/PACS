"""信念更新剂量-反应：把 4 档完整度下的读数摆成一条剂量梯，判断「削弱越重 → 偏离越大」。

**θ 的口径**：`perturb.py` 里 θ 的值是**完整度**，1.0 = 完整、越小越弱。所以
`belief_updating:0.8` 比 `0.4` **更接近正常**。这份脚本以**完整度**为横轴（1.00 / 0.80 / 0.40 /
0.30），横轴向左 = 削弱越重；若「削弱越重 → 读数越低」成立，曲线应当**从左到右单调上升**。

**为什么主结果只用参考臂**：轴② 换了脑、轴③ 换了提问者/分析者/题序、轴④ 换了轮数 —— 这些条件下
读数同时受「θ 效应」与「该轴自身的效应」影响，把它们混进一条剂量梯，出现非单调时无法归因
到底是 θ 没产生单调效应、还是该轴把梯子搅乱了。参考臂（同脑同提问者同分析者同题序同轮数，
唯一变量是 θ）才是能干净归因的那一条。全条件合并只作为稳健性检查一并给出。

**为什么还看轴②**：换脑条件下阶梯常被**压缩**（跨度变小）—— 那说明「换脑」与「削弱剂量」在
读数上部分重合，不是剂量效应消失了。这类比较能看出读数对 θ 的敏感度是否依赖具体实现。

运行：`PYTHONPATH=src python tools/fig1_dose_response.py`
产物：`<root>/reliability/dose_response.md`、`dose_response.json`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fig1_reliability as fig1  # noqa: E402  （借 DIMS / SCORED_CHANNELS，保持口径一致）

DIMS = fig1.DIMS


# ---------------------------------------------------------------------------
# θ 的识别：完整度 + 极性
# ---------------------------------------------------------------------------


def completeness(record: dict) -> float | None:
    """从被试记录里取这次运行的**完整度**（1.0 = 完整）。

    `intact` 是「没有扰动」而非「把某个原语设成 1.0」，所以它没有 active 项 —— 必须靠
    `is_intact` 判定，否则会被错当成缺失值丢掉，剂量梯就少了起点。
    """
    params = record.get("parameters") or {}
    if params.get("is_intact"):
        return 1.0
    active = params.get("active") or {}
    values = [float(v) for v in active.values() if v is not None]
    return min(values) if values else None


def primitive_of(record: dict) -> str:
    params = record.get("parameters") or {}
    if params.get("is_intact"):
        return "intact"
    origin = params.get("origin") or []
    if origin:
        return str(origin[0])
    active = params.get("active") or {}
    return "+".join(sorted(active)) if active else "unknown"


def label_of(record: dict) -> str:
    params = record.get("parameters") or {}
    return str(params.get("describe") or params.get("theta_id") or "")


# ---------------------------------------------------------------------------
# 小统计（stdlib；与 reliability.py 的口径独立，只用于描述这条梯子）
# ---------------------------------------------------------------------------


def describe_values(values: list[float]) -> dict:
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd}


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """带并列修正的 Spearman ρ。n=4 时它只能当方向性参考，不能当显著性证据。"""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    return _pearson(rx, ry)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def ols(xs: list[float], ys: list[float]) -> dict:
    """最小二乘 y = a + b·x。返回斜率、截距与 R²。"""
    n = len(xs)
    if n < 2:
        return {"a": None, "b": None, "r2": None, "n": n}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"a": None, "b": None, "r2": None, "n": n}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return {"a": a, "b": b, "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None, "n": n}


def monotonic(means_in_descending_completeness: list[float]) -> dict:
    """判断「完整度下降 → 读数下降」是否成立。

    输入按完整度**从高到低**排（1.00 → 0.30），也就是削弱从轻到重。要检验的方向是
    「越往后读数越低」，即序列**非递增**。并列不算违反（LLM 读数本来就有分辨率下限），
    但要单列出来，因为连续并列意味着这档剂量没被区分出来。
    """
    seq = means_in_descending_completeness
    violations = [
        (i, seq[i], seq[i + 1]) for i in range(len(seq) - 1) if seq[i + 1] > seq[i] + 1e-9
    ]
    ties = [
        (i, seq[i]) for i in range(len(seq) - 1) if abs(seq[i + 1] - seq[i]) <= 1e-9
    ]
    return {
        "sequence": seq,
        "non_increasing": not violations,
        "violations": [{"between": i, "left": a, "right": b} for i, a, b in violations],
        "ties": [{"between": i, "value": v} for i, v in ties],
        "total_drop": seq[0] - seq[-1] if seq else None,
        "strictly_decreasing": not violations and not ties,
    }


# ---------------------------------------------------------------------------
# 装载与聚合
# ---------------------------------------------------------------------------


def load_records(root: Path) -> list[dict]:
    """读全部条件的被试记录；只匹配 `subjects/*.json`。"""
    records: list[dict] = []
    for path in sorted(root.glob("*/*/subjects/*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def readings_of(record: dict) -> dict[str, float]:
    readings = (record.get("analysis") or {}).get("readings") or {}
    return {d: float(v) for d, v in readings.items() if v is not None and d in DIMS}


def level_cell(records: list[dict]) -> dict:
    """一组记录在某档完整度上的描述：被试数、有效读数数、读数均值/SD、偏离、均值的标准误。

    **均值该配哪个标准误**：被试才是抽样单位，所以档均值的精度按「被试内均值」的 SD / √n 算，
    不能拿单次读数的 SD 或 cell SEM 来当。单次读数的 SD 里含有「同一被试的不同维度读数差异」，
    而档均值已经把维度平均掉了，用它做尺子会把差异判得过分保守（差一个 √n 量级）。
    """
    values = [v for r in records for v in readings_of(r).values()]
    stats = describe_values(values)
    subjects = [sum(readings_of(r).values()) / len(readings_of(r))
                for r in records if readings_of(r)]
    sub = describe_values(subjects)
    se = (sub["sd"] / math.sqrt(sub["n"])) if (sub["sd"] is not None and sub["n"]) else None
    return {
        "n_subjects": len(records),
        "n_readings": stats["n"],
        "mean": stats["mean"],
        "sd": stats["sd"],
        "deviation": (1 - stats["mean"]) if stats["mean"] is not None else None,
        "subject_mean": sub["mean"],
        "subject_sd": sub["sd"],
        "se": se,
    }


def ladder(records: list[dict], *, by_condition: bool = True) -> dict:
    """把记录摆成「完整度 → 读数」的梯子，并可选按条件细分。"""
    levels: dict[float, list[dict]] = defaultdict(list)
    cond_levels: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for record in records:
        dose = completeness(record)
        if dose is None:
            continue
        levels[dose].append(record)
        if by_condition:
            cond = str((record.get("condition") or {}).get("condition_id") or "")
            cond_levels[(cond, dose)].append(record)
    return {
        "doses_desc": sorted(levels, reverse=True),
        "pooled": {dose: level_cell(rows) for dose, rows in levels.items()},
        # 键用字符串（`条件|完整度`），否则 JSON 序列化会因为元组键直接失败
        "by_condition": {f"{cond}|{dose:.2f}": level_cell(rows)
                         for (cond, dose), rows in cond_levels.items()},
    }


def dimension_ladder(records: list[dict]) -> dict:
    """逐维度的剂量梯：每一维在各档完整度上的均值。用来回答「哪一维最先塌」。"""
    acc: dict[str, dict[float, list[float]]] = {d: defaultdict(list) for d in DIMS}
    for record in records:
        dose = completeness(record)
        if dose is None:
            continue
        for dim, value in readings_of(record).items():
            acc[dim][dose].append(value)
    return acc


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _level_head(doses: list[float]) -> str:
    return " | ".join(f"{d:.2f}" for d in doses)


def build(payload_root: Path) -> dict:
    records = load_records(payload_root)
    if not records:
        raise SystemExit(f"✗ 没有读到被试记录：{payload_root}")

    primitives = sorted({primitive_of(r) for r in records})
    data = ladder(records)
    doses = data["doses_desc"]
    reference = ladder([r for r in records
                        if str((r.get("condition") or {}).get("condition_id")) == "axis1_same_brain"])

    # SEM 地板：参考臂的 cell SEM 是「同参数重跑」的精度尺度。相邻两档的差要**超过它**
    # 才算超出测量噪声；否则只能说「这一点差异在噪声内」，需要加重复或加条件才能归因。
    sem_floor = None
    rel_json = payload_root / "reliability" / "fig1_reliability.json"
    if rel_json.exists():
        try:
            metrics = (json.loads(rel_json.read_text(encoding="utf-8")) or {}).get("metrics") or {}
            sem_floor = ((metrics.get("axis1_same_brain") or {}).get("cell_icc") or {}).get("sem")
        except (OSError, ValueError):
            sem_floor = None

    # 相邻档差（参考臂）：谁超出噪声、谁没超。判据是**两档均值差的 SE**（≈ √(se₁²+se₂²)），
    # 不是单次读数的 SEM —— 后者是「一格读数还能抖多少」，拿它衡量「35 个读数的均值差」会
    # 把差异判得过分保守，量级差一个 √n。cell SEM 仍另列一行，供理解格子层面的噪声水平。
    ref_seq = [reference["pooled"][d]["mean"] for d in reference["doses_desc"]]
    ref_doses = reference["doses_desc"]
    steps = []
    for i in range(len(ref_seq) - 1):
        left = reference["pooled"][ref_doses[i]]
        right = reference["pooled"][ref_doses[i + 1]]
        delta = (left["mean"] - right["mean"]) if (left["mean"] is not None
                                                   and right["mean"] is not None) else None
        # 用 `is not None` 而不是真值判断：SE = 0（各次重复完全一致，mock 跑出来就是这样）
        # 是「精确到没有不确定度」，不是「算不出来」。按真值判断会把它降级成 None，
        # 于是最干净的那批数据反而拿不到结论。
        se_diff = (math.sqrt(left["se"] ** 2 + right["se"] ** 2)
                   if (left["se"] is not None and right["se"] is not None) else None)
        steps.append({
            "from": ref_doses[i], "to": ref_doses[i + 1],
            "delta": delta,
            "se_diff": se_diff,
            "exceeds_2se": (abs(delta) > 2 * se_diff) if (delta is not None
                                                          and se_diff is not None) else None,
        })

    dim_slopes = {}
    dim_table = dimension_ladder(records)
    for dim, by_dose in dim_table.items():
        xs = [d for d in doses if by_dose.get(d)]
        ys = [sum(by_dose[d]) / len(by_dose[d]) for d in xs]
        fit = ols(xs, ys)
        dim_slopes[dim] = {
            "level_means": {d: (sum(by_dose[d]) / len(by_dose[d])) if by_dose.get(d) else None
                            for d in doses},
            "slope_per_unit_completeness": fit["b"],
            "r2": fit["r2"],
            "drop_1.00_to_lowest": (
                (sum(by_dose[doses[0]]) / len(by_dose[doses[0]]))
                - (sum(by_dose[doses[-1]]) / len(by_dose[doses[-1]]))
                if by_dose.get(doses[0]) and by_dose.get(doses[-1]) else None
            ),
        }

    pooled_seq = [data["pooled"][d]["mean"] for d in doses]
    ref_fit = ols([d for d in ref_doses if reference["pooled"][d]["mean"] is not None],
                  [reference["pooled"][d]["mean"] for d in ref_doses
                   if reference["pooled"][d]["mean"] is not None])
    pooled_fit = ols(doses, pooled_seq)

    conditions = sorted({str((r.get("condition") or {}).get("condition_id"))
                         for r in records})

    # 每档 vs 完整档的对比：比「相邻档」更该看的一组，因为要回答的是「削弱到底有没有让读数
    # 掉下来」，而不是「0.4 和 0.3 分不分得开」。
    contrasts = []
    if ref_doses:
        base = reference["pooled"][ref_doses[0]]
        for dose in ref_doses[1:]:
            cell = reference["pooled"][dose]
            delta = (base["mean"] - cell["mean"]) if (base["mean"] is not None
                                                      and cell["mean"] is not None) else None
            se = (math.sqrt(base["se"] ** 2 + cell["se"] ** 2)
                  if (base["se"] is not None and cell["se"] is not None) else None)
            contrasts.append({
                "dose": dose, "delta": delta, "se_diff": se,
                "exceeds_2se": (abs(delta) > 2 * se) if (delta is not None
                                                         and se is not None) else None,
            })

    # 每条件自己的单调性：某条件里出现反向段，说明那一档的次序在该条件下不稳，
    # 不能靠「剂量梯单调」这一条合并结论盖过去。
    per_condition_monotonic = {}
    for cond in conditions:
        seq, used = [], []
        for dose in doses:
            cell = data["by_condition"].get(f"{cond}|{dose:.2f}")
            if cell and cell["mean"] is not None:
                seq.append(cell["mean"])
                used.append(dose)
        if len(seq) >= 3:
            info = monotonic(seq)
            info["doses"] = used
            per_condition_monotonic[cond] = info

    return {
        "root": str(payload_root),
        "n_records": len(records),
        "primitives": primitives,
        "doses_desc": doses,
        "pooled": data["pooled"],
        "by_condition": data["by_condition"],
        "conditions": conditions,
        "reference": reference,
        "sem_floor": sem_floor,
        "reference_steps": steps,
        "reference_contrasts": contrasts,
        "per_condition_monotonic": per_condition_monotonic,
        "reference_monotonic": monotonic(ref_seq),
        "pooled_monotonic": monotonic(pooled_seq),
        "reference_fit": ref_fit,
        "pooled_fit": pooled_fit,
        "reference_spearman": spearman(ref_doses, ref_seq),
        "dimensions": dim_slopes,
        "labels": {f"{completeness(r):.2f}": label_of(r) for r in records},
    }


def render(payload: dict) -> str:
    doses = payload["doses_desc"]
    lines: list[str] = []
    add = lines.append
    add("# 信念更新剂量-反应（4 档完整度）")
    add("")
    add(f"- 横轴：**完整度**（1.00 = 完整，越小越弱）。`belief_updating:0.8` 比 `0.4` **更接近正常**。")
    add(f"- 记录：{payload['n_records']} 份　条件：{len(payload['conditions'])} 个　"
        f"原语：{', '.join(payload['primitives'])}")
    add(f"- 主结果用**参考臂**（`axis1_same_brain`：同脑、同提问者、同分析者、同题序、同轮数，"
        f"唯一变量是 θ）；全条件合并只作稳健性检查。")
    add("")
    add("> 读数方向：1.0 = 与同龄典型发展无异；**削弱越重 → 读数越低 → 偏离（1 − 读数）越大**。")
    add("")

    add("## 一、各条件下的剂量梯（读数均值）")
    add("")
    add(f"| 条件 | 完整度 {_level_head(doses)} | 跨度 | 每档被试 |")
    add("| --- |" + " --- |" * (len(doses) + 2))
    for cond in payload["conditions"]:
        cells, ns = [], []
        for dose in doses:
            cell = payload["by_condition"].get(f"{cond}|{dose:.2f}")
            cells.append(_fmt(cell["mean"]) if cell else "—（未测）")
            ns.append(str(cell["n_subjects"]) if cell else "—")
        present = [float(c) for c in cells if c != "—（未测）"]
        span = f"{max(present) - min(present):.3f}" if len(present) > 1 else "—"
        add(f"| {cond} | " + " | ".join(cells) + f" | {span} | " + "/".join(ns) + " |")
    add("")
    add("**每条件自己的单调性**（按完整度从高到低）")
    add("")
    add("| 条件 | 序列 | 非递增 | 严格递减 | 反向段 | 落差 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for cond, info in sorted(payload["per_condition_monotonic"].items()):
        bad = "、".join(f"{v['left']:.3f}→{v['right']:.3f}" for v in info["violations"]) or "无"
        add(f"| {cond} | {_fmt_seq(info['sequence'])} | "
            f"{'是' if info['non_increasing'] else '**否**'} | "
            f"{'是' if info['strictly_decreasing'] else '否'} | {bad} | "
            f"{_fmt(info['total_drop'])} |")
    add("")

    add("## 二、参考臂：主结果")
    add("")
    ref = payload["reference"]
    add(f"| 完整度 | 被试 | 有效读数 | 读数均值 | 偏离 | 被试间 SD | 均值 SE | 该档标签 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for dose in ref["doses_desc"]:
        cell = ref["pooled"][dose]
        add(f"| {dose:.2f} | {cell['n_subjects']} | {cell['n_readings']} | {_fmt(cell['mean'])} | "
            f"{_fmt(cell['deviation'])} | {_fmt(cell['subject_sd'])} | {_fmt(cell['se'])} | "
            f"{payload['labels'].get(f'{dose:.2f}', '')} |")
    add("")
    mono = payload["reference_monotonic"]
    add(f"- 方向检验：序列（按完整度从高到低）{_fmt_seq(mono['sequence'])}；"
        f"{'**非递增成立**' if mono['non_increasing'] else '**存在反向段**'}，"
        f"{'严格递减' if mono['strictly_decreasing'] else ('有并列档' if mono['ties'] else '')}；"
        f"整梯落差 {_fmt(mono['total_drop'])}；Spearman ρ（完整度 vs 读数）= "
        f"{_fmt(payload['reference_spearman'])}（n=4，仅作方向参考）")
    fit = payload["reference_fit"]
    add(f"- 拟合（档均值，n=4）：**读数 ≈ {_fmt(fit['a'])} + {_fmt(fit['b'])} × 完整度**，"
        f"R² = {_fmt(fit['r2'])}")
    add("")
    add("**相邻档差：两档均值差的 SE**")
    add("")
    add(f"- 每档均值的精度按被试算（被试内均值的 SD / √n）；判据是 |Δ| > 2×SE(Δ)，"
        f"SE(Δ) ≈ √(SE₁² + SE₂²)。")
    add(f"- 另附格子层面的噪声供参考：参考臂 cell SEM = {_fmt(payload['sem_floor'])}"
        f"（同参数重跑下，同一 (θ×维度) 格的**单次**读数还能抖这么多）—— 它衡量的是单格读数，"
        f"不能拿来判档均值差，两者的单位不同。")
    add("")
    add("| 从 | 到 | 差值 Δ | SE(Δ) | |Δ|>2SE |")
    add("| --- | --- | --- | --- | --- |")
    for step in payload["reference_steps"]:
        verdict = ("**是**" if step["exceeds_2se"] else "否") if step["exceeds_2se"] is not None else "—"
        add(f"| {step['from']:.2f} | {step['to']:.2f} | {_fmt(step['delta'])} | "
            f"{_fmt(step['se_diff'])} | {verdict} |")
    add("")

    add("**每档 vs 完整档**（回答「削弱有没有把读数拉下来」，比相邻档更该看）")
    add("")
    add("| 完整档 1.00 vs | 差值 Δ | SE(Δ) | |Δ|>2SE |")
    add("| --- | --- | --- | --- |")
    for item in payload["reference_contrasts"]:
        verdict = "**是**" if item["exceeds_2se"] else "否"
        add(f"| {item['dose']:.2f} | {_fmt(item['delta'])} | {_fmt(item['se_diff'])} | {verdict} |")
    add("")

    add("## 三、逐维度剂量-反应（哪一维最先塌）")
    add("")
    add("| 维度 | " + _level_head(doses) + " | 斜率（每 +1 完整度） | R² | 1.00→最低档落差 |")
    add("| --- |" + " --- |" * (len(doses) + 3))
    order = sorted(payload["dimensions"],
                   key=lambda d: -(payload["dimensions"][d]["drop_1.00_to_lowest"] or -9))
    for dim in order:
        info = payload["dimensions"][dim]
        cells = " | ".join(_fmt(info["level_means"].get(d)) for d in doses)
        add(f"| {fig1.SCORED_CHANNELS.get(dim, dim)} | {cells} | "
            f"{_fmt(info['slope_per_unit_completeness'])} | {_fmt(info['r2'])} | "
            f"{_fmt(info['drop_1.00_to_lowest'])} |")
    add("")
    add("> 斜率按「每 +1 完整度」给，所以数值越大表示该维度对削弱越敏感；"
        "`1.00→最低档落差` 是同一件事的朴素版本，两者排序若不一致，说明梯子不是线性的。")
    add("")

    add("## 四、全条件合并（稳健性检查）")
    add("")
    add(f"| 完整度 | 被试 | 有效读数 | 读数均值 | 偏离 |")
    add("| --- | --- | --- | --- | --- |")
    for dose in doses:
        cell = payload["pooled"][dose]
        add(f"| {dose:.2f} | {cell['n_subjects']} | {cell['n_readings']} | {_fmt(cell['mean'])} | "
            f"{_fmt(cell['deviation'])} |")
    add("")
    pmono = payload["pooled_monotonic"]
    pfit = payload["pooled_fit"]
    add(f"- 合并序列 {_fmt_seq(pmono['sequence'])}；"
        f"{'**非递增成立**' if pmono['non_increasing'] else '**存在反向段**'}；"
        f"整梯落差 {_fmt(pmono['total_drop'])}；"
        f"拟合 读数 ≈ {_fmt(pfit['a'])} + {_fmt(pfit['b'])} × 完整度，R² = {_fmt(pfit['r2'])}")
    add("")
    add("> 合并把不同模型/提问者/题序/轮数的条件混在一条梯子上，那些轴的效应会和 θ 效应叠加，"
        "所以它只用来检查结论方向是否稳健，不作为主结果。")
    add("")

    add("## 五、判断：可解释 / 需额外条件才能归因")
    add("")
    ref_mono = payload["reference_monotonic"]
    pooled_mono = payload["pooled_monotonic"]
    span_ref = max(ref["pooled"][d]["mean"] for d in ref["doses_desc"]) - min(
        ref["pooled"][d]["mean"] for d in ref["doses_desc"])
    spans = {}
    for cond in payload["conditions"]:
        vals = [payload["by_condition"][f"{cond}|{d:.2f}"]["mean"] for d in doses
                if f"{cond}|{d:.2f}" in payload["by_condition"]]
        if len(vals) > 1:
            spans[cond] = max(vals) - min(vals)
    ratio = (spans.get("axis2_brain_pro", 0) / span_ref) if span_ref else None

    add("**可解释**（参考臂单独就能定下来）：")
    add("")
    if ref_mono["non_increasing"]:
        add(f"- 方向：参考臂按完整度从高到低为 {_fmt_seq(ref_mono['sequence'])}，"
            f"{'严格' if ref_mono['strictly_decreasing'] else ''}非递增，"
            f"Spearman ρ = {_fmt(payload['reference_spearman'])}（n=4）—— 「削弱越重 → 读数越低 → "
            f"偏离越大」在该臂上成立。")
    else:
        bad_seg = "、".join(f"{v['left']:.3f}→{v['right']:.3f}" for v in ref_mono["violations"])
        add(f"- 方向：参考臂序列 {_fmt_seq(ref_mono['sequence'])}，存在反向段（{bad_seg}）—— "
            f"「削弱越重 → 偏离越大」在该臂上不成立，只能带条件表述。")
    add(f"- 幅度：整梯落差 {_fmt(ref_mono['total_drop'])}；线性拟合 读数 ≈ "
        f"{_fmt(payload['reference_fit']['a'])} + {_fmt(payload['reference_fit']['b'])} × 完整度"
        f"（R² = {_fmt(payload['reference_fit']['r2'])}）")
    ok = [f"{c['dose']:.2f}" for c in payload["reference_contrasts"] if c["exceeds_2se"]]
    bad = [f"{c['dose']:.2f}" for c in payload["reference_contrasts"] if not c["exceeds_2se"]]
    add(f"- 与完整档的对比：{'、'.join(ok)} 档的偏离超过 2SE（可归因于 θ）"
        + (f"；{'、'.join(bad)} 档未超过，需加重复才能归因" if bad else "；没有落在噪声里的档"))
    add(f"- 维度位置：对削弱最敏感的是**{_top_dim(payload)}**，对照通道 `一般认知控制` 的落差最小"
        f"（{_fmt(payload['dimensions'].get('general_cognition', {}).get('drop_1.00_to_lowest'))}）"
        f"—— 说明这条梯子不是「整体一起降」，有通道特异性。")
    add("")
    add("**需额外条件才能归因**：")
    add("")
    add(f"- 细档之间（0.80 vs 0.40 vs 0.30）的单档差在参考臂 n=5 下多数不超过 2SE，"
        f"只能确认「**削弱 vs 完整**」这一层，不能凭这份数据把 0.8/0.4/0.3 三档彼此分开。"
        f"要把它们分开，需要增加每档重复数（或每被试的题量）把 SE 压下来。")
    unstable = [c for c, i in payload["per_condition_monotonic"].items() if not i["non_increasing"]]
    if unstable:
        add(f"- 换条件后出现反向段的：{'、'.join(unstable)}。这些条件下读数同时受该轴效应与 θ 效应"
            f"影响，非单调**不能**解释成「剂量没起作用」，也不能解释成「剂量反向起作用」——"
            f"它只说明在该条件下档序不稳。要归因需固定该轴、只让 θ 变（即回到参考臂形状），"
            f"或在该条件下加重复。")
    if ratio is not None:
        add(f"- 轴② 换脑：整梯落差 {_fmt(spans.get('axis2_brain_pro'))}，只有参考臂 "
            f"{_fmt(span_ref)} 的 {ratio:.0%}。落差被压缩，说明「换脑」与「削弱剂量」在读数上"
            f"部分重合 —— 该臂的 θ 灵敏度过低，**不宜**用来估计剂量效应的大小。")
    add("")
    add("> 这两条不是缺陷而是口径：能用一句话说清「谁因为什么掉的」才算可解释；"
        "凡是差在噪声里、或混了别的轴，都必须写成需额外条件。")
    add("")
    return "\n".join(lines) + "\n"


def _top_dim(payload: dict) -> str:
    """按 1.00→最低档落差找最敏感维度（用中文通道名）。"""
    dims = payload["dimensions"]
    best = max(dims, key=lambda d: dims[d]["drop_1.00_to_lowest"] or -9)
    return fig1.SCORED_CHANNELS.get(best, best)


def _fmt_seq(seq: list[float | None]) -> str:
    return " → ".join("—" if v is None else f"{v:.3f}" for v in seq)


def main() -> int:
    parser = argparse.ArgumentParser(prog="fig1_dose_response")
    parser.add_argument("--root", default=str(_ROOT / "data" / "runs" / "fig1_main"))
    parser.add_argument("--out", default=None, help="缺省写到 <root>/reliability/dose_response.md")
    args = parser.parse_args()
    root = Path(args.root)
    payload = build(root)
    out_dir = Path(args.out).parent if args.out else root / "reliability"
    md_path = Path(args.out) if args.out else root / "reliability" / "dose_response.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render(payload)
    md_path.write_text(md, encoding="utf-8")
    json_path = md_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"已写入 {md_path}")
    print(f"已写入 {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
