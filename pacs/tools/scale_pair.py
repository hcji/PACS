"""两两组合（465 个 pair）· 规模化数据产生 —— 单维度批次的组合臂。

矩阵：**C(31,2) = 465 个 pair × 3 次重复 = 1395 个新被试**，两个 primitive 的完整度都固定
**0.8（轻度）**。轮次、模型、题池、基线全部与 `scale_single.py` 的单维度批次一致。

## 为什么单独一个脚本

组合批的回答对象完全不同：单维度批问「把某个认知过程削弱到某一档，表型上出现什么」，
组合批问「**两个过程同时削弱时，损伤是可加的、超加的（协同），还是次加的（拮抗）**」。
聚合层因此不是剂量-反应/特异性，而是 **交互效应**（`analysis/heterogeneity.py` 的
`evaluate_interactions`）与**机制异质性**（`find_convergence`）。

编排与装盘部分**一行都不重写**：逐单元格运行、失败重试、分片、按单元格续跑、逐被试记录、
扁平表全部直接调 `scale_single`（它自己又复用了 `fig1_reliability`）。`scale_single.py`
与 `fig1_reliability.py` 均**不改动**。

## 可加性参照来自已经跑完的单维度批次（关键设计）

`evaluate_interactions` 的可加性预期是 `single_a + single_b − baseline`，并且它只在
**单因素与组合处于同一 dose** 时才做比较（`abs(la - lvl) > 1e-6` 就跳过）。所以：

- 组合臂统一取 **0.8**，正好与单维度批次的轻度档对齐；
- 单因素参照**不重跑**，直接读 `data/runs/scale_single` 里 1 个 primitive 被削到 0.8 的那些被试；
- 读之前有**硬门**（`single_reference_issues`）：协议、三方模型、题池逐位一致才允许用，
  否则整批的 `expected_additive` 都建立在不可比的参照上（而这不会报错，只会算出一个错的数）。

## 必须先对重复求平均，否则交互效应会算错

`evaluate_interactions` 内部是 `singles[pid] = (level, value)` —— **覆盖赋值**。直接把 3 次
重复的剖面喂进去，它只会静默采用列表里**最后一个**重复，而不是均值；组合侧同理
（`pairs[(a,b,lvl)] = value`）。这不会报错，只会让 465 个交互量各自建立在一个随机抽样上。
因此本脚本先按（机制组合）把重复**取均值**再送进去（`average_groups`），噪声降到约 1/√3。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fig1_reliability as fig1  # noqa: E402
import scale_single as s1  # noqa: E402
from pacs import DEFAULT_AGE_MONTHS, SIMULATION_NOTICE, __version__  # noqa: E402
from pacs.analysis import heterogeneity as hg  # noqa: E402
from pacs.analysis.stats import mean  # noqa: E402
from pacs.brain import examiner_bank as eb  # noqa: E402
from pacs.brain.perturb import Perturbation, compose, primitive_zh, single  # noqa: E402
from pacs.config import PROJECT_ROOT, Settings  # noqa: E402
from pacs.ontology.primitives import DEFAULT_ACTIVE  # noqa: E402
from pacs.phenotype import (  # noqa: E402
    CONTROL_DIM,
    PHENOTYPE_DIMS,
    SCORED_CHANNELS,
    PhenotypeProfile,
    channel_label,
    synthetic_count,
)

#: 组合臂的受损程度：两个 primitive 都取轻度 0.8（与单维度批的轻度档同剂量，可加性比较才成立）
PAIR_DOSE = 0.8
#: 每个组合的重复次数
PAIR_REPLICATES = s1.DEFAULT_REPLICATES

AXIS = "scale_pair"
AXIS_NAME = "两两组合（轻度 0.8）"
#: 协议与单维度批完全相同 —— 直接引用它的常量，避免两批悄悄分叉
PROTOCOL = dict(s1.PROTOCOL)
DISCLAIMER = s1.DISCLAIMER

#: 单维度批次的默认位置（可加性参照来源）
SINGLE_DIR_REL = Path("data") / "runs" / "scale_single"

LIMITATIONS: tuple[str, ...] = (
    "组合臂的受损程度统一为轻度 0.8，**只有一个剂量档**：因此本批能判「可加/超加/次加」，"
    "但**不能**回答「交互随剂量怎么变」——那需要组合臂也做剂量梯度。",
    "可加性参照是**另一个批次**（单维度批的 0.8 档）的读数，不是同一批同时测的。两批的"
    "协议/模型/题池逐位一致（见第 0 节校验），但它们仍是两次独立运行 —— 跨批比较的"
    "不确定性没有被算进交互量的区间里。",
    "3 次重复（取均值后）仍是低强度：单次对话的逐维读数噪声约 ±0.08–0.13，交互效应往往比"
    "单因素效应小一个量级，因此大量 pair 的 `interaction` 会落在噪声带内 —— 报告给出的是"
    "**哪些组合的偏离超过了噪声**，而不是「每个组合的交互真值」。",
    "`evaluate_interactions` 只在单因素与组合**同剂量**时比较，因此本批全部 465 对都参与"
    "（单维度批的 0.8 档覆盖全部 31 个 primitive）；缺任一单因素参照的对会被静默跳过，"
    "报告里显式给出被跳过的对数。",
    "单因素设计不识别高阶交互：本批只做**二阶**组合，三阶及以上的联合损伤不在范围内。",
    "仿真产物（虚拟被试），非真实儿童数据；仅用于方法学研究，不用于诊断。",
)


def limitations_for(n_pairs: int, n_skipped: int) -> list[str]:
    """把第 4 条改写成与本批实际覆盖率相符的表述。"""
    fourth = (
        f"本批 {n_pairs} 对里，**{n_skipped} 对**因缺少同剂量的单因素参照而被交互分析跳过"
        + "（本批全部 465 对都应有参照，出现跳过说明单维度批的 0.8 档有缺口）。"
        if n_skipped else
        f"本批 {n_pairs} 对**全部**都有同剂量的单因素参照（单维度批的 0.8 档覆盖 31 个 primitive），"
        "没有对被跳过。"
    )
    return [fourth if i == 4 else item for i, item in enumerate(LIMITATIONS, start=1)]


# ---------------------------------------------------------------------------
# 组合网格
# ---------------------------------------------------------------------------


def all_pairs(primitives: Sequence[str]) -> list[tuple[str, str]]:
    """全部二元组合，**确定性顺序**（itertools.combinations 按传入顺序）。"""
    return list(itertools.combinations(primitives, 2))


def pair_condition_id(a: str, b: str) -> str:
    """`pair_<a>__<b>`。用双下划线分隔，使 `a`/`b` 可以从 id 里无歧义地还原。"""
    return f"pair_{a}__{b}"


def pair_conditions(
    *, primitives: Sequence[str] = DEFAULT_ACTIVE, dose: float = PAIR_DOSE,
    replicates: int = PAIR_REPLICATES, models: tuple[str, str, str],
) -> list[fig1.Condition]:
    """465 个条件：**一个 pair 一个条件**。

    一 pair 一条件的好处与单维度批相同：条件是分片与按单元格续跑的单位，某片崩了只影响自己；
    而且 465 个条件分给 N 片后负载天然均衡（每片约 465/N 个）。
    """
    brain, examiner, analyst = models
    out: list[fig1.Condition] = []
    for a, b in all_pairs(primitives):
        out.append(
            fig1.Condition(
                condition_id=pair_condition_id(a, b),
                axis=AXIS,
                axis_name=AXIS_NAME,
                label=(f"{primitive_zh(a)} + {primitive_zh(b)}：两者完整度均 {dose:g}，"
                       f"{replicates} 次重复"),
                brain_model=brain, examiner_model=examiner, analyst_model=analyst,
                seed=PROTOCOL["seed"], n_turns=PROTOCOL["n_turns"],
                max_turns=PROTOCOL["max_turns"], replicates=replicates,
                n_thetas=1, probe_mode=PROTOCOL["probe_mode"],
                note=f"二阶组合扰动：{a} 与 {b} 同时削到 {dose:g}",
            )
        )
    return out


def pair_thetas(condition: fig1.Condition, *, dose: float = PAIR_DOSE) -> list[Perturbation]:
    """一个条件的 θ：两个 primitive 同时削到同一剂量（`compose` 取最弱者，此处两者相等）。"""
    a, b = condition.condition_id.removeprefix("pair_").split("__", 1)
    return [compose(single(a, dose, seed=PROTOCOL["seed"]),
                    single(b, dose, seed=PROTOCOL["seed"]),
                    seed=PROTOCOL["seed"])]


def _cost(conditions: Sequence[fig1.Condition], dose: float) -> dict:
    cells = sum(len(pair_thetas(c, dose=dose)) * c.replicates for c in conditions)
    lower = sum(len(pair_thetas(c, dose=dose)) * c.replicates * (c.n_turns * 2 + 1)
                for c in conditions)
    upper = sum(len(pair_thetas(c, dose=dose)) * c.replicates * (c.max_turns * 2 + 1)
                for c in conditions)
    return {
        "n_conditions": len(conditions), "n_cells": cells,
        "calls_lower": lower, "calls_upper": upper,
        "formula": "单元格数 × (轮数 × 2 + 1)：每轮一次提问 + 一次作答，每被试一次分析者读数",
    }


# ---------------------------------------------------------------------------
# 可加性参照：读已经跑完的单维度批次（带硬门）
# ---------------------------------------------------------------------------


def single_reference_issues(
    single_dir: Path, *, model_id: str, probe_keys: Sequence[str], protocol: dict, dose: float
) -> list[str]:
    """校验单维度批是否与本批同协议 —— 不通过就不能拿它当可加性参照。

    判据与基线复用的那道门同源（协议常量、三方模型、题池逐位一致），另加一条：参照批次必须
    真的覆盖了本批的剂量档。最后一条容易漏：协议全对但剂量档不同（比如只有 0.4），
    `evaluate_interactions` 会把每一对都静默跳过，报告里只剩一张空表。
    """
    issues: list[str] = []
    manifest_path = single_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"参照批次的 manifest.json 不存在：{manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"参照批次 manifest.json 读不出来：{exc}"]

    ref_protocol = manifest.get("protocol") or {}
    for key, expected in protocol.items():
        actual = ref_protocol.get(key)
        if actual != expected:
            issues.append(f"协议不一致：{key} 参照={actual!r}，本批={expected!r}")
    models = manifest.get("models") or {}
    for role in ("brain", "examiner", "analyst"):
        if models.get(role) != model_id.split(":")[-1]:
            issues.append(f"模型不一致：{role} 参照={models.get(role)!r}，本批={model_id!r}")
    ref_keys = list(manifest.get("probe_keys") or [])
    if ref_keys != list(probe_keys):
        issues.append(f"**题池不一致**（{len(ref_keys)} vs {len(probe_keys)}）—— "
                      "参照批次与本批问的不是同一批问题，可加性比较不成立")
    doses = [float(d) for d in (manifest.get("doses") or [])]
    if not any(abs(d - dose) < 1e-9 for d in doses):
        issues.append(f"参照批次没有 {dose:g} 这个剂量档（它有 {doses}）—— "
                      "交互分析会把每一对都跳过，只剩空表")
    return issues


def load_single_reference(single_dir: Path, *, dose: float) -> list[dict]:
    """读参照批次里「恰好一个 primitive 被削到 `dose`」的被试记录。

    **只收剂量完全相等的**：把 0.4 的单因素混进来，`evaluate_interactions` 内部的
    `singles[pid]` 会被后出现的那条覆盖，然后剂量校验又把该 pair 整条跳过 ——
    表现为「没有任何交互结果」，而看不出是参照混了剂量。
    """
    out: list[dict] = []
    if not (single_dir / "cells").exists():
        return out
    for path in sorted((single_dir / "cells").glob("*/subjects/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        params = payload.get("parameters") or {}
        if float(params.get("generic_degradation") or 0.0) > 0:
            continue
        active = params.get("active") or {}
        if len(active) != 1:
            continue
        if abs(float(next(iter(active.values()))) - dose) > 1e-9:
            continue
        payload["_source_path"] = str(path)
        out.append(payload)
    return out


# ---------------------------------------------------------------------------
# 交互分析
# ---------------------------------------------------------------------------


def average_groups(
    profiles: Sequence[PhenotypeProfile], key: Callable[[PhenotypeProfile], tuple]
) -> list[PhenotypeProfile]:
    """按 `key` 把重复剖面**取均值**合成一条。

    为什么非做不可：`evaluate_interactions` 里 `singles[pid] = (level, value)` 与
    `pairs[(a, b, lvl)] = value` 都是**覆盖赋值**，喂 3 次重复只会采纳列表里最后一个 ——
    不报错，但 465 个交互量各自建立在一个随机抽样上。先平均把噪声降到约 1/√k。
    """
    groups: dict[tuple, list[PhenotypeProfile]] = {}
    for profile in profiles:
        groups.setdefault(key(profile), []).append(profile)

    out: list[PhenotypeProfile] = []
    for group_key, bucket in sorted(groups.items(), key=lambda kv: str(kv[0])):
        first = bucket[0]
        dims: dict[str, float | None] = {}
        for ch in SCORED_CHANNELS:
            values = [float(p.dims[ch]) for p in bucket if p.dims.get(ch) is not None]
            dims[ch] = sum(values) / len(values) if values else None
        confidences = [p.confidence for p in bucket if p.confidence is not None]
        out.append(
            PhenotypeProfile(
                theta_id=f"avg_{'_'.join(group_key)}",
                model_id=first.model_id,
                run_id=first.run_id,
                age_months=first.age_months,
                theta=dict(first.theta),
                generic_degradation=0.0,
                dims=dims,
                n_items=sum(p.n_items for p in bucket),
                n_parsed=sum(p.n_parsed for p in bucket),
                replicate=0,
                design="replicate_mean",
                note=f"由 {len(bucket)} 次重复取平均",
                instrument=first.instrument,
                confidence=(sum(confidences) / len(confidences)) if confidences else None,
                provider_kind="mock" if any(p.synthetic for p in bucket) else "llm",
            )
        )
    return out


def _interaction_row(a: str, b: str, dose: float, ch: str, value: "hg.InteractionResult") -> dict:
    return {
        "primitive_a": a, "primitive_b": b, "dose": dose, "channel": ch,
        "channel_zh": channel_label(ch), "single_a": value.single_a, "single_b": value.single_b,
        "combined": value.combined, "baseline": value.baseline,
        "expected_additive": value.expected_additive, "interaction": value.interaction,
        "kind": value.kind,
    }


def interaction_analysis(
    combined: Sequence[PhenotypeProfile], *, dose: float, channels: Sequence[str] = tuple(SCORED_CHANNELS)
) -> dict:
    """逐通道的交互效应 + 逐对的汇总。"""
    per_channel: dict[str, list[dict]] = {}
    for ch in channels:
        results = hg.evaluate_interactions(list(combined), ch, dose=dose)
        per_channel[ch] = [_interaction_row(r.primitive_a, r.primitive_b, r.dose, ch, r)
                           for r in results]

    # 逐对汇总：交互量在 6 个表型维度上的均值（线性，所以等价于「平均读数层面」的交互）
    by_pair: dict[tuple[str, str], dict[str, dict]] = {}
    for ch, rows in per_channel.items():
        for row in rows:
            key = (row["primitive_a"], row["primitive_b"])
            by_pair.setdefault(key, {})[ch] = row

    summaries: list[dict] = []
    for (a, b), by_ch in sorted(by_pair.items()):
        phenotype = {ch: v["interaction"] for ch, v in by_ch.items() if ch in PHENOTYPE_DIMS}
        if not phenotype:
            continue
        interactions = [v["interaction"] for v in by_ch.values()]
        combined_values = [v["combined"] for v in by_ch.values()]
        baseline_values = [v["baseline"] for v in by_ch.values()]
        kinds: dict[str, int] = {}
        for v in by_ch.values():
            kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
        worst = min(by_ch.items(), key=lambda kv: (kv[1]["interaction"], kv[0]))
        summaries.append({
            "primitive_a": a, "primitive_b": b,
            "label": f"{primitive_zh(a)} + {primitive_zh(b)}",
            "n_channels": len(by_ch),
            "combined_mean": round(mean(combined_values), 4),
            "baseline_mean": round(mean(baseline_values), 4),
            "delta_mean": round(mean(combined_values) - mean(baseline_values), 4),
            "expected_additive_mean": round(mean([v["expected_additive"] for v in by_ch.values()]), 4),
            "interaction_mean": round(mean(interactions), 4),
            "interaction_phenotype_mean": round(mean(list(phenotype.values())), 4),
            "most_interacting_channel": worst[0],
            "most_interacting_kind": worst[1]["kind"],
            "kinds": kinds,
            "channels": {ch: {"single_a": v["single_a"], "single_b": v["single_b"],
                              "combined": v["combined"], "expected_additive": v["expected_additive"],
                              "interaction": v["interaction"], "kind": v["kind"]}
                         for ch, v in by_ch.items()},
        })

    kind_by_channel: dict[str, dict[str, int]] = {}
    for ch, rows in per_channel.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        kind_by_channel[ch] = counts
    overall: dict[str, int] = {}
    for counts in kind_by_channel.values():
        for kind, n in counts.items():
            overall[kind] = overall.get(kind, 0) + n

    return {
        "dose": dose,
        "n_pairs": len(summaries),
        "n_interaction_rows": sum(len(v) for v in per_channel.values()),
        "kind_overall": overall,
        "kind_by_channel": kind_by_channel,
        "pairs": summaries,
        "by_channel": per_channel,
    }


def convergence_analysis(combined: Sequence[PhenotypeProfile], *, dose: float) -> dict:
    """机制异质性：每个表型终点上，「同表型不同机制」的候选集（含单因素与组合）。

    这是本项目核心主张的直接检验面 —— 高阶能力的损伤通常源于**多个基础过程的组合异常**，
    因此要问的是「把某个维度压到同一水平，有多少种彼此可区分的机制组合能做到」。
    """
    out: dict[str, dict] = {}
    for ch in PHENOTYPE_DIMS:
        try:
            result = hg.find_convergence(list(combined), ch, dose=dose, include_pairs=True)
        except (ValueError, KeyError, TypeError) as exc:
            out[ch] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if result is None:
            out[ch] = {"n_candidates": 0, "note": "没有候选机制把该维度压到阈值以下"}
            continue
        payload = result.to_dict()
        payload["n_pairs_among_candidates"] = sum(
            1 for c in payload.get("candidates") or [] if len(c.get("members") or []) > 1
        )
        out[ch] = payload
    return out


def aggregate(
    pair_records: Sequence[dict], single_records: Sequence[dict],
    baseline_records: Sequence[dict], *, dose: float,
) -> dict:
    """把三个来源的剖面合成：预平均 → 逐通道交互效应 → 逐对汇总 → 机制异质性。"""
    pair_profiles = s1.rehydrate(pair_records)
    single_profiles = s1.rehydrate(single_records)
    baseline_profiles = s1.rehydrate(baseline_records)

    avg_singles = average_groups(single_profiles, lambda p: (p.active_primitives()[0],))
    avg_pairs = average_groups(pair_profiles, lambda p: tuple(sorted(p.active_primitives())))
    combined = baseline_profiles + avg_singles + avg_pairs

    # 用公开入口（`specificity.baseline_profile` = 完整心智剖面取平均），不碰 heterogeneity 的私有函数
    baseline_avg = s1.sp.baseline_profile(combined)
    interactions = interaction_analysis(combined, dose=dose)
    convergence = convergence_analysis(combined, dose=dose)

    n_pairs_total = len(pair_profiles) / max(1, PAIR_REPLICATES)
    return {
        "dose": dose,
        "replicates": PAIR_REPLICATES,
        "n_pair_profiles": len(pair_profiles),
        "n_pair_cells_expected": round(n_pairs_total),
        "n_single_reference_profiles": len(single_profiles),
        "n_baseline_profiles": len(baseline_profiles),
        "n_profiles_synthetic": synthetic_count(list(combined)),
        "n_singles_after_average": len(avg_singles),
        "n_pairs_after_average": len(avg_pairs),
        "n_pairs_skipped": max(0, round(n_pairs_total) - interactions["n_pairs"]),
        "baseline_aggregate": (
            None if baseline_avg is None else {
                "note": baseline_avg.note,
                "dims": {ch: (None if baseline_avg.dims.get(ch) is None
                              else round(float(baseline_avg.dims[ch]), 4))
                         for ch in SCORED_CHANNELS},
            }
        ),
        "interactions": interactions,
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


KIND_ZH = {
    "superadditive": "超加（协同损伤）",
    "additive": "可加",
    "subadditive": "次加（拮抗）",
}


def render_analysis_md(
    result: dict, *, models: tuple[str, str, str], provider: str,
    single_source: str, checked: Sequence[str], limitations: Sequence[str],
) -> str:
    lines: list[str] = []
    inter = result["interactions"]
    lines.append("# 两两组合（465 pair）· 交互效应分析")
    lines.append("")
    lines.append(f"- provider：`{provider}`")
    lines.append(f"- 模型：脑 `{models[0]}` / 提问者 `{models[1]}` / 分析者 `{models[2]}`")
    lines.append(f"- 组合臂：**{result['n_pair_cells_expected']} 对 × {result['replicates']} 次重复**，"
                 f"两个 primitive 完整度均为 {result['dose']:g}")
    lines.append(f"- 可加性参照：`{single_source}` 的 {result['n_single_reference_profiles']} 份"
                 f"单因素剖面（同一剂量 {result['dose']:g}）")
    lines.append(f"- 基线：{result['n_baseline_profiles']} 份完整心智剖面")
    lines.append(f"- 交互量定义：`interaction = combined − (single_a + single_b − baseline)`；"
                 "**负 = 比可加更差 = 超加性损伤**")
    if result["n_profiles_synthetic"]:
        lines.append("")
        lines.append(f"> ⚠ 本批含 {result['n_profiles_synthetic']} 份 mock（规则式构造）剖面，"
                     "**不能作为科学结论的依据**。")
    lines.append("")
    lines.append("## 0. 参照与基线校验")
    lines.append("")
    lines += _table([[item] for item in checked], ["校验项"])
    lines.append("")
    base = result.get("baseline_aggregate")
    if base:
        lines += _table([[channel_label(ch), f"{base['dims'][ch]:.3f}"
                          if base["dims"][ch] is not None else "—"] for ch in SCORED_CHANNELS],
                        ["通道", "基线均值"])
    else:
        lines.append("⚠ 没有可用基线 → 交互效应无法判定。")
    lines.append("")
    lines.append("## 1. 覆盖与预平均")
    lines.append("")
    lines.append(f"- 组合剖面 {result['n_pair_profiles']} 份 → 取平均后 **{result['n_pairs_after_average']}** 条")
    lines.append(f"- 单因素参照 {result['n_single_reference_profiles']} 份 → 取平均后 "
                 f"**{result['n_singles_after_average']}** 条（应为 31 个 primitive）")
    lines.append(f"- 交互分析覆盖 **{inter['n_pairs']}** 对；"
                 f"被跳过 {result['n_pairs_skipped']} 对（缺同剂量单因素参照）")
    lines.append("")
    lines.append("> 为什么要先取平均：`evaluate_interactions` 内部对 singles/pairs 是**覆盖赋值**，"
                 "直接喂重复剖面只会静默采用最后一个重复，而不是均值。")
    lines.append("")

    lines.append("## 2. 交互类型分布（逐通道）")
    lines.append("")
    lines.append("每格 = 该通道上 465 对里各类交互的计数。")
    lines.append("")
    headers = ["通道"] + [KIND_ZH.get(k, k) for k in ("superadditive", "additive", "subadditive")]
    rows = []
    for ch in SCORED_CHANNELS:
        counts = inter["kind_by_channel"].get(ch) or {}
        rows.append([channel_label(ch)] + [counts.get(k, 0)
                                           for k in ("superadditive", "additive", "subadditive")])
    rows.append(["**合计**"] + [inter["kind_overall"].get(k, 0)
                                for k in ("superadditive", "additive", "subadditive")])
    lines += _table(rows, headers)
    lines.append("")

    pairs = inter["pairs"]
    lines.append("## 3. 最超加（协同损伤）的 15 对")
    lines.append("")
    lines.append("按 7 个计分通道的平均交互量升序 —— 越负 = 两个过程同时削弱造成的损伤越超过可加预期。")
    lines.append("")
    lines += _table(
        [[p["label"], f"{p['combined_mean']:.3f}", f"{p['expected_additive_mean']:.3f}",
          f"{p['interaction_mean']:+.3f}", f"{p['interaction_phenotype_mean']:+.3f}",
          channel_label(p["most_interacting_channel"])]
         for p in sorted(pairs, key=lambda p: p["interaction_mean"])[:15]],
        ["组合", "实测均值", "可加预期", "交互量", "表型维均值", "交互最强通道"],
    )
    lines.append("")
    lines.append("## 4. 最次加（拮抗）的 15 对")
    lines.append("")
    lines.append("交互量为正 = 两个过程同时削弱反而比可加预期**更轻**。")
    lines.append("")
    lines += _table(
        [[p["label"], f"{p['combined_mean']:.3f}", f"{p['expected_additive_mean']:.3f}",
          f"{p['interaction_mean']:+.3f}", f"{p['interaction_phenotype_mean']:+.3f}",
          channel_label(p["most_interacting_channel"])]
         for p in sorted(pairs, key=lambda p: -p["interaction_mean"])[:15]],
        ["组合", "实测均值", "可加预期", "交互量", "表型维均值", "交互最强通道"],
    )
    lines.append("")
    lines.append("## 5. 逐对汇总（全部组合）")
    lines.append("")
    lines.append("`实测均值`/`可加预期` 为 7 个通道读数的平均；`交互量` 为其差。"
                 "`协同/可加/拮抗` 是该对在 7 个通道上各自的判定计数。")
    lines.append("")
    lines += _table(
        [[p["label"], f"{p['delta_mean']:+.3f}", f"{p['interaction_mean']:+.3f}",
          p["kinds"].get("superadditive", 0), p["kinds"].get("additive", 0),
          p["kinds"].get("subadditive", 0)]
         for p in sorted(pairs, key=lambda p: p["interaction_mean"])],
        ["组合", "相对基线 Δ", "交互量", "协同", "可加", "拮抗"],
    )
    lines.append("")

    lines.append("## 6. 机制异质性：同表型、不同机制")
    lines.append("")
    lines.append("对每个表型维度问「有多少种彼此可区分的机制（含单因素与组合）能把它压到阈值以下」。")
    lines.append("")
    rows = []
    for ch, payload in result["convergence"].items():
        if "error" in payload:
            rows.append([channel_label(ch), "—", "—", f"计算失败：{payload['error'][:40]}"])
            continue
        rows.append([
            channel_label(ch), payload.get("n_candidates", 0),
            payload.get("n_pairs_among_candidates", 0),
            "、".join(channel_label(d) for d, _ in (payload.get("distinguishing_dims") or [])[:3]) or "—",
        ])
    lines += _table(rows, ["表型维度", "候选机制数", "其中组合机制", "可区分的维度（前 3）"])
    lines.append("")

    lines.append("## 7. 必须随本报告一并引用的局限")
    lines.append("")
    lines += [f"{i}. {item}" for i, item in enumerate(limitations, start=1)]
    lines.append("")
    lines.append(SIMULATION_NOTICE)
    lines.append(DISCLAIMER)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 清单与摘要
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path, *, conditions: Sequence[fig1.Condition], dose: float, models: tuple[str, str, str],
    settings: Settings, protocol: dict, cost: dict, probe_keys: Sequence[str],
    rubric_keys: Sequence[str], baseline_source: str, baseline_records: Sequence[dict],
    baseline_issues_found: Sequence[str], single_source: str, single_issues_found: Sequence[str],
    single_records: Sequence[dict], shard: tuple[int, int] | None,
) -> None:
    grid = []
    for condition in conditions:
        for theta in pair_thetas(condition, dose=dose):
            grid.append({
                "condition_id": condition.condition_id,
                "theta_id": theta.theta_id,
                "design": theta.design,
                "describe": theta.describe(),
                "members": list(theta.origin),
                "active": {k: round(float(v), 4) for k, v in theta.active_perturbations().items()},
                "values": {k: round(float(v), 4) for k, v in theta.values.items()},
                "replicates": condition.replicates,
                "n_turns": condition.n_turns, "max_turns": condition.max_turns,
            })
    fig1._write_json(path, {
        "tool": "scale_pair",
        "created_at": fig1.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pacs_version": __version__,
        "design": "pair",
        "axis": AXIS, "axis_name": AXIS_NAME,
        "pair_dose": dose,
        "replicates_per_pair": PAIR_REPLICATES,
        "n_pairs": len(conditions),
        "models": {"brain": models[0], "examiner": models[1], "analyst": models[2]},
        "provider": settings.llm.provider, "base_url": settings.llm.base_url,
        "protocol": dict(protocol),
        "age_months": DEFAULT_AGE_MONTHS,
        "cost": cost,
        "shard": None if shard is None else {"index": shard[0], "total": shard[1]},
        "condition_ids": [c.condition_id for c in conditions],
        "theta_grid": grid,
        "probe_keys": list(probe_keys), "rubric_keys": list(rubric_keys),
        "baseline": {
            "source": baseline_source, "n_records": len(baseline_records),
            "protocol_issues": list(baseline_issues_found),
            "reused": not baseline_issues_found and bool(baseline_records),
        },
        "single_reference": {
            "source": single_source, "dose": dose, "n_records": len(single_records),
            "protocol_issues": list(single_issues_found),
            "usable": not single_issues_found and bool(single_records),
        },
        "limitations": list(LIMITATIONS),
        "notice": SIMULATION_NOTICE, "disclaimer": DISCLAIMER,
    })


def write_summary_md(
    path: Path, *, records: Sequence[dict], failures: Sequence[dict], models: tuple[str, str, str],
    baseline_source: str, baseline_records: Sequence[dict], single_source: str,
    single_issues_found: Sequence[str], cost: dict, out_dir: Path,
    limitations: Sequence[str], incomplete: bool = False,
) -> None:
    lines = ["# 两两组合（465 pair）· 运行摘要", ""]
    lines.append(f"- 生成时间：{fig1.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 模型：脑 `{models[0]}` / 提问者 `{models[1]}` / 分析者 `{models[2]}`")
    lines.append(f"- 已完成被试：{len(records)} / 计划 {cost.get('n_cells')}")
    lines.append(f"- 组合剂量：{PAIR_DOSE:g}（两个 primitive 同时削弱）")
    lines.append(f"- 基线：{len(baseline_records)} 份来自 `{baseline_source}`")
    lines.append(f"- 可加性参照：`{single_source}`")
    lines.append(f"- 调用数预估：下界 {cost['calls_lower']}，上界 {cost['calls_upper']}")
    lines.append("")
    if incomplete:
        lines.append(f"> ⚠ **本批不完整**：只跑完 {len(records)}/{cost.get('n_cells')} 个单元格。"
                     "聚合结果只覆盖已跑完的部分。补齐：`--shard i/N --resume --yes` 后再 "
                     "`--aggregate-only`。")
        lines.append("")
    if single_issues_found:
        lines.append("## ⚠ 可加性参照校验未通过")
        lines.append("")
        lines += [f"- {item}" for item in single_issues_found]
        lines.append("")
        lines.append("→ 参照不可用时**不做**可加性比较（宁可不给结论，也不要一个建立在"
                     "不可比参照上的数）。")
        lines.append("")

    diag = fig1._diagnostics(records)
    lines.append("## 协议诊断")
    lines.append("")
    lines.append(f"- 实际轮数：均值 {diag.get('n_turns_run_mean')}，"
                 f"范围 {diag.get('n_turns_run_min')}–{diag.get('n_turns_run_max')}")
    lines.append(f"- 分析者解析失败率：{diag.get('parse_failure_rate')}")
    lines.append(f"- 读数来源：{diag.get('provider_kinds')}（`synthetic`={diag.get('synthetic')}）")
    lines.append(f"- 未覆盖通道计数：{diag.get('uncovered_channel_counts')}")
    lines.append(f"- 证据偏薄（不足 5 问）通道计数：{diag.get('under_quota_channel_counts')}")
    lines.append("")

    if failures:
        lines.append("## 失败单元格")
        lines.append("")
        lines += _table(
            [[f.get("condition_id", ""), f.get("replicate"), "、".join(f.get("theta_ids") or [])]
             for f in failures],
            ["条件", "重复", "θ"],
        )
        lines.append("")

    lines.append("## 产物")
    lines.append("")
    lines.append(f"- `{out_dir / 'manifest.json'}` —— 本批设置的参数（465 对网格 + 协议 + 成本 + 两个参照出处）")
    lines.append(f"- `{out_dir / 'cells'}/<pair>/subjects/*.json` —— 逐被试完整记录")
    lines.append(f"- `{out_dir / 'flat' / 'all_subjects.csv'}` / `readings_long.csv` / `数据字典.md`")
    lines.append(f"- `{out_dir / 'analysis' / 'scale_pair_analysis.md'}` —— 交互效应报告")
    lines.append(f"- `{out_dir / 'analysis' / 'scale_pair_analysis.json'}`")
    lines.append("")
    lines.append("## 局限")
    lines.append("")
    lines += [f"{i}. {item}" for i, item in enumerate(limitations, start=1)]
    lines.append("")
    lines.append(SIMULATION_NOTICE)
    lines.append(DISCLAIMER)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scale_pair",
        description="两两组合规模化数据产生：C(31,2)=465 对 × 3 重复，轻度 0.8，固定 50 轮",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/scale_pair.py --dry-run\n"
            "  python tools/scale_pair.py --provider mock --limit-pairs 5\n"
            "  python tools/scale_pair.py --shard 1/16 --resume --yes\n"
            "  python tools/scale_pair.py --aggregate-only\n"
        ),
    )
    parser.add_argument("--provider", choices=("mock", "openai"), default=None)
    parser.add_argument("--model", default=None, help="三方角色统一使用的模型名")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=PROTOCOL["max_tokens"],
                        dest="max_tokens", help=f"默认 {PROTOCOL['max_tokens']}（钉死为单维度批的取值）")
    parser.add_argument("--analyst-max-tokens", type=int, default=None, dest="analyst_max_tokens")

    parser.add_argument("--dose", type=float, default=PAIR_DOSE,
                        help=f"组合的受损程度，两个 primitive 同取该值（默认 {PAIR_DOSE:g}）")
    parser.add_argument("--replicates", type=int, default=PAIR_REPLICATES)
    parser.add_argument("--turns", type=int, default=PROTOCOL["n_turns"])
    parser.add_argument("--max-turns", type=int, default=PROTOCOL["max_turns"], dest="max_turns")
    parser.add_argument("--text-limit", type=int, default=PROTOCOL["text_limit"], dest="text_limit")
    parser.add_argument("--seed", type=int, default=PROTOCOL["seed"])
    parser.add_argument("--probe-pool", type=int, default=PROTOCOL["probe_pool"], dest="probe_pool")
    parser.add_argument("--probe-mode", choices=("auto", "scenario", "question"),
                        default=PROTOCOL["probe_mode"], dest="probe_mode")
    parser.add_argument("--rubric-limit", type=int, default=PROTOCOL["rubric_limit"],
                        dest="rubric_limit")

    parser.add_argument("--primitives", default=None,
                        help="只在这些 primitive 之间组合（逗号分隔；默认全部 31 个 → 465 对）")
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="只跑前 N 对（确定性顺序）—— 干跑与试点用，不影响正式网格")
    parser.add_argument("--conditions", default=None, help="只跑这些条件 id（逗号分隔）")

    parser.add_argument("--single-dir", default=None,
                        help=f"可加性参照目录（默认 {SINGLE_DIR_REL}）")
    parser.add_argument("--baseline-dir", default=None, help="完整心智基线目录")
    parser.add_argument("--refresh-baseline", action="store_true",
                        help="不复用外部基线，改为把完整心智加进本批网格重跑")
    parser.add_argument("--allow-reference-mismatch", action="store_true",
                        help="参照/基线协议校验不通过时仍继续（聚合会缺参照）")

    parser.add_argument("--shard", default=None, help="分片 `i/N`（1 起）")
    parser.add_argument("--out", default=None, help="产物根目录（默认 data/runs/scale_pair）")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shard = s1._parse_shard(args.shard)
    settings = s1._settings_from_args(args)
    protocol = {
        "n_turns": args.turns, "max_turns": args.max_turns, "text_limit": args.text_limit,
        "probe_mode": args.probe_mode, "probe_pool": args.probe_pool,
        "rubric_limit": args.rubric_limit, "seed": args.seed, "max_tokens": args.max_tokens,
        "temperature": (args.temperature if args.temperature is not None else PROTOCOL["temperature"]),
    }
    analyst_max_tokens = args.analyst_max_tokens or args.max_tokens

    model = settings.llm.model
    models = (model, model, model)
    model_id = f"openai:{model}" if settings.llm.provider == "openai" else f"mock:{model}"

    out_dir = Path(args.out) if args.out else (settings.data_dir / "runs" / "scale_pair")
    single_dir = (Path(args.single_dir) if args.single_dir else (settings.base_dir / SINGLE_DIR_REL))
    baseline_dir = (Path(args.baseline_dir) if args.baseline_dir
                    else (settings.base_dir / s1.BASELINE_DIR_REL))

    primitives = DEFAULT_ACTIVE
    if args.primitives:
        wanted = [p.strip() for p in args.primitives.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in DEFAULT_ACTIVE]
        if unknown:
            raise SystemExit(f"✗ 未知 primitive：{'、'.join(unknown)}")
        primitives = tuple(wanted)

    full_conditions = pair_conditions(primitives=primitives, dose=args.dose,
                                      replicates=args.replicates, models=models)
    if args.limit_pairs:
        full_conditions = full_conditions[: args.limit_pairs]
    selected = list(full_conditions)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
        known = {c.condition_id for c in selected}
        missing = wanted - known
        if missing:
            raise SystemExit(f"✗ 未知条件：{'、'.join(sorted(missing))}")
        selected = [c for c in selected if c.condition_id in wanted]
    run_conditions = s1.select_shard(selected, shard)
    cost = _cost(selected, args.dose)
    cost_shard = _cost(run_conditions, args.dose)

    probes = eb.dialogue_probes(goals=fig1.DIMS, n=protocol["probe_pool"],
                                seed=protocol["seed"], mode=protocol["probe_mode"])
    rubric = eb.observation_rubric(goals=fig1.DIMS, limit=protocol["rubric_limit"],
                                   seed=protocol["seed"])
    probe_keys = [i.get("key", "") for i in probes]
    rubric_keys = [i.get("key", "") for i in rubric]

    print("=" * 90)
    print("两两组合（465 pair）· 规模化数据产生")
    print("=" * 90)
    print(f"provider={settings.llm.provider}  model={model}  base_url={settings.llm.base_url}"
          f"  api_key={'已配置' if settings.llm.api_key else '缺失'}")
    print(f"组合：{len(primitives)} 个 primitive 两两组合 → {len(full_conditions)} 对；"
          f"每个 primitive 完整度 {args.dose:g}；每对 {args.replicates} 次重复")
    print(f"协议：{protocol['n_turns']} 轮基础 / 上限 {protocol['max_turns']} / "
          f"字数上限 {protocol['text_limit']} / seed {protocol['seed']} / "
          f"题集 {len(probes)} 条 / max_tokens {protocol['max_tokens']}")
    print(f"产物目录：{out_dir}")

    # ---- 基线（与单维度批同一道门）----
    baseline_records: list[dict] = []
    baseline_issues: list[str] = []
    if args.refresh_baseline:
        print("基线：`--refresh-baseline` —— 注意本脚本未生成基线臂，聚合会缺基线")
    elif settings.llm.provider == "mock":
        print("基线：mock 运行**不接**真实 provider 的基线")
    else:
        baseline_records = s1.load_baseline_records(baseline_dir)
        baseline_issues = s1.baseline_issues(baseline_dir, model_id=model_id,
                                             probe_keys=probe_keys, protocol=protocol)
        ids = s1.baseline_profile_model_ids(baseline_records)
        if ids and ids != {model_id}:
            baseline_issues.append(f"基线 model_id={sorted(ids)} ≠ 本批 {model_id!r}")
        print(f"基线复用：{len(baseline_records)} 份（{baseline_dir}）"
              + ("  ✓ 协议/模型/题池一致" if not baseline_issues else "  ✗ 校验未通过"))

    # ---- 可加性参照（本批的硬门）----
    single_records: list[dict] = []
    single_issues: list[str] = []
    if settings.llm.provider == "mock":
        print("可加性参照：mock 运行不与真实参照比较")
    else:
        single_issues = single_reference_issues(single_dir, model_id=model_id,
                                               probe_keys=probe_keys, protocol=protocol,
                                               dose=args.dose)
        single_records = load_single_reference(single_dir, dose=args.dose)
        if not single_records and not single_issues:
            single_issues.append(f"参照目录里没有 {args.dose:g} 档的单因素被试记录：{single_dir}")
        print(f"可加性参照：{len(single_records)} 份（{single_dir}）"
              + ("  ✓ 协议/模型/题池一致、剂量齐备" if not single_issues else "  ✗ 校验未通过"))

    if baseline_issues or single_issues:
        for item in list(baseline_issues) + list(single_issues):
            print(f"    - {item}")
        if not args.allow_reference_mismatch and not args.dry_run:
            print("  → 用 `--allow-reference-mismatch` 显式接受，或先补齐参照批次。")
            return 2
    print()

    if shard:
        print(f"分片 {shard[0]}/{shard[1]}：本片 {len(run_conditions)} 对")
    print(f"本片：条件 {cost_shard['n_conditions']} 个　单元格 {cost_shard['n_cells']} 个　"
          f"调用数 {cost_shard['calls_lower']}–{cost_shard['calls_upper']}")
    print(f"全批：条件 {cost['n_conditions']} 个　单元格 {cost['n_cells']} 个　"
          f"调用数 {cost['calls_lower']}（下界）– {cost['calls_upper']}（上界）")

    if args.dry_run:
        print()
        for condition in run_conditions[:10]:
            print(f"  {condition.condition_id:<48}{condition.replicates} 次重复　{condition.label}")
        if len(run_conditions) > 10:
            print(f"  …… 另有 {len(run_conditions) - 10} 对")
        print()
        print("（--dry-run：未调用任何模型）")
        print(SIMULATION_NOTICE)
        print(DISCLAIMER)
        return 0

    if args.aggregate_only:
        records: list[dict] = []
        for condition in selected:
            records.extend(fig1.load_condition_records(out_dir / "cells" / condition.condition_id))
        print(f"从磁盘读回 {len(records)} 份组合被试记录，开始聚合…")
        return _finish(out_dir, records, single_records, baseline_records, args=args,
                       shard=None, elapsed=None, selected=selected, models=models,
                       protocol=protocol, probe_keys=probe_keys, rubric_keys=rubric_keys,
                       single_dir=single_dir, baseline_dir=baseline_dir,
                       single_issues=single_issues, baseline_issues=baseline_issues)

    if not run_conditions:
        print("✗ 分片后没有要跑的条件")
        return 1

    if settings.llm.provider != "mock" and cost_shard["calls_upper"] > fig1.CONFIRM_THRESHOLD:
        if not args.yes:
            print(f"\n⚠ 真实 provider 且调用数上界 {cost_shard['calls_upper']} > "
                  f"{fig1.CONFIRM_THRESHOLD}：加 `--yes` 确认。")
            return 1
        if not settings.brain_ready():
            print("✗ 真实 provider 但缺 api_key。")
            return 1
        fig1.preflight_check(settings, list(dict.fromkeys(models)))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (out_dir / "manifest.json" if shard is None
                     else out_dir / "shards" / f"shard_{shard[0]}_of_{shard[1]}_manifest.json")
    write_manifest(manifest_path, conditions=selected, dose=args.dose, models=models,
                   settings=settings, protocol=protocol, cost=cost, probe_keys=probe_keys,
                   rubric_keys=rubric_keys, baseline_source=str(baseline_dir),
                   baseline_records=baseline_records, baseline_issues_found=baseline_issues,
                   single_source=str(single_dir), single_issues_found=single_issues,
                   single_records=single_records, shard=shard)

    all_records: list[dict] = []
    all_failures: list[dict] = []
    started = fig1.datetime.datetime.now()
    for position, condition in enumerate(run_conditions, start=1):
        print(f"\n▶ [{position}/{len(run_conditions)}] {condition.condition_id}　{condition.label}")
        try:
            records, failures = s1.run_condition(
                condition, settings=settings, thetas=pair_thetas(condition, dose=args.dose),
                out_dir=out_dir, model_id=model_id, probes=probes, rubric=rubric,
                text_limit=protocol["text_limit"], max_tokens=protocol["max_tokens"],
                analyst_max_tokens=analyst_max_tokens, reuse_existing=args.resume,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 本条件整体失败（{type(exc).__name__}: {str(exc)[:120]}），继续下一个")
            all_failures.append({"condition_id": condition.condition_id, "replicate": None,
                                 "theta_ids": [], "reasons": {"整体失败": str(exc)[:200]}})
            continue
        all_records.extend(records)
        for failure in failures:
            all_failures.append({**failure, "condition_id": condition.condition_id})

    elapsed = (fig1.datetime.datetime.now() - started).total_seconds() / 60.0
    print(f"\n本片完成：{len(all_records)} 个被试，用时 {elapsed:.1f} 分钟")
    return _finish(out_dir, all_records, single_records, baseline_records, args=args, shard=shard,
                   elapsed=elapsed, selected=selected, models=models, protocol=protocol,
                   probe_keys=probe_keys, rubric_keys=rubric_keys, single_dir=single_dir,
                   baseline_dir=baseline_dir, single_issues=single_issues,
                   baseline_issues=baseline_issues, shard_scope=run_conditions,
                   shard_failures=all_failures)


def _finish(
    out_dir: Path, records: Sequence[dict], single_records: Sequence[dict],
    baseline_records: Sequence[dict], *, args: argparse.Namespace, shard: tuple[int, int] | None,
    elapsed: float | None, selected: Sequence[fig1.Condition], models: tuple[str, str, str],
    protocol: dict, probe_keys: Sequence[str], rubric_keys: Sequence[str], single_dir: Path,
    baseline_dir: Path, single_issues: Sequence[str], baseline_issues: Sequence[str],
    shard_scope: Sequence[fig1.Condition] = (), shard_failures: Sequence[dict] = (),
) -> int:
    """收尾。`shard is None` 才写 flat/ 与 analysis/（分片绝不写，避免互相覆盖）。

    `selected` 是**全批**（进 manifest）；`shard_scope` 是**本片实际负责的**条件（进分片小结）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = s1._settings_from_args(args)
    cost = _cost(selected, args.dose)

    if shard is not None:
        scope = list(shard_scope) or list(selected)
        expected = sum(len(pair_thetas(c, dose=args.dose)) * c.replicates for c in scope)
        s1._write_shard_summary(out_dir, shard=shard, records=records, failures=shard_failures,
                                conditions=scope, elapsed=elapsed)
        print(f"\n分片小结：{out_dir / 'shards' / f'shard_{shard[0]}_of_{shard[1]}.md'}")
        print(f"本片完整性：{len(records)}/{expected}")
        print("（分片不写 flat/ 与 analysis/；全部跑完后 `--aggregate-only` 统一出）")
        print()
        print(SIMULATION_NOTICE)
        print(DISCLAIMER)
        return 3 if (shard_failures or len(records) < expected) else 0

    expected = cost["n_cells"]
    print(f"\n单元格完整性：{len(records)}/{expected}")
    incomplete = len(records) < expected
    if incomplete:
        print(f"  ⚠ 缺 {expected - len(records)} 个单元格：报告只覆盖已跑完的部分。")

    result = None
    usable = not single_issues and bool(single_records)
    if records and usable:
        print("\n交互分析…")
        result = aggregate(records, single_records, baseline_records, dose=args.dose)
        checked = ["外部基线：协议/模型/题池逐位一致"]
        checked.append("可加性参照（单维度批）：协议/模型/题池逐位一致、剂量齐备")
        limitations = limitations_for(result["n_pair_cells_expected"], result["n_pairs_skipped"])
        (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
        fig1._write_json(out_dir / "analysis" / "scale_pair_analysis.json", result)
        (out_dir / "analysis" / "scale_pair_analysis.md").write_text(
            render_analysis_md(result, models=models, provider=settings.llm.provider,
                               single_source=str(single_dir), checked=checked,
                               limitations=limitations),
            encoding="utf-8",
        )
        inter = result["interactions"]
        print("  交互类型总览：", inter["kind_overall"])
        print(f"  交互分析覆盖 {inter['n_pairs']} 对（跳过 {result['n_pairs_skipped']}）")
        print("  扁平表：", s1.write_flat_dataset(out_dir, records))
    elif records:
        print("\n⚠ 可加性参照不可用 → 不写交互报告（宁可不给结论，也不要错的结论）")
        limitations = list(LIMITATIONS)
    else:
        print("没有被试记录，跳过聚合。")
        limitations = list(LIMITATIONS)

    write_summary_md(out_dir / "summary.md", records=records, failures=list(shard_failures),
                     models=models, baseline_source=str(baseline_dir),
                     baseline_records=baseline_records, single_source=str(single_dir),
                     single_issues_found=single_issues, cost=cost, out_dir=out_dir,
                     limitations=limitations, incomplete=incomplete)
    write_manifest(out_dir / "manifest.json", conditions=selected, dose=args.dose, models=models,
                   settings=settings, protocol=protocol, cost=cost,
                   probe_keys=probe_keys, rubric_keys=rubric_keys,
                   baseline_source=str(baseline_dir), baseline_records=baseline_records,
                   baseline_issues_found=baseline_issues, single_source=str(single_dir),
                   single_issues_found=single_issues, single_records=single_records, shard=None)

    print()
    print("产物：")
    print(f"    {out_dir / 'manifest.json'}（本批设置的参数）")
    print(f"    {out_dir / 'cells'}/<pair>/subjects/*.json（逐被试完整记录）")
    print(f"    {out_dir / 'flat' / 'all_subjects.csv'}（宽表）")
    print(f"    {out_dir / 'analysis' / 'scale_pair_analysis.md'}（交互效应报告）")
    print(f"    {out_dir / 'summary.md'}（运行摘要）")
    print()
    print(SIMULATION_NOTICE)
    print(DISCLAIMER)
    return 3 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
