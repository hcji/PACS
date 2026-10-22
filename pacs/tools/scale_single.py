"""单维度受损 · 规模化数据产生（`design=single`）。

矩阵：**31 个 core primitive × 2 个程度（轻度 0.8 / 重度 0.4）× 3 次重复 = 186 个新被试**，
另加通用退化对照（FR-11 Control 1）3 个。轮次固定 50 轮（「7 通道 × 5 问」的配额走完后
若仍有通道不足，补问至上限 100）。

## 为什么单独一个脚本

Fig.1（`fig1_reliability.py`）问的是「同一批 θ 重复测，读数稳不稳」；本脚本问的是
「把某个认知过程削弱到某一档，表型上出现什么、剂量关系连不连续」。两者是两套条件的语义，
混进一个脚本会让数据集与报告互相污染（尤其是报告里的轴名与结论措辞）。

复用 `fig1_reliability.py` 里与实验语义无关的**编排 + 装盘**部分（`Condition` /
`build_record` / `_measurement_gap` / provider 工厂 / 扁平表列定义），因为这些已经踩过坑：
逐单元 `run_design` 调用、失败重试、单元格级续跑、缺口判据、utf-8-sig 导出。
本脚本**一行测量逻辑都不复制** —— 建心智、提问、读数、逐维表型全部走 agent 的公开入口
（`engine.run_design` → `agents.measure` → `agents.dialogue`）。

## 完整心智基线是复用来的

剂量-反应必须在 dose=1.0 处有锚点（`analysis/dose_response.py` 的 `single_perturbation_series`），
特异性判定也必须有基线（`analysis/specificity.py` 无基线直接返回空）。这批基线**不重跑**，
直接读 Fig.1 主运行里那 5 份完整心智剖面：

    data/runs/fig1_main/axis1/axis1_same_brain/       # θ = theta_f75b38eb1106，r0..r4

复用有**硬前提**，脚本在开跑前逐条校验（见 `baseline_issues`），任一条不满足就拒绝复用
并要求显式 `--refresh-baseline`（改为把完整心智加进本批网格重跑 5 次）：

1. 三方模型同名同端点；
2. `n_turns/max_turns/text_limit/probe_mode/seed/max_tokens/temperature` 逐项一致；
3. **题池逐位一致** —— `dialogue_probes(goals, n=100, seed=42, mode="auto")` 的 key 序列与
   基线 `condition.json` 里记的 `probe_keys` 完全相同。题池一变，基线就不是同一个协议下的
   测量了（这一条最容易在改题库后静默失效）。

## 协议常量被钉死为基线的取值

`max_tokens` 固定 **8000**（**不是** config.json 里的 2048）：Fig.1 主运行就是用 8000 跑的，
换成 2048 会让正文被截断成半截 JSON，而且基线立刻不可比。

## 用法

```bash
# 1) 看整张网格与成本上界（不调用任何模型，仍会校验基线协议）
python tools/scale_single.py --dry-run

# 2) mock 全矩阵干跑（秒级，验证管线；表型是构造的，不是证据）
python tools/scale_single.py --provider mock --out data/runs/scale_single_mock

# 3) 真实 provider 试点单个 primitive（6 个单元）
python tools/scale_single.py --primitives belief_updating --yes

# 4) 真实 provider 全量，6 路并行（每片一个进程）
python tools/scale_single.py --shard 1/6 --resume --yes
python tools/scale_single.py --shard 2/6 --resume --yes
...

# 5) 跑完后聚合（也可随时对已跑完的部分出报告）
python tools/scale_single.py --aggregate-only
```

产物（`--out`，默认 `data/runs/scale_single`）：

    manifest.json                 本批设置的参数：完整 θ 网格、协议、成本、基线出处与指纹
    cells/<condition_id>/
      condition.json              条件级参数 + probe_keys/rubric_keys + 失败记录
      subjects/NNN_theta_xxx_rK.json
                                  单个被试：parameters（θ）/ run_parameters /
                                  dialogue（逐轮全文）/ analysis（7 通道数值 + 结论）/
                                  profile / blind_audit
    flat/all_subjects.csv         一行一被试（宽表，含 7 维读数与结论）
    flat/readings_long.csv        一行一 被试×维度（长表）
    flat/数据字典.md
    analysis/scale_analysis.json  聚合结果（机器读）
    analysis/scale_analysis.md    聚合报告（人读：逐格均值 + 剂量-反应 + 特异性 + 机制剖面）
    summary.md                    全批一览、失败与协议问题

## 盲法

θ 只出现在 `parameters` 与 `blind_audit` 两个区块；`dialogue` 与 `analysis` 区块里没有 θ、
也没有被削弱的 primitive 名字（由 `fig1_reliability.build_record` 保证，本脚本不改动它）。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fig1_reliability as fig1  # noqa: E402  （编排/装盘复用；import 无副作用）
from pacs import (  # noqa: E402
    DEFAULT_AGE_MONTHS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TEXT_LIMIT,
    SIMULATION_NOTICE,
    __version__,
)
from pacs.analysis import dose_response as dr  # noqa: E402
from pacs.analysis import mechanism_profile as mp  # noqa: E402
from pacs.analysis import specificity as sp  # noqa: E402
from pacs.analysis.stats import mean, stdev  # noqa: E402
from pacs.brain import examiner_bank as eb  # noqa: E402
from pacs.brain.perturb import (  # noqa: E402
    Perturbation,
    generic_degradation_control,
    primitive_zh,
    single,
)
from pacs.config import PROJECT_ROOT, Settings  # noqa: E402
from pacs.engine import run_design  # noqa: E402
from pacs.llm.provider import LLMError, get_provider  # noqa: E402
from pacs.ontology.primitives import DEFAULT_ACTIVE  # noqa: E402
from pacs.phenotype import (  # noqa: E402
    CONTROL_DIM,
    PHENOTYPE_DIMS,
    SCORED_CHANNELS,
    PhenotypeProfile,
    channel_label,
    synthetic_count,
)

# ---------------------------------------------------------------------------
# 设计常量
# ---------------------------------------------------------------------------

#: 两个程度：轻度 0.8、重度 0.4（都在 `DEFAULT_DOSE_LEVELS` 里）
DEFAULT_DOSES: tuple[float, ...] = (0.8, 0.4)
#: 每个（primitive × 程度）单元的重复次数。3 次是**下限**：特异性判定的自校准集中度门槛
#: 要求 `len(series) >= 3` 才用重复估计标准误（见 `specificity.py`），低于 3 就只有固定门槛。
DEFAULT_REPLICATES = 3

#: 通用退化对照（FR-11 Control 1）。判据本身要求重复 ≥ 3 才自校准；
#: `evaluate_generic_degradation` 的 docstring 建议强度 ≥ 0.5，否则单次读数噪声（±0.08–0.13）
#: 与效应同量级、判不出来。
GENERIC_LEVEL = 0.5
GENERIC_REPLICATES = 3
GENERIC_CONDITION_ID = "generic_control_050"

#: 单元格重试之间的等待秒数。端点抖动往往持续几十秒，立刻重试常撞在同一段故障上；
#: 稍等一会儿再打，成功率明显更高，而代价可以忽略（一顿饭的时长换掉一次白跑）。
RETRY_BACKOFF_SECONDS = 8.0

#: 复用基线的条件 id（重跑时用 `--refresh-baseline`）
BASELINE_CONDITION_ID = "baseline_intact"
BASELINE_REPLICATES = 5
#: Fig.1 主运行里完整心智基线的落盘位置（相对项目根）
BASELINE_DIR_REL = Path("data") / "runs" / "fig1_main" / "axis1" / "axis1_same_brain"

AXIS = "scale"
AXIS_NAME = "单维度剂量放大"

#: 协议常量 —— **钉死为基线的取值**（见模块 docstring）。
#: 改任何一项都会使复用的完整心智基线不再是同一协议下的测量。
PROTOCOL: dict = {
    "n_turns": fig1.BASE_TURNS,              # 50
    "max_turns": fig1.BASE_MAX_TURNS,        # 100
    "text_limit": DEFAULT_TEXT_LIMIT,        # 100
    "probe_mode": "auto",
    "probe_pool": fig1.PROBE_POOL,           # 100
    "rubric_limit": fig1.RUBRIC_LIMIT,       # 8
    "seed": 42,
    # ⚠ 不是 config.json 的 2048：Fig.1 主运行用 8000，改回去既会截断输出又让基线不可比
    "max_tokens": 8000,
    "temperature": 0.3,
}

DISCLAIMER = "用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。"

#: 本批必须随数据一并记录的局限（写进报告，不写就是过度声称）
LIMITATIONS: tuple[str, ...] = (
    "3 次重复是特异性自校准门槛的下限（`specificity.py` 要求 ≥3 才用重复估计标准误）："
    "逐 primitive 判定的置信区间会宽，这是本次设计的既定强度，不是可忽略的细节。",
    "真实 provider 不接收 seed、temperature 固定 0.3：同一 θ 的重复之间测到的是服务端采样"
    "噪声 + 分析者给分噪声，不是「同一仪器同一样本」的复现。",
    "50 轮只比「7 通道 × 5 问 = 35」的配额多 15 轮余量：仍可能有通道补问（实际轮数 >50）"
    "或证据偏薄（`under_quota_channels` 非空），逐条件报 n_turns_run 分布。",
    # 第 4 条随基线的**实际来源**改写（见 `limitations_for`）
    "基线的来源与协议一致性见第 0 节。",
    "单因素设计**不识别交互**：组合效应需要 pair 设计，本批不给任何交互/协同结论。",
    "仿真产物（虚拟被试），非真实儿童数据；仅用于方法学研究，不用于诊断。",
)


def limitations_for(origin: str, baseline_n: int) -> list[str]:
    """把第 4 条局限改写成与基线**实际来源**相符的表述。

    静态写死「基线是复用数据」在两种情况下都会说错：本批自带基线臂时它不是复用数据；
    完全没有基线时，真正要提醒的是「判定为空不是没效应，而是没测到」。
    """
    if origin == "reused":
        fourth = (
            "**完整心智基线是复用数据**（Fig.1 主运行的 5 份重复）：来源路径、重复次数与"
            "协议校验结果见第 0 节。协议任一项不一致时基线作废、改为重跑"
            "（`--refresh-baseline`），不做静默降级。"
        )
    elif origin == "internal":
        fourth = (
            f"基线是**本批自带**的完整心智臂（{baseline_n} 份，`--refresh-baseline`）：与扰动"
            "被试同批同协议，但它与 Fig.1 主运行不是同一批测量，跨批比较时须注意。"
        )
    else:
        fourth = (
            "**本批没有可用基线**：剂量-反应缺 dose=1.0 锚点，特异性与机制剖面无法判定 —— "
            "这是「没测到」，不是「没效应」。"
        )
    return [fourth if i == 4 else item for i, item in enumerate(LIMITATIONS, start=1)]


# ---------------------------------------------------------------------------
# θ 网格与条件表
# ---------------------------------------------------------------------------


def build_conditions(
    *,
    primitives: Sequence[str] = DEFAULT_ACTIVE,
    doses: Sequence[float] = DEFAULT_DOSES,
    replicates: int = DEFAULT_REPLICATES,
    models: tuple[str, str, str],
    include_generic: bool = True,
    include_baseline: bool = False,
) -> list[fig1.Condition]:
    """展开条件表：**一个 primitive 一个条件**，条件内是「各程度 × 各重复」的单元格。

    为什么一 primitive 一条件（而不是一整个批次一个条件）：条件是**续跑与分片的单位**。
    31 个条件可以均匀分给 N 个分片进程，某片崩了只影响自己那条；一个巨型条件则要么全跑
    要么重来。条件 id 就是 `single_<primitive>`，可读可筛（`--conditions`）。
    """
    brain, examiner, analyst = models
    dose_text = " / ".join(f"{d:g}" for d in doses)
    out: list[fig1.Condition] = []

    if include_baseline:
        out.append(
            fig1.Condition(
                condition_id=BASELINE_CONDITION_ID,
                axis=AXIS,
                axis_name=AXIS_NAME,
                label=f"完整心智（无扰动）重复 {BASELINE_REPLICATES} 次 —— 剂量-反应的 dose=1.0 锚点",
                brain_model=brain, examiner_model=examiner, analyst_model=analyst,
                seed=PROTOCOL["seed"], n_turns=PROTOCOL["n_turns"],
                max_turns=PROTOCOL["max_turns"], replicates=BASELINE_REPLICATES,
                n_thetas=1, probe_mode=PROTOCOL["probe_mode"],
                reference=True,
                note="本批自带的基线臂（--refresh-baseline 时启用）",
            )
        )

    for pid in primitives:
        out.append(
            fig1.Condition(
                condition_id=f"single_{pid}",
                axis=AXIS,
                axis_name=AXIS_NAME,
                label=f"{primitive_zh(pid)}：完整度 {dose_text} 各 {replicates} 次重复",
                brain_model=brain, examiner_model=examiner, analyst_model=analyst,
                seed=PROTOCOL["seed"], n_turns=PROTOCOL["n_turns"],
                max_turns=PROTOCOL["max_turns"], replicates=replicates,
                n_thetas=len(doses), probe_mode=PROTOCOL["probe_mode"],
                note=f"单因素扰动：只削弱 {pid}，其余 {len(DEFAULT_ACTIVE) - 1} 个 primitive 保持完整",
            )
        )

    if include_generic:
        out.append(
            fig1.Condition(
                condition_id=GENERIC_CONDITION_ID,
                axis=AXIS,
                axis_name=AXIS_NAME,
                label=f"通用退化对照 {GENERIC_LEVEL:g}（非特异：所有通道应一起变差）",
                brain_model=brain, examiner_model=examiner, analyst_model=analyst,
                seed=PROTOCOL["seed"], n_turns=PROTOCOL["n_turns"],
                max_turns=PROTOCOL["max_turns"], replicates=GENERIC_REPLICATES,
                n_thetas=1, probe_mode=PROTOCOL["probe_mode"],
                note="FR-11 Control 1：用来区分「特异机制」与「什么都变差」",
            )
        )
    return out


def thetas_for(condition: fig1.Condition, *, doses: Sequence[float]) -> list[Perturbation]:
    """一个条件的设计表（θ 列表）。顺序即落盘文件名里的 `position`。"""
    if condition.condition_id == BASELINE_CONDITION_ID:
        return [fig1_intact()]
    if condition.condition_id == GENERIC_CONDITION_ID:
        return [generic_degradation_control(GENERIC_LEVEL, seed=PROTOCOL["seed"])]
    pid = condition.condition_id.removeprefix("single_")
    return [single(pid, float(dose), seed=PROTOCOL["seed"]) for dose in doses]


def fig1_intact() -> Perturbation:
    """完整心智（所有 primitive = 1.0）。θ 指纹必须与 Fig.1 基线一致，否则基线对不上。"""
    from pacs.brain.perturb import intact

    return intact(seed=PROTOCOL["seed"])


def select_shard(conditions: Sequence[fig1.Condition], shard: tuple[int, int] | None):
    """轮转分片：`(index, total)` → 该分片负责的条件。保证并集 == 全集且互不相交。"""
    if shard is None:
        return list(conditions)
    index, total = shard
    if total < 1:
        raise SystemExit("✗ 分片总数必须 ≥ 1")
    if not (1 <= index <= total):
        raise SystemExit(f"✗ 分片序号必须在 1..{total}，收到 {index}")
    return [c for i, c in enumerate(conditions) if i % total == index - 1]


# ---------------------------------------------------------------------------
# 完整心智基线的复用与校验
# ---------------------------------------------------------------------------


def load_baseline_records(base_dir: Path | None) -> list[dict]:
    """读回 Fig.1 主运行里的完整心智被试记录（只取真·完整心智：无扰动、无通用退化）。"""
    if base_dir is None or not (base_dir / "subjects").exists():
        return []
    out: list[dict] = []
    for path in sorted((base_dir / "subjects").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        params = payload.get("parameters") or {}
        if params.get("is_intact") and not params.get("generic_degradation"):
            payload["_source_path"] = str(path)
            out.append(payload)
    return out


def baseline_issues(
    base_dir: Path,
    *,
    model_id: str,
    probe_keys: Sequence[str],
    protocol: dict,
) -> list[str]:
    """逐条校验基线是否与本批同协议。返回问题清单，空 = 可以复用。

    校验的是**记录下来的协议**（`condition.json` 的 `run_parameters`）与**题池实际内容**，
    而不是「看起来差不多」。题池那一条尤其重要：改过题库之后 key 序列会变，而
    `run_parameters` 里的 `n_probe_items` 仍是 100，光看条数察觉不到。
    """
    issues: list[str] = []
    cond_path = base_dir / "condition.json"
    if not cond_path.exists():
        return [f"基线的 condition.json 不存在：{cond_path}"]
    try:
        condition = json.loads(cond_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"基线 condition.json 读不出来：{exc}"]

    params = condition.get("run_parameters") or {}
    for key, expected in (
        ("n_turns", protocol["n_turns"]),
        ("max_turns", protocol["max_turns"]),
        ("text_limit", protocol["text_limit"]),
        ("probe_mode", protocol["probe_mode"]),
        ("seed", protocol["seed"]),
        ("n_probe_items", protocol["probe_pool"]),
        ("n_rubric_items", protocol["rubric_limit"]),
        ("max_tokens", protocol["max_tokens"]),
    ):
        actual = params.get(key)
        if actual != expected:
            issues.append(f"协议不一致：{key} 基线={actual!r}，本批={expected!r}")
    base_temp = params.get("temperature")
    if base_temp is not None and abs(float(base_temp) - float(protocol["temperature"])) > 1e-9:
        issues.append(f"协议不一致：temperature 基线={base_temp!r}，本批={protocol['temperature']!r}")

    for key in ("brain_model_id", "examiner_model_id", "analyst_model_id"):
        actual = condition.get(key)
        if actual != model_id:
            issues.append(f"模型不一致：{key} 基线={actual!r}，本批={model_id!r}")

    base_keys = list(condition.get("probe_keys") or [])
    if base_keys != list(probe_keys):
        if len(base_keys) != len(probe_keys):
            detail = f"条数 {len(base_keys)} vs {len(probe_keys)}"
        else:
            diff = [i for i, (a, b) in enumerate(zip(base_keys, probe_keys)) if a != b]
            detail = f"首个不同的位置 #{diff[0] + 1}（{base_keys[diff[0]]} vs {probe_keys[diff[0]]}），共 {len(diff)} 处"
        issues.append(f"**题池不一致**（{detail}）—— 题库或选题参数变过，基线不再是同一协议的测量")

    return issues


def baseline_profile_model_ids(records: Sequence[dict]) -> set[str]:
    return {
        str((r.get("profile") or {}).get("model_id") or "")
        for r in records
    }


# ---------------------------------------------------------------------------
# 运行一个条件
# ---------------------------------------------------------------------------


def run_condition(
    condition: fig1.Condition,
    *,
    settings: Settings,
    thetas: Sequence[Perturbation],
    out_dir: Path,
    model_id: str,
    probes: Sequence[dict],
    rubric: Sequence[dict],
    text_limit: int,
    max_tokens: int,
    analyst_max_tokens: int,
    reuse_existing: bool = False,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """跑一个条件：每个 (θ, 重复) 单元格单独一次 `run_design`，失败单独重试。

    循环形状照抄 `fig1_reliability.run_condition`（那里已经踩过坑：内建 replicate 计数会在
    重试后错位并覆盖已落盘文件；一次调用放多个 design 时 `on_dialogue` 与 θ 无法可靠配对）。
    只改一处：run_id 前缀用 `scale_`，不沿用 `fig1_` —— 落盘里的 run_id 是溯源字段，
    写错前缀会让人以为这些被试属于 Fig.1 实验。
    """
    run_id = f"scale_{condition.condition_id}"
    cond_dir = out_dir / "cells" / condition.condition_id
    subjects_dir = cond_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    brain = fig1.provider_for(settings.llm, condition.brain_model)
    examiner = fig1.provider_for(settings.llm, condition.examiner_model)
    analyst = fig1.provider_for(
        fig1.replace(settings.llm, max_tokens=analyst_max_tokens), condition.analyst_model
    )

    run_parameters = {
        "n_turns": condition.n_turns, "max_turns": condition.max_turns,
        "text_limit": text_limit, "probe_mode": condition.probe_mode,
        "seed": condition.seed, "base_seed": PROTOCOL["seed"],
        "n_probe_items": len(probes), "n_rubric_items": len(rubric),
        "replicates": condition.replicates, "n_thetas": len(thetas),
        "age_months": DEFAULT_AGE_MONTHS, "cell_attempts": fig1.CELL_ATTEMPTS,
        "max_tokens": max_tokens, "analyst_max_tokens": analyst_max_tokens,
        "temperature": getattr(settings.llm, "temperature", None),
    }
    index_of = {theta.theta_id: position for position, theta in enumerate(thetas, start=1)}

    records: list[dict] = []
    failures: list[dict] = []
    done: set[tuple[str, int]] = set()
    conflicts: list[str] = []

    if reuse_existing:
        for path in sorted(subjects_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            previous = payload.get("condition") or {}
            params = payload.get("run_parameters") or {}
            theta_id = (payload.get("parameters") or {}).get("theta_id", "")
            replicate = previous.get("replicate")
            # 只复用「配置一致」的单元格：换过轮数/额度/θ 的旧产物混进来会让同一条件里
            # 出现两套协议，而下游按条件聚合时看不出来。
            if previous.get("condition_id") != condition.condition_id:
                continue
            if (params.get("n_turns"), params.get("max_turns"), params.get("max_tokens")) != (
                    condition.n_turns, condition.max_turns, max_tokens):
                conflicts.append(
                    f"{path.name}（轮数 {params.get('n_turns')}/{params.get('max_turns')}、"
                    f"额度 {params.get('max_tokens')}）")
                continue
            if theta_id not in index_of or replicate is None:
                conflicts.append(f"{path.name}（θ 或重复号无法识别）")
                continue
            gap = fig1._measurement_gap(payload)
            if gap:
                conflicts.append(f"{path.name}（{gap}，需重测）")
                continue
            records.append(payload)
            done.add((theta_id, int(replicate)))
        if records:
            print(f"    复用已落盘单元格 {len(records)} 个（--resume，按单元格而非按条件续跑）")
        for note in conflicts[:5]:
            print(f"    ⚠ 忽略无法复用的旧产物：{note}")
        if len(conflicts) > 5:
            print(f"    ⚠ ……另有 {len(conflicts) - 5} 个同类旧产物（协议不一致，全部重跑）")

    for replicate in range(condition.replicates):
        missing = [t for t in thetas if (t.theta_id, replicate) not in done]
        for attempt in range(1, fig1.CELL_ATTEMPTS + 1):
            if not missing:
                break
            still: list[Perturbation] = []
            for theta in missing:
                position = index_of[theta.theta_id]
                captured: list[tuple[object, dict]] = []
                profiles: list[object] = []
                # 会话 id 必须逐单元格唯一，且**不得出现 primitive 名字**：它会被写进
                # 落盘记录的 `dialogue` 区块，而那个区块是「分析者视角」的产物 —— 里面出现
                # 被削弱的认知过程名字，就等于把 ground truth 塞进了读数旁边（本脚本是一
                # primitive 一条件，condition_id 里就带着名字，直接用它会泄漏）。
                # 用不透明的 θ 指纹代替（与 `session.theta_id` 同一套指纹，不可反推）。
                cell_run_id = f"scale_{theta.theta_id}_r{replicate:02d}i{position:03d}a{attempt}"

                def on_dialogue(session, feedback) -> None:
                    captured.append((session, feedback))

                def on_profile(profile) -> None:
                    profiles.append(profile)

                try:
                    run = run_design(
                        brain, [theta],
                        run_id=cell_run_id, model_id=model_id, seed=PROTOCOL["seed"],
                        examiner=examiner, analyst=analyst,
                        probes=probes, probe_mode=condition.probe_mode,
                        dialogue_turns=condition.n_turns,
                        dialogue_max_turns=condition.max_turns,
                        text_limit=text_limit, rubric_items=rubric,
                        on_dialogue=on_dialogue, on_profile=on_profile,
                    )
                except Exception as exc:  # noqa: BLE001 —— 见下方注释，这里必须接住全部异常
                    # **不能只接 `LLMError`**。批量运行里两万次调用碰上几次网络层故障是常态，
                    # 而 `http.client.IncompleteRead`、连接重置、读取超时是从 `urllib` 直接
                    # 冒出来的，**不是** LLMError。只接 LLMError 的后果实测过：8 个分片里有 7 个
                    # 在同一个端点抖动窗口内被 IncompleteRead 打穿、整个进程退出 ——
                    # 一次瞬时故障毁掉整片几百次调用，而失败的那一格连重试机会都没有。
                    still.append(theta)
                    print(f"    ⚠ r{replicate} {theta.theta_id} 尝试 {attempt}/"
                          f"{fig1.CELL_ATTEMPTS} 抛错：{type(exc).__name__}: {str(exc)[:120]}")
                    time.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                if not captured or not profiles:
                    reason = run.failures[0][1] if run.failures else "未返回会话（原因未记录）"
                    still.append(theta)
                    # 立刻打出来：只在整个条件结束后汇总，会让一整轮的故障几小时后才被发现
                    print(f"    ⚠ r{replicate} {theta.theta_id} "
                          f"尝试 {attempt}/{fig1.CELL_ATTEMPTS} 失败：{reason}")
                    continue
                session, feedback = captured[0]
                record = fig1.build_record(
                    condition=condition, index=position,
                    session_id=fig1._plain(getattr(session, "session_id", cell_run_id)),
                    perturbation=theta, session=session, feedback=feedback,
                    profile=profiles[0], run_id=run_id,
                    run_parameters={**run_parameters, "replicate": replicate,
                                    "attempt": attempt, "cell_run_id": cell_run_id},
                )
                record["condition"]["replicate"] = replicate      # 重复编号由驱动决定
                gap = fig1._measurement_gap(record)
                if gap:
                    # 调用成功但产出残缺（截断/无读数/无结论）→ 不是一次可用的测量，重试
                    still.append(theta)
                    print(f"    ⚠ r{replicate} {theta.theta_id} 尝试 {attempt}/"
                          f"{fig1.CELL_ATTEMPTS} 判为本次测量失败：{gap}，重试")
                    continue
                fig1._write_json(
                    subjects_dir / f"{position:03d}_{theta.theta_id}_r{replicate}.json", record
                )
                records.append(record)
                if verbose:
                    info = record["dialogue"]["session"]
                    print(f"    [{position:03d}·r{replicate}] {theta.theta_id}　"
                          f"{info['n_turns_run']} 轮　"
                          f"未覆盖 {len(info['uncovered_channels'])} 通道　"
                          f"平均 {fig1._fmt(record['analysis']['mean_reading'])}　"
                          f"{fig1._likeness_phrase(record['analysis']['asd_likeness'])[:14]}"
                          + (f"　（第 {attempt} 次尝试成功）" if attempt > 1 else ""))
            missing = still
        if missing:
            failures.append({
                "replicate": replicate,
                "theta_ids": [t.theta_id for t in missing],
                "reasons": {t.theta_id: "重试耗尽" for t in missing},
            })
            print(f"    ✗ r{replicate} 仍有 {len(missing)} 个单元格未完成，已放弃本轮")

    expected = len(thetas) * condition.replicates
    fig1._write_json(cond_dir / "condition.json", {
        **condition.to_dict(),
        "run_id": run_id,
        "brain_model_id": f"{brain.name}:{brain.model}",
        "examiner_model_id": f"{examiner.name}:{examiner.model}",
        "analyst_model_id": f"{analyst.name}:{analyst.model}",
        "run_parameters": run_parameters,
        "n_records": len(records), "n_expected": expected,
        "failures": failures,
        "probe_keys": [i.get("key", "") for i in probes],
        "rubric_keys": [i.get("key", "") for i in rubric],
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    if len(records) < expected:
        print(f"    ⚠ 本条件只完成 {len(records)}/{expected} 个单元格")
    return records, failures


# ---------------------------------------------------------------------------
# 聚合：各维度数值 + 结论
# ---------------------------------------------------------------------------


def rehydrate(entries: Sequence[dict]) -> list[PhenotypeProfile]:
    """从落盘 JSON 重建剖面 —— 分片并行跑完后只能靠这一步聚合。

    接受两种形状：完整的被试记录（取它的 `profile` 区块）或裸的剖面载荷本身。
    两种都要收，是因为「基线」既可能以完整记录的形式传进来（读 Fig.1 的 subjects/*.json），
    也可能以剖面 dict 的形式传进来（测试与下游复用）。
    """
    out: list[PhenotypeProfile] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("profile")
        payload = nested if isinstance(nested, dict) else entry
        if not payload.get("theta"):
            continue
        try:
            out.append(PhenotypeProfile.from_dict(payload))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _cell_key(record: dict) -> tuple[str, float] | None:
    """该被试属于哪个（primitive, dose）单元格；通用退化/完整心智返回 None。"""
    params = record.get("parameters") or {}
    if float(params.get("generic_degradation") or 0.0) > 0:
        return None
    active = params.get("active") or {}
    if len(active) != 1:
        return None
    pid, dose = next(iter(active.items()))
    return str(pid), round(float(dose), 4)


def cell_table(
    records: Sequence[dict], baseline_dims: dict[str, float | None] | None
) -> list[dict]:
    """逐格（primitive × 程度）的通道均值 ± 重复间标准差，以及相对基线的 Δ。

    这是「各维度数值」的表：n 列给重复次数，读者能看出这一格有几个重复在支撑。
    """
    groups: dict[tuple[str, float], list[dict]] = {}
    for record in records:
        key = _cell_key(record)
        if key is None:
            continue
        groups.setdefault(key, []).append(record)

    rows: list[dict] = []
    for (pid, dose) in sorted(groups, key=lambda k: (k[0], -k[1])):
        bucket = groups[(pid, dose)]
        row = {
            "primitive_id": pid,
            "primitive_zh": primitive_zh(pid),
            "dose": dose,
            "n_replicates": len(bucket),
            "channels": {},
            "dominant_channel": "",
            "dominant_delta": None,
            "likeness_levels": {},
        }
        deltas: dict[str, float] = {}
        for ch in SCORED_CHANNELS:
            values = [
                float((r.get("analysis") or {}).get("readings", {}).get(ch))
                for r in bucket
                if (r.get("analysis") or {}).get("readings", {}).get(ch) is not None
            ]
            entry: dict = {"n": len(values)}
            if values:
                entry["mean"] = round(mean(values), 4)
                entry["sd"] = round(stdev(values), 4) if len(values) > 1 else None
                base = (baseline_dims or {}).get(ch)
                if base is not None:
                    entry["baseline"] = round(float(base), 4)
                    entry["delta"] = round(mean(values) - float(base), 4)
                    deltas[ch] = entry["delta"]
            row["channels"][ch] = entry
        phenotype_deltas = {ch: deltas[ch] for ch in PHENOTYPE_DIMS if ch in deltas}
        if phenotype_deltas:
            dominant = min(phenotype_deltas.items(), key=lambda kv: (kv[1], kv[0]))
            row["dominant_channel"] = dominant[0]
            row["dominant_delta"] = round(dominant[1], 4)
        for record in bucket:
            level = ((record.get("analysis") or {}).get("asd_likeness") or {}).get("level") or "unknown"
            row["likeness_levels"][level] = row["likeness_levels"].get(level, 0) + 1
        rows.append(row)
    return rows


def aggregate(
    records: Sequence[dict],
    baseline_records: Sequence[dict],
    *,
    doses: Sequence[float],
) -> dict:
    """把两个来源的剖面合成聚合结果：剂量-反应、特异性（每档一次）、机制剖面、通用退化对照。

    `model_id` 一律传 None：本批是单模型数据集，交给分析层按 model_id 过滤只会引入
    「基线 model_id 与本批不一致 → 静默返回空判定」的失败模式；model_id 的一致性
    在开跑前由 `baseline_issues` 显式校验。
    """
    profiles = rehydrate(records)
    baseline_profiles = rehydrate(baseline_records)
    combined = baseline_profiles + profiles

    fits = dr.build_dose_response(combined)
    specificities = {dose: sp.evaluate_specificity(combined, dose=dose) for dose in doses}
    generic = sp.evaluate_generic_degradation(combined)
    mprofile = mp.estimate_mechanism_profile(combined)

    base = sp.baseline_profile(combined)
    baseline_dims = (
        {ch: base.dims.get(ch) for ch in SCORED_CHANNELS} if base is not None else None
    )
    cells = cell_table(records, baseline_dims)
    # 实际参与基线平均的完整心智剖面数 —— 它可能来自外部复用，也可能来自本批自带的
    # 完整心智臂（`--refresh-baseline`），报告必须说清是哪一种。
    baseline_n = sum(
        1 for p in combined
        if p.generic_degradation <= 0 and p.theta and all(v >= 0.999 for v in p.theta.values())
    )

    return {
        "n_profiles_new": len(profiles),
        "n_profiles_baseline": len(baseline_profiles),
        "n_profiles_synthetic": synthetic_count(combined),
        "baseline_n": baseline_n,
        "baseline_origin": (
            "reused" if baseline_profiles else ("internal" if baseline_n else "none")
        ),
        "baseline_aggregate": (
            None if base is None else {
                "theta_id": base.theta_id, "note": base.note,
                "dims": {ch: (None if base.dims.get(ch) is None else round(float(base.dims[ch]), 4))
                         for ch in SCORED_CHANNELS},
            }
        ),
        "cells": cells,
        "dose_response": [f.to_dict() for f in fits],
        "dose_response_summary": dr.summarize(fits),
        "specificity": {
            f"{float(dose):g}": [v.to_dict() for v in specificities[dose]] for dose in doses
        },
        "specificity_summary": {
            f"{float(dose):g}": sp.summarize(specificities[dose]) for dose in doses
        },
        "dominant_distribution": {
            f"{float(dose):g}": sp.dominant_distribution(specificities[dose]) for dose in doses
        },
        "mechanism_profile": mprofile.to_dict(),
        "generic_degradation": None if generic is None else {
            "level": generic.level, "control_value": round(generic.control_value, 4),
            "mean_other_value": round(generic.mean_other_value, 4),
            "is_nonspecific": generic.is_nonspecific, "reason": generic.reason,
        },
    }


def _table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def render_analysis_md(
    result: dict,
    *,
    records: Sequence[dict],
    models: tuple[str, str, str],
    provider: str,
    baseline_source: str,
    baseline_checked: Sequence[str],
    doses: Sequence[float],
) -> str:
    """人读的聚合报告：逐格数值 → 剂量-反应 → 特异性 → 机制剖面 → 通用退化 → 逐 primitive 结论。"""
    lines: list[str] = []
    lines.append("# 单维度受损 · 规模化数据集分析")
    lines.append("")
    lines.append(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"- provider：`{provider}`")
    lines.append(f"- 模型：脑 `{models[0]}` / 提问者 `{models[1]}` / 分析者 `{models[2]}`")
    lines.append(f"- 本批网格被试：{result['n_profiles_new']}（不含外部基线）")
    origin = {
        "reused": f"复用外部完整心智基线（{result['n_profiles_baseline']} 份，来自 {baseline_source}）",
        "internal": f"本批自带的完整心智臂（{result['baseline_n']} 份，`--refresh-baseline`）",
        "none": "**无**（剂量-反应缺 dose=1.0 锚点；特异性与机制剖面无法判定）",
    }[result.get("baseline_origin", "none")]
    lines.append(f"- 基线：{origin}")
    lines.append(f"- 剂量档位：{' / '.join(f'{d:g}' for d in doses)}；每格重复：见下表 `n`")
    lines.append("- 读数方向：**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**；Δ 为相对完整心智基线的变化")
    if result["n_profiles_synthetic"]:
        lines.append("")
        lines.append(f"> ⚠ **本批含 {result['n_profiles_synthetic']} 份 mock（规则式构造）剖面** —— "
                     "mock 的表型结构是构造出来的，不是 LLM 涌现的，**不能作为科学结论的依据**。")
    lines.append("")
    lines.append("## 0. 基线来源与校验")
    lines.append("")
    if baseline_checked:
        lines += _table([[item] for item in baseline_checked], ["校验项"])
    else:
        lines.append("无检查项。")
    lines.append("")
    base = result.get("baseline_aggregate")
    if base:
        lines.append("基线聚合读数（"
                     f"{base['note']}）：")
        lines.append("")
        lines += _table(
            [[channel_label(ch), f"{base['dims'][ch]:.3f}" if base['dims'][ch] is not None else "—"]
             for ch in SCORED_CHANNELS],
            ["通道", "基线均值"],
        )
    else:
        lines.append("⚠ 没有可用的完整心智基线 → 剂量-反应与特异性都无法判定。")
    lines.append("")

    # ---- 逐格数值 ----
    lines.append("## 1. 逐格逐维数值（primitive × 程度 × 通道）")
    lines.append("")
    lines.append("每格 = 该（primitive, 程度）下所有重复的通道均值；括号里是重复间标准差（n≥2 时）。"
                 "Δ 相对基线。")
    lines.append("")
    headers = ["primitive", "程度", "n"] + [channel_label(ch) for ch in SCORED_CHANNELS]
    rows = []
    for cell in result["cells"]:
        row = [cell["primitive_zh"], f"{cell['dose']:g}", cell["n_replicates"]]
        for ch in SCORED_CHANNELS:
            entry = cell["channels"].get(ch) or {}
            if entry.get("mean") is None:
                row.append("—")
                continue
            text = f"{entry['mean']:.2f}"
            if entry.get("sd") is not None:
                text += f"({entry['sd']:.2f})"
            if entry.get("delta") is not None:
                text += f" Δ{entry['delta']:+.2f}"
            row.append(text)
        rows.append(row)
    lines += _table(rows, headers)
    lines.append("")

    # ---- 剂量-反应 ----
    lines.append("## 2. 剂量-反应（`analysis/dose_response.py`，逐通道拟合；主导通道由数据给出）")
    lines.append("")
    lines.append(f"判定分布：{result['dose_response_summary']}")
    lines.append("")
    lines += _table(
        [[f["primitive_id"], f["dominant_dim"] or "—", f"{f['slope']:+.3f}", f"{f['r2']:.2f}",
          f["n_points"], f["direction"], f["verdict"]] for f in result["dose_response"]],
        ["primitive", "主导通道", "slope", "r²", "点数", "方向", "判定"],
    )
    lines.append("")

    # ---- 特异性 ----
    for dose in doses:
        key = f"{float(dose):g}"
        lines.append(f"## 3.{key} 特异性 @ 程度 {key}（主导维度由数据给出，对照通道只做稳定性判据）")
        lines.append("")
        lines.append(f"判定分布：{result['specificity_summary'][key]}；"
                     f"主导维度分布：{result['dominant_distribution'][key]}")
        lines.append("")
        lines += _table(
            [[v["primitive_id"], v["dominant_dim"] or "—", f"{v['dominant_delta']:+.2f}",
              f"{v['concentration']:+.2f}", f"{v['control_delta']:+.2f}", v["n_replicates"],
              v["verdict"]] for v in result["specificity"][key]],
            ["primitive", "主导维度", "Δ主导", "集中度", "Δ对照", "n", "判定"],
        )
        lines.append("")

    # ---- 机制剖面 ----
    lines.append("## 4. 机制—维度剖面（数据驱动的测量产物，禁止回写为先验）")
    lines.append("")
    mprofile = result["mechanism_profile"]
    rows = []
    for row in mprofile.get("rows", []):
        deltas = row.get("deltas") or {}
        top = sorted(deltas.items(), key=lambda kv: kv[1])[:3]
        rows.append([
            row["primitive_id"], f"{row['dose']:g}",
            channel_label(row["dominant_channel"]) if row["dominant_channel"] else "—",
            ", ".join(f"{channel_label(ch)}{v:+.2f}" for ch, v in top) or "—",
            row["n_replicates"],
        ])
    lines += _table(rows, ["primitive", "程度", "主导通道", "下降最大的 3 个通道", "n"])
    for note in mprofile.get("notes") or []:
        lines.append("")
        lines.append(f"注：{note}")
    lines.append("")

    # ---- 通用退化对照 ----
    lines.append("## 5. 通用退化对照（FR-11 Control 1）")
    lines.append("")
    generic = result.get("generic_degradation")
    if generic:
        lines.append(f"- 水平：{generic['level']:.2f}　对照通道均值：{generic['control_value']:.3f}"
                     f"　其它维度均值：{generic['mean_other_value']:.3f}")
        lines.append(f"- 判定：**{'确认非特异整体下降' if generic['is_nonspecific'] else '尚不足以确认通用退化'}**")
        lines.append(f"- 理由：{generic['reason']}")
    else:
        lines.append("本批没有通用退化对照被试（`--no-generic`），此项无法判定。")
    lines.append("")

    # ---- 逐 primitive 结论 ----
    lines.append("## 6. 逐 primitive 结论摘要")
    lines.append("")
    lines.append("把上面几节的判定按 primitive 合成一条：剂量-反应是否连续 + 各档的特异性判定。"
                 "**这是聚合层结论**；每个被试的探查结论（约 350 字）在逐被试 JSON 的 "
                 "`analysis.conclusion` 与 `all_subjects.csv` 的 `conclusion` 列。")
    lines.append("")
    spec_by_dose = {key: {v["primitive_id"]: v for v in vals}
                    for key, vals in result["specificity"].items()}
    rows = []
    for fit in result["dose_response"]:
        pid = fit["primitive_id"]
        parts = [f"剂量-反应 {fit['verdict']}"]
        for dose in doses:
            verdict = spec_by_dose.get(f"{float(dose):g}", {}).get(pid)
            if verdict:
                parts.append(f"{float(dose):g} → {verdict['verdict']}")
        rows.append([pid, primitive_zh(pid), "；".join(parts)])
    lines += _table(rows, ["primitive", "中文名", "结论"])
    lines.append("")

    lines.append("## 7. 必须随本报告一并引用的局限")
    lines.append("")
    limitations = limitations_for(result.get("baseline_origin", "none"), result.get("baseline_n", 0))
    lines += [f"{i}. {item}" for i, item in enumerate(limitations, start=1)]
    lines.append("")
    lines.append(SIMULATION_NOTICE)
    lines.append(DISCLAIMER)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 扁平表与数据字典
# ---------------------------------------------------------------------------


def write_flat_dataset(out_dir: Path, records: Sequence[dict]) -> dict:
    """两张规整表 + 数据字典。

    列定义**复用 `fig1_reliability` 的**（`FLAT_SUBJECT_COLUMNS` / `FLAT_READING_COLUMNS` /
    `_flat_subject_row`），因此本批与 Fig.1 主运行的扁平表列完全一致、可直接合表比较。
    唯一自写的是**路径构造**：`fig1_reliability.write_flat_dataset` 用记录里的 `axis_dir`
    拼路径，而 `Condition.to_dict()` 并不写这个键，导致那边的 `subject_json_path` 少一层
    目录、指向不存在的文件（`data/runs/fig1_main/axis1_same_brain/subjects/...`）。
    """
    if not records:
        return {"n_subjects": 0}
    flat_dir = out_dir / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    reading_rows = []
    for index, record in enumerate(records, start=1):
        condition = record.get("condition") or {}
        stem = (f"{record.get('subject_index') or index:03d}_"
                f"{(record.get('parameters') or {}).get('theta_id', 'theta')}"
                f"_r{condition.get('replicate')}")
        json_path = (out_dir / "cells" / str(condition.get("condition_id", ""))
                     / "subjects" / f"{stem}.json")
        subject_rows.append(fig1._flat_subject_row(record, json_path))

        dimensions = (record.get("analysis") or {}).get("dimensions") or {}
        for dim in fig1.DIMS:
            item = dimensions.get(dim) or {}
            reading_rows.append({
                "run_id": record.get("run_id", ""),
                "condition_id": condition.get("condition_id", ""),
                "axis": condition.get("axis", ""),
                "theta_id": (record.get("parameters") or {}).get("theta_id", ""),
                "subject_index": record.get("subject_index"),
                "replicate": condition.get("replicate"),
                "session_id": record.get("session_id", ""),
                "dim": dim,
                "dim_label": SCORED_CHANNELS.get(dim, dim),
                "is_control": dim not in PHENOTYPE_DIMS,
                "score": item.get("score"),
                "typicality": item.get("typicality"),
                "deviation": item.get("deviation"),
                "confidence": item.get("confidence"),
                "observations": "；".join(item.get("observations") or []),
                "task_measurements": "；".join(
                    f"{m.get('task', '')} → {m.get('measurement', '')}"
                    for m in (item.get("task_measurements") or [])
                ),
            })

    with (flat_dir / "all_subjects.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fig1.FLAT_SUBJECT_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(subject_rows)
    with (flat_dir / "readings_long.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fig1.FLAT_READING_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reading_rows)

    (flat_dir / "数据字典.md").write_text(_data_dictionary(), encoding="utf-8")
    return {"n_subjects": len(subject_rows), "n_readings": len(reading_rows), "dir": str(flat_dir)}


def _data_dictionary() -> str:
    rows = [
        ("run_id", "运行号（本项目里一个条件一个 run_id）"),
        ("condition_id", "条件 id：`single_<primitive>` / `generic_control_050` / `baseline_intact`"),
        ("axis", f"比较轴：恒为 `{AXIS}`（单维度剂量放大）"),
        ("condition_label", "条件的中文描述（哪个 primitive、哪几档、几次重复）"),
        ("brain_model", "模拟大脑（被扰动的心智）模型名"),
        ("examiner_model", "提问者模型名"),
        ("analyst_model", "分析者模型名（读数由它判出）"),
        ("seed", f"题目集合与运行种子（本批固定 {PROTOCOL['seed']}）"),
        ("n_turns_set", f"**设定**的基础轮数（本批固定 {PROTOCOL['n_turns']}）"),
        ("max_turns_set", f"**设定**的轮数上限（本批固定 {PROTOCOL['max_turns']}）"),
        ("n_turns_run", "**实际**发生的轮数（配额满足即收工，故可能小于设定值；也可能补问到上限）"),
        ("replicates_set", "该条件计划的重复次数（本批每个程度 3 次）"),
        ("subject_index", "被试在该条件设计表里的序号（同一 θ 的重复共享）"),
        ("replicate", "同一 θ 的第几次重复（0 起）"),
        ("session_id", "会话号（与逐被试 JSON 的逐轮记录同键）"),
        ("theta_id", "θ 指纹（由取值派生；同 θ 同 id）"),
        ("theta_describe", "θ 的中文描述，如「[单因素] 信念更新 0.40」"),
        ("theta_active", "被削弱的认知过程及完整度；完整心智写「完整心智」"),
        ("generic_degradation", "通用退化水平（0 = 无）"),
        ("<7 个维度列>", "该维读数，**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**；"
                         "`general_cognition` 是对照通道（不进雷达图）"),
        ("mean_reading", "7 个通道读数的平均"),
        ("confidence", "分析者自评置信度（对证据偏薄的通道会主动降低）"),
        ("likeness_present / likeness_level", "表型印象：是否表现出孤独症样特征 + 程度档位"
                                               "（none/mild/moderate/marked）"),
        ("conclusion_chars / conclusion", "**分析者结论**（约 350 字病例报告）的字符数 / 截断到 400 字的正文"),
        ("uncovered_channels", "整段对话**一个问题都没问到**的通道（其读数没有证据支撑）"),
        ("under_quota_channels / n_under_quota", "被问到过但**问题数不足 5** 的通道（证据偏薄）"),
        ("n_examiner_turns / n_improvised_turns / improvised_turn_rate",
         "提问轮数 / 提问者没用上题库条目的轮数 / 两者之比"),
        ("over_limit_turns", "问题或回答超出字数上限的轮数（只告警、不截断）"),
        ("parse_error", "分析者输出解析失败的原因（空 = 正常）"),
        ("provider_kind / synthetic", "读数来源：`llm` 真实模型 / `mock` 规则式替身。"
                                      "`synthetic=True` 的剖面**不能当作 LLM 涌现证据**"),
        ("subject_json_path", "该被试的完整 JSON（含**逐轮对话全文**）的相对路径"),
    ]
    lines = [
        "# 数据字典（单维度受损 · 规模化数据集）", "",
        "## `all_subjects.csv`（宽表：一行 = 一个被试）", "",
        "| 列 | 含义 |", "| --- | --- |",
    ]
    lines += [f"| `{name}` | {desc} |" for name, desc in rows]
    lines += [
        "", "## `readings_long.csv`（长表：一行 = 一个 被试 × 维度）", "",
        "| 列 | 含义 |", "| --- | --- |",
        "| `dim` / `dim_label` | 维度 id / 中文名 |",
        "| `is_control` | 是否对照通道（`general_cognition`，不进雷达图） |",
        "| `typicality` | 该维读数（0..1，1 = 典型）—— **主字段** |",
        "| `score` | 同一读数的 0–100 表示 |",
        "| `deviation` | 相对基线的偏离量 |",
        "| `confidence` | 该维的分析者自评置信度 |",
        "| `observations` | 该维的观察证据（分析者给出的要点） |",
        "| `task_measurements` | 任务级测量（哪一轮、测到了什么） |",
        "", "## 关联与合表", "",
        "- 两张表都带 `session_id`，可与 `cells/<condition_id>/subjects/*.json`（含逐轮对话全文"
        "与逐题 `probe_key`）按 `session_id` 关联。",
        "- 跨实验合并：本批与 `fig1_main` 的扁平表**列完全一致**，按 `theta_id` + `dim` 即可对齐。",
        "", "## 方向与口径", "",
        "- 读数方向：**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**（与 θ 的完整度同向）。",
        "- θ 的方向：**1.0 = 该认知过程完整**；被削弱的 primitive 其完整度即「程度」档位。",
        "- 心智年龄固定 36 个月，**不是参数**。",
        "- 单因素设计**不识别交互**：同一被试只有一个 primitive 被削弱。",
        "- `synthetic=True` 的行来自 mock provider，**不可作为科学结论的依据**。",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 清单与汇总
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    *,
    conditions: Sequence[fig1.Condition],
    doses: Sequence[float],
    models: tuple[str, str, str],
    settings: Settings,
    protocol: dict,
    cost: dict,
    probe_keys: Sequence[str],
    rubric_keys: Sequence[str],
    baseline_source: str,
    baseline_records: Sequence[dict],
    baseline_issues_found: Sequence[str],
    shard: tuple[int, int] | None,
) -> None:
    """本批**设置的参数**全表：网格、协议、成本、基线出处与校验结果。

    这是「这批数据是怎么产生的」的唯一权威出处：离开它，落盘的 JSON 只是一堆读数。
    """
    grid = []
    for condition in conditions:
        for theta in thetas_for(condition, doses=doses):
            grid.append({
                "condition_id": condition.condition_id,
                "theta_id": theta.theta_id,
                "design": theta.design,
                "describe": theta.describe(),
                "active": {k: round(float(v), 4) for k, v in theta.active_perturbations().items()},
                "generic_degradation": round(float(theta.generic_degradation), 4),
                "values": {k: round(float(v), 4) for k, v in theta.values.items()},
                "replicates": condition.replicates,
                "n_turns": condition.n_turns,
                "max_turns": condition.max_turns,
            })
    payload = {
        "tool": "scale_single",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pacs_version": __version__,
        "model_version": DEFAULT_MODEL_VERSION,
        "design": "single",
        "axis": AXIS,
        "axis_name": AXIS_NAME,
        "doses": [float(d) for d in doses],
        "replicates_per_dose": DEFAULT_REPLICATES,
        "models": {"brain": models[0], "examiner": models[1], "analyst": models[2]},
        "provider": settings.llm.provider,
        "base_url": settings.llm.base_url,
        "protocol": dict(protocol),
        "age_months": DEFAULT_AGE_MONTHS,
        "cost": cost,
        "shard": None if shard is None else {"index": shard[0], "total": shard[1]},
        "n_conditions": len(conditions),
        "condition_ids": [c.condition_id for c in conditions],
        "theta_grid": grid,
        "probe_keys": list(probe_keys),
        "rubric_keys": list(rubric_keys),
        "baseline": {
            "source": baseline_source,
            "n_records": len(baseline_records),
            "theta_ids": sorted({(r.get("parameters") or {}).get("theta_id", "")
                                 for r in baseline_records}),
            "protocol_issues": list(baseline_issues_found),
            "reused": not baseline_issues_found and bool(baseline_records),
        },
        "limitations": list(LIMITATIONS),
        "notice": SIMULATION_NOTICE,
        "disclaimer": DISCLAIMER,
    }
    fig1._write_json(path, payload)


def write_summary_md(
    path: Path,
    *,
    records: Sequence[dict],
    failures: Sequence[dict],
    models: tuple[str, str, str],
    baseline_source: str,
    baseline_records: Sequence[dict],
    baseline_issues_found: Sequence[str],
    cost: dict,
    out_dir: Path,
    limitations: Sequence[str] = LIMITATIONS,
    incomplete: bool = False,
) -> None:
    """全批一览：完成情况、协议诊断、失败清单、产物索引。"""
    lines = ["# 单维度受损 · 规模化数据集（运行摘要）", ""]
    lines.append(f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 模型：脑 `{models[0]}` / 提问者 `{models[1]}` / 分析者 `{models[2]}`")
    lines.append(f"- 已完成被试：{len(records)} / 计划 {cost.get('n_cells')}")
    lines.append(f"- 基线：{len(baseline_records)} 份，来源 `{baseline_source}`")
    lines.append(f"- 调用数预估：下界 {cost['calls_lower']}，上界 {cost['calls_upper']}")
    lines.append("")
    if incomplete:
        lines.append(f"> ⚠ **本批不完整**：只跑完 {len(records)}/{cost.get('n_cells')} 个单元格。")
        lines.append("> 下面的聚合结果**只覆盖已跑完的部分**，不能当选整批读。"
                     "补齐：按分片重跑 `--shard i/N --resume --yes`，再 `--aggregate-only`。")
        lines.append("")

    if baseline_issues_found:
        lines.append("## ⚠ 基线协议校验未通过")
        lines.append("")
        lines += [f"- {item}" for item in baseline_issues_found]
        lines.append("")
        lines.append("→ 这一批的聚合报告**没有可用基线**，剂量-反应与特异性无法判定。"
                     "请用 `--refresh-baseline` 把完整心智加进网格重跑。")
        lines.append("")

    diag = fig1._diagnostics(records)
    lines.append("## 协议诊断")
    lines.append("")
    lines.append(f"- 实际轮数：均值 {diag.get('n_turns_run_mean')}，"
                 f"范围 {diag.get('n_turns_run_min')}–{diag.get('n_turns_run_max')}")
    lines.append(f"- 分析者解析失败率：{diag.get('parse_failure_rate')}")
    lines.append(f"- 分析者自评置信度均值：{diag.get('confidence_mean')}")
    lines.append(f"- 读数来源：{diag.get('provider_kinds')}"
                 f"（`synthetic`={diag.get('synthetic')}）")
    lines.append(f"- 未覆盖通道计数：{diag.get('uncovered_channel_counts')}")
    lines.append(f"- 证据偏薄（不足 5 问）通道计数：{diag.get('under_quota_channel_counts')}")
    if diag.get("improvised_turn_rate") is not None:
        lines.append(f"- 提问者自造题比例：{diag['improvised_turn_rate']}")
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
        lines.append("→ `--resume` 只补这些单元格，已完成的不会重跑。")
        lines.append("")

    lines.append("## 产物")
    lines.append("")
    lines.append(f"- `{out_dir / 'manifest.json'}` —— 本批设置的参数（完整 θ 网格 + 协议 + 成本 + 基线出处）")
    lines.append(f"- `{out_dir / 'cells'}/<condition_id>/subjects/*.json` —— 逐被试：参数 / 逐轮对话 / 7 维读数 / 结论")
    lines.append(f"- `{out_dir / 'flat' / 'all_subjects.csv'}` —— 宽表，一行一被试")
    lines.append(f"- `{out_dir / 'flat' / 'readings_long.csv'}` —— 长表，一行一 被试×维度")
    lines.append(f"- `{out_dir / 'flat' / '数据字典.md'}`")
    lines.append(f"- `{out_dir / 'analysis' / 'scale_analysis.md'}` —— 聚合报告（逐格数值 + 剂量-反应 + 特异性 + 机制剖面）")
    lines.append(f"- `{out_dir / 'analysis' / 'scale_analysis.json'}` —— 同一份结果的机器可读版")
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
        prog="scale_single",
        description="单维度受损规模化数据产生：31 primitive × 2 程度 × 3 重复，固定 50 轮",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/scale_single.py --dry-run\n"
            "  python tools/scale_single.py --provider mock\n"
            "  python tools/scale_single.py --primitives belief_updating --yes\n"
            "  python tools/scale_single.py --shard 1/6 --resume --yes\n"
            "  python tools/scale_single.py --aggregate-only\n"
        ),
    )
    parser.add_argument("--provider", choices=("mock", "openai"), default=None,
                        help="LLM 提供方（默认读 config.json；mock 为离线试跑）")
    parser.add_argument("--model", default=None,
                        help=f"三方角色统一使用的模型名（默认取 config.json，"
                             f"基线复用时必须是基线记录里的那个）")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=PROTOCOL["max_tokens"],
                        dest="max_tokens",
                        help=f"所有角色的单次输出额度上限（默认 {PROTOCOL['max_tokens']}，"
                             "**钉死为 Fig.1 基线的取值**，不是 config.json 的 2048）")
    parser.add_argument("--analyst-max-tokens", type=int, default=None,
                        dest="analyst_max_tokens",
                        help="分析者单独的额度上限（默认与 --max-tokens 相同，即与基线同协议）")

    parser.add_argument("--doses", default=",".join(f"{d:g}" for d in DEFAULT_DOSES),
                        help="程度档位，逗号分隔（默认 0.8,0.4；1.0 = 不削弱，会被剔除）")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                        help=f"每个（primitive × 程度）的重复次数（默认 {DEFAULT_REPLICATES}）")
    parser.add_argument("--turns", type=int, default=PROTOCOL["n_turns"],
                        help=f"基础轮数（默认 {PROTOCOL['n_turns']}）")
    parser.add_argument("--max-turns", type=int, default=PROTOCOL["max_turns"],
                        dest="max_turns",
                        help=f"补问上限（默认 {PROTOCOL['max_turns']}）")
    parser.add_argument("--text-limit", type=int, default=PROTOCOL["text_limit"],
                        dest="text_limit",
                        help=f"问题与回答的字数上限（默认 {PROTOCOL['text_limit']}；超出只告警）")
    parser.add_argument("--seed", type=int, default=PROTOCOL["seed"])
    parser.add_argument("--probe-pool", type=int, default=PROTOCOL["probe_pool"],
                        dest="probe_pool", help=f"题集大小（默认 {PROTOCOL['probe_pool']}）")
    parser.add_argument("--probe-mode", choices=("auto", "scenario", "question"),
                        default=PROTOCOL["probe_mode"], dest="probe_mode")
    parser.add_argument("--rubric-limit", type=int, default=PROTOCOL["rubric_limit"],
                        dest="rubric_limit")

    parser.add_argument("--primitives", default=None,
                        help="只跑这些 primitive（逗号分隔；默认全部 31 个）")
    parser.add_argument("--conditions", default=None, help="只跑这些条件 id（逗号分隔）")
    parser.add_argument("--no-generic", action="store_true",
                        help="不加通用退化对照（FR-11 Control 1）")
    parser.add_argument("--refresh-baseline", action="store_true",
                        help="不复用 Fig.1 的完整心智基线，改为把完整心智加进本批网格重跑")
    parser.add_argument("--allow-baseline-mismatch", action="store_true",
                        help="基线协议校验不通过时仍继续（只影响运行；聚合仍按实际可用基线算）")
    parser.add_argument("--baseline-dir", default=None,
                        help=f"基线目录（默认 {BASELINE_DIR_REL}）")

    parser.add_argument("--shard", default=None, help="分片 `i/N`（1 起）—— 多进程并行用")
    parser.add_argument("--out", default=None, help="产物根目录（默认 data/runs/scale_single）")
    parser.add_argument("--base-dir", default=None, help="项目根目录（默认自动定位）")
    parser.add_argument("--resume", action="store_true", help="跳过已完成的单元格")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="只从已落盘产物出聚合报告，不调用模型")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印网格、成本与基线校验结果，不调用模型")
    parser.add_argument("--yes", action="store_true", help="确认大规模真实调用")
    return parser.parse_args(argv)


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.load(base_dir=Path(args.base_dir) if args.base_dir else PROJECT_ROOT)
    if args.provider:
        settings.llm.provider = args.provider
    if args.model:
        settings.llm.model = args.model
    if args.base_url:
        settings.llm.base_url = args.base_url
    if args.api_key:
        settings.llm.api_key = args.api_key
    if args.temperature is not None:
        settings.llm.temperature = args.temperature
    if args.max_tokens is not None:
        if args.max_tokens < 256:
            raise SystemExit(f"✗ --max-tokens={args.max_tokens} 太小，正文会被截断成空")
        settings.llm.max_tokens = args.max_tokens
    settings.ensure_dirs()
    return settings


def _protocol_from_args(args: argparse.Namespace) -> dict:
    return {
        "n_turns": args.turns,
        "max_turns": args.max_turns,
        "text_limit": args.text_limit,
        "probe_mode": args.probe_mode,
        "probe_pool": args.probe_pool,
        "rubric_limit": args.rubric_limit,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature if args.temperature is not None else PROTOCOL["temperature"],
    }


def _parse_doses(text: str) -> tuple[float, ...]:
    out: list[float] = []
    for chunk in str(text).replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            dose = float(chunk)
        except ValueError:
            raise SystemExit(f"✗ 程度必须是数值，收到 {chunk!r}")
        if not (0.0 < dose < 1.0):
            if dose >= 0.999:
                print(f"  跳过 degree={dose:g}（= 不削弱，与完整心智重复）")
                continue
            raise SystemExit(f"✗ 程度必须落在 (0, 1)，收到 {dose}")
        out.append(dose)
    if not out:
        raise SystemExit("✗ 至少要有一个有效的程度档位")
    return tuple(dict.fromkeys(out))


def _parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    if "/" not in text:
        raise SystemExit("✗ --shard 形如 `1/6`")
    left, right = text.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        raise SystemExit(f"✗ --shard 形如 `1/6`，收到 {text!r}")


def _select_conditions(
    args: argparse.Namespace, conditions: Sequence[fig1.Condition]
) -> list[fig1.Condition]:
    selected = list(conditions)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
        known = {c.condition_id for c in selected}
        missing = wanted - known
        if missing:
            raise SystemExit(f"✗ 未知条件：{'、'.join(sorted(missing))}")
        selected = [c for c in selected if c.condition_id in wanted]
    if args.primitives:
        wanted = {p.strip() for p in args.primitives.split(",") if p.strip()}
        unknown = wanted - set(DEFAULT_ACTIVE)
        if unknown:
            raise SystemExit(f"✗ 未知 primitive：{'、'.join(sorted(unknown))}")
        keep = {f"single_{p}" for p in wanted} | {
            c.condition_id for c in selected
            if c.condition_id in (BASELINE_CONDITION_ID, GENERIC_CONDITION_ID)
        }
        selected = [c for c in selected if c.condition_id in keep]
    return selected


def _cost(conditions: Sequence[fig1.Condition], doses: Sequence[float]) -> dict:
    cells = sum(len(thetas_for(c, doses=doses)) * c.replicates for c in conditions)
    lower = sum(len(thetas_for(c, doses=doses)) * c.replicates * (c.n_turns * 2 + 1)
                for c in conditions)
    upper = sum(len(thetas_for(c, doses=doses)) * c.replicates * (c.max_turns * 2 + 1)
                for c in conditions)
    return {
        "n_conditions": len(conditions),
        "n_cells": cells,
        "calls_lower": lower,
        "calls_upper": upper,
        "formula": "单元格数 × (轮数 × 2 + 1)：每轮一次提问 + 一次作答，每被试一次分析者读数",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doses = _parse_doses(args.doses)
    shard = _parse_shard(args.shard)
    settings = _settings_from_args(args)
    protocol = _protocol_from_args(args)
    analyst_max_tokens = args.analyst_max_tokens or args.max_tokens

    # 三方角色一律走 settings.llm：本批要把三个角色钉在同一个模型上（与基线同协议），
    # 所以刻意不读 config.json 的 examiner_llm —— 那里配的另一个模型会让基线不可比。
    model = settings.llm.model
    models = (model, model, model)
    model_id = f"openai:{model}" if settings.llm.provider == "openai" else f"mock:{model}"

    out_dir = Path(args.out) if args.out else (settings.data_dir / "runs" / "scale_single")
    baseline_dir = (Path(args.baseline_dir) if args.baseline_dir
                    else (settings.base_dir / BASELINE_DIR_REL))
    baseline_source = str(baseline_dir)

    # 全批网格与「本进程要跑的网格」分开：全批进 manifest（设置过什么参数），
    # 本片进成本核算与执行（分片进程只跑自己那一片）。
    full_conditions = build_conditions(
        doses=doses, replicates=args.replicates, models=models,
        include_generic=not args.no_generic, include_baseline=args.refresh_baseline,
    )
    selected = _select_conditions(args, full_conditions)
    run_conditions = select_shard(selected, shard)
    cost_full = _cost(selected, doses)
    cost = _cost(run_conditions, doses)

    # 题池：一份，全条件复用 —— 与 Fig.1 主运行同参数，基线才可比。
    probes = eb.dialogue_probes(goals=fig1.DIMS, n=protocol["probe_pool"],
                                seed=protocol["seed"], mode=protocol["probe_mode"])
    rubric = eb.observation_rubric(goals=fig1.DIMS, limit=protocol["rubric_limit"],
                                   seed=protocol["seed"])
    probe_keys = [i.get("key", "") for i in probes]
    rubric_keys = [i.get("key", "") for i in rubric]

    print("=" * 90)
    print("单维度受损 · 规模化数据产生（design=single）")
    print("=" * 90)
    print(f"provider={settings.llm.provider}  model={model}  base_url={settings.llm.base_url}"
          f"  api_key={'已配置' if settings.llm.api_key else '缺失'}")
    print(f"程度档位={'/'.join(f'{d:g}' for d in doses)}  每个程度重复={args.replicates}")
    print(f"三方角色一律用同一个模型（与基线同协议）：{model}")
    print(f"协议：{protocol['n_turns']} 轮基础 / 上限 {protocol['max_turns']} / "
          f"字数上限 {protocol['text_limit']} / seed {protocol['seed']} / "
          f"题集 {len(probes)} 条 / 提纲 {len(rubric)} 条 / "
          f"max_tokens {protocol['max_tokens']}（分析者 {analyst_max_tokens}）")
    print(f"心智年龄固定 {DEFAULT_AGE_MONTHS} 个月（不是参数）")
    print(f"产物目录：{out_dir}")
    print()

    # ---- 基线校验（硬门）----
    baseline_records: list[dict] = []
    issues: list[str] = []
    if args.refresh_baseline:
        print("基线：`--refresh-baseline` → **不复用**，完整心智已加入本批网格"
              f"（重跑 {BASELINE_REPLICATES} 次）")
    elif settings.llm.provider == "mock":
        # mock 的读数由规则构造，真实 LLM 的基线是**另一种仪器**的读数。混在一起算 Δ，
        # 出来的数看着正常却毫无意义 —— 宁可没有基线，也不要一个假的。mock 想验证聚合
        # 全链路就用 `--refresh-baseline`，让它在 mock 内部自带一份同源基线。
        print("基线：**不接**真实 provider 的基线（本次是 mock 运行，读数由规则构造，"
              "与真实 LLM 基线不同源；要跑通聚合请加 `--refresh-baseline`）")
    else:
        baseline_records = load_baseline_records(baseline_dir)
        if not baseline_records:
            issues.append(f"基线目录下没有可用的完整心智被试记录：{baseline_dir}")
        else:
            issues = baseline_issues(baseline_dir, model_id=model_id,
                                     probe_keys=probe_keys, protocol=protocol)
            ids = baseline_profile_model_ids(baseline_records)
            if ids != {model_id}:
                issues.append(f"基线剖面的 model_id={sorted(ids)} 与本批 {model_id!r} 不一致"
                              "（会让分析层的基线聚合静默返回 None）")
        print(f"基线复用：{len(baseline_records)} 份来自 {baseline_dir}")
        if issues:
            print("✗ 基线协议校验未通过：")
            for item in issues:
                print(f"    - {item}")
            print("  → 题池/协议/模型任一不一致，基线就不是同一协议下的测量。")
            print("     请用 `--refresh-baseline` 把完整心智加进本批重跑，"
                  "或用 `--allow-baseline-mismatch` 显式接受（聚合会缺基线）。")
            if not args.allow_baseline_mismatch and not args.dry_run:
                return 2
        else:
            print("  ✓ 协议、三方模型、题池逐位一致 → 基线可复用")
    print()

    if shard:
        print(f"分片 {shard[0]}/{shard[1]}：本片 {len(run_conditions)} 个条件，"
              f"全批 {len(selected)} 个条件")
    print(f"本片：条件 {cost['n_conditions']} 个　单元格 {cost['n_cells']} 个　"
          f"调用数 {cost['calls_lower']}（下界）– {cost['calls_upper']}（上界）")
    print(f"全批：条件 {cost_full['n_conditions']} 个　单元格 {cost_full['n_cells']} 个　"
          f"调用数 {cost_full['calls_lower']}（下界）– {cost_full['calls_upper']}（上界）")
    if run_conditions:
        print()
        for condition in run_conditions:
            thetas = thetas_for(condition, doses=doses)
            print(f"  {condition.condition_id:<44}{condition.replicates} 重复 × "
                  f"{len(thetas)} θ = {condition.replicates * len(thetas):>2} 被试　"
                  f"{condition.label}")

    if args.dry_run:
        print()
        print("（--dry-run：未调用任何模型）")
        print("提示：真实 provider 建议先 `--primitives <一个> --yes` 跑单条件，"
              "确认读数、逐轮记录与解析都正常，再分片全量。")
        print()
        print(SIMULATION_NOTICE)
        print(DISCLAIMER)
        return 0

    # ---- 只看聚合（全部条件跑完后统一出报告）----
    if args.aggregate_only:
        records: list[dict] = []
        for condition in selected:
            records.extend(fig1.load_condition_records(out_dir / "cells" / condition.condition_id))
        if not baseline_records:
            baseline_records = load_baseline_records(baseline_dir)
        print(f"从磁盘读回 {len(records)} 份被试记录，开始聚合…")
        return _finish(out_dir, records, baseline_records, doses=doses, models=models,
                       baseline_source=baseline_source, failures=[], cost=cost_full,
                       baseline_issues_found=issues, publish=True, shard=None, elapsed=None,
                       selected_conditions=selected, shard_scope=[], settings=settings,
                       protocol=protocol, probe_keys=probe_keys, rubric_keys=rubric_keys)

    if not run_conditions:
        print("✗ 分片后没有要跑的条件")
        return 1

    # ---- 真实 provider 的确认门 ----
    if settings.llm.provider != "mock" and cost["calls_upper"] > fig1.CONFIRM_THRESHOLD:
        if not args.yes:
            print()
            print(f"⚠ 真实 provider 且调用数上界 {cost['calls_upper']} > "
                  f"{fig1.CONFIRM_THRESHOLD}：加 `--yes` 确认。")
            return 1
        if not settings.brain_ready():
            print("✗ 真实 provider 但缺 api_key：检查 config.json 或 PACS_LLM_API_KEY。")
            return 1
        fig1.preflight_check(settings, list(dict.fromkeys(models)))

    # ---- 先把「设置的参数」落盘：即使跑到一半崩了，这批是怎么配的也还在。
    # 分片进程写**自己那一片**的清单（`shards/shard_i_of_n_manifest.json`），不写批级的
    # `manifest.json`：6 个进程同时往同一个文件写，最后一个覆盖前面的，读到的 shard 字段
    # 是随机的。批级 manifest 只由 publish 路径（全量单进程或 --aggregate-only）写。
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (out_dir / "manifest.json" if shard is None
                     else out_dir / "shards" / f"shard_{shard[0]}_of_{shard[1]}_manifest.json")
    write_manifest(manifest_path, conditions=selected, doses=doses, models=models,
                   settings=settings, protocol=protocol, cost=cost_full, probe_keys=probe_keys,
                   rubric_keys=rubric_keys, baseline_source=baseline_source,
                   baseline_records=baseline_records, baseline_issues_found=issues, shard=shard)

    # ---- 跑 ----
    all_records: list[dict] = []
    all_failures: list[dict] = []
    started = datetime.datetime.now()
    for position, condition in enumerate(run_conditions, start=1):
        print(f"\n▶ [{position}/{len(run_conditions)}] {condition.condition_id}　{condition.label}")
        try:
            records, failures = run_condition(
                condition, settings=settings, thetas=thetas_for(condition, doses=doses),
                out_dir=out_dir, model_id=model_id, probes=probes, rubric=rubric,
                text_limit=protocol["text_limit"], max_tokens=protocol["max_tokens"],
                analyst_max_tokens=analyst_max_tokens, reuse_existing=args.resume,
            )
        except Exception as exc:  # noqa: BLE001
            # 一个条件整体崩掉不该带走整片剩下的条件：记下来继续跑，跑完用 `--resume` 补。
            print(f"  ✗ 本条件整体失败（{type(exc).__name__}: {str(exc)[:120]}），"
                  "跳过并继续下一个条件")
            all_failures.append({"condition_id": condition.condition_id, "replicate": None,
                                 "theta_ids": [], "reasons": {"整体失败": str(exc)[:200]}})
            continue
        all_records.extend(records)
        for failure in failures:
            all_failures.append({**failure, "condition_id": condition.condition_id})

    elapsed = (datetime.datetime.now() - started).total_seconds() / 60.0
    print(f"\n本片完成：{len(all_records)} 个被试，用时 {elapsed:.1f} 分钟")
    if all_failures:
        print(f"失败单元格：{sum(len(f['theta_ids']) for f in all_failures)} 个"
              "（跑完后 `--resume` 只补这些）")

    return _finish(out_dir, all_records, baseline_records, doses=doses, models=models,
                   baseline_source=baseline_source, failures=all_failures, cost=cost_full,
                   baseline_issues_found=issues, publish=shard is None, shard=shard,
                   elapsed=elapsed, selected_conditions=selected,
                   shard_scope=run_conditions, settings=settings,
                   protocol=protocol, probe_keys=probe_keys, rubric_keys=rubric_keys)


def _write_shard_summary(
    out_dir: Path,
    *,
    shard: tuple[int, int],
    records: Sequence[dict],
    failures: Sequence[dict],
    conditions: Sequence[fig1.Condition],
    elapsed: float | None,
) -> None:
    """分片只写自己这一片的小结 —— **绝不碰 flat/ 与 analysis/**。

    这是分片并行最容易踩的坑：6 个进程若各自把「本片记录」写成 `flat/all_subjects.csv`，
    最后落盘的批次数据就只有 1/6 的被试，而且没人看得出来（文件看起来完整、行数也像那么回事）。
    统一出报告的入口只有一个：全部跑完后的 `--aggregate-only`。
    """
    lines = [f"# 分片 {shard[0]}/{shard[1]} 运行小结", ""]
    lines.append(f"- 时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 本片条件：{len(conditions)} 个")
    lines.append(f"- 本片完成被试：{len(records)}")
    if elapsed is not None:
        lines.append(f"- 用时：{elapsed:.1f} 分钟")
    lines.append("")
    lines.append("## 本片条件")
    lines.append("")
    lines += _table(
        [[c.condition_id, c.replicates, c.n_thetas, c.label] for c in conditions],
        ["条件", "重复", "θ 数", "描述"],
    )
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
    lines.append("## 下一步")
    lines.append("")
    lines.append("- 全部跑完后执行：`python tools/scale_single.py --aggregate-only`")
    lines.append("  （它读全部条件、统一出 `flat/` 与 `analysis/`，分片进程不写这两处）")
    lines.append("")
    lines.append(SIMULATION_NOTICE)
    lines.append(DISCLAIMER)
    path = out_dir / "shards" / f"shard_{shard[0]}_of_{shard[1]}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finish(
    out_dir: Path,
    records: Sequence[dict],
    baseline_records: Sequence[dict],
    *,
    doses: Sequence[float],
    models: tuple[str, str, str],
    baseline_source: str,
    failures: Sequence[dict],
    cost: dict,
    baseline_issues_found: Sequence[str],
    publish: bool,
    shard: tuple[int, int] | None,
    elapsed: float | None,
    selected_conditions: Sequence[fig1.Condition],
    shard_scope: Sequence[fig1.Condition],
    settings: Settings,
    protocol: dict,
    probe_keys: Sequence[str],
    rubric_keys: Sequence[str],
) -> int:
    """收尾。`publish=True` 才写 `flat/` + `analysis/` + `summary.md` + `manifest.json`。

    分片进程一律 `publish=False`（见 `_write_shard_summary`）。
    `selected_conditions` 是**全批**（进 manifest）；`shard_scope` 是**本片实际负责的**条件
    （进分片小结）。两者不能混：混了分片小结会写「本片条件 32 个」而它其实只跑了 4 个。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not publish:
        expected = sum(len(thetas_for(c, doses=doses)) * c.replicates for c in shard_scope)
        _write_shard_summary(out_dir, shard=shard, records=records, failures=failures,
                             conditions=shard_scope, elapsed=elapsed)
        print(f"\n分片小结：{out_dir / 'shards' / f'shard_{shard[0]}_of_{shard[1]}.md'}")
        print(f"本片完整性：{len(records)}/{expected}")
        print("（分片不写 flat/ 与 analysis/，避免各片互相覆盖；全部跑完后 `--aggregate-only` 统一出）")
        print()
        print(SIMULATION_NOTICE)
        print(DISCLAIMER)
        # 有失败或没跑满就以非 0 退出：启动脚本靠退出码判断这片刻底跑完没有。
        return 3 if (failures or len(records) < expected) else 0

    expected_cells = cost.get("n_cells")
    print(f"\n单元格完整性：{len(records)}/{expected_cells}")
    incomplete = bool(expected_cells) and len(records) < expected_cells
    if incomplete:
        print(f"  ⚠ 缺 {expected_cells - len(records)} 个单元格：这份报告**只覆盖已跑完的部分**，"
              "不是全批。补齐后再跑一次 `--aggregate-only`。")

    result: dict | None = None
    if records:
        print("\n聚合…")
        result = aggregate(records, baseline_records, doses=doses)
        # 校验项随基线**实际来源**给：只有外部复用才谈得上「协议逐位一致」；
        # 本批自带臂是另一份测量，不能套用这句话。
        checked = {
            "reused": ["外部基线：协议 / 三方模型 / 题池逐位一致 → 可复用"],
            "internal": ["未做外部基线校验：本批自带完整心智臂（--refresh-baseline）"],
            "none": ["**无可用基线**"] + list(baseline_issues_found),
        }[result.get("baseline_origin", "none")]
        (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
        fig1._write_json(out_dir / "analysis" / "scale_analysis.json", result)
        (out_dir / "analysis" / "scale_analysis.md").write_text(
            render_analysis_md(result, records=records, models=models,
                               provider=settings.llm.provider,
                               baseline_source=baseline_source, baseline_checked=checked,
                               doses=doses),
            encoding="utf-8",
        )
        print("  基线聚合：", "有" if result["baseline_aggregate"] else "**无**（判定会缺基线）")
        print("  剂量-反应判定分布：", result["dose_response_summary"])
        for dose in doses:
            key = f"{float(dose):g}"
            print(f"  特异性 @{key} 判定分布：", result["specificity_summary"][key])
        if result["generic_degradation"]:
            print("  通用退化对照：",
                  "确认非特异" if result["generic_degradation"]["is_nonspecific"] else "未确认")
        print("  扁平表：", write_flat_dataset(out_dir, records))
    else:
        print("没有被试记录，跳过聚合。")

    write_summary_md(out_dir / "summary.md", records=records, failures=failures, models=models,
                     baseline_source=baseline_source, baseline_records=baseline_records,
                     baseline_issues_found=baseline_issues_found, cost=cost, out_dir=out_dir,
                     limitations=limitations_for(
                         (result or {}).get("baseline_origin", "none"),
                         (result or {}).get("baseline_n", 0)),
                     incomplete=incomplete)
    write_manifest(out_dir / "manifest.json", conditions=selected_conditions, doses=doses,
                   models=models, settings=settings, protocol=protocol, cost=cost,
                   probe_keys=probe_keys, rubric_keys=rubric_keys,
                   baseline_source=baseline_source, baseline_records=baseline_records,
                   baseline_issues_found=baseline_issues_found, shard=None)

    print()
    print("产物：")
    print(f"    {out_dir / 'manifest.json'}（本批设置的参数）")
    print(f"    {out_dir / 'cells'}/<condition_id>/subjects/*.json（逐被试完整记录）")
    print(f"    {out_dir / 'flat' / 'all_subjects.csv'}（宽表）")
    print(f"    {out_dir / 'flat' / 'readings_long.csv'}（长表）")
    print(f"    {out_dir / 'analysis' / 'scale_analysis.md'}（聚合报告：数值 + 结论）")
    print(f"    {out_dir / 'summary.md'}（运行摘要）")
    print()
    print(SIMULATION_NOTICE)
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
