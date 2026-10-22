"""Fig.1 系统可靠性实验：批量调用 PACS 心智，比较轴①～④，并算出可靠性指标。

四条比较轴（同一个「心智」被重复测量，看读数稳不稳）：

| 轴 | 变化的是什么 | 回答的问题 |
|---|---|---|
| ① 同一脑重复 | 什么都不变（只重跑） | 这套仪器本身的噪声有多大 |
| ② 不同 LLM | 模拟大脑换成另一个模型 | 换个「心智」实现，读数还可比吗 |
| ③ 不同 examiner / analyst / 题序 | 提问者、分析者、题目顺序 | 读数依赖谁在问、谁在判、按什么顺序问吗 |
| ④ 不同轮次 | 对话轮数（25 / 50 / 100） | 问多问少会改变结论吗 |

**硬约束（已遵守）**：本脚本与 `reliability.py` 都在 `tools/` 下，`src/pacs/` **零改动**、
`batch_probe.py` 也不动。三方模型分离通过给 `engine.run_design` 显式传 `examiner=` / `analyst=`
实现 —— 那是它公开的参数，属调用方权利，不是改核心过程代码。逐题 provenance 直接取自
`on_dialogue(session, feedback)` 回调里 `session.turns` 本来就带的 `probe_key` 等字段，
所以连「让智能体补保存输出」都不需要动包内代码。

**轮次协议**：轴①②③ 用 `n_turns=50, max_turns=100` —— 跑满 50 个基础轮后，只要 7 个计分通道
的配额（每通道 ≥5 问）都已满足就立刻收工，仍有缺口才补问，最多补到 100。
轴④ 用 `n_turns=L, max_turns=L` **硬固定**在 L 轮（配额检查只在最后一轮触发，无法延长）。

**题目集合**：每个 (seed, mode) 生成**一份 100 条的清单**，条件内所有被试复用；
轮数按位置天然截断（第 i 轮用 `probe_items[i-1]`）。于是 L=25 用的正是 L=50 的前 25 题，
形成嵌套结构，尽量把「轮数效应」与「题集效应」分开。

用法：

    # 1) 先看矩阵与成本（不调用模型）
    python tools/fig1_reliability.py --dry-run

    # 2) 离线跑通管线（Mock，秒级；mock 的可靠性必然≈1，只验证脚本）
    python tools/fig1_reliability.py --provider mock --mini

    # 3) 真实批量（BYOK；建议先跑单条件验证再放开）
    python tools/fig1_reliability.py --model-a deepseek-flash --model-b deepseek-v4-pro --yes
    python tools/fig1_reliability.py --conditions axis1_same_brain --yes

产物：

    data/runs/fig1/<轴>/<条件>/subjects/NNN_<θ>.json   逐被试（参数+对话全文+读数+剖面+条件元信息）
    data/runs/fig1/<轴>/<条件>/condition.json          该条件元信息 + 该条件的可靠性指标
    data/runs/fig1/reliability/fig1_reliability.json   全部指标
    data/runs/fig1/reliability/fig1_reliability.md     人读版（按轴分节）
    data/runs/fig1/reliability/fig1_panels.json        可直接喂绘图脚本的面板数据

设计说明、指标口径、统计实现踩过的坑、以及**必须一并声明的局限**见
`tools/可靠性实验.md`。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import reliability as rel  # noqa: E402
from pacs import (  # noqa: E402
    DEFAULT_AGE_MONTHS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TEXT_LIMIT,
    SIMULATION_NOTICE,
    __version__,
)
from pacs.agents.dialogue import (  # noqa: E402
    ABSOLUTE_MAX_TURNS,
    DEFAULT_N_TURNS,
    MAX_N_TURNS,
)
from pacs.brain import examiner_bank as eb  # noqa: E402
from pacs.brain.perturb import Perturbation, intact, single  # noqa: E402
from pacs.config import PROJECT_ROOT, Settings  # noqa: E402
from pacs.engine import run_design  # noqa: E402
from pacs.llm.provider import ChatMessage, LLMError, get_provider  # noqa: E402
from pacs.phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS  # noqa: E402
from pacs.phenotype.conclusion import ASD_LIKENESS_LEVEL_ZH  # noqa: E402

DIMS = tuple(SCORED_CHANNELS)

#: 与 `pacs run` 同口径的护栏
CONFIRM_THRESHOLD = 200
#: 分析者观察提纲条数（与 `cmd_run` 一致）
RUBRIC_LIMIT = 8
#: 基础轮数与补问上限（轴①②③ 的协议）
BASE_TURNS = 50
BASE_MAX_TURNS = 100
#: 轴④ 的轮次档位
TURN_LEVELS = (25, 50, 100)
#: 每个 (seed, mode) 生成的题集大小 —— 取最大轮数，使低轮次用的是高轮次题集的前缀
PROBE_POOL = 100
#: 默认的三组注入参数：完整心智基线 + 两个单因素扰动（真值跨度约 0.15，ICC 才有意义）
DEFAULT_THETA_SPEC = "intact,belief_updating:0.3,salience_assignment:0.3"

DISCLAIMER = "用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。"

#: 轴的 ASCII 键（目录名 / 面板键 / 比较表键统一用它，避免圈码与 ASCII 混用）。
AXIS_KEYS = {"①": "axis1", "②": "axis2", "③": "axis3", "④": "axis4"}
#: 轴的中文名（报告标题用）
AXIS_NAMES = {"①": "同一脑重复", "②": "不同 LLM（模拟大脑）",
              "③": "不同 examiner / analyst / 题序", "④": "不同轮次"}


# ---------------------------------------------------------------------------
# 序列化小工具（本脚本自己的一份；不跨文件 import 私有函数，避免耦合）
# ---------------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _round(value, digits: int = 4):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _plain(value, default: str = "") -> str:
    return default if value is None else str(value)


def _seq(value) -> list:
    return list(value or [])


def _one_line(text, limit: int = 120) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _likeness_phrase(likeness: dict) -> str:
    likeness = likeness or {}
    present = likeness.get("present")
    level_zh = ASD_LIKENESS_LEVEL_ZH.get(str(likeness.get("level") or ""), likeness.get("level") or "")
    if present is None:
        return f"未给出（{level_zh or '—'}）"
    return ("表现出孤独症样表型特征" if present else "未表现出孤独症样表型特征") + f" · 程度{level_zh}"


# ---------------------------------------------------------------------------
# 条件表
# ---------------------------------------------------------------------------


@dataclass
class Condition:
    """一个条件 = 一次 `run_design` 调用 = 一张 θ×重复 的设计表。"""

    condition_id: str
    axis: str
    axis_name: str
    label: str
    brain_model: str
    examiner_model: str
    analyst_model: str
    seed: int
    n_turns: int
    max_turns: int
    replicates: int
    n_thetas: int = 3
    probe_mode: str = "auto"
    reference: bool = False
    note: str = ""
    #: 本条件只用 θ 清单的**前 N 个**（None = 全部）。
    #: 用途：补跑一个新 θ 时，某些条件是**专门的对照臂**（例如轴② 是「换脑模型」的对照），
    #: 往里面加新 θ 会改变那条对照轴要回答的问题，也会平白多跑几格。
    theta_limit: int | None = None

    @property
    def n_subjects(self) -> int:
        return self.n_thetas * self.replicates

    @property
    def axis_dir(self) -> str:
        """轴目录名（短名，避免与条件 id 重复成 `axis1_same_brain/axis1_same_brain/`）。"""
        return AXIS_KEYS.get(self.axis, f"axis_{self.axis}")

    def describe(self) -> str:
        return (f"脑={self.brain_model} 提问者={self.examiner_model} 分析者={self.analyst_model} "
                f"seed={self.seed} 轮次={self.n_turns}/{self.max_turns} 重复={self.replicates}")

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id, "axis": self.axis, "axis_name": self.axis_name,
            "label": self.label, "brain_model": self.brain_model,
            "examiner_model": self.examiner_model, "analyst_model": self.analyst_model,
            "seed": self.seed, "n_turns": self.n_turns, "max_turns": self.max_turns,
            "replicates": self.replicates, "n_subjects": self.n_subjects,
            "probe_mode": self.probe_mode, "reference": self.reference, "note": self.note,
            "theta_limit": self.theta_limit,
        }


def build_conditions(
    model_a: str, model_b: str, replicates: int,
    levels: Sequence[int] = TURN_LEVELS, reference_turns: int = BASE_TURNS,
    mini: bool = False,
) -> list[Condition]:
    """按四条轴展开条件表。

    `model_a` / `model_b` 是两个可比较的模型（默认 flash / pro）。
    轴① 是**共享参照臂**：轴② 拿它当 A 模型那一半、轴③ 拿它当比较基准、轴④ 拿它当
    与 `reference_turns` 相同的那一档，所以它的重复次数最多（条件间的比较都靠它）。

    重复次数由 `replicates` 决定：参照臂与轴④ 用满 `replicates`，轴③ 的次要比对用
    `max(2, replicates - 2)`（重复太少的条件算不出像样的 ICC，但也不必和参照臂一样多）。
    `mini` 只用来把一切压到 2 次、便于秒级干跑。
    """
    rep_ref = 2 if mini else max(2, replicates)
    rep_small = 2 if mini else max(2, replicates - 2)

    def cond(cid, axis, axis_name, label, brain, examiner, analyst, seed,
             n_turns, max_turns, reps, reference=False, note="") -> Condition:
        return Condition(
            condition_id=cid, axis=axis, axis_name=axis_name, label=label,
            brain_model=brain, examiner_model=examiner, analyst_model=analyst,
            seed=seed, n_turns=n_turns, max_turns=max_turns, replicates=reps,
            reference=reference, note=note,
        )

    out: list[Condition] = []
    # ① 同一脑重复（= 轴②的 flash 臂 = 轴④的 L=50 档）
    out.append(cond(
        "axis1_same_brain", "①", "同一脑重复",
        f"同参数重跑 {rep_ref} 次（参考臂）", model_a, model_a, model_a, 42,
        BASE_TURNS, BASE_MAX_TURNS, rep_ref, reference=True,
        note="共享参照臂：轴②的 A 臂、轴③的比较基准、轴④的 L=50 档都由它充当",
    ))
    # ② 不同 LLM：只换模拟大脑
    out.append(cond(
        "axis2_brain_pro", "②", "不同 LLM（模拟大脑）",
        f"模拟大脑换 {model_b}（提问者/分析者不变）", model_b, model_a, model_a, 42,
        BASE_TURNS, BASE_MAX_TURNS, rep_ref,
        note="只换心智实现；提问者与分析者固定，避免把「换脑」与「换判官」混在一起",
    ))
    # ③ 不同 examiner / analyst / 题序
    out.append(cond(
        "axis3_examiner_pro", "③", "不同 examiner/analyst/题序",
        f"提问者换 {model_b}", model_a, model_b, model_a, 42,
        BASE_TURNS, BASE_MAX_TURNS, rep_small,
    ))
    out.append(cond(
        "axis3_analyst_pro", "③", "不同 examiner/analyst/题序",
        f"分析者换 {model_b}", model_a, model_a, model_b, 42,
        BASE_TURNS, BASE_MAX_TURNS, rep_small,
        note="这是分析侧的更换：读数由谁判出来",
    ))
    out.append(cond(
        "axis3_probe_seed7", "③", "不同 examiner/analyst/题序",
        "题目顺序改变（seed=7）", model_a, model_a, model_a, 7,
        BASE_TURNS, BASE_MAX_TURNS, rep_small,
        note="dialogue_probes 前 6 位对任何 seed 都相同，所以改变的只是第 7 位起的题序",
    ))
    # ④ 不同轮次：与**参照臂实际轮数**相同的档位由参照臂充当，不重复跑
    for level in levels:
        if level == reference_turns:
            continue
        out.append(cond(
            f"axis4_turns_{level}", "④", "不同轮次",
            f"硬固定 {level} 轮", model_a, model_a, model_a, 42,
            level, level, rep_ref,
            note=(f"{level} < 35（7 通道 × 5 问）时必有通道配额不足，"
                  f"其读数是分析者按提示词给的保守中性值" if level < 35 else ""),
        ))
    return out


def parse_theta_spec(spec: str) -> list[Perturbation]:
    """解析 θ 清单：`intact,belief_updating:0.3,salience_assignment:0.3`。"""
    out: list[Perturbation] = []
    for chunk in str(spec).split(","):
        item = chunk.strip()
        if not item:
            continue
        if item in ("intact", "完整", "基线"):
            out.append(intact(seed=42))
            continue
        if ":" not in item:
            raise SystemExit(f"✗ θ 写法应为 `primitive:剂量` 或 `intact`，收到 {item!r}")
        primitive, _, dose_text = item.partition(":")
        try:
            dose = float(dose_text)
        except ValueError:
            raise SystemExit(f"✗ 剂量不是数字：{item!r}")
        if not (0.0 < dose <= 1.0):
            raise SystemExit(f"✗ 剂量必须落在 (0, 1]（1.0 = 不削弱），收到 {dose}")
        try:
            out.append(single(primitive.strip(), dose, seed=42))
        except KeyError as exc:
            raise SystemExit(f"✗ {exc}")
    if not out:
        raise SystemExit("✗ θ 清单为空")
    return out


# ---------------------------------------------------------------------------
# 逐被试记录
# ---------------------------------------------------------------------------


def build_record(
    *,
    condition: Condition,
    index: int,
    session_id: str,
    perturbation: Perturbation,
    session,
    feedback: dict,
    profile,
    run_id: str,
    run_parameters: dict,
) -> dict:
    """一个被试的全部产物：条件 / 参数 / 对话全文 / 读数 / 剖面 / 盲法审计。

    `session.turns` **原样保存**（含 `probe_key`、`question`、`focus`、`chars`、`over_limit`）——
    逐题 provenance 是题目顺序轴与「提问者自造题比例」的唯一证据来源。
    `session.probe_keys` 是**计划**清单（100 条），逐轮的 `probe_key` 才是实际用了哪条。
    """
    readings = {dim: _round((feedback.get("dims") or {}).get(dim)) for dim in DIMS}
    details = feedback.get("dim_details") or {}
    dimensions = {}
    for dim in DIMS:
        detail = details.get(dim) or {}
        score = detail.get("score")
        if score is None and readings[dim] is not None:
            score = round(readings[dim] * 100, 1)
        dimensions[dim] = {
            "label": SCORED_CHANNELS.get(dim, dim),
            "control": dim not in PHENOTYPE_DIMS,
            "score": score,
            "typicality": readings[dim],
            "deviation": _round(detail.get("deviation"), 1),
            "confidence": _round(detail.get("confidence")),
            "observations": _seq(detail.get("observations")),
            "task_measurements": _seq(detail.get("task_measurements")),
        }
    turns = _seq(getattr(session, "turns", None))
    examiner_turns = [t for t in turns if t.get("role") == "examiner"]
    improvised = sum(1 for t in examiner_turns if not t.get("probe_key"))
    return {
        "run_id": run_id,
        "subject_index": index,
        "session_id": session_id,
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "condition": {
            **condition.to_dict(),
            "replicate": getattr(profile, "replicate", None),
        },

        "parameters": {
            "theta_id": perturbation.theta_id,
            "design": perturbation.design,
            "origin": list(perturbation.origin),
            "seed": perturbation.seed,
            "note": perturbation.note,
            "is_intact": perturbation.is_intact,
            "generic_degradation": _round(perturbation.generic_degradation),
            "active": {k: _round(v) for k, v in perturbation.active_perturbations().items()},
            "describe": perturbation.describe(),
        },
        "run_parameters": run_parameters,

        "dialogue": {
            "session": {
                "session_id": _plain(getattr(session, "session_id", "")),
                "theta_id": _plain(getattr(session, "theta_id", "")),
                "brain_model": _plain(getattr(session, "brain_model", "")),
                "examiner_model": _plain(getattr(session, "examiner_model", "")),
                "age_months": getattr(session, "age_months", DEFAULT_AGE_MONTHS),
                "n_turns": getattr(session, "n_turns", 0),
                "max_turns": getattr(session, "max_turns", 0),
                "n_turns_run": getattr(session, "n_turns_run", 0),
                "n_brain_turns": getattr(session, "n_brain_turns", 0),
                "seed": getattr(session, "seed", 0),
                "text_limit": getattr(session, "text_limit", 0),
                "probe_mode": _plain(getattr(session, "probe_mode", "")),
                "probe_keys": _seq(getattr(session, "probe_keys", None)),
                "question_counts": dict(getattr(session, "question_counts", None) or {}),
                "uncovered_channels": _seq(getattr(session, "uncovered_channels", None)),
                "under_quota_channels": _seq(getattr(session, "under_quota_channels", None)),
                "over_limit_turns": getattr(session, "over_limit_turns", 0),
                "ts": _plain(getattr(session, "ts", "")),
            },
            "turns": [_jsonable(t) for t in turns],
            "n_turns": len(turns),
            "n_examiner_turns": len(examiner_turns),
            "n_improvised_turns": improvised,
            "problems": _seq(getattr(session, "problems", None)),
        },

        "analysis": {
            "analyst_model": feedback.get("model_id", ""),
            "dimensions": dimensions,
            "readings": readings,
            "mean_reading": _round(feedback.get("mean_reading")),
            "confidence": _round(feedback.get("confidence")),
            "conclusion": feedback.get("conclusion") or "",
            "conclusion_chars": feedback.get("conclusion_chars"),
            "conclusion_target_chars": feedback.get("conclusion_target_chars"),
            "conclusion_notes": _seq(feedback.get("conclusion_notes")),
            "asd_likeness": feedback.get("asd_likeness") or {},
            "narrative": feedback.get("narrative") or "",
            "notes": feedback.get("notes") or "",
            "parse_error": feedback.get("error") or "",
        },

        "profile": profile.to_dict() if hasattr(profile, "to_dict") else _jsonable(profile),

        "blind_audit": {
            "note": "分析者看不到的 ground truth（被削弱的认知过程），仅供事后追溯，不参与读数计算。",
            "active_perturbations": {k: _round(v) for k, v in perturbation.active_perturbations().items()},
            "generic_degradation": _round(perturbation.generic_degradation),
            "describe": perturbation.describe(),
        },
    }


def load_condition_records(directory: Path) -> list[dict]:
    """从磁盘读回一个条件的全部被试记录（续跑与重新汇总都要用）。"""
    records: list[dict] = []
    subjects = directory / "subjects"
    if not subjects.exists():
        return records
    for path in sorted(subjects.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    records.sort(key=lambda r: int(r.get("subject_index") or 0))
    return records


# ---------------------------------------------------------------------------
# 单个条件的运行
# ---------------------------------------------------------------------------


def provider_for(base_llm, model: str):
    """按模型名造 provider（provider 实例与模型是一一绑定的）。"""
    return get_provider(replace(base_llm, model=model))


#: 分析者单独的输出额度上限（默认值，可用 `--analyst-max-tokens` 覆盖）。
#: 为什么要单独给：分析者的产出是一整段 JSON（7 段逐维记录 + 约 350 字结论 + 判定依据），
#: 是三方里最长的；推理型模型还会先花掉一大截额度在思维链上。核心包里的 `ANALYST_MIN_TOKENS`
#: 是 8000，但对 v4-pro 当分析者**不够** —— 实测在 8000 下 4/4 份输出全被 `max_tokens` 截断
#: （`finish_reason=length`），其中两份只救回 5/7 与 2/7 个维度的读数，且没有结论与档位判定。
#: 32768 已在两个模型上实测被端点接受（65536 也接受，留有余量即可）。
#: 这是**上限而非用量**：模型写多长还是多长，给足只是不让它被腰斩。
ANALYST_OUTPUT_TOKENS = 32768

#: 判定「完整测量」的四个条件，逐条都有实测来源（见 `_measurement_gap`）。
def _measurement_gap(record: dict) -> str:
    """这次测量**缺了什么**；返回空串表示是一份完整测量。

    分析者的产出一份 JSON（7 个维度 + 结论 + 档位判定）。四种残缺都会**静默落盘**，
    在下游只表现为「缺值」或「读数异常」，看不出是仪器故障：

    1. `parse_error` 非空 —— JSON 没解析完整（被 `max_tokens` 截断时最常见）。实测踩过：
       v4-pro 当分析者时 4/4 份全部被截断，其中两份只救回 5/7 与 2/7 个维度的读数。
    2. 一个维度读数都没有 —— 记录看起来完成了，其实一个数都没测到。
    3. 结论为空 —— 结论级一致度就无从计算，而它是本实验的主要指标之一。
    4. 档位判定缺失（`asd_likeness.present` 为 None）—— 二元/序数判定是结论级主指标。

    只按「有没有抛异常」判成败不够：上面四种情况**都没有抛异常**，调用本身是成功的。

    **刻意不要求「7 个维度全部有值」**：轮数少时（如轴④ 的 L=25，低于 7×5 的配额）本来就会
    有通道没被探查到，分析者对这些通道给 None 是**合法**结果，硬要求 7 维齐全会把有效测量
    判成失败、反而多丢格子。（截断造成的缺维另有 `parse_error` 兜住：实测被截断的 4 份全都带 `parse_error`，没有漏网。）
    """
    analysis = record.get("analysis") or {}
    if (analysis.get("parse_error") or "").strip():
        return f"分析者输出未完整解析（{(analysis.get('parse_error') or '')[:40]}）"
    readings = analysis.get("readings") or {}
    if not any(value is not None for value in readings.values()):
        return f"一个维度读数都没有（{len(readings)} 个维度全为 None）"
    if not (analysis.get("conclusion") or "").strip():
        return "结论为空"
    if (analysis.get("asd_likeness") or {}).get("present") is None:
        return "档位判定缺失（asd_likeness.present 为 None）"
    return ""


def _has_readings(record: dict) -> bool:
    """这份记录是否可用于可靠性统计（= 是否是一份完整测量）。"""
    return not _measurement_gap(record)


#: 单个 (θ, 重复) 单元格最多尝试几次。真实端点会出现**瞬时**故障（超时、5xx、空响应），
#: `run_design` 遇到 `LLMError` 只在 `failures` 里记一笔就跳过整条被试。若不管，矩阵就会不平衡，
#: 而下游 `matrix_from_records` 只保留测量次数达到最多的目标 —— **整组单元格会被静默丢掉**。
#: 实测踩过：一轮 10 个单元格坏了 3 个，故障集中在几分钟的端点抖动窗口内，重跑全部成功。
CELL_ATTEMPTS = 3


def run_condition(
    condition: Condition,
    *,
    settings: Settings,
    thetas: Sequence[Perturbation],
    out_dir: Path,
    text_limit: int,
    probe_pool: dict[int, list[dict]],
    verbose: bool = True,
    reuse_existing: bool = False,
    analyst_max_tokens: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """跑一个条件：同一批 θ 重复 `replicates` 次，产物落 `out_dir/<轴>/<条件>/`。

    按「轮」跑，一轮 = 所有 θ 各测一次（对应一个 `replicate` 编号）。**每个单元格单独一次
    `run_design` 调用**，失败就单独重试，最多 `CELL_ATTEMPTS` 次；仍失败才记进返回的失败列表。

    为什么不用 `run_design` 内建的 replicate 计数：它把「第几次出现」当作重复序号，一旦某次
    尝试失败后重跑，序号就会错位、覆盖已落盘的文件。这里由驱动自己按下标打 `replicate`，
    补跑既不污染重复编号，也保证「同一 (θ, replicate) 只落一个文件」。

    为什么一次调用只放一个 design：`run_design` 不返回会话对象，会话只能从 `on_dialogue`
    拿到，而 `on_dialogue` 不带 θ 标识。一批多个 design 时，要靠 `session.theta_id` 之类的
    属性把回调对回 θ —— 一旦这个属性缺失或改名，配对会**静默**错位。一次一个 design 则配对
    唯一，不依赖任何非契约属性。

    `reuse_existing=True` 时先扫 `subjects/`，把已落盘且配置一致的单元格读回来当作已完成，
    使 `--resume` 精确到单元格而不是条件。

    返回 `(records, failures)`；`failures` 形如 `[{replicate, theta_ids, reasons}]`。
    """
    run_id = f"fig1_{condition.condition_id}"
    cond_dir = out_dir / condition.axis_dir / condition.condition_id
    subjects_dir = cond_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    brain = provider_for(settings.llm, condition.brain_model)
    examiner = provider_for(settings.llm, condition.examiner_model)
    # 分析者的输出最长（7 段逐维记录 + 约 350 字结论 + 判定依据），推理型模型还要先花掉
    # 一大截思维链额度。共用额度会让它被截断成半截 JSON —— 实测 v4-pro 当分析者时 4/4 份
    # 全部截断。所以它单独一个额度，且**不影响脑与提问者**（那两方的协议保持全条件一致）。
    analyst_ceiling = int(analyst_max_tokens or getattr(settings.llm, "max_tokens", 0) or 0)
    analyst = provider_for(replace(settings.llm, max_tokens=analyst_ceiling),
                           condition.analyst_model)
    model_id = f"{brain.name}:{brain.model}"

    # 本条件实际使用的 θ：`theta_limit` 让对照臂只用前 N 个 θ（见 Condition.theta_limit）
    active = list(thetas)
    if condition.theta_limit is not None:
        if condition.theta_limit < 2:
            raise SystemExit(f"✗ {condition.condition_id} 的 theta_limit="
                             f"{condition.theta_limit} 太小，ICC 无从谈起")
        active = active[:condition.theta_limit]

    probes = probe_pool[condition.seed]
    rubric = eb.observation_rubric(goals=DIMS, limit=RUBRIC_LIMIT, seed=42)

    run_parameters = {
        "n_turns": condition.n_turns, "max_turns": condition.max_turns,
        "text_limit": text_limit, "probe_mode": condition.probe_mode,
        "seed": condition.seed, "base_seed": 42,
        "n_probe_items": len(probes), "n_rubric_items": len(rubric),
        "replicates": condition.replicates, "n_thetas": len(active),
        "theta_limit": condition.theta_limit,
        "age_months": DEFAULT_AGE_MONTHS, "cell_attempts": CELL_ATTEMPTS,
        # 输出额度直接决定「正文会不会被思维链挤成空」，是协议的一部分，必须逐被试留痕
        "max_tokens": int(getattr(settings.llm, "max_tokens", 0) or 0),
        "analyst_max_tokens": analyst_ceiling,
        "temperature": getattr(settings.llm, "temperature", None),
    }
    index_of = {theta.theta_id: position for position, theta in enumerate(active, start=1)}

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
            # 只复用「配置一致」的产物：换过轮数/θ 的旧文件混进来会让同一条件里出现两套协议
            if previous.get("condition_id") != condition.condition_id:
                continue
            if (params.get("n_turns"), params.get("max_turns"), params.get("max_tokens")) != (
                    condition.n_turns, condition.max_turns, run_parameters["max_tokens"]):
                conflicts.append(
                    f"{path.name}（轮数 {params.get('n_turns')}/{params.get('max_turns')}、"
                    f"额度 {params.get('max_tokens')}）")
                continue
            if theta_id not in index_of or replicate is None:
                conflicts.append(f"{path.name}（θ 或重复号无法识别）")
                continue
            gap = _measurement_gap(payload)
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
        missing = [t for t in active if (t.theta_id, replicate) not in done]
        for attempt in range(1, CELL_ATTEMPTS + 1):
            if not missing:
                break
            still: list[Perturbation] = []
            for theta in missing:
                position = index_of[theta.theta_id]
                captured: list[tuple[object, dict]] = []
                profiles: list[object] = []
                # 会话 id 必须逐单元格唯一：run_design 用 `{run_id}-{index:03d}`，而这里每次
                # 调用都只有 1 个 design（index 恒为 001），唯一性只能由 run_id 携带
                cell_run_id = f"{run_id}_r{replicate:02d}_i{position:03d}a{attempt}"

                def on_dialogue(session, feedback) -> None:
                    captured.append((session, feedback))

                def on_profile(profile) -> None:
                    profiles.append(profile)

                try:
                    run = run_design(
                        brain, [theta],
                        run_id=cell_run_id, model_id=model_id, seed=42,
                        examiner=examiner, analyst=analyst,
                        probes=probes, probe_mode=condition.probe_mode,
                        dialogue_turns=condition.n_turns, dialogue_max_turns=condition.max_turns,
                        text_limit=text_limit, rubric_items=rubric,
                        on_dialogue=on_dialogue, on_profile=on_profile,
                    )
                except Exception as exc:  # noqa: BLE001 - 任何异常都只该毁掉这一格，不该带走整批
                    # provider 只把「网络错误」归到 URLError/TimeoutError/OSError 三类里，而
                    # HTTP 响应被中途截断（`http.client.IncompleteRead`）**不属于这三类** ——
                    # 它会一路穿透到 main，把跑了几小时的整批带崩。实测踩过：并行 5 个进程
                    # 在同一分钟被同一个截断问题全部杀死，各丢一格。所以这里用宽口径兜住：
                    # 传输层的任何异常都降级成「本次尝试失败」，由单元格重试处理。
                    still.append(theta)
                    print(f"    ⚠ r{replicate} {theta.theta_id} "
                          f"尝试 {attempt}/{CELL_ATTEMPTS} 失败：{type(exc).__name__}: "
                          f"{str(exc)[:120]}，重试")
                    continue
                if not captured or not profiles:
                    reason = run.failures[0][1] if run.failures else "未返回会话（原因未记录）"
                    still.append(theta)
                    # 立刻打出来：只在整个条件结束后汇总，会让一整轮的故障几小时后才被发现
                    print(f"    ⚠ r{replicate} {theta.theta_id} "
                          f"尝试 {attempt}/{CELL_ATTEMPTS} 失败：{reason}")
                    continue
                session, feedback = captured[0]
                record = build_record(
                    condition=condition, index=position,
                    session_id=_plain(getattr(session, "session_id", cell_run_id)),
                    perturbation=theta, session=session, feedback=feedback,
                    profile=profiles[0], run_id=run_id,
                    run_parameters={**run_parameters, "replicate": replicate,
                                    "attempt": attempt, "cell_run_id": cell_run_id},
                )
                record["condition"]["replicate"] = replicate      # 重复编号由驱动决定
                gap = _measurement_gap(record)
                if gap:
                    # 调用成功但产出残缺 → 这不是一次可用的测量，重试而不是落盘
                    still.append(theta)
                    print(f"    ⚠ r{replicate} {theta.theta_id} "
                          f"尝试 {attempt}/{CELL_ATTEMPTS} 判为本次测量失败：{gap}，重试")
                    continue
                _write_json(subjects_dir / f"{position:03d}_{theta.theta_id}_r{replicate}.json",
                            record)
                records.append(record)
                if verbose:
                    info = record["dialogue"]["session"]
                    print(f"    [{position:03d}·r{replicate}] {theta.theta_id}　"
                          f"{info['n_turns_run']} 轮　"
                          f"未覆盖 {len(info['uncovered_channels'])} 通道　"
                          f"平均 {_fmt(record['analysis']['mean_reading'])}　"
                          f"{_likeness_phrase(record['analysis']['asd_likeness'])[:14]}"
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
    condition_payload = {
        **condition.to_dict(),
        "run_id": run_id, "brain_model_id": model_id,
        "examiner_model_id": f"{examiner.name}:{examiner.model}",
        "analyst_model_id": f"{analyst.name}:{analyst.model}",
        "run_parameters": run_parameters,
        "n_records": len(records), "n_expected": expected,
        "failures": failures,
        "probe_keys": [i.get("key", "") for i in probes],
        "rubric_keys": [i.get("key", "") for i in rubric],
        "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(cond_dir / "condition.json", condition_payload)
    if len(records) < expected:
        print(f"    ⚠ 本条件只完成 {len(records)}/{expected} 个单元格"
              "（下游只会保留测量次数齐全的目标）")
    return records, failures






# ---------------------------------------------------------------------------
# 指标聚合：把一个条件的产物算成可靠性数字
# ---------------------------------------------------------------------------


def _readings(record: dict) -> dict:
    return (record.get("analysis") or {}).get("readings") or {}


def _theta_of(record: dict) -> str:
    """取记录里的 θ 标识。

    本脚本的记录把 θ 放在 `parameters.theta_id`、重复放在 `condition.replicate`（嵌套结构），
    而 `reliability` 的适配器默认读顶层 `theta_id`/`replicate`。两者必须显式对上，
    否则每个记录都解析成空串、所有单元格塌成一组 —— 矩阵形状看着正常，ICC 却是错的。
    """
    return str((record.get("parameters") or {}).get("theta_id") or "")


def _replicate_of(record: dict) -> int:
    return int((record.get("condition") or {}).get("replicate") or 0)


def condition_cell_means(records: Sequence[dict]) -> dict[tuple[str, str], float]:
    """(θ, 维度) → 该单元格在重复之间的均值。用于跨条件比较。"""
    acc: dict[tuple[str, str], list[float]] = {}
    for record in records:
        theta = (record.get("parameters") or {}).get("theta_id", "")
        readings = _readings(record)
        for dim in DIMS:
            value = readings.get(dim)
            if value is None:
                continue
            acc.setdefault((theta, dim), []).append(float(value))
    return {key: sum(values) / len(values) for key, values in acc.items()}


def _diagnostics(records: Sequence[dict]) -> dict:
    """轮次轴与分析质量的共变量 —— 这些必须和读数一起报，否则无法解释可靠性差异。"""
    if not records:
        return {"n_records": 0}
    turns = [int((r["dialogue"]["session"] or {}).get("n_turns_run") or 0) for r in records]
    uncovered: dict[str, int] = {}
    under_quota: dict[str, int] = {}
    for record in records:
        session = record["dialogue"]["session"]
        for channel in session.get("uncovered_channels") or []:
            uncovered[channel] = uncovered.get(channel, 0) + 1
        for channel in session.get("under_quota_channels") or []:
            under_quota[channel] = under_quota.get(channel, 0) + 1
    examiner_turns = sum(int(r["dialogue"].get("n_examiner_turns") or 0) for r in records)
    improvised = sum(int(r["dialogue"].get("n_improvised_turns") or 0) for r in records)
    confidences = [r["analysis"]["confidence"] for r in records
                   if r["analysis"].get("confidence") is not None]
    parse_failures = sum(1 for r in records if r["analysis"].get("parse_error"))
    return {
        "n_records": len(records),
        "n_turns_run_mean": _round(sum(turns) / len(turns), 2),
        "n_turns_run_min": min(turns),
        "n_turns_run_max": max(turns),
        "n_turns_run_distinct": sorted(set(turns)),
        "uncovered_channel_counts": uncovered,
        "under_quota_channel_counts": under_quota,
        "improvised_turn_rate": _round(improvised / examiner_turns) if examiner_turns else None,
        "confidence_mean": _round(sum(confidences) / len(confidences)) if confidences else None,
        "parse_failure_rate": _round(parse_failures / len(records)),
        "provider_kinds": sorted({str((r.get("profile") or {}).get("provider_kind")) for r in records}),
        "synthetic": any(bool((r.get("profile") or {}).get("synthetic")) for r in records),
    }


def condition_metrics(records: Sequence[dict]) -> dict:
    """一个条件的全套可靠性指标。

    主数字是 **cell 矩阵**的 ICC：目标 = (θ × 维度) 共 21 个单元格、重复 = 各次运行。
    为什么不用「目标 = θ」的 3 目标形状当主数字：3 个目标的 ICC 置信区间宽到没有信息量
    （`reliability.ICC_CI_MIN_TARGETS` 会直接拒绝给区间）。逐维度 ICC 仍照报，但要看清它没有区间。
    """
    if not records:
        return {"n_records": 0}

    cell_matrix, dropped_cells = rel.cell_matrix_from_records(
        records, DIMS, lambda r, d: _readings(r).get(d),
        target_fn=_theta_of, replicate_fn=_replicate_of,
    )
    # 形状自检分两步，因为「键对不上」和「格稀疏」是两回事，混在一起会把正常数据判死：
    #   1) 分组键自检：(θ × 维度) 的组合**是否存在**。少一个就说明 target_fn/维度名与记录结构对不上，
    #      这属于必须失败的错误（21 格塌成 7 格而 ICC 看着合理，正是这类错误）。
    #   2) 稀疏自检：组合都在、只是某格**可用重复数不够** → 该格被 `matrix_from_records` 丢掉，
    #      不伪造数据。这是真实的测量缺口（分析者可以对某一维留 null），只记录、不抛错。
    expected_cells = len({_theta_of(r) for r in records}) * len(DIMS)
    seen_cells = {
        f"{_theta_of(r)}|{dim}" for r in records for dim in DIMS
        if _readings(r).get(dim) is not None
    }
    if cell_matrix and len(seen_cells) != expected_cells:
        raise ValueError(
            f"cell 矩阵形状异常：得到 {len(seen_cells)} 个 (θ × 维度) 组合，应为 {expected_cells} 个 "
            f"(θ × {len(DIMS)} 维度) —— 分组键很可能与记录结构不匹配"
        )
    sparse_cells = sorted(dropped_cells)   # 可用重复数不足、已剔出 ICC 的格（见上面的稀疏自检）
    overall = rel.reliability_report(cell_matrix) if cell_matrix else {"n_targets": 0}

    per_dim: dict[str, dict | None] = {}
    for dim in DIMS:
        matrix, _ = rel.matrix_from_records(
            records, lambda r, d=dim: _readings(r).get(d),
            target_fn=_theta_of, replicate_fn=_replicate_of,
        )
        per_dim[dim] = rel.icc21(matrix).to_dict() if matrix else None

    # 结论级一致度：档位（结构化、可辩护）+ 结论文本（词汇层近似）
    levels: dict[str, dict[int, str]] = {}
    conclusions: dict[str, list[str]] = {}
    for record in records:
        theta = (record.get("parameters") or {}).get("theta_id", "")
        replicate = int((record.get("condition") or {}).get("replicate") or 0)
        likeness = record["analysis"].get("asd_likeness") or {}
        levels.setdefault(theta, {})[replicate] = str(likeness.get("level") or "unknown")
        conclusions.setdefault(theta, []).append(record["analysis"].get("conclusion") or "")
    likeness_rows = [[levels[t][k] for k in sorted(levels[t])] for t in sorted(levels)]
    per_theta_conclusion = {t: rel.conclusion_similarity(v) for t, v in sorted(conclusions.items())}
    conclusion_agg = {
        "n_theta": len(per_theta_conclusion),
        "jaccard_bigram_mean": _round(
            sum(c["jaccard_bigram"] for c in per_theta_conclusion.values() if c["jaccard_bigram"] is not None)
            / max(1, sum(1 for c in per_theta_conclusion.values() if c["jaccard_bigram"] is not None))
        ),
        "chars_mean": _round(
            sum(c["chars_mean"] for c in per_theta_conclusion.values() if c["chars_mean"] is not None)
            / max(1, sum(1 for c in per_theta_conclusion.values() if c["chars_mean"] is not None))
        , 1),
        "per_theta": per_theta_conclusion,
    }

    # 目标水平的一致性：每个 θ 的 7 维剖面在重复之间是否稳
    theta_vectors: dict[str, list[list[float]]] = {}
    for record in records:
        theta = (record.get("parameters") or {}).get("theta_id", "")
        readings = _readings(record)
        vector = [readings.get(d) for d in DIMS]
        if all(v is not None for v in vector):
            theta_vectors.setdefault(theta, []).append([float(v) for v in vector])
    theta_profiles = {}
    for theta, vectors in sorted(theta_vectors.items()):
        if len(vectors) < 2:
            continue
        theta_profiles[theta] = {
            "n_replicates": len(vectors),
            "pairwise": rel.pairwise_agreement(vectors),
            "rank_stability": rel.rank_stability(vectors, dims=DIMS),
        }

    return {
        "n_records": len(records),
        "cell_icc": overall,
        "cell_matrix_shape": [len(cell_matrix), len(cell_matrix[0]) if cell_matrix else 0],
        "dropped_cells": dropped_cells,
        "sparse_cells": sparse_cells,
        "per_dimension_icc": per_dim,
        "likeness_agreement": rel.categorical_agreement(likeness_rows) if likeness_rows else None,
        "likeness_by_theta": levels,
        "conclusion_agreement": conclusion_agg,
        "theta_profile_stability": theta_profiles,
        "diagnostics": _diagnostics(records),
    }


def compare_conditions(
    base_records: Sequence[dict], other_records: Sequence[dict],
    base_id: str, other_id: str,
) -> dict:
    """两个条件在同一批 (θ, 维度) 单元格上的一致性（跨模型 / 跨提问者 / 跨轮次的比较）。"""
    left = condition_cell_means(base_records)
    right = condition_cell_means(other_records)
    keys = sorted(set(left) & set(right))
    if len(keys) < 3:
        return {"base": base_id, "other": other_id, "n_cells": len(keys),
                "note": "共同单元格不足 3 个，无法比较"}
    xs = [left[k] for k in keys]
    ys = [right[k] for k in keys]
    profile = rel.profile_agreement(xs, ys)
    ba = rel.bland_altman(xs, ys)
    return {
        "base": base_id, "other": other_id, "n_cells": len(keys),
        "profile_agreement": profile.to_dict() if profile else None,
        "bland_altman": ba.to_dict() if ba else None,
        "paired_icc": rel.icc_point([xs, ys]).__round__(4) if len(xs) >= 2 else None,
    }


# ---------------------------------------------------------------------------
# 报告与面板
# ---------------------------------------------------------------------------


# 报告与面板
# ---------------------------------------------------------------------------


def _axis_rows(axis: str, conditions: Sequence[Condition], metrics: dict) -> list[dict]:
    """把一个轴的各条件汇总成表格行（报告与面板共用）。"""
    rows = []
    for condition in conditions:
        if condition.axis != axis or condition.condition_id not in metrics:
            continue
        got = metrics[condition.condition_id]
        cell = got.get("cell_icc") or {}
        icc = cell.get("icc") or {}
        profile = cell.get("profile_agreement") or {}
        likeness = got.get("likeness_agreement") or {}
        conclusion = got.get("conclusion_agreement") or {}
        diag = got.get("diagnostics") or {}
        rows.append({
            "condition": condition.condition_id,
            "label": condition.label,
            "describe": condition.describe(),
            "note": condition.note,
            "n_records": got.get("n_records"),
            "n_targets": cell.get("n_targets"),
            "n_sparse_cells": len(got.get("dropped_cells") or []),
            "sparse_cells": got.get("dropped_cells") or [],
            "n_turns": condition.n_turns,
            "max_turns": condition.max_turns,
            "icc": icc.get("icc"),
            "icc_ci_low": icc.get("ci_low"),
            "icc_ci_high": icc.get("ci_high"),
            "icc_ci_method": icc.get("ci_method"),
            "sem": cell.get("sem"),
            "sem_ci_low": cell.get("sem_ci_low"),
            "sem_ci_high": cell.get("sem_ci_high"),
            "cv_mean": cell.get("cv_mean"),
            "pearson": profile.get("pearson"),
            "spearman": profile.get("spearman"),
            "within_tolerance": profile.get("within_tolerance"),
            "mean_abs_diff": profile.get("mean_abs_diff"),
            "likeness_kappa": likeness.get("kappa"),
            "likeness_weighted_kappa": likeness.get("kappa_weighted"),
            "likeness_unanimous": likeness.get("unanimous_rate"),
            "conclusion_jaccard": conclusion.get("jaccard_bigram_mean"),
            "turns_mean": diag.get("n_turns_run_mean"),
            "improvised_rate": diag.get("improvised_turn_rate"),
            "confidence_mean": diag.get("confidence_mean"),
            "parse_failure_rate": diag.get("parse_failure_rate"),
            "uncovered_channel_counts": diag.get("uncovered_channel_counts"),
            "synthetic": diag.get("synthetic"),
        })
    return rows


def build_panels(conditions: Sequence[Condition], metrics: dict, comparisons: dict) -> dict:
    """给绘图脚本的面板数据（每面板 = 一组「条件 × 指标 × 值 ± 区间」）。"""
    panels: dict[str, dict] = {}
    for axis, name in AXIS_NAMES.items():
        rows = _axis_rows(axis, conditions, metrics)
        if rows:
            panels[AXIS_KEYS[axis]] = {"axis": axis, "name": name, "rows": rows}
    return {
        "panels": panels,
        "comparisons": comparisons,
        "direction": ("读数 1.0 = 与同龄典型发展无异，0.0 = 严重偏离；"
                      "ICC 越高越可靠，CV / SEM / 一致性界限越小越可靠"),
    }


def conditions_from_payload(payload: dict) -> list[Condition]:
    """报告层从 payload 重建条件对象（只为了拿标签与轴）。"""
    out = []
    for item in payload.get("conditions", []):
        out.append(Condition(
            condition_id=item["condition_id"], axis=item["axis"], axis_name=item["axis_name"],
            label=item["label"], brain_model=item["brain_model"],
            examiner_model=item["examiner_model"], analyst_model=item["analyst_model"],
            seed=item["seed"], n_turns=item["n_turns"], max_turns=item["max_turns"],
            replicates=item["replicates"], n_thetas=item.get("n_thetas", 3),
            theta_limit=item.get("theta_limit"),
            probe_mode=item.get("probe_mode", "auto"),
            reference=item.get("reference", False), note=item.get("note", ""),
        ))
    return out


LIMITATIONS = [
    "**Mock 不是证据**：mock provider 是确定性规则替身，其可靠性恒为 1 —— 那验证的是管线，"
    "不是 LLM 涌现行为。本实验的结论必须来自真实 provider 的运行。",
    "**真实运行没有 seed 可用**：provider 请求里只有 model / messages / temperature / max_tokens，"
    "不发送 seed，temperature 固定取配置值（默认 0.3）。所以轴① 测的是**服务端采样噪声**，"
    "不是「同种子可复现」这种确定性主张。",
    "**轴④ 的 L=25 低于配额下限**（7 通道 × 5 问 = 35）：实测该档是**每个通道都被问到至少一次、"
    "但远达不到每通道 5 问的配额**（0 个 `uncovered_channels`、每被试约 5.6 个 `under_quota_channels`），"
    "7 个维度仍都有读数 —— 那些读数是分析者按提示词指示给出的**保守中性值**，"
    "属于「证据量不足」而非「缺测」。报告里必须与 50 / 100 档分开表述，并给出配额不足的通道数。",
    "**题集与轮数耦合已被缓解但未消除**：所有条件复用同一份清单、轮数按位置截断，"
    "使 L=25 用的正是 L=50 的前 25 题；但提问者**自造题**的轮次会打破严格嵌套，"
    "因此诊断表里给出「自造题率」供核对。",
    "**轴③ 的 seed 只改变第 7 位起的题序**：`dialogue_probes` 的前 6 位由目标通道确定性选出，"
    "对任何 seed 都相同；观察提纲对所有 seed 完全一致（其随机源未被使用）。",
    "**ICC 区间的实际覆盖率略低于名义值**：(θ × 7 维度) 目标数在 21～28 一档、每目标 5 重复下，"
    "jackknife 区间实测覆盖 0.91–0.93（而非 0.95），且目标数少于 10 时不给区间。"
    "精度陈述应优先引用 SEM 的精确 χ² 区间。",
    "**逐维度 ICC 只有 θ 数个目标**（3 θ 时 3 个、4 θ 时 4 个），点估计可用但不可给区间，"
    "也不应据此排名单个维度。",
    "**结论级相似度是词汇层近似**：字符 n-gram Jaccard 衡量用词与句式，两个意思相同但措辞不同的"
    "结论会被判为低相似。结论级可靠性的主指标应看 `asd_likeness` 档位一致度（κ 与完全一致率）。",
    "**本实验不写入项目数据库**（产物全在文件系统），因此 `pacs analyze` / `pacs report` "
    "看不到这批数据。",
    "**单次输出额度是协议的一部分，且额度不足会伪装成「测量完成」**：正文被 `max_tokens` 截断成"
    "**空串**时 provider 会报错（`finish_reason=length`），但截断成**半句**时会被当作完整回答收下、"
    "不留标记。实测失败率：脑/提问者共用 8000 时一轮 6 次尝试坏 1 次；分析者共用 8000 时 v4-pro 的"
    "输出 **4/4 份全被截断**（两份只救回 5/7 与 2/7 个维度的读数，且无结论与档位判定）。"
    "因此分析者单独给了 32768（已是默认值），并把「完整测量」作为入矩阵的门槛。",
    "**残缺产物的四道判据**（`parse_error` 非空 / 一个维度都没有 / 结论为空 / 档位缺失）都被显式判为"
    "「本次测量失败」并重试，续跑也会拒绝复用残缺产物。**刻意不要求 7 维齐全**：低轮数下未被探查到的"
    "通道给 None 是合法结果，硬要求齐全反而会把有效测量误判成失败。",
    "**单元格级重试与补漏**：单个 (θ, 重复) 失败最多重试 3 次；整轮跑完仍缺格的条件下会再补跑一次。"
    "实测踩过端点抖动窗口导致一格三次尝试全超时（丢 2 格）、以及分析者输出完全无法解析（重试后成功）。"
    "报告里的 n 是**实际入矩阵的目标数**；某条件若缺格，其数字不能与其他条件直接比。",
]


def write_report(out_dir: Path, payload: dict) -> None:
    """落盘 fig1_reliability.json / fig1_panels.json / fig1_reliability.md。"""
    rel_dir = out_dir / "reliability"
    rel_dir.mkdir(parents=True, exist_ok=True)
    _write_json(rel_dir / "fig1_reliability.json", payload)
    _write_json(rel_dir / "fig1_panels.json", payload.get("panels", {}))

    conditions = conditions_from_payload(payload)
    lines: list[str] = []
    add = lines.append
    add(f"# Fig.1 系统可靠性 · {payload['run_id']}")
    add("")
    add(f"- 生成时间：{payload['created_at']}")
    add(f"- 心智年龄：{payload['age_months']} 个月（固定常量，不可配置）")
    add(f"- θ 清单：{payload['theta_summary']}")
    add(f"- 轮次协议：轴①②③ 用 {payload['base_turns']}/{payload['base_max_turns']}"
        f"（跑满 {payload['base_turns']} 轮后配额满足即收工，仍有缺口补到 {payload['base_max_turns']}）；"
        f"轴④ 硬固定到该档")
    add(f"- 题集：每个 (seed, mode) 一份 {payload['probe_pool']} 条清单，全条件复用，轮数按位置截断")
    add(f"- 被试总数：{payload['n_subjects']}　预计调用："
        f"{payload['cost']['calls_lower_bound']}（基础轮）～{payload['cost']['calls_upper_bound']}（上界）")
    add("")
    add("> 读数方向：**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**；"
        "ICC 越高越可靠，CV / SEM / Bland-Altman 界限越小越可靠。")
    add("")
    add(f"> {SIMULATION_NOTICE}")
    add(">")
    add(f"> {DISCLAIMER}")
    add("")

    if payload.get("synthetic_any"):
        add("## ⚠ 合成数据告警")
        add("")
        add("本批包含 **mock（规则式替身）** 剖面。mock 是确定性的，可靠性必然≈1 —— "
            "那是 mock 规则的性质，**不是** LLM 涌现行为的可靠性。这份结果只能用于验证管线。")
        add("")

    for axis, name in AXIS_NAMES.items():
        rows = _axis_rows(axis, conditions, payload["metrics"])
        if not rows:
            continue
        add(f"## 轴{axis} · {name}")
        add("")
        add("| 条件 | 被试 | ICC(2,1) | ICC 区间 | 方法 | SEM | SEM 区间 | CV | 剖面 Pearson | "
            "落在容差内 | 档位 κ | 档位完全一致 | 结论 Jaccard | 平均轮数 | 自造题率 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            icc_ci = (f"[{_fmt(row['icc_ci_low'])}, {_fmt(row['icc_ci_high'])}]"
                      if row["icc_ci_low"] is not None else "—（目标数不足）")
            sem_ci = (f"[{_fmt(row['sem_ci_low'])}, {_fmt(row['sem_ci_high'])}]"
                      if row["sem_ci_low"] is not None else "—")
            add(f"| {row['label']} | {row['n_records']} | {_fmt(row['icc'])} | {icc_ci} | "
                f"{row['icc_ci_method']} | {_fmt(row['sem'])} | {sem_ci} | {_fmt(row['cv_mean'])} | "
                f"{_fmt(row['pearson'])} | {_fmt(row['within_tolerance'])} | "
                f"{_fmt(row['likeness_kappa'])} | {_fmt(row['likeness_unanimous'])} | "
                f"{_fmt(row['conclusion_jaccard'])} | {_fmt(row['turns_mean'], 1)} | "
                f"{_fmt(row['improvised_rate'])} |")
        add("")
        notes = [row for row in rows if row.get("note")]
        for row in notes:
            add(f"- _{row['label']}：{row['note']}_")
        if notes:
            add("")
        # 稀疏格必须自己说出来：某格可用重复数不够时它被剔出 ICC（不补值、不伪造），
        # 于是该条件的 target 数比别处少，跨条件比 ICC 时要看这一行。
        sparse = [row for row in rows if row.get("n_sparse_cells")]
        for row in sparse:
            names = "、".join(str(c).replace("|", " × ") for c in row["sparse_cells"])
            add(f"- ⚠ _{row['label']}：{row['n_sparse_cells']} 个 (θ×维度) 格可用重复数少于其他格，"
                f"已从 ICC 剔除（不补值）：{names}；该条件 cell 目标数 = {row['n_targets']}_")
        if sparse:
            add("")

        comparisons = payload["comparisons"].get(AXIS_KEYS[axis], []) or []
        if comparisons:
            add("**跨条件一致度**（同一批 (θ×维度) 单元格的配对比较）")
            add("")
            add("| 基准 | 对照 | 单元格 | 剖面 Pearson | 平均绝对差 | 最大绝对差 | 落在容差内 | 偏倚 |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for item in comparisons:
                if item.get("n_cells", 0) < 3:
                    add(f"| {item['base']} | {item['other']} | {item.get('n_cells')} | — | — | — | — | — |")
                    continue
                profile = item.get("profile_agreement") or {}
                bland = item.get("bland_altman") or {}
                add(f"| {item['base']} | {item['other']} | {item['n_cells']} | "
                    f"{_fmt(profile.get('pearson'))} | {_fmt(profile.get('mean_abs_diff'))} | "
                    f"{_fmt(profile.get('max_abs_diff'))} | {_fmt(profile.get('within_tolerance'))} | "
                    f"{_fmt(bland.get('bias'))} |")
            add("")

    # 目标数随 θ 清单变化（3 θ → 3/21，4 θ → 4/28），所以不能写死。以各条件里出现过的 θ 数为准。
    counts = [len((got.get("likeness_by_theta") or {})) for got in payload["metrics"].values()]
    n_thetas = max(counts) if counts else len(payload.get("thetas") or [])
    n_cells = n_thetas * len(DIMS)
    add(f"## 逐维度 ICC（目标 = θ，仅 {n_thetas} 个目标 → 不给区间）")
    add("")
    add("| 条件 | " + " | ".join(SCORED_CHANNELS[d] for d in DIMS) + " |")
    add("| --- |" + " --- |" * len(DIMS))
    for condition in conditions:
        got = payload["metrics"].get(condition.condition_id)
        if not got:
            continue
        per_dim = got.get("per_dimension_icc") or {}
        cells = " | ".join(_fmt((per_dim.get(d) or {}).get("icc")) for d in DIMS)
        add(f"| {condition.label} | {cells} |")
    add("")
    add(f"{n_thetas} 个目标的 ICC 置信区间宽到没有信息量，因此 `reliability.ICC_CI_MIN_TARGETS` 会拒绝给出区间；"
        f"主数字请引 cell 矩阵（{n_cells} 个目标）那一行，精度陈述请引 SEM 的精确 χ² 区间。")
    add("")

    add("## 诊断（解释可靠性差异的共变量）")
    add("")
    add("| 条件 | 平均轮数 | 轮数分布 | 自造题率 | 平均置信度 | 解析失败率 | 未覆盖通道 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for condition in conditions:
        got = payload["metrics"].get(condition.condition_id)
        if not got:
            continue
        diag = got.get("diagnostics") or {}
        uncovered = diag.get("uncovered_channel_counts") or {}
        uncovered_text = "；".join(
            f"{SCORED_CHANNELS.get(k, k)}×{v}" for k, v in sorted(uncovered.items())
        ) or "无"
        add(f"| {condition.label} | {_fmt(diag.get('n_turns_run_mean'), 1)} | "
            f"{diag.get('n_turns_run_distinct')} | {_fmt(diag.get('improvised_turn_rate'))} | "
            f"{_fmt(diag.get('confidence_mean'))} | {_fmt(diag.get('parse_failure_rate'))} | "
            f"{_one_line(uncovered_text, 60)} |")
    add("")

    add("## 已知局限（引用这些数字时必须一并声明）")
    add("")
    for index, item in enumerate(payload["limitations"], start=1):
        add(f"{index}. {item}")
    add("")

    (rel_dir / "fig1_reliability.md").write_text("\n".join(lines), encoding="utf-8")


# 扁平数据集导出：让这批结果能被下游分析直接接走
# ---------------------------------------------------------------------------

#: `all_subjects.csv` 的列顺序（宽表：一行 = 一个被试）
FLAT_SUBJECT_COLUMNS = [
    "run_id", "condition_id", "axis", "axis_name", "condition_label",
    "brain_model", "examiner_model", "analyst_model", "seed",
    "n_turns_set", "max_turns_set", "n_turns_run", "replicates_set",
    "subject_index", "replicate", "session_id", "saved_at",
    "theta_id", "theta_describe", "theta_active", "generic_degradation",
    *DIMS,
    "mean_reading", "confidence",
    "likeness_present", "likeness_level",
    "conclusion_chars", "conclusion",
    "uncovered_channels", "under_quota_channels", "n_under_quota",
    "n_examiner_turns", "n_improvised_turns", "improvised_turn_rate",
    "over_limit_turns", "n_probe_keys_planned",
    "parse_error", "provider_kind", "synthetic", "subject_json_path",
]

#: `readings_long.csv` 的列（长表：一行 = 一个 被试×维度）
FLAT_READING_COLUMNS = [
    "run_id", "condition_id", "axis", "theta_id", "subject_index", "replicate",
    "session_id", "dim", "dim_label", "is_control",
    "score", "typicality", "deviation", "confidence", "observations", "task_measurements",
]


def _flat_subject_row(record: dict, json_path: Path) -> dict:
    condition = record.get("condition") or {}
    parameters = record.get("parameters") or {}
    session = (record.get("dialogue") or {}).get("session") or {}
    analysis = record.get("analysis") or {}
    dialogue = record.get("dialogue") or {}
    profile = record.get("profile") or {}
    likeness = analysis.get("asd_likeness") or {}
    readings = analysis.get("readings") or {}

    examiner_turns = int(dialogue.get("n_examiner_turns") or 0)
    improvised = int(dialogue.get("n_improvised_turns") or 0)
    active = parameters.get("active") or {}
    under_quota = list(session.get("under_quota_channels") or [])
    row = {
        "run_id": record.get("run_id", ""),
        "condition_id": condition.get("condition_id", ""),
        "axis": condition.get("axis", ""),
        "axis_name": condition.get("axis_name", ""),
        "condition_label": condition.get("label", ""),
        "brain_model": condition.get("brain_model", ""),
        "examiner_model": condition.get("examiner_model", ""),
        "analyst_model": condition.get("analyst_model", ""),
        "seed": condition.get("seed"),
        "n_turns_set": condition.get("n_turns"),
        "max_turns_set": condition.get("max_turns"),
        "n_turns_run": session.get("n_turns_run"),
        "replicates_set": condition.get("replicates"),
        "subject_index": record.get("subject_index"),
        "replicate": condition.get("replicate"),
        "session_id": record.get("session_id", ""),
        "saved_at": record.get("saved_at", ""),
        "theta_id": parameters.get("theta_id", ""),
        "theta_describe": parameters.get("describe", ""),
        "theta_active": "；".join(f"{k}={v}" for k, v in sorted(active.items())) or "完整心智",
        "generic_degradation": parameters.get("generic_degradation"),
        "mean_reading": analysis.get("mean_reading"),
        "confidence": analysis.get("confidence"),
        "likeness_present": likeness.get("present"),
        "likeness_level": likeness.get("level", ""),
        "conclusion_chars": analysis.get("conclusion_chars"),
        "conclusion": _one_line(analysis.get("conclusion"), 400),
        "uncovered_channels": "；".join(
            SCORED_CHANNELS.get(c, c) for c in (session.get("uncovered_channels") or [])
        ),
        "under_quota_channels": "；".join(SCORED_CHANNELS.get(c, c) for c in under_quota),
        "n_under_quota": len(under_quota),
        "n_examiner_turns": examiner_turns,
        "n_improvised_turns": improvised,
        "improvised_turn_rate": _round(improvised / examiner_turns) if examiner_turns else None,
        "over_limit_turns": session.get("over_limit_turns"),
        "n_probe_keys_planned": len(session.get("probe_keys") or []),
        "parse_error": analysis.get("parse_error", ""),
        "provider_kind": profile.get("provider_kind", ""),
        "synthetic": profile.get("synthetic"),
        "subject_json_path": str(json_path),
    }
    for dim in DIMS:
        row[dim] = readings.get(dim)
    return row


def write_flat_dataset(out_dir: Path, records: Sequence[dict]) -> dict:
    """把全部被试摊平成两张可直接接走的表 —— 供后续分析（其他论文、跨实验合并）复用。

    为什么除了逐被试 JSON 还要这两张表：JSON 是**完整留档**（含对话全文，几 MB），
    适合回看；而下游统计/绘图需要的是**规整表**，让它自己去遍历几百个 JSON 既慢又容易读错字段。

    - `all_subjects.csv`：宽表，一行 = 一个被试，含 7 维读数与全部条件/诊断字段（Excel 可开）
    - `readings_long.csv`：长表，一行 = 一个「被试 × 维度」，最便于分组统计与画图
    - `数据字典.md`：每一列的含义与口径

    文件名固定、`--out` 目录独立，因此这批结果不会被后续运行覆盖或混淆。
    """
    if not records:
        return {"n_subjects": 0}

    flat_dir = out_dir / "flat"
    flat_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = []
    reading_rows = []
    for index, record in enumerate(records, start=1):
        condition = record.get("condition") or {}
        session = (record.get("dialogue") or {}).get("session") or {}
        stem = f"{record.get('subject_index') or index:03d}_" \
               f"{(record.get('parameters') or {}).get('theta_id', 'theta')}_r{condition.get('replicate')}"
        json_path = out_dir / condition.get("axis_dir", "") / condition.get("condition_id", "") \
            / "subjects" / f"{stem}.json"
        subject_rows.append(_flat_subject_row(record, json_path))

        dimensions = (record.get("analysis") or {}).get("dimensions") or {}
        for dim in DIMS:
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

    # utf-8-sig：Excel 双击打开中文不乱码
    with (flat_dir / "all_subjects.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAT_SUBJECT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(subject_rows)
    with (flat_dir / "readings_long.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAT_READING_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reading_rows)

    (flat_dir / "数据字典.md").write_text(_data_dictionary(), encoding="utf-8")
    return {
        "n_subjects": len(subject_rows),
        "n_readings": len(reading_rows),
        "dir": str(flat_dir),
    }


def _data_dictionary() -> str:
    """数据字典：让拿到这批结果的人不必读代码就能用。"""
    rows = [
        ("run_id", "运行号（一次 fig1_reliability.py 调用）"),
        ("condition_id", "条件 id（`axis1_same_brain` / `axis2_brain_pro` / …）"),
        ("axis", "比较轴：①同一脑重复 ②不同 LLM ③不同 examiner/analyst/题序 ④不同轮次"),
        ("condition_label", "条件的中文描述"),
        ("brain_model", "模拟大脑（被扰动的心智）模型名"),
        ("examiner_model", "提问者模型名"),
        ("analyst_model", "分析者模型名（读数由它判出）"),
        ("seed", "题目集合与运行种子。轴③ 的 seed 条件靠它改题序"),
        ("n_turns_set", "**设定**的基础轮数"),
        ("max_turns_set", "**设定**的轮数上限"),
        ("n_turns_run", "**实际**发生的轮数（配额满足即收工，故可能小于上限）"),
        ("replicates_set", "该条件计划的重复次数"),
        ("subject_index", "被试在设计表里的序号（同一 θ 的重复共享）"),
        ("replicate", "同一 θ 的第几次重复（0 起）"),
        ("session_id", "会话号（逐轮记录的键）"),
        ("theta_id", "θ 指纹（由取值派生；同 θ 同 id）"),
        ("theta_describe", "θ 的中文描述，如「[单因素] 信念更新 0.20」"),
        ("theta_active", "被削弱的认知过程及完整度；完整心智写「完整心智」"),
        ("generic_degradation", "通用退化水平（0 = 无）"),
        ("<7 个维度列>", "该维读数，**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**；"
                         "`general_cognition` 是对照通道（不进雷达图）"),
        ("mean_reading", "7 个通道读数的平均"),
        ("confidence", "分析者自评置信度（对证据偏薄的通道会主动降低）"),
        ("likeness_present / likeness_level", "表型印象：是否表现出孤独症样特征 + 程度档位"
                                               "（none/mild/moderate/marked）"),
        ("conclusion_chars / conclusion", "探查结论的字符数 / 截断到 400 字的正文"),
        ("uncovered_channels", "整段对话**一个问题都没问到**的通道（其读数没有证据支撑）"),
        ("under_quota_channels / n_under_quota", "被问到过但**问题数不足 5** 的通道（证据偏薄）"),
        ("n_examiner_turns / n_improvised_turns / improvised_turn_rate",
         "提问轮数 / 提问者没用上题库条目的轮数 / 两者之比"),
        ("over_limit_turns", "问题或回答超出字数上限的轮数（只告警、不截断）"),
        ("parse_error", "分析者输出解析失败的原因（空 = 正常）"),
        ("provider_kind / synthetic", "读数来源：`llm` 真实模型 / `mock` 规则式替身。"
                                      "`synthetic=True` 的剖面**不能当作 LLM 涌现证据**"),
        ("subject_json_path", "该被试的完整 JSON（含对话全文）相对路径"),
    ]
    lines = ["# 数据字典", "", "## `all_subjects.csv`（宽表：一行 = 一个被试）", "",
             "| 列 | 含义 |", "| --- | --- |"]
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
        "", "## 与谁合表", "",
        "两张表都带 `run_id` / `condition_id` / `theta_id` / `session_id`，"
        "可直接与逐被试 JSON（`subjects/*.json`，含**逐轮对话全文**与逐题 `probe_key`）按",
        "`session_id` 关联；跨实验合并时按 `theta_id` + `dim` 对齐。",
        "", "## 方向与口径", "",
        "- 读数方向：**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**（与 θ 的完整度同向）。",
        "- 心智年龄固定 36 个月，**不是参数**。",
        "- `n_turns_run < n_turns_set` 说明配额在设定轮数内已满足、提前收工；",
        "  若 `n_under_quota > 0` 则说明证据偏薄，该被试的读数**要打折看**。",
        "- `synthetic=True` 的行来自 mock provider，**不可作为科学结论的依据**。",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fig1_reliability",
        description="Fig.1 系统可靠性实验：轴①同一脑重复 ②不同LLM ③不同examiner/analyst/题序 ④不同轮次",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/fig1_reliability.py --dry-run\n"
            "  python tools/fig1_reliability.py --provider mock --mini\n"
            "  python tools/fig1_reliability.py --model-a deepseek-flash --model-b deepseek-v4-pro --yes\n"
            "  python tools/fig1_reliability.py --conditions axis1_same_brain --yes\n"
        ),
    )
    parser.add_argument("--provider", choices=("mock", "openai"), default=None,
                        help="LLM 提供方（默认读 config.json；mock 为离线试跑）")
    parser.add_argument("--model-a", default=None,
                        help="A 模型名（默认取 config.json 的 model）")
    parser.add_argument("--model-b", default=None,
                        help="B 模型名（默认把 A 里的 flash 换成 pro；没有 flash 则追加 -alt）")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--analyst-max-tokens", type=int, default=ANALYST_OUTPUT_TOKENS,
                        dest="analyst_max_tokens",
                        help=f"分析者单独的输出额度上限（默认 {ANALYST_OUTPUT_TOKENS}）。分析者输出最长"
                             "（7 段逐维记录 + 约 350 字结论 + 判定依据），推理型模型还要先花掉一大截"
                             "思维链额度：共用脑/提问者的额度会把 JSON 截断成半截 —— 实测 v4-pro 当"
                             "分析者时 4/4 份全被截断，其中两份只救回 5/7 与 2/7 个维度的读数。"
                             "**默认就给足**，免得忘了加这个参数就跑出一批残缺数据")
    parser.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                        help="所有角色的单次输出额度上限（默认用 config.json 的值）。"
                             "推理型模型会把额度花在思维链上：额度太小会导致正文被截断成空，"
                             "表现为「提问者返回空内容（finish_reason=length）」——"
                             "那是**调用失败**，不是心智沉默")

    parser.add_argument("--thetas", default=DEFAULT_THETA_SPEC,
                        help=f"θ 清单，形如 `intact,belief_updating:0.3`（默认 {DEFAULT_THETA_SPEC}）")
    parser.add_argument("--replicates", type=int, default=None,
                        help="参照条件的重复次数（其余条件按其 3/5 取；默认 5）")
    parser.add_argument("--turns", type=int, default=BASE_TURNS,
                        help=f"轴①②③ 的基础轮数（默认 {BASE_TURNS}）")
    parser.add_argument("--max-turns", type=int, default=BASE_MAX_TURNS,
                        help=f"轴①②③ 的补问上限（默认 {BASE_MAX_TURNS}）")
    parser.add_argument("--turn-levels", default=None,
                        help=f"轴④ 的轮次档位，逗号分隔（默认 {'/'.join(map(str, TURN_LEVELS))}）")
    parser.add_argument("--probe-pool", type=int, default=PROBE_POOL,
                        help=f"每个 (seed, mode) 生成的题集大小（默认 {PROBE_POOL}）")
    parser.add_argument("--probe-mode", choices=("auto", "scenario", "question"), default="auto")
    parser.add_argument("--text-limit", type=int, default=None,
                        help=f"问题与回答的字数上限（默认 {DEFAULT_TEXT_LIMIT}；超出只告警）")

    parser.add_argument("--axes", default=None, help="只跑这些轴，如 `①②` 或 `1,4`")
    parser.add_argument("--conditions", default=None, help="只跑这些条件 id，逗号分隔")
    parser.add_argument("--theta-limit", default=None, dest="theta_limit",
                        help="按条件限定 θ 数量，形如 `axis2_brain_pro=3`（逗号分隔可给多个）。"
                             "补跑新 θ 时用来把**对照臂**摘出去：往对照轴里加新 θ 会改变它"
                             "要回答的问题，也会多跑几格")
    parser.add_argument("--mini", action="store_true",
                        help="缩微矩阵（每条件 2 个 θ × 2 次重复，轴④只留 2 档）—— 用于干跑管线")
    parser.add_argument("--out", default=None, help="产物根目录（默认 data/runs/fig1）")
    parser.add_argument("--base-dir", default=None, help="项目根目录（默认自动定位）")
    parser.add_argument("--resume", action="store_true", help="跳过产物已齐全的条件")
    parser.add_argument("--no-report", action="store_true", dest="no_report",
                        help="只跑测量、不生成报告与扁平数据集。并行拆分多个进程时必须加："
                             "每个进程只认识自己那批条件，各自写报告会互相覆盖、且内容残缺")
    parser.add_argument("--dry-run", action="store_true", help="只打印矩阵与成本，不调用模型")
    parser.add_argument("--yes", action="store_true", help="确认大规模真实调用")
    return parser.parse_args(argv)


#: 已知的「同族对照模型」配对。按名字猜是很危险的：端点支持的模型名未必是 `-pro`
#: （实测 `deepseek-pro` 会 400，正确的对照是 `deepseek-v4-pro`），而猜错会让整批运行
#: 在真实调用时全线失败。因此这里只登记**已验证过**的配对，其余情况要求显式传 `--model-b`。
KNOWN_MODEL_PAIRS = {
    "deepseek-flash": "deepseek-v4-pro",
}


def _derive_model_b(model_a: str) -> str | None:
    """推导对照模型名。**推不出来就返回 None**，由调用方要求显式指定。"""
    return KNOWN_MODEL_PAIRS.get(model_a)


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.load(base_dir=Path(args.base_dir) if args.base_dir else PROJECT_ROOT)
    if args.provider:
        settings.llm.provider = args.provider
    if args.model_a:
        settings.llm.model = args.model_a
    if args.base_url:
        settings.llm.base_url = args.base_url
    if args.api_key:
        settings.llm.api_key = args.api_key
    if args.temperature is not None:
        settings.llm.temperature = args.temperature
    if args.max_tokens is not None:
        if args.max_tokens < 256:
            raise SystemExit(f"✗ --max-tokens={args.max_tokens} 太小，正文会被截断成空")
        # 驱动只用 settings.llm 造三方 provider（脑 / 提问者 / 分析者），所以改这一处即可；
        # 分析者在 provider 内另有 8000 的下限，这里是**抬高的共同上限**。
        settings.llm.max_tokens = args.max_tokens
    settings.ensure_dirs()
    return settings


def _select_conditions(args: argparse.Namespace, conditions: list[Condition]) -> list[Condition]:
    selected = conditions
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",") if c.strip()}
        missing = wanted - {c.condition_id for c in selected}
        if missing:
            raise SystemExit(f"✗ 未知条件：{'、'.join(sorted(missing))}")
        selected = [c for c in selected if c.condition_id in wanted]
    if args.axes:
        wanted = {a.strip().lstrip("轴") for a in args.axes.replace("①", "1").replace("②", "2")
                  .replace("③", "3").replace("④", "4").split(",") if a.strip()}
        selected = [c for c in selected if c.axis in wanted]
    return selected


def _turn_levels(args: argparse.Namespace) -> tuple[int, ...]:
    if not args.turn_levels:
        return TURN_LEVELS
    levels = []
    for chunk in str(args.turn_levels).replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            levels.append(int(chunk))
        except ValueError:
            raise SystemExit(f"✗ 轮次档位必须是整数，收到 {chunk!r}")
    if not levels:
        return TURN_LEVELS
    for level in levels:
        if not (1 <= level <= ABSOLUTE_MAX_TURNS):
            raise SystemExit(f"✗ 轮次档位必须在 1..{ABSOLUTE_MAX_TURNS}，收到 {level}")
    return tuple(sorted(set(levels)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = _settings_from_args(args)

    model_a = settings.llm.model
    model_b = args.model_b or _derive_model_b(model_a)
    if model_b is None:
        raise SystemExit(
            f"✗ 无法自动推导与 {model_a!r} 对照的模型名。请显式指定，例如：\n"
            f"    --model-a {model_a} --model-b <对照模型名>\n"
            "  （按名字猜模型名有风险：端点支持的名称未必是 `-pro` 这种形式，"
            "猜错会让整批运行在真实调用时全线失败。）"
        )
    if model_b == model_a:
        raise SystemExit("✗ --model-b 与 A 模型相同，轴②将测不出东西")

    thetas = parse_theta_spec(args.thetas)
    if len(thetas) < 2:
        raise SystemExit("✗ 至少需要 2 个 θ —— 只有 1 个时 ICC 无从谈起（没有组间方差）")
    replicates = args.replicates or (2 if args.mini else 5)
    levels = _turn_levels(args)
    conditions = build_conditions(
        model_a, model_b, replicates, levels=levels,
        reference_turns=args.turns, mini=args.mini,
    )
    limits: dict[str, int] = {}
    for pair in (args.theta_limit or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"✗ --theta-limit 需要 `条件id=数量` 形式，收到 {pair!r}")
        key, _, raw = pair.partition("=")
        limits[key.strip()] = int(raw)
    for condition in conditions:
        condition.n_thetas = len(thetas)
        if condition.condition_id in limits:
            limit = limits[condition.condition_id]
            if not 2 <= limit <= len(thetas):
                raise SystemExit(f"✗ --theta-limit {condition.condition_id}={limit} 越界"
                                 f"（需在 2~{len(thetas)} 之间）")
            condition.theta_limit = limit
            condition.n_thetas = limit
        if condition.axis != "④":          # 轴①②③ 统一走「基础轮 + 补问上限」协议
            condition.n_turns = args.turns
            condition.max_turns = args.max_turns
    conditions = _select_conditions(args, conditions)

    base_turns = args.turns
    base_max = args.max_turns
    text_limit = args.text_limit if args.text_limit is not None else DEFAULT_TEXT_LIMIT

    out_dir = Path(args.out) if args.out else (settings.data_dir / "runs" / "fig1")
    run_id = f"fig1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    total_subjects = sum(c.n_subjects for c in conditions)
    lower = sum(c.n_subjects * (c.n_turns * 2 + 1) for c in conditions)
    upper = sum(c.n_subjects * (c.max_turns * 2 + 1) for c in conditions)

    print("=" * 90)
    print("Fig.1 系统可靠性实验 · 批量调用 PACS 心智")
    print("=" * 90)
    print(settings.describe())
    print(f"  A 模型（脑/提问者/分析者默认） ：{model_a}")
    print(f"  B 模型（对照）                 ：{model_b}")
    print(f"  θ 清单（{len(thetas)} 组）              ："
          + "；".join(f"{t.theta_id} {t.describe(include_design=False)}" for t in thetas))
    print(f"  心智年龄                       ：{DEFAULT_AGE_MONTHS} 个月（固定常量，不可配置）")
    print(f"  轴①②③ 轮次协议             ：{base_turns} 基础轮 / 补问上限 {base_max}")
    print(f"  轴④ 轮次档位               ：{'/'.join(map(str, sorted(levels)))}（各自硬固定）")
    print(f"  题集                           ：每个 (seed, mode) {args.probe_pool} 条，全条件复用")
    analyst_ceiling = args.analyst_max_tokens or args.max_tokens or settings.llm.max_tokens
    print(f"  脑/提问者输出额度              ：{settings.llm.max_tokens} token"
          + ("　（--max-tokens 覆盖）" if args.max_tokens is not None
             else "　（来自 config.json）"))
    print(f"  分析者输出额度                 ：{analyst_ceiling} token"
          + ("　（单独指定——它输出最长，额度不足会把 JSON 截断）"
             if analyst_ceiling != settings.llm.max_tokens else "　（与上同）"))
    print(f"  被试总数                       ：{total_subjects}")
    print(f"  预计模型调用                   ：{lower}（基础轮）～ {upper}（上界）")
    print(f"  产物目录                       ：{out_dir}")
    print()

    print("条件表：")
    print(f"  {'条件':<26}{'轴':<4}{'脑':<18}{'提问者':<18}{'分析者':<18}{'seed':>5}{'轮次':>9}{'重复':>5}{'被试':>5}")
    for condition in conditions:
        print(f"  {condition.condition_id:<26}{condition.axis:<4}{condition.brain_model:<18}"
              f"{condition.examiner_model:<18}{condition.analyst_model:<18}{condition.seed:>5}"
              f"{condition.n_turns:>5}/{condition.max_turns:<3}{condition.replicates:>5}"
              f"{condition.n_subjects:>5}")
    print()

    if args.dry_run:
        print("（--dry-run：只打印矩阵与成本，不调用任何模型）")
        _dry_run_checks(settings, model_a, model_b)
        return 0

    if settings.llm.provider != "mock" and upper > CONFIRM_THRESHOLD and not args.yes:
        print(f"✗ 真实 provider 下调用数上界 {upper} 超过安全阈值 {CONFIRM_THRESHOLD}。"
              "\n  这会消耗实际额度。确认请加 --yes，或用 --conditions 先跑单条件、"
              "或用 --mini 缩微矩阵干跑。")
        return 2
    if settings.llm.provider != "mock" and not settings.brain_ready():
        print("✗ 模拟大脑未配置好（真实 provider 需要 api_key）。"
              "请检查 config.json，或设置 PACS_LLM_API_KEY。")
        return 2

    preflight_check(settings, [c.brain_model for c in conditions]
                             + [c.examiner_model for c in conditions])

    # 题集：按条件用到的 seed 各生成一份（保证同 seed 同题序，跨条件可比）
    seeds = sorted({c.seed for c in conditions})
    print(f"生成题集（seed={seeds}，每个 {args.probe_pool} 条）…")
    probe_pool = {
        seed: eb.dialogue_probes(goals=DIMS, n=args.probe_pool, seed=seed, mode=args.probe_mode)
        for seed in seeds
    }
    for seed, items in probe_pool.items():
        print(f"  seed={seed}: {len(items)} 条　前 3 条 {[i.get('key') for i in items[:3]]}")
    print()

    metrics: dict[str, dict] = {}
    condition_records: dict[str, list[dict]] = {}
    all_cell_failures: dict[str, list[dict]] = {}
    started = time.time()
    for condition in conditions:
        cond_dir = out_dir / condition.axis_dir / condition.condition_id
        if args.resume and _condition_complete(cond_dir, condition.n_subjects):
            print(f"▶ {condition.condition_id}：产物已齐全，跳过（--resume）")
            condition_records[condition.condition_id] = load_condition_records(cond_dir)
            metrics[condition.condition_id] = condition_metrics(
                condition_records[condition.condition_id])
            continue
        print(f"▶ {condition.condition_id} · {condition.axis_name} · {condition.label}")
        print(f"    {condition.describe()}")
        try:
            records, cell_failures = run_condition(
                condition, settings=settings, thetas=thetas, out_dir=out_dir,
                text_limit=text_limit, probe_pool=probe_pool,
                reuse_existing=args.resume,
                analyst_max_tokens=args.analyst_max_tokens or args.max_tokens,
            )
        except LLMError as exc:
            print(f"    ✗ 运行中断：{exc}")
            return 2
        except Exception as exc:  # noqa: BLE001 - 单条件出意外不该让整批白跑
            print(f"    ✗ {condition.condition_id} 中断（{type(exc).__name__}）：{str(exc)[:160]}")
            print("      跳过本条件继续后面的；跑完用 --resume 回来补齐即可")
            continue
        if cell_failures:
            # 单元格残缺会让下游静默丢目标，所以必须冒泡到最终报告，不能只留在本条件的日志里
            all_cell_failures[condition.condition_id] = cell_failures
        condition_records[condition.condition_id] = records
        metrics[condition.condition_id] = condition_metrics(records)
        cell = (metrics[condition.condition_id].get("cell_icc") or {})
        icc = (cell.get("icc") or {})
        print(f"    → cell ICC={_fmt(icc.get('icc'))}　"
              f"SEM={_fmt(cell.get('sem'))}　"
              f"档位 κ={_fmt((metrics[condition.condition_id].get('likeness_agreement') or {}).get('kappa'))}")
        print()

    # 补漏：某个条件若因瞬时端点故障缺格（单元格重试已用满），整轮跑完后再单独补一次。
    # 这是最后一道 —— 缺格会让下游丢掉整组目标，而报告里的 n 变小却看不出原因。
    for condition in conditions:
        done_ok = [r for r in (condition_records.get(condition.condition_id) or [])
                   if _has_readings(r)]
        if len(done_ok) >= condition.n_subjects:
            continue
        print(f"▶ 补漏 {condition.condition_id}：可用 {len(done_ok)}/{condition.n_subjects} 格，再补一次")
        try:
            records, gap_failures = run_condition(
                condition, settings=settings, thetas=thetas, out_dir=out_dir,
                text_limit=text_limit, probe_pool=probe_pool,
                reuse_existing=True,
                analyst_max_tokens=args.analyst_max_tokens or args.max_tokens,
            )
        except LLMError as exc:
            print(f"    ✗ 补漏中断：{exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"    ✗ 补漏中断（{type(exc).__name__}）：{str(exc)[:160]}")
            continue
        condition_records[condition.condition_id] = records
        metrics[condition.condition_id] = condition_metrics(records)
        if gap_failures:
            all_cell_failures[condition.condition_id] = gap_failures

    # 跨条件比较
    reference = next((c for c in conditions if c.reference), None)
    comparisons: dict[str, list[dict]] = {}
    if reference and reference.condition_id in condition_records:
        base_records = condition_records[reference.condition_id]
        for condition in conditions:
            if condition.condition_id == reference.condition_id:
                continue
            if condition.condition_id not in condition_records:
                continue
            item = compare_conditions(
                base_records, condition_records[condition.condition_id],
                reference.condition_id, condition.condition_id,
            )
            comparisons.setdefault(AXIS_KEYS.get(condition.axis, condition.axis), []).append(item)

    # 轴④ 相邻档位也互相比一次（看轮数增加是否收敛）
    level_conditions = sorted(
        (c for c in conditions if c.axis == "④" and c.condition_id in condition_records),
        key=lambda c: c.n_turns,
    )
    if reference and reference.condition_id in condition_records:
        level_conditions = [reference] + level_conditions
    # 参照臂与 L=25 档的对比上面已作为「参照 vs 该条件」加过一次，这里去重，
    # 否则报告里同一对条件会出现两行完全相同的数字。
    seen_pairs = {
        (item.get("base"), item.get("other"))
        for items in comparisons.values() for item in items
    }
    for left, right in zip(level_conditions, level_conditions[1:]):
        if left.condition_id == right.condition_id:
            continue
        if (left.condition_id, right.condition_id) in seen_pairs:
            continue
        item = compare_conditions(
            condition_records[left.condition_id], condition_records[right.condition_id],
            left.condition_id, right.condition_id,
        )
        comparisons.setdefault(AXIS_KEYS["④"], []).append(item)

    # 汇总产物
    theta_summary = "；".join(
        f"{t.theta_id}={t.describe(include_design=False)}" for t in thetas
    )
    all_records = [r for records in condition_records.values() for r in records]
    payload = {
        "run_id": run_id,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
        "out_dir": str(out_dir),
        "pacs_version": __version__,
        "model_version": DEFAULT_MODEL_VERSION,
        "age_months": DEFAULT_AGE_MONTHS,
        "base_turns": base_turns,
        "base_max_turns": base_max,
        "turn_levels": sorted(levels),
        "probe_pool": args.probe_pool,
        "theta_summary": theta_summary,
        "thetas": [
            {"theta_id": t.theta_id, "design": t.design,
             "active": {k: v for k, v in t.active_perturbations().items()},
             "describe": t.describe()}
            for t in thetas
        ],
        "model_a": model_a,
        "model_b": model_b,
        "conditions": [c.to_dict() for c in conditions],
        "n_subjects": total_subjects,
        "n_records": len(all_records),
        "cost": {"calls_lower_bound": lower, "calls_upper_bound": upper},
        "synthetic_any": any(
            bool((r.get("profile") or {}).get("synthetic")) for r in all_records
        ),
        "provider": settings.llm.provider,
        "metrics": metrics,
        "comparisons": comparisons,
        "cell_failures": all_cell_failures,
        "limitations": LIMITATIONS,
        "blind_note": ("提问者与分析者全程看不到 θ：分析者接口签名里没有 θ、提示词里也没有。"
                       "各被试 JSON 的 blind_audit 小节是事后记录，不参与任何读数计算。"),
        "仿真声明": SIMULATION_NOTICE,
        "用途声明": DISCLAIMER,
    }
    payload["panels"] = build_panels(conditions, metrics, comparisons)
    if args.no_report:
        # 并行拆分时每个进程只认识自己那批条件：这时写报告会互相覆盖且内容残缺，
        # 所以各进程只落测量产物，报告留到最后单独跑一次全量汇总。
        print("（--no-report：跳过报告与扁平数据集，稍后用不带该参数的 --resume 汇总）")
        print(f"  本进程完成的条件：{sorted(condition_records)}")
        return 0
    write_report(out_dir, payload)
    # 扁平数据集：供后续/其他分析直接接走（宽表 + 长表 + 数据字典）
    flat_info = write_flat_dataset(out_dir, all_records)
    payload["flat_dataset"] = flat_info
    # 把扁平数据集的信息补进 json（report 已写过一次，这里覆盖成含该字段的版本）
    _write_json(out_dir / "reliability" / "fig1_reliability.json", payload)

    print("=" * 90)
    print("完成")
    print("=" * 90)
    print(f"  被试记录：{len(all_records)}")
    print(f"  耗时    ：{payload['elapsed_seconds']}s")
    if all_cell_failures:
        lost = sum(len(round_failure["theta_ids"])
                   for rounds in all_cell_failures.values() for round_failure in rounds)
        print(f"  ⚠ 残缺单元格：{lost} 个（重试 {CELL_ATTEMPTS} 次仍失败）"
              f"，涉及 {len(all_cell_failures)} 个条件 —— 这些 (θ, 重复) 在可靠性统计里")
        print("    会因测量次数不足被丢弃，报告中的 n 已按实际入矩阵的目标数计算。")
    print(f"  产物目录：{out_dir}")
    print(f"    {out_dir / 'reliability' / 'fig1_reliability.md'}（人读）")
    print(f"    {out_dir / 'reliability' / 'fig1_reliability.json'}")
    print(f"    {out_dir / 'reliability' / 'fig1_panels.json'}（喂绘图脚本）")
    print(f"    {out_dir / 'flat' / 'all_subjects.csv'}（宽表，一行一被试）")
    print(f"    {out_dir / 'flat' / 'readings_long.csv'}（长表，一行一 被试×维度）")
    print(f"    {out_dir / 'flat' / '数据字典.md'}")
    print()
    print(SIMULATION_NOTICE)
    print(DISCLAIMER)
    return 0


def _condition_complete(cond_dir: Path, expected: int) -> bool:
    if not (cond_dir / "condition.json").exists():
        return False
    subjects = cond_dir / "subjects"
    if not subjects.exists():
        return False
    return len(list(subjects.glob("*.json"))) >= expected


def preflight_check(settings: Settings, models: Sequence[str]) -> None:
    """真实 provider 下先探一次每个模型名 —— 在开跑前把「模型名写错」这类错误挡掉。

    为什么值得单独做这一步：批量运行是**顺序**的，模型名写错时前几个条件可能已经跑了几小时，
    后面才因为 400 全线崩掉。一次几十 token 的探测就能避免。实测踩过：`deepseek-pro`
    在端点上不存在（正确名字是 `deepseek-v4-pro`），而按名字猜「flash→pro」恰好猜错。
    """
    if settings.llm.provider == "mock":
        return
    print("模型名预检（每个模型各一次极小调用）…")
    for model in dict.fromkeys(models):          # 去重保序
        provider = get_provider(replace(settings.llm, model=model))
        try:
            provider.chat([ChatMessage("user", "回复一个字：好")], max_tokens=1024)
        except Exception as exc:  # noqa: BLE001 - 要把原始错误原样呈现给用户
            raise SystemExit(
                f"✗ 模型 {model!r} 不可用：{type(exc).__name__}: {str(exc)[:200]}\n"
                "  请用 --model-a / --model-b 指定端点**实际支持**的模型名。"
            )
        print(f"  ✓ {model}")


def _dry_run_checks(settings: Settings, model_a: str, model_b: str) -> None:
    if settings.llm.provider == "mock":
        print("提示：provider=mock（规则式替身）。mock 完全确定性 → 可靠性必然≈1，"
              "只能验证管线，不能作为论文结论。")
    elif not settings.brain_ready():
        print("⚠ 真实 provider 但缺 api_key：正式跑之前请检查 config.json 或 PACS_LLM_API_KEY。")
    else:
        print(f"✓ 真实 provider 就绪：{settings.llm.provider} / {settings.llm.base_url}")
    print(f"  将使用的两个模型：{model_a}（A）与 {model_b}（B）")
    print("  运行前建议：先 `--conditions axis1_same_brain` 跑单条件，确认 ICC/κ 落在合理区间。")


if __name__ == "__main__":
    sys.exit(main())
