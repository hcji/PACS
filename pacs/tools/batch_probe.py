"""批量探针：一次跑多个「人工心智」，把每个被试的**参数 / 对话过程 / 分析结果**存成文件。

为什么要有这个脚本：`pacs run` 把剖面逐条写进 SQLite（给下游分析用），但对话全文、逐维细节、
结论原文只躺在库里 —— 想看「这个 θ 下心智到底说了什么、分析者凭什么这么判」就得翻数据库。
本脚本把一次批量运行的**全部产物**摊成可以直接阅读与统计的文件。

**它不自造任何测量逻辑**：建心智、提问者提问、分析者读数、逐维表型，全部走 agent 已有的
公开入口（`engine.run_design` → `measure_phenotype` → `run_dialogue` / `analyze_dialogue`），
与 `pacs run` 是同一条链路。因此 —— 改 agent（提示词、补问配额、解析规则、新增维度）后，
本脚本不改也能跟着变；它只做「编排 + 落盘」两件事，一行测量逻辑都不复制。

一次批量运行 = 一张 θ 设计表 × 每个 θ 一个虚拟被试：

    1. 建心智   VirtualAgent(provider, θ) 并 initialize()（`SELF_STATE: OK` 是可审计的「上线」）
    2. 设参数   θ 由 `brain.perturb` 的采样器生成（与 `pacs run` 同一套设计模式）
    3. 分析探查 `run_dialogue` 提问者 LLM 提问 → 心智作答 → `analyze_dialogue` 分析者 LLM 逐维读数
    4. 保存结果 参数 θ / 对话逐轮 / 逐维数值与结论 / 表型剖面 → JSON + CSV + Markdown

产物（默认 `<--out>` = `data/runs/probe_<run_id>/`）：

    manifest.json       运行级参数：模型、轮数与字数上限、条目库清单、完整 θ 设计表、成本上界
    subjects/NNN_<θ>.json   单个被试：参数 + 对话过程 + 分析结果 + 剖面 + 盲法审计
    subjects/NNN_<θ>.md     同上，人读版（对话全文 + 结论）
    index.csv           行 = 被试，列 = θ 摘要 + 各维度数值 + 结论（Excel 直接打开）
    summary.md          全部被试一览、结论全文、失败与协议问题

用法：

    # 1) 先看设计表与成本预估（不调用任何模型）
    python tools/batch_probe.py --design single --doses 1.0,0.8,0.6 --dry-run

    # 2) 离线跑通全链路（Mock 心智，无需 Key）—— 验证脚本本身
    python tools/batch_probe.py --provider mock --design single --limit 2

    # 3) 真实 LLM 批量（BYOK：推荐用环境变量 PACS_LLM_API_KEY / PACS_EXAMINER_LLM_API_KEY）
    python tools/batch_probe.py --design single --doses 1.0,0.8,0.6 --limit 20 --yes

    # 4) 单个 θ 定点复查
    python tools/batch_probe.py --primitive salience_assignment --dose 0.2 --yes

注意：Mock 是**规则式前向模型**，它的表型结构由 θ 与题目敏感度算出 —— 用来验证管线可以，
不能当成 LLM 涌现的证据（产物里 `provider_kind` / `synthetic` 会如实标出来）。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

# 允许直接 `python tools/batch_probe.py` 运行（无需先 pip install / 设 PYTHONPATH）。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pacs import (  # noqa: E402
    DEFAULT_AGE_MONTHS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_TEXT_LIMIT,
    QUESTIONS_PER_CHANNEL,
    SIMULATION_NOTICE,
    __version__,
)
from pacs.agents.dialogue import (  # noqa: E402
    ABSOLUTE_MAX_TURNS,
    DEFAULT_N_TURNS,
    MAX_N_TURNS,
)
from pacs.brain import examiner_bank as eb  # noqa: E402
from pacs.brain.perturb import (  # noqa: E402
    DEFAULT_DOSE_LEVELS,
    Perturbation,
    generic_degradation_control,
    intact,
    sample_design,
    single,
)
from pacs.config import PROJECT_ROOT, Settings  # noqa: E402
from pacs.engine import new_run_id, run_design  # noqa: E402
from pacs.llm.provider import LLMError, get_examiner_provider, get_provider  # noqa: E402
from pacs.phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS  # noqa: E402
from pacs.phenotype.conclusion import ASD_LIKENESS_LEVEL_ZH  # noqa: E402

# 与 `pacs run` 同口径的护栏：真实 provider 下调用数（上界）超过它就要求 --yes。
CONFIRM_THRESHOLD = 200

#: 分析者的观察提纲条数 —— 与 `cmd_run` 一致（`observation_rubric(goals=..., limit=8)`）。
RUBRIC_LIMIT = 8

DESIGN_MODES = ("single", "pair", "lhs", "intact", "generic", "all")

DISCLAIMER = "用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。"


# ---------------------------------------------------------------------------
# 序列化与格式化小工具（只管装盘，不碰任何测量口径）
# ---------------------------------------------------------------------------


def _jsonable(value):
    """把任意结构收敛成可 JSON 序列化的形式（防御性：分析者输出是 LLM 给的，字段不定）。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload) -> None:
    """写 JSON —— 中文不转义、缩进 2（与项目其余产物的落盘格式一致）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _r(value, digits: int = 4):
    """读数取整到固定小数位（None 保持 None，别把「没读数」写成 0）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def _plain(value, default: str = "") -> str:
    """取一个「可能是 None」的字符串字段（会话缺失时兜底成空串，而不是写成 'None'）。"""
    return default if value is None else str(value)


def _seq(value) -> list:
    """取一个「可能是 None」的序列字段。"""
    return list(value or [])


def _one_line(text, limit: int = 160) -> str:
    """把长文本压成表格里能放的一行。"""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _likeness_phrase(likeness: dict) -> str:
    """把结构化表型印象写成一句话（不替分析者下结论）。"""
    likeness = likeness or {}
    present = likeness.get("present")
    level = str(likeness.get("level") or "unknown")
    level_zh = ASD_LIKENESS_LEVEL_ZH.get(level, level)
    if present is None:
        return f"未给出（程度档位：{level_zh}）"
    verdict = "表现出孤独症样表型特征" if present else "未表现出孤独症样表型特征"
    return f"{verdict} · 程度{level_zh}"


def _likeness_short(likeness: dict) -> str:
    """进度行用的缩写版（整数表、进度行都放得下）。"""
    likeness = likeness or {}
    if likeness.get("present") is None:
        return "未给出"
    return ("有孤独症样特征 · " if likeness.get("present") else "无孤独症样特征 · ") + _plain(
        ASD_LIKENESS_LEVEL_ZH.get(str(likeness.get("level") or ""), likeness.get("level") or "")
    )


def _threshold(value, default=0.0):
    """CLI 传来的可缺省数值阈值。"""
    return default if value is None else value


def _doses(raw) -> list[float]:
    """解析剂量档位：接受 `1.0,0.8,0.6`（逗号分隔）与 `1.0 0.8 0.6`（空格分隔）。

    完整度 1.0 等于「不削弱」，采样器会把它当成与基线重复而跳过，因此这里显式剔除并说明，
    免得看到「档位写了 3 个、被试却只有 2 档」时以为是脚本出错。
    """
    if not raw:
        return list(DEFAULT_DOSE_LEVELS)
    values: list[float] = []
    for chunk in str(raw).replace("，", ",").replace(",", " ").split():
        try:
            values.append(float(chunk))
        except ValueError:
            raise SystemExit(f"✗ --doses 只能是数字（逗号或空格分隔），收到 {chunk!r}")
    if not values:
        return list(DEFAULT_DOSE_LEVELS)
    for value in values:
        if not (0.0 < value <= 1.0):
            raise SystemExit(f"✗ 剂量必须在 (0, 1] 之间（1.0 = 不削弱），收到 {value}")
    kept = [v for v in values if v < 0.999]
    if not kept:
        raise SystemExit(
            "✗ 所有剂量都是 1.0（= 不削弱，等价于完整心智基线）。"
            "请给一个小于 1.0 的档位（如 0.2/0.4/0.6），或用 --design intact 直接跑基线。"
        )
    if len(kept) != len(values):
        print(f"（已剔除此档位 1.0：完整度 1.0 = 不削弱，与完整心智基线重复）")
    return kept


# ---------------------------------------------------------------------------
# 设计表：θ 网格（采样器全部来自 agent，脚本不自己造 θ）
# ---------------------------------------------------------------------------


def build_designs(args: argparse.Namespace) -> list[Perturbation]:
    """按 `--design` 生成 θ 设计表。与 `cmd_run` 用同一批采样器，口径完全一致。"""
    if args.primitive:
        return [single(args.primitive, _threshold(args.dose, 0.2), seed=args.seed)]

    mode = args.design
    doses = tuple(_doses(args.doses))
    if mode == "all":
        # 与 cmd_run --mode all 同款拼装：单因素 + 组合 + 通用退化对照（Control 1）
        return (
            sample_design("single", doses=doses, seed=args.seed)
            + sample_design("pair", pair_dose=_threshold(args.dose, 0.2),
                            seed=args.seed, max_pairs=args.max_pairs)
            + [generic_degradation_control(0.5, seed=args.seed + rep) for rep in range(5)]
        )
    if mode == "generic":
        return [generic_degradation_control(_threshold(args.dose, 0.5), seed=args.seed)]
    return sample_design(
        mode=mode,
        doses=doses,
        seed=args.seed,
        n_samples=args.n_samples,
        pair_dose=_threshold(args.dose, 0.2),
        max_pairs=args.max_pairs,
    )


def apply_limit(designs: list[Perturbation], limit: int | None) -> list[Perturbation]:
    """按上限裁剪设计，但**保持代表性**（与 `cli._apply_limit` 同一个出发点）。

    为什么不能直接 `designs[:limit]`：设计表的前段全是完整心智基线的重复（默认 5 次），
    硬截断会让 `--limit 3` 变成「跑 3 遍基线」—— 既得不到任何扰动读数，也白花额度。

    这里的取舍是：基线重复压到 1 次、通用退化对照压到 1 次（两者是全部比较的锚点，
    必须各留一个），剩下的预算按确定性等距抽样分给扰动网格。`--limit < 3` 时会如实提示
    「这一批里没有扰动被试」。
    """
    if not limit or limit >= len(designs):
        return designs

    intacts = [d for d in designs if d.is_intact]
    controls = [d for d in designs if not d.is_intact and d.generic_degradation > 0]
    rest = [d for d in designs if not d.is_intact and d.generic_degradation <= 0]

    keep: list[Perturbation] = intacts[:1] + controls[:1]
    budget = max(0, limit - len(keep))
    if budget == 0:
        kept = intacts[:1][:limit] or rest[:limit] or controls[:1][:limit]
        print(f"  ⚠ --limit {limit} 小于「1 个基线 + 1 个通用退化对照」，"
              f"本批只有 {len(kept)} 个被试，且不含任何扰动被试。")
        return kept
    if len(rest) <= budget:
        return keep + rest

    stride = len(rest) / budget
    sampled: list[Perturbation] = []
    seen: set[str] = set()
    for i in range(budget):
        candidate = rest[min(len(rest) - 1, int(i * stride))]
        if candidate.theta_id not in seen:
            seen.add(candidate.theta_id)
            sampled.append(candidate)
    return keep + sampled


# ---------------------------------------------------------------------------
# 运行级参数（进 manifest.json）
# ---------------------------------------------------------------------------


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """按 CLI 参数覆盖 config.json —— 与 `cli._build_settings` 同一套优先级。"""
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
    # 提问者 / 分析者：只覆盖显式给出的项，其余跟随模拟大脑（与 `cli` / 界面一致）
    if args.examiner_provider:
        settings.examiner.provider = args.examiner_provider
    if args.examiner_model:
        settings.examiner.model = args.examiner_model
    if args.examiner_base_url:
        settings.examiner.base_url = args.examiner_base_url
    if args.examiner_api_key:
        settings.examiner.api_key = args.examiner_api_key
    if args.same_provider_for_all_roles:
        # 三方同模型：提问者 / 分析者直接复用模拟大脑那套配置
        settings.examiner = settings.llm
    elif args.provider == "mock" and not any(
        (args.examiner_provider, args.examiner_model, args.examiner_base_url, args.examiner_api_key)
    ):
        # `--provider mock` 是一句「这次不调模型」的承诺：若提问者/分析者仍指向 config.json 里
        # 的真实端点，整批对话会真的打到付费 API —— 与「只是想离线试跑」的意图正好相反。
        # 因此显式给出 mock 且没单独配第二套时，让三方都走 mock（要混搭就显式给 --examiner-provider）。
        settings.examiner.provider = "mock"
    settings.ensure_dirs()
    return settings


def _provider_descriptor(settings, kind: str) -> dict:
    """记录**用哪套 provider 跑的**。Key 一律不落盘，只写「配没配」。"""
    if kind == "examiner":
        # `examiner_effective()` 会处理「同端点缺 Key 时回落模拟大脑的 Key」
        eff = settings.examiner_effective()
        return {
            "role": "提问者 / 分析者",
            "provider": eff.provider,
            "model": eff.model,
            "base_url": eff.base_url,
            "temperature": eff.temperature,
            "api_key_configured": bool(eff.api_key),
        }
    return {
        "role": "模拟大脑（被扰动的心智）",
        "provider": settings.llm.provider,
        "model": settings.llm.model,
        "base_url": settings.llm.base_url,
        "temperature": settings.llm.temperature,
        "api_key_configured": bool(settings.llm.api_key),
    }


def _design_payload(design: Perturbation, index: int) -> dict:
    """θ 的完整记录 —— 这是「设置的参数」区块。"""
    return {
        "index": index,
        "theta_id": design.theta_id,
        "design": design.design,
        "origin": list(design.origin),
        "seed": design.seed,
        "note": design.note,
        "is_intact": design.is_intact,
        "generic_degradation": round(float(design.generic_degradation), 4),
        # 被削弱的 primitive 及其完整度（只列 < 1.0 的）
        "active": {k: round(float(v), 4) for k, v in design.active_perturbations().items()},
        "values": {k: round(float(v), 4) for k, v in design.values.items()},
    }


def _manifest(
    *,
    run_id: str,
    settings: Settings,
    args: argparse.Namespace,
    designs: list[Perturbation],
    probes: list[dict],
    rubric: list[dict],
    turns: int,
    max_turns: int,
    text_limit: int,
    model_id: str,
    out_dir: Path,
    started: str,
    finished: str,
    elapsed: float,
    run_payload: dict,
    subjects: list[dict],
) -> dict:
    """运行级参数：把「这批是怎么跑出来的」完整写下来，使产物离开本机也能被读懂与复现。"""
    base_calls = len(designs) * (turns * 2 + 1)
    total_calls = len(designs) * (max_turns * 2 + 1)
    failed = {f[0] for f in run_payload.get("failures") or []}
    return {
        "脚本": "tools/batch_probe.py",
        "说明": "批量调用 PACS 心智：建心智 → 设参数 → 分析探查 → 保存结果",
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": round(elapsed, 1),
        "out_dir": str(out_dir),
        "pacs_version": __version__,
        "model_version": DEFAULT_MODEL_VERSION,
        "age_months": DEFAULT_AGE_MONTHS,
        "age_months_note": "心智年龄是固定常量，不可配置（参照线随参数漂移读数就不可比）",
        "instrument": run_payload.get("instrument", "dialogue"),
        "brain_model_id": model_id,
        "run_parameters": {
            "n_turns": turns,
            "n_turns_note": f"基础轮数（默认 {DEFAULT_N_TURNS} = 计分通道数 × 每通道 {QUESTIONS_PER_CHANNEL} 个问题）",
            "max_turns": max_turns,
            "max_turns_note": f"轮数上限，含按配额自动补问的轮次（默认 {MAX_N_TURNS}，硬顶 {ABSOLUTE_MAX_TURNS}）",
            "text_limit": text_limit,
            "text_limit_note": "问题与回答的字数上限；超出只告警、不截断",
            "probe_mode": args.probe_mode,
            "seed": args.seed,
            "design_mode": args.design,
            "doses": _doses(args.doses),
            "dose": args.dose,
            "limit": args.limit,
            "initialize": not args.no_init,
            "same_provider_for_all_roles": bool(args.same_provider_for_all_roles),
            "goals": list(SCORED_CHANNELS),
            "control_channel": [ch for ch in SCORED_CHANNELS if ch not in PHENOTYPE_DIMS],
        },
        "llm": {
            "brain": _provider_descriptor(settings, "brain"),
            "examiner": _provider_descriptor(settings, "examiner"),
        },
        "item_bank": {
            "n_probes": len(probes),
            "probe_keys": [i.get("key", "") for i in probes],
            "probe_titles": [i.get("source_name", "") for i in probes],
            "n_rubric": len(rubric),
            "rubric_keys": [i.get("key", "") for i in rubric],
        },
        "cost_estimate": {
            "subjects": len(designs),
            "calls_lower_bound": base_calls,
            "calls_upper_bound": total_calls,
            "formula": "被试数 × (轮数×2 + 1)：每轮一次提问 + 一次作答，全批一次分析者读数",
        },
        "n_designs": len(designs),
        "n_subjects_completed": len(run_payload.get("profiles") or []),
        "n_subjects_failed": len(failed),
        "designs": [_design_payload(d, i) for i, d in enumerate(designs, start=1)],
        "subjects": subjects,
        "failures": [list(f) for f in run_payload.get("failures") or []],
        "盲法声明": (
            "提问者与分析者全程看不到 θ：分析者接口签名里没有 θ，提示词里也没有。"
            "各被试 JSON 的「盲法审计」小节是**事后**记录，不参与任何读数计算。"
        ),
        "仿真声明": SIMULATION_NOTICE,
        "用途声明": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# 单个被试：把「参数 + 对话 + 分析结果」装成一条记录
# ---------------------------------------------------------------------------


def _dimensions(feedback: dict, goals: tuple[str, ...]) -> dict:
    """逐维读数：数值 + 该维的细节（观察证据 / 任务测量 / 偏离量 / 置性度）。

    `dim_details` 由分析者 LLM 给出，字段不保证齐全，因此这里逐字段兜底。
    """
    readings = feedback.get("dims") or {}
    details = feedback.get("dim_details") or {}
    out: dict[str, dict] = {}
    for dim in goals:
        detail = details.get(dim) or {}
        score = detail.get("score")
        typicality = readings.get(dim)
        if typicality is None:
            typicality = detail.get("typicality")
        if score is None and isinstance(typicality, (int, float)):
            score = round(float(typicality) * 100, 1)
        out[dim] = {
            "label": SCORED_CHANNELS.get(dim, dim),
            "control": dim not in PHENOTYPE_DIMS,
            "score": score,
            "typicality": _r(typicality),
            "deviation": _r(detail.get("deviation"), 1),
            "confidence": _r(detail.get("confidence")),
            "observations": list(detail.get("observations") or []),
            "task_measurements": list(detail.get("task_measurements") or []),
        }
    return out


def build_subject_record(
    *,
    run_id: str,
    model_id: str,
    index: int,
    session_id: str,
    perturbation: Perturbation,
    session,
    feedback: dict,
    profile,
    goals: tuple[str, ...],
    run_parameters: dict,
    provider_kind: str,
    synthetic: bool,
) -> dict:
    """一个被试的全部产物：参数 θ / 对话过程 / 分析结果（各维数值 + 结论）/ 剖面。

    分区是刻意的：`parameters` 与 `blind_audit` 里才有 θ，`analysis` / `transcript` 里
    一律没有 —— 免得下游把 ground truth 与读数混在一起做（那就不是盲法读数了）。

    `session` 允许为 None（只在「回调次序意外」的兜底路径上发生）：那种情况下表型与参数
    照常保存，对话区块如实留空，并记一条协议问题 —— 不为了字段好看去编造对话。
    """
    dimensions = _dimensions(feedback, goals)
    session_note = "" if session is not None else "本次回调没有带来会话记录，对话区块为空"
    return {
        "run_id": run_id,
        "subject_index": index,
        "session_id": session_id or (session.session_id if session is not None else ""),
        "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "brain_model_id": model_id,
        "provider_kind": provider_kind,
        "synthetic": synthetic,
        "problems": ([session_note] if session_note else []),

        # ---- ① 设置的参数 ----
        "parameters": _design_payload(perturbation, index),
        "run_parameters": run_parameters,

        # ---- ② 对话过程（提问者 LLM ↔ 被扰动的心智）----
        #: `session.theta_id` 是 agent 自己写进会话的**不透明指纹**（仅用于把这段对话与
        #: `parameters` 区块对上），不是 θ 内容；提问者与分析者从未见过它。
        "dialogue": {
            "session": {
                "session_id": _plain(getattr(session, "session_id", "")),
                "theta_id": _plain(getattr(session, "theta_id", ""), perturbation.theta_id),
                "brain_model": _plain(getattr(session, "brain_model", "")),
                "examiner_model": _plain(getattr(session, "examiner_model", "")),
                "age_months": getattr(session, "age_months", DEFAULT_AGE_MONTHS),
                "n_turns": getattr(session, "n_turns", 0),
                "max_turns": getattr(session, "max_turns", 0),
                "n_turns_run": getattr(session, "n_turns_run", 0),
                "n_brain_turns": getattr(session, "n_brain_turns", 0),
                "seed": getattr(session, "seed", 0),
                "text_limit": getattr(session, "text_limit", 0),
                "mode": _plain(getattr(session, "mode", "")),
                "probe_mode": _plain(getattr(session, "probe_mode", "")),
                "probe_keys": _seq(getattr(session, "probe_keys", None)),
                "probe_sources": _seq(getattr(session, "probe_sources", None)),
                "probe_dims": _seq(getattr(session, "probe_dims", None)),
                "probe_primitives": _seq(getattr(session, "probe_primitives", None)),
                "focus_channels": _seq(getattr(session, "focus_channels", None)),
                "question_counts": dict(getattr(session, "question_counts", None) or {}),
                "uncovered_channels": _seq(getattr(session, "uncovered_channels", None)),
                "under_quota_channels": _seq(getattr(session, "under_quota_channels", None)),
                "over_limit_turns": getattr(session, "over_limit_turns", 0),
                "ts": _plain(getattr(session, "ts", "")),
            },
            "turns": [
                {
                    "turn": t.get("turn"),
                    "role": t.get("role"),
                    "role_zh": "提问者" if t.get("role") == "examiner" else "心智",
                    "content": t.get("content"),
                    "focus": t.get("focus") or "",
                }
                for t in (getattr(session, "turns", None) or [])
            ],
            "n_turns": len(getattr(session, "turns", None) or []),
            "problems": _seq(getattr(session, "problems", None)),
        },

        # ---- ③ 分析结果（分析者 LLM，盲法）----
        "analysis": {
            "analyst_model": feedback.get("model_id", ""),
            "dimensions": dimensions,
            "readings": {dim: dimensions[dim]["typicality"] for dim in goals},
            "mean_reading": _r(feedback.get("mean_reading")),
            "confidence": _r(feedback.get("confidence")),
            "conclusion": feedback.get("conclusion") or "",
            "conclusion_chars": feedback.get("conclusion_chars"),
            "conclusion_target_chars": feedback.get("conclusion_target_chars"),
            "conclusion_notes": list(feedback.get("conclusion_notes") or []),
            "asd_likeness": feedback.get("asd_likeness") or {},
            "narrative": feedback.get("narrative") or "",
            "evidence": list(feedback.get("evidence") or []),
            "notes": feedback.get("notes") or "",
            "parse_error": feedback.get("error") or "",
            "parse_ok": not bool(feedback.get("error")),
            "question_counts": dict(feedback.get("question_counts") or {}),
            "under_quota_channels": list(feedback.get("under_quota_channels") or []),
            "raw_output": feedback.get("raw") or "",
        },

        # ---- ④ 表型剖面（θ → B）----
        "profile": profile.to_dict() if hasattr(profile, "to_dict") else _jsonable(profile),

        # ---- ⑤ 盲法审计（事后追溯用，不参与读数）----
        "blind_audit": {
            "note": "这里是分析者**看不到**的 ground truth（被削弱的认知过程），仅供事后追溯。",
            "active_perturbations": {
                k: round(float(v), 4) for k, v in perturbation.active_perturbations().items()
            },
            "generic_degradation": round(float(perturbation.generic_degradation), 4),
            "design": perturbation.design,
            "theta_id": perturbation.theta_id,
            "describe": perturbation.describe(),
        },
    }


def render_subject_md(record: dict) -> str:
    """单个被试的人读版：参数 → 逐轮对话 → 各维数值 → 结论。"""
    params = record["parameters"]
    analysis = record["analysis"]
    session = record["dialogue"]["session"]
    audit = record["blind_audit"]
    lines: list[str] = []
    add = lines.append

    add(f"# 被试 {record['subject_index']:03d} · {params['theta_id']}")
    add("")
    add(f"- 运行号：`{record['run_id']}`　会话号：`{record['session_id']}`")
    add(f"- 保存时间：{record['saved_at']}　模型：`{record['brain_model_id']}`")
    add(f"- 心智年龄：{session['age_months']} 个月（固定常量）")
    add(f"- 读数来源：`{record['provider_kind']}`"
        + ("（**合成剖面**：规则式 mock 构造，不是 LLM 涌现）" if record["synthetic"] else ""))
    add("")

    add("## ① 设置的参数")
    add("")
    add(f"- 扰动描述：{params['describe'] if 'describe' in params else audit['describe']}")
    add(f"- 设计：`{params['design']}`　θ 指纹：`{params['theta_id']}`　种子：{params['seed']}")
    add(f"- 通用退化：{_fmt(params['generic_degradation'])}")
    if params["active"]:
        add("")
        add("| 被削弱的认知过程 | 完整度 |")
        add("| --- | --- |")
        for name, value in params["active"].items():
            add(f"| {name} | {_fmt(value)} |")
    else:
        add("")
        add("（完整心智：所有认知过程均为 1.0，无扰动）")
    add("")
    add(f"- 基础轮数 {session['n_turns']} / 上限 {session['max_turns']} / 实际 "
        f"{session['n_turns_run']} 轮（心智作答 {session['n_brain_turns']} 次）")
    add(f"- 字数上限 {session['text_limit']} 字/轮　探查方式 `{session['probe_mode']}`　"
        f"用量库条目 {len(session['probe_keys'])} 条")
    counts = session["question_counts"]
    if counts:
        add("- 逐通道问题数：" + "、".join(
            f"{SCORED_CHANNELS.get(k, k)} {v}" for k, v in counts.items()
        ))
    if session["uncovered_channels"]:
        add("- ⚠ 未被探查的通道：" + "、".join(
            SCORED_CHANNELS.get(c, c) for c in session["uncovered_channels"]
        ))
    if session["under_quota_channels"]:
        add("- ⚠ 问题数不足配额的通道：" + "、".join(
            SCORED_CHANNELS.get(c, c) for c in session["under_quota_channels"]
        ))
    add("")

    add("## ② 对话过程")
    add("")
    add(f"提问者：`{session['examiner_model']}`　心智：`{session['brain_model']}`")
    add("")
    for turn in record["dialogue"]["turns"]:
        focus = f"　〔{SCORED_CHANNELS.get(turn['focus'], turn['focus'])}〕" if turn["focus"] else ""
        add(f"**第 {turn['turn']} 轮 · {turn['role_zh']}**{focus}")
        add("")
        add(f"> {turn['content']}")
        add("")

    add("## ③ 分析结果（分析者 LLM，盲法）")
    add("")
    add(f"分析者：`{analysis['analyst_model']}`　平均读数：{_fmt(analysis['mean_reading'])}　"
        f"置性度：{_fmt(analysis['confidence'])}")
    add("")
    add("| 维度 | 分数(0-100) | 典型性(0-1) | 偏离 | 置性度 | 观察证据 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for dim, item in analysis["dimensions"].items():
        mark = "（对照通道，不进雷达图）" if item["control"] else ""
        evidence = "；".join(item["observations"][:3]) or "—"
        add(f"| {item['label']}{mark} | {_fmt(item['score'], 1)} | {_fmt(item['typicality'])} | "
            f"{_fmt(item['deviation'], 1)} | {_fmt(item['confidence'])} | {_one_line(evidence, 90)} |")
    add("")
    tasks: list[str] = []
    for dim, item in analysis["dimensions"].items():
        for measurement in item["task_measurements"]:
            tasks.append(
                f"- [{item['label']}] {measurement.get('task', '')} → {measurement.get('measurement', '')}"
            )
    if tasks:
        add("**任务级测量**")
        add("")
        lines.extend(tasks)
        add("")

    add(f"**表型印象**：{_likeness_phrase(analysis['asd_likeness'])}")
    add("")
    add("**探查结论（病例报告式总结）**")
    add("")
    add("> " + (analysis["conclusion"] or "（分析者没有给出结论）").replace("\n", "\n> "))
    add("")
    add(f"（{analysis['conclusion_chars'] or 0} 字 / 约 {analysis['conclusion_target_chars'] or '—'} 字）")
    for note in analysis["conclusion_notes"]:
        add(f"- ⚠ 结论提示：{note}")
    if analysis["narrative"]:
        add("")
        add(f"**叙述**：{analysis['narrative']}")
    if analysis["notes"]:
        add("")
        add("**分析者备注**")
        add("")
        for line in str(analysis["notes"]).splitlines():
            if line.strip():
                add(f"- {line}")
    if analysis["parse_error"]:
        add("")
        add(f"⚠ 分析者输出解析失败：{analysis['parse_error']}")
    if record["problems"]:
        add("")
        add("**协议问题**")
        add("")
        for problem in record["problems"]:
            add(f"- {problem}")

    add("")
    add("## ④ 表型剖面（θ → B）")
    add("")
    profile = record.get("profile") or {}
    if not profile:
        add("（这一版还没有剖面：本文件在「对话完成、剖面未回」的瞬间落盘过一版，"
            "剖面到齐后会被覆盖成完整版。若一直是这一版，说明该被试没产出剖面。）")
    else:
        add(f"- 对话完成率：{_fmt(profile.get('parse_rate'))}"
            f"（{profile.get('n_parsed')}/{profile.get('n_items')} 轮）")
        add(f"- 承载来源：`{profile.get('provider_kind')}`　合成：{_fmt(profile.get('synthetic'))}")
        add("")
        add("```json")
        add(json.dumps(_jsonable(profile), ensure_ascii=False, indent=2))
        add("```")

    add("")
    add("## ⑤ 盲法审计（分析者看不到的 ground truth）")
    add("")
    add(f"{audit['note']}")
    add("")
    add(f"- 扰动：{audit['describe']}")
    add(f"- 设计：`{audit['design']}`　θ：`{audit['theta_id']}`　"
        f"通用退化：{_fmt(audit['generic_degradation'])}")
    if audit["active_perturbations"]:
        add("- 被削弱的过程：" + "、".join(
            f"{k}={v:.2f}" for k, v in audit["active_perturbations"].items()
        ))
    else:
        add("- 被削弱的过程：（无，完整心智）")
    add("")
    add("---")
    add("")
    add(SIMULATION_NOTICE)
    add("")
    add(DISCLAIMER)
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 汇总产物：index.csv / summary.md
# ---------------------------------------------------------------------------


def _index_columns() -> list[str]:
    base = [
        "subject_index", "theta_id", "design", "generic_degradation", "active_perturbations",
        "primitive_count", "describe",
    ]
    return base + list(SCORED_CHANNELS) + [
        "mean_reading", "confidence", "parse_rate", "n_turns_run", "n_brain_turns",
        "uncovered_channels", "asd_likeness", "provider_kind", "synthetic",
        "conclusion_chars", "conclusion", "session_id", "problems",
    ]


def _index_row(record: dict) -> dict:
    params = record["parameters"]
    analysis = record["analysis"]
    session = record["dialogue"]["session"]
    profile = record["profile"]
    row = {
        "subject_index": record["subject_index"],
        "theta_id": params["theta_id"],
        "design": params["design"],
        "generic_degradation": params["generic_degradation"],
        "active_perturbations": "；".join(
            f"{k}={v:.2f}" for k, v in params["active"].items()
        ) or "（完整心智）",
        "primitive_count": len(params["active"]),
        "describe": record["blind_audit"]["describe"],
        "mean_reading": analysis["mean_reading"],
        "confidence": analysis["confidence"],
        "parse_rate": profile.get("parse_rate"),
        "n_turns_run": session["n_turns_run"],
        "n_brain_turns": session["n_brain_turns"],
        "uncovered_channels": "；".join(
            SCORED_CHANNELS.get(c, c) for c in session["uncovered_channels"]
        ),
        "asd_likeness": _likeness_phrase(analysis["asd_likeness"]),
        "provider_kind": record["provider_kind"],
        "synthetic": record["synthetic"],
        "conclusion_chars": analysis["conclusion_chars"],
        "conclusion": _one_line(analysis["conclusion"], 300),
        "session_id": record["session_id"],
        "problems": _one_line("；".join(record["problems"]), 200),
    }
    for dim in SCORED_CHANNELS:
        row[dim] = (analysis["readings"] or {}).get(dim)
    return row


def write_index_csv(path: Path, records: list[dict]) -> None:
    """行 = 被试，列 = 参数摘要 + 各维度数值 + 结论。

    用 `utf-8-sig` 落盘：Excel 双击打开中文不乱码（普通 utf-8 在中文 Windows 上会花）。
    """
    columns = _index_columns()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(_index_row(record))


def write_summary_md(path: Path, manifest: dict, records: list[dict]) -> None:
    """全部被试一览 + 各被试结论全文 + 失败与协议问题。"""
    lines: list[str] = []
    add = lines.append
    subjects = manifest["subjects"]

    add(f"# 批量探针汇总 · {manifest['run_id']}")
    add("")
    add(f"- 运行时间：{manifest['started_at']} → {manifest['finished_at']}"
        f"（{manifest['elapsed_seconds']} 秒）")
    add(f"- 心智模型：`{manifest['brain_model_id']}`　"
        f"提问者/分析者：`{manifest['llm']['examiner']['model']}`"
        f"（{manifest['llm']['examiner']['provider']}）")
    add(f"- 心智年龄：{manifest['age_months']} 个月（固定常量）")
    add(f"- 设计：`{manifest['run_parameters']['design_mode']}`　"
        f"被试数：{manifest['n_designs']}　完成：{manifest['n_subjects_completed']}　"
        f"失败：{manifest['n_subjects_failed']}")
    add(f"- 轮数：基础 {manifest['run_parameters']['n_turns']} / 上限 "
        f"{manifest['run_parameters']['max_turns']}　字数上限："
        f"{manifest['run_parameters']['text_limit']}")
    add(f"- 预计模型调用：{manifest['cost_estimate']['calls_lower_bound']}"
        f"（基础轮）～ {manifest['cost_estimate']['calls_upper_bound']}（补问至上限）")
    add(f"- 产物目录：`{manifest['out_dir']}`")
    add("")
    add(f"> {manifest['仿真声明']}")
    add(">")
    add(f"> {manifest['用途声明']}")
    add("")

    add("## 1. 被试一览")
    add("")
    add("| # | θ 指纹 | 扰动（盲法审计） | 平均读数 | 表型印象 | 结论字数 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for subject in subjects:
        add(f"| {subject['subject_index']:03d} | `{subject['theta_id']}` | "
            f"{_one_line(subject['describe'], 60)} | {_fmt(subject['mean_reading'])} | "
            f"{_one_line(subject['asd_likeness'], 40)} | {subject['conclusion_chars'] or 0} |")
    add("")

    add("## 2. 各维度读数")
    add("")
    add("方向约定：**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**；"
        "`general_cognition` 是对照通道（不进雷达图）。")
    add("")
    header = "| # | " + " | ".join(
        f"{SCORED_CHANNELS[d]}" + ("（对照通道）" if d not in PHENOTYPE_DIMS else "")
        for d in SCORED_CHANNELS
    ) + " | 平均 |"
    add(header)
    add("| --- |" + " --- |" * (len(SCORED_CHANNELS) + 1))
    for subject in subjects:
        cells = " | ".join(_fmt(subject["readings"].get(d)) for d in SCORED_CHANNELS)
        add(f"| {subject['subject_index']:03d} | {cells} | {_fmt(subject['mean_reading'])} |")
    add("")

    add("## 3. 结论全文")
    add("")
    for record in records:
        params = record["parameters"]
        add(f"### 被试 {record['subject_index']:03d} · `{params['theta_id']}`")
        add("")
        add(f"- 扰动：{record['blind_audit']['describe']}")
        add(f"- 表型印象：{_likeness_phrase(record['analysis']['asd_likeness'])}")
        add(f"- 平均读数：{_fmt(record['analysis']['mean_reading'])}　"
            f"置性度：{_fmt(record['analysis']['confidence'])}")
        add("")
        add("> " + (record["analysis"]["conclusion"] or "（分析者没有给出结论）")
            .replace("\n", "\n> "))
        add("")

    problems = [r for r in records if r["problems"] or r["analysis"]["parse_error"]]
    add("## 4. 协议问题")
    add("")
    if not problems:
        add("（无）")
    for record in problems:
        add(f"- 被试 {record['subject_index']:03d}："
            + "；".join(record["problems"] + (
                [f"分析者解析失败：{record['analysis']['parse_error']}"]
                if record["analysis"]["parse_error"] else []
            )))
    add("")

    if manifest["failures"]:
        add("## 5. 失败的被试")
        add("")
        for theta_id, reason in manifest["failures"]:
            add(f"- `{theta_id}`：{reason}")
        add("")

    add("## 6. 逐维细节（分数 / 证据 / 任务测量）")
    add("")
    for record in records:
        add(f"### 被试 {record['subject_index']:03d}")
        add("")
        add("| 维度 | 分数(0-100) | 典型性 | 偏离 | 观察证据 | 任务级测量 |")
        add("| --- | --- | --- | --- | --- | --- |")
        for dim, item in record["analysis"]["dimensions"].items():
            evidence = "；".join(item["observations"][:3]) or "—"
            tasks = "；".join(
                f"{m.get('task', '')} → {m.get('measurement', '')}"
                for m in item["task_measurements"]
            ) or "—"
            add(f"| {item['label']} | {_fmt(item['score'], 1)} | {_fmt(item['typicality'])} | "
                f"{_fmt(item['deviation'], 1)} | {_one_line(evidence, 80)} | {_one_line(tasks, 80)} |")
        add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="batch_probe",
        description="批量调用 PACS 心智：建心智 → 设参数 → 分析探查 → 保存结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/batch_probe.py --design single --doses 1.0,0.8,0.6 --dry-run\n"
            "  python tools/batch_probe.py --provider mock --design single --limit 2\n"
            "  python tools/batch_probe.py --design single --limit 20 --yes\n"
            "  python tools/batch_probe.py --primitive salience_assignment --dose 0.2 --yes\n"
        ),
    )
    # 与 `pacs` CLI 同名的 provider 覆盖项（默认读 config.json + 环境变量）
    parser.add_argument("--provider", choices=("mock", "openai"), default=None, help="LLM 提供方")
    parser.add_argument("--model", default=None, help="模型名，如 deepseek-chat / gpt-4o-mini")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容端点")
    parser.add_argument("--api-key", default=None, help="API Key（更推荐环境变量 PACS_LLM_API_KEY）")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--examiner-provider", choices=("mock", "openai"), default=None,
                        help="提问者 / 分析者用的提供方（默认与模拟大脑相同）")
    parser.add_argument("--examiner-model", default=None, help="提问者 / 分析者用的模型名")
    parser.add_argument("--examiner-base-url", default=None)
    parser.add_argument("--examiner-api-key", default=None,
                        help="提问者 / 分析者用的 Key（或环境变量 PACS_EXAMINER_LLM_API_KEY）")
    parser.add_argument("--same-provider-for-all-roles", action="store_true",
                        help="三方同模型：提问者 / 分析者直接复用模拟大脑那套配置")

    # 设计表（θ 怎么来）
    parser.add_argument("--design", choices=DESIGN_MODES, default="single",
                        help="θ 设计模式（默认 single；generic 为通用退化对照）")
    parser.add_argument("--doses", default=None,
                        help=f"单因素扰动的剂量档位，逗号或空格分隔"
                             f"（默认 {'/'.join(str(d) for d in DEFAULT_DOSE_LEVELS)}）")
    parser.add_argument("--dose", type=float, default=None,
                        help="组合 / 通用退化扰动使用的统一完整度")
    parser.add_argument("--primitive", default=None,
                        help="只跑这一个 primitive 的单因素扰动（给了它就忽略 --design）")
    parser.add_argument("--n-samples", type=int, default=32, help="LHS 采样数")
    parser.add_argument("--max-pairs", type=int, default=None, help="组合扰动对数上限")
    parser.add_argument("--limit", type=int, default=None,
                        help="被试数上限（保留完整心智基线与通用退化对照，其余等距抽样）")

    # 探查参数（默认与 agent 一致：不传就用包内常量）
    parser.add_argument("--turns", type=int, default=None,
                        help=f"每位被试的**基础**观察轮数（默认 {DEFAULT_N_TURNS}）")
    parser.add_argument("--max-turns", type=int, default=None,
                        help=f"轮数上限，含按配额自动补问（默认 {MAX_N_TURNS}，硬顶 {ABSOLUTE_MAX_TURNS}）")
    parser.add_argument("--text-limit", type=int, default=None,
                        help=f"问题与回答的字数上限（默认 {DEFAULT_TEXT_LIMIT}；超出只告警、不截断）")
    parser.add_argument("--probe-mode", choices=("auto", "scenario", "question"), default="auto",
                        help="内置条目库的探查形态")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-init", action="store_true", help="跳过提示词初始化步骤")

    # 产物与执行
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None,
                        help="产物目录（默认 data/runs/probe_<run_id>）")
    parser.add_argument("--base-dir", default=None, help="项目根目录（默认自动定位）")
    parser.add_argument("--resume", action="store_true",
                        help="续跑：跳过产物目录里已经完成的被试")
    parser.add_argument("--dry-run", action="store_true", help="只打印设计表与成本预估，不调用模型")
    parser.add_argument("--yes", action="store_true", help="确认大规模真实调用")
    return parser.parse_args(argv)


def _resolve_turns(args: argparse.Namespace) -> tuple[int, int, int]:
    """轮数与字数上限 —— 不传就用包内默认（与 agent 一致，句柄不写死数字）。"""
    turns = args.turns or DEFAULT_N_TURNS
    if not (1 <= turns <= ABSOLUTE_MAX_TURNS):
        raise SystemExit(f"✗ 基础轮数必须在 1..{ABSOLUTE_MAX_TURNS} 之间，收到 {turns}")
    max_turns = args.max_turns or max(MAX_N_TURNS, turns)
    if not (turns <= max_turns <= ABSOLUTE_MAX_TURNS):
        raise SystemExit(
            f"✗ 轮数上限必须落在 {turns}..{ABSOLUTE_MAX_TURNS}（且不小于基础轮数），收到 {max_turns}"
        )
    text_limit = args.text_limit or DEFAULT_TEXT_LIMIT
    return turns, max_turns, text_limit


def _completed_indices(subjects_dir: Path) -> dict[str, int]:
    """续跑用：读出产物目录里已完成的被试（θ 指纹 → 原序号）。"""
    done: dict[str, int] = {}
    if not subjects_dir.exists():
        return done
    for path in sorted(subjects_dir.glob("*_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        theta_id = str((payload.get("parameters") or {}).get("theta_id") or "")
        if theta_id:
            done[theta_id] = int(payload.get("subject_index") or 0)
    return done


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        turns, max_turns, text_limit = _resolve_turns(args)
    except SystemExit as exc:
        print(exc)
        return 2

    settings = _settings_from_args(args)
    designs = apply_limit(build_designs(args), args.limit)
    dose_levels = _doses(args.doses)

    # 条目库与观察提纲：与 `cmd_run` 用同一批选材参数（goals / n / seed / mode / limit）
    goals = tuple(SCORED_CHANNELS)
    probes = eb.dialogue_probes(goals=goals, n=turns, seed=args.seed, mode=args.probe_mode)
    rubric = eb.observation_rubric(goals=goals, limit=RUBRIC_LIMIT, seed=args.seed)

    run_id = args.run_id or new_run_id("probe")
    out_dir = Path(args.out) if args.out else (settings.data_dir / "runs" / f"probe_{run_id}")
    subjects_dir = out_dir / "subjects"

    base_calls = len(designs) * (turns * 2 + 1)
    upper_calls = len(designs) * (max_turns * 2 + 1)

    print("=" * 78)
    print("批量探针 · 建心智 → 设参数 → 分析探查 → 保存结果")
    print("=" * 78)
    print(settings.describe())
    print(f"  运行号      ：{run_id}")
    print(f"  心智年龄    ：{DEFAULT_AGE_MONTHS} 个月（固定常量，不可配置）")
    print(f"  设计模式    ：{args.design}"
          + (f"　剂量档位：{'/'.join(str(d) for d in dose_levels)}"
             if not args.primitive else f"　单因素：{args.primitive}"))
    print(f"  被试数      ：{len(designs)}"
          + (f"（--limit {args.limit} 已按代表性裁剪，保留基线与通用退化对照）"
             if args.limit else ""))
    print(f"  基础轮数    ：{turns}（每个方面至少 {QUESTIONS_PER_CHANNEL} 个问题的下限）")
    print(f"  轮数上限    ：{max_turns}（某方面问题数不足时自动补问，到上限为止）")
    print(f"  字数上限    ：{text_limit} 字/轮（超出只告警、不截断）")
    print(f"  探查方式    ：{args.probe_mode}　用量库条目 {len(probes)} 条　"
          f"观察提纲 {len(rubric)} 条")
    print(f"  预计调用数  ：{base_calls}（基础轮）～ {upper_calls}（补问至上限）")
    print(f"  产物目录    ：{out_dir}")
    print()

    if args.dry_run:
        print("（--dry-run：只打印设计表，不调用任何模型）")
        print()
        print(f"{'#':>4}  {'θ 指纹':<20}{'设计':<14}{'通用退化':>8}  扰动")
        for index, design in enumerate(designs, start=1):
            print(f"{index:>4}  {design.theta_id:<20}{design.design:<14}"
                  f"{design.generic_degradation:>8.2f}  {design.describe(include_design=False)}")
        print()
        _dry_run_checks(settings)
        return 0

    # 成本闸门：与 `pacs run` 同口径（真实 provider 下超阈值要 --yes）
    if settings.llm.provider != "mock" and upper_calls > CONFIRM_THRESHOLD and not args.yes:
        print(
            f"✗ 真实 provider 下调用数最多 {upper_calls}（上界）超过安全阈值 {CONFIRM_THRESHOLD}。"
            "\n  这会消耗实际额度。确认请加 --yes，或用 --limit 先小规模试跑。"
        )
        return 2

    # 就绪检查：mock 不需要 Key（离线可跑），因此只对真实 provider 要求已配置。
    brain_ok = settings.llm.provider == "mock" or settings.brain_ready()
    examiner_ok = settings.examiner.provider == "mock" or settings.examiner_ready()
    if not brain_ok:
        print("✗ 模拟大脑未配置好（真实 provider 需要 api_key）。"
              "请检查 config.json，或设置 PACS_LLM_API_KEY。")
        return 2
    if not examiner_ok:
        print("✗ 提问者 / 分析者未配置好（真实 provider 需要 api_key）。"
              "请检查 config.json 的 examiner_llm，或设置 PACS_EXAMINER_LLM_API_KEY；"
              "只想离线试跑请用 --provider mock。")
        return 2

    provider = get_provider(settings.llm)
    examiner = get_examiner_provider(settings)
    model_id = f"{provider.name}:{provider.model}"

    # 续跑：跳过已经落盘的被试。序号仍按**原始设计表**编号，因此续跑不会打乱产物命名。
    completed = _completed_indices(subjects_dir) if args.resume else {}
    index_of: dict[str, int] = {d.theta_id: i for i, d in enumerate(designs, start=1)}
    todo = [d for d in designs if d.theta_id not in completed]
    if completed:
        print(f"  续跑        ：已完成 {len(completed)} 个，本次待跑 {len(todo)} 个")
        print()
    if not todo:
        # 全部已完成：不重跑，但汇总必须基于**目录里已有的产物**重建 ——
        # 传空表会把 index.csv / summary.md 覆盖成空文件，那是比不重建更糟的结果。
        print("✓ 所有被试都已完成，无需重跑；从已有产物重建汇总。")
        existing = _merge_records(subjects_dir, [])
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _write_summaries(out_dir, run_id, settings, args, designs, probes, rubric,
                         turns, max_turns, text_limit, model_id, existing,
                         {"profiles": existing, "failures": []}, 0.0, now)
        print(f"  已有被试    ：{len(existing)} 个")
        print(f"  产物目录    ：{out_dir}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    pending: dict[str, dict] = {}
    records: list[dict] = []
    run_parameters = {
        "n_turns": turns, "max_turns": max_turns, "text_limit": text_limit,
        "probe_mode": args.probe_mode, "seed": args.seed, "design_mode": args.design,
        "initialize": not args.no_init, "goals": list(goals),
    }

    def finalize(entry: dict, profile) -> dict | None:
        """把「会话 + 分析者反馈 + 剖面」装成一条被试记录。

        θ 不经会话对象传递（`DialogueSession` 只带 `theta_id` 指纹，不带 θ 本身 ——
        这正是盲法的体现），因此这里按指纹回查本次设计表拿到那份 θ。
        """
        session = entry["session"]
        if session is None and profile is None:
            return None
        perturbation = perturbation_of(
            getattr(session, "theta_id", None) or getattr(profile, "theta_id", ""),
            designs,
        )
        if perturbation is None:
            return None
        provider_kind = getattr(profile, "provider_kind", None) or (
            "mock" if getattr(provider, "name", "") == "mock" else "llm"
        )
        return build_subject_record(
            run_id=run_id,
            model_id=model_id,
            index=entry["index"],
            session_id=(session.session_id if session is not None
                        else f"{run_id}-{entry['index']:03d}"),
            perturbation=perturbation,
            session=session,
            feedback=entry["feedback"] or {},
            profile=profile,
            goals=goals,
            run_parameters=run_parameters,
            provider_kind=provider_kind,
            synthetic=bool(getattr(profile, "synthetic", provider_kind == "mock")),
        )

    def save_subject(record: dict, *, announce: bool) -> None:
        """落盘一个被试（JSON + Markdown）。

        `announce=False` 用于「对话到手、剖面未回」的那一版：它会被剖面到齐后的完整版覆盖，
        因此不重复打进度行（否则每个被试会打两行、时长统计还会翻倍）。
        """
        index = record["subject_index"] or 0
        stem = f"{index:03d}_{record['parameters']['theta_id']}"
        _write_json(subjects_dir / f"{stem}.json", record)
        (subjects_dir / f"{stem}.md").write_text(render_subject_md(record), encoding="utf-8")
        if announce:
            analysis = record["analysis"]
            print(f"  [{index:03d}] ✓ {record['parameters']['theta_id']}"
                  f"　{record['blind_audit']['describe'][:40]}"
                  f"　平均 {_fmt(analysis['mean_reading'])}"
                  f"　表型：{_likeness_short(analysis['asd_likeness'])}")

    def on_dialogue(session, feedback) -> None:
        """对话与读数到手 → 先落一版（含参数与逐轮对话），跑完剖面再覆盖成完整版。"""
        entry = {
            "session": session, "feedback": feedback,
            "index": index_of.get(session.theta_id, 0),
        }
        pending[session.theta_id] = entry
        record = finalize(entry, None)
        if record is not None:
            save_subject(record, announce=False)
            records.append(record)

    def on_profile(profile) -> None:
        """剖面到手 → 补全并覆盖同一文件，得到「参数 + 对话 + 读数 + 剖面」。

        两个回调各写一次同一个文件：若在剖面之前中断，先落的那版仍在（至少不丢对话）；
        文件名由序号与 θ 指纹决定，因此两次指向同一路径。`records` 里也把先前那版换掉，
        避免汇总重复计数。
        """
        entry = pending.pop(profile.theta_id, None)
        if entry is None:
            # 没见到对应会话（回调次序意外）：照常落盘，对话区块留空并记一条协议问题。
            entry = {"session": None, "feedback": {},
                     "index": index_of.get(profile.theta_id, 0)}
        record = finalize(entry, profile)
        if record is None:
            return
        save_subject(record, announce=True)
        for position, existing in enumerate(records):
            if existing["parameters"]["theta_id"] == record["parameters"]["theta_id"]:
                records[position] = record
                break
        else:
            records.append(record)

    started_wall = datetime.datetime.now()
    started = time.time()
    print("开始逐被试运行（建心智 → 对话探针 → 分析者读数 → 落盘）")
    print()
    try:
        run = run_design(
            provider,
            todo,
            run_id=run_id,
            model_id=model_id,
            seed=args.seed,
            initialize=not args.no_init,
            examiner=examiner,
            probes=probes,
            probe_mode=args.probe_mode,
            dialogue_turns=turns,
            dialogue_max_turns=max_turns,
            text_limit=text_limit,
            rubric_items=rubric,
            on_profile=on_profile,
            on_dialogue=on_dialogue,
        )
    except LLMError as exc:
        print(f"\n✗ 运行中断：{exc}")
        return 2
    elapsed = time.time() - started

    print()
    print("=" * 78)
    print("运行完成")
    print("=" * 78)
    print(f"  表型剖面数  ：{len(run.profiles)}")
    print(f"  初始化成功率：{run.init_ok_rate:.0%}")
    if run.profiles:
        rates = [p.parse_rate for p in run.profiles]
        print(f"  对话完成率  ：{sum(rates) / len(rates):.1%}")
    print(f"  耗时        ：{elapsed:.1f}s")
    if run.failures:
        print(f"  ⚠ 失败 {len(run.failures)} 项：")
        for theta_id, reason in run.failures[:5]:
            print(f"    - {theta_id}: {reason}")

    # 续跑时把上一次已完成、本次没重跑的产物也读回来，让汇总覆盖整批被试。
    all_records = _merge_records(subjects_dir, records)
    _write_summaries(
        out_dir, run_id, settings, args, designs, probes, rubric,
        turns, max_turns, text_limit, model_id, all_records, run.to_dict(),
        elapsed, started_wall.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print()
    print(f"  产物目录：{out_dir}")
    print(f"    manifest.json　subjects/*.json　subjects/*.md　index.csv　summary.md")
    print()
    print(SIMULATION_NOTICE)
    print(DISCLAIMER)
    return 0


def _dry_run_checks(settings) -> None:
    """干跑时把「跑之前就该知道的事」也说清楚，免得开跑才发现配置不对。"""
    if settings.llm.provider == "mock":
        print("提示：当前 provider = mock（规则式替身）。它只验证管线，不是 LLM 涌现证据。")
    elif not settings.brain_ready():
        print("⚠ 模拟大脑未配置好（缺 api_key）：真实运行前请检查 config.json 或 PACS_LLM_API_KEY。")
    else:
        print(f"✓ 模拟大脑配置就绪：{settings.llm.provider} / {settings.llm.model}")
    eff = settings.examiner_effective()
    print(f"  提问者 / 分析者：{eff.provider} / {eff.model}"
          + ("（与模拟大脑相同）" if (eff.provider, eff.model, eff.base_url)
             == (settings.llm.provider, settings.llm.model, settings.llm.base_url)
             else "（独立模型）"))
    if settings.examiner.provider != "mock" and not settings.examiner_ready():
        print("⚠ 提问者 / 分析者缺 api_key：批次会跑到第一轮对话才失败。")


def perturbation_of(theta_id: str, designs: list[Perturbation]) -> Perturbation | None:
    """按 θ 指纹回查本次设计表里的那一份 θ。

    θ 不随会话或剖面传递（`DialogueSession` / `PhenotypeProfile` 都只带指纹）——
    这是盲法的一部分。产物里要写「设置了什么参数」，就得从本地设计表按指纹取回。
    """
    for design in designs:
        if design.theta_id == theta_id:
            return design
    return None


def _merge_records(subjects_dir: Path, fresh: list[dict]) -> list[dict]:
    """把目录里已有的被试产物全部读回来（续跑时用它汇总整批，而不是只汇总本次）。"""
    records: dict[int, dict] = {}
    for path in sorted(subjects_dir.glob("*_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        index = int(payload.get("subject_index") or 0)
        records[index] = payload
    for record in fresh:
        records[record["subject_index"]] = record
    return [records[k] for k in sorted(records)]


def _write_summaries(
    out_dir, run_id, settings, args, designs, probes, rubric,
    turns, max_turns, text_limit, model_id, records, run_payload, elapsed, started_at,
) -> None:
    """落盘汇总产物：manifest.json / index.csv / summary.md。"""
    finished = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subjects = [
        {
            "subject_index": r["subject_index"],
            "theta_id": r["parameters"]["theta_id"],
            "design": r["parameters"]["design"],
            "theta_note": r["parameters"].get("note", ""),
            "describe": r["blind_audit"]["describe"],
            "active_perturbations": r["parameters"]["active"],
            "generic_degradation": r["parameters"]["generic_degradation"],
            "readings": r["analysis"]["readings"],
            "mean_reading": r["analysis"]["mean_reading"],
            "confidence": r["analysis"]["confidence"],
            "asd_likeness": _likeness_phrase(r["analysis"]["asd_likeness"]),
            "conclusion_chars": r["analysis"]["conclusion_chars"],
            "n_turns_run": r["dialogue"]["session"]["n_turns_run"],
            "provider_kind": r["provider_kind"],
            "synthetic": r["synthetic"],
            "file": f"subjects/{r['subject_index']:03d}_{r['parameters']['theta_id']}.json",
        }
        for r in records
    ]
    manifest = _manifest(
        run_id=run_id, settings=settings, args=args, designs=designs, probes=probes,
        rubric=rubric, turns=turns, max_turns=max_turns, text_limit=text_limit,
        model_id=model_id, out_dir=out_dir, started=started_at, finished=finished,
        elapsed=elapsed, run_payload=run_payload, subjects=subjects,
    )
    _write_json(out_dir / "manifest.json", manifest)
    write_index_csv(out_dir / "index.csv", records)
    write_summary_md(out_dir / "summary.md", manifest, records)


if __name__ == "__main__":
    sys.exit(main())
