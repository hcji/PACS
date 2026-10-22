"""pacs 命令行入口。

典型用法（真实 LLM，BYOK）：

    # 1) 配置 Key（推荐用环境变量，不落盘）
    export PACS_LLM_API_KEY=<your-api-key>            # Windows: set PACS_LLM_API_KEY=<your-api-key>

    # 2) 先通过提示词初始化「人造大脑」，确认它对认知架构的自我报告
    pacs init-brain --provider openai --model gpt-4o-mini

    # 3) 查看扰动后的架构提示词（审计扰动是怎么作用于心智的）
    pacs prompt --primitive social_reward_weighting --dose 0.2

    # 4) 跑扰动设计 → 生成表型
    pacs run --provider openai --model gpt-4o-mini --mode single --limit 3 --yes

    # 5) 分析机制 / 分型 / 特征差异，生成 Atlas，冻结预测
    pacs analyze --endpoint social_communication
    pacs report
    pacs freeze

可视化界面（推荐入口，浏览器操作）：

    pacs serve                   # 启动界面；默认 mock，无需 Key 即可试跑
    pacs serve --provider openai --model gpt-4o-mini --api-key <your-api-key>

离线体验（不需要 Key，用规则式 Mock 心智）：

    pacs demo                    # 一键跑通全链路并生成 Atlas
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import DEFAULT_MODEL_VERSION, __version__
from .agents.virtual_agent import VirtualAgent
from .analysis import (
    counterfactual,
    dose_response,
    heterogeneity,
    mechanism_profile,
    specificity,
    subtypes,
)
from .brain import examiner_bank as eb
from .phenotype import PHENOTYPE_DIMS, SCORED_CHANNELS
from .phenotype import PhenotypeProfile
from .brain.architecture import build_brain_prompt
from .brain.perturb import (
    DEFAULT_DOSE_LEVELS,
    generic_degradation_control,
    intact,
    sample_design,
    single,
)
from .config import PROJECT_ROOT, Settings
from .engine import new_run_id, run_design
from .freeze import build_freeze, save_freeze, verify_freeze
from .agents.dialogue import (
    ABSOLUTE_MAX_TURNS,
    DEFAULT_N_TURNS,
    MAX_N_TURNS,
)
from . import (
    DEFAULT_AGE_MONTHS,
    DEFAULT_TEXT_LIMIT,
    MAX_TEXT_LIMIT,
    QUESTIONS_PER_CHANNEL,
)
from .phenotype.conclusion import ASD_LIKENESS_LEVEL_ZH, CONCLUSION_CHARS_TARGET
from .llm.provider import LLMError, get_examiner_provider, get_provider
from .llm.registry import profile_for
from .ontology.assays import (
    primitives_without_assay,
    summary_table as assay_table,
    validate_assays,
)
from .ontology.primitives import (
    DEFAULT_ACTIVE,
    PRIMITIVES,
    summary_table as primitive_table,
    validate_ontology,
)
from .ontology.systems import SYSTEMS, validate_systems
from .report import Atlas, build_atlas, render_atlas, write_atlas
from .store.db import Store

DISCLAIMER = "用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。"
CONFIRM_THRESHOLD = 200     # 真实 provider 下超过这个调用数需要 --yes


def _header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _likeness_phrase(likeness: dict) -> str:
    """把结构化表型印象写成一句话（缺值时如实说「未给出」，不替分析者下结论）。"""
    present = (likeness or {}).get("present")
    level = str((likeness or {}).get("level") or "unknown")
    if present is None:
        return f"未给出（程度档位：{level}）"
    verdict = "表现出孤独症样表型特征" if present else "未表现出孤独症样表型特征"
    return f"{verdict} · 程度 {ASD_LIKENESS_LEVEL_ZH.get(level, level)}"


def _wrap(text: str, width: int = 72) -> list[str]:
    """按宽度折行（终端里把结论读成人话，而不是一整行溢出屏幕）。"""
    lines: list[str] = []
    for paragraph in str(text or "").splitlines() or [""]:
        while len(paragraph) > width:
            lines.append(paragraph[:width])
            paragraph = paragraph[width:]
        lines.append(paragraph)
    return lines


def _build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings.load(base_dir=args.base_dir or PROJECT_ROOT)
    if getattr(args, "provider", None):
        settings.llm.provider = args.provider
    if getattr(args, "model", None):
        settings.llm.model = args.model
    if getattr(args, "base_url", None):
        settings.llm.base_url = args.base_url
    if getattr(args, "api_key", None):
        settings.llm.api_key = args.api_key
    if getattr(args, "temperature", None) is not None:
        settings.llm.temperature = args.temperature

    # 提问者 / 分析者：只覆盖显式给出的项，其余跟随模拟大脑（与界面一致）
    if getattr(args, "examiner_provider", None):
        settings.examiner.provider = args.examiner_provider
    if getattr(args, "examiner_model", None):
        settings.examiner.model = args.examiner_model
    if getattr(args, "examiner_base_url", None):
        settings.examiner.base_url = args.examiner_base_url
    if getattr(args, "examiner_api_key", None):
        settings.examiner.api_key = args.examiner_api_key

    settings.ensure_dirs()
    return settings


def _apply_limit(designs: list, limit: int | None) -> list:
    """按上限裁剪设计，但**保持代表性**，而不是从前面硬截断。

    为什么不能直接 `designs[:limit]`：设计列表的前段全是单因素扰动，硬截断会
    ① 丢掉全部组合扰动（Control 2 / 交互效应无从评估）；② 丢掉通用退化对照
    （Control 1 变成「缺失」）。用户为了控成本设了 limit，却在不知情的情况下
    失去了两个关键对照，Atlas 会报「incomplete」。
    因此这里：先保留完整心智基线与通用退化对照，其余按确定性等距抽样。
    """
    if not limit or limit >= len(designs):
        return designs
    keep: list = []
    rest: list = []
    for d in designs:
        if d.is_intact or d.generic_degradation > 0:
            keep.append(d)
        else:
            rest.append(d)
    budget = max(0, limit - len(keep))
    if budget == 0:
        return keep
    if len(rest) <= budget:
        return keep + rest
    stride = len(rest) / budget
    sampled = [rest[min(len(rest) - 1, int(i * stride))] for i in range(budget)]
    # 去重（等距抽样可能撞到同一下标）
    seen: set[str] = set()
    unique: list = []
    for d in sampled:
        if d.theta_id not in seen:
            seen.add(d.theta_id)
            unique.append(d)
    return keep + unique


def _open_store(settings: Settings) -> Store:
    return Store(settings.db_path)


def _load_profiles(store: Store, run_id: str | None, model_id: str | None) -> list[PhenotypeProfile]:
    profiles = store.load_profiles(run_id=run_id, model_id=model_id)
    if not profiles and not run_id:
        profiles = store.load_profiles()
    return profiles


# ---------------------------------------------------------------------------
# 只读命令
# ---------------------------------------------------------------------------


def cmd_ontology(args: argparse.Namespace) -> int:
    _header("认知基本单元本体（Level 0）")
    print(primitive_table())
    _header("Level 1 高阶认知（由 primitive 组合涌现，不直接扰动）")
    for s in SYSTEMS:
        print(f"\n{s.name_zh}（{s.id}）")
        print(f"  成员：{' + '.join(s.members)}")
        print(f"  涌现逻辑：{s.emergence_rule}")
        print(f"  可证伪预期：{s.falsifiable_expectation}")
    _header("每个 primitive 的独立 assay（FR-3）")
    print(assay_table())
    _header("自检")
    problems = validate_ontology() + validate_systems() + validate_assays()
    missing = primitives_without_assay()
    if missing:
        problems.append(f"未派生 assay：{', '.join(missing)}")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("  ✓ 本体 / 高阶构念 / assay 自检全部通过，且每个 primitive 都有独立 assay 与特异性对照。")
    print()
    print(DISCLAIMER)
    return 0


def cmd_vocab(args: argparse.Namespace) -> int:
    """打印**测量口径**与探查登记。

    这里刻意不做任何「机制 → 维度」的映射展示：构造属性与测量维度之间没有预设对应，
    维度读数由分析者从对话得出（`pacs analyze` 的 ②.1/②.4 是从数据估计出来的剖面）。
    """
    from .phenotype import DIM_ANCHORS

    _header("计分通道（表型向量 B 的键，顺序固定）")
    for ch, label in SCORED_CHANNELS.items():
        role = "表型维度（进雷达图）" if ch in PHENOTYPE_DIMS else "对照通道（不进雷达图，进评分与特异性判据）"
        print(f"  {ch:<40}{label:<24}{role}")
    print()
    _header("各通道的观察锚点（进提问者与分析者的提示词）")
    for ch, label in SCORED_CHANNELS.items():
        anchor = DIM_ANCHORS.get(ch) or {}
        print()
        print(f"{label}（{ch}）")
        print(f"  看什么：{anchor.get('watch', '—')}")
        print(f"  低分：{anchor.get('low', '—')}")
        print(f"  高分：{anchor.get('high', '—')}")
    print()
    _header("每个 primitive 的探查登记（用什么探查，不声称维度）")
    print(assay_table())
    print()
    print("注：对照通道没有内置条目，由提问者 LLM 自造一道日常问题；")
    print("    由测量估计出的机制—维度剖面见 `pacs analyze` 或 Atlas ②.4。")
    print()
    print(DISCLAIMER)
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    perturbation = (
        intact()
        if args.primitive == "intact"
        else single(args.primitive, args.dose, note=args.note or "")
    )
    _header(f"认知架构提示词（θ = {perturbation.theta_id}）")
    print(f"扰动：{perturbation.describe()}")
    print(f"指纹：{perturbation.theta_id}")
    _header("系统提示词（这就是「人造大脑」的初始化内容）")
    print(build_brain_prompt(perturbation))
    return 0


def cmd_init_brain(args: argparse.Namespace) -> int:
    """先通过提示词初始化人造大脑，并让模型报告自状态。"""
    settings = _build_settings(args)
    if args.primitive == "intact":
        perturbation = intact()
    elif args.primitive == "generic":
        perturbation = generic_degradation_control(args.dose)
    else:
        perturbation = single(args.primitive, args.dose)

    _header("初始化人造大脑")
    print(settings.describe())
    print()
    print(f"扰动配置：{perturbation.describe()}")
    try:
        provider = get_provider(settings.llm)
    except LLMError as exc:
        print(f"\n✗ {exc}")
        return 2

    agent = VirtualAgent(provider, perturbation, model_id=f"{provider.name}:{provider.model}")
    print(f"模型档案：{profile_for(provider.name, provider.model).__dict__}")
    print("\n正在下发认知架构 + 初始化指令 …")
    started = time.time()
    try:
        result = agent.initialize()
    except LLMError as exc:
        print(f"\n✗ 初始化失败：{exc}")
        return 2
    elapsed = time.time() - started
    _header("心智的自我报告")
    print(result.self_report or "(空)")
    print()
    print(f"初始化标记 SELF_STATE: OK → {'✓ 成功' if result.ok else '✗ 未检出'}")
    print(f"耗时 {elapsed:.1f}s ｜ θ = {result.theta_id}")
    if not result.ok:
        print("提示：真实模型可能不严格守格式；这不必然是失败，但建议检查提示词或换模型。")
    print()
    print(DISCLAIMER)
    return 0 if result.ok else 1


def _turns_of(args: argparse.Namespace) -> int:
    """本次运行的**基础**轮数：默认 = 每个计分通道各 5 个问题的下限（7 × 5 = 35）。"""
    turns = int(getattr(args, "turns", None) or DEFAULT_N_TURNS)
    if not (1 <= turns <= ABSOLUTE_MAX_TURNS):
        raise SystemExit(f"✗ 轮数必须在 1..{ABSOLUTE_MAX_TURNS} 之间，收到 {turns}")
    return turns


def _max_turns_of(args: argparse.Namespace, n_turns: int) -> int:
    """本次运行的轮数**上限**（含按配额自动补问的轮次）。"""
    limit = int(getattr(args, "max_turns", None) or MAX_N_TURNS)
    if not (n_turns <= limit <= ABSOLUTE_MAX_TURNS):
        raise SystemExit(
            f"✗ 轮数上限必须在 {n_turns}..{ABSOLUTE_MAX_TURNS} 之间（且不小于基础轮数），"
            f"收到 {limit}"
        )
    return limit


def _text_limit_of(args: argparse.Namespace) -> int:
    """问题与回答的字数上限（默认 100；超出只告警，不截断）。"""
    limit = int(getattr(args, "text_limit", None) or DEFAULT_TEXT_LIMIT)
    if not (1 <= limit <= MAX_TEXT_LIMIT):
        raise SystemExit(f"✗ 字数上限必须在 1..{MAX_TEXT_LIMIT} 之间，收到 {limit}")
    return limit


def cmd_sample(args: argparse.Namespace) -> int:
    designs = sample_design(
        mode=args.mode,
        doses=tuple(args.doses),
        seed=args.seed,
        n_samples=args.n_samples,
        pair_dose=args.dose,
        max_pairs=args.max_pairs,
    )
    turns = _turns_of(args)
    max_turns = _max_turns_of(args, turns)
    base_calls = len(designs) * (turns * 2 + 1)
    total_calls = len(designs) * (max_turns * 2 + 1)
    _header(f"扰动设计（mode={args.mode}）")
    print(f"被试数（θ 数）：{len(designs)}")
    print(f"每位被试的基础轮数：{turns}（每个计分通道各 "
          f"{QUESTIONS_PER_CHANNEL} 个问题的下限）")
    print(f"每位被试的轮数上限：{max_turns}（某方面问题数不足时自动补问，到上限为止）")
    print(f"预计模型调用数：{base_calls}（基础轮）～ {total_calls}（补问至上限）")
    print()
    for d in designs[:20]:
        print(f"  {d.theta_id}  {d.describe()}")
    if len(designs) > 20:
        print(f"  …（其余 {len(designs) - 20} 个）")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    with _open_store(settings) as store:
        _header("本地运行记录")
        runs = store.list_runs()
        if not runs:
            print("（无）")
        for r in runs:
            print(f"  {r['run_id']:<30} model={r['model_id']:<32} n_profiles={r['n_profiles']:<5} {r['created_at']}")
        _header("库统计")
        for table, count in store.counts().items():
            print(f"  {table:<16}{count}")
        print(f"\n数据库：{settings.db_path}")
    return 0


# ---------------------------------------------------------------------------
# 运行命令
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    if args.mode == "all":
        designs = (
            sample_design("single", doses=tuple(args.doses), seed=args.seed)
            + sample_design("pair", pair_dose=args.dose, seed=args.seed, max_pairs=args.max_pairs)
            + [generic_degradation_control(0.5, seed=args.seed + rep) for rep in range(5)]
        )
    else:
        designs = sample_design(
            mode=args.mode,
            doses=tuple(args.doses),
            seed=args.seed,
            n_samples=args.n_samples,
            pair_dose=args.dose,
            max_pairs=args.max_pairs,
        )
    if args.limit:
        designs = _apply_limit(designs, args.limit)

    turns = _turns_of(args)
    max_turns = _max_turns_of(args, turns)
    text_limit = _text_limit_of(args)
    # 闸门按**上界**算：配额补问会把轮数从基础轮数推到上限，等到那时再拦已经来不及了。
    base_calls = len(designs) * (turns * 2 + 1)
    total_calls = len(designs) * (max_turns * 2 + 1)
    _header("运行扰动设计")
    print(settings.describe())
    print(f"模式={args.mode}  被试数={len(designs)}  基础轮数={turns}  轮数上限={max_turns}"
          f"  字数上限={text_limit}")
    print(f"预计模型调用数：{base_calls}（基础轮）～ {total_calls}（补问至上限）")
    if settings.llm.provider != "mock" and total_calls > CONFIRM_THRESHOLD and not args.yes:
        print(
            f"\n✗ 真实 provider 下调用数最多 {total_calls}（上界）超过安全阈值 {CONFIRM_THRESHOLD}。"
            "\n  这会消耗实际额度。确认请加 --yes，或用 --limit 先小规模试跑。"
        )
        return 2

    try:
        provider = get_provider(settings.llm)
    except LLMError as exc:
        print(f"\n✗ {exc}")
        return 2

    run_id = args.run_id or new_run_id("run")
    model_id = f"{provider.name}:{provider.model}"

    with _open_store(settings) as store:
        for d in designs:
            store.save_perturbation(d)
        store.save_run(run_id, model_id, {"mode": args.mode, "n_designs": len(designs)})
        store.audit("run_start", f"{run_id} mode={args.mode} designs={len(designs)}")

        started = time.time()

        def progress(index: int, total: int, perturbation) -> None:
            if index % max(1, total // 20) == 0 or index == total:
                print(f"  [{index}/{total}] {perturbation.theta_id} {perturbation.describe()[:60]}")

        def on_profile(profile: PhenotypeProfile) -> None:
            store.save_profile(profile)

        try:
            result = run_design(
                provider,
                designs,
                run_id=run_id,
                model_id=model_id,
                seed=args.seed,
                initialize=not args.no_init,
                progress=progress,
                on_profile=on_profile,
                examiner=get_examiner_provider(settings),
                probes=eb.dialogue_probes(
                    goals=tuple(SCORED_CHANNELS),
                    n=turns, seed=args.seed, mode=args.probe_mode,
                ),
                probe_mode=args.probe_mode,
                dialogue_turns=turns,
                dialogue_max_turns=max_turns,
                text_limit=text_limit,
                rubric_items=eb.observation_rubric(
                    goals=tuple(SCORED_CHANNELS),
                    limit=8, seed=args.seed,
                ),
            )
        except LLMError as exc:
            print(f"\n✗ 运行中断：{exc}")
            return 2

        store.audit("run_done", f"{run_id} profiles={len(result.profiles)}")
        elapsed = time.time() - started

    _header("运行完成")
    print(f"run_id      ：{run_id}")
    print(f"表型剖面数  ：{len(result.profiles)}")
    print(f"初始化成功率：{result.init_ok_rate:.0%}")
    print(f"对话完成率  ：{sum(p.parse_rate for p in result.profiles) / max(1, len(result.profiles)):.2%}")
    print(f"耗时        ：{elapsed:.1f}s")
    if result.failures:
        print(f"\n⚠ {len(result.failures)} 项失败：")
        for theta_id, reason in result.failures[:5]:
            print(f"  - {theta_id}: {reason}")
    print()
    print(f"下一步：pacs analyze --run-id {run_id} --endpoint {args.endpoint}")
    return 0


# ---------------------------------------------------------------------------
# 分析 / 导出 / 报告 / 冻结
# ---------------------------------------------------------------------------


def _atlas_from_args(args: argparse.Namespace) -> tuple[Atlas, Settings] | tuple[None, Settings]:
    settings = _build_settings(args)
    with _open_store(settings) as store:
        profiles = _load_profiles(store, getattr(args, "run_id", None), getattr(args, "model_id", None))
    if not profiles:
        print("✗ 库里没有表型数据。先跑 `pacs run`（离线可加 --provider mock）。")
        return None, settings
    if not profiles:
        print("✗ 库里没有可用的表型数据。")
        return None, settings
    model_profiles = [
        profile_for(*(m.split(":", 1) if ":" in m else (m, m)))
        for m in sorted({p.model_id for p in profiles})
    ]
    atlas = build_atlas(
        profiles,
        model_version=DEFAULT_MODEL_VERSION,
        endpoint_dim=getattr(args, "endpoint", "social_communication_pragmatics"),
        model_profiles=model_profiles,
        threshold=getattr(args, "threshold", 0.70),
        dose=getattr(args, "dose", None),
    )
    return atlas, settings


def cmd_analyze(args: argparse.Namespace) -> int:
    atlas, _settings = _atlas_from_args(args)
    if atlas is None:
        return 1
    _header(f"机制分析：表型终点 = {atlas.endpoint_dim}")
    print(f"被试数 {atlas.n_profiles}；模型 {', '.join(atlas.model_ids)}")
    print()
    print("① 扰动特异性")
    print(specificity.render_table(atlas.specificity_verdicts) if atlas.specificity_verdicts else "  （无单因素数据）")
    print()
    print("② 剂量-反应")
    print(dose_response.render_table(atlas.dose_fits) if atlas.dose_fits else "  （数据不足）")
    print()
    if atlas.generic_verdict:
        print("③ 通用退化对照（Control 1）")
        print("  " + atlas.generic_verdict.reason)
        print()
    print("④ 机制异质性（同表型不同机制）")
    print(heterogeneity.render_convergence(atlas.convergence) if atlas.convergence else "  （无数据）")
    print()
    print("⑤ 计算亚型")
    print(subtypes.render_subtypes(atlas.subtypes, atlas.subtype_scores))
    print()
    print("⑥ 跨模型收敛")
    if atlas.convergence_matrix:
        for pid, row in sorted(atlas.convergence_matrix.items()):
            cells = "  ".join(f"{m}={row.get(m, 'n/a')}" for m in atlas.model_ids)
            print(f"  {pid:<34}{cells}   → {row.get('_verdict')}")
    else:
        print("  （单模型；需 ≥2 个模型才可评估）")
    print()
    print("⑦ 反事实预测与人类实验反向设计")
    print(
        counterfactual.render_experiment_design(atlas.experiment_design, atlas.counterfactuals)
        if atlas.experiment_design
        else "  （无候选机制）"
    )
    print()
    print(DISCLAIMER)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    atlas, settings = _atlas_from_args(args)
    if atlas is None:
        return 1
    md_path = settings.exports_dir / "atlas.md"
    json_path = settings.exports_dir / "atlas.json"
    write_atlas(atlas, md_path, json_path)
    _header("Atlas 报告已生成")
    print(f"  Markdown：{md_path}")
    print(f"  JSON    ：{json_path}")
    print()
    print(render_atlas(atlas)[:1200])
    print("\n…（完整内容见文件）")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    atlas, settings = _atlas_from_args(args)
    if atlas is None:
        return 1
    with _open_store(settings) as store:
        profiles = _load_profiles(store, args.run_id, args.model_id)
        perturbations = []
        for p in profiles:
            perturbations.append(
                {
                    "theta_id": p.theta_id,
                    "design": p.design,
                    "theta": p.theta,
                    "generic_degradation": p.generic_degradation,
                }
            )
        snapshot = build_freeze(
            perturbations=perturbations,
            convergence=atlas.convergence,
            counterfactuals=atlas.counterfactuals,
            model_version=DEFAULT_MODEL_VERSION,
        )
        problems = verify_freeze(snapshot)
        out_dir = save_freeze(snapshot, settings.exports_dir)
        store.save_freeze(snapshot.freeze_id, snapshot.to_dict())
        store.audit("freeze", snapshot.freeze_id)

    _header("冻结完成（FR-15）")
    print(f"freeze_id：{snapshot.freeze_id}")
    print(f"产物目录 ：{out_dir}")
    print()
    print("被冻结的五类对象及其摘要：")
    for name, value in snapshot.digests.items():
        print(f"  {name:<26}{value}")
    if problems:
        print("\n⚠ 一致性校验问题：")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\n✓ 冻结快照一致性校验通过。")
    print()
    for note in snapshot.notes:
        print(f"  · {note}")
    print()
    print(DISCLAIMER)
    return 0


# ---------------------------------------------------------------------------
# 可视化界面
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    """启动本地可视化界面。"""
    from .api.server import serve

    settings = _build_settings(args)
    if args.api_key:
        settings.llm.api_key = args.api_key
    serve(settings, host=args.host, port=args.port, open_browser=not args.no_browser,
          auto_init=not args.no_auto_init)
    return 0


def cmd_brains(args: argparse.Namespace) -> int:
    """列出大脑库（保存/切换见界面「① 模型设置」，或 API /api/brains）。"""
    settings = _build_settings(args)
    with _open_store(settings) as store:
        brains = store.list_brains()
        active_id = store.get_meta("active_brain_id")
        _header(f"大脑库（{len(brains)} 个）")
        if not brains:
            print("  还没有保存的大脑。可用 `pacs run` 或界面「② 扰动注入」产生并入库。")
            print()
            print(DISCLAIMER)
            return 0
        for record in brains:
            payload = record.get("payload") or {}
            mark = "← 当前" if record["brain_id"] == active_id else ""
            print(f"  [{record['brain_id']}] {record['name']} {mark}")
            print(f"       θ = {record['theta_id']}  来源={record.get('source')}  "
                  f"更新={record.get('updated_at')}")
            print(f"       {payload.get('describe', '')}")
        print()
        print("  切换当前大脑：POST /api/brains/activate {\"brain_id\": \"...\"}（界面「① 模型设置」可点选）")
        print()
        print(DISCLAIMER)
    return 0


def cmd_dialogue(args: argparse.Namespace) -> int:
    """对话探针：提问者 LLM 提问 → 模拟大脑作答 → 分析者 LLM 形成表型反馈。"""
    from .agents.dialogue import analyze_dialogue, run_dialogue
    from .agents.virtual_agent import VirtualAgent
    from .llm.provider import get_examiner_provider

    settings = _build_settings(args)
    if args.primitive and args.primitive not in DEFAULT_ACTIVE:
        print(f"未知 primitive：{args.primitive}（可用：{', '.join(DEFAULT_ACTIVE)}）")
        return 2
    if args.primitive:
        perturbation = single(args.primitive, args.dose, seed=args.seed)
    else:
        perturbation = intact(seed=args.seed)

    turns = _turns_of(args)
    max_turns = _max_turns_of(args, turns)
    text_limit = _text_limit_of(args)
    _header("表型观察 · 对话探针")
    print(f"  模拟大脑 θ：{perturbation.theta_id}  {perturbation.describe()}")
    print(f"  心智年龄  ：{DEFAULT_AGE_MONTHS} 个月（固定值，不可配置）")
    print(f"  基础轮数  ：{turns}（每个方面各 5 个问题的下限）")
    print(f"  轮数上限  ：{max_turns}（某方面问题数不足 5 个时自动补问，到上限为止）")
    print(f"  字数上限  ：{text_limit} 字/轮（问题与回答；超出只告警，不截断）")
    print(f"  预计调用  ：{turns * 2 + 1}（基础轮）～ {max_turns * 2 + 1}（补问至上限）")
    print("  提问者与分析者对 θ 盲法：它们看不到完整度参数，其判断作为独立读数。")
    print()

    brain_provider = get_provider(settings.llm)
    examiner = get_examiner_provider(settings)
    model_id = f"{brain_provider.name}:{brain_provider.model}"
    agent = VirtualAgent(brain_provider, perturbation, model_id=model_id,
                         seed=args.seed, text_limit=text_limit)

    probes = probe_items(settings, args) if getattr(args, "probe_mode", None) else None
    session = run_dialogue(
        agent, examiner, probe_items=probes, probe_mode=getattr(args, "probe_mode", None) or "auto",
        n_turns=turns, max_turns=max_turns, text_limit=text_limit, seed=args.seed,
        on_turn=lambda turn: print(
            f"  [{'提问者' if turn['role'] == 'examiner' else '大脑  '}] {turn['content']}"
        ),
    )
    for problem in session.problems:
        print(f"  ! {problem}")
    print(f"  · 实际 {session.n_turns_run} 轮；逐方面问题数："
          + "、".join(f"{SCORED_CHANNELS.get(k, k)} {v}" for k, v in session.question_counts.items()))
    print()

    feedback = analyze_dialogue(
        examiner, session,
    )
    _header("表型反馈（分析者 LLM，盲法）")
    if feedback.get("error"):
        print(f"  ! 分析者输出解析问题：{feedback['error']}")
    for dim, value in (feedback.get("dims") or {}).items():
        print(f"  {PHENOTYPE_DIMS.get(dim, dim):<20s} {value}")
    print(f"  平均读数  ：{feedback.get('mean_reading')}")
    print(f"  置信度    ：{feedback.get('confidence')}")
    print(f"  叙述      ：{feedback.get('narrative')}")
    likeness = feedback.get("asd_likeness") or {}
    print()
    print("  ── 探查结论（病例报告式总结）──")
    print(f"  表型印象：{_likeness_phrase(likeness)}")
    for line in _wrap(feedback.get("conclusion") or "（分析者没有给出结论）"):
        print(f"    {line}")
    print(f"  （{feedback.get('conclusion_chars') or 0} 字 / 约 "
          f"{feedback.get('conclusion_target_chars') or CONCLUSION_CHARS_TARGET} 字）")
    for note in (feedback.get("conclusion_notes") or []):
        print(f"  ! 结论：{note}")
    for note in (feedback.get("notes") or "").splitlines():
        if note.strip():
            print(f"  备注      ：{note}")
    print()
    print(f"  盲法审计（分析者看不到的 ground truth）："
          f"{perturbation.active_perturbations() or '（完整心智）'}"
          f"  通用退化={perturbation.generic_degradation:.2f}")
    print()

    with _open_store(settings) as store:
        store.save_dialogue_session(session.session_id, session.to_dict())
        for index, turn in enumerate(session.turns):
            store.save_dialogue_turn(session.session_id, index, turn["role"], turn["content"], turn)
        store.save_feedback(session.session_id, session.theta_id,
                            feedback.get("model_id", ""), feedback)
        store.audit("cli_dialogue", f"{session.session_id} turns={session.n_brain_turns}")
    print(f"  已入库：session={session.session_id}（界面「③ 表型观察 → 历史对话」可回看）")
    print()
    print(DISCLAIMER)
    return 0


# ---------------------------------------------------------------------------
# 一键离线演示
# ---------------------------------------------------------------------------


def cmd_bank(args: argparse.Namespace) -> int:
    """浏览观察者内置条目库（来源：AsdKB；随包分发）。"""
    from .brain import examiner_bank as eb

    try:
        stats = eb.bank_stats()
    except FileNotFoundError as exc:
        print(exc)
        return 2

    if args.layer:
        return _print_reference_layer(args.layer)

    _header(f"内置条目库（{stats['items_total']} 条 · 来源 {eb.source_meta()['source']['name']}）")
    print(f"  量表      : {stats['scales']} 套（{stats['scales_with_items']} 套有逐题）")
    modes = stats.get("by_probe_mode", {})
    bases = stats.get("by_probe_basis", {})
    print(f"  可探查条目       : {stats['probe_ready']}"
          f"（直接提问 {modes.get('question', 0)} · 情境式 {modes.get('scenario', 0)}）"
          f"　仅作观察要点 : {stats['observation_only']}")
    print(f"  探查来源         : 人工撰写 {bases.get('curated', 0)}"
          f" · 模板派生 {bases.get('template', 0)}")
    print(f"  覆盖 36 月龄     : {stats['covers_36_months']} 条")
    print(f"  施测方式         : {stats['by_admin_zh']}")
    print(f"  参考层           : 症状 {stats['symptoms']} · 诊断标准 "
          f"{stats['diagnostic_criteria']} · 循证干预 {stats['treatments']}"
          "（用 --layer 查看）")

    if args.rules:
        rules = eb.mapping_rules()
        print()
        _header("映射规则")
        print("  维度判定优先级：" + " → ".join(rules["dim_priority"]))
        print(f"  （{rules.get('dim_priority_note','')}）")
        print()
        for dim, keywords in rules["dim_keywords"].items():
            print(f"  {dim} ← " + "、".join(keywords[:8]) + " …")
        print()
        for sid, dims in rules["scale_dims"].items():
            print(f"  {sid} → " + "、".join(f"{k}={v}" for k, v in dims.items()))
        print()
        print("  规则全文见 src/pacs/data/examiner_bank.json 的 mapping_rules，")
        print("  界面「③ 表型观察 → 映射规则与来源」可展开查看。")
        print()
        print(DISCLAIMER)
        return 0

    page = eb.list_items(
        dim=args.dim or "",
        primitive=args.primitive or "",
        admin=args.admin or "",
        probe_mode=args.probe_mode or "",
        source_id=args.scale or "",
        probe_ready=args.probe_ready,
        age=args.age,
        query=args.query or "",
        limit=args.limit,
        offset=0,
    )
    print()
    print(f"  筛选后 {page['total']} 条（显示前 {len(page['items'])} 条）")
    print()
    dims_zh = eb.dims_map()
    for item in page["items"]:
        dims = "、".join(dims_zh.get(d, d) for d in (item.get("dims") or {})) or "—"
        prims = "、".join(item.get("primitives") or {}) or "—"
        print(f"  [{item['key']}] {item['text_zh']}")
        if item.get("probe_zh"):
            label = "情境式探查" if item.get("probe_mode") == "scenario" else "直接提问"
            for index, line in enumerate(item["probe_zh"].split("\n")):
                print(f"      {label}：{line}" if index == 0 else f"              {line}")
            if (item.get("scenario") or {}).get("observe"):
                print(f"      观察目标：{item['scenario']['observe']}")
        else:
            print(f"      观察要点：{item.get('observation') or '—'}")
        print(f"      {eb.ADMIN_ZH.get(item['admin'], item['admin'])}"
              f" | 适用年龄 {item.get('age_hint') or '—'} | {dims} | {prims}")
        print(f"      出处：{item['source_ref']}")
    print()
    print("  用 `pacs dialogue --turns N` 跑对话探针（会按年龄自动选题），")
    print("  或在界面「③ 表型观察 → 内置条目库」勾选条目。")
    print()
    print(DISCLAIMER)
    return 0


def _print_reference_layer(layer: str) -> int:
    """打印 AsdKB 的三个参考层：症状 / 诊断标准 / 循证干预。"""
    from .brain import examiner_bank as eb

    if layer == "symptoms":
        _header(f"症状层（{len(eb.symptoms())} 条）")
        for s in eb.symptoms():
            dims = "、".join(eb.dims_map().get(d, d) for d in s["dims"])
            print(f"  {s['id']:10s} [{dims}] {s['name_zh']}")
            print(f"       源数据提及题数 {s['mentioned_in_questions']}"
                  "（量表级覆盖，不可用于逐题路由）")
    elif layer == "criteria":
        _header(f"诊断标准条目（{len(eb.criteria())} 条）")
        for c in eb.criteria():
            domain = "受限重复行为域" if c["domain"] == "rrb" else "社交沟通域"
            print(f"  {c['id']:4s} [{domain}] {c['text_zh'][:70]}")
    else:
        _header(f"循证干预（{len(eb.treatments())} 条 · {eb.treatments()[0]['type']}）")
        for t in eb.treatments():
            print(f"  {t['id']:12s} {t['name_zh']}（{t['name_en']}）")
            print(f"       {t['introduction'][:80]}")
    print()
    print(DISCLAIMER)
    return 0


def probe_items(settings, args) -> list:
    """按界面同款规则从内置条目库选题（供 CLI 的对话探针使用）。"""
    from .brain import examiner_bank as eb

    return eb.dialogue_probes(
        goals=(),
        n=int(getattr(args, "turns", None) or DEFAULT_N_TURNS),
        seed=getattr(args, "seed", 42),
        mode=getattr(args, "probe_mode", None) or "auto",
    )


def cmd_demo(args: argparse.Namespace) -> int:
    """用规则式 Mock 心智跑通全链路（无需 Key、无需网络），产出全部交付物。"""
    args.provider = "mock"
    settings = _build_settings(args)
    _header("PACS 离线演示（Mock 心智，无需 API Key）")
    print("注意：Mock 是**规则式前向模型**，表型结构由 θ 与题目敏感度算出 —— 它验证的是")
    print("管线与统计逻辑，不是 LLM 的涌现行为。真实结论必须用真实 provider 跑。")
    print()

    designs = (
        sample_design("single", doses=tuple(args.doses), seed=args.seed)
        + sample_design("pair", pair_dose=args.dose, seed=args.seed, max_pairs=args.max_pairs)
        + [generic_degradation_control(0.5, seed=args.seed + rep) for rep in range(5)]
    )
    provider = get_provider(settings.llm)
    run_id = args.run_id or new_run_id("demo")
    model_id = f"{provider.name}:{provider.model}"

    with _open_store(settings) as store:
        for d in designs:
            store.save_perturbation(d)
        store.save_run(run_id, model_id, {"mode": "demo", "n_designs": len(designs)})

        def on_profile(profile: PhenotypeProfile) -> None:
            store.save_profile(profile)

        started = time.time()
        demo_turns = _turns_of(args)
        result = run_design(
            provider,
            designs,
            run_id=run_id,
            model_id=model_id,
            seed=args.seed,
            initialize=not args.no_init,
            on_profile=on_profile,
            examiner=get_examiner_provider(settings),
            probes=eb.dialogue_probes(
                goals=tuple(SCORED_CHANNELS),
                n=demo_turns, seed=args.seed, mode=args.probe_mode,
            ),
            probe_mode=args.probe_mode,
            dialogue_turns=demo_turns,
            dialogue_max_turns=_max_turns_of(args, demo_turns),
            text_limit=_text_limit_of(args),
            rubric_items=eb.observation_rubric(
                goals=tuple(SCORED_CHANNELS),
                limit=8, seed=args.seed,
            ),
        )
        elapsed = time.time() - started

    _header("运行完成")
    print(f"run_id={run_id}  被试数={len(result.profiles)}  耗时={elapsed:.1f}s")
    print(f"初始化成功率={result.init_ok_rate:.0%}  对话完成率={sum(p.parse_rate for p in result.profiles)/max(1,len(result.profiles)):.1%}")

    # 分析
    args.run_id = run_id
    atlas, settings = _atlas_from_args(args)
    if atlas is None:
        return 1
    md_path = settings.exports_dir / "atlas.md"
    json_path = settings.exports_dir / "atlas.json"
    write_atlas(atlas, md_path, json_path)

    with _open_store(settings) as store:
        profiles = [p for p in store.load_profiles(run_id=run_id) if p.generic_degradation <= 0]
        snapshot = build_freeze(
            perturbations=[
                {"theta_id": p.theta_id, "design": p.design, "theta": p.theta,
                 "generic_degradation": p.generic_degradation}
                for p in profiles
            ],
            convergence=atlas.convergence,
            counterfactuals=atlas.counterfactuals,
            model_version=DEFAULT_MODEL_VERSION,
        )
        out_dir = save_freeze(snapshot, settings.exports_dir)
        store.save_freeze(snapshot.freeze_id, snapshot.to_dict())
        store.audit("demo_done", f"{run_id} freeze={snapshot.freeze_id}")

    _header("交付物")
    print(f"  表型数据库   ：{settings.db_path}")
    print(f"  Atlas 报告   ：{md_path}")
    print(f"  Atlas JSON   ：{json_path}")
    print(f"  冻结快照     ：{out_dir}")
    print(f"  freeze_id    ：{snapshot.freeze_id}")
    print()

    _header("机制异质性摘要（核心结果）")
    print(heterogeneity.render_convergence(atlas.convergence) if atlas.convergence else "（无候选机制）")
    print()
    print(subtypes.render_subtypes(atlas.subtypes, atlas.subtype_scores))
    print()
    print(DISCLAIMER)
    return 0


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pacs",
        description="可扰动人工认知系统：扰动基础认知过程，产生孤独症样行为表型，研究其机制与分型。",
    )
    parser.add_argument("--version", action="version", version=f"pacs {__version__}")
    parser.add_argument("--base-dir", type=Path, default=None, help="项目根目录（默认取包所在项目）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_llm_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=("mock", "openai"), default=None, help="LLM 提供方")
        p.add_argument("--model", default=None, help="模型名，如 gpt-4o-mini / deepseek-chat / qwen2.5")
        p.add_argument("--base-url", default=None, help="OpenAI 兼容端点")
        p.add_argument("--api-key", default=None, help="API Key（更推荐用环境变量 PACS_LLM_API_KEY）")
        p.add_argument("--temperature", type=float, default=None)
        # 第二套 API：表型观察里的提问者 / 分析者（「另外的 LLM」）。不给则跟随 --provider 那套。
        p.add_argument("--examiner-provider", choices=("mock", "openai"), default=None,
                       help="提问者 / 分析者用的提供方（默认与模拟大脑相同）")
        p.add_argument("--examiner-model", default=None, help="提问者 / 分析者用的模型名")
        p.add_argument("--examiner-base-url", default=None, help="提问者 / 分析者用的端点")
        p.add_argument("--examiner-api-key", default=None,
                       help="提问者 / 分析者用的 Key（或环境变量 PACS_EXAMINER_LLM_API_KEY）")

    def add_run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=("single", "pair", "lhs", "intact", "all"), default="single")
        p.add_argument("--doses", type=float, nargs="+", default=list(DEFAULT_DOSE_LEVELS))
        p.add_argument("--dose", type=float, default=0.2, help="组合扰动的统一 dose（默认取最强档，便于与单因素同剂量比较）")
        p.add_argument("--n-samples", type=int, default=32, help="LHS 采样数")
        p.add_argument("--max-pairs", type=int, default=None, help="组合扰动对数上限")
        p.add_argument("--turns", type=int, default=None,
                       help="每位被试的**基础**观察轮数（默认 35 = 7 个计分通道 × 每通道 5 个问题）")
        p.add_argument("--probe-mode", choices=("auto", "scenario", "question"), default="auto",
                       help="内置条目库的探查形态")
        p.add_argument("--max-turns", type=int, default=None,
                       help="轮数上限，含按配额自动补问的轮次（默认 100，硬顶 500）")
        p.add_argument("--text-limit", type=int, default=None,
                       help="问题与回答的字数上限（默认 100；超出只告警，不截断）")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--limit", type=int, default=None, help="被试数上限（成本控制）")
        p.add_argument("--run-id", default=None)
        p.add_argument("--no-init", action="store_true", help="跳过提示词初始化步骤")
        p.add_argument("--yes", action="store_true", help="确认大规模真实调用")

    def add_analysis_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", default=None)
        p.add_argument("--model-id", default=None)
        p.add_argument("--endpoint", default="social_communication_pragmatics", choices=tuple(PHENOTYPE_DIMS))
        p.add_argument("--threshold", type=float, default=0.70, help="表型「受损」阈值（chance-corrected 尺度）")
        p.add_argument("--dose", type=float, default=None, help="只在指定 dose 上做机制分析")

    p = sub.add_parser("ontology", help="打印认知本体与 assay，并做自检")
    p.set_defaults(func=cmd_ontology)

    p = sub.add_parser("vocab", help="打印计分通道、观察锚点与探查登记（不做机制→维度映射）")
    p.set_defaults(func=cmd_vocab)

    p = sub.add_parser("prompt", help="打印某一扰动下的人造大脑提示词")
    p.add_argument("--primitive", default="intact")
    p.add_argument("--dose", type=float, default=0.4)
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("init-brain", help="先通过提示词初始化人造大脑并查看自我报告")
    add_llm_options(p)
    p.add_argument("--primitive", default="intact", help="intact | generic | <primitive_id>")
    p.add_argument("--dose", type=float, default=0.4)
    p.set_defaults(func=cmd_init_brain)

    p = sub.add_parser("sample", help="预览扰动设计规模（不调用模型）")
    add_run_options(p)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("runs", help="列出本地运行记录")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("run", help="运行扰动设计并生成表型")
    add_llm_options(p)
    add_run_options(p)
    p.add_argument("--endpoint", default="social_communication_pragmatics", choices=tuple(PHENOTYPE_DIMS))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="机制 / 分型 / 特征差异分析")
    add_analysis_options(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report", help="生成 Atlas 报告（Markdown + JSON）")
    add_analysis_options(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("freeze", help="生成冻结快照")
    add_analysis_options(p)
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("serve", help="启动可视化界面（浏览器操作，推荐入口）")
    add_llm_options(p)
    p.add_argument("--host", default="127.0.0.1", help="只建议本机地址")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--no-auto-init", action="store_true",
                   help="跳过启动时自动初始化正常大脑（CI / 自动化保持确定性）")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("brains", help="列出大脑库（可保存/切换的模拟大脑）")
    p.set_defaults(func=cmd_brains)

    p = sub.add_parser("bank", help="浏览观察者内置条目库（来源：AsdKB）")
    p.add_argument("--dim", default=None, choices=tuple(PHENOTYPE_DIMS), help="按表型维度过滤")
    p.add_argument("--primitive", default=None, choices=DEFAULT_ACTIVE, help="按可扰动基础过程过滤")
    p.add_argument("--admin", default=None,
                   choices=("caregiver_report", "clinician_rating", "self_report", "observation"),
                   help="按施测方式过滤")
    p.add_argument("--scale", default=None, help="按量表 id 过滤（如 tools19）")
    p.add_argument("--probe-ready", action="store_true", help="只看可探查条目（有探查语句）")
    p.add_argument("--probe-mode", default=None, choices=("question", "scenario", "none"),
                   help="按探查方式过滤：直接提问 / 情境式 / 不可探查")
    p.add_argument("--age", type=int, default=None, help="按年龄（月龄）过滤")
    p.add_argument("--query", default=None, help="关键词（题干 / 探查语句 / 量表名）")
    p.add_argument("--layer", default=None, choices=("symptoms", "criteria", "treatments"),
                   help="改为打印参考层：症状 / 诊断标准 / 循证干预")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--rules", action="store_true", help="只打印映射规则与来源摘要")
    p.set_defaults(func=cmd_bank)

    p = sub.add_parser("dialogue", help="表型观察：对话探针（提问者 → 模拟大脑 → 分析者）")
    add_llm_options(p)
    p.add_argument("--primitive", default=None, help="要扰动的 primitive（默认完整心智）")
    p.add_argument("--dose", type=float, default=0.2, help="完整度（越低扰动越强）")
    p.add_argument("--turns", type=int, default=None,
                   help="基础对话轮数（默认 35 = 7 个计分通道 × 每通道 5 个问题）")
    p.add_argument("--max-turns", type=int, default=None,
                   help="轮数上限，含按配额自动补问的轮次（默认 100，硬顶 500）")
    p.add_argument("--text-limit", type=int, default=None,
                   help="问题与回答的字数上限（默认 100；超出只告警，不截断）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-mode", default=None, choices=("auto", "scenario", "question"),
                   help="探查方式：情境式（设定场景后问「这种情况你会怎么处理」）/ 直接提问")
    p.set_defaults(func=cmd_dialogue)

    p = sub.add_parser("demo", help="一键离线跑通全链路（Mock 心智，无需 Key）")
    add_run_options(p)
    p.add_argument("--endpoint", default="social_communication_pragmatics", choices=tuple(PHENOTYPE_DIMS))
    p.add_argument("--threshold", type=float, default=0.70)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
