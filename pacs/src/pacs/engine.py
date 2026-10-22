"""运行引擎：把「扰动设计 × 模型」跑成表型数据（FR-7 的批量编排层）。

一条设计 = 一个虚拟被试。对每个被试：先用提示词初始化心智（`initialize`），再测量表型。

**测量只有一条通道**：提问者 LLM 与被扰动的心智对话，分析者 LLM 从这段对话里读出
逐维表型 —— 这个读数就是表型向量 `B`。表型结构要么由 LLM 涌现，要么就没有。

逐条落盘（由 `store` 负责），因此中断后可跳过已完成被试（NFR-5）。
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from . import DEFAULT_TEXT_LIMIT
from .agents.measure import measure_phenotype
from .agents.virtual_agent import BrainInitResult, VirtualAgent
from .brain.perturb import Perturbation
from .llm.provider import LLMError, LLMProvider
from .phenotype import PhenotypeProfile

ProgressFn = Callable[[int, int, Perturbation], None]
DialogueFn = Callable[[object, dict], None]

#: 测量工具的记录值；用于在报告里写出「这批剖面是怎么测出来的」。
INSTRUMENT = "dialogue"


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class DesignRun:
    """一次设计运行的全部产物。"""

    run_id: str
    model_id: str
    designs: list[Perturbation]
    profiles: list[PhenotypeProfile] = field(default_factory=list)
    init_results: list[BrainInitResult] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)   # (theta_id, 错误信息)
    instrument: str = INSTRUMENT

    @property
    def init_ok_rate(self) -> float:
        if not self.init_results:
            return 0.0
        return sum(1 for r in self.init_results if r.ok) / len(self.init_results)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "instrument": self.instrument,
            "n_designs": len(self.designs),
            "n_profiles": len(self.profiles),
            "init_ok_rate": round(self.init_ok_rate, 4),
            "failures": self.failures,
        }


def run_design(
    provider: LLMProvider,
    designs: Iterable[Perturbation],
    *,
    run_id: str | None = None,
    model_id: str | None = None,
    seed: int = 42,
    initialize: bool = True,
    progress: ProgressFn | None = None,
    on_profile: Callable[[PhenotypeProfile], None] | None = None,
    examiner: LLMProvider | None = None,
    analyst: LLMProvider | None = None,
    probes: Sequence[dict] = (),
    probe_mode: str = "auto",
    dialogue_turns: int | None = None,
    dialogue_max_turns: int | None = None,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    rubric_items: Sequence[dict] = (),
    on_dialogue: DialogueFn | None = None,
) -> DesignRun:
    """跑完一整个扰动设计。

    `initialize=True` 时每个被试先做一次提示词初始化（可审计的「大脑上线」）；
    初始化失败不中断运行（记录进 `failures`），以便在报告中暴露协议问题。

    表型由一次对话探针涌现：提问者 LLM 提问、心智作答、分析者 LLM 给出逐维读数。
    `examiner`/`analyst` 缺省时退回脑用的 `provider`（mock provider 按 `role` 分派，
    因此离线链路完全不需另配 API）。

    **年龄不是参数**：心智年龄固定为 `DEFAULT_AGE_MONTHS`（见该常量）；`dialogue_turns`
    是基础轮数、`dialogue_max_turns` 是补问上限、`text_limit` 是问题与回答的字数上限。
    """
    design_list = list(designs)
    resolved_model_id = model_id or f"{getattr(provider, 'name', 'unknown')}:{getattr(provider, 'model', '?')}"
    run = DesignRun(
        run_id=run_id or new_run_id(),
        model_id=resolved_model_id,
        designs=design_list,
    )
    examiner_provider = examiner or provider
    analyst_provider = analyst or provider
    total = len(design_list)
    replicate_seen: dict[str, int] = {}
    for index, perturbation in enumerate(design_list, start=1):
        # 同一个 θ 可能出现多次（参照条件需要重复测量以降低噪声）→ 记为不同 replicate，
        # 否则它们在库中会因主键相同而互相覆盖。
        replicate = replicate_seen.get(perturbation.theta_id, 0)
        replicate_seen[perturbation.theta_id] = replicate + 1
        agent = VirtualAgent(
            provider,
            perturbation,
            model_id=resolved_model_id,
            seed=seed + index,
            text_limit=text_limit,
        )
        if initialize:
            try:
                init_result = agent.initialize()
                run.init_results.append(init_result)
                if not init_result.ok:
                    run.failures.append(
                        (perturbation.theta_id, "初始化未返回 SELF_STATE: OK（协议标记缺失）")
                    )
            except LLMError as exc:
                run.failures.append((perturbation.theta_id, f"初始化失败：{exc}"))

        try:
            measured = measure_phenotype(
                agent,
                examiner_provider,
                analyst_provider,
                run_id=run.run_id,
                probes=probes,
                probe_mode=probe_mode,
                n_turns=dialogue_turns,
                max_turns=dialogue_max_turns,
                seed=seed + index,
                text_limit=text_limit,
                rubric_items=rubric_items,
                # 会话 id 必须逐被试唯一：批量运行时一秒内会产生几十段对话，
                # 用默认的时间戳 id 会互相覆盖，丢掉除最后一个之外的全部记录。
                session_id=f"{run.run_id}-{index:03d}",
            )
        except LLMError as exc:
            run.failures.append((perturbation.theta_id, f"对话测量失败：{exc}"))
            if progress:
                progress(index, total, perturbation)
            continue
        profile = measured.profile
        profile.replicate = replicate
        run.profiles.append(profile)
        if on_dialogue:
            on_dialogue(measured.session, measured.feedback)
        if on_profile:
            on_profile(profile)
        if progress:
            progress(index, total, perturbation)
    return run


def run_models(
    providers: dict[str, LLMProvider],
    designs: Iterable[Perturbation],
    *,
    run_id_prefix: str = "run",
    **kwargs,
) -> dict[str, DesignRun]:
    """对多个模型跑同一设计（FR-10 跨模型对照）。"""
    design_list = list(designs)
    runs: dict[str, DesignRun] = {}
    for label, provider in providers.items():
        runs[label] = run_design(
            provider,
            design_list,
            run_id=f"{run_id_prefix}_{label}",
            model_id=f"{getattr(provider, 'name', label)}:{getattr(provider, 'model', '?')}",
            **kwargs,
        )
    return runs
