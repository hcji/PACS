"""虚拟被试（FR-7）：把 LLM 当作一个被扰动过的人工心智来使用。

一个 `VirtualAgent` = 一份认知架构系统提示（由 θ 生成）+ 一个 provider。
`initialize()` 是**先通过提示词初始化**这一步：先让心智确认架构并报告自状态，
再开始接受观察 —— 保证扰动在任务之前已经生效，且初始化动作可审计。

**它只出声，不判分**：心智对提问者的每一轮问题自由作答，不做选择式应答；
读表型由分析者 LLM 负责。
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from ..brain import prompts
from ..brain.architecture import (
    brain_fingerprint,
    build_brain_prompt,
    build_brain_turn,
    build_init_prompt,
)
from ..brain.perturb import Perturbation
from ..llm.parse import has_self_state_ok
from ..llm.provider import ChatMessage, LLMProvider
from .. import DEFAULT_TEXT_LIMIT


def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class BrainInitResult:
    """大脑初始化的结果记录（审计用）。"""

    model_id: str
    theta_id: str
    ok: bool
    self_report: str
    raw: str
    ts: str = field(default_factory=now_iso)


class VirtualAgent:
    """一个由 θ 定义的人工心智。"""

    def __init__(
        self,
        provider: LLMProvider,
        perturbation: Perturbation,
        *,
        model_id: str | None = None,
        seed: int = 42,
        temperature: float | None = None,
        text_limit: int = DEFAULT_TEXT_LIMIT,
    ) -> None:
        self.provider = provider
        self.perturbation = perturbation
        self.model_id = model_id or f"{getattr(provider, 'name', 'unknown')}:{getattr(provider, 'model', '?')}"
        self.seed = seed
        self.temperature = temperature
        # 回答的字数上限：与提问者的问题上限是同一个数（见 `pacs.DEFAULT_TEXT_LIMIT`），
        # 写进架构提示的「行为准则」——两侧都约束，超长只告警不截断。
        self.text_limit = text_limit
        self._system_prompt: str | None = None

    # ---------- 身份 ----------

    @property
    def theta_id(self) -> str:
        return self.perturbation.theta_id

    @property
    def fingerprint(self) -> str:
        return brain_fingerprint(self.perturbation)

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_brain_prompt(
                self.perturbation, text_limit=self.text_limit
            )
        return self._system_prompt

    # ---------- 初始化（提示词层面的「上线」）----------

    def initialize(self) -> BrainInitResult:
        """先通过提示词把心智初始化，再开始任务。"""
        messages = [
            ChatMessage("system", self.system_prompt),
            ChatMessage("user", build_init_prompt()),
        ]
        raw = self.provider.chat(
            messages, temperature=self.temperature, seed=self.seed
        )
        self._messages_prefix = messages
        return BrainInitResult(
            model_id=self.model_id,
            theta_id=self.theta_id,
            ok=has_self_state_ok(raw),
            self_report=raw.strip()[:400],
            raw=raw,
        )

    # ---------- 作答 ----------

    def reply(
        self,
        history: list[ChatMessage],
        *,
        turn: int = 0,
        item: object = None,
        stage: str = "probe",
        turn_index: int | None = None,
        turn_total: int | None = None,
    ) -> str:
        """表型观察 / 直接提问里的一句回应（自由文本）。

        `history` 是已经积累的对话消息（提问者的 `user` / 心智的 `assistant` 交替），
        系统提示由本方法置于首位，调用方不必自己拼。
        `item` 是本轮使用的内置探查条目（可选）：真实 provider 忽略它，mock 用它把
        「选择性受损」算出来 —— 只有知道这一轮问的是哪条探查，才谈得上按维度区分回应质量。

        按协议阶段框住这一轮（只影响**发给模型的这一份**，history 里仍是提问者的原话）：
        - `stage="probe"` —— 表型观察的对话探针：完整心智走阶段 2「基础认知校准」，
          被扰动的心智走阶段 7「连续行为任务」（`turn_index`/`turn_total` 是它在序列里的位置）；
        - `stage="interview"` —— ③ 里由用户本人提问的那一路，按阶段 6「单次任务提问」包一层
          （不刻意正常或异常、不展示也不回避某种特征、不推断实验目的）。
        """
        messages = [ChatMessage("system", self.system_prompt), *history]
        if messages[-1].role == "user" and stage == "interview":
            messages[-1] = ChatMessage(
                "user",
                prompts.single_task(
                    task="回答提问者的问题（这是一次直接访谈，不是量表任务）",
                    stimulus="（见 Question）",
                    question=messages[-1].content,
                ),
            )
        elif messages[-1].role == "user":
            index = turn_index if turn_index is not None else max(1, turn)
            total = turn_total if turn_total is not None else index
            messages[-1] = ChatMessage(
                "user",
                build_brain_turn(
                    messages[-1].content,
                    index=index,
                    total=total,
                    intact=self.perturbation.is_intact,
                ),
            )
        return self.provider.chat(
            messages,
            temperature=self.temperature,
            perturbation=self.perturbation,
            seed=self.seed + turn,
            role="brain",
            item=item,
        )

    def __repr__(self) -> str:  # pragma: no cover - 展示用
        return f"VirtualAgent({self.model_id}, {self.perturbation.theta_id})"
