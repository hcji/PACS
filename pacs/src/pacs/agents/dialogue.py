"""表型观察的对话探针（规则 5）：**另外的 LLM** 提问 → 模拟大脑作答 → 分析者形成表型反馈。

三方分工（这是「盲法」的落点）：
- **模拟大脑** —— 由 θ 定义的人工心智，**只有它**拿到完整度参数（θ 只作它的提示词输入）；
- **提问者**（另外的 LLM）—— 提出日常情境与问题，**看不到 θ**；
- **分析者**（另外的 LLM）—— 读完对话后给出各维度的表型读数，**同样看不到 θ**。

**两道隔离，都必须守住**：
1. **θ 只能输入给 LLM。** 本模块不把 θ 交给任何别的东西：提示词里没有，`analyze_dialogue`
   的入参里也没有 —— 因此它**不可能**把 θ 写进读数。
2. **表型只能由提问与观察得到。** 读数只来自「提问者提问 → 大脑作答 → 分析者读对话」；
   分析者的输出里不附带 θ 基准真值，观察通路上也就不存在「读数与扰动并排」这种已知条件的
   关联。要比较 θ 与读数，那是**测量之外**的工作（`pacs analyze` 等离线分析），不在这里。

表型读数方向与 θ 一致：**1.0 = 典型/完好，0.0 = 严重受损**。
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..brain.architecture import (
    DEFAULT_GOALS,
    build_analyst_prompt,
    build_analyst_system_prompt,
    build_examiner_system_prompt,
    build_examiner_turn_prompt,
    render_transcript,
)
from .. import (
    DEFAULT_AGE_MONTHS,
    DEFAULT_TEXT_LIMIT,
    MAX_TEXT_LIMIT,
    QUESTIONS_PER_CHANNEL,
)
from ..phenotype import CONTROL_DIM, SCORED_CHANNELS
from ..brain.examiner_bank import render_probe_plan, render_rubric
from ..llm.parse import (
    parse_analyst_json,
    parse_examiner_focus,
    parse_examiner_question,
    presentation_of,
)
from ..llm.provider import ChatMessage, LLMError, LLMProvider

#: 基础轮数 = 通道数 × 每通道问题数（7 × 5 = 35）。跑完基础轮数后仍有通道不足 5 个问题，
#: 就会**继续补问**（见 `run_dialogue` 的 `must_focus`），直到每通道够数或到 `max_turns`。
DEFAULT_N_TURNS = QUESTIONS_PER_CHANNEL * len(SCORED_CHANNELS)

#: 提问者一轮最多尝试几次产出可解析的问题。三次而不是两次：真实模型会偶发返回空内容或
#: 整个丢掉格式，而一次失败会终止整段对话（协议下限 35 轮就此作废）；多给一次机会的
#: 成本只是两次调用，收益是整段会话不残。
EXAMINER_ATTEMPTS = 3

#: 分析者的输出额度不再由这里给：它是**按角色**在 provider 里兜底的下限
#: （`llm/provider.py` 的 `ANALYST_MIN_TOKENS`）—— 放在会被调用的那一层，配置里调大
#: `max_tokens` 才能对它生效。这一层只负责声明「这一轮是分析者」（`role="analyst"`）。

#: 对话轮数上限的**默认值**（可由 CLI / API / 界面自定义，`max_turns` 参数）。
#: 成本 = 每轮 2 次调用（提问者 + 大脑），末尾再 +1 次分析者：100 轮 ≈ 201 次调用。
MAX_N_TURNS = 100

#: 上限的硬顶：防止误填一个天文数字把额度打光。超过它一律拒绝。
ABSOLUTE_MAX_TURNS = 500

TurnFn = Callable[[dict], None]


def new_session_id() -> str:
    return "dlg_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class DialogueSession:
    """一段对话探针会话：提问者与模拟大脑的逐轮记录。"""

    session_id: str
    theta_id: str
    brain_id: str = ""
    brain_model: str = ""
    examiner_model: str = ""
    age_months: int = DEFAULT_AGE_MONTHS       # 固定 36 月，不可配置（见 DEFAULT_AGE_MONTHS）
    n_turns: int = 0                            # 基础轮数（计划的「每通道 ≥5」下限）
    max_turns: int = MAX_N_TURNS                # 本轮允许的最大轮数（含补问）
    n_turns_run: int = 0                        # 实际发起的轮数（含补问轮）
    goals: tuple[str, ...] = ()
    seed: int = 42
    text_limit: int = DEFAULT_TEXT_LIMIT        # 问题/回答的字数上限（超出只告警）
    turns: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: 内置题库出处记账：这次对话用了哪些内置题、覆盖了哪些维度/primitive
    #: "probe" = 提问者 LLM 驱动；"interview" = 用户直接提问
    mode: str = "probe"
    probe_mode: str = "auto"
    probe_keys: list[str] = field(default_factory=list)
    probe_sources: list[str] = field(default_factory=list)
    probe_dims: list[str] = field(default_factory=list)
    probe_primitives: list[str] = field(default_factory=list)
    #: 逐轮自报的方面（提问者 FOCUS）；自行出题的轮次也据此归属
    focus_channels: list[str] = field(default_factory=list)
    #: 逐通道的**问题计数**（按提问者自报的 FOCUS 记 1；该轮焦点为空/非法则不计入任何通道）。
    #: 这是「每通道至少 5 个问题」的判据来源，也是补问轮选目标的依据。
    question_counts: dict[str, int] = field(default_factory=dict)
    #: 整段对话**一个问题都没有**的计分通道：它们的读数没有证据支撑，必须能被下游
    #: （剖面/报告）读出来，而不是静默当成典型。
    uncovered_channels: list[str] = field(default_factory=list)
    #: 被问到过、但问题数不足「每通道 5 个」的通道 —— 证据偏薄，读数要打折看。
    under_quota_channels: list[str] = field(default_factory=list)
    #: 超出字数上限的轮数（只告警、不截断）：问题或回答任一超长即计入。
    over_limit_turns: int = 0
    ts: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def transcript(self) -> str:
        return render_transcript(self.turns)

    @property
    def n_brain_turns(self) -> int:
        return sum(1 for t in self.turns if t.get("role") == "brain")

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "theta_id": self.theta_id,
            "brain_id": self.brain_id,
            "brain_model": self.brain_model,
            "examiner_model": self.examiner_model,
            "age_months": self.age_months,
            "n_turns": self.n_turns,
            "max_turns": self.max_turns,
            "n_turns_run": self.n_turns_run,
            "goals": list(self.goals),
            "seed": self.seed,
            "text_limit": self.text_limit,
            "turns": self.turns,
            "problems": self.problems,
            "mode": self.mode,
            "probe_mode": self.probe_mode,
            "probe_keys": self.probe_keys,
            "probe_sources": self.probe_sources,
            "probe_dims": self.probe_dims,
            "probe_primitives": self.probe_primitives,
            "focus_channels": self.focus_channels,
            "question_counts": self.question_counts,
            "uncovered_channels": self.uncovered_channels,
            "under_quota_channels": self.under_quota_channels,
            "over_limit_turns": self.over_limit_turns,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DialogueSession":
        return cls(
            session_id=str(data.get("session_id") or new_session_id()),
            theta_id=str(data.get("theta_id") or ""),
            brain_id=str(data.get("brain_id") or ""),
            brain_model=str(data.get("brain_model") or ""),
            examiner_model=str(data.get("examiner_model") or ""),
            age_months=DEFAULT_AGE_MONTHS,
            n_turns=int(data.get("n_turns") or 0),
            max_turns=int(data.get("max_turns") or MAX_N_TURNS),
            n_turns_run=int(data.get("n_turns_run") or 0),
            goals=tuple(data.get("goals") or ()),
            seed=int(data.get("seed") or 42),
            text_limit=int(data.get("text_limit") or DEFAULT_TEXT_LIMIT),
            turns=list(data.get("turns") or []),
            problems=list(data.get("problems") or ()),
            mode=str(data.get("mode") or "probe"),
            probe_mode=str(data.get("probe_mode") or "auto"),
            probe_keys=list(data.get("probe_keys") or ()),
            probe_sources=list(data.get("probe_sources") or ()),
            probe_dims=list(data.get("probe_dims") or ()),
            probe_primitives=list(data.get("probe_primitives") or ()),
            focus_channels=list(data.get("focus_channels") or ()),
            question_counts={
                str(k): int(v) for k, v in (data.get("question_counts") or {}).items()
            },
            uncovered_channels=list(data.get("uncovered_channels") or ()),
            under_quota_channels=list(data.get("under_quota_channels") or ()),
            over_limit_turns=int(data.get("over_limit_turns") or 0),
            ts=str(data.get("ts") or ""),
        )


def run_dialogue(
    brain,
    examiner: Optional[LLMProvider] = None,
    *,
    questions: Sequence[str] | None = None,
    probe_items: Sequence[dict] | None = None,
    probe_mode: str = "auto",
    n_turns: int = DEFAULT_N_TURNS,
    max_turns: int = MAX_N_TURNS,
    adapt: bool = True,
    goals: Sequence[str] = (),
    seed: int = 42,
    session_id: str | None = None,
    brain_id: str = "",
    text_limit: int = DEFAULT_TEXT_LIMIT,
    on_turn: TurnFn | None = None,
) -> DialogueSession:
    """驱动一段对话探针。

    提问内容的三种来源（优先级从高到低）：
    1. `questions` —— 固定问题清单（离线回放 / 上传量表）；
    2. `probe_items` —— **内置题库**选出的探查题（`brain/examiner_bank.py`）：它们的
       `probe_zh` 会作为「计划探查的方面」写进提问者的系统提示，由 LLM 用自己的话提问；
    3. 都不给 —— 提问者自由发挥（最贴近「另外的 LLM 提问」的原始形态）。

    每产生一轮就调用 `on_turn(turn)` —— 界面据此做**逐轮可视化**（服务端把它写进任务状态，
    前端轮询增量渲染，因此不需要 SSE）。

    **轮数与补问**：先走 `n_turns` 个基础轮，然后按**配额**决定是否继续 —— 每个计分通道
    至少要 `QUESTIONS_PER_CHANNEL` 个问题（按提问者自报的 `FOCUS` 计数，一条题库条目即使
    覆盖多个维度也只记 1）。仍有通道不够时，逐轮把**缺口最紧的那一个**写进提问者的当轮指令
    （`must_focus`）继续补问，直到每通道够数或到 `max_turns` 停。
    `adapt=False`（或给了固定 `questions` 清单）时是定长对话：跑满 `n_turns` 就结束 ——
    固定清单没有「再生成一个问题」的来源，补问无从谈起。

    **字数**：`text_limit` 同时渲染进两侧提示词（提问者的问题、心智的回答）。超长的轮次
    **照常记录、不截断、不重试**，只在该轮标 `over_limit` 并在结尾记一条汇总问题。
    """
    if not (1 <= n_turns <= ABSOLUTE_MAX_TURNS):
        raise ValueError(f"轮数必须在 1..{ABSOLUTE_MAX_TURNS} 之间，收到 {n_turns}")
    if not (n_turns <= max_turns <= ABSOLUTE_MAX_TURNS):
        raise ValueError(
            f"轮数上限必须在 {n_turns}..{ABSOLUTE_MAX_TURNS} 之间（且不小于基础轮数），"
            f"收到 {max_turns}"
        )
    if not (1 <= text_limit <= MAX_TEXT_LIMIT):
        raise ValueError(f"字数上限必须在 1..{MAX_TEXT_LIMIT} 之间，收到 {text_limit}")
    if examiner is None and not questions:
        raise ValueError("必须提供 examiner 或 questions 之一")

    goals = tuple(goals) if goals else tuple(SCORED_CHANNELS)
    probe_items = list(probe_items or ())
    fixed = bool(questions)
    # 定长模式（固定清单 / 显式关闭自适应）跑满基础轮数；自适应模式允许补到上限
    run_limit = n_turns if (fixed or not adapt) else max_turns
    session = DialogueSession(
        session_id=session_id or new_session_id(),
        theta_id=brain.theta_id,
        brain_id=brain_id,
        brain_model=brain.model_id,
        examiner_model=f"{examiner.name}:{examiner.model}" if examiner else "（固定问题清单）",
        age_months=DEFAULT_AGE_MONTHS,
        n_turns=n_turns,
        max_turns=max_turns,
        goals=goals,
        seed=seed,
        text_limit=text_limit,
    )
    # 题库出处记账：用了哪些内置题、它们各自映射到哪些维度/primitive，都可追溯
    if probe_items:
        session.probe_keys = [str(i.get("key")) for i in probe_items]
        session.probe_sources = sorted({str(i.get("source_id")) for i in probe_items})
        session.probe_dims = sorted({d for i in probe_items for d in (i.get("dims") or {})})
        session.probe_primitives = sorted(
            {p for i in probe_items for p in (i.get("primitives") or {})}
        )

    session.probe_mode = probe_mode
    plan = render_probe_plan(probe_items, mode=probe_mode) if probe_items else ""
    # 计划本身覆盖了哪些通道（不是「问过哪些」）：mock 提问者据此判断某个方面是否
    # 题库里已有参考条目 —— 没有就要自己出题（规则 7/8 的离线版）。
    plan_dims = tuple(sorted({d for i in probe_items for d in (i.get("dims") or {})}))
    system_prompt = build_examiner_system_prompt(
        goals=goals, turn_budget=n_turns, probe_plan=plan,
        # 把「题库里没有条目的方面」点名要求自造题：否则这件事是静默的。
        # 注意这只是**探查设计的覆盖情况**，与 θ、与「机制该体现在哪个维度」都无关。
        covered_dims=plan_dims,
        text_limit=text_limit,
    )
    history: list = []          # 给模拟大脑的多轮消息历史（user/assistant 交替）

    question_counts: dict[str, int] = {g: 0 for g in goals}
    unattributed_turns: list[int] = []      # 自报焦点缺失/非法的轮次（不计入任何通道）
    over_limit_turns: list[int] = []        # 问题或回答超出字数上限的轮次
    turn_index = 0
    while turn_index < run_limit:
        turn_index += 1
        questions_asked = sum(question_counts.values())
        # 补问目标：缺口最紧的通道（并列按 goals 顺序）。基础轮内不指定 ——
        # 基础轮的目的是把每个方面各铺开一遍，那时按缺口指派反而会让前几轮反复问同一件事。
        must_focus = ""
        if not fixed and adapt and turn_index > n_turns:
            deficits = _quota_deficits(question_counts, goals)
            must_focus = deficits[0] if deficits else ""
        question = ""
        presentation = ""
        focus = ""
        if questions:
            question = str(questions[turn_index - 1]).strip() if turn_index <= len(questions) else ""
            if not question:
                session.problems.append(f"第 {turn_index} 轮：问题清单已用尽")
                break
            presentation = question
        else:
            # 把「还没被探查过的方面」逐轮告诉提问者：题库只是参考，选题要有依据，
            # 否则轮数用满之后剩下的通道只能靠运气（分析者对它们只能给保守值）
            remaining = tuple(g for g in goals if question_counts.get(g, 0) == 0)
            question, presentation, focus = _ask(
                examiner, system_prompt, session, turn_index, n_turns, seed,
                remaining=remaining,
                must_focus=must_focus,
                final=(turn_index == run_limit),
                # 这一轮的下发条目（可能为 None = 由提问者自行出题）：
                # 真实 provider 只把它当成「计划里的一条参考」，mock 用它保证
                # 「问的话 / 记录的 probe_key / 自报的 FOCUS」三者一致
                item=probe_items[turn_index - 1] if turn_index <= len(probe_items) else None,
                plan_dims=plan_dims,
            )

        if not question:
            session.problems.append(
                f"第 {turn_index} 轮：提问者连续两次没有产出可解析的问题，对话在此结束"
            )
            break

        # 记录与下发都用**完整呈现**（情境 + 提问），而不是只留解析出的那一句问题：
        # 情境式探查里儿童回应的是整个情境，分析者也需要知道当时设定了什么。
        focus = focus if focus in goals else ""
        if focus:
            # 计数口径：按提问者**自报的焦点**记 1 个问题（不按条目覆盖的维度去摊）
            question_counts[focus] = question_counts.get(focus, 0) + 1
        else:
            unattributed_turns.append(turn_index)
        # 这一轮到底用了哪条条目：由 `_brain_probe` 判定（它问的是「这一轮在问的方面在不在
        # 该条目的维度里」）。自行出题的轮次记为 `None`，而不是照抄位置上的那条 —— 否则记录会
        # 声称用了一条实际没用的条目，`probe_key` 就成了假出处。
        probe = _brain_probe(probe_items, turn_index, focus)
        used_key = None if probe is None or probe.get("improvised") else probe.get("key")
        examiner_turn = {
            "turn": turn_index,
            "role": "examiner",
            "content": presentation,
            "question": question,
            "probe_key": used_key,
            # 这一轮**主要**在问哪个通道：提问者自报（FOCUS）。自行出题的轮次靠它归属，
            # 否则逐轮归属里会出现空洞，分析者与离线 mock 都失去维度线索。
            "focus": focus,
            "chars": len(presentation),
            "over_limit": len(presentation) > text_limit,
        }
        if examiner_turn["over_limit"]:
            over_limit_turns.append(turn_index)
        session.turns.append(examiner_turn)
        if on_turn:
            on_turn(dict(examiner_turn))

        history.append(_user(presentation))
        answer = brain.reply(
            history,
            turn=turn_index,
            item=probe,
            # 逐轮的协议阶段框（阶段 2 / 阶段 7）：对话本身是一串连续的行为决策。
            # 总数用**本次会话允许的上限**（而不是基础轮数）：自适应补问会把轮数拖到 35 以后，
            # 拿基础轮数去标就会出现「第 36/35 轮」这种自相矛盾的框。
            turn_index=turn_index,
            turn_total=run_limit,
        )
        history.append(_assistant(answer))

        brain_turn = {
            "turn": turn_index,
            "role": "brain",
            "content": answer,
            "chars": len(answer),
            "over_limit": len(answer) > text_limit,
        }
        if brain_turn["over_limit"]:
            over_limit_turns.append(turn_index)
        session.turns.append(brain_turn)
        if on_turn:
            on_turn(dict(brain_turn))

        # 配额已满就收工：基础轮走完之前不停（基础轮本来就要铺满每个方面），
        # 之后只要没有缺口就没有继续补问的理由 —— 多问一轮就是多两次模型调用。
        if turn_index >= n_turns and not fixed and adapt:
            if not _quota_deficits(question_counts, goals):
                break

    session.n_turns_run = sum(1 for t in session.turns if t.get("role") == "examiner")
    session.focus_channels = [
        str(t.get("focus") or "") for t in session.turns if t.get("role") == "examiner"
    ]
    session.question_counts = {g: question_counts.get(g, 0) for g in goals}
    counts = session.question_counts
    # 三类通道分开记：**一个问题都没有**（没有证据）与**问到过但不够配额**（证据偏薄）
    # 是两件不同的事，下游（分析者提示词、剖面）对它们的处理也不同。
    session.uncovered_channels = [g for g in goals if counts.get(g, 0) == 0]
    session.under_quota_channels = [
        g for g in goals if 0 < counts.get(g, 0) < QUESTIONS_PER_CHANNEL
    ]
    session.over_limit_turns = len(set(over_limit_turns))
    if session.uncovered_channels:
        # 不静默：整段对话都没问到的通道没有证据，分析者只能给保守值 —— 这必须留在记录里，
        # 而且要结构化（`uncovered_channels`）以便下游把它标成「无证据」而不是「表现典型」。
        session.problems.append(
            "以下方面整段对话都没有被探查到，其读数没有证据支撑："
            + "、".join(SCORED_CHANNELS.get(g, g) for g in session.uncovered_channels)
            + f"（覆盖 {len(goals) - len(session.uncovered_channels)}/{len(goals)} 个方面）"
        )
    if session.under_quota_channels:
        session.problems.append(
            f"以下方面被问到过，但问题数不足 {QUESTIONS_PER_CHANNEL} 个（证据偏薄，读数要打折看）："
            + "、".join(
                f"{SCORED_CHANNELS.get(g, g)}（{counts.get(g, 0)} 个）"
                for g in session.under_quota_channels
            )
            + f"；已问 {session.n_turns_run} 轮（上限 {max_turns}）"
        )
    if unattributed_turns:
        session.problems.append(
            "以下轮次提问者没有自报有效的焦点，该轮不计入任何通道的问题数："
            + "、".join(f"第 {i} 轮" for i in unattributed_turns[:8])
            + (f" 等 {len(unattributed_turns)} 轮" if len(unattributed_turns) > 8 else "")
        )
    if over_limit_turns:
        session.problems.append(
            f"有 {session.over_limit_turns} 轮的问题或回答超过 {text_limit} 字上限"
            "（照原样记录，未截断）："
            + "、".join(f"第 {i} 轮" for i in sorted(set(over_limit_turns))[:8])
            + (f" 等 {session.over_limit_turns} 轮" if session.over_limit_turns > 8 else "")
        )
    return session


def _quota_deficits(question_counts: dict[str, int], goals: Sequence[str]) -> list[str]:
    """还没问够 `QUESTIONS_PER_CHANNEL` 个问题的通道，**缺口最紧的排前面**。

    并列时按 `goals` 顺序（调用方给的目标顺序）—— 必须有确定的次序，否则补问轮的目标
    会随字典顺序漂移，同一段对话在不同机器上补出不同的内容。
    """
    order = {goal: index for index, goal in enumerate(goals)}
    deficits = [g for g in goals if question_counts.get(g, 0) < QUESTIONS_PER_CHANNEL]
    deficits.sort(key=lambda g: (question_counts.get(g, 0), order[g]))
    return deficits


def _ask(examiner, system_prompt: str, session: DialogueSession, turn_index: int,
         n_turns: int, seed: int, *,
         remaining: tuple[str, ...] = (),
         must_focus: str = "",
         final: bool = False,
         item: dict | None = None,
         plan_dims: tuple[str, ...] = ()) -> tuple[str, str, str]:
    """向提问者要一次呈现；空则重试一次（第二次加重语气要求只输出问题）。

    返回 `(解析出的问题, 完整呈现, 自报方面)` —— 情境式探查下前两者不同：呈现包含情境描述。

    `must_focus` 是补问轮必须探查的那一个方面（由调用方按配额缺口算出）；`final` 表示
    这一轮确实是最后一轮（轮数上限用尽）。

    `item` / `plan_dims` 只对 mock 有意义（真实 provider 用不到，它在提示词里已经看到计划）：
    题库里有**多条条目的探查语句逐字相同**，靠文本回查认不出这一轮用的是哪一条，
    于是自报的 `FOCUS` 可能落到错误的通道上。把它们传下去，mock 就不必猜。
    """
    transcript = session.transcript()
    # 三次机会：真实模型偶尔会返回空内容或把输出格式整个丢掉，而**一次失败就终止整段
    # 对话**的代价太大 —— 基础 35 轮的协议会退化成两三轮的残缺会话，后面所有读数都没证据。
    for attempt in range(EXAMINER_ATTEMPTS):
        prompt = build_examiner_turn_prompt(
            turn_index=turn_index, turn_budget=n_turns, transcript=transcript,
            remaining=remaining, must_focus=must_focus, final=final,
        )
        if attempt:
            prompt += (
                "\n\n（上一次没有收到可解析的问题。请**只**按下面三行输出，不要解释、不要用代码块：\n"
                "第一行：问题或情境本身；\n"
                "第二行：`QUESTION: ` 加同一句问题；\n"
                "第三行：`FOCUS: ` 加一个方面 id。）"
            )
        raw = examiner.chat(
            [_system(system_prompt), _user(prompt)],
            role="examiner",
            seed=seed + turn_index - 1,
            context={
                "probe_item": item,
                "plan_dims": list(plan_dims),
                "must_focus": must_focus,
            },
        )
        question = parse_examiner_question(raw)
        if question:
            return question, presentation_of(raw, question), parse_examiner_focus(raw)
    return "", "", ""


def _brain_probe(probe_items: list, turn_index: int, focus: str) -> dict | None:
    """这一轮下发给模拟大脑的「探查条目」。

    用题库条目（它有逐 primitive 的敏感权重）；提问者**自己出题**的轮次没有条目，
    此时按自报的方面合成一个伪条目：权重取「该通道在条目库里的 primitive 亲和」，
    并标 `improvised` 说明这不是真条目的权重，而是对「这道题大概在考察什么」的估计。

    「这一轮用的是条目还是自造题」的判据与 `turn_channels` 一致（提问者自报的方面
    不在该条目的维度里 = 它没照条目问）：两处若不一致，会出现「答题按 A 条目估、
    证据记到 B 通道」的错配 —— 对照通道那一轮尤其致命，条目权重会让它随扰动塌下去，
    而它必须对单一机制不敏感（阴性预测）。
    """
    item = probe_items[turn_index - 1] if turn_index <= len(probe_items) else None
    if item is not None and (not focus or focus in (item.get("dims") or {})):
        return item
    if not focus:
        return item
    if focus == CONTROL_DIM:
        # 对照通道按构造**不对任何单一机制敏感**（原来那批零敏感度题目就是这个作用）：
        # 只有通用退化才该压低它。没有这一条，mock 会拿「全局最弱环节」去答这一轮，
        # 于是强扰动下对照通道一起塌下去 —— 阴性预测在离线上就废了。
        return {
            "key": f"focus:{focus}",
            "dims": {focus: 1.0},
            "primitives": {},
            "insensitive": True,
            # 这也是「自行出题」的一轮（题库里没有对照通道的题）
            "improvised": True,
            "probe_mode": "improvised",
            "source_id": "improvised",
        }
    from ..brain.examiner_bank import channel_signature

    signature = channel_signature(focus)
    if not signature:
        return None
    return {
        "key": f"focus:{focus}",
        "dims": {focus: 1.0},
        # 自造题没有真实条目权重，这里给的是「该通道大致会考察到哪些基础过程」的集合；
        # mock 侧对这种轮次取集合内最弱的一环（见 `MockProvider._probe_ability`），
        # 所以权重一律 1.0 —— 真实 provider 下这个探测条目根本不参与判断。
        "primitives": {pid: 1.0 for pid in signature},
        "improvised": True,
        "probe_mode": "improvised",
        "source_id": "improvised",
    }


def analyze_dialogue(
    analyst: LLMProvider,
    session: DialogueSession,
    *,
    goals: Sequence[tuple[str, str]] | None = None,
    scale_items: list[dict] | None = None,
    rubric_items: Sequence[dict] | None = None,
) -> dict:
    """调用分析者 LLM 对整段对话形成表型反馈。

    `rubric_items` 来自内置条目库：该月龄适用的**观察要点**（知情者报告 / 临床评定条目）。
    它不含 θ，因此给分析者提供锚点的同时不破坏盲法。

    `session.uncovered_channels`（整段对话没被探查到的方面）会写进分析者的输入：这不是 θ，
    也不是预期的维度归属，而是**证据覆盖情况** —— 让分析者知道哪些方面它手里没有证据，
    从而给保守值并写进 notes，而不是按对话读起来「挺正常」就给高分。
    `session.under_quota_channels`（问到过、但问题数不足配额的方面）同样披露，但分列一段：
    它们**有**证据、只是偏薄，与「完全没有证据」的处理不该混为一谈。

    **函数签名里没有 θ，也不该有。** 读数只能由这段对话读出来；任何「把 ground truth 传进来
    以便并排展示」的入参都会让扰动与表型产生一条已知条件的关联，那是本项目的构造效度所不允许的。
    """
    goal_pairs = tuple(goals) if goals else DEFAULT_GOALS
    messages = [
        _system(build_analyst_system_prompt(
            goal_pairs,
            rubric=render_rubric(list(rubric_items or ())),
        )),
        _user(build_analyst_prompt(
            session.transcript(),
            scale_items=scale_items,
            tasks=list(_items_by_keys(list(session.probe_keys))),
            uncovered=tuple(
                (ch, SCORED_CHANNELS.get(ch, ch)) for ch in session.uncovered_channels
            ),
            under_quota=tuple(
                (ch, SCORED_CHANNELS.get(ch, ch), int(session.question_counts.get(ch, 0)))
                for ch in session.under_quota_channels
            ),
            quota=QUESTIONS_PER_CHANNEL,
        )),
    ]
    analysis_error = ""
    try:
        raw = analyst.chat(
            messages,
            role="analyst",     # 额度下限按角色给：分析者的输出最长（见 provider）
            seed=session.seed,
            # 逐轮的维度归属（按轮次顺序）。这是给 mock 分析者的确定性上下文，让它能按维度
            # 分别汇总证据；真实模型忽略它，靠自己读对话判断。它不含 θ，故不破坏盲法。
            context={"probe_dims": _probe_dims_by_turn(session)},
        )
    except LLMError as exc:
        # 分析者调用失败也要留下**一条说明白原因**的反馈记录：直接抛出去的话，界面只会看到
        # 「没有读数」，看起来像这具心智根本没被观察过 —— 与「分析者没吐出东西」是两回事。
        raw, analysis_error = "", f"{type(exc).__name__}: {exc}"
    # 正则兜底能认的通道：观察目标 + 上传量表的条目 id。给了这份清单，解析器就不会把
    # 截断输出里的 `"confidence": 0.8` 之类当成一个维度读出来（那会造出一条假读数）。
    allowed_channels = {dim for dim, _label in goal_pairs}
    allowed_channels |= {
        str(item.get("id") or item.get("key") or "") for item in (scale_items or ())
    }
    # 一律交给解析器：空 raw 也要拿到它那句「分析者返回空内容」——
    # 否则「provider 返回空串」这条路径会连原因都丢掉，界面只剩一片空白。
    payload, error = parse_analyst_json(raw, allowed=allowed_channels)
    if analysis_error:
        error = analysis_error

    model_id = f"{analyst.name}:{analyst.model}"
    feedback = {
        "session_id": session.session_id,
        "theta_id": session.theta_id,
        "brain_id": session.brain_id,
        "model_id": model_id,
        "n_turns": session.n_turns,            # 计划的基础轮数
        "n_turns_run": session.n_turns_run,    # 实际轮数（含补问轮）
        "max_turns": session.max_turns,
        "n_brain_turns": session.n_brain_turns,
        "question_counts": dict(session.question_counts),
        "under_quota_channels": list(session.under_quota_channels),
        "dims": {},
        "dim_details": {},
        "confidence": None,
        "evidence": [],
        "narrative": "",
        # 探查结论：约 350 字的病例报告式总结 + 结构化表型印象。解析不出时留空键
        # （界面据此显示「未给出结论」），而不是编一段出来。
        "conclusion": "",
        "conclusion_chars": 0,
        "conclusion_notes": [],
        "asd_likeness": {"present": None, "level": "unknown", "basis": []},
        "notes": "",
        "error": error,
        "raw": raw,
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "probe_keys": session.probe_keys,
        "rubric_keys": [str(i.get("key")) for i in (rubric_items or ())],
    }
    if error:
        feedback["notes"] = (
            f"分析者没有给出可用的读数（{error}）。这次读数不可用，请检查提问者/分析者的输出"
            "额度（max_tokens）与模型类型（推理型模型会把额度花在思维链上），然后重新观察。"
        )
    if payload:
        feedback.update({
            "dims": payload.get("dims") or {},
            "dim_details": payload.get("dim_details") or {},
            "confidence": payload.get("confidence"),
            "evidence": payload.get("evidence") or [],
            "narrative": payload.get("narrative") or "",
            # 解析出了问题但救回了读数（截断/格式不合约定）：这条必须写在明面上 ——
            # 否则一份「只读到两个维度」的部分读数会看起来像完整读数。
            "notes": (
                f"⚠ 分析者输出格式有问题（{error}）—— 这次读数可能不完整。"
                + (f" {payload.get('notes')}" if payload.get("notes") else "")
                if error else (payload.get("notes") or "")
            ),
            "conclusion": payload.get("conclusion") or "",
            "conclusion_chars": payload.get("conclusion_chars") or 0,
            "conclusion_target_chars": payload.get("conclusion_target_chars"),
            "conclusion_notes": payload.get("conclusion_notes") or [],
            "asd_likeness": payload.get("asd_likeness")
            or {"present": None, "level": "unknown", "basis": []},
        })
        feedback["mean_reading"] = (
            round(sum(feedback["dims"].values()) / len(feedback["dims"]), 4)
            if feedback["dims"] else None
        )
    return feedback


def turn_channels(item: dict | None, focus: str) -> list[str]:
    """一轮对话归属到哪些通道 —— 归属与覆盖度**共用这一条规则**，不许两处各算一遍。

    规则（按提问者的实际行为分三种情形）：
    1. 自报的方面**不在**该轮计划条目的维度里 → 说明这一轮提问者没照条目问，按它自报的算；
    2. 否则有条目就按条目（条目带逐维度权重，比自报的单通道准得多）；
    3. 没有条目但有自报 → 按自报（自行出题的轮次靠它归属）；
    4. 都没有 → 空列表：该轮不进任何通道的证据，不硬塞给某个通道。
    """
    item_dims = sorted((item or {}).get("dims") or ())
    if focus and focus not in item_dims:
        return [focus]
    if item_dims:
        return item_dims
    return [focus] if focus else []


def _probe_dims_by_turn(session: DialogueSession) -> list[list[str]]:
    """逐轮的维度归属（按轮次顺序，供 mock 分析者区分选择性受损）。"""
    items = {
        str(item.get("key")): item
        for item in _items_by_keys(list(session.probe_keys))
    }
    out: list[list[str]] = []
    for turn in session.turns:
        if turn.get("role") != "examiner":
            continue
        out.append(turn_channels(items.get(str(turn.get("probe_key"))), str(turn.get("focus") or "")))
    return out


def _items_by_keys(keys: list[str]) -> list[dict]:
    from ..brain import examiner_bank as eb

    return eb.items_by_keys(keys) if keys else []


def _system(content: str) -> ChatMessage:
    return ChatMessage("system", content)


def _user(content: str) -> ChatMessage:
    return ChatMessage("user", content)


def _assistant(content: str) -> ChatMessage:
    return ChatMessage("assistant", content)
