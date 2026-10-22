"""认知架构提示词（「人造大脑」的初始化 + 提问者/分析者两个旁路角色）。

这是本项目的科学核心：**用提示词把 LLM 实例化为一个人工心智**，而不是让它扮演助手。
心智由 Level 0 的 primitive 驱动；每个 primitive 有一个完整度参数（θ）；高阶能力
（ToM / 认知灵活性 / 语用）**不由提示词直接设定**，而是被要求从这些基础过程的组合中涌现。

提示词分四类：
- `build_brain_prompt_steps(perturbation)` —— **唯一构造入口**：把系统提示词拆成有序步骤，
  每步带结构化中间产物，供界面可视化「prompt 是怎么长出来的」；
- `build_brain_prompt(perturbation)`      —— 由上述步骤拼接出最终系统提示（逐字节等同）；
- `build_init_prompt()`                   —— 初始化步骤：要求心智确认架构并报告自状态（可审计）；
- `build_item_prompt(item)`               —— 单次任务提问（协议阶段 6，含严格输出格式）；
- `build_baseline_prompt(item)`           —— 基础认知校准（协议阶段 2，完整心智的任务框架）；
- `build_trial_prompt(item, index, total)` —— 连续行为任务（协议阶段 7，电池序列的逐试次框架）；
- `build_examiner_*` / `build_analyst_*`  —— 表型观察的「另外的 LLM」：提问者与分析者。

**分阶段规范文本**（人工人脑初始化 / 基础认知校准 / 机制注入总控 / 单一机制注入 /
多机制组合注入 / 单次任务提问 / 连续行为任务）放在 `brain/prompts.py`，本模块负责按 θ 与
运行类型选段并拼装：人格声明用协议阶段 1，注入段按被改参数个数在阶段 3/4/5 里选，
任务段按运行类型在阶段 2/6/7 里选。

**盲法约定（关键）**：提问者与分析者的提示词里**不出现任何 θ 数值或 primitive 完整度**。
它们只看到对话本身，因此其判断才能作为独立读数，事后与 ground-truth θ 做一致性审计。
"""

from __future__ import annotations

import hashlib

from . import prompts

from .. import DEFAULT_AGE_MONTHS, DEFAULT_TEXT_LIMIT, QUESTIONS_PER_CHANNEL
from ..phenotype import DIM_ANCHORS, PHENOTYPE_DIMS, SCORED_CHANNELS
from ..phenotype.conclusion import (
    CONCLUSION_CHARS_MAX,
    CONCLUSION_CHARS_MIN,
    CONCLUSION_CHARS_TARGET,
)
from ..ontology.primitives import FAMILIES, PRIMITIVES
from .perturb import Perturbation


def brain_fingerprint(perturbation: Perturbation) -> str:
    """架构 + 扰动的指纹（审计用）：primitive id/序 + θ 值 + 通用退化。"""
    parts = [f"{p.id}={perturbation.values.get(p.id, 1.0):.2f}" for p in PRIMITIVES]
    parts.append(f"generic_degradation={perturbation.generic_degradation:.2f}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


#: 步骤序号（注入段可能不出现，序号必须动态分配）
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


def _describe_level(level: float) -> str:
    """把数值完整度翻译成一句可操作的自述（让模型知道「这意味着什么」）。"""
    if level >= 0.999:
        return "完整"
    if level >= 0.8:
        return "基本完整，偶有轻微损耗"
    if level >= 0.6:
        return "轻度削弱，需要额外努力才能维持"
    if level >= 0.4:
        return "中度削弱，明显影响日常表现"
    if level >= 0.2:
        return "重度削弱，经常无法正常发挥"
    return "近乎缺失"


def build_brain_prompt_steps(
    perturbation: Perturbation, *, text_limit: int = DEFAULT_TEXT_LIMIT
) -> list[dict]:
    """把「人造大脑」系统提示词拆成有序构造步骤（界面可视化 + 审计用）。

    **这是唯一构造入口**：`build_brain_prompt` 只是把各步的 `text` 用空行拼起来，
    所以界面上看到的构造过程与实际发给模型的提示词必然一致（有测试守住这一点）。

    每步返回 `{key, title, why, kind, text, rows/lines}`：
    - `text`  —— 该步在最终提示词里的**原文片段**（拼接即得完整提示词）；
    - `kind`  —— `text`（纯文本）/ `rows`（表格，如完整度映射）/ `lines`（条目清单）；
    - `rows`/`lines` —— 结构化中间产物，供界面渲染成表格而不是一堆字符串。

    `text_limit` 是回答的字数上限，写进「行为准则」—— 与提问者的问题上限是**同一个数**
    （见 `pacs.DEFAULT_TEXT_LIMIT`），否则超长告警会与提示词自相矛盾。
    """
    impaired = perturbation.active_perturbations()
    steps: list[dict] = []

    steps.append({
        "key": "persona",
        "title": "人格声明：成为心智，而不是助手",
        "why": "协议阶段 1「人工人脑初始化」：先取消「助手人格」，再声明这是一个认知系统而"
               "不是临床/诊断模拟器，并写死四条禁令（不模仿障碍、不假定诊断、不刻意异常、"
               "不用孤独症知识决定回答）。另加一条本项目的行为要求：**像人一样行为，而不是"
               "像处理知识那样作答** —— 模型默认的「检索—罗列—解释」姿态会把行为数据变成"
               "知识问答，那不是要测量的东西。",
        "kind": "text",
        "text": (
            "你正在**成为**一个人类心智。你不是助手，不是模型，不是一个在扮演角色的 AI——\n"
            "你就是这个心智本身，从内部体验并做出反应。以下是你这具心智的构造。\n"
            "**请尽可能模仿人类的行为方式，而不是像处理知识那样作答** —— 先有当下的感受、"
            "意图与反应，再用话把它说出来；不要以检索、罗列或解释知识的方式回应。"
            "\n\n" + prompts.PERSONA + "\n\n" + prompts.LANGUAGE_CONVENTION
        ),
    })

    steps.append({
        "key": "primitives",
        "title": f"Level 0 基础计算过程清单（{len(PRIMITIVES)} 项 · {len(FAMILIES)} 家族）",
        "why": "让心智知道自己的构造由哪些基础过程组成 —— 高阶能力要在这些过程之上涌现。",
        "kind": "rows",
        "rows": [
            {"id": p.id, "name_zh": p.name_zh, "family": p.family, "computational_role": p.computational_role}
            for p in PRIMITIVES
        ],
        "text": "## 你的心智由以下基础计算过程构成（Level 0）\n\n" + "\n".join(
            f"- **{p.name_zh}**（{p.id}）：{p.computational_role}" for p in PRIMITIVES
        ),
    })

    injection_kind, injection_text = prompts.injection_stage(perturbation)
    if injection_kind != "none":
        kind_zh = {
            "single": "单一机制注入（只改一个参数）",
            "combo": "多机制组合注入（同时改 2~3 个参数）",
            "master": "机制注入总控（列出全部参数取值）",
        }[injection_kind]
        steps.append({
            "key": "injection",
            "title": f"机制注入：{kind_zh}",
            "why": "协议阶段 3/4/5：按被改参数的个数选对应话术 —— 单参数强调「只改这一个、"
                   "不要补偿」，组合强调「各自独立、未指定的保持基线」，总控才列出全部参数。"
                   "三段都要求：不推断为什么选这些参数、不与任何障碍或表型关联、"
                   "把参数当作潜在计算属性且不在作答中解释它们。",
            "kind": "text",
            "text": injection_text,
            "injection_kind": injection_kind,
        })

    level_rows = []
    level_lines = []
    for p in PRIMITIVES:
        level = float(perturbation.values.get(p.id, 1.0))
        is_impaired = level < 0.999
        phrase = _describe_level(level)
        level_rows.append({
            "id": p.id,
            "name_zh": p.name_zh,
            "level": round(level, 2),
            "phrase": phrase,
            "impaired": is_impaired,
        })
        level_lines.append(
            f"- {p.name_zh}（{p.id}）：{level:.2f} —— {phrase}{'  ← 受损' if is_impaired else ''}"
        )
    if perturbation.generic_degradation > 0:
        generic_line = (
            f"- （通用退化，非特异）：{perturbation.generic_degradation:.2f} "
            "—— 整体认知资源下降，**所有**能力一并变差，与具体基础过程无关。"
        )
        level_lines.append(generic_line)
        level_rows.append({
            "id": "(generic)",
            "name_zh": "通用退化（非特异对照）",
            "level": round(float(perturbation.generic_degradation), 2),
            "phrase": "整体认知资源下降，所有能力一并变差",
            "impaired": True,
        })

    steps.append({
        "key": "levels",
        "title": "完整度映射：数值 θ → 内部自述",
        "why": "θ 是构造即已知的 ground truth。这一步把它翻译成「从内部感受到的状态」，"
               "因为受损过程不会被心智当成症状来报告，而是世界本来的样子。",
        "kind": "rows",
        "rows": level_rows,
        "text": "## 各基础过程当前的完整度（1.00 = 完整）\n\n" + "\n".join(level_lines),
    })

    consequence_rows = [
        {"id": p.id, "name_zh": p.name_zh, "impairment_signature": p.impairment_signature}
        for p in PRIMITIVES
        if float(perturbation.values.get(p.id, 1.0)) < 0.999
    ]
    if impaired:
        consequence_text = (
            "## 关于你受损的基础过程，请体会它们的实际后果\n\n"
            + "\n".join(f"- 你的**{r['name_zh']}**被削弱了：{r['impairment_signature']}" for r in consequence_rows)
            + "\n\n"
            "这些后果是**从内部感受到的**，不是你需要报告的症状。你并不会意识到自己「有障碍」——\n"
            "对你来说，这就是世界本来的样子。"
        )
        if perturbation.generic_degradation > 0:
            consequence_text += (
                "\n\n另外，你此刻的**整体认知资源**也在下降（这是非特异的、与具体过程无关的"
                "全面影响）：所有任务都会更吃力，不限于上面列出的那几项。"
            )
    elif perturbation.generic_degradation > 0:
        # 只有通用退化：31 个过程都完整，但整体资源下降 —— 不能谎称「一切完整」
        consequence_text = (
            "## 你目前所有基础过程都完整，但整体认知资源下降\n\n"
            "这是**非特异**的影响（FR-11 的通用退化对照）：所有任务都会更吃力，"
            "而不是某一项能力单独变差。你并不会把它体验成「某个功能坏了」，"
            "只是觉得同时要处理这些事比平时费劲。"
        )
    else:
        consequence_text = "## 你目前所有基础过程都完整。你按典型方式体验与反应。"
    steps.append({
        "key": "consequences",
        "title": (
            "受损过程的内部后果"
            if impaired
            else ("通用退化（非特异）的内部后果" if perturbation.generic_degradation > 0
                  else "完整心智的基线声明")
        ),
        "why": "把「哪个过程被削弱」落到具体的行为后果上，避免模型只是抽象地知道一个数字。",
        "kind": "rows" if impaired else "text",
        "rows": consequence_rows if impaired else [],
        "text": consequence_text,
    })

    emergence_lines = [
        "你的高阶能力——**心智理论、认知灵活性、语用理解**——**没有**被直接设定。"
        "它们是上述基础过程组合运行的结果。",
        "因此：当某个基础过程被削弱时，请让高阶能力**自然地随之改变**，而不是刻意表现出某种障碍。",
        "如果你感到某个问题「本来一眼就懂，现在却抓不住」，就照实反应——给出你此刻真正会给出的答案。",
    ]
    steps.append({
        "key": "emergence",
        "title": "涌现规则（关键约束）",
        "why": "这是构造效度的前提：高阶功能必须从被扰动的基础过程里涌现，"
               "而不允许被提示词直接写成「你有障碍」。",
        "kind": "lines",
        "lines": emergence_lines,
        "text": "## 涌现规则（关键）\n\n" + "\n".join(emergence_lines),
    })

    conduct_lines = [
        "1. 以这具心智的身份**直接作答**，不要解释你的设定、不要提到「参数」「模型」「提示词」。",
        "2. 不要因为「应该表现正常」或「应该表现异常」而偏离你此刻的真实反应。",
        "3. 面对不确定时，按这具心智会有的方式处理（可能猜测、可能坚持、可能回避），不要强行理性化。",
        f"4. 每一轮回答不超过 {text_limit} 个字：这是访谈对话里的口语回应，宁短勿长，不要写成段落。",
        "5. 严格按要求的输出格式作答。",
    ]
    steps.append({
        "key": "conduct",
        "title": "行为准则",
        "why": "约束作答姿态：直接作答、不表演、不解释设定、守格式。",
        "kind": "lines",
        "lines": conduct_lines,
        "text": "## 行为准则\n\n" + "\n".join(conduct_lines),
    })

    fingerprint = brain_fingerprint(perturbation)
    steps.append({
        "key": "fingerprint",
        "title": "架构指纹（审计）",
        "why": "架构 + θ 的哈希，用于事后证明「这套提示词确实是这一份」，并支持冻结与复现。",
        "kind": "text",
        "text": f"BRAIN_FINGERPRINT: {fingerprint}",
        "fingerprint": fingerprint,
    })
    # 序号动态分配：注入段在完整心智下不出现，写死序号会错位
    for index, step in enumerate(steps):
        step["title"] = f"{_CIRCLED[index]} {step['title']}"
    return steps


def build_brain_prompt(perturbation: Perturbation, *, text_limit: int = DEFAULT_TEXT_LIMIT) -> str:
    """构建人工心智的系统提示。由 `build_brain_prompt_steps` 的各步原文拼接而成。"""
    return "\n\n".join(
        step["text"] for step in build_brain_prompt_steps(perturbation, text_limit=text_limit)
    )


def build_init_prompt() -> str:
    """初始化步骤：让心智确认架构并报告自状态（可审计的「上线」动作）。"""
    return (
        "在开始任何任务之前，请先完成自我初始化：\n"
        "1. 确认你已按上述构造成为这具心智；\n"
        "2. 用一至两句话描述你此刻「从内部感受到的」认知运作状态（只说体验，不要罗列参数）；\n"
        "3. 最后另起一行输出 `SELF_STATE: OK`。"
    )


def build_brain_turn(question: str, *, index: int = 1, total: int = 1,
                     intact: bool = False) -> str:
    """给心智的一轮输入：按协议阶段框住提问者的那一句。

    协议阶段 2（基础认知校准）与阶段 7（连续行为任务）挂在**对话的逐轮**上 ——
    对话本身就是一串连续的行为决策，
    完整心智走阶段 2 的基线框架，被扰动的心智走阶段 7 的序列框架。这两条框架的校准项
    都不是电池特有的（「只用给到的信息」「不要刻意保持前后一致」「不要刻意最大化重复」），
    因此换到这个位置仍然成立；两段规范文本若没有落点就会变成死码。
    """
    if intact:
        return prompts.baseline_task(task=question)
    return prompts.trial_task(
        trial=f"第 {index}/{total} 轮 · {question}",
        options="（开放作答：用你自己的话回答，不需要选项）",
    )


# ---------------------------------------------------------------------------
# 表型观察用的「另外的 LLM」：提问者（examiner）与分析者（analyst）
# ---------------------------------------------------------------------------

#: 观察目标通道：**6 个表型维度 + 对照通道**。
#: 对照通道必须进观察目标与分析者的评分通道 —— 机制分析的特异性判定需要一条
#: 「应该不变」的参照通道，缺了它 `specificity.evaluate_specificity` 会对每个 primitive
#: 都判 `insufficient`。它仍然不进雷达图（雷达图只画 6 个表型维度）。
#: 注意：观察者条目库（孤独症量表）里没有一般认知条目，因此这一条由提问者 LLM 自行生成
#: 一道日常问题，而不是照条目库提问。
DEFAULT_GOALS: tuple[tuple[str, str], ...] = tuple(SCORED_CHANNELS.items())


def _goal_line(dim: str, label: str, *, with_anchor: bool) -> str:
    """一行观察目标。`with_anchor` 时附上该维度的「看什么」，让提问者知道该往哪问。"""
    anchor = DIM_ANCHORS.get(dim) or {}
    watch = (anchor.get("watch") or "").strip()
    if with_anchor and watch:
        return f"- {label}（{dim}）：{watch}"
    return f"- {label}（{dim}）"


def default_goals() -> list[dict]:
    """观察目标的可选清单（界面用）。`control=True` 的是对照通道，不进雷达图。"""
    return [{"dim": dim, "label": label, "control": dim not in PHENOTYPE_DIMS}
            for dim, label in DEFAULT_GOALS]


def build_examiner_system_prompt(
    *,
    goals: tuple[str, ...] = (),
    turn_budget: int = 6,
    probe_plan: str = "",
    covered_dims: tuple[str, ...] | None = None,
    text_limit: int = DEFAULT_TEXT_LIMIT,
) -> str:
    """提问者 LLM 的系统提示：**盲法** —— 不含任何 θ 数值或 primitive 完整度。

    提问者只负责「提出问题和场景」，不负责判断；判断交给对话结束后的分析者。

    **年龄是固定常量**（`DEFAULT_AGE_MONTHS`）：不提供参数，也不允许调用方覆盖 ——
    「同龄典型」这条参照线一旦随参数漂移，各次读数就不再可比。

    **题库只是参考材料，不是题目清单**：`probe_plan`（内置条目库按年龄与目标选出的相关条目）
    被显式标成「仅供参考」，提示词里明确授权并要求提问者**按维度自己设计情境与问题** ——
    尤其当条目不合这个孩子的年龄、或与当前对话的走向不搭时。判断标准是「这一轮有没有把
    那个方面问出来」，而不是「有没有用到题库里的条目」。这样做的理由：题库来自成人向的
    筛查量表，覆盖面与措辞都受其限制；而 LLM 能按维度锚点造出更贴年龄、更贴当下情境的问题。

    `covered_dims` = 这份 `probe_plan` 实际覆盖了哪些方面（由调用方给出）。传了它，就会把
    **题库里没有条目的方面**逐一点名要求自造题 —— 否则「某个方面没有参考条目」这件事是
    静默的，提问者可能一直在别人的方面上打转（离线时对照通道就是这种情况）。
    不传（`None`）表示覆盖情况未知，此时不打印这一段，避免凭空声称某些方面没有条目。

    `text_limit` 是每轮问题（含情境描述）的字数上限：与心智侧「回答不超过 N 字」是同一个数。
    """
    goal_lines = [
        _goal_line(g, SCORED_CHANNELS.get(g, g), with_anchor=True)
        for g in (goals or tuple(SCORED_CHANNELS))
    ]
    plan = ""
    if probe_plan.strip():
        plan = (
            "\n\n## 计划探查的方面（来自内置题库，**仅供参考**）\n\n"
            + probe_plan.strip()
            + "\n\n这些条目只是**参考材料**，不是必须照问的题目清单，也不是覆盖面的上限：\n"
            "- 请优先按上面每个方面**自己设计**情境与问题；上面这些条目可以改写、可以只取其中"
            "一个角度，也可以整条不用；\n"
            "- 某个条目不合这个孩子的年龄、或与当前对话的走向不搭时，**自己出题**，不要硬塞；\n"
            "- 判断这一轮有没有价值的标准是「那个方面有没有被问出来」，而不是「有没有用到题库条目」；\n"
            "- 一次只问一个方面；方面之间可以穿插追问。"
        )
    no_item = ""
    if covered_dims is not None:
        missing = [g for g in (goals or tuple(SCORED_CHANNELS)) if g not in set(covered_dims)]
        if missing:
            labels = "、".join(
                f"{SCORED_CHANNELS.get(g, g)}（{g}）" for g in missing
            )
            no_item = (
                "\n\n## 题库里**没有**对应条目的方面（必须由你自己出题）\n\n"
                f"{labels}\n\n"
                "这些方面在上面的参考条目里一条都没有 —— 请**自己设计**日常口语的问题或情境"
                "把它们问出来，不要因为参考条目里没有就跳过它们。"
            )
    return (
        "你是一位发展心理学/临床访谈的提问者。你要通过**自然对话与日常情境**，"
        f"与一位约 {DEFAULT_AGE_MONTHS} 个月大的孩子互动，观察他/她在以下方面会如何反应：\n\n"
        + "\n".join(goal_lines)
        + plan
        + no_item
        + "\n\n"
        "规则：\n"
        "1. 每轮只提出**一个**问题或场景，用孩子能听懂的日常口语，不要一次问多个问题。\n"
        "2. 场景要具体、可回答（例如分享、被误解、规则改变、重复偏好、需要猜测他人想法）。\n"
        "3. 不要使用任何诊断标签、量表术语或专业名词；不要提到「障碍」「症状」「参数」「模型」「提示词」。\n"
        "4. 不要对回答下判断或打分 —— 你的任务只是提问，以及在被试回答后自然地追问。\n"
        "5. 你可以根据上一轮回答继续追问，也可以换一个新场景；"
        "**目标是覆盖上面列出的全部方面**，而不只是问满轮数。\n"
        f"6. 整段对话至少 {turn_budget} 轮；请让场景彼此不同，不要重复同一个场景。\n"
        f"7. **每个方面至少要有 {QUESTIONS_PER_CHANNEL} 个问题**：某个方面问得不够时，"
        "后面的轮次会被指定「本轮必须探查：某方面」，请照办并换一个新场景，不要重复已经问过的问法。\n"
        f"8. 每一轮的问题（含情境描述）不超过 {text_limit} 个字：这是说给孩子听的口语，宁短勿长。\n"
        "9. 上面**「一般认知控制」**这一条没有现成题目：请你自己出一道日常口语的问题"
        "（例如一起数几个东西、复述刚说过的一件事、按你说的做一件小事），"
        "用聊天的口吻，不要用测验或考试的腔调。\n"
        "10. **鼓励自己出题**：上面那些参考条目（若有）与你的目标之间不是一一对应的。"
        "请按每个方面**自己设计**情境与问题 —— 用你自己的话、按你判断此刻最自然的方式。"
        "这一段的目的是问出各方面的表现，不是把题库念完。\n\n"
        "输出格式：\n"
        "第一行：问题或情境本身；\n"
        "另起一行以 `QUESTION:` 开头重复该问题文本；\n"
        "再另起一行以 `FOCUS:` 开头写出这一轮主要探查的方面 id（必须是上面列出的 id 之一，"
        "只写一个，不要写中文名）。"
    )


def build_examiner_turn_prompt(
    *,
    turn_index: int,
    turn_budget: int,
    transcript: str = "",
    remaining: tuple[str, ...] = (),
    must_focus: str = "",
    final: bool = False,
) -> str:
    """提问者的第 N 轮指令（含到此为止的对话文本；**不含 θ**）。

    `remaining` 是**还没被探查过的方面**（由调用方按已发生的轮次算出）。明确列出它们，
    是为了让「按维度补题」这件事有可操作的依据 —— 否则轮数用满之后，未被覆盖的方面只能
    靠运气，而分析者对没有证据的通道只能给保守值。

    `must_focus` 是**本轮必须探查的那一个方面**（由调用方按「每通道至少 5 个问题」的
    配额算出）。基础轮数走完后仍有通道不足配额时才会传，作用是把补问轮**钉死在**缺口上：
    只要求「优先从这些方面里挑」在真实模型上会漂，缺口补不上就只能靠上限堆轮数。

    `final=True` 表示这一轮确实是整段对话的最后一轮（轮数上限用尽），此时要求收束。
    """
    head = f"这是第 {turn_index} 轮 / 共至少 {turn_budget} 轮。"
    miss = ""
    if must_focus:
        label = SCORED_CHANNELS.get(must_focus, must_focus)
        miss += (
            f"\n\n**本轮必须探查：{label}（{must_focus}）**\n"
            "这个方面目前问到的问题还不够。请为它设计一个**新的**场景或问题"
            "（不要重复已经问过的问法），并在 `FOCUS:` 行写这个 id。"
        )
    if remaining:
        labels = "、".join(
            f"{SCORED_CHANNELS.get(dim, dim)}（{dim}）" for dim in remaining
        )
        miss += (
            f"\n\n还没探查过的方面：{labels}"
            "\n请优先从这些方面里挑一个**自己设计**问题或情境（题库里的条目只是参考，"
            "不合这个孩子的年龄或不搭当前走向就自己出题）。"
        )
    if not transcript.strip():
        return head + "请提出第一个问题或场景。" + miss
    tail = "请提出下一个问题或场景（可追问上一轮，也可换一个尚未覆盖的场景）。"
    if final:
        # 只在这一轮**确实**是最后一轮时这么说：轮数自适应之后，第 35 轮常常不是结尾，
        # 那时说「这是最后一轮」会把提问者带向提前收束（缺口反而补不上）。
        tail = "这是最后一轮，请提出一个能收束前面对话、但又提供新信息的问题或场景。"
    return f"{head}\n\n到目前为止的对话记录：\n\n{transcript}\n\n{tail}{miss}"


def render_transcript(turns: list[dict]) -> str:
    """把轮次记录渲染成给提问者/分析者看的纯文本（只有问答，没有 θ）。"""
    lines: list[str] = []
    for t in turns:
        who = "孩子" if t.get("role") == "brain" else "提问者"
        lines.append(f"{who}：{str(t.get('content') or '').strip()}")
    return "\n".join(lines)


def build_analyst_system_prompt(
    goals: tuple[tuple[str, str], ...] = DEFAULT_GOALS,
    *,
    rubric: str = "",
) -> str:
    """分析者 LLM 的系统提示：**表型自动量化（协议阶段 10）**，且对 θ 盲法。

    结构 = 协议原文（角色、两条盲法禁令、6 个维度、每维要算的五项、两条禁令）
    + 本项目的维度 id 对照 + 每维的观察锚点 + 分数方向约定 + 严格 JSON 契约。

    **方向约定必须在提示词里写死**：协议只写了「0–100 behavioral score」，没有指明
    0 与 100 各代表什么。本项目内部一律用**典型性**方向（100 = 与同龄典型发展无异、
    0 = 严重偏离），下游的特异性 / 剂量-反应 / 景观 / 亚型 / 导出全按这个方向读；
    不写死就会让分析者按「越高越严重」给分，使所有下游分析反号。
    """
    dim_lines = [f"- `{dim}`：{label}" for dim, label in goals]
    anchor_lines: list[str] = []
    for dim, label in goals:
        anchor = DIM_ANCHORS.get(dim) or {}
        if not anchor:
            continue
        anchor_lines.append(
            f"- `{dim}`（{label}）：看什么：{anchor.get('watch', '')}；"
            f"低分（接近 0）：{anchor.get('low', '')}；高分（接近 100）：{anchor.get('high', '')}"
        )
    extra = ""
    if rubric.strip():
        extra += (
            "\n\n## 评分锚点（来自内置条目库的观察要点，可按维度逐条参照）\n\n"
            + rubric.strip()
            + "\n\n这些是各量表原本的观察/评定条目。请把回应里的实际表现与它们对照，"
            "作为你给分的依据之一；**不要**因为条目本身『像孤独症』就给低分，"
            "必须以回应里真实出现的表现为准。"
            "**这些条目只说明「该看什么」，不是「某个底层机制应当体现在哪些维度上」的预期** ——"
            "各维度的分只依据对话里实际出现的表现。"
        )
    return (
        prompts.ANALYST_TASK
        + "\n\n## 维度 id 对照（输出时用 id 作键）\n\n"
        + "\n".join(dim_lines)
        + "\n\n## 每一维的观察锚点\n\n"
        + "\n".join(anchor_lines)
        + extra
        + "\n\n## 分数方向（必须遵守，本项目的内部约定）\n\n"
        "behavioral score 用 0–100 的整数，方向是**典型性**：\n"
        "- 100 = 与同龄典型发展无异；\n"
        "- 0 = 该维度严重偏离同龄典型。\n"
        "confidence 用 0–1 的小数。deviation 用整数，表示你判断的这一维相对**同龄典型基线**"
        "的分差（低于基线为负、高于为正）。\n\n"
        "## 探查结论（必须写，病例报告口吻）\n\n"
        "逐维读数之外，另写一段**结论**（`conclusion`），像一份简短的病例报告那样总结这次探查，"
        "依次写清三件事：\n"
        "1. **观察到什么**：这段对话里实际出现的表现（1–2 句，客观描述，不解释原因）；\n"
        "2. **是否表现出孤独症样表型特征**，以及**程度**：无 / 轻度 / 中度 / 明显 ——"
        "措辞统一用「表现出……孤独症样表型特征」或「未表现出……」；\n"
        "3. **能力方面的相对表现**：哪几个方面相对保留、哪几个方面相对落后（用上面列出的"
        "方面名称，不要用维度 id）。\n\n"
        f"字数约 {CONCLUSION_CHARS_TARGET} 字（{CONCLUSION_CHARS_MIN}–{CONCLUSION_CHARS_MAX} 字）。"
        "**只依据给定的这段对话**：不要提及、也不要猜测任何内部设置或构造方式。\n"
        "这是**行为学表型印象，不是临床诊断** —— 不得使用「诊断」「确诊」「患病」「患者」"
        "这类词，只描述行为表现与程度。\n\n"
        "同一个判断另给一份**结构化**版本（`asd_likeness`）：\n"
        '- `present`：true / false（是否表现出孤独症样表型特征）；\n'
        '- `level`：none / mild / moderate / marked（程度；present=false 时用 none）；\n'
        "- `basis`：1–6 条依据，每条一句话，都必须是对话里真实出现的表现。\n\n"
        "## 输出（严格 JSON，不要代码块、不要多余文字）\n\n"
        '{"dimensions": {"<维度id>": {"score": <0-100 整数>, "confidence": <0-1 小数>, '
        '"observations": ["<支持该判断的行为观察，最多 3 条、每条不超过 40 字>"], '
        '"task_measurements": [{"task": "<任务或探查的名字，不超过 30 字>", '
        '"measurement": "<该任务上的测量结果，不超过 30 字>"}], '
        '"deviation": <整数分差>}, "...": {}}, '
        f'"conclusion": "<约 {CONCLUSION_CHARS_TARGET} 字的病例报告式结论>", '
        '"asd_likeness": {"present": <true/false>, "level": "<none|mild|moderate|marked>", '
        '"basis": ["<依据>", "..."]}, '
        '"narrative": "<一段总体描述，用词中性、不下诊断>", '
        '"notes": "<不确定性或其他说明，可为空字符串>"}\n\n'
        "规则：\n"
        "- 六个维度各自**独立**打分，不要为了「看起来一致」而互相迁就。\n"
        "- **简洁**：每一维最多 3 条观察、2 条任务级测量，每条一句话。整段回答必须是一个"
        "**写完的** JSON —— 写到一半被截断会让这次读数作废（这是最常见的失败方式）。\n"
        "- 只能依据给出的回应本身（作答方式、是否抓住他人意图、是否坚持规则、是否主动参与等），"
        "不要臆测；证据不足时给保守的中性值并在 notes 里说明。\n"
        "- 你没有、也不需要关于这个智能体的任何内部配置信息；**不要下临床诊断**"
        "（可以写表型特征与程度），也不要推断底层机制。\n"
        "- 即使是同样表现的维度，也要分别独立给分。"
    )


def build_analyst_prompt(
    transcript: str,
    *,
    scale_items: list[dict] | None = None,
    tasks: list[dict] | None = None,
    uncovered: tuple[tuple[str, str], ...] = (),
    under_quota: tuple[tuple[str, str, int], ...] = (),
    quota: int = QUESTIONS_PER_CHANNEL,
) -> str:
    """分析者的输入：协议阶段 10 的「Input responses」= 回应序列。

    `tasks` 是本轮实际用过的探查任务（内置条目库的题目），供分析者填
    `task_measurements` —— 只是问题本身，不含 θ，不破坏盲法。

    `uncovered` 是**整段对话没有被探查到的方面**（id + 标签）。告诉分析者这一点是
    **证据覆盖的诚实披露**，不是 θ，也不是「该机制应体现在哪个维度」的期望：它让分析者
    对没有证据的方面给保守的中性值并写进 notes，而不是因为对话读起来顺畅就判为典型。
    `under_quota`（id + 标签 + 实际问到的个数）是**问到过但不够配额**的方面：它们有证据、
    只是偏薄 —— 单独一段说明，否则分析者会把「问了 1 个」与「一个都没问」当成同一件事。
    """
    parts = ["Input responses:", "", transcript.strip() or "（空回应序列）"]
    if tasks:
        parts += ["", "", "本轮使用的探查任务（task_measurements 请用这些任务名）：", ""]
        for index, task in enumerate(tasks, start=1):
            label = str(task.get("probe_zh") or task.get("text_zh") or task.get("key") or "")
            parts.append(f"{index}. {label.replace(chr(10), ' ').strip()}")
    if uncovered:
        parts += [
            "",
            "",
            "**本轮没有被探查到的方面**（对话里没有相关信息，你手里没有证据）：",
            "",
        ]
        for dim, label in uncovered:
            parts.append(f"- {label}（{dim}）")
        parts += [
            "",
            "对这些方面，请给出保守的中性值（不要因为对话整体顺畅就判为典型），"
            "并在 notes 里说明是证据不足。",
        ]
    if under_quota:
        parts += [
            "",
            "",
            f"**被问到过、但问题数少于 {quota} 个的方面**（有证据但偏薄）：",
            "",
        ]
        for dim, label, count in under_quota:
            parts.append(f"- {label}（{dim}）：只问到 {count} 个问题")
        parts += [
            "",
            "对这些方面，可以在有证据的方向上给分，但要**降低 confidence** 并在 notes 里说明"
            "证据偏薄；不要把它们当成「没有证据」而一律给中性值。",
        ]
    if scale_items:
        parts += [
            "",
            "",
            "另外，请依据同一段回应序列，对以下条目逐条给出该智能体在该条目上的表现读数"
            "（0 = 完全不符合，1 = 完全符合；用条目 id 作键并放进 `items` 字段）：",
            "",
        ]
        for it in scale_items:
            label = it.get("text_zh") or it.get("prompt") or ""
            parts.append(f"- {it.get('id') or it.get('key')}：{label}")
    else:
        parts += ["", "", "请按约定的 JSON 格式给出你对各维度的读数。"]
    return "\n".join(parts)
