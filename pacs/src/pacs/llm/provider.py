"""LLM 接入层（FR-10）：BYOK + 离线确定性 Mock。

两种 provider：
- `OpenAICompatProvider` —— 走 OpenAI 兼容 `/chat/completions`（GPT / DeepSeek / Qwen / 本地
  推理服务均可），**仅用标准库 urllib**，不需要 httpx；
- `MockProvider` —— 确定性**规则式前向模型**：由 θ 与探查条目的敏感权重算出一个可复现的回答。
  它不产生任何 API 费用、不需要网络，用于离线开发、CI 与格式校验（与 `autism_twin` 用规则
  世界模型做冷启动同理）。注意：mock 路径下的表型结构是**构造出来的**，不是 LLM 涌现的；
  真实结论必须用真实 provider 跑。

同一个 provider 承担四种角色（由 `role` 参数区分）：
- `""`（默认）—— 心智初始化；
- `"brain"`  —— 表型观察的对话中，模拟大脑对提问者作答；
- `"examiner"` —— 提问者的下一个问题或场景；
- `"analyst"`  —— 分析者对整段对话的表型打分（严格 JSON）。

`""` 角色用于被观察的心智自由作答。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Protocol

from ..phenotype import SCORED_CHANNELS
from ..config import LLMSettings

#: 对话角色的合法取值（供调用方自检，避免把角色打错字而静默退化成初始化路径）
ROLES = ("", "brain", "examiner", "analyst")

#: 分析者的输出额度下限：它要吐一整段 JSON（7 个通道的逐维记录 + 约 350 字结论 +
#: 判定依据），比一次问答的额度大得多。512 这种默认值会被截断成半截 JSON，
#: 解析失败后整个读数就空了 —— 这个数不是调优，是「够不够写完一个答案」。
#: **它是下限而不是定值**：`① 模型设置` 里的 `max_tokens` 更大时以配置为准，
#: 否则界面上那句「把 max_tokens 调大」对分析者根本不生效（分析者的输出最长，
#: 偏偏也只有它对配置免疫）。
#: 8000 这个量级是实测出来的：一次真实会话里分析者的可见输出写到 967 字就被截断，
#: 而额度只给了 3000 —— 推理型模型会把大头的额度花在思维链上，留给正文的远少于额度。
#: 端点若不认这么大的额度（HTTP 400 提到 max_tokens），会自动退回配置值重试（见 `chat`）。
ANALYST_MIN_TOKENS = 8000



class LLMError(RuntimeError):
    """LLM 调用失败（网络 / 鉴权 / 限流 / 响应格式）。"""


class _EmptyContentError(LLMError):
    """provider 返回了**空回答**（不是「被试沉默」，是这一次调用没产出内容）。"""


@dataclass
class ChatMessage:
    role: str          # system | user | assistant
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


def _uniform(*parts: object) -> float:
    """确定性伪随机数 ∈ [0,1)：同输入同输出（跨进程、跨平台稳定）。"""
    seed = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16)


#: 表型印象档位（`asd_likeness.level`）：由平均读数粗分，与界面上的中文对照见
#: `phenotype.conclusion.ASD_LIKENESS_LEVEL_ZH`。
def _mock_level(mean: float) -> str:
    if mean < 0.35:
        return "marked"
    if mean < 0.55:
        return "moderate"
    if mean < 0.75:
        return "mild"
    return "none"


def _mock_conclusion(
    scored_dims: dict[str, float],
    dims: list[str],
    scored: list[tuple[str, str, float]],
    unprobed: list[str],
) -> dict:
    """mock 的探查结论（病例报告口吻，约 350 字）+ 结构化表型印象。

    它是**规则式替身**：按平均读数分档、按维度读数的高低排「相对保留 / 相对落后」。
    结论里明写这一点 —— 否则一段读起来像临床判读的文字会被人当成 LLM 的判断。
    """
    from ..phenotype.conclusion import ASD_LIKENESS_LEVEL_ZH

    mean = sum(scored_dims.values()) / len(scored_dims) if scored_dims else 0.0
    level = _mock_level(mean)
    present = level != "none"
    weak = sum(1 for _q, reply, _s in scored if _reply_quality(reply) in ("poor", "minimal"))
    ranked = sorted(scored_dims.items(), key=lambda kv: kv[1])
    label = lambda dim: SCORED_CHANNELS.get(dim, dim)  # noqa: E731
    kept = "、".join(label(d) for d, _v in ranked[-2:][::-1]) or "（无读数）"
    lagged = "、".join(label(d) for d, _v in ranked[:2]) or "（无读数）"

    text = (
        f"本次共完成 {len(scored)} 轮问答，孩子在 {len(scored) - weak} 轮里给出了具体回应，"
        f"在 {weak} 轮里回应简短、回避或答非所问。"
        f"综合各方面看，这段对话{'表现出' if present else '未表现出'}孤独症样表型特征，"
        f"程度为{ASD_LIKENESS_LEVEL_ZH[level]}。"
        f"能力方面，{kept}相对保留；{lagged}相对落后，其余方面表现居中。"
        f"回应在场景切换后的适应性、对他人意图的追踪、以及是否愿意展开细节上差异较明显，"
        f"这与各维度读数并不齐平是一致的。"
        + (
            f"另有{'、'.join(label(d) for d in unprobed)}在本次对话中没有被探查到，"
            "无法作判断。"
            if unprobed else ""
        )
        + "需要说明：以上是**基于对话的行为学表型印象**，不是临床诊断；"
        "它只描述这段问答里出现过的表现，不涉及任何内部构造或成因。"
        "本段由离线 mock 规则式产出（按回应长度与回避词频推定），不是 LLM 的判断，"
        "真实结论必须用真实 provider 重跑。"
    )
    basis = [
        f"{label(dim)}读数 {value:.2f}（1 = 与同龄典型发展无异）"
        for dim, value in ranked[:3]
    ]
    return {
        "conclusion": text,
        "asd_likeness": {
            "present": present,
            "level": level,
            "basis": basis or ["本次没有产生可用的维度读数"],
        },
    }


class LLMProvider(Protocol):
    """provider 统一接口。

    `item` / `perturbation` / `seed` / `context` 只被 Mock 使用：真实 provider 把这些信息
    要么编码在提示词里、要么根本不需要，因此一律忽略。
    """

    name: str
    model: str

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        item: object = None,
        perturbation: object = None,
        seed: int = 0,
        role: str = "",
        context: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Mock：确定性规则式前向模型
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mock 的对话探针素材（确定性、规则式）
# ---------------------------------------------------------------------------

#: 提问者的固定问题池（覆盖各表型维度的常见场景）；mock 下按轮次取用。
_DIALOGUE_QUESTIONS: tuple[str, ...] = (
    "你在幼儿园搭积木，旁边的小朋友突然把你搭好的房子推倒了，你会怎么办？",
    "老师今天把座位换了，你必须坐到另一个小朋友旁边，你会怎么想？",
    "如果你看到好朋友在哭，你会做什么？",
    "妈妈让你把最喜欢的玩具分给弟弟玩，你愿意吗？为什么？",
    "老师说完「先洗手再吃点心」，但你的好朋友直接去拿点心了，你会怎么做？",
    "有个小朋友说「我知道你想玩这个」，可他说的跟你想的不一样，你会怎么办？",
    "今天玩一个你从来没玩过的新游戏，你会想去试试吗？",
    "如果你找不到自己的水杯了，你会怎么做？",
    "画画的时候，老师说要涂成红色，可你想涂成蓝色，你会怎么办？",
    "小朋友抢了你的椅子，你告诉他「这是我的」，他还是不还，你会怎么做？",
    "你正在看的图画书被翻到了下一页，你会怎么办？",
    "妈妈今天走另一条路去幼儿园，你会说什么？",
    "如果有小朋友问你「今天早上吃了什么」，你会怎么告诉他？",
    "你在玩小汽车，别的小朋友说要一起玩，但只肯玩两分钟，你会怎么做？",
    "老师让你把积木按颜色排成一排，你会怎么排？",
    "你搭好的桥倒了，你会先做什么？",
    "有一个声音很响的吸尘器在响，你会怎么做？",
    "今天点心是你没吃过的，你会先做什么？",
    "爸爸说他今天很累，你会怎么做？",
    "你讲一件事的时候，别人没听懂，你会怎么办？",
    "刚刚老师说了哪三件事？你记得几件？",
    "你想玩秋千，但已经有小朋友在玩了，你会怎么做？",
    "如果游戏规则突然改成「谁都不能说话」，你会怎么办？",
)

#: 回避 / 未回应类标记（用于 mock 分析者粗判回应质量）。
#: 注意不要收录「不说」这类会误伤的子串 —— 「我先不说话」是一种策略，不是回避。
_WEAK_MARKERS: tuple[str, ...] = ("不知道", "不想说", "不清楚", "没什么", "就那样", "沉默")

#: 四档回应质量 → 表型读数（1.0 = 典型）
#: 提问者**自己出题**时用的问句（按通道）。题库覆盖不到的通道只能靠这里 —— 离线时
#: 只有对照通道属于这种，所以它必须有一道像样的日常问题，而不是随便抓一句社交问句。
_IMPROVISED_QUESTIONS: dict[str, str] = {
    "general_cognition": "我们一起数数看：桌上有三个杯子，我再放两个，一共几个呀？",
}

_QUALITY_SCORE = {"rich": 0.88, "mid": 0.62, "poor": 0.35, "minimal": 0.12}

#: 每档的确定性回应模板（长度与措辞与 `_reply_quality` 的分档规则保持一致）
_DIALOGUE_REPLIES: dict[str, tuple[str, ...]] = {
    "rich": (
        "我会先看看他在做什么，然后告诉他我也想玩，我们可以轮流来。",
        "我会等一会儿再问一次；如果他还是不愿意，我就换一个我们都喜欢的玩法。",
        "他大概是想要那个玩具吧，我会说「先给你玩，等一下轮到我」。",
        "我先不说话，看看他还想不想继续，然后再慢慢靠近一点。",
    ),
    "mid": (
        "嗯……我会给他玩一下。",
        "我会先去问一下老师。",
        "那我就照着做吧。",
        "我有点不明白，等一下再说。",
    ),
    "poor": (
        "我不知道该怎么说，就是不太想回答。",
        "这个我不清楚，他想要什么我也说不上来。",
    ),
    "minimal": (
        "不知道。",
        "……",
        "不想说。",
        "沉默，没有回答。",
    ),
}


def _reply_quality(text: str) -> str:
    """把一句儿童回应粗分成四档（规则式、确定性）。

    这正是 mock 路径「构造性」的所在：真实 LLM 下这些档位应由模型自身表现涌现，
    这里只是为了让对话通道在离线环境下可被端到端验证。
    """
    t = text.strip()
    if len(t) <= 3:
        return "minimal"
    if any(m in t for m in _WEAK_MARKERS):
        return "minimal" if len(t) < 12 else "poor"
    if len(t) >= 20:
        return "rich"
    if len(t) >= 8:
        return "mid"
    return "poor"


class MockProvider:
    """离线确定性 provider。

    对被扰动的心智：由 `ability = Σw·θ / Σw × (1 − generic_degradation)` 映射到回应质量，
    再用确定性均匀数决定这一次的措辞档位。
    对初始化等非对话调用：返回确定性的骨架文本。
    """

    name = "mock"

    def __init__(self, model: str = "mock-cognitive-mind") -> None:
        self.model = model

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        item: object = None,
        perturbation: object = None,
        seed: int = 0,
        role: str = "",
        context: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # `max_tokens` 只对真实 provider 有意义（mock 的输出是规则式的，本来就短）
        if role == "examiner":
            return self._ask_question(messages, seed, context)
        if role == "analyst":
            return self._analyze(messages, context)
        if role == "brain":
            return self._reply_dialogue(messages, perturbation, seed, item)
        if perturbation is not None:
            return self._reply_dialogue(messages, perturbation, seed, None)
        # 大脑初始化 / 其他调用：给出确定性骨架回应
        return self._acknowledge(messages)

    # ---------- 对话探针：模拟大脑作答 ----------

    @staticmethod
    def _probe_ability(probe, perturbation) -> float:
        """这一轮探查所敏感的基础过程，在 θ 下的加权平均完整度。

        为什么要按条目加权而不是取全局最弱值：心智不是「整体调暗」，而是**选择性**受损。
        `social_reward_weighting` 掉到 0.2 的心智，回答「积木被推倒了怎么办」会明显贫乏，
        回答「每天走同一条路会不会不舒服」却可以很正常。取全局最弱值会把这种选择性抹平，
        让所有维度的读数一起塌下去 —— 那样下游的机制异质性分析就没有任何信号可用。
        """
        values = getattr(perturbation, "values", None) or {}
        if (probe or {}).get("insensitive"):
            # 明确标了「不敏感」（对照通道）：只有通用退化能压低它，单一机制不影响
            return 1.0
        prims = (probe or {}).get("primitives") or {}
        if not prims:
            # 没有条目信息（固定问题清单）：退化为最弱环节
            return min(values.values()) if values else 1.0
        weights = {p: float(w) for p, w in prims.items() if p in values}
        if not weights:
            return 1.0
        if (probe or {}).get("improvised"):
            # 提问者当场自造的题：mock 无从知道这道题实际会加载哪些基础过程，只知道它冲哪个
            # 通道去的。取该通道画像里**最弱的一环** —— 「只要这个通道依赖的某个过程受损，
            # 问到这个通道时回应就会显出贫乏」。若照条目那样取加权平均，单一 primitive 的损伤
            # 会被同通道其它完好的 primitive 摊平（实测 attention_shifting=0.2 的读数与完整
            # 心智完全一致），离线管线就再也看不到选择性受损。这是 mock 的近似，不是真实机制。
            return min(float(values[p]) for p in weights)
        total = sum(weights.values())
        if total <= 0:
            return 1.0
        return sum(float(values[p]) * w for p, w in weights.items()) / total

    def _reply_dialogue(self, messages, perturbation, seed: int, probe=None) -> str:
        """对话中模拟大脑的回应：由「这一轮问的是什么」对应的 θ 决定回应丰富程度。

        规则式前向模型：能力 = 该条目所敏感基础过程的加权完整度 × (1 − 通用退化)，
        再映射到四档模板。真实 LLM 下这一步应当由提示词里描述的架构自然涌现。
        """
        ability = self._probe_ability(probe, perturbation)
        generic = float(getattr(perturbation, "generic_degradation", 0.0))
        ability = max(0.0, min(1.0, ability * (1.0 - generic)))
        tier = (
            "rich" if ability >= 0.8
            else "mid" if ability >= 0.5
            else "poor" if ability >= 0.3
            else "minimal"
        )
        pool = _DIALOGUE_REPLIES[tier]
        key = (probe or {}).get("key") or ""
        idx = int(_uniform(seed, getattr(perturbation, "theta_id", "theta"), key, "dialogue")
                  * len(pool))
        return pool[min(idx, len(pool) - 1)]

    # ---------- 提问者 ----------

    _PLAN_LINE = re.compile(r"^\s*\d+\.\s*\[[^\]]*\]\s*(.+)$")
    #: 计划行里的条目前缀（`3. [维度名…] `），回查条目时要先剥掉
    _PLAN_PREFIX = re.compile(r"^\s*\d+\.\s*\[[^\]]*\]\s*")

    @staticmethod
    def _norm_probe(text: str) -> str:
        """把条目文本与计划行归一化到可比形态：去掉所有空白与「 ｜ 」分隔符。

        计划行把条目里的换行渲染成了「 ｜ 」，逐字包含匹配会漏掉带换行的长条目
        （实测 7 条里 3 条匹配不上，于是 `_focus_line` 退回 `goals[0]`，
        而那 3 轮的证据会被 `turn_channels` 整体挪到错误的通道上）。
        """
        return re.sub(r"[\s｜|]+", "", text or "")

    @classmethod
    def _bank_item(cls, text: str) -> dict | None:
        """按条目文本回查条目库 —— **只在直接调用 `chat` 时的兜底路径**。

        真实调用方（`dialogue._ask`）通过 `context["probe_item"]` 直接给出这一轮用的是哪一条，
        mock 因此不必猜。为什么不能只靠文本回查：题库里**有多条条目的探查语句逐字相同**
        （同一情境模板挂在不同量表上，例如 `tools10:1` 与 `tools0:55`），文本匹配只能给出一对
        孪生条目中的一个，认错就把 `FOCUS` 报成别的通道 —— 而 `FOCUS` 一错，
        `turn_channels` 会把该轮证据整体挪到那个通道。

        这里仍做归一化后的**最长**匹配：提示词把条目里的换行渲染成「 ｜ 」，逐字包含匹配会
        漏掉长条目；先到先得又会命中更短的孪生条目。两个坑都堵上之后，兜底路径才勉强可用。
        """
        from ..brain import examiner_bank as eb

        line = cls._PLAN_PREFIX.sub("", (text or "").strip())
        probe = cls._norm_probe(line.split("提问：", 1)[0])
        needle = cls._norm_probe(line)
        best: dict | None = None
        best_len = 0
        for item in eb._all_items():
            zh = cls._norm_probe(item.get("probe_zh"))
            if not zh:
                continue
            if (zh in needle or (probe and probe in zh)) and len(zh) > best_len:
                best, best_len = item, len(zh)
        return best
    #: 通道 id 的形状（小写字母开头）。目标行与「还没探查过的方面」行都用它 findall，
    #: 不能只匹配**第一个**括号组：对照通道的标签里还有一层括号
    #: （`- 一般认知控制（对照通道）（general_cognition）：…`），只看第一组会漏掉它。
    _ID_RE = re.compile(r"[（(]([a-z][a-z0-9_]*)[）)]")

    def _ask_question(self, messages: list[ChatMessage], seed: int,
                      context: Optional[dict] = None) -> str:
        """照用**参考条目**（有的话轮换着用）；题库覆盖不到的方面就自己出题。

        真实模型会用自己的话改写这些方面、也会自己造题；mock 照用条目文本，但**同样输出
        `FOCUS:`** —— 这是提问者的输出契约（自报这一轮在问哪个通道），离线路径也必须跑到，
        否则「自行出题的轮次如何做维度归属」这件事就没被验证过。

        计划信息来自 `context`（`probe_item` / `plan_dims`，**mock 专用**，与分析者的
        `probe_dims` 同一性质）：`probe_item` 是这一轮真正下发给大脑的那条条目 —— 用它而不是
        回查文本，才能保证「问的话、记的 `probe_key`、自报的 `FOCUS`」三者一致。没有 context
        （测试里直接调 `chat`）时，退回从提示词里解析计划行。

        自行出题的判据照着提示词的规则来：轮提示里点名了「还没探查过的方面」，而题库条目
        覆盖不到它（离线时只有对照通道属于这种）→ 自己出一道并自报该通道。
        """
        ctx = context or {}
        system = next((m.content for m in messages if m.role == "system"), "")
        user_prompt = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        goals: list[str] = []
        for line in system.splitlines():
            if not line.lstrip().startswith("- "):
                continue
            ids = self._ID_RE.findall(line)
            if ids:
                goals.append(ids[0])
        plan: list[str] = []
        if "计划探查的方面" in system:
            plan = [
                m.group(1).strip()
                for line in system.splitlines()
                if (m := self._PLAN_LINE.match(line))
            ]
        # 已经问过的问题及其次数（从对话记录里读）——补问同一通道时不能逐字重复，
        # 重复的场景对儿童式探查没有新信息，只会白烧两轮调用。
        asked = Counter(
            question for question, _reply in self._pairs_of(user_prompt) if question
        )

        # 补问轮：提示词点名「本轮必须探查」某个方面 —— 照办（这是配额的兜底手段）
        must = str(ctx.get("must_focus") or "")
        if must and must in goals:
            question = self._improvise(must, asked)
            return f"{question}\nQUESTION: {question}\nFOCUS: {must}"

        # 题库覆盖不到的、且还没探查过的方面 → 自己出题（mock 版的「鼓励自己出题」）
        plan_dims = set(ctx.get("plan_dims") or ())
        if not plan_dims:
            for text in plan:
                plan_dims |= set((self._bank_item(text) or {}).get("dims") or {})
        for target in self._ID_RE.findall(user_prompt):
            if target in goals and target not in plan_dims:
                question = self._improvise(target, asked)
                return f"{question}\nQUESTION: {question}\nFOCUS: {target}"

        item = ctx.get("probe_item")
        if item:
            return self._render_item(item)
        if plan:
            focus = self._focus_of(self._bank_item(plan[seed % len(plan)]), goals)
            return self._present(plan[seed % len(plan)], focus)
        # 完全没有计划（自由提问 / 用户关闭题库）：自己出一道，并自报这一轮在问哪个通道
        question = _DIALOGUE_QUESTIONS[seed % len(_DIALOGUE_QUESTIONS)]
        focus = f"\nFOCUS: {goals[seed % len(goals)]}" if goals else ""
        return f"{question}\nQUESTION: {question}{focus}"

    @staticmethod
    def _improvise(target: str, asked: "Counter[str]") -> str:
        """给某个通道自己出一道**还没问过**的题（mock 版的「按维度自己设计问题」）。

        问句池按通道取一个确定的偏移，然后挑第一条本次对话里还没出现过的：
        同一通道的多个问题因此互不重复，且补问次序完全可复现（不依赖随机数）。

        池子被问完时（通道被反复补问）挑**问得最少**的那条再来一次，并在文本里写明这是
        第几次问它 —— 宁可换个说法再说一次，也不要凭空声称「已经问够了」，更不要逐字重复：
        逐字重复对儿童式探查没有新信息，只是白烧两轮调用。
        """
        pool = list(_DIALOGUE_QUESTIONS)
        preferred = _IMPROVISED_QUESTIONS.get(target)
        if preferred and not asked.get(preferred):
            return preferred
        def times_used(text: str) -> int:
            """这条问句被问过几次 —— 连带编号后缀的那些一起算（否则每次都算「第 2 次」）。"""
            return sum(n for asked_text, n in asked.items() if asked_text.startswith(text))

        offset = int(_uniform(target, "improvise") * len(pool))
        rarest: tuple[str, int] | None = None
        for step in range(len(pool)):
            candidate = pool[(offset + step) % len(pool)]
            count = times_used(candidate)
            if count == 0:
                return candidate
            if rarest is None or count < rarest[1]:
                rarest = (candidate, count)
        candidate, count = rarest or (pool[offset], 0)
        return f"{candidate}（同一件事换个说法，第 {count + 1} 次）"

    def _render_item(self, item: dict) -> str:
        """把这一轮的探查条目渲染成提问者的输出：呈现 + `QUESTION:` + `FOCUS:`。"""
        text = (item.get("probe_zh") or item.get("question_zh") or "").strip()
        return self._present(text, self._focus_of(item, ()))

    @staticmethod
    def _focus_of(item: Optional[dict], goals: list[str]) -> str:
        """这一轮的 `FOCUS`：取条目权重最大的维度。

        同权重时按 `SCORED_CHANNELS` 的**固定通道顺序**取首个命中 —— 不用字母序：
        字母序决胜是隐式的、语义上无依据的排序（`max(dims, key=(w, name))` 实际按维度 id
        字典序最大者胜），会让两个并列维度之间的归属随命名偶然翻转。
        mock 没有「意图」，只能按条目的维度权重定；真实模型自报的是它自己的判断。
        条目未知时退回首观察目标（而不是静默不给 —— 少了 `FOCUS`，这一轮就没有归属）。
        """
        dims = (item or {}).get("dims") or {}
        if dims:
            order = list(SCORED_CHANNELS)
            ranked = sorted(
                dims.items(),
                key=lambda kv: (
                    -float(kv[1]),
                    order.index(kv[0]) if kv[0] in order else len(order),
                ),
            )
            return f"\nFOCUS: {ranked[0][0]}"
        return f"\nFOCUS: {goals[0]}" if goals else ""

    @staticmethod
    def _present(text: str, focus: str) -> str:
        """情境式条目渲染成「情境 + 处理方式提问」两段（这才是儿童实际听到的内容），
        其余条目直接呈现；`QUESTION:` 行只给问题本身，便于上层解析出干净的问题。"""
        if "情境：" in text:
            scenario, _, ask = text.partition("提问：")
            presentation = (scenario.replace("情境：", "（设定情境）")
                            .replace(" ｜ ", "\n").strip())
            ask = ask.strip() or "这种情况你会怎么处理？"
            return f"{presentation}\n{ask}\nQUESTION: {ask}{focus}"
        return f"{text}\nQUESTION: {text}{focus}"

    # ---------- 分析者（表型打分）----------

    @staticmethod
    def _pairs_of(user_text: str) -> list[tuple[str, str]]:
        """把记录拆成 (提问者的呈现, 孩子的回应) 有序对。"""
        pairs: list[tuple[str, str]] = []
        last_question = ""
        for line in user_text.splitlines():
            if line.startswith("提问者："):
                last_question = line.split("：", 1)[1].strip()
            elif line.startswith("孩子："):
                pairs.append((last_question, line.split("：", 1)[1].strip()))
        return pairs

    def _analyze(self, messages: list[ChatMessage], context: Optional[dict] = None) -> str:
        """由对话里孩子的回应质量推定各维度读数（规则式，盲法：不看 θ）。

        维度取自分析者系统提示里列出的目标，因此上传量表（自定义维度）也能走通。
        **每个维度只吃探查过该维度的那些轮次** —— 这正是「选择性受损」能被观察到的原因：
        社会认知相关的几轮贫乏、而常规/兴趣相关的几轮照常，这两件事必须落在不同的维度上，
        否则机制异质性分析拿不到任何区分维度。

        `context["probe_dims"]` 是逐轮的维度归属（由调用方按轮次顺序给出，mock 专用）。
        """
        sys_text = next((m.content for m in messages if m.role == "system"), "")
        user_text = "\n".join(m.content for m in messages if m.role == "user")
        dims = re.findall(r'-\s*"([^"]+)"', sys_text)
        if not dims:
            dims = list(SCORED_CHANNELS)

        pairs = self._pairs_of(user_text)
        scored = [(question, reply, _QUALITY_SCORE[_reply_quality(reply)])
                  for question, reply in pairs]
        probe_dims = list((context or {}).get("probe_dims") or ())

        # 逐维度收集证据：第 i 轮的维度 = probe_dims[i]
        per_dim: dict[str, list[float]] = {}
        for index, (_question, _reply, score) in enumerate(scored):
            if index >= len(probe_dims):
                continue
            for dim in probe_dims[index] or ():
                per_dim.setdefault(dim, []).append(score)

        overall = (
            sum(score for _q, _r, score in scored) / len(scored) if scored else 0.5
        )
        out_dims: dict[str, Optional[float]] = {}
        unprobed: list[str] = []
        for dim in dims:
            evidence = per_dim.get(dim)
            if evidence:
                value = sum(evidence) / len(evidence)
            elif per_dim:
                # 本轮没有探查过这一维 → **没有证据**，不能给「典型值」。
                # 这里必须与「探查过但表现贫乏」分开，也不能把「没问过」判成「表现正常」：
                # 那会把未覆盖的维度系统性地推向典型，下游（特异性/剂量-反应）就会以为
                # 该维度「在扰动下不变」——凭空造出一个阴性结果。故记 None（无证据）。
                out_dims[dim] = None
                unprobed.append(dim)
                continue
            else:
                value = overall
            jitter = (_uniform(dim, "analyst-jitter") - 0.5) * 0.06
            out_dims[dim] = round(max(0.05, min(0.95, value + jitter)), 2)

        evidence = [
            {
                "dim": (probe_dims[i] or [dims[i % len(dims)]])[0] if i < len(probe_dims) else dims[0],
                "quote": reply[:60],
                "reading": "回应较短或回避，参与与展开有限。" if _reply_quality(reply)
                           in ("poor", "minimal") else "回应具体、能展开情境与人称视角。",
            }
            for i, (_q, reply, _s) in enumerate(scored[:3])
        ]
        # 逐维的行为记录（协议阶段 10）：score 0–100（100 = 与同龄典型发展无异）、
        # confidence、支持该判断的观察、任务级测量、相对同龄典型基线的分差。
        # **没有证据的维度不进记录**（不填 0–100 的分数）——否则「没问过」会被读成一个读数。
        dimensions: dict[str, dict] = {}
        for dim in dims:
            if out_dims.get(dim) is None:
                continue
            value = out_dims[dim]
            turn_measures = [
                {"task": f"第 {index + 1} 轮探查", "measurement": f"回应质量 {score:.2f}"}
                for index, (_q, _r, score) in enumerate(scored)
                if index < len(probe_dims) and dim in (probe_dims[index] or ())
            ][:3]
            dimensions[dim] = {
                "score": round(value * 100),
                "confidence": 0.8 if len(turn_measures) >= 2 else (0.5 if turn_measures else 0.4),
                "observations": [
                    e["reading"] for e in evidence if e.get("dim") == dim
                ] or ["回应具体、能展开情境与人称视角。"],
                "task_measurements": turn_measures,
                "deviation": round((value - 1.0) * 100),
            }
        scored_dims = {k: v for k, v in out_dims.items() if v is not None}
        mean = sum(scored_dims.values()) / len(scored_dims) if scored_dims else 0.0
        payload = {
            "dims": out_dims,
            "dimensions": dimensions,
            "confidence": 0.8 if len(scored) >= 4 else 0.5,
            "evidence": evidence,
            "narrative": (
                f"共 {len(scored)} 轮回应，总体表现读数约 {mean:.2f}。"
                "不同方面的回应在具体性与视角转换上有差异，故各维度读数并不齐平。"
            ),
            "notes": "（mock 规则式分析：读数由回应长度与回避词频推定，不是 LLM 的判断；"
                     "deviation 以同龄典型基线 = 100 分计。真实结论必须用真实 provider 跑。）"
                     + (f"本轮未探查：{'、'.join(unprobed)}（记为空读数，不按典型计）。"
                        if unprobed else ""),
            **_mock_conclusion(scored_dims, dims, scored, unprobed),
        }
        return json.dumps(payload, ensure_ascii=False)

    # ---------- 初始化回应 ----------

    @staticmethod
    def _acknowledge(messages: list[ChatMessage]) -> str:
        sys_text = next((m.content for m in messages if m.role == "system"), "")
        fp = ""
        for line in sys_text.splitlines():
            if line.startswith("BRAIN_FINGERPRINT:"):
                fp = line.split(":", 1)[1].strip()
                break
        return (
            "认知架构已载入。我已作为该人工心智开始运行：\n"
            f"- 指纹：{fp or '(未提供)'}\n"
            "- 基础过程参数已按给定完整度生效；高阶能力将由这些基础过程组合涌现，我不直接设定它们。\n"
            "SELF_STATE: OK"
        )


# ---------------------------------------------------------------------------
# OpenAI 兼容 provider（GPT / DeepSeek / Qwen / 本地服务）
# ---------------------------------------------------------------------------

class OpenAICompatProvider:
    """OpenAI 兼容 chat completions。BYOK：Key 只从本地配置/环境变量读取，不上报。"""

    name = "openai"

    def __init__(self, settings: LLMSettings) -> None:
        if not settings.api_key:
            raise LLMError(
                "未配置 API Key。请设置环境变量 PACS_LLM_API_KEY，"
                "或在 config.json 的 llm.api_key 中填写（config.json 已在 .gitignore 中）。"
            )
        self.settings = settings
        self.model = settings.model
        self.base_url = settings.base_url.rstrip("/")

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        item: object = None,
        perturbation: object = None,
        seed: int = 0,
        role: str = "",
        context: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # role/seed/item/context 只用于 mock 的角色分流与确定性；
        # 真实调用把它们编码在提示词里，故此处忽略。
        # 额度：调用方给的就用它，否则用配置里的。
        tokens = int(max_tokens or self.settings.max_tokens)
        # 分析者一次要写完一整段 JSON（逐维记录 + 约 350 字结论 + 判定依据），是输出最长的
        # 角色：额度**既不低于下限、也不低于用户配置**。把配置也并进来是必需的 —— 界面上
        # 那句「把 max_tokens 调大」就是冲它说的，而定值会把那句话对它彻底屏蔽掉。
        if role == "analyst":
            tokens = max(tokens, int(self.settings.max_tokens), ANALYST_MIN_TOKENS)
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            payload["max_tokens"] = tokens
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                # 取正文这一步会抛 `_EmptyContentError`（空回答），由下面的分支决定重试或报错
                content = _content_of(data, role)
                return content
            except urllib.error.HTTPError as exc:  # 4xx/5xx
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                last_error = LLMError(f"HTTP {exc.code}：{detail or exc.reason}")
                # 端点不认我们要求的额度（有些模型的输出上限比下限还低）：退回配置值再来一次。
                # 不这么做的话，「额度要多了」会让整次观察直接失败 —— 那比额度小更难查。
                if (exc.code == 400 and "max_tokens" in detail.lower() and attempt == 0
                        and tokens > int(self.settings.max_tokens)):
                    tokens = max(256, int(self.settings.max_tokens))
                    continue
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LLMError(f"网络错误：{exc}")
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error from exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise LLMError(f"响应格式异常：{exc}") from exc
            except _EmptyContentError as exc:
                # 空回答重试一次：偶发的空响应是有的。重试仍空就**报错，不返回空串** ——
                # 空串会被当成「被试没回答 / 分析者没给读数」混进记录，把一次失败的调用
                # 变成一条假的行为数据（分析者会据此压低读数），这比直接失败更糟。
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise exc from None
        raise last_error or LLMError("未知 LLM 错误")


def _content_of(data: dict, role: str) -> str:
    """从 chat completions 响应里取正文；空正文一律报错并说明**为什么可能空**。

    真实 provider 返回空 `content` 的常见原因（这几种都必须让用户看见，而不是静默吞掉）：
    - **推理型模型**：思维链占用全部额度，`content` 为空串而答案还在 `reasoning_content`
      里（`finish_reason="length"`）—— 额度调大或换非推理模型即可；
    - 输出被 `max_tokens` 截断到只剩零个字符；
    - 服务端内容过滤。
    """
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"响应格式异常：缺少 choices（{exc}）") from exc
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = ""
    if str(content).strip():
        return str(content)
    finish = str(choice.get("finish_reason") or "未提供")
    reasoning = str(message.get("reasoning_content") or "")
    hint = ""
    if reasoning:
        hint = (f"；该模型的思维链输出了 {len(reasoning)} 字，可能把 max_tokens "
                "耗在了 reasoning_content 上（请调大 max_tokens 或换用非推理模型）")
    elif finish == "length":
        hint = "；输出在产出正文前就被 max_tokens 截断（请调大 max_tokens）"
    who = {"brain": "模拟大脑", "examiner": "提问者", "analyst": "分析者"}.get(role, "模型")
    raise _EmptyContentError(f"{who}返回空内容（finish_reason={finish}）{hint}")


def get_provider(settings: LLMSettings) -> LLMProvider:
    """按设置构造 provider。`mock` 不需要 Key，保证无网络可跑。"""
    if settings.provider == "openai":
        return OpenAICompatProvider(settings)
    return MockProvider()


def get_examiner_provider(settings) -> LLMProvider:
    """构造「另外的 LLM」（提问者 / 分析者）—— 用独立的第二套配置。

    未单独配置时 `Settings.load` 已让它跟随模拟大脑；这里只是显式取用，
    防止调用方误用同一对象而丢掉「另一个模型」的独立性。
    """
    # 走 `examiner_effective()`：同端点缺 Key 时回落模拟大脑的 Key，
    # 与 `Settings.examiner_ready()` 读同一个判定，避免「说可用、调用却说没 Key」。
    effective = getattr(settings, "examiner_effective", None)
    return get_provider(effective() if callable(effective) else settings.examiner)
