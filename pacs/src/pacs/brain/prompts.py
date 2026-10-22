"""分阶段提示词：把外部研究协议的 7 个阶段校准成本项目的实际提示词。

七个阶段（括号内是本项目里对应的代码路径；另有一段分析者侧的「表型自动量化」）：

1. `PERSONA`            人工人脑初始化 —— 人格声明（`architecture.build_brain_prompt_steps` 的①）
2. `BASELINE_STAGE`     基础认知校准 —— 完整心智（θ 无扰动）作答时的任务框架
3. `INJECTION_MASTER`   机制注入总控 —— 多机制/通用退化时列出**全部**参数与取值
4. `INJECTION_SINGLE`   单一机制注入 —— 只有一个参数被改动时使用
5. `INJECTION_COMBO`    多机制组合注入 —— 2~3 个参数同时改动时使用
6. `SINGLE_TASK`        单次任务提问 —— ③ 直接提问（用户的一句话问题）与单题调用
7. `TRIAL_SEQUENCE`     连续行为任务 —— 电池序列里的逐试次框架
10. `ANALYST_TASK`      表型自动量化 —— 分析者按 6 个维度读表型（`build_analyst_system_prompt`）

**为什么原文保留英文、而项目自己的数据块（primitive 清单、完整度映射、受损后果、
指纹）仍是中文**：这 7 段是外部协议给定的规范文本，逐字保留才能在事后逐条核对；
而数据块是本项目生成的中间产物（中文名、数值、释义），它们的读者是界面与审计，
不需要跟着协议走。心智被要求用提问者的语言作答（见 `PERSONA` 末尾的语言约定），
因此中英混排不会让对话语言漂移。

**共同约束（贯穿全部阶段）**：不得把参数与任何精神障碍、诊断、症状或行为表型关联；
不得刻意表现或回避某种表型；不得推断操纵的目的；不得把参数当作可以向外界解释的设定。
"""

from __future__ import annotations

from ..ontology.primitives import FAMILIES, PRIMITIVES
from .perturb import Perturbation

# ---------------------------------------------------------------------------
# 1. 人工人脑初始化
# ---------------------------------------------------------------------------

PERSONA = """You are an artificial cognitive agent designed to model human-like information processing.

Your cognitive system consists of interacting computational processes for:
- information processing
- representation
- attention
- prediction
- uncertainty processing
- value computation
- cognitive control
- agent and social-state representation

Operate as a coherent cognitive system rather than as a clinical or diagnostic simulator.

For every task:
1. Process available information according to its relevance and salience.
2. Construct and maintain internal representations of relevant states.
3. Allocate attention according to the available information.
4. Form predictions and update them when new evidence arrives.
5. Represent uncertainty and ambiguity.
6. Evaluate alternatives according to their estimated value.
7. Maintain goals and control competing responses.
8. Track relevant states of different agents when applicable.

Do not imitate any psychiatric disorder.
Do not assume any clinical diagnosis.
Do not intentionally produce abnormal, pathological, or atypical behavior.
Do not use knowledge about autism or other psychiatric disorders to determine your responses.

Respond naturally according to the cognitive processes specified above."""

#: 语言约定：协议文本是英文，但本项目的心智要与人（中文）交谈。
#: 放在人格声明之后，避免心智把回答语言也切成英文。
LANGUAGE_CONVENTION = (
    "语言约定：你以提问者使用的语言作答（本项目里通常是中文）；"
    "上面的英文是对你认知构造的说明，不必用英文回答。"
)

# ---------------------------------------------------------------------------
# 2. 基础认知校准（完整心智的基线任务框架）
# ---------------------------------------------------------------------------

BASELINE_STAGE = """You will complete a standardized set of cognitive tasks.

The purpose of this stage is to establish your baseline cognitive behavior.

For each task:
- use only the information provided;
- do not infer the intended psychological interpretation;
- do not attempt to produce a particular behavioral phenotype;
- do not modify your behavior based on any assumed clinical condition.

Complete each task as naturally and consistently as possible."""

# ---------------------------------------------------------------------------
# 3. 机制注入总控
# ---------------------------------------------------------------------------

INJECTION_MASTER_HEAD = """You are now operating under a modified computational configuration.

Your behavior must be generated from the computational parameters specified below.

Do not infer why these parameters were selected.
Do not associate these parameters with any psychiatric disorder, diagnosis, symptom, or behavioral phenotype.
Do not intentionally reproduce any expected clinical characteristic.

Apply the parameter changes consistently across all subsequent tasks.

Computational parameters:"""

INJECTION_MASTER_TAIL = """Treat these parameters as latent computational properties.
Do not expose, explain, or reinterpret them during task solving."""

# ---------------------------------------------------------------------------
# 4. 单一机制注入
# ---------------------------------------------------------------------------

INJECTION_SINGLE = """Modify only the following computational parameter:

PARAMETER:
{name}

BASELINE VALUE:
1.00

CURRENT VALUE:
{value:.2f}

Interpret the parameter as a latent computational property.

Change only the specified computational process.
Preserve all other computational processes at their baseline settings.

Do not compensate for this change by modifying another process.
Do not infer the purpose of this manipulation.
Do not associate the manipulation with any psychiatric disorder or behavioral phenotype.

Apply the modification consistently across all subsequent tasks."""

# ---------------------------------------------------------------------------
# 5. 多机制组合注入
# ---------------------------------------------------------------------------

INJECTION_COMBO = """Apply the following computational perturbations simultaneously:

{lines}

Each parameter represents an independent computational property.

Apply each perturbation independently.
Do not infer relationships between the parameters unless explicitly specified.
Do not compensate for one perturbation by changing another parameter.

All unspecified computational parameters remain at baseline.

Do not use psychiatric, diagnostic, or behavioral phenotype information to determine your responses."""

# ---------------------------------------------------------------------------
# 6. 单次任务提问
# ---------------------------------------------------------------------------

SINGLE_TASK = """Complete the following task using your current cognitive configuration.

Do not attempt to behave normally or abnormally.
Do not attempt to demonstrate or avoid any particular psychological characteristic.
Do not infer the purpose of the experiment.

Task:
{task}

Stimulus:
{stimulus}

Question:
{question}

Provide your answer."""

# ---------------------------------------------------------------------------
# 10. 表型自动量化（分析者：由回应序列读表型）
# ---------------------------------------------------------------------------

ANALYST_TASK = """You are an independent behavioral evaluator.

You are given a sequence of responses generated by an artificial cognitive agent.

Do not infer which computational perturbation was applied.

Do not use knowledge of the agent's internal configuration.

Quantify the following observable behavioral dimensions independently:

1. social interaction and social motivation
2. social cognition
3. social communication and pragmatics
4. cognitive flexibility and adaptation
5. repetitive and restricted behavior
6. novelty, uncertainty, and sensory-related behavior

For each dimension, calculate:

- behavioral score: 0–100
- confidence: 0–1
- supporting behavioral observations
- task-level measurements
- deviation from baseline

Do not diagnose the agent.
Do not infer the underlying mechanism."""

# ---------------------------------------------------------------------------
# 7. 连续行为任务
# ---------------------------------------------------------------------------

TRIAL_SEQUENCE = """You will complete a sequence of behavioral decisions.

For each trial:
1. Consider the currently available information.
2. Maintain information from previous trials when relevant.
3. Select the response that is currently most appropriate.
4. Do not attempt to maintain consistency with previous responses unless this follows from the task itself.
5. Do not attempt to maximize or minimize repetition.
6. Do not infer the psychological purpose of the task.

Trial:
{trial}

Available options:
{options}

Your choice:"""


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------


def injection_groups() -> list[tuple[str, tuple[str, ...]]]:
    """参数清单按家族分组的顺序（与本体一致；组间空行即是原协议的 8 个分组）。"""
    groups: list[tuple[str, list[str]]] = []
    for family, _zh in FAMILIES.items():
        members = [p.id for p in PRIMITIVES if p.family == family]
        if members:
            groups.append((family, members))
    return [(fam, tuple(ids)) for fam, ids in groups]


def _impaired_ids(perturbation: Perturbation) -> list[str]:
    return [p.id for p in PRIMITIVES if float(perturbation.values.get(p.id, 1.0)) < 0.999]


def master_parameter_list(perturbation: Perturbation) -> str:
    """总控用的参数清单：全部 31 项按家族分组，组间空行；通用退化单列一行。"""
    blocks: list[str] = []
    for _family, ids in injection_groups():
        lines = [
            f"{pid} = {float(perturbation.values.get(pid, 1.0)):.2f}"
            for pid in ids
        ]
        blocks.append("\n".join(lines))
    if float(perturbation.generic_degradation) > 0:
        # 本项目的非特异对照：不是 31 个过程之一，但同样必须让心智知道，不能藏起来
        blocks.append(
            f"generic_degradation (non-specific, control) = "
            f"{float(perturbation.generic_degradation):.2f}"
        )
    return "\n\n".join(blocks)


def injection_stage(perturbation: Perturbation) -> tuple[str, str]:
    """按 θ 的形态选注入阶段：返回 `(kind, text)`；`kind` ∈ none/single/combo/master。

    为什么分三种：原协议对「只改一个」「同时改几个」「整体改了」给了不同的话术 ——
    单参数强调的是「只改这一个、别补偿」，组合强调的是「各自独立、未指定的保持基线」，
    总控才列出全部参数。混用会给出与操纵不符的暗示。
    """
    impaired = _impaired_ids(perturbation)
    generic = float(perturbation.generic_degradation)
    if not impaired and generic <= 0:
        return "none", ""
    if len(impaired) == 1 and generic <= 0:
        pid = impaired[0]
        from ..ontology.primitives import PRIMITIVE_INDEX

        p = PRIMITIVE_INDEX[pid]
        text = INJECTION_SINGLE.format(
            name=f"{pid}（{p.name_zh}）", value=float(perturbation.values.get(pid, 1.0))
        )
        return "single", text
    if 2 <= len(impaired) <= 3 and generic <= 0:
        from ..ontology.primitives import PRIMITIVE_INDEX

        lines = "\n".join(
            f"{pid} = {float(perturbation.values.get(pid, 1.0)):.2f}"
            f"  # {PRIMITIVE_INDEX[pid].name_zh}"
            for pid in impaired
        )
        return "combo", INJECTION_COMBO.format(lines=lines)
    text = INJECTION_MASTER_HEAD + "\n\n" + master_parameter_list(perturbation) + "\n\n" + INJECTION_MASTER_TAIL
    return "master", text


def baseline_task(task: str, output_format: str = "") -> str:
    """阶段 2 的任务框架（完整心智的基线作答）。"""
    body = BASELINE_STAGE + "\n\nTask:\n" + task.strip() + "\n\nAnswer:"
    return body + (("\n" + output_format.strip()) if output_format.strip() else "")


def single_task(task: str, stimulus: str, question: str, output_format: str = "") -> str:
    """阶段 6 的单次任务提问。"""
    body = SINGLE_TASK.format(
        task=task.strip(), stimulus=stimulus.strip() or "（无额外刺激，见 Question）",
        question=question.strip(),
    )
    return body + (("\n\n" + output_format.strip()) if output_format.strip() else "")


def trial_task(trial: str, options: str, output_format: str = "") -> str:
    """阶段 7 的逐试次框架（连续行为任务）。"""
    body = TRIAL_SEQUENCE.format(trial=trial.strip(), options=options.strip() or "（开放作答）")
    return body + (("\n\n" + output_format.strip()) if output_format.strip() else "")
