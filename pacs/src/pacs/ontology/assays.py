"""每个 primitive 的独立 assay（FR-3）—— **探查登记**，不是维度对应表。

设计决策：**不另写一套题目**，而是从**观察者内置条目库**（`brain/examiner_bank.py` 读到的
441 条量表条目，每条带 `primitives` 权重）里派生。这样「assay」与「观察时实际会问到的内容」
共享同一批条目，避免两套内容漂移，同时满足 FR-3 的五项条件：

1. 有定义          → `primitive.definition`（`ontology/primitives.py`）
2. 有独立行为测试   → `primary_probe`（该 primitive 权重最高的**可探查**条目）
3. 可以单独扰动     → `perturb.single(...)`
4. 有 dose-response → `dose_levels`
5. 有 specificity control → `control_probes`（对该 primitive **零敏感度**的条目）

**红线：本模块只登记「用什么探查这个机制、哪些条目是对照、dose 档位」，不登记任何
「这个机制应当在哪个测量维度上体现」的对应关系。** 表型维度读数一律由分析者 LLM 从
「提问者提问 → 心智作答」的对话里得出（`phenotype` 与 `analysis/specificity.py`），
判据也从数据本身得出（哪个维度下降最多、效应是否集中、对照通道是否稳定），
不预设期望维度。这条红线与 `README.md`「扰动与表型之间不允许有任何已知条件的关联」一致，
由 `tests/test_no_preset_mapping.py` 守住。

本模块从观察者条目库派生，只描述探查设计本身：不声明 `target_dim`（条目 → 主维度）
或 `dim_map_targets`（primitive → 外部模型维度）这类预设对应。
**语义上的差别必须说清楚**：条目库的一条是
「观察者该问什么/该看什么」，不是一道有客观正确率指标的题；因此 assay 的「独立行为测试」
现在是一次**结构化观察**（情境 + 提问 + 观察要点），而不是一个可自动判对的选项。

**全局对照任务**：FR-9.2 要求「扰动某机制不应搞坏一般认知」。条目库是孤独症量表，
**不含一般认知条目**（`general_cognition` 在条目库里 0 条），因此对照通道没有内置探题 ——
它由提问者 LLM 自己生成一道日常问题，由分析者 LLM 从同一段对话里评分（见
`phenotype/dims.py` 与 `analysis/specificity.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..brain import examiner_bank as bank
from .primitives import DEFAULT_DOSE_LEVELS, PRIMITIVES, Primitive


@dataclass(frozen=True)
class AssayProbe:
    """一条用作 assay 的条目库条目（只保留 assay 需要的字段）。

    **不含维度归属**：条目在题库里的 `dims` 标签属于题目设计元数据，不进入本登记 ——
    否则它就成了一条「机制 → 测量维度」的预设对应。
    """

    key: str
    text: str
    sensitivity: dict[str, float]
    probe_mode: str
    source_id: str
    age_hint: str


@dataclass(frozen=True)
class Assay:
    """一个 primitive 的独立 assay 设计（探查登记）。"""

    primitive_id: str
    primary_probe: AssayProbe
    sensitivity_weight: float
    control_probes: tuple[AssayProbe, ...]
    dose_levels: tuple[float, ...]
    expected_direction: str

    @property
    def n_controls(self) -> int:
        return len(self.control_probes)


def _probe(item: dict) -> AssayProbe:
    return AssayProbe(
        key=str(item.get("key") or ""),
        text=str(item.get("probe_zh") or ""),
        sensitivity={k: float(v) for k, v in (item.get("primitives") or {}).items()},
        probe_mode=str(item.get("probe_mode") or "none"),
        source_id=str(item.get("source_id") or ""),
        age_hint=str(item.get("age_hint") or ""),
    )


def _sensitivity(probe: AssayProbe, primitive_id: str) -> float:
    return float(probe.sensitivity.get(primitive_id, 0.0))


def _probe_ready_items() -> list[dict]:
    """可用于观察的条目（有探查文本、非「仅观察要点」）。"""
    out: list[dict] = []
    for item in bank._all_items():
        if item.get("probe_mode") == "none":
            continue
        if not str(item.get("probe_zh") or "").strip():
            continue
        out.append(item)
    return out


def _probes_for(primitive_id: str) -> list[tuple[dict, float]]:
    """对该 primitive 有权重的可探查条目（权重降序，同权重按键名稳定排序）。"""
    scored = [(_p, _sensitivity(_p, primitive_id)) for _p in map(_probe, _probe_ready_items())]
    return sorted(
        [(i, w) for i, w in scored if w > 0],
        key=lambda pair: (-pair[1], pair[0].key),
    )


def _control_pool(primitive_id: str, limit: int = 4) -> tuple[AssayProbe, ...]:
    """对该 primitive 零敏感度的可探查条目（按 key 稳定排序）。

    条目库不含一般认知条目，因此这里只能取「不含该 primitive 的其它条目」——
    它们提供的是「扰动这一个机制时，别的方面不该一起变差」的参照面。
    """
    zeros = [
        probe for probe in map(_probe, _probe_ready_items())
        if _sensitivity(probe, primitive_id) == 0.0
    ]
    zeros.sort(key=lambda p: p.key)
    return tuple(zeros[:limit])


def build_assays(primitives: tuple[Primitive, ...] = PRIMITIVES) -> tuple[Assay, ...]:
    """为每个 primitive 派生一个 assay。条目库未覆盖的会被标记出来（本体缺口）。"""
    assays: list[Assay] = []
    for p in primitives:
        scored = _probes_for(p.id)
        if not scored:
            # 本体有定义但条目库未覆盖 → 显式暴露，而不是静默跳过（FR-1 标准 2 未达标）
            continue
        primary, weight = scored[0]
        assays.append(
            Assay(
                primitive_id=p.id,
                primary_probe=primary,
                sensitivity_weight=weight,
                control_probes=_control_pool(p.id),
                dose_levels=p.dose_levels or DEFAULT_DOSE_LEVELS,
                expected_direction=(
                    f"{p.name_zh} ↓ → 观察「{primary.text}」时表现变差；"
                    "对照条目方面基本不变（本条不指定该机制应体现在哪个测量维度）"
                ),
            )
        )
    return tuple(assays)


ASSAYS: tuple[Assay, ...] = build_assays()
ASSAY_INDEX: dict[str, Assay] = {a.primitive_id: a for a in ASSAYS}


def get(primitive_id: str) -> Assay:
    try:
        return ASSAY_INDEX[primitive_id]
    except KeyError:
        raise KeyError(
            f"primitive {primitive_id} 没有派生出 assay（条目库未覆盖它，属本体缺口）"
        ) from None


def primitives_without_assay() -> tuple[str, ...]:
    """有本体定义、但条目库未覆盖的 primitive —— FR-1 标准 2 未达标项。"""
    return tuple(p.id for p in PRIMITIVES if p.id not in ASSAY_INDEX)


def validate_assays() -> list[str]:
    """自检 FR-1 标准 2/4/5：有独立探查、有 dose 梯度、有零敏感度对照。

    不再校验任何维度归属 —— 探查登记里本就没有维度字段（见模块红线）。
    """
    problems: list[str] = []
    for pid in primitives_without_assay():
        problems.append(f"{pid} 未派生 assay（条目库未覆盖该 primitive，FR-1 标准 2 未达标）")
    for a in ASSAYS:
        if a.sensitivity_weight <= 0:
            problems.append(f"{a.primitive_id} 的 assay 敏感度为 0")
        if not a.primary_probe.text.strip():
            problems.append(f"{a.primitive_id} 的主探查条目没有探查文本（无法实际提问）")
        if a.n_controls == 0:
            problems.append(f"{a.primitive_id} 没有 specificity control（FR-1 标准 5 未达标）")
        if len(a.dose_levels) < 2:
            problems.append(f"{a.primitive_id} 的 dose 梯度少于 2 档")
    return problems


def summary_table() -> str:
    lines = ["每个 primitive 的独立探查登记（FR-3，来源：观察者内置条目库）", "=" * 96]
    lines.append(f"{'primitive':<34}{'主探查条目':<16}{'权重':<7}{'对照数':<8}dose 档数")
    lines.append("-" * 96)
    for a in ASSAYS:
        lines.append(
            f"{a.primitive_id:<34}{a.primary_probe.key:<16}{a.sensitivity_weight:<7.2f}"
            f"{a.n_controls:<8}{len(a.dose_levels)}"
        )
    missing = primitives_without_assay()
    if missing:
        lines.append("")
        lines.append(f"⚠ 未派生 assay：{', '.join(missing)}")
    lines.append("")
    lines.append("注：本表只登记「用什么探查、哪些是对照、dose 档位」；")
    lines.append("    不登记「该机制应在哪个测量维度上体现」—— 维度读数由分析者从对话得出。")
    return "\n".join(lines)
