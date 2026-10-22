"""扰动引擎（FR-4、FR-5）。

`Perturbation` = 一个 θ 向量（primitive → 完整度，1.0 = 完整）+ 一个**非特异的通用退化**参数。

- θ 是**构造即已知的** ground truth：它不需要被估计，是我们设定并施加的。这正是它能充当
  Model 1 缺失的 `latent state` 列的原因（PRD §1.3）。
- `theta_id` 由 θ 的**取值**确定性派生（同 θ 同 id），便于去重与复现。
- 采样支持：单因素剂量梯度（`sample_singles`）、二阶组合（`sample_pairs`）、
  Latin hypercube（`sample_lhs`）。不穷举全部组合（FR-5 要求 2）。
- `generic_degradation` 是 FR-11 Control 1：整体能力随机下降，用于区分
  「特异的认知扰动」与「什么都变差」。
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass

from ..ontology.primitives import DEFAULT_ACTIVE, DEFAULT_DOSE_LEVELS, INTACT, PRIMITIVES

DESIGNS = ("intact", "single", "pair", "lhs", "generic_control")

#: 设计模式的中文名。`describe()` 把它写进描述，界面与 CLI 因此共用同一套措辞。
DESIGN_ZH: dict[str, str] = {
    "intact": "完整心智",
    "single": "单因素",
    "pair": "组合",
    "composed": "组合",      # compose() 在合成多个扰动时写的就是这个值
    "lhs": "分层采样",
    "custom": "自定义",
    "generic_control": "通用退化对照",
    "baseline_aggregate": "基线均值",
}

#: primitive id → 中文名。描述里**不出现英文 id**：界面与 CLI 都用中文过程名；
#: 原始 id 只在需要追溯时以等宽小字或 title 提示出现。
_PRIMITIVE_ZH: dict[str, str] = {p.id: p.name_zh for p in PRIMITIVES}


def primitive_zh(primitive_id: str) -> str:
    """过程的中文名；本体里没有的 id（如候补 primitive）原样返回，不隐藏它。"""
    return _PRIMITIVE_ZH.get(primitive_id, primitive_id)

# LHS 采样下界：完整度不取 0（保留最小能力，避免数值退化）
LHS_LOW = 0.2


@dataclass(frozen=True)
class Perturbation:
    """一次扰动配置 = 一个虚拟被试的 θ。"""

    values: dict[str, float]
    generic_degradation: float = 0.0
    design: str = "intact"
    origin: tuple[str, ...] = ()
    seed: int = 42
    note: str = ""

    # ---------- 派生属性 ----------

    @property
    def theta_id(self) -> str:
        """θ 的身份：仅由取值（与通用退化）派生 —— 同 θ 同 id。"""
        items = sorted((k, round(float(v), 4)) for k, v in self.values.items())
        payload = ";".join(f"{k}={v}" for k, v in items)
        payload += f"|g={round(float(self.generic_degradation), 4)}"
        return "theta_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def active_perturbations(self) -> dict[str, float]:
        """被削弱（< 1.0）的 primitive 及其完整度。"""
        return {k: v for k, v in sorted(self.values.items()) if v < 0.999}

    @property
    def is_intact(self) -> bool:
        return not self.active_perturbations() and self.generic_degradation <= 0.0

    def level_of(self, primitive_id: str) -> float:
        return float(self.values.get(primitive_id, INTACT))

    def describe(self, *, include_design: bool = True) -> str:
        """人类可读的扰动描述：**中文过程名 + 完整度**。

        界面与 CLI 都用这一份措辞，因此这里不出现英文 primitive id（原始 id 留待需要追溯时
        以等宽小字或 title 提示展示）。`include_design=False` 用于生成大脑默认名称这类
        不该带设计标签的场合，例如「显著性分配 0.30」而不是「[自定义] 显著性分配 0.30」。
        """
        if self.is_intact and self.generic_degradation <= 0:
            return "完整心智（无扰动）"
        parts = [f"{primitive_zh(k)} {v:.2f}" for k, v in self.active_perturbations().items()]
        if self.generic_degradation > 0:
            parts.append(f"通用退化 {self.generic_degradation:.2f}")
        body = "、".join(parts) if parts else "无扰动"
        if not include_design:
            return body
        return f"[{DESIGN_ZH.get(self.design, self.design)}] {body}"

    def describe_ids(self) -> str:
        """同一批扰动的 id 形式，供 title 提示使用（可追溯但不占据界面主语言）。"""
        if self.is_intact and self.generic_degradation <= 0:
            return "intact"
        parts = [f"{k}={v:.2f}" for k, v in self.active_perturbations().items()]
        if self.generic_degradation > 0:
            parts.append(f"generic_degradation={self.generic_degradation:.2f}")
        return "、".join(parts)

    # ---------- 序列化 ----------

    def to_dict(self) -> dict:
        return {
            "theta_id": self.theta_id,
            "values": {k: round(float(v), 4) for k, v in self.values.items()},
            "generic_degradation": round(float(self.generic_degradation), 4),
            "design": self.design,
            "origin": list(self.origin),
            "seed": self.seed,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Perturbation":
        return cls(
            values={k: float(v) for k, v in (data.get("values") or {}).items()},
            generic_degradation=float(data.get("generic_degradation", 0.0)),
            design=data.get("design", "custom"),
            origin=tuple(data.get("origin") or ()),
            seed=int(data.get("seed", 42)),
            note=data.get("note", ""),
        )

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"Perturbation({self.theta_id}: {self.describe()})"


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------

def intact_values(primitives: tuple[str, ...] = DEFAULT_ACTIVE) -> dict[str, float]:
    return {p: INTACT for p in primitives}


def intact(seed: int = 42, note: str = "") -> Perturbation:
    """完整心智（所有 primitive = 1.0）—— FR-9.1 baseline validation 的起点。"""
    return Perturbation(
        values=intact_values(), generic_degradation=0.0, design="intact", origin=(), seed=seed, note=note
    )


def single(primitive_id: str, dose: float, seed: int = 42, note: str = "") -> Perturbation:
    """单因素扰动：只削弱一个 primitive，其余保持完整（FR-1 标准 3「可单独扰动」）。"""
    if primitive_id not in DEFAULT_ACTIVE:
        raise KeyError(f"未知 primitive：{primitive_id}")
    if not (0.0 < dose <= 1.0):
        raise ValueError(f"dose 必须落在 (0, 1]，收到 {dose}")
    values = intact_values()
    values[primitive_id] = float(dose)
    return Perturbation(
        values=values,
        generic_degradation=0.0,
        design="single",
        origin=(primitive_id,),
        seed=seed,
        note=note or f"{primitive_id} 降至 {dose:.2f}",
    )


def from_values(
    values: dict[str, float],
    *,
    design: str = "custom",
    generic_degradation: float = 0.0,
    origin: tuple[str, ...] | None = None,
    seed: int = 42,
    note: str = "",
) -> Perturbation:
    """由给定取值构造（未列出的 primitive 视为完整）。"""
    merged = intact_values()
    for k, v in values.items():
        merged[k] = float(v)
    if origin is None:
        origin = tuple(sorted(k for k, v in merged.items() if v < 0.999))
    return Perturbation(
        values=merged,
        generic_degradation=float(generic_degradation),
        design=design,
        origin=origin,
        seed=seed,
        note=note,
    )


def generic_degradation_control(level: float, seed: int = 42) -> Perturbation:
    """FR-11 Control 1：所有 primitive 保持完整，但整体能力随机下降 `level`。

    判据：若该控制下**所有**任务（含对照通道）都变差 → 说明观测到的表型可能是通用退化，
    而非特异的认知机制。
    """
    if not (0.0 <= level < 1.0):
        raise ValueError(f"generic_degradation 必须落在 [0, 1)，收到 {level}")
    return Perturbation(
        values=intact_values(),
        generic_degradation=float(level),
        design="generic_control",
        origin=(),
        seed=seed,
        note=f"通用退化 {level:.2f}（非特异对照）",
    )


def compose(*perturbations: Perturbation, seed: int | None = None, note: str = "") -> Perturbation:
    """组合多个扰动（FR-5）。

    合并规则：同一 primitive 取**最弱**（min）——两个扰动同时作用于同一过程时，
    系统只保留更差的那一档；`generic_degradation` 取 max。
    """
    if not perturbations:
        raise ValueError("compose 至少需要一个扰动")
    values = intact_values()
    generic = 0.0
    origin: list[str] = []
    for p in perturbations:
        for k, v in p.values.items():
            values[k] = min(values.get(k, INTACT), float(v))
        generic = max(generic, float(p.generic_degradation))
    for k, v in values.items():
        if v < 0.999 and k not in origin:
            origin.append(k)
    return Perturbation(
        values=values,
        generic_degradation=generic,
        design=perturbations[0].design if len(perturbations) == 1 else "composed",
        origin=tuple(sorted(origin)),
        seed=seed if seed is not None else perturbations[0].seed,
        note=note or " + ".join(p.describe() for p in perturbations),
    )


# ---------------------------------------------------------------------------
# 采样（FR-5）
# ---------------------------------------------------------------------------

def sample_singles(
    primitives: tuple[str, ...] = DEFAULT_ACTIVE,
    doses: tuple[float, ...] = DEFAULT_DOSE_LEVELS,
    seed: int = 42,
    include_intact: bool = True,
    n_intact: int = 5,
) -> list[Perturbation]:
    """单因素 × 剂量梯度设计：每个 primitive 在每个 dose 上各一个被试（FR-3 的 dose-response）。

    `n_intact`：完整心智基线的**重复次数**（默认 5）。基线是全部比较的参照，而单次剖面
    的抽样噪声约 ±0.09；重复 k 次取平均可把基线噪声降到约 1/√k。
    """
    designs: list[Perturbation] = []
    if include_intact:
        for rep in range(max(1, n_intact)):
            designs.append(intact(seed=seed + rep, note=f"baseline#{rep + 1}"))
    for pid in primitives:
        for dose in doses:
            if dose >= 0.999:      # dose=1.0 即完整，避免与 baseline 重复
                continue
            designs.append(single(pid, dose, seed=seed))
    return designs


def sample_pairs(
    primitives: tuple[str, ...] = DEFAULT_ACTIVE,
    dose: float = 0.4,
    seed: int = 42,
    max_pairs: int | None = None,
) -> list[Perturbation]:
    """二阶组合设计（FR-5 要求 1）：所有二元组合在同一 dose 上各一个被试。

    用于估计**交互效应**（可加性 / 超加性 / 拮抗），并据以实测相关结构。
    为控制规模可用 `max_pairs` 截断（按确定性顺序）。
    """
    designs: list[Perturbation] = []
    for a, b in itertools.combinations(primitives, 2):
        designs.append(
            compose(single(a, dose, seed=seed), single(b, dose, seed=seed),
                    seed=seed, note=f"{a}+{b} @ {dose:.2f}")
        )
        if max_pairs is not None and len(designs) >= max_pairs:
            break
    return designs


def sample_lhs(
    primitives: tuple[str, ...] = DEFAULT_ACTIVE,
    n_samples: int = 32,
    seed: int = 42,
    low: float = LHS_LOW,
    snap_levels: tuple[float, ...] | None = None,
) -> list[Perturbation]:
    """Latin hypercube 采样（FR-5 要求 2）：在 15 维空间中分层覆盖，不穷举。

    实现：对每个维度把 [low, 1] 分成 n_samples 层，各层随机取一点并独立置换；
    因而每个维度在每一层恰好落一个样本。确定性（显式 seed）。
    """
    if n_samples < 1:
        raise ValueError("n_samples 必须 ≥ 1")
    d = len(primitives)
    rng = random.Random(seed)
    # 每个维度：层内抖动 + 层序置换
    jitter = [[rng.random() for _ in range(n_samples)] for _ in range(d)]
    perms: list[list[int]] = []
    for _ in range(d):
        order = list(range(n_samples))
        rng.shuffle(order)
        perms.append(order)

    designs: list[Perturbation] = []
    for i in range(n_samples):
        values: dict[str, float] = {}
        for dim_idx, pid in enumerate(primitives):
            stratum = perms[dim_idx][i]
            u = (stratum + jitter[dim_idx][i]) / n_samples
            level = low + u * (1.0 - low)
            if snap_levels:
                level = min(snap_levels, key=lambda lv: abs(lv - level))
            values[pid] = round(min(1.0, max(low, level)), 4)
        designs.append(
            from_values(values, design="lhs", seed=seed, note=f"LHS #{i}")
        )
    return designs


def sample_design(
    mode: str = "single",
    primitives: tuple[str, ...] = DEFAULT_ACTIVE,
    doses: tuple[float, ...] = DEFAULT_DOSE_LEVELS,
    seed: int = 42,
    n_samples: int = 32,
    pair_dose: float = 0.4,
    max_pairs: int | None = None,
) -> list[Perturbation]:
    """统一采样入口（CLI `pacs sample --mode ...`）。"""
    if mode == "single":
        return sample_singles(primitives, doses, seed)
    if mode == "pair":
        return sample_pairs(primitives, pair_dose, seed, max_pairs)
    if mode == "lhs":
        return sample_lhs(primitives, n_samples, seed)
    if mode == "intact":
        return [intact(seed=seed)]
    raise ValueError(f"未知采样模式：{mode}（可选：single / pair / lhs / intact）")


def interaction_sign(a: float, b: float, ab: float, baseline: float) -> str:
    """交互类型判定（以「损伤量」为尺度）。

    `a`/`b` = 单独扰动时的表型值；`ab` = 组合扰动时的表型值；`baseline` = 完整心智的表型值。
    返回 additive（可加）| superadditive（超加，即协同）| subadditive（次加，即拮抗）。
    """
    expected = a + b - baseline          # 纯可加时的预期
    tol = 0.05 * max(1e-6, abs(baseline - expected) + abs(expected))
    if ab < expected - tol:
        return "superadditive"
    if ab > expected + tol:
        return "subadditive"
    return "additive"
