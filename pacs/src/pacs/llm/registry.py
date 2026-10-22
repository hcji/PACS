"""模型档案（FR-10）：支持「≥3 个模型、含不同架构/训练范式」的对照。

`Plan.txt` 第五阶段的要求：如果某个表型只在单一模型上出现，审稿人会说是 artifact。
因此需要登记每个模型的家族、架构与训练范式，并给出跨模型收敛判定。
"""

from __future__ import annotations

from dataclasses import dataclass

# 已知模型的家族 / 架构 / 训练范式（用于跨模型收敛分析；未登记者标 unknown）
KNOWN_MODELS: dict[str, tuple[str, str, str]] = {
    # name_hint: (family, architecture, training_regime)
    "gpt-4o": ("GPT", "dense transformer (proprietary)", "RLHF / instruction-tuned"),
    "gpt-4o-mini": ("GPT", "dense transformer (proprietary)", "RLHF / instruction-tuned"),
    "gpt-4.1": ("GPT", "dense transformer (proprietary)", "RLHF / instruction-tuned"),
    "gpt-3.5-turbo": ("GPT", "dense transformer (proprietary)", "RLHF / instruction-tuned"),
    "claude": ("Claude", "dense transformer (proprietary)", "RLHF / constitution AI"),
    "deepseek-chat": ("DeepSeek", "MoE transformer", "SFT + RL"),
    "deepseek-reasoner": ("DeepSeek", "MoE transformer", "RL (reasoning)"),
    "qwen": ("Qwen", "dense transformer (open weights)", "SFT + DPO"),
    "qwen2.5": ("Qwen", "dense transformer (open weights)", "SFT + DPO"),
    "llama": ("Llama", "dense transformer (open weights)", "SFT + RLHF"),
    "mistral": ("Mistral", "dense/MoE transformer (open weights)", "SFT"),
    "gemini": ("Gemini", "MoE transformer (proprietary)", "RLHF"),
    "glm": ("GLM", "dense transformer", "SFT"),
    "mock-cognitive-mind": ("Mock", "rule-based forward model", "not applicable (offline)"),
}


@dataclass
class ModelProfile:
    """一个模型的档案。"""

    model_id: str
    provider: str
    model: str
    family: str
    architecture: str
    training_regime: str
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "model": self.model,
            "family": self.family,
            "architecture": self.architecture,
            "training_regime": self.training_regime,
            "notes": self.notes,
        }


def _lookup(model: str) -> tuple[str, str, str]:
    lowered = model.lower()
    for key, value in KNOWN_MODELS.items():
        if key in lowered:
            return value
    return ("unknown", "unknown", "unknown")


def profile_for(
    provider_name: str, model: str, *, notes: str = ""
) -> ModelProfile:
    """由 provider/model 名构造档案；未登记的模型标 unknown（不猜测）。"""
    family, architecture, regime = _lookup(model)
    return ModelProfile(
        model_id=f"{provider_name}:{model}",
        provider=provider_name,
        model=model,
        family=family,
        architecture=architecture,
        training_regime=regime,
        notes=notes,
    )


def architecture_distinct(profiles: list[ModelProfile]) -> bool:
    """FR-10 的要求：≥3 个模型且架构/训练范式不完全相同。"""
    if len(profiles) < 3:
        return False
    architectures = {p.architecture for p in profiles}
    regimes = {p.training_regime for p in profiles}
    return len(architectures) > 1 or len(regimes) > 1


def render_profiles(profiles: list[ModelProfile]) -> str:
    lines = ["模型档案（FR-10）", "=" * 78]
    lines.append(f"{'model_id':<32}{'家族':<12}{'架构':<34}训练范式")
    lines.append("-" * 78)
    for p in profiles:
        lines.append(
            f"{p.model_id:<32}{p.family:<12}{p.architecture:<34}{p.training_regime}"
        )
    lines.append("")
    lines.append(
        "架构异质性判定："
        + ("满足（存在不同架构或训练范式）" if architecture_distinct(profiles) else "不满足（<3 个或完全同构）")
    )
    return "\n".join(lines)
