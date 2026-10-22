"""配置：BYOK LLM 接入 + 路径管理。

两套 LLM 配置：
- `llm`      —— 模拟大脑（被扰动的人工心智）；
- `examiner` —— **另外的 LLM**：表型观察里负责提问、以及对话结束后负责分析与表型打分。
  未单独配置时**跟随模拟大脑**，这样只配一套 Key 的用户也能直接跑通。

优先级：环境变量 > config.json > 默认值。
运行时零第三方依赖（stdlib only）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# src/pacs/config.py -> src/pacs -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PROVIDER = "PACS_LLM_PROVIDER"
ENV_API_KEY = "PACS_LLM_API_KEY"
ENV_MODEL = "PACS_LLM_MODEL"
ENV_BASE_URL = "PACS_LLM_BASE_URL"

ENV_EXAMINER_PROVIDER = "PACS_EXAMINER_LLM_PROVIDER"
ENV_EXAMINER_API_KEY = "PACS_EXAMINER_LLM_API_KEY"
ENV_EXAMINER_MODEL = "PACS_EXAMINER_LLM_MODEL"
ENV_EXAMINER_BASE_URL = "PACS_EXAMINER_LLM_BASE_URL"

PROVIDERS = ("mock", "openai")


@dataclass
class LLMSettings:
    """LLM 接入设置（BYOK：Key 仅本地，不上报）。"""

    provider: str = "mock"          # mock | openai
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.3
    #: 单词调用允许的最大输出 token 数。默认给得宽一些：真实模型（尤其推理型）会把额度
    #: 花在思维链上，512 这类小额度会让正文直接为空 —— 空正文不是「模型沉默」，
    #: 而是这次调用失败（见 `llm/provider.py` 的空回答处理）。分析者另有更大的额度下限。
    max_tokens: int = 2048
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"未知 provider：{self.provider}（可选：{PROVIDERS}）")


@dataclass
class RunSettings:
    n_instances_per_item: int = 1
    max_concurrency: int = 1


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    examiner: LLMSettings = field(default_factory=LLMSettings)
    run: RunSettings = field(default_factory=RunSettings)
    base_dir: Path = PROJECT_ROOT

    # ---------- 门禁（界面打开时要求先配置 LLM API）----------

    def brain_ready(self) -> bool:
        """模拟大脑是否已可真实调用：非 mock 且已填 Key。"""
        return self.llm.provider != "mock" and bool(self.llm.api_key)

    def same_endpoint(self) -> bool:
        """提问者 / 分析者与模拟大脑是不是**同一个端点**（同厂商 + 同 base_url）。

        用来判定「能不能共用一把 Key」：端点相同才谈得上共用，否则拿大脑的 Key 去请求
        另一个厂商的服务只会得到 401。
        """
        return (
            self.examiner.provider == self.llm.provider
            and self.examiner.base_url.rstrip("/") == self.llm.base_url.rstrip("/")
        )

    def examiner_key(self) -> str:
        """提问者 / 分析者实际可用的 Key：自己的优先，**同端点**时回落到模拟大脑的。

        为什么要有这条回落：README 承诺「只配一套 Key 也能跑通」。但只要 `examiner_llm`
        在配置里出现过（界面保存配置就会写这一段），它的 `api_key` 就是空串，于是
        「未单独配置」被当成「没配 Key」，门禁永远挂着「请配置真实 API」——
        用户明明已经填了 Key。回落到同一个端点上的 Key，才让那句承诺真的成立。
        """
        if self.examiner.api_key:
            return self.examiner.api_key
        if self.examiner.provider != "mock" and self.same_endpoint():
            return self.llm.api_key
        return ""

    def examiner_ready(self) -> bool:
        """提问者 / 分析者是否已可真实调用（含「同端点共用一把 Key」）。"""
        return self.examiner.provider != "mock" and bool(self.examiner_key())

    def examiner_effective(self) -> LLMSettings:
        """真正交给提问者 / 分析者的设置：把回落到的 Key 填进去。

        判定说「可用」而调用时又说「没配 Key」是最难查的一类不一致 ——
        两处必须读同一个 Key，因此 readiness 与 provider 构造都走这里。
        """
        key = self.examiner_key()
        if key and key != self.examiner.api_key:
            return replace(self.examiner, api_key=key)
        return self.examiner

    def is_configured(self) -> bool:
        """界面门禁用：大脑与提问者都可用才算配置完成（离线 mock 需用户显式选择）。"""
        return self.brain_ready() and self.examiner_ready()

    # ---------- 派生路径 ----------

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def ontology_dir(self) -> Path:
        return self.data_dir / "ontology"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pacs.db"

    @property
    def config_path(self) -> Path:
        return self.base_dir / "config.json"

    # ---------- 加载 / 保存 ----------

    @classmethod
    def load(cls, path: Path | str | None = None, base_dir: Path | str | None = None) -> "Settings":
        """从 config.json + 环境变量构建设置。文件不存在时用默认值（mock，离线可跑）。"""
        root = Path(base_dir) if base_dir else PROJECT_ROOT
        cfg_path = Path(path) if path else root / "config.json"
        raw: dict = {}
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))

        llm_raw = dict(raw.get("llm") or {})
        exam_raw = dict(raw.get("examiner_llm") or {})
        run_raw = dict(raw.get("run") or {})
        # 心智年龄已固定为 DEFAULT_AGE_MONTHS：配置文件里的 observation/battery.age_months
        # （以及任何其他月龄来源）一律**忽略**，不再参与任何计算。

        # 环境变量覆盖（BYOK 推荐方式：不把 Key 写进文件）
        if os.environ.get(ENV_PROVIDER):
            llm_raw["provider"] = os.environ[ENV_PROVIDER]
        if os.environ.get(ENV_API_KEY):
            llm_raw["api_key"] = os.environ[ENV_API_KEY]
        if os.environ.get(ENV_MODEL):
            llm_raw["model"] = os.environ[ENV_MODEL]
        if os.environ.get(ENV_BASE_URL):
            llm_raw["base_url"] = os.environ[ENV_BASE_URL]

        if os.environ.get(ENV_EXAMINER_PROVIDER):
            exam_raw["provider"] = os.environ[ENV_EXAMINER_PROVIDER]
        if os.environ.get(ENV_EXAMINER_API_KEY):
            exam_raw["api_key"] = os.environ[ENV_EXAMINER_API_KEY]
        if os.environ.get(ENV_EXAMINER_MODEL):
            exam_raw["model"] = os.environ[ENV_EXAMINER_MODEL]
        if os.environ.get(ENV_EXAMINER_BASE_URL):
            exam_raw["base_url"] = os.environ[ENV_EXAMINER_BASE_URL]

        # 只保留 dataclass 认识的键
        known_llm = {k: v for k, v in llm_raw.items() if k in LLMSettings.__dataclass_fields__}
        known_exam = {k: v for k, v in exam_raw.items() if k in LLMSettings.__dataclass_fields__}
        known_run = {k: v for k, v in run_raw.items() if k in RunSettings.__dataclass_fields__}

        brain_llm = LLMSettings(**known_llm)
        # 提问者/分析者：以模拟大脑为底再叠加自己的覆盖项 —— 未单独配置时即跟随模拟大脑，
        # 避免「只填了一套 Key，提问者却静默退化成 mock」这种难以察觉的降级。
        examiner_llm = replace(brain_llm, **known_exam)

        return cls(
            llm=brain_llm,
            examiner=examiner_llm,
            run=RunSettings(**known_run),
            base_dir=root,
        )

    def save(self, path: Path | str | None = None, include_key: bool = False) -> Path:
        """写出 config.json。默认**不写入 API Key**（避免密钥落盘）。"""
        cfg_path = Path(path) if path else self.config_path
        llm = asdict(self.llm)
        examiner = asdict(self.examiner)
        if not include_key:
            llm["api_key"] = ""
            examiner["api_key"] = ""
        payload = {
            "llm": llm,
            "examiner_llm": examiner,
            "run": asdict(self.run),
            # 不再写入 age_months：心智年龄是固定常量（DEFAULT_AGE_MONTHS），不可配置
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return cfg_path

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.runs_dir, self.exports_dir, self.ontology_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- 展示 ----------

    def describe(self) -> str:
        key = "已配置" if self.llm.api_key else "未配置"
        examiner_key = "已配置" if self.examiner.api_key else "未配置"
        return (
            f"provider={self.llm.provider}  model={self.llm.model}  "
            f"base_url={self.llm.base_url}  api_key={key}\n"
            f"examiner: provider={self.examiner.provider}  model={self.examiner.model}  "
            f"api_key={examiner_key}\n"
            f"base_dir={self.base_dir}"
        )
