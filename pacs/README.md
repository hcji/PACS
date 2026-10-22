# PACS 智能体：技术文档

**Perturbable Artificial Cognitive System（PACS）** 把 LLM 实例化为一个可操纵的人工心智：用 31 个基础认知过程的完整度向量 θ 构造提示词，通过提问者与分析者两个旁路 LLM 读出 6 个表型维度与 1 个对照通道。

本文是智能体的技术文档：安装、配置、CLI、HTTP 接口、观察协议、存储与测试。科学设计与本体定义见 `src/pacs/ontology/` 与仓库根 README。

---

## 1. 运行要求

| 层 | 要求 |
|---|---|
| 智能体本体 | Python ≥ 3.10，**运行时零第三方依赖**（HTTP 用标准库 `urllib`，数据模型用 dataclass） |
| 测试 | `pytest` |
| 结果表 / 图件 | `openpyxl`、`numpy`、`pandas`、`scipy`、`matplotlib`（见仓库根目录 `requirements.txt`） |

Web 界面由标准库 `http.server` 提供，不需要 FastAPI。`pyproject.toml` 中的 `api` 可选依赖仅用于外部 ASGI 形态。

```bash
# 在仓库根目录
conda env create -f environment.yml && conda activate pacs
# 或者：python -m pip install -r requirements.txt
```

---

## 2. 安装与运行

智能体以 src-layout 组织，未安装时把 `src` 加入 `PYTHONPATH` 即可：

```bash
cd pacs
set PYTHONPATH=src            # Windows cmd
                              # PowerShell: $env:PYTHONPATH="src"
                              # Linux / macOS: export PYTHONPATH=src

python -m pacs.cli --version
```

也可以安装为包：

```bash
cd pacs
python -m pip install -e .
pacs --version
```

---

## 3. 快速开始

### 3.1 离线体验

```bash
python -m pacs.cli demo
```

`demo` 用规则式 mock 心智跑通：扰动设计 → 逐被试对话 → 分析者读数 → 机制分析 → Atlas → 冻结快照。全程不需要 API Key，也不访问网络。

> Mock 是确定性前向模型，验证的是管线与统计逻辑，不是 LLM 的涌现行为。mock 剖面会被标为 `provider_kind="mock"` / `synthetic=true`。

### 3.2 查看本体与提示词

```bash
python -m pacs.cli ontology
python -m pacs.cli vocab
python -m pacs.cli prompt --primitive social_reward_weighting --dose 0.2
```

`ontology` 打印 31 个基础过程、Level 1 组合与每个过程的探查登记，并做自检；`vocab` 打印 7 个计分通道与观察锚点（不做机制→维度映射）；`prompt` 打印某一扰动下实际发送给模拟大脑的系统提示词。

### 3.3 真实模型

```bash
# 推荐用环境变量传 Key，不落盘
set PACS_LLM_API_KEY=<your-api-key>
set PACS_LLM_PROVIDER=openai

python -m pacs.cli init-brain --provider openai --model gpt-4o-mini
python -m pacs.cli sample --mode all --limit 30
python -m pacs.cli run --provider openai --model gpt-4o-mini --mode all --limit 30 --yes
```

批量运行在调用数上界超过 200 时要求 `--yes`，防止误花额度。

---

## 4. 配置

### 4.1 配置文件

复制 `config.example.json` 为 `config.json`，填写模拟大脑与提问者/分析者两套端点：

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "api_key": "<your-api-key>",
    "temperature": 0.3,
    "max_tokens": 8000,
    "timeout": 60.0
  },
  "examiner_llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "api_key": "<your-api-key>",
    "temperature": 0.3,
    "max_tokens": 8000,
    "timeout": 60.0
  }
}
```

`examiner_llm` 未填写时，提问者/分析者继承 `llm`（同厂商、同 base_url 时只配一套 Key 即可）。要接到另一家端点，必须单独填写它的 Key。

### 4.2 环境变量

| 变量 | 作用 |
|---|---|
| `PACS_LLM_PROVIDER` / `PACS_LLM_API_KEY` / `PACS_LLM_MODEL` / `PACS_LLM_BASE_URL` | 模拟大脑 |
| `PACS_EXAMINER_LLM_PROVIDER` / `PACS_EXAMINER_LLM_API_KEY` / `PACS_EXAMINER_LLM_MODEL` / `PACS_EXAMINER_LLM_BASE_URL` | 提问者 / 分析者 |

**API Key 不落盘**：界面保存配置默认只写非敏感项；需要重启后仍然有效，勾选「写入 config.json」或使用环境变量。

### 4.3 输出额度

`max_tokens` 对模拟大脑与提问者生效；分析者另有下限 `ANALYST_MIN_TOKENS = 8000`，实际额度取 `max(配置值, 8000)`。推理型模型会把额度花在思维链上，建议模拟大脑/提问者给 8000，分析者给 32768。输出被截断时，解析器会按括号栈补齐并标注「读数可能不完整」，不会把空串当成被试回答。

---

## 5. CLI

```text
pacs {ontology,vocab,prompt,init-brain,sample,runs,run,analyze,report,freeze,serve,brains,bank,dialogue,demo}
```

| 命令 | 作用 |
|---|---|
| `ontology` | 打印 31 个 primitive、Level 1 组合、每个 primitive 的探查登记，并做自检 |
| `vocab` | 打印计分通道、观察锚点与探查登记 |
| `prompt` | 打印某一扰动下的模拟大脑提示词（可审计） |
| `init-brain` | 提示词初始化人工心智并查看自我报告 |
| `sample` | 预览设计规模与预计模型调用数（不调用模型） |
| `run` | 跑扰动设计 → 观察每位被试并存库 |
| `dialogue` | 单次对话探针：提问者 → 模拟大脑 → 分析者，并打印探查结论 |
| `analyze` | 机制 / 分型 / 特征差异 / 反事实分析 |
| `report` | 生成 Atlas 报告（Markdown + JSON） |
| `freeze` | 生成冻结快照（五类对象 + 摘要 id） |
| `runs` | 列出本地运行记录 |
| `brains` | 列出大脑库（保存 / 切换 / 重命名 / 删除） |
| `bank` | 浏览观察者内置条目库（AsdKB 派生，441 条） |
| `serve` | 启动本地可视化界面 |
| `demo` | 一键离线跑通全链路 |

常用参数：

| 参数 | 说明 |
|---|---|
| `--turns` | **基础**轮数（默认 35） |
| `--max-turns` | 补问上限（默认 100，硬顶 500） |
| `--text-limit` | 问题/回答字数上限（默认 100） |
| `--mode single\|pair\|lhs\|intact\|generic\|all` | 扰动设计模式 |
| `--provider mock\|openai` | 模型提供方 |
| `--limit N` | 被试数上限 |
| `--yes` | 确认真实批量调用 |

---

## 6. 可视化界面与 HTTP 接口

```bash
python -m pacs.cli serve                 # 默认 127.0.0.1，mock 模式可直接进入
python -m pacs.cli serve --port 8080
```

界面有三个工作区：

1. **模型设置**：模拟大脑与提问者/分析者两套 API，以及大脑库；
2. **扰动注入**：系统生成或手动调整 θ，查看提示词构造过程，注入并重新初始化；
3. **表型观察**：当前大脑的读数雷达图、历史对话、用户直接提问；提问者/分析者的过程默认收起。

主要 HTTP 路由：

| 路由 | 作用 |
|---|---|
| `GET /api/boot` | 启动状态、当前配置、当前大脑 |
| `GET/POST /api/config` | 读取 / 保存模型配置 |
| `GET /api/brains` · `POST /api/brains/{save,activate,rename,delete}` | 大脑库 |
| `POST /api/perturb/generate` · `POST /api/inject` | 生成扰动、注入并初始化 |
| `POST /api/prompt` | 返回提示词构造步骤 |
| `POST /api/dialogue/start` · `GET /api/dialogue/sessions` · `POST /api/dialogue/analyze` | 对话观察与读数 |
| `POST /api/interview/{start,ask}` · `GET /api/interview/sessions` | 用户直接提问 |
| `GET/POST /api/itemsets` · `POST /api/itemsets/delete` | 自定义条目集 |
| `GET /api/examiner/bank` | 内置条目库 |
| `GET /api/analysis` · `GET /api/profiles` · `GET /api/runs` | 受试集合层面的分析 |
| `GET /api/freeze` · `GET /api/atlas-md` | 冻结快照与 Atlas |
| `GET /api/job` · `GET /api/download` | 后台任务状态与产物下载 |

---

## 7. 观察协议

| 口径 | 取值 | 说明 |
|---|---|---|
| 心智年龄 | 固定 36 个月 | 参照线不能随参数漂移 |
| 基础轮数 | 35 = 7 通道 × 5 问 | 每个计分通道都要有证据 |
| 计数 | 按提问者自报的 `focus` 每轮记 1 个问题 | 与实际问答一一对应 |
| 补问 | 不足配额的通道写入当轮指令 | 只要求「优先问」在真实模型上会漂 |
| 轮数上限 | 默认 100，硬顶 500 | 成本旋钮，不改变协议下限 |
| 字数上限 | 默认 100 字 | 超出只告警，不截断 |
| 分析者结论 | 约 350 字 + 结构化表型印象 | 行为学印象，不是临床诊断 |

方向约定：**1.0 = 与同龄典型发展无异，0.0 = 严重偏离**。

---

## 8. 数据与存储

| 路径 | 内容 |
|---|---|
| `data/pacs.db` | SQLite：运行、剖面、大脑库、对话、条目集、冻结快照 |
| `data.7z` | 五个批次的逐被试对话数据压缩包（解压后为 `data/runs/`） |
| `data/runs/` | 批量实验的逐被试文件（解压 `data.7z` 后生成，不纳入版本库） |
| `data/exports/atlas.{md,json}` | `pacs report` 的九章 Atlas |
| `data/exports/freeze_<id>/` | `pacs freeze` 的冻结快照 |

解压命令（在 `pacs/` 目录下）：

```bash
7z x data.7z
```

解压后各批次的 `flat/数据字典.md` 给出逐字段说明。

---

## 9. 代码结构

```
src/pacs/
├── ontology/     # 31 个 primitive、8 个家族、Level 1 组合、探查登记
├── brain/        # 认知架构提示词、θ 与扰动设计、条目库运行时
├── agents/       # 虚拟被试、对话探针、LLM 涌现测量
├── phenotype/    # 6+1 通道词表、观察锚点、剖面记录、结论口径
├── analysis/     # 特异性、剂量-反应、景观、异质性、亚型、反事实、贝叶斯
├── llm/          # provider 抽象（mock + OpenAI 兼容）、输出解析、模型注册表
├── store/        # SQLite 存储
├── api/          # 本地 HTTP 服务与任务状态
├── web/          # 零构建前端（原生 ES 模块）
├── data/         # 内置条目库 examiner_bank.json
├── engine.py     # 设计运行编排
├── report.py     # Atlas 报告
├── freeze.py     # 冻结协议
└── cli.py        # 命令行入口
```

---

## 10. 测试

```bash
cd pacs
set PYTHONPATH=src
python -m pytest tests
```

当前 **399 项**：398 通过，1 项在缺少外部 AsdKB 源数据时自动跳过。测试覆盖构造效度守卫、盲法、轮次配额、输出解析、冻结协议、HTTP API、Web 界面与条目库生成链。

---

## 11. 设计约束与局限

- **零训练**：扰动完全由提示词实现，不微调任何模型。
- **运行时零依赖**：核心包只用标准库。
- **盲法**：θ 只进模拟大脑提示词；提问者与分析者的输入不含 θ。
- **无预设映射**：不存在「机制 → 测量维度」的对应表；主导维度由数据给出。
- **合成数据标记**：mock 剖面标 `provider_kind="mock"` / `synthetic=true`，不冒充 LLM 涌现。
- **不做诊断**：输出是方法学研究材料，不用于医学诊断、不替代专业评估。

---

## 12. 相关文档

| 文档 | 内容 |
|---|---|
| [`tools/可靠性实验.md`](tools/可靠性实验.md) | Fig.1 可靠性实验的设计、指标与报告口径 |
