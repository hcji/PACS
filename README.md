# PACS：可扰动人工认知系统

**Perturbable Artificial Cognitive System（PACS）** 用提示词把大语言模型实例化为一个**人工心智**，对它的基础认知计算过程施加选择性扰动，观察复杂行为表型如何涌现，并研究孤独症样的产生机制、计算分型与特征差异。

```
认知基础过程 → 选择性扰动 → 表型涌现 → 机制拟合 → 可证伪预测 → 人类验证
```

本仓库是论文配套的代码、对话数据集、结果表与图件发布版，包含：

- 能独立运行的智能体本体（`pacs/`）；
- 五个实验批次的逐被试对话数据（压缩包 `pacs/data.7z`，解压后为 `pacs/data/runs/`）；
- 四张结果表、四张主图及其指标清单（`results/`）；
- 从对话数据重算表格与图件的全部脚本（`results/scripts/`）。

> **用途声明**：本项目是方法学研究材料。所有被试都是 LLM 实例化的**虚拟心智**，不是真实儿童数据；不得用于医学诊断，也不替代专业评估。

---

## 目录

- [1. 系统概览](#1-系统概览)
- [2. 仓库结构](#2-仓库结构)
- [3. 三分钟上手](#3-三分钟上手)
- [4. 接入真实模型](#4-接入真实模型)
- [5. 对话数据集](#5-对话数据集)
- [6. 结果表与图件](#6-结果表与图件)
- [7. 复现表格与图件](#7-复现表格与图件)
- [8. 文档索引](#8-文档索引)
- [9. 测试](#9-测试)
- [10. 局限与使用边界](#10-局限与使用边界)
- [11. 许可与引用](#11-许可与引用)

---

## 1. 系统概览

### 1.1 研究逻辑

真实孤独症研究里，底层认知机制无法被直接操纵，研究者只能从症状组合反推机制，而症状与机制是多对多的。PACS 把方向倒过来：**先构造机制，再看表型**。

系统把心智拆成三层：

```
Level 0   31 个可独立扰动的基础认知过程（8 个家族），完整度向量 θ = (P₁ … P₃₁)
             │  选择性扰动：1.0 = 完整
             ▼
Level 1   高阶认知（心智理论、认知灵活性、语用解释）—— 由基础过程组合涌现，不直接扰动
             │
             ▼
Level 2   行为表型：6 个表型维度 + 1 个对照通道
```

31 个基础过程的操作性定义、理论依据与计算角色见 `pacs/src/pacs/ontology/primitives.py`；6 个表型维度与对照通道见 `pacs/src/pacs/phenotype/dims.py`。

### 1.2 三方角色与盲法

| 角色 | 职责 | 能看到 | 看不到 |
|---|---|---|---|
| **模拟大脑** | 被 θ 扰动的人工心智 | 自己的完整度参数、提问者的逐轮问题 | 提问者与分析者的存在及判断 |
| **提问者** | 另一个 LLM，设计情境并提问 | 观察目标、参考条目库、逐轮对话 | θ、primitive 名称、完整度数值 |
| **分析者** | 另一个 LLM，阅读整段对话并逐维评分 | 对话全文、评分锚点、证据覆盖情况 | θ、内部构造 |

θ 只进入模拟大脑的提示词。提问者与分析者的输入里没有 θ，`analyze_dialogue()` 的函数签名里也没有 θ 参数——读数在代码层面不可能被扰动参数直接影响。

### 1.3 测量协议

每次观察都按同一套口径执行：

| 口径 | 取值 |
|---|---|
| 心智年龄 | 固定 **36 个月**，无配置入口 |
| 基础轮数 | **35 轮** = 7 个计分通道 × 每通道 5 个问题 |
| 补问 | 基础轮后仍不足配额的通道，逐轮写入提问者的当轮指令继续补问 |
| 轮数上限 | 默认 **100**（硬顶 500） |
| 问题/回答字数 | 默认 **≤100 字**；超出只告警，不截断 |
| 分析者输出 | 逐维读数（0–100，100 = 与同龄典型无异）+ 约 350 字病例报告式结论 |

一轮对话 = 2 次模型调用（提问 + 作答），末尾加 1 次分析者调用。基础 35 轮约 71 次调用/被试；补问至 100 轮最多约 201 次。

### 1.4 构造效度约束

- **不存在预设的「机制 → 测量维度」对应表**。哪个维度受影响由数据给出：取 6 个表型维度里下降最多者为主导维度，再看效应集中度与对照通道是否稳定。
- **没有证据 ≠ 表现典型**。对话没有探查到的通道记空读数，并在 `uncovered_channels` 中列出，不填典型值。
- **合成数据不得冒充涌现**。离线 mock 剖面标 `provider_kind="mock"` / `synthetic=true`，报告带免责说明。
- 这些约束由测试套件中的 AST 扫描、提示词盲法检查与合成数据标记守卫。

---

## 2. 仓库结构

```
PACS-publish/
├── README.md                        本文
├── LICENSE                          MIT
├── environment.yml                  conda 环境（表格与图件）
├── requirements.txt                 pip 依赖（表格与图件）
│
├── pacs/                            智能体本体与实验驱动
│   ├── src/pacs/                    核心包（运行时零第三方依赖）
│   ├── tests/                       399 项测试
│   ├── tools/                       实验驱动与条目库生成器
│   ├── data.7z                      五个批次的逐被试对话数据（解压为 data/runs/）
│   ├── pyproject.toml
│   ├── config.example.json          配置模板
│   ├── README.md                    智能体技术文档
│   └── tools/可靠性实验.md            Fig.1 可靠性实验设计与指标
│
└── results/                         结果表、图件与复现脚本
    ├── scripts/                     9 个脚本
    ├── data/                        四张结果表、候选关联表、对话个案集
    └── figure/                      Fig.1–4 的 png/pdf、metrics 与图注说明
```

发布包约 **31 MB**，其中 **21 MB** 是 `pacs/data.7z` 对话数据压缩包；解压后为 `pacs/data/runs/`（约 108 MB，未纳入版本库）。

---

## 3. 三分钟上手

智能体本体只需要 Python ≥ 3.10，运行时**零第三方依赖**。用离线 mock 心智即可跑通全链路：

```bash
cd pacs
set PYTHONPATH=src            # Windows cmd
                              # PowerShell: $env:PYTHONPATH="src"
                              # Linux / macOS: export PYTHONPATH=src

python -m pacs.cli --version          # pacs 0.1.0
python -m pacs.cli ontology           # 31 个基础过程与探查登记自检
python -m pacs.cli demo               # 一键离线跑通全链路（Mock，无需 API Key）
python -m pacs.cli serve              # 打开可视化界面（默认 127.0.0.1）
```

跑测试套件：

```bash
cd pacs
set PYTHONPATH=src
python -m pytest tests
```

当前测试规模：**399 项**（398 通过、1 项在缺少外部 AsdKB 源数据时自动跳过）。

---

## 4. 接入真实模型

复制配置模板，填写 provider、model、base_url 与 api_key：

```bash
cd pacs
copy config.example.json config.json       # Windows
# cp config.example.json config.json       # Linux / macOS
```

也可以用环境变量覆盖，不落盘：

| 变量 | 作用 |
|---|---|
| `PACS_LLM_PROVIDER` / `PACS_LLM_API_KEY` / `PACS_LLM_MODEL` / `PACS_LLM_BASE_URL` | 模拟大脑 |
| `PACS_EXAMINER_LLM_*` | 提问者 / 分析者（未配置时跟随模拟大脑） |

任何 OpenAI 兼容端点均可使用（DeepSeek、OpenAI、本地 vLLM / Ollama 等）。

```bash
# 初始化一具完整心智，查看自我报告
python -m pacs.cli init-brain --provider openai --model gpt-4o-mini

# 查看某一扰动下的心智提示词
python -m pacs.cli prompt --primitive social_reward_weighting --dose 0.2

# 预览设计规模与调用数（不调用模型）
python -m pacs.cli sample --mode all --limit 30

# 小规模真实运行
python -m pacs.cli run --provider openai --model gpt-4o-mini --mode single --limit 3
```

**不要**把带 API Key 的 `config.json` 提交到版本库。

---

## 5. 对话数据集

对话数据以 `pacs/data.7z` 发布。解压后得到 `pacs/data/runs/`，每个批次包含：

```
<run>/
├── cells/<condition>/subjects/<index>_theta_<id>_r<rep>.json   逐被试：参数 + 逐轮对话 + 读数 + 剖面
├── flat/all_subjects.csv                                       一行一被试的宽表
├── analysis/                                                   批次聚合结果
├── manifest.json                                               运行参数、模型、协议、成本
└── summary.md                                                  人读摘要
```

| 批次 | 设计 | 被试数 | 用途 |
|---|---|---|---|
| `fig1_main` | 7 个条件 × 4 档 θ，重复 3–5 次 | 116 | `result_1`、Fig.1 |
| `fig1_realcheck` | 同 θ 的补充参照臂（5 份完整心智 + 5 份扰动） | 10 | 对话证据分析的行为参照 |
| `scale_single` | 31 个 primitive × 2 档 × 3 重复 + 3 个通用退化对照 | 189 | `result_2`、Fig.2、Fig.3 |
| `scale_pair` | C(31,2) = 465 对 × 3 重复 | 1395 | `result_3`、Fig.4 |
| `scale_single_dose_merged_5dose_5rep` | 5 个 primitive × 5 档 × 5 重复 | 125 | `result_4`、Fig.2 面板 F、Fig.3 Q3 |

单个被试 JSON 的五个区块：

| 区块 | 内容 |
|---|---|
| `parameters` | θ、通用退化、设计、来源、θ 指纹 |
| `run_parameters` | 轮数、上限、字数上限、种子、探查方式 |
| `dialogue` | 会话元信息与**逐轮** `{turn, role, content, focus}` |
| `analysis` | 7 通道读数、逐维细节、分析者结论、表型印象 |
| `profile` | 表型剖面、provider 类型、`synthetic` 标记、未覆盖通道 |

解压命令（在 `pacs/` 目录下）：

```bash
7z x data.7z
```

Windows 也可以右键 `data.7z` 选择「解压到当前位置」。解压后各批次的 `flat/数据字典.md` 给出逐字段说明。

---

## 6. 结果表与图件

### 6.1 结果表

| 文件 | 内容 | 来源批次 |
|---|---|---|
| `results/data/result_1_robustness.xlsx` | 116 名被试 × 16 列；四轴可靠性实验 | `fig1_main` |
| `results/data/result_2_univariate.xlsx` | 189 名被试 × 13 列；单维度扰动批 | `scale_single` |
| `results/data/result_3_multivariable.xlsx` | 1395 名被试；两两组合批 | `scale_pair` |
| `results/data/result_4_does_response.xlsx` | 125 名被试；五档剂量阶梯 | `scale_single_dose_merged_5dose_5rep` |

每张工作簿的第一张表 `by_subject` 为一行一被试；后续表为逐格聚合、剂量-反应与特异性判定。

### 6.2 图件

| 图 | 文件 | 内容 |
|---|---|---|
| **Fig.1** | `results/figure/figure1_pacs_reliability.{png,pdf}` | 系统可靠性：同一脑重测、换模型、换提问者/分析者/题序、换轮次 |
| **Fig.2** | `results/figure/figure2_univariate_phenotypes.{png,pdf}` | 单维度扰动批的表型热图、曼哈顿图、二部网络、剂量-反应 |
| **Fig.3** | `results/figure/figure3_dialogue_evidence.{png,pdf}` | 对话证据：受损 vs 完整心智、同维度不同机制、同机制不同剂量 |
| **Fig.4** | `results/figure/figure4_interaction_stability.{png,pdf}` | 组合效应的稳定性：协同、缓冲、新表型与重复一致性 |

每张图旁边有 `figure*_metrics.json`：图中每个数字的完整清单、面板口径与复现所需字段。

对话个案集的完整版在 `results/data/dialogue_evidence_casebook.{md,xlsx}`；候选关联清单在 `results/data/figure2_link_candidates.{csv,md}`。

---

## 7. 复现表格与图件

环境：

```bash
conda env create -f environment.yml && conda activate pacs
# 或者：python -m pip install -r requirements.txt
```

`results/scripts/` 下的脚本按文件位置解析项目根，并自动把 `pacs/src`、`pacs/tools` 加入 `sys.path`，**不需要设置 `PYTHONPATH`**。

### 7.1 结果表

先解压对话数据（只需一次）：

```bash
cd pacs
7z x data.7z
cd ..
```

```bash
cd results/scripts

# result_2（完整 16 列导出）与 result_3
python export_result_tables.py

# result_4（五档剂量阶梯）
python export_dose_supplement.py \
  --merged-run pacs/data/runs/scale_single_dose_merged_5dose_5rep \
  --doses 0.2,0.4,0.6,0.8,0.9 --replicates 5 \
  --by-subject-layout result2 --no-ladder --freeze-panes A44 \
  --xlsx-name result_4_does_response.xlsx --out-dir ../data
```

`result_2_univariate.xlsx` 的发布版是 **13 列**（省略三个模型列）；`export_result_tables.py` 默认导出 **16 列**，两者行序与全部读数相同。`result_1_robustness.xlsx` 由 Fig.1 上游链生成，随包提供，不需要重跑。

### 7.2 图件

按以下顺序运行（后面的脚本依赖前面的产物）：

```bash
cd results/scripts

# Fig.1：可靠性 → figure1_* + figure1_metrics.json
python fig1_visualize.py

# Fig.2：单维度表型 → figure2_* + figure2_metrics.json
python fig2_visualize.py

# Fig.2 的候选关联清单 → figure2_link_candidates.{csv,md}
python fig2_link_candidates.py

# 对话证据个案分析 → dialogue_evidence_casebook.{md,xlsx} + dialogue_evidence_metrics.json
python analyze_dialogue_evidence.py

# Fig.3：对话证据 → figure3_* + figure3_metrics.json
python fig3_dialogue_evidence.py

# Fig.4：组合效应稳定性 → figure4_* + figure4_metrics.json
python fig4_interaction_stability.py
```

所有脚本在落盘前自带一致性检查（例如 Fig.2 的 62 个判定逐行对账、Fig.3 的 186 个 p 值重算、Fig.4 的 10 条链式硬门）；检查不过就拒绝出图。

---

## 8. 文档索引

| 文档 | 内容 |
|---|---|
| [`pacs/README.md`](pacs/README.md) | 智能体技术文档：安装、CLI、HTTP 路由、配置、存储、测试 |
| [`pacs/tools/可靠性实验.md`](pacs/tools/可靠性实验.md) | Fig.1 可靠性实验的四条轴、统计指标与报告口径 |

---

## 9. 测试

```bash
cd pacs
set PYTHONPATH=src
python -m pytest tests
```

测试覆盖：构造效度守卫（禁止预设映射）、提示词盲法、轮次配额与字数上限、分析者输出解析、冻结协议、HTTP API、Web 界面、条目库生成链。当前 **399 项**，398 通过、1 项在缺少外部 AsdKB 源数据时跳过。

---

## 10. 局限与使用边界

1. **全部被试是模拟心智**。任何关于孤独症机制的结论都必须来自真实模型的运行；mock 路径只验证管线与统计逻辑。
2. **prompt 级扰动的稳定性是核心未验证假设**。若某个基础过程在真实 LLM 上不能产生稳定、特异的效应，应按预注册标准剔除。
3. **表型粒度受分析者稳定性限制**。逐维读数来自 LLM 对一段对话的解读，同一次会话内各维相关、跨会话才独立；提高分辨力应增加独立重复，而不是增加题目。
4. **本研究不做诊断**。机制拟合回答的是「这具心智更符合哪种计算机制」，不是 `LLM score → ASD/TD`。
5. **人类验证不在本仓库范围**。本仓库只提供可证伪的、已冻结的预测与配套代码。
6. **条目库的逐题映射由本项目派生**。AsdKB 源数据的逐题症状关联实为量表级覆盖，因此维度/症状/标准关联按显式规则派生，并逐条标注依据。

> 仿真提供先验与机制假设，不提供真理；真实人类数据负责逐步校正先验。

---

## 11. 许可与引用

代码与数据以 [MIT](LICENSE) 许可发布。

如果使用本仓库，请引用：

```bibtex
@software{pacs2026,
  title  = {PACS: Perturbable Artificial Cognitive System},
  year   = {2026},
  note   = {Research code and data release},
  url    = {https://github.com/hcji/PACS}
}
```

> **用途声明**：本项目为方法学研究材料，全部被试为模拟产物，**不用于医学诊断**、不替代专业评估。
