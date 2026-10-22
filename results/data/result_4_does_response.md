# 剂量阶梯（0.9 / 0.8 / 0.6 / 0.4 / 0.2）· 合并视图

## 这份表是什么

把 5 个档位合到同一张剂量-反应曲线与同一份逐档特异性表里的视图。
其中一部分档位是**更早批次跑好的复用数据**，本次没有重跑（逐档来源见下）。

- primitive（5 个）：agent_state_updating、belief_updating、rule_updating、self_other_distinction、social_reward_weighting
- 档位：0.9 / 0.8 / 0.6 / 0.4 / 0.2
- 重复：每（primitive × 档）5 次
- 协议：50 轮（配额不足补至 100）/ seed 42 / 题池 100 @ seed 42 / text_limit 100 / max_tokens 8000 / temperature 0.3 / 三方同为 deepseek-flash
- 完整心智基线：复用 `pacs/data/runs/fig1_main/axis1/axis1_same_brain` 的 5 份（协议 / 三方模型 / 题池逐位一致，每次开跑前都校验），不重跑
- 通用退化对照：**未纳入**（`--no-generic`）
- 心智年龄：固定 36 个月

### 来源批次

- pacs\data\runs\scale_single_dose_merged_5dose_5rep\cells

- 行数构成：`by_subject` 125 行 —— **30 行**可回指结果 2（0.4 / 0.8 两档里更早批次跑的那 3 次重复，**未重跑**），其余 **95 行**为本轮新增或重跑
- 档位覆盖：`summary_by_cell` 25 行 = 5 primitive × 5 档

### ⚠ `row_id` 的格式与结果 2/3 不同（列名与列序仍一致）

本文件的 `row_id` 是 `<条件>#<剂量>#<被试序号>_r<重复>`（如 `belief_updating#0.20#001_r0`），比结果 2/3 多一段剂量。

原因是结果 2/3 的「被试序号」是**该批次剂量表里的位次**、不是剂量本身：
不同批次的同一位次对应不同剂量。照抄旧格式会让多档合并后大量行 `row_id` 相同
（同一个 `#001_r0` 同时指数个不同剂量的测量）。一个自称是行标识的列不能重复，
所以这里嵌入了剂量。数据本身不带歧义：`mind_setting`（`belief_updating 0.20`）
与 `completeness`（`0.2`）两列都写着剂量。

## 三张表怎么读

| 表 | 一行是什么 | 用途 |
|---|---|---|
| `by_subject` | 一个被试 | 逐被试 7 维读数 + 分析者结论全文 |
| `summary_by_cell` | 一个（primitive × 档） | 逐维 mean/delta、主导维度、特异性判定、剂量-反应斜率与 r² |
| `dose_ladder` | 一个（primitive × 维度） | 各档 delta 横排 + 最重档 + 极差 + 严格递降步数，看阶梯形状 |

`delta` 一律 = 该格均值 − 完整心智基线均值；负值 = 比同龄典型发展更差。
`completeness` 就是剂量本身（该认知过程的**保留程度**）：1.0 = 完整，0.2 = 几乎不可用。

## 产物出处

| 层 | 路径 |
|---|---|
| 合并运行目录（5 档原始记录） | `D:\PACS\pacs\data\runs\scale_single_dose_merged_5dose_5rep` |
| └ 装配出处（逐份记录来源、θ 指纹） | `D:\PACS\pacs\data\runs\scale_single_dose_merged_5dose_5rep\PROVENANCE.md` |
| └ 聚合报告（逐格数值 + 结论） | `D:\PACS\pacs\data\runs\scale_single_dose_merged_5dose_5rep\analysis\scale_analysis.md` |
| 各来源批次原始运行 | 见上方「来源批次」 |
| 本表 | `D:\PACS\results\data\result_4_does_response.xlsx` |

逐被试的**完整对话过程**在运行目录的 `cells/<条件>/subjects/*.json` 的 `dialogue.turns` 里（逐轮原文），本表只带分析者结论（`analyst_summary` 列）。

## 必须一并引用的局限

1. **复用档位与本次新跑的档位不是同一批测量**：同源、协议逐项一致、题池逐位相同，但**误差结构不同源** —— 阶梯上复用档的重复间离散度不能与新跑的档位直接比。
2. **0.2 档接近「该认知过程几乎不可用」**：可能触发整体性塌陷，使特异性判定
   集体落到 `fail_control` / `fail_diffuse`。**这是剂量过高时该设计的已知行为，
   不是测量失败**，结论里必须写清而不能当成异常丢弃。
3. 每档 3 次重复是特异性自校准门槛的下限（`specificity.py` 要求 ≥3 才用重复
   估计标准误）：门槛本身会随重复噪声在档位之间摆动（实测同一 primitive 的四档
   门槛从 0.05 摆到 0.32），所以**逐档判定之间的差异主要反映门槛在动，不足以当作「某档特异、另一档不特异」的证据**。
4. 真实 provider 不接收 seed、temperature 固定 0.3：同一 θ 的重复之间测到的是
   服务端采样噪声 + 分析者给分噪声，不是「同一仪器同一样本」的复现。
5. 50 轮只比「7 通道 × 5 问 = 35」的配额多 15 轮余量，仍可能有补问（实际轮数 >50）。
6. 单因素设计**不识别交互**：本表不给任何交互/协同结论。
7. 仿真产物（虚拟被试），非真实儿童数据；仅用于方法学研究，不用于诊断。

用途声明：本工具为方法学研究材料，不用于医学诊断、不替代专业评估。
