# Banking77-FinIntent-SupCon：开发规划

## 1. 规划原则

本计划依据老师的建议收缩项目范围：先把 TF-IDF + SVM、BERT 和 SetFit 基线做扎实，再依次验证 SupCon 与 hard negative mining。所有结论以 few-shot 实验、Macro-F1、per-class F1 和 confusion matrix 为依据，可视化只作辅助。

当前阶段不纳入开放集识别、主动学习、推理服务、Docker、监控或更多大模型基线。只有核心实验完成且结论稳定后，才评估是否扩展。

## 2. 当前基线

截至当前版本：

- BANKING77 原始训练集、测试集和类别文件已在仓库中；
- EDA、固定训练/验证划分、重复文本泄漏检查及划分清单已经完成；
- 项目研究问题、方法边界和实验口径已确定；
- TF-IDF + SVM、BERT、SetFit 三组 baseline 及统一训练产物已经实现，三种方法的四种预算均已完成 `seed=42` Version 1 运行；SetFit 在 few-shot 下形成有效基线，但 BERT 少样本实验发生预测塌缩，需要修正训练预算；
- 旧文档中的历史单次结果缺少当前仓库内的代码与产物支撑，不纳入最终结论。

Version 1 的详细结果、问题与修正方案见 [FIRST_EXPERIMENT_REPORT.md](FIRST_EXPERIMENT_REPORT.md)。

因此下一步应先修复少样本 BERT、SetFit 训练历史与运行元数据，再完成多随机种子基线；基线通过可信度检查后再进入 SupCon 阶段。

## 3. 目标目录与产物契约

计划形成以下结构：

```text
.
├── data/banking_data/               # BANKING77 原始数据
├── docs/
│   ├── PROJECT.md                    # 项目介绍
│   └── DEVELOPMENT_PLAN.md           # 开发规划
├── src/
│   ├── config.py                     # 路径、设置、种子与公共超参数
│   ├── data/prepare_data.py          # train/val 划分与 few-shot 抽样
│   ├── baselines/                    # TF-IDF、BERT、SetFit
│   ├── supcon/                       # 模型、损失、采样与训练
│   ├── analysis/                     # 指标汇总、混淆分析与出图
│   └── run_experiments.py            # 统一实验入口
├── tests/                            # 数据、损失与最小运行测试
├── outputs/runs/                     # 每次实验的小型可核验产物
├── requirements.txt
└── README.md
```

每个 `(method, setting, seed)` 运行至少保存：

```text
outputs/runs/<method>/<setting>/seed<seed>/
├── config.json          # 数据、模型和训练参数
├── metrics.json         # Macro-F1 及逐类指标
├── predictions.csv      # y_true、y_pred、置信度
└── env.json             # Python、依赖、设备与版本信息
```

checkpoint 和 embedding 可以本地保存但不提交；小型指标与预测文件应保留，便于复核最终表格。

## 4. 实验协议

### 4.1 数据划分

1. 从官方 10,003 条训练数据中按类别分层划分训练池与固定验证集；
2. 官方 3,080 条测试数据保持冻结，只在模型方案确定后评估；
3. 5/10/20-shot 样本只从训练池抽取；
4. 固定验证集划分，few-shot 样本抽取随随机种子变化；
5. 检查并披露官方划分中已有的重复文本，确保新划分不引入额外泄漏，并记录所有样本索引或哈希。

建议主实验使用 3 个随机种子；若计算资源允许，再扩展到 5 个。不得只汇报最优 seed。

### 4.2 公平对照

BERT、SupCon 和 SupCon + hard negatives 之间统一：

- backbone 与 tokenizer；
- 最大序列长度；
- train/val/test 划分；
- 随机种子集合；
- 分类头与 checkpoint 选择指标；
- 训练步数或等价计算预算；
- 评估和结果保存代码。

超参数只能根据验证集确定，不得根据测试集反复调整。

### 4.3 结果口径

主结果表固定为：

| Method | full | 5-shot | 10-shot | 20-shot |
|---|---:|---:|---:|---:|
| TF-IDF + Linear SVM | mean ± std | mean ± std | mean ± std | mean ± std |
| BERT | mean ± std | mean ± std | mean ± std | mean ± std |
| SetFit | mean ± std | mean ± std | mean ± std | mean ± std |
| BERT + SupCon | mean ± std | mean ± std | mean ± std | mean ± std |
| BERT + SupCon + HN | mean ± std | mean ± std | mean ± std | mean ± std |

除 Macro-F1 外，每次运行必须保留 per-class F1 与完整预测，供混淆分析使用。

## 5. 开发阶段

### 阶段 0：建立可复现实验骨架

任务：

- 新建并固定依赖清单；
- 实现统一配置、随机种子和设备选择；
- 实现分层 train/val 划分与 full、5/10/20-shot 数据集；
- 实现 Macro-F1、per-class P/R/F1 和 confusion matrix；
- 实现统一运行目录、配置记录和断点续跑；
- 增加数据划分、标签映射和指标计算测试。

验收标准：

- 同一 seed 重复准备数据时样本完全一致；
- train 与 val 无重叠，官方 train/test 的已有重复已记录，77 类标签映射一致；
- 一个虚拟模型可以走通“训练—预测—保存—汇总”全流程；
- 测试集未参与 checkpoint 或超参数选择。

### 阶段 1：完成三组基线

按计算成本从低到高实现：

1. TF-IDF（word 1–2 gram）+ Linear SVM；
2. BERT 全参数微调，以验证集 Macro-F1 选择 checkpoint；
3. SetFit 对比学习 + 分类头。

每种方法先在 5-shot 上做 smoke test，再运行四个数据设置和全部 seeds。记录训练耗时，并检查异常结果，例如 full 明显差于少样本或不同 seed 波动过大。

验收标准：三种方法的所有设置均产生统一格式的配置、指标和预测文件，主结果表不再有空格子。

### 阶段 2：实现标准 SupCon

任务：

- 实现数值稳定的监督对比损失；
- 实现类平衡 batch，确保每个 anchor 至少有一个正样本；
- 采用“两阶段训练”：先学习对比表示，再冻结 encoder 训练线性分类头；
- 对比 BERT baseline 与 BERT + SupCon；
- 为损失退化、梯度有限值和 batch 正样本条件增加测试。

验收标准：四个数据设置、全部 seeds 均完成，且 SupCon 与 BERT 的差异能够由统一产物复算。

### 阶段 3：加入 hard negative mining

第一版只实现 batch 内相似度加权：异类样本与 anchor 越相似，负样本权重越高。实现需满足：

- hard-negative 权重关闭或趋于均匀时退化为标准 SupCon；
- 使用 log-sum-exp 等稳定计算，避免低温度下溢出；
- 记录 hard-negative 温度等关键配置；
- 不在本阶段同时加入动态混淆图等额外机制。

验收标准：完成 `BERT + SupCon + HN` 的完整实验矩阵，并能隔离 HN 相对标准 SupCon 的增量。

### 阶段 4：消融与混淆分析

围绕老师提出的问题分析结果：

1. 比较 baseline、SupCon、SupCon + HN 的 Macro-F1；
2. 计算每类 F1 的变化，列出提升最大和下降最大的类别；
3. 从 baseline confusion matrix 中选出高频混淆 intent pairs；
4. 比较三组方法在这些类别对上的双向误判数；
5. 检查整体提升是否只来自少数容易类别；
6. 如有 embedding，再补充 t-SNE；不以视觉分离代替指标。

验收标准：至少给出一张主结果表、一张高混淆类别对表和一组 confusion matrix，并能回答“提升是否集中在易混淆类别”。

### 阶段 5：形成最终报告

报告按“问题—方法—实验协议—结果—误差分析—局限”组织。只写入可由仓库产物复核的数字，并明确报告负结果和异常结果。

完成标准：

- 所有核心实验无缺失格子；
- 结果为多 seed 的 mean ± std；
- SupCon 和 HN 的贡献分别有消融证据；
- 结论同时覆盖总体表现与易混淆 intent pairs；
- README 的命令、结果和文档链接与仓库一致。

## 6. 执行顺序与判定门

```text
实验骨架
   ↓
TF-IDF / BERT / SetFit 基线
   ↓  基线可信且产物齐全
标准 SupCon
   ↓  能隔离 SupCon 增量
SupCon + hard negatives
   ↓  能隔离 HN 增量
per-class F1 + confusion matrix + 报告
```

- **Gate 1：基线可信。** 若 full 结果异常或 seed 波动过大，先排查数据、标签、收敛和 checkpoint，不继续堆方法。
- **Gate 2：SupCon 可解释。** 若整体提升不足，检查收益是否稳定集中在 few-shot 或高混淆类别；不得只挑有利类别。
- **Gate 3：HN 有独立贡献。** 若 HN 无显著增益，如实报告并分析 false negatives、batch 构造和温度敏感性，不继续增加复杂模块掩盖结果。

## 7. 暂不开展的扩展

以下内容保留为核心实验完成后的候选方向，不属于当前里程碑：

- confusion-aware 动态难负例；
- 开放集或 OOS 意图识别；
- 主动学习与标注成本优化；
- 更多 backbone、LoRA 或大模型基线；
- API、Docker、在线监控与交互式 Demo。

扩展的前提是主实验已经完整、可复现，并且新增方向服务于明确的研究问题。
