# Banking77-FinIntent-SupCon：项目介绍

## 1. 项目定位

本项目研究金融客服场景中的细粒度意图分类问题：当多个意图的措辞非常相似、每类又只有少量标注样本时，如何仍然准确地区分用户意图。

项目使用 BANKING77 数据集，以传统机器学习和预训练语言模型为基线，重点验证监督对比学习（Supervised Contrastive Learning，SupCon）与难负例挖掘（Hard Negative Mining）能否改善易混淆意图的分类效果。

核心研究问题是：

> 在不同 few-shot 标注预算下，SupCon 与 hard negative mining 是否不仅提高整体 Macro-F1，而且确实减少语义相近意图之间的误判？

## 2. 问题背景

意图分类负责把用户查询路由到相应的业务流程。BANKING77 的 77 个标签都属于银行服务领域，类别边界往往只由少量关键词或事件状态决定，例如：

| 易混淆意图 | 关键区别 |
|---|---|
| `declined_card_payment` / `declined_transfer` | 卡支付被拒绝 / 转账被拒绝 |
| `card_arrival` / `order_physical_card` | 查询卡片到达状态 / 申请实体卡 |
| `request_refund` / `Refund_not_showing_up` | 发起退款 / 退款尚未到账 |
| `top_up_failed` / `topping_up_by_card` | 充值失败 / 咨询银行卡充值 |

这类任务有两个主要困难：

- 类别多且语义接近，普通分类损失未必能显式拉开相似类别的表示；
- 新意图和低频意图的标注成本高，需要考察少样本条件下的稳定性。

## 3. 数据集

本项目只使用 `data/banking_data/` 中的 BANKING77 数据。仓库 `data/` 下的 NLU++ 和 span extraction 数据来自同一上游数据仓库，不属于本项目当前实验范围。

| 属性 | 数值 |
|---|---:|
| 训练集 | 10,003 条 |
| 测试集 | 3,080 条 |
| 意图类别 | 77 类 |
| 语言 | 英语 |
| 训练集每类样本数 | 35–187 条 |
| 测试集每类样本数 | 40 条 |

训练集存在一定类别不均衡，但不将其夸大为严重长尾。官方测试集保持冻结，只用于最终评估；模型选择和超参数调整应使用从官方训练集分出的验证集。

## 4. 方法范围

项目遵循“先建立可信基线，再验证核心方法”的顺序。

### 4.1 基线

1. **TF-IDF + Linear SVM**：提供快速、可解释的词面特征基线。
2. **BERT 微调**：提供标准的预训练语言模型基线。
3. **SetFit**：提供面向少样本场景的对比学习基线。

### 4.2 核心方法

SupCon 使用标签构造正负样本关系，使同类表示更接近、异类表示更远。对样本 $i$，其监督对比损失为：

$$
L_i=-\frac{1}{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(z_i\cdot z_p/\tau)}
{\sum_{a\ne i}\exp(z_i\cdot z_a/\tau)}
$$

其中 $P(i)$ 是 batch 内与样本 $i$ 同类的样本集合，$\tau$ 是温度系数。训练时需要采用类平衡 batch，保证每个 anchor 都有同类正样本。

Hard negative mining 在异类样本中提高高相似度样本的权重，使模型更关注最容易混淆的类别边界。当前阶段只验证清晰、可控的 batch 内相似度加权方案，不同时引入动态混淆图、开放集识别或其他扩展。

## 5. 实验设计

### 5.1 数据设置

| 设置 | 标注预算 |
|---|---:|
| full | 使用训练池全部样本 |
| 5-shot | 每类 5 条，共 385 条 |
| 10-shot | 每类 10 条，共 770 条 |
| 20-shot | 每类 20 条，共 1,540 条 |

少样本从训练池按类别抽样；验证集不计入 few-shot 标注预算。少样本抽样和模型训练使用多个随机种子，最终报告均值与标准差。

### 5.2 对照与消融

主对照包括 TF-IDF + SVM、BERT 和 SetFit。核心消融固定相同的数据划分、BERT backbone、训练预算和评估代码，只改变对比学习模块：

| 实验组 | 目的 |
|---|---|
| BERT baseline | 建立标准分类基线 |
| BERT + SupCon | 测量监督对比学习的增量 |
| BERT + SupCon + hard negatives | 测量难负例挖掘的额外增量 |

### 5.3 评估指标

- **Macro-F1**：主指标，平等衡量 77 个类别；
- **per-class Precision / Recall / F1**：定位收益和退化发生在哪些类别；
- **confusion matrix**：检查语义相近 intent pair 的双向误判；
- **mean ± std**：反映多随机种子下的稳定性；
- **训练耗时**：作为方法成本的辅助指标。

t-SNE 等表示可视化只作为辅助证据，不替代定量结果。

## 6. 预期结论形式

项目不预设 SupCon 或 hard negatives 一定有效。最终结论至少回答：

1. 不同标注预算下，哪种方法的 Macro-F1 最好且最稳定；
2. SupCon 相对 BERT baseline 的增益是否稳定；
3. hard negatives 是否进一步减少高混淆意图对的误判；
4. 改进是否以其他类别退化或更多训练成本为代价；
5. 若总体提升不显著，是否仍在少样本或特定混淆类别上有一致收益。

## 7. 项目产出

完成后的最小交付物包括：

- 可复现的数据划分与训练入口；
- 三类基线与两类 SupCon 实验；
- 每次运行的配置、指标和逐样本预测；
- 主结果表、per-class F1、confusion matrix 和典型错误案例；
- 基于真实结果撰写的实验结论。

## 8. 当前状态与文档导航

当前仓库已经完成数据准备、EDA、统一基线代码与 Version 1 单随机种子运行。TF-IDF、BERT、SetFit 的四种标注预算均已生成完整结果：SetFit 在 few-shot 下表现最好，TF-IDF 在 full 下表现最好，BERT 少样本运行则出现预测塌缩。Version 1 属于流程验证与问题定位，不作为最终多随机种子结论。

具体任务顺序、验收标准和产物规范见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)，Version 1 实验分析见 [FIRST_EXPERIMENT_REPORT.md](FIRST_EXPERIMENT_REPORT.md)。
