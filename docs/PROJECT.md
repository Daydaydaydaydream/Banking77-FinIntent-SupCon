# Banking77-FinIntent-SupCon 项目文档

> 面向金融客服场景的细粒度意图分类，融合监督对比学习（SupCon）与难负例挖掘，
> 在标注资源受限（few-shot）条件下提升语义高度重叠意图类别的区分能力。

---

## 1. 项目概述

本项目针对金融客服意图分类任务中两类核心挑战——**细粒度语义重叠**与**少样本标注**——开展方法研究。实验数据采用 PolyAI 发布的 BANKING77 基准数据集，方法上以监督对比学习（Supervised Contrastive Learning, SupCon）为核心，叠加难负例挖掘（hard-negative weighting）变体，通过多组基线与消融实验量化各模块的增量贡献。

项目采用"基线优先、核心方法验证、少样本评估"的技术路线：

1. **基线对照**：建立 TF-IDF+SVM、BERT 微调、原生 SetFit 三组基线，在全量与少样本设置下确立标准性能参照。
2. **核心方法**：实现 SupCon 损失函数及其难负例加权变体，验证其是否能在表征层面拉开易混淆意图的特征间距。
3. **消融分析**：对比 `Baseline`、`Baseline + SupCon`、`Baseline + SupCon + 难负例挖掘` 三组设置，明确各模块的增量贡献。

---

## 2. 项目背景与研究问题

### 2.1 任务定义

意图分类（Intent Classification）是客服聊天机器人中的核心模块。给定一条用户输入的简短查询（utterance），系统需预测其所属意图类别，以路由至对应应答策略或业务流程。形式化表述为：

$$f: x \rightarrow y, \quad y \in \{c_1, c_2, \dots, c_{77}\}$$

其中 $x$ 为用户查询文本，$y$ 为预测的意图标签。

### 2.2 两大现实难题

#### （1）细粒度、语义高度重叠的意图类别

金融客服场景中存在大量业务相近、表述相似的细分意图。以 BANKING77 为例，77 个类别同属银行服务域，许多意图在语义上仅有微妙差异：

| 易混淆意图对 | 语义差异 |
|---|---|
| `declined_card_payment` vs `declined_transfer` | 均为"交易被拒绝"，区别在于支付方式是卡支付还是转账 |
| `card_arrival` vs `order_physical_card` | 均涉及实体卡，前者询问卡是否到达，后者请求办理新卡 |
| `request_refund` vs `Refund_not_showing_up` | 均与退款相关，前者发起退款请求，后者投诉退款未到账 |
| `top_up_failed` vs `topping_up_by_card` | 均涉及账户充值，前者报告充值失败，后者咨询充值方式 |

此类细粒度重叠使通用预训练模型在分类边界上极易产生混淆，直接影响客服机器人的路由准确率。

#### （2）标注成本高昂与少样本场景

真实金融客服业务中的意图标注面临以下约束：

- **标注依赖领域专家**：金融意图的准确区分依赖业务知识，普通标注人员难以胜任，导致标注成本高、周期长。
- **长尾意图样本稀缺**：大量细分意图在实际业务中出现频率低，仅能收集到少量带标签样本（few-shot）。
- **意图体系持续演进**：银行业务持续更新，新意图不断涌现，无法为每个新意图积累充足标注数据。

### 2.3 研究问题

本项目的核心研究问题为：**在细粒度语义重叠 + 少样本标注的双重约束下，如何构建高效、鲁棒的金融客服意图分类模型？**

---

## 3. 数据集

### 3.1 数据集概述

BANKING77 由 PolyAI 发布，是金融客服领域公认的细粒度意图分类基准，涵盖真实在线银行场景下用户向客服提出的简短自然语言查询。

### 3.2 基本统计

| 属性 | 数值 |
|---|---|
| 总样本数 | 13,083 条 |
| 训练集 | 10,003 条 |
| 测试集 | 3,080 条 |
| 意图类别数 | 77 个 |
| 语言 | 英语 |
| 平均查询长度 | 约 11–12 个词（短文本） |

### 3.3 意图覆盖范围

77 个意图按功能可归为若干大类：

- **银行卡管理**：`card_arrival`、`activate_my_card`、`lost_or_stolen_card`、`card_swallowed` 等
- **转账与支付**：`declined_transfer`、`declined_card_payment`、`failed_transfer`、`cancel_transfer` 等
- **账户充值**：`topping_up_by_card`、`top_up_failed`、`top_up_limits` 等
- **交易与争议**：`transaction_charged_twice`、`request_refund`、`extra_charge_on_statement` 等
- **账户与安全**：`terminate_account`、`age_limit`、`verify_my_identity`、`pin_blocked` 等

### 3.4 数据集特点与挑战

- **细粒度标注**：77 个类别均为同一金融域内的细分意图，类别间边界模糊。
- **语义高度重叠**：多对意图在字面上极为相似，是模型混淆的主要来源。
- **类别分布不均衡**：训练集中各意图样本数差异较大，部分长尾意图样本稀缺。
- **短文本特性**：查询平均仅十余词，上下文信息有限，进一步加大细粒度区分难度。

---

## 4. 方法论

### 4.1 基线方法

在核心方法之外，建立三组基线作为性能参照：

#### （1）TF-IDF + LinearSVC

经典词袋模型基线。采用 sublinear TF-IDF 特征（word 1-2 gram），送入带类别平衡权重的线性支持向量机（LinearSVC）。该基线训练快速，可作为深度学习模型的下界参照。

实现要点：

- `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)`
- `LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)`

#### （2）BERT 微调

以 `bert-base-uncased` 为骨干，在线性分类头下进行全参数微调。鉴于 Banking77 查询为短文本，输入截断至 64 token。设备支持 MPS（Apple Silicon）与 CUDA，无 GPU 时回退 CPU。

#### （3）原生 SetFit

采用 `setfit` 库的两阶段训练：先对 `sentence-transformers/all-MiniLM-L6-v2` 的 body 做对比学习微调，再在冻结表征上训练逻辑回归头（`use_differentiable_head=False`）。少样本设置沿用 SetFit 的规范配置（较多对比迭代次数），全量设置减少迭代次数以控制耗时。

### 4.2 核心方法：监督对比学习（SupCon）

#### 4.2.1 损失函数

监督对比学习（Khosla et al., NeurIPS 2020）利用标签信息，将同类样本的表征拉近、异类样本的表征推远。设 batch 内样本经 L2 归一化后得到特征 $\mathbf{z}$，对 anchor $i$，其正样本集合为 $P(i)$（同类、排除自身），$A(i)$ 为 batch 内除自身外所有样本：

$$L_i = -\frac{1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

其中 $\tau$ 为温度系数，本项目取 $\tau = 0.07$。

#### 4.2.2 难负例加权变体

标准 SupCon 的分母对所有负样本等权处理，无法针对性地惩罚高混淆的跨类样本对。难负例加权变体按负样本与 anchor 的相似度重新分配权重：

$$w_{i,n} = \mathrm{softmax}_{n \in N(i)}\!\left(\mathbf{z}_i \cdot \mathbf{z}_n / \tau_{hn}\right) \cdot |N(i)|$$

其中 softmax 仅在负样本集合 $N(i)$ 上归一化，$|N(i)|$ 为负样本数量。分母相应变为：

$$\mathrm{denom}_i = \sum_{p \in P(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau) + \sum_{n \in N(i)} w_{i,n} \cdot \exp(\mathbf{z}_i \cdot \mathbf{z}_n / \tau)$$

$\tau_{hn}$ 控制权重向最难负样本集中的程度：$\tau_{hn}$ 越小，权重越集中于相似度最高的负样本；当 $\tau_{hn} \to \infty$ 时，softmax 趋于均匀分布，$w_{i,n} \to 1$，目标严格退化为标准 SupCon，从而保证消融实验的可控性。

#### 4.2.3 两阶段训练策略

模型由三部分组成：BERT encoder、投影头（projection head）、分类头（classification head）。训练分为两个阶段：

1. **阶段一（表征学习）**：训练 encoder 与投影头，以监督对比损失优化。encoder 输出 `[CLS]` 池化表示 $\mathbf{h}$，投影头将其映射为 L2 归一化的对比嵌入 $\mathbf{z}$（维度 128）。
2. **阶段二（线性分类）**：冻结 encoder，仅训练分类头，以交叉熵损失优化。

采用两阶段而非端到端联合训练的原因在于：对比目标与分类目标若同时作用于 encoder，二者在优化方向上可能相互干扰；分阶段训练使表征学习与判别学习解耦，保证对比损失专注于塑造特征空间结构。

#### 4.2.4 类平衡 batch 采样

监督对比损失要求 batch 内存在同类正样本。为此，每个 batch 通过类平衡采样构造：随机选取 `batch_classes` 个类别，每个类别取 `batch_per_class` 个样本。该设计确保 5-shot 场景下对比损失始终能观察到同类正样本。各数据设置下的采样参数如下：

| 设置 | batch_classes | batch_per_class |
|---|---|---|
| full | 16 | 8 |
| 5-shot | 16 | 5 |
| 10-shot | 16 | 8 |
| 20-shot | 16 | 8 |

---

## 5. 实验设计

### 5.1 数据设置

在原始 BANKING77 训练集上构造四种数据设置（固定随机种子 `seed=42`，`shuffle + groupby.head(k)` 分层采样）：

| 数据设置 | 训练样本数 | 说明 |
|---|---|---|
| full | 10,003 | 原始全量训练集 |
| 5-shot | 385 | 每类 5 条标注 |
| 10-shot | 770 | 每类 10 条标注 |
| 20-shot | 1,540 | 每类 20 条标注 |

测试集在所有设置下保持一致（3,080 条），保证结果可比。

### 5.2 基线实验

三组基线（TF-IDF+SVM、BERT 微调、原生 SetFit）在四种数据设置下分别训练与评估，统一报告测试集 Macro-F1。

### 5.3 消融实验

在 BERT 骨干上设置三组消融，量化各模块的增量贡献：

| 消融组 | 方法 |
|---|---|
| `baseline_bert` | BERT 直接微调（Baseline） |
| `supcon` | BERT + 监督对比预训练（Baseline + SupCon） |
| `supcon_hn` | BERT + SupCon + 难负例加权（Baseline + SupCon + HN） |

### 5.4 评估指标

- **主指标**：Macro-F1（对 77 个类别取算术平均的 F1，缓解类别不均衡影响）。
- **辅助指标**：per-class F1（观察易混淆意图对的逐类提升）、混淆矩阵（观察误判率变化）。
- **可视化证据**：t-SNE 特征降维（直观展示特征分离效果，作为辅助证据，核心结论以定量指标为准）。

---

## 6. 系统架构与代码组织

### 6.1 目录结构

```
.
├── data/
│   └── banking_data/            # BANKING77 原始数据 (train.csv / test.csv / categories.json)
├── src/
│   ├── config.py                # 路径、随机种子、数据集常量（单一事实来源）
│   ├── data/prepare_data.py     # 构建全量 + 5/10/20-shot 数据集
│   ├── utils/
│   │   ├── data.py              # 数据加载与 label map 读取
│   │   └── metrics.py           # Macro-F1 / per-class F1 / 混淆矩阵
│   ├── baselines/
│   │   ├── tfidf_svm.py         # 基线 1：TF-IDF + LinearSVC
│   │   ├── bert_ft.py           # 基线 2：bert-base-uncased 微调
│   │   └── setfit_baseline.py   # 基线 3：原生 SetFit
│   ├── supcon/
│   │   ├── losses.py            # SupConLoss（含难负例加权变体）
│   │   ├── model.py             # SupConModel：encoder + 投影头 + 分类头
│   │   └── train.py             # 两阶段训练 + 类平衡 batch 采样
│   ├── analysis/
│   │   ├── visualize.py         # 混淆矩阵 + 易混淆对子矩阵 + t-SNE
│   │   └── report.py            # 生成 report.md
│   ├── run_baselines.py         # 基线编排入口（断点续跑）
│   └── run_experiments.py       # 消融实验编排入口（断点续跑）
├── outputs/                     # 生成的中间数据与结果（gitignore）
│   ├── datasets/                # 预处理后的训练/测试 split 与 label map
│   ├── results/                 # results.json / ablation_results.json
│   ├── analysis/                # 预测与 embedding 的 npy 持久化
│   └── figures/                 # 混淆矩阵与 t-SNE 图
├── docs/PROJECT.md              # 本文档
├── requirements.txt             # 钉死版本的依赖清单
├── README.md                    # 项目速览
└── dev-plan.md                  # 项目开发计划
```

### 6.2 数据流

```
data/banking_data/{train,test}.csv + categories.json
        │
        ▼  src/data/prepare_data.py
outputs/datasets/  (label_map.json, test.csv, train_{full,5shot,10shot,20shot}.csv)
        │
        ├──▶ baselines/tfidf_svm.py ────────┐
        ├──▶ baselines/bert_ft.py ──────────┤
        ├──▶ baselines/setfit_baseline.py ──┤
        ├──▶ supcon/train.py ───────────────┤
        │                                     ▼
        │                            outputs/results/*.json (Macro-F1 / per-class F1)
        │                                     │
        └──▶ analysis/visualize.py ◀──────────┤  (读 outputs/analysis/*.npy)
                                              ▼
                              outputs/figures/*.png
                                              │
                                              ▼
                              analysis/report.py ──▶ report.md
```

### 6.3 关键设计决策

| 决策 | 理由 |
|---|---|
| 两阶段训练（SupCon 后线性分类） | 解耦表征学习与判别学习，避免对比与分类目标相互干扰 |
| 类平衡 batch 采样 | 保证 5-shot 场景下对比损失始终能观察到同类正样本 |
| 难负例加权退化为标准 SupCon | $\tau_{hn} \to \infty$ 时权重均匀，保证消融实验可控 |
| 统一报告 Macro-F1 | 缓解 77 类长尾分布对指标的影响，保证跨方法可比 |
| 断点续跑机制 | 结果增量写入 JSON，跳过已完成项，支持部分重跑 |

---

## 7. 运行方式

### 7.1 环境安装

```bash
pip install -r requirements.txt
```

**版本约束**（重要）：

- `setfit 1.1.3` 强制依赖 `transformers<5.0` 与 `sentence-transformers 3.x`。升级 `transformers` 至 5.x 或 `sentence-transformers` 至 6.x 前，必须同步升级 `setfit`，否则会出现 `default_logdir` 等导入错误。
- 钉死版本见 `requirements.txt`（torch 2.14.0 / transformers 4.57.6 / sentence-transformers 3.4.1 / setfit 1.1.3 等）。

**国内网络**：模型下载需走 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 7.2 数据准备

```bash
cd src
python -m data.prepare_data
```

一次性生成全量与 5/10/20-shot 训练集、共享测试集与 label map，写入 `outputs/datasets/`。

### 7.3 运行基线

```bash
# 运行全部基线（自动跳过已完成项，可断点续跑）
python -m run_baselines --baseline all --setting all

# 运行单个基线 / 单个数据设置
python -m run_baselines --baseline bert_ft --setting 5shot
python -m baselines.tfidf_svm full      # 单独运行
```

### 7.4 运行消融实验

```bash
# 运行全部消融组与数据设置
python -m run_experiments --method all --setting all

# 单独运行 SupCon + 难负例挖掘
python -m supcon.train 5shot --hn
```

### 7.5 可视化与报告

```bash
python -m analysis.visualize   # 生成混淆矩阵与 t-SNE 图到 outputs/figures/
python -m analysis.report      # 汇总结果生成 report.md
```

---

## 8. 当前实验结果

> 注：以下为当前已完成的结果。SetFit 的 10-shot / 20-shot 与 SupCon 消融实验尚在运行中。

### 8.1 基线 Macro-F1（测试集）

| 方法 | full | 5-shot | 10-shot | 20-shot |
|---|---|---|---|---|
| TF-IDF + LinearSVC | 0.8924 | 0.5126 | 0.6509 | 0.7597 |
| BERT 微调 | 0.7964 | 0.6531 | 0.8057 | 0.8680 |
| SetFit | 0.8707 | 0.7340 | — | — |

### 8.2 阶段性观察

- **SetFit 在少样本场景优势明显**：5-shot 下 SetFit（0.7340）显著高于 BERT 微调（0.6531）与 TF-IDF+SVM（0.5126），体现对比式少样本微调对小样本的适配性。
- **TF-IDF+SVM 在 full 设置下表现突出**（0.8924，高于 BERT 微调的 0.7964）。归因于 Banking77 为短文本、词表集中的领域数据，TF-IDF 的词/字符 n-gram 共现特征在全量数据下已具备较强判别力；而 BERT full 仅训练 5 epoch、学习率 2e-5，存在欠拟合可能。此现象需在最终报告中单独标注并解释，避免误导。

---

## 9. 环境依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| torch | 2.14.0 | 深度学习框架（MPS/CUDA） |
| transformers | 4.57.6 | BERT 模型与 Trainer |
| sentence-transformers | 3.4.1 | SetFit 骨干 |
| setfit | 1.1.3 | 少样本对比微调框架 |
| datasets | 5.0.1 | HuggingFace 数据集接口 |
| accelerate | 1.14.0 | 分布式/混合精度训练支持 |
| scikit-learn | 1.9.0 | SVM、指标、t-SNE |
| scipy | 1.18.1 | 科学计算 |
| pandas | 3.0.5 | 数据处理 |
| numpy | 2.5.2 | 数值计算 |

---

## 10. 后续计划

1. **补跑 SetFit**：完成 10-shot / 20-shot 设置，补全基线对照表。
2. **核心消融实验**：运行 `run_experiments`，完成 `Baseline` / `+SupCon` / `+SupCon+HN` 三组消融，观察提升是否集中在易混淆意图对上。
3. **分析与可视化**：生成混淆矩阵与 t-SNE 图，从表征层面验证特征分离效果。
4. **报告撰写**：汇总实验数据与图表，完成最终实验报告，重点分析 SupCon 与难负例挖掘在细粒度语义重叠场景下的增量贡献。

---

> 注：本文档部分内容可能由 AI 生成。
