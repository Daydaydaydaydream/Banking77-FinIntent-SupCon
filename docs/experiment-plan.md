# Banking77-FinIntent-SupCon：后续实验规划

本文档是 [`job-oriented-improvement-roadmap.md`](job-oriented-improvement-roadmap.md) 的落地版本。路线图回答"项目应该补齐哪些能力"，本文档回答"按什么顺序、改哪些文件、以什么标准判定完成"。

文档只描述实验设计与工程改动，不含进度承诺。所有耗时量级均标注为待校准估计。

---

## 1. 现状盘点

### 1.1 已有结果

`outputs/results/results.json` 当前记录：

| 方法 | full | 5-shot | 10-shot | 20-shot |
|---|---|---|---|---|
| TF-IDF + LinearSVC | 0.8924 | 0.5126 | 0.6509 | 0.7597 |
| BERT 微调 | 0.7964 | 0.6531 | 0.8057 | 0.8680 |
| 原生 SetFit | 0.8707 | 0.7340 | — | — |
| SupCon | — | — | — | — |
| SupCon + 难负例 | — | — | — | — |

### 1.2 三个必须先处理的事实

**（1）BERT full 结果存在异常，且不止一处。** 除路线图 §2.6 指出的 full(0.7964) < TF-IDF(0.8924) 之外，还存在第二个异常：BERT 的 full(0.7964) 低于其自身 10-shot(0.8057) 与 20-shot(0.8680)。同一模型、同一测试集，训练数据从 770 条增加到 10,003 条反而变差，这不能由噪声解释。两处异常需一并排查，否则 BERT 作为 SupCon 的对照系不可信。

**（2）SupCon 与难负例的实证结果完全缺失。** 项目标题与核心主张所依赖的模块尚未产出任何数字（`outputs/results/ablation_results.json` 不存在），`outputs/analysis/` 目录也未生成。这是项目从"课程实验"到"可展示项目"之间的唯一硬缺口。

**（3）当前实验划分不支持任何 checkpoint 选择。** 三个基线均设置 `eval_strategy="no"`、`save_strategy="no"`，全程无验证集，训练结束即用测试集评估。这意味着当前所有数字都建立在"固定 epoch 数 + 无模型选择"之上，且测试集被反复观察。一旦引入验证集，现有数字全部需要重跑。

### 1.3 代码级缺口

| 位置 | 问题 | 影响 |
|---|---|---|
| `src/baselines/bert_ft.py:88-89` | `eval_strategy="no"`、`save_strategy="no"` | 无验证集、无 checkpoint、无早停 |
| `src/baselines/bert_ft.py:117-122` | 只保存 `y_true`/`y_pred`，无 embedding、无概率 | 分析流程跳过 BERT |
| `src/baselines/setfit_baseline.py:72-77` | 只返回 `macro_f1` | 无法与其余方法做样本级对比 |
| `src/supcon/train.py:290-323` | 训练后不保存模型与配置 | 无法复现预测、无法服务化 |
| `src/supcon/train.py:123` | 仅用 `np.random.default_rng` | torch / DataLoader 随机性未固定 |
| `src/supcon/losses.py:91` | `torch.exp(logits)` 直接计算 | 换温度或混合精度时存在溢出风险 |
| `src/analysis/visualize.py` | `load_analysis` 强制要求 embedding 存在 | 无 embedding 的方法被整体跳过 |
| `src/config.py:30` | 单一 `SEED`，无多种子配置与 `VAL_RATIO` | 不支持多随机种子实验 |
| `.gitignore:221` | `outputs/` 全部忽略 | 公开仓库看不到结果与图表 |
| `README.md:5` | 链接指向 `dev-plan.md`，实际在 `docs/` | 首页链接失效 |

---

## 2. 规划总览

按"先让实验可信，再让方法成立，最后让系统可用"的顺序分为七个阶段。阶段 A 是所有后续工作的前置条件，不可跳过。

| 阶段 | 目标 | 对应路线图 | 是否必做 |
|---|---|---|---|
| A | 可复现性、验证集与统一产物契约 | §2.3, §3.1–3.4 | 必做 |
| B | 排查 BERT full 异常，确立可信对照系 | §2.6 | 必做 |
| C | 补齐实验矩阵（SetFit 10/20-shot、SupCon、SupCon+HN） | §2.1 | 必做 |
| D | 多随机种子与显著性检验 | §2.2, §2.5 | 必做 |
| E | 对齐条件下的完整消融 | §2.4 | 必做 |
| F | Confusion-aware 难负例挖掘 | §4.1 | 差异化主贡献 |
| G | 开放集识别 或 主动学习（二选一） | §4.2, §4.3 | 视目标岗位 |
| H | 工程闭环（服务、Docker、测试、benchmark） | §5 | 视目标岗位 |

阶段 A–D 产出"可信的基准结果"，是项目的最低可展示门槛。阶段 E–F 构成技术贡献。阶段 G–H 决定项目是否能作为单一代表作在面试中完整讲解。

---

## 3. 阶段 A：实验基础设施

### A.1 统一随机性控制

新增 `src/utils/seed.py`，提供 `seed_everything(seed, deterministic: bool = True)`，统一设置 `random`、`numpy`、`torch`、`torch.cuda`，并写入 `PYTHONHASHSEED`。MPS 侧设置 `torch.mps.manual_seed`，同时记录 `torch.use_deterministic_algorithms` 的实际状态。

要求：每个训练入口在数据加载之前调用一次；`DataLoader` 的 `generator` 由同一 seed 派生；`_balanced_batches` 的 `np.random.default_rng(SEED + epoch)` 改为由 `seed_everything` 派生，避免 epoch 编号与全局种子耦合。

### A.2 分层训练/验证划分

在 `src/data/prepare_data.py` 中增加分层划分，仅从原始 10,003 条训练集中切分：

- 训练集 9,000 条，验证集 1,003 条，按 `category` 分层（`train_test_split(stratify=...)`）；
- 测试集 3,080 条保持官方划分不变，冻结，仅用于最终评估；
- few-shot 的 k 条样本**只从训练集部分抽取**，验证集不计入标注预算，并在报告中显式说明该口径。

同时增加 `--dedup` 开关，用于路线图 §3.6 提及的去重敏感性实验：剔除 train/test 间 6 条标准化后完全相同的文本，保留官方划分结果作为主结果，去重结果单独报告。

> 影响说明：引入验证集改变了训练集构成，1.1 节中**全部数字（含 TF-IDF）都需重跑**。因此阶段 A 必须在阶段 C 之前完成，顺序不可颠倒。

### A.3 统一产物契约

新增 `src/utils/artifacts.py`，定义 `RunArtifacts`，所有方法（TF-IDF、BERT、SetFit、SupCon 及其变体）写入同一目录结构：

```
outputs/runs/<method>/<setting>/seed<k>/
├── metrics.json        # macro_f1, accuracy, micro_f1, per-class P/R/F1
├── predictions.csv     # y_true, y_pred, top1~top3 intent, confidence
├── embeddings.npy      # 测试集 encoder 表示（TF-IDF 存稀疏矩阵）
├── config.json         # 全部超参数 + setting + seed
├── env.json            # 依赖版本、设备、torch 确定性状态
└── model/              # 权重、tokenizer、label_map、最优 epoch 与验证集指标
```

`metrics.json` 与 `predictions.csv` 纳入版本管理（修正 `.gitignore` 只忽略 `outputs/runs/**/model/` 与临时文件），使公开仓库包含核心结果。

对应代码改动：

- `bert_ft.py`：`eval_strategy="epoch"`、`save_strategy="epoch"`、`load_best_model_at_end=True`、`metric_for_best_model="macro_f1"`、`save_total_limit=2`；通过 `output_hidden_states` 提取 `[CLS]` 作为 embedding，并保存 softmax 概率。
- `setfit_baseline.py`：补齐预测、概率与 body embedding，接口与其余方法一致。
- `supcon/train.py`：阶段 2 以验证集 Macro-F1 选择 checkpoint，训练结束保存 encoder、投影头、分类头与 label map。

### A.4 修复分析流程

`src/analysis/visualize.py` 的 `load_analysis` 改为 embedding 可选：无 embedding 时仍生成混淆矩阵、per-class F1 条形图与置信度分布，仅跳过 t-SNE。同时让 `analysis/report.py` 支持按 `<method>/<setting>/seed<k>` 聚合出 mean ± std。

### A.5 验收标准

- 同一配置连续运行两次，测试集预测逐条一致（`predictions.csv` 完全相同）；
- 四类方法均产出完整的 `outputs/runs/...` 目录树；
- 分析脚本对无 embedding 的方法不再抛错；
- `outputs/runs/**/metrics.json` 进入 git，checkpoint 不进 git。

---

## 4. 阶段 B：BERT full 异常排查

这是整条流水线的可信度前提。当前 BERT 既低于 TF-IDF，又低于自身 20-shot 结果，在原因澄清之前，SupCon 相对 BERT 的任何提升都无法解读。

排查按成本从低到高进行，每步记录到 `docs/bert-full-diagnosis.md`：

1. **收敛曲线**：开启验证集评估与训练日志，绘制 train/val loss 与 val Macro-F1 曲线。判定是欠拟合（曲线未收敛）、过拟合（val 早于 train 恶化）还是异常震荡。
2. **标签一致性**：校验 `label_map.json` 在数据构建、训练、评估三处的映射方向一致（`id2label` 与 `label2id` 不可混用），并抽查 20 条样本的 `text → label_id → intent` 全链路。
3. **超参扫描（full）**：`lr ∈ {2e-5, 3e-5, 5e-5}` × `epochs ∈ {3, 5, 8}`，共 9 次，其余设置固定。
4. **设备一致性**：同一配置在 MPS 与 CPU 各运行一次，比较 Macro-F1 与预测差异。
5. **重复样本影响**：在去重后的训练集上重跑 full，观察差异。
6. **对照 RF/线性探针**：固定 BERT 表示训练逻辑回归头（见阶段 E），判断问题出在表示质量还是微调过程。

**判定门（Gate 1）**：若 val 曲线正常收敛且 `lr=2e-5, epochs=5` 为最优，说明原配置合理，则异常源于任务本身——短文本 + 77 类细粒度场景下，TF-IDF 的词面特征对 BANKING77 具有较强的判别力，该现象本身是值得写入报告的发现。若扫描后 full Macro-F1 提升至 0.85 以上，则原配置的 epoch/lr 设置需修正，阶段 C–E 全部使用新配置。

---

## 5. 阶段 C：补齐实验矩阵

在阶段 A 的划分与新配置下补齐缺失格子：

| 方法 | full | 5-shot | 10-shot | 20-shot |
|---|---|---|---|---|
| TF-IDF + LinearSVC | 重跑 | 重跑 | 重跑 | 重跑 |
| Frozen BERT + Linear Probe | 新增 | 新增 | 新增 | 新增 |
| BERT 微调 | 重跑 | 重跑 | 重跑 | 重跑 |
| 原生 SetFit | 重跑 | 重跑 | **新增** | **新增** |
| SupCon（两阶段） | **新增** | **新增** | **新增** | **新增** |
| SupCon + 难负例 | **新增** | **新增** | **新增** | **新增** |
| CE + λ·SupCon（联合训练） | 新增 | 新增 | 新增 | 新增 |

指标口径统一为：Macro-F1（主）、Accuracy、Micro-F1、per-class P/R/F1，以及训练时长、峰值内存、参数量、模型磁盘大小。

阶段 C 的运行即为阶段 D 的 seed=42 分支，不重复计算。

---

## 6. 阶段 D：多随机种子与显著性

### D.1 种子设置

`src/config.py` 增加 `SEEDS = [42, 1, 2]`（为主结果），条件允许时扩展到 5 个种子。需固定 / 变化的量：

| 项目 | 处理 |
|---|---|
| 模型初始化 | 随种子变化 |
| batch 顺序、类别采样 | 随种子变化 |
| few-shot 样本抽取 | 随种子变化 |
| 数据划分（阶段 A.2） | 固定，所有种子共用同一划分 |

few-shot 的种子敏感性需单独报告，因为抽取方差通常大于训练方差。

### D.2 报告方式

- 主表：`Macro-F1 = mean ± std`，同时保留每个种子的原始 `metrics.json`，不筛选最优结果；
- 显著性：对测试集做**分层 paired bootstrap**（1000 次重采样，测试集每类 40 条，需保类），报告 ΔMacro-F1 的 95% 置信区间与双侧 p 值；
- 补充：种子数 ≥5 时增加 Wilcoxon signed-rank 检验。

**判定门（Gate 2）**：若 SupCon 相对 BERT 的 Macro-F1 均值提升 < 0.5pp 且 bootstrap 置信区间覆盖 0，则调整主张口径——把核心论点从"提升整体精度"改为"在 5-shot 与高混淆类别对上提升稳定性与可分性"，并以 per-class F1 的分布变化、t-SNE 分离度、混淆对误判率为证据。该情形下阶段 G 的权重相应提高。

---

## 7. 阶段 E：对齐消融

### E.1 公平性约束

对照组之间除被检验的模块外全部一致：backbone（`bert-base-uncased`）、tokenizer、`MAX_LEN=64`、数据划分、随机种子、计算预算（以步数或 FLOPs 计，而非 epoch 数）、checkpoint 选择方式、分类头结构、评估代码。

### E.2 实验组

| 组别 | 训练内容 | 作用 |
|---|---|---|
| Frozen BERT + Linear Probe | 冻结 encoder，仅训练线性头 | 衡量原始表示质量 |
| BERT + CE | 端到端微调 | 标准基线 |
| SupCon + Linear Probe | 阶段 1 对比学习 + 冻结 encoder 线性头 | 隔离对比学习的表示贡献 |
| SupCon + HN + Linear Probe | 增加难负例加权 | 隔离难负例的增量贡献 |
| CE + λ·SupCon | 单阶段联合训练，λ ∈ {0.1, 0.5, 1.0} | 比较两阶段与联合训练 |

`src/supcon/train.py` 增加 `--joint` 与 `--lambda` 参数以支持最后一组。

### E.3 计算预算对齐

两阶段方法的总步数与端到端微调需可比。记录每个方法的累计训练步数、墙钟时间与峰值显存，在报告中一并给出，避免"增加算力换性能"的质疑。

---

## 8. 阶段 F：Confusion-aware 难负例挖掘

这是项目的差异化主贡献。当前 `src/supcon/losses.py:93-105` 的难负例权重完全由 batch 内余弦相似度决定，未利用金融意图间的真实混淆结构。

### F.1 方法设计

1. 用阶段 C 的 BERT 基线生成完整混淆矩阵；
2. 由双向误判率构建 intent confusion graph：`w(c_i, c_j) = (n_ij + n_ji) / (n_i + n_j)`；
3. 训练时按该权重优先将高混淆类别放入同一 batch（替换当前均匀的 `rng.choice`）；
4. 对混淆图中的高权重类别对赋予更高的损失权重；
5. 每 N 个 epoch 用当前模型刷新一次混淆图（动态版），与一次性静态图对照；
6. 对照组：随机负例、batch 内相似度负例（现方法）、混淆图负例。

### F.2 数值稳定性与实现细节

- 分母改用 log-sum-exp 实现，替代 `losses.py:91` 的 `torch.exp(logits)`，避免低温度或 AMP 下溢出；
- 新增 `detach_weights` 开关，比较难负例权重是否阻断梯度；
- 记录每类的正样本数，防止 balanced batch 在高混淆类别对上产生 false negative。

### F.3 待回答问题

- 提升是否集中在高混淆 intent pair（如 `declined_card_payment` / `declined_transfer`）？
- 是否损害原本易分类的类别（per-class F1 的降幅分布）？
- 不同 `hard_negative_temperature` 下是否稳定？
- 动态混淆图是否优于静态图？
- 相对 SupCon 的额外训练成本是多少（时间、内存）？

`src/supcon/train.py` 新增 `--hn-mode {none,similarity,confusion_static,confusion_dynamic}`，与 `run_experiments.py` 的 `METHODS` 字典对接。

---

## 9. 阶段 G：深入方向（二选一）

路线图 §7.2 建议只选一个方向做深。选择依据是目标岗位：偏 Applied AI 选 G.1，偏算法研究选 G.2。

### G.1 开放集意图识别与安全回退

数据侧：构造 OOS 集合（BANKING77 之外的金融/通用查询）与多意图、模糊输入子集，模拟新意图漂移。

方法与指标：temperature scaling 置信度校准；报告 AUROC、AUPR、FPR@95TPR、ECE、Coverage–Accuracy 曲线，以及不同拒识阈值下的自动处理率与错误路由率。

### G.2 主动学习与标注成本

在 5/10/20-shot 之外，比较 Random、Entropy、Margin、Embedding Diversity、Uncertainty+Diversity、Confusion-aware 六种采样策略，输出标注量–Macro-F1 曲线，回答"达到目标性能需多少标注""相比随机采样节省多少"。

成本控制：主动学习轮次较多，建议缩减为 3 个预算点 × 2 个种子 × 8 轮。

---

## 10. 阶段 H：工程闭环

| 子项 | 内容 |
|---|---|
| 推理服务 | FastAPI，支持单条/批量、Top-K、置信度阈值、未知意图拒识、健康检查；响应含 `intent`、`confidence`、`top_k`、`route_to_human`、`model_version` |
| 可复现部署 | Dockerfile、一键启动、模型下载流程、环境变量与配置文件说明、示例请求响应 |
| 性能测试 | CPU 单条 P50/P95 延迟、MPS/CUDA 延迟、batch size–吞吐曲线、模型加载时间、峰值内存、磁盘大小 |
| 漂移监控 | 离线模拟输入长度分布、类别分布、embedding drift、低置信度比例、unknown 比例与每类错误率的变化，给出告警条件与人工处理流程 |
| 测试与 CI | SupCon loss 数值与梯度测试、hard-negative 退化性质、balanced batch 正样本保证、label map 一致性、数据划分无交叉、API schema、小样本 smoke test |

---

## 11. 执行顺序与计算预算

### 11.1 依赖顺序

```
A（基础设施）
 └─► B（BERT 诊断，Gate 1）
      └─► C（矩阵补齐）
           └─► D（多种子 + 显著性，Gate 2）
                ├─► E（对齐消融）
                └─► F（confusion-aware HN，Gate 3）
                     └─► G（OOS 或 主动学习）
                          └─► H（工程闭环 + README 重写）
```

E 与 F 可并行；H 的推理服务部分依赖 C 产出的 checkpoint。

### 11.2 运行次数

| 阶段 | 训练运行数 | 说明 |
|---|---|---|
| B | 11 | 9 次超参扫描 + 2 次设备对照，均为 full |
| C+D | 36 | 4 settings × 3 方法（BERT / SupCon / SupCon+HN）× 3 seeds |
| C+D 附加 | 12 | SetFit 4 settings × 3 seeds（轻量） |
| E | 24 | 联合训练 12 次 + 冻结探针 12 次（仅头部，成本约 1/10） |
| F | 12 | 4 settings × 3 seeds |
| 合计 | ≈ 95 | 其中 full-data 训练约 30 次 |

单次耗时量级（**估计值，需在阶段 A 完成后用统一计时实测校准**）：BERT-base、`seq_len=64`、Apple MPS 下 full 约 5–15 min，few-shot 约 2–5 min。据此估算总墙钟时间约 10–20 GPU 小时，不含超参扫描失败重跑。建议在阶段 A 完成后先跑一次完整计时，再据此裁剪种子数与扫描范围。

### 11.3 三个判定门

| 门 | 触发点 | 若不通过 |
|---|---|---|
| Gate 1 | B 结束后 | 按 §4 结论修正 BERT 配置，或将其作为分析性发现写入报告 |
| Gate 2 | D 结束后 | 按 §6 调整主张口径，提高阶段 G 权重 |
| Gate 3 | F 结束后 | 难负例无显著增益时，如实报告负结果并做机制分析（false negative 问题），或退回 confusion-aware 采样单模块 |

三个判定门均设有退路，任一模块失败不阻塞项目收尾。

---

## 12. 需要修改的文件清单

| 文件 | 改动 |
|---|---|
| `src/utils/seed.py` | 新增：统一随机性控制 |
| `src/utils/artifacts.py` | 新增：统一产物契约 |
| `src/utils/metrics.py` | 补充 accuracy、micro-F1、分层 bootstrap 检验 |
| `src/config.py` | 增加 `SEEDS`、`VAL_RATIO`、`RUNS_DIR` |
| `src/data/prepare_data.py` | 分层 train/val 划分、`--dedup` 开关 |
| `src/baselines/bert_ft.py` | 验证集评估、checkpoint 保存、embedding 与概率导出 |
| `src/baselines/setfit_baseline.py` | 补齐预测、概率、embedding 导出 |
| `src/baselines/tfidf_svm.py` | 接入产物契约；修正 n-gram 注释与实现不一致问题 |
| `src/supcon/train.py` | seed、模型保存、val 选点、`--joint/--lambda`、`--hn-mode` |
| `src/supcon/losses.py` | log-sum-exp 稳定实现、`detach_weights`、混淆权重注入 |
| `src/analysis/visualize.py` | embedding 可选 |
| `src/analysis/report.py` | 多种子聚合、mean ± std、显著性输出 |
| `src/run_experiments.py` | 支持 seed 维度与 hn-mode 维度的编排 |
| `.gitignore` | 仅忽略 checkpoint 与临时文件，保留结果与图表 |
| `README.md` | 修复 `dev-plan.md` 链接、补全目录树、增加结果表与复现命令 |
| `tests/` | 新增：loss 数值测试、划分无交叉、API schema、smoke test |

---

## 13. 最终结果表口径

阶段 D 结束后，主结果表按下表固定列结构，直接用于 README 与报告：

| Method | full | 5-shot | 10-shot | 20-shot | Params | Train time | P95 latency |
|---|---|---|---|---|---|---|---|
| TF-IDF + LinearSVC | | | | | | | |
| Frozen BERT + Linear Probe | | | | | | | |
| BERT + CE | | | | | | | |
| SetFit | | | | | | | |
| SupCon | | | | | | | |
| SupCon + HN | | | | | | | |
| Confusion-aware HN | | | | | | | |

每个单元格填 `mean ± std`（3–5 seeds），显著性标注相对当前列最强基线的 paired bootstrap 结果。附一张高混淆类别对的混淆矩阵对比图与一张 t-SNE 图作为视觉证据。

README 首页按路线图 §6 的结构组织，并优先展示本表、混淆对比图与系统架构图。
