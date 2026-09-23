# Version 1 基线实验报告

## 1. 报告范围

本报告总结 `outputs/version1/` 中保存的第一版完整基线实验。Version 1 已覆盖 3 种方法、4 种标注预算和统一的 `seed=42`，共 12 次运行：

- TF-IDF + Linear SVM；
- BERT 全参数微调；
- SetFit；
- 5-shot、10-shot、20-shot 和 full。

本轮已经能够比较三种基线，但仍属于探索性单随机种子结果。特别是 BERT 的少样本训练出现明显预测塌缩，因此不能把 Version 1 直接作为最终论文结论，也不能据此宣称某种深度学习方法在统计意义上稳定优于另一种方法。

## 2. 实验设置

### 2.1 数据与标注预算

| 项目 | 数值 |
|---|---:|
| 意图类别 | 77 |
| 训练池 | 9,000 |
| 固定验证集 | 1,003 |
| 官方测试集 | 3,080 |
| 测试集每类样本 | 40 |

少样本训练集从训练池中按类别抽取：

| Setting | 每类训练样本 | 训练样本总数 |
|---|---:|---:|
| 5-shot | 5 | 385 |
| 10-shot | 10 | 770 |
| 20-shot | 20 | 1,540 |
| full | 不限 | 9,000 |

验证集固定不变，官方测试集不参与 checkpoint 选择。Version 1 的完整运行产物见 [`outputs/version1/runs/`](../outputs/version1/runs/)。

### 2.2 方法配置

1. **TF-IDF + Linear SVM**：word 1–2 gram、sublinear TF、`C=1.0`、类别平衡；
2. **BERT**：`bert-base-uncased` 全参数微调，5 epochs，batch size 32，学习率 `2e-5`，最大长度 64，按验证集 Macro-F1 选择 checkpoint；
3. **SetFit**：`sentence-transformers/all-MiniLM-L6-v2`，少样本设置使用 20 次对比样本迭代，full 设置使用 1 次迭代，再训练分类头。

## 3. 总体结果

### 3.1 Validation 与 Test 指标

| Method | Setting | Validation Macro-F1 | Test Macro-F1 | Test Accuracy | 训练时间（秒） |
|---|---|---:|---:|---:|---:|
| TF-IDF + SVM | 5-shot | 0.5091 | 0.5519 | 0.5620 | 0.02 |
| TF-IDF + SVM | 10-shot | 0.6331 | 0.6649 | 0.6708 | 0.05 |
| TF-IDF + SVM | 20-shot | 0.7348 | 0.7567 | 0.7584 | 0.08 |
| TF-IDF + SVM | full | **0.8639** | **0.8879** | **0.8880** | 0.37 |
| BERT | 5-shot | 0.0063 | 0.0077 | 0.0279 | 71.27 |
| BERT | 10-shot | 0.0298 | 0.0351 | 0.0633 | 113.02 |
| BERT | 20-shot | 0.2643 | 0.2676 | 0.3019 | 191.78 |
| BERT | full | 0.8527 | 0.8644 | 0.8744 | 1,009.27 |
| SetFit | 5-shot | **0.7273** | **0.7557** | **0.7604** | 196.50 |
| SetFit | 10-shot | **0.7639** | **0.8062** | **0.8094** | 337.97 |
| SetFit | 20-shot | **0.8147** | **0.8325** | **0.8354** | 599.60 |
| SetFit | full | 0.8545 | 0.8778 | 0.8779 | 208.90 |

汇总数据和主图可直接查看：

- [`test_metrics_summary.csv`](../outputs/version1/figures/model_results/test_metrics_summary.csv)
- [`test_macro_f1_comparison.png`](../outputs/version1/figures/model_results/test_macro_f1_comparison.png)

Version 1 呈现出三个清楚的现象：

- **SetFit 是当前最强的 few-shot 基线。** 它在 5-shot、10-shot 和 20-shot 上均取得最高 Test Macro-F1；
- **TF-IDF + SVM 是当前最强的 full-data 基线。** full Test Macro-F1 为 0.8879，比 SetFit 高 0.0101，比 BERT 高 0.0235；
- **BERT 只有 full 设置可以视为正常基线。** 三个少样本设置均存在不同程度的预测塌缩，当前数值不适合与 SupCon 做公平比较。

### 3.2 SetFit 相对 TF-IDF 的增量

| Setting | Validation Macro-F1 增量 | Test Macro-F1 增量 | 测试集配对重采样 95% 区间 |
|---|---:|---:|---:|
| 5-shot | +0.2182 | +0.2038 | [0.1850, 0.2219] |
| 10-shot | +0.1308 | +0.1412 | [0.1237, 0.1579] |
| 20-shot | +0.0799 | +0.0758 | [0.0602, 0.0907] |
| full | -0.0095 | -0.0101 | [-0.0211, 0.0022] |

配对重采样按测试类别分层，运行 500 次。前三个少样本预算下，SetFit 的优势在当前测试样本上较稳定；full 设置的区间跨越 0，不能根据本轮样本认定两者有稳定差异。该区间只描述测试样本重采样的不确定性，不包含 few-shot 抽样、参数初始化和训练随机性，不能替代多随机种子实验。

逐类比较也支持这一趋势：SetFit 相对 TF-IDF 在 5-shot、10-shot 和 20-shot 下分别改善了 71、67 和 59 个类别；full 下则是 35 类改善、38 类下降、4 类持平，说明 full 场景不存在全面优势。

## 4. 模型行为分析

### 4.1 SetFit：少样本下有效且覆盖完整

SetFit 在四种设置中都预测出了全部 77 个类别，没有出现 Test F1 为 0 的类别。5-shot 的预测类别分布归一化熵约为 0.992，表明预测没有集中到少数标签。

5-shot 下，相对 TF-IDF 改善最大的类别包括：

| Class | F1 增量 |
|---|---:|
| `top_up_failed` | +0.573 |
| `atm_support` | +0.535 |
| `failed_transfer` | +0.492 |
| `transaction_charged_twice` | +0.461 |
| `top_up_by_bank_transfer_charge` | +0.418 |
| `declined_cash_withdrawal` | +0.415 |

不过，SetFit 仍然存在稳定的难类。5-shot 下 `topping_up_by_card` 的 F1 只有 0.314，`balance_not_updated_after_bank_transfer`、`why_verify_identity`、`beneficiary_not_allowed` 和 `supported_cards_and_currencies` 也低于或接近 0.53。20-shot 中 `why_verify_identity` 的 F1 反而降至 0.122，说明增加样本并没有自动解决身份验证类别边界问题。

### 4.2 BERT：少样本预测塌缩

| Setting | 预测出的类别数 | Test F1 为 0 的类别数 | 预测分布归一化熵 |
|---|---:|---:|---:|
| 5-shot | 24 / 77 | 66 / 77 | 0.463 |
| 10-shot | 48 / 77 | 46 / 77 | 0.611 |
| 20-shot | 68 / 77 | 13 / 77 | 0.807 |
| full | 75 / 77 | 2 / 77 | 0.990 |

5-shot 验证 Macro-F1 在第 4 个 epoch 最高也只有 0.0063，验证损失从约 4.357 降至 4.295；77 类随机预测的交叉熵约为 `ln(77)=4.34`，说明模型基本没有学到可用的类别边界。10-shot 和 20-shot 虽有所改善，但仍远未充分收敛。full 设置则正常收敛，最后验证损失约为 0.79。

根本问题是所有数据预算都固定为 5 epochs。5-shot、10-shot 和 20-shot 的总 optimizer steps 远少于 full，同时前 10% 仍用于 warmup。相同 epoch 并不等于相同训练预算，全参数微调在极少样本下也更容易受到初始化和数据顺序影响。

### 4.3 易混淆类别与 hard negative 候选

当前错误集中在业务对象相同、但状态或用户诉求不同的类别组：

| 类别组 | 主要区别 |
|---|---|
| `verify_my_identity` / `why_verify_identity` / `unable_to_verify_identity` | 如何验证、为何验证、无法验证 |
| `pending_top_up` / `top_up_failed` / `top_up_reverted` | 充值处理中、充值失败、充值退回 |
| `get_disposable_virtual_card` / `virtual_card_not_working` | 获取虚拟卡、虚拟卡无法使用 |
| `balance_not_updated_after_bank_transfer` / `balance_not_updated_after_cheque_or_cash_deposit` | 银行转账后余额未更新、现金或支票存入后余额未更新 |
| `card_payment_wrong_exchange_rate` / `wrong_exchange_rate_for_cash_withdrawal` | 卡支付汇率、取现汇率 |

5-shot 中，SetFit 相对 TF-IDF 明显减少了部分双向误判：

| Intent pair | TF-IDF | SetFit |
|---|---:|---:|
| `pending_top_up` ↔ `top_up_failed` | 21 | 5 |
| `atm_support` ↔ `card_acceptance` | 20 | 1 |
| `exchange_via_app` ↔ `exchange_charge` | 15 | 1 |
| `get_disposable_virtual_card` ↔ `disposable_card_limits` | 21 | 8 |
| `verify_my_identity` ↔ `why_verify_identity` | 30 | 22 |

但并非所有类别对都改善。20-shot 中 `verify_my_identity` ↔ `why_verify_identity` 的双向误判由 TF-IDF 的 8 次增加到 SetFit 的 27 次。这组类别应作为后续 SupCon + hard negatives 的重点对象，同时也用于检查难负例是否带来副作用。

这些候选来自测试集的事后误差分析，只能用于提出研究假设。后续 hard negative 构造和超参数选择必须基于训练集与验证集，不能直接用测试混淆矩阵反复调参。

## 5. 时间成本

TF-IDF 的训练时间不足 1 秒，仍然是最适合做流程检查和强词面基线的方法。当前设备与实现下，BERT 推理速度约为每秒 331–339 条，SetFit 约为每秒 2,367–2,862 条，SetFit 大约快 7–9 倍。

训练时间不能直接横向解读。SetFit 的 few-shot 设置使用 20 次对比样本迭代，而 full 只使用 1 次，因此出现“full 比 20-shot 更快”的非单调现象。这是不同训练预算造成的，不代表数据越多训练越快。后续公平比较应同时报告 optimizer steps、epoch/iteration 和 wall-clock time。

## 6. 当前问题与局限

1. **只有一个随机种子。** 当前不能估计 few-shot 抽样和深度模型初始化的波动，也不能报告正式的 `mean ± std`。
2. **BERT 少样本基线无效。** 如果直接与 SupCon 比较，提升可能仅来自训练步骤或 batch 构造更充分，而不是对比学习本身。
3. **SetFit 训练历史为空。** 四次运行的 `training_history.json` 均为 `[]`，当前代码很可能读取了外层 trainer 状态，而实际日志保存在内部 Sentence Transformers trainer 中。
4. **部分元数据含旧绝对路径。** `best_checkpoint` 和可视化 manifest 仍保存旧机器路径，移动到 `outputs/version1/` 后已经失效。后续应保存相对路径或在归档时重写。
5. **checkpoint 体积较大。** 四个 SetFit 中间 checkpoint 合计约 1 GB；当前 `--no-save-model` 只控制最终模型，没有完全禁止训练器保存中间 checkpoint。
6. **测试集已用于探索性分析。** Version 1 已查看测试指标与混淆关系，后续配置选择必须严格只看验证集，测试集只用于冻结方案后的最终评估。

## 7. 下一阶段计划

### 7.1 修复并冻结可信 BERT baseline

- 将 few-shot 的固定 5 epochs 改为最低 optimizer steps 或设置更高的最大 epoch，并用验证集早停；
- 在验证集小范围比较学习率、warmup、冻结 encoder 与只训练分类头等稳定化方案；
- 自动记录预测类别覆盖数、零 F1 类别数和预测分布熵，把塌缩运行标为异常；
- 只有当三个 few-shot 设置都覆盖绝大多数类别并明显优于随机状态后，才冻结 BERT 配置。

### 7.2 修复产物记录

- 从正确的内部 trainer 导出 SetFit 训练历史；
- 将 checkpoint 和 manifest 路径改为相对路径；
- 让 `--no-save-model` 同时关闭或清理中间 checkpoint；
- 在汇总表中补充训练步数，避免只用时间比较不同预算。

### 7.3 完成多随机种子基线

固定配置后，至少运行 3 个随机种子。主结果报告 `mean ± std`，同时保留每个 seed 的 per-class F1、预测文件与 confusion matrix。如果深度模型波动较大，应先报告稳定性问题，而不是只选择最好的一次。

### 7.4 开展核心消融

按以下顺序运行，并保持相同 backbone、数据、随机种子、验证规则和等价训练预算：

1. BERT baseline；
2. BERT + SupCon；
3. BERT + SupCon + hard negatives。

最终不仅比较 Macro-F1，还要检查高混淆 intent pair 的双向误判是否稳定下降、改善是否集中于 few-shot、是否导致其他类别退化，以及性能提升是否值得额外训练成本。

## 8. Version 1 结论

Version 1 已经形成三种方法、四种标注预算的完整单 seed 基线。最明确的结果是：**SetFit 在少样本条件下显著优于 TF-IDF，而 TF-IDF 在 full-data 条件下仍是最强且成本最低的基线。** 这说明对比式句向量方法确实适合当前 few-shot 场景，也为继续验证 SupCon 提供了合理依据。

与此同时，BERT 的三个少样本结果发生严重预测塌缩，尚不能作为可信的 SupCon 对照；单一随机种子、SetFit 日志缺失和训练预算不等价也限制了结论强度。下一步应先修复 BERT 与产物记录，完成多随机种子基线，再开展 `baseline → SupCon → SupCon + hard negatives` 消融。Version 1 已经支持提出 hard negative 的候选类别组，但还不足以宣称 SupCon 或 hard negative mining 已经有效。
