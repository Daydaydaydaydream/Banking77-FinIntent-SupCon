# 实验结果

> 历史 HN 结果不可用于评估难负例收益：复核发现旧实现计算了加权分母，却没有用它更新最终返回的损失，权重未参与优化。表中旧数值仅为历史记录，须使用修复后的实现重新运行。
本报告的统计由 `outputs/runs/` 下的 `metrics.json` 与 `confusion_matrix.csv` 重新汇总得到，主结果为多随机种子的均值与样本标准差；单种子分析单独注明。

## 1. 主结果表

测试集 Macro-F1，`mean ± std` over seeds [42, 43, 44]。⚠ 表示该运行被标记为类别塌缩，不可信。

| Method | 5-shot | 10-shot | 20-shot | full |
|---|---:|---:|---:|---:|
| TF-IDF + Linear SVM | 0.5494 ± 0.0040 | 0.6622 ± 0.0027 | 0.7599 ± 0.0031 | 0.8879 ± 0.0000 |
| SetFit | 0.7500 ± 0.0229 | 0.7978 ± 0.0105 | 0.8219 ± 0.0057 | 0.8768 ± 0.0049 |
| BERT (baseline) | 0.6968 ± 0.0111 | 0.8137 ± 0.0071 | 0.8629 ± 0.0057 | 0.9236 ± 0.0081 |
| BERT + SupCon | 0.6554 ± 0.0043 | 0.7954 ± 0.0061 | 0.8545 ± 0.0031 | 0.9299 ± 0.0048 |
| BERT + SupCon + HN | 0.6552 ± 0.0038 | 0.7951 ± 0.0044 | 0.8528 ± 0.0055 | 0.9276 ± 0.0023 |

## 2. 随机种子稳定性

单一种子的结果无法区分方法差异与划分噪声。下表给出每个 (方法, 设置) 的种子波动范围。

| method | 5shot | 10shot | 20shot | full |
|---|---|---|---|---|
| bert | 0.022 | 0.0142 | 0.0108 | 0.0151 |
| setfit | 0.0406 | 0.0204 | 0.0111 | 0.0098 |
| supcon | 0.0081 | 0.012 | 0.0056 | 0.0092 |
| supcon_hn | 0.0076 | 0.008 | 0.0111 | 0.0046 |
| tfidf_svm | 0.0071 | 0.0054 | 0.0062 | 0.0 |

## 3. 训练方案配对比较

以 BERT (baseline) 为对照，使用相同 BERT backbone、数据划分与评估口径，但训练阶段、采样和计算成本不同；这里比较训练方案，不能将差异全部归因于损失函数。「worst seed」是三种子里最差的那次配对差值；「consistent」表示三次全部为正。

| Comparison | Setting | paired seeds | mean delta | worst seed | consistent |
|---|---|---:|---:|---:|---|
| BERT + SupCon vs BERT (baseline) | 5shot | 3 | -0.0414 ± 0.0075 | -0.0468 | no |
| BERT + SupCon vs BERT (baseline) | 10shot | 3 | -0.0182 ± 0.0125 | -0.0326 | no |
| BERT + SupCon vs BERT (baseline) | 20shot | 3 | -0.0084 ± 0.0039 | -0.0118 | no |
| BERT + SupCon vs BERT (baseline) | full | 3 | +0.0063 ± 0.0123 | -0.0068 | no |
| BERT + SupCon + HN vs BERT (baseline) | 5shot | 3 | -0.0416 ± 0.0074 | -0.0478 | no |
| BERT + SupCon + HN vs BERT (baseline) | 10shot | 3 | -0.0186 ± 0.0113 | -0.0313 | no |
| BERT + SupCon + HN vs BERT (baseline) | 20shot | 3 | -0.0101 ± 0.0071 | -0.0177 | no |
| BERT + SupCon + HN vs BERT (baseline) | full | 3 | +0.0040 ± 0.0090 | -0.0055 | no |

## 4. 预声明易混淆类别对

以下类别组在训练开始前依据标签语义与业务分组预先声明，不使用测试集混淆矩阵来选择类别对——否则评估集会泄漏进方法选择。双向误判数按 a→b 与 b→a 分别统计后相加。

| 组 | 成员 |
|---|---|
| `identity_verification` | `verify_my_identity`, `why_verify_identity`, `verify_source_of_funds`, `unable_to_verify_identity` |
| `top_up_state` | `pending_top_up`, `top_up_failed`, `top_up_reverted`, `top_up_by_bank_transfer_charge` |
| `card_delivery` | `card_arrival`, `order_physical_card`, `card_delivery_estimate`, `card_not_working` |
| `declined_transaction` | `declined_card_payment`, `declined_transfer`, `declined_cash_withdrawal` |
| `balance_not_updated` | `balance_not_updated_after_bank_transfer`, `balance_not_updated_after_cheque_or_cash_deposit` |
| `exchange_rate` | `card_payment_wrong_exchange_rate`, `wrong_exchange_rate_for_cash_withdrawal`, `exchange_rate` |
| `refund` | `request_refund`, `Refund_not_showing_up` |
| `virtual_card` | `get_disposable_virtual_card`, `disposable_card_limits`, `virtual_card_not_working` |


### 4.1 跨 3 个种子的一致性（bert → supcon）

单一种子上某个类别对「改善」，可能只是错误转移到了未检查的类别对上。只有在该类别对上**每个种子的误判数都严格下降／上升**才计为一致改善／恶化；持平不计入。组改善按跨种子均值计算。

| Setting | 组改善 | 三种子均改善的类别对 | 三种子均恶化的类别对 |
|---|---:|---:|---:|
| 5shot | 5/8 | 0/29 | 1/29 |
| 10shot | 3/8 | 1/29 | 4/29 |
| 20shot | 3/8 | 0/29 | 1/29 |
| full | 5/8 | 0/29 | 0/29 |


### 4.2 逐组双向误判数变化（5-shot，三种子均值）

| 组 | BERT (baseline) | BERT + SupCon | delta | 方向 |
|---|---:|---:|---:|---|
| `identity_verification` | 45.3 | 41.3 | -4.0 | 改善 |
| `virtual_card` | 25.0 | 24.0 | -1.0 | 改善 |
| `exchange_rate` | 19.7 | 19.3 | -0.3 | 改善 |
| `balance_not_updated` | 5.7 | 5.3 | -0.3 | 改善 |
| `refund` | 13.7 | 13.3 | -0.3 | 改善 |
| `card_delivery` | 22.3 | 22.7 | +0.3 | 恶化 |
| `declined_transaction` | 9.7 | 10.3 | +0.7 | 恶化 |
| `top_up_state` | 39.0 | 46.7 | +7.7 | 恶化 |

组改善 5/8；三种子均改善的类别对 0/29；三种子均恶化 1/29。


### 4.3 hard negatives 的同一检验（5-shot）

| 组 | BERT + SupCon | BERT + SupCon + HN | delta | 方向 |
|---|---:|---:|---:|---|
| `top_up_state` | 46.7 | 45.0 | -1.7 | 改善 |
| `card_delivery` | 22.7 | 22.7 | +0.0 | 持平 |
| `balance_not_updated` | 5.3 | 5.3 | +0.0 | 持平 |
| `declined_transaction` | 10.3 | 10.7 | +0.3 | 恶化 |
| `refund` | 13.3 | 13.7 | +0.3 | 恶化 |
| `virtual_card` | 24.0 | 25.0 | +1.0 | 恶化 |
| `exchange_rate` | 19.3 | 21.0 | +1.7 | 恶化 |
| `identity_verification` | 41.3 | 43.7 | +2.3 | 恶化 |

组改善 1/8；三种子均改善的类别对 0/29；三种子均恶化 1/29。


### 逐对双向误判变化（BERT → SupCon，5-shot，seed 42，展示变化最大的 15 对）

| 组 | label A | label B | BERT | SupCon | delta | 改善 |
|---|---|---|---:|---:|---:|---|
| `identity_verification` | `verify_my_identity` | `why_verify_identity` | 24 | 13 | -11 | 是 |
| `identity_verification` | `unable_to_verify_identity` | `why_verify_identity` | 5 | 13 | +8 | 否 |
| `refund` | `Refund_not_showing_up` | `request_refund` | 17 | 12 | -5 | 是 |
| `card_delivery` | `card_arrival` | `card_delivery_estimate` | 20 | 23 | +3 | 否 |
| `card_delivery` | `card_arrival` | `card_not_working` | 0 | 3 | +3 | 否 |
| `card_delivery` | `card_delivery_estimate` | `order_physical_card` | 5 | 2 | -3 | 是 |
| `balance_not_updated` | `balance_not_updated_after_bank_transfer` | `balance_not_updated_after_cheque_or_cash_deposit` | 12 | 9 | -3 | 是 |
| `virtual_card` | `disposable_card_limits` | `get_disposable_virtual_card` | 23 | 26 | +3 | 否 |
| `identity_verification` | `unable_to_verify_identity` | `verify_my_identity` | 4 | 2 | -2 | 是 |
| `top_up_state` | `top_up_failed` | `top_up_reverted` | 3 | 5 | +2 | 否 |
| `card_delivery` | `card_arrival` | `order_physical_card` | 2 | 0 | -2 | 是 |
| `declined_transaction` | `declined_card_payment` | `declined_cash_withdrawal` | 6 | 4 | -2 | 是 |
| `exchange_rate` | `card_payment_wrong_exchange_rate` | `wrong_exchange_rate_for_cash_withdrawal` | 15 | 17 | +2 | 否 |
| `exchange_rate` | `exchange_rate` | `wrong_exchange_rate_for_cash_withdrawal` | 4 | 2 | -2 | 是 |
| `virtual_card` | `get_disposable_virtual_card` | `virtual_card_not_working` | 18 | 16 | -2 | 是 |


预声明类别对中有 12/29 组双向误判下降。


## 5. 产物完整性

- 已完成运行：60
- 标记为塌缩：0
- 塌缩运行：无

受控 5-shot 另有 9 组（见第 6 节），与本节的 60 组分别汇总，不混入同一张表。

## 6. 受控联合训练正式结果（RTX 5060 Ti，5-shot，9 组）

2026-10-27 在 RTX 5060 Ti 16GB / WSL2（CUDA 13.0、torch 2.14.0、BF16、SDPA、梯度检查点）上完成 5-shot × CE / CE + SupCon / CE + SupCon + HN × seeds 42/43/44，共 9 组。三组共享 BERT-base 编码器、类平衡采样（每批每类 2 条，有效 batch 154）、学习率 5e-5、固定 300 次更新、每 50 步验证、最大长度 64，按验证 Macro-F1 恢复最佳状态后才计算测试指标；只有损失函数不同。

### 6.1 效果与成本

| 方法 | Test Macro-F1 mean ± std | Validation Macro-F1 mean | 训练+周期验证秒数 | 测试条/秒 | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|
| CE | 0.6831 ± 0.0112 | 0.6608 | 83.9 | 3448.7 | 2.13 |
| CE + SupCon | 0.7013 ± 0.0050 | 0.6790 | 84.2 | 3442.0 | 2.14 |
| CE + SupCon + HN | 0.7019 ± 0.0034 | 0.6767 | 84.3 | 3452.7 | 2.14 |

`train_seconds` 含训练、周期验证和最佳状态 CPU 拷贝，不含模型加载、分词与最终保存；测试吞吐经过预热，含数据装载、前向与分数回传，不含分词。峰值显存是该次训练与评估中 PyTorch 分配的峰值，不等于系统总占用。

### 6.2 配对差值

| Comparison | Setting | seeds | mean delta | worst seed | 正向种子 |
|---|---|---:|---:|---:|---:|
| CE + SupCon − CE | 5shot | 3 | +0.0182 ± 0.0078 | +0.0095 | 3/3 |
| CE + SupCon + HN − CE + SupCon | 5shot | 3 | +0.0005 ± 0.0032 | −0.0032 | 2/3 |

对比损失的增量在三个种子下方向一致；HN 的增量在 seed 44 转负，不成立。三个种子只能说明当前划分与协议下的方向，不足以支持普遍性结论。

### 6.3 预声明易混淆类别对

沿用第 4 节的 29 个类别对，统计 CE → CE + SupCon 的双向误判数变化（三种子均值）：

- 没有任何类别对在三个种子上同时改善；2 对同时恶化：`top_up_failed ↔ top_up_reverted`（+2.33）、`Refund_not_showing_up ↔ request_refund`（+2.67）。
- 29 对合计误判数 +2.0 次，即 Macro-F1 增益没有集中在预声明的易混淆类别上。
- 组级合计（负值为误判减少）：`exchange_rate` −3.00、`balance_not_updated` −1.33、`top_up_state` −1.33；`virtual_card` 持平；`declined_transaction` +0.67、`card_delivery` +2.00、`identity_verification` +2.33、`refund` +2.67。

### 6.4 产物与局限

- 产物根目录 `outputs/job_ready_gpu_v2_retry/`：`protocol.json`（固定协议、环境、模型哈希）、`summary.csv`、`runs.csv`、`paired_comparisons.json`、`delivery/`（报告、效果/成本表、逐类变化、29 对混淆、真实错误案例、`selected_model.json`）。每组另有 `config.json`、`env.json`、`metrics.json`、`training_history.json`、逐条与验证集预测、混淆矩阵。
- 模型权重（约 5 GB）只保留在运行机器上，不进入 Git；仓库保留可复算指标的文本产物。
- 首次运行在验证分数一致性检查处失败：BF16 下并列最高分类分数时，导出与选模使用了不同并列规则。修复统一了并列规则，并用保留的 checkpoint 重导出，未重新训练、未改变选模规则、未放宽容差；正式产物位于带 `_retry` 的目录。
- 本节只覆盖 5-shot、3 个种子、9 组；与第 1 节的 60 组训练预算不同，两组数字不可逐行相减。
- 配置只按预声明协议与验证规则选择；测试集结果此前已被查看，本节不描述为从未接触测试集的确认性研究。
