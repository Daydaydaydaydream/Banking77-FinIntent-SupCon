# 实验结果

本报告的每个数字都由 `outputs/runs/` 下的 `metrics.json` 重新汇总得到，没有任何硬编码值。所有结果为多随机种子的均值与标准差。

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

## 3. 逐步消融

以 BERT (baseline) 为对照，在相同 backbone、数据划分、批次构造与评估代码下逐级比较。「worst seed」是三种子里最差的那次配对差值；「consistent」表示三次全部为正。

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

单一种子上某个类别对「改善」，可能只是错误转移到了未检查的类别对上。因此只有在该类别对上**每个种子都改善**才计为改善。

| Setting | 组改善 | 三种子均改善的类别对 | 三种子均恶化的类别对 |
|---|---:|---:|---:|
| 5shot | 5/8 | 0/29 | 9/29 |
| 10shot | 3/8 | 1/29 | 14/29 |
| 20shot | 3/8 | 0/29 | 15/29 |
| full | 5/8 | 0/29 | 15/29 |


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

组改善 5/8；三种子均改善的类别对 0/29；三种子均恶化 9/29。


### 4.3 hard negatives 的同一检验（5-shot）

| 组 | BERT + SupCon | BERT + SupCon + HN | delta | 方向 |
|---|---:|---:|---:|---|
| `top_up_state` | 46.7 | 45.0 | -1.7 | 改善 |
| `card_delivery` | 22.7 | 22.7 | +0.0 | 恶化 |
| `balance_not_updated` | 5.3 | 5.3 | +0.0 | 恶化 |
| `declined_transaction` | 10.3 | 10.7 | +0.3 | 恶化 |
| `refund` | 13.3 | 13.7 | +0.3 | 恶化 |
| `virtual_card` | 24.0 | 25.0 | +1.0 | 恶化 |
| `exchange_rate` | 19.3 | 21.0 | +1.7 | 恶化 |
| `identity_verification` | 41.3 | 43.7 | +2.3 | 恶化 |

组改善 1/8；三种子均改善的类别对 0/29；三种子均恶化 21/29。


### 逐对双向误判变化（BERT → SupCon，5-shot）

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
