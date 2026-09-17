# Banking77-FinIntent-SupCon

面向 BANKING77 的金融客服细粒度意图分类项目。项目先建立 TF-IDF + SVM、BERT 和 SetFit 基线，再重点验证监督对比学习（SupCon）与 hard negative mining 能否在 full、5-shot、10-shot 和 20-shot 设置下改善语义相近类别的区分。

评估以 Macro-F1 为主，同时使用 per-class F1 和 confusion matrix 检查收益是否真正集中在易混淆 intent pairs；可视化只作为辅助证据。

## 文档

- [项目介绍](docs/PROJECT.md)：研究问题、数据集、方法范围、实验设计与预期产出。
- [开发规划](docs/DEVELOPMENT_PLAN.md)：实施阶段、实验协议、验收标准与范围边界。

## 当前状态

仓库当前保留 BANKING77 原始数据与规划文档。实验代码、依赖清单和可核验结果需要按开发规划重新建立，因此暂不在首页展示未经当前产物验证的实验数字。

## 数据

本项目只使用 `data/banking_data/`：

- `train.csv`：10,003 条，77 类；
- `test.csv`：3,080 条，每类 40 条；
- `categories.json`：类别列表。

`data/` 下的 NLU++ 与 span extraction 文件来自上游数据仓库，不在当前实验范围内。
