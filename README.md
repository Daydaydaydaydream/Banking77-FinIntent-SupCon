# Banking77-FinIntent-SupCon

面向 BANKING77 的金融客服细粒度意图分类项目。项目先建立 TF-IDF + SVM、BERT 和 SetFit 基线，再重点验证监督对比学习（SupCon）与 hard negative mining 能否在 full、5-shot、10-shot 和 20-shot 设置下改善语义相近类别的区分。

评估以 Macro-F1 为主，同时使用 per-class F1 和 confusion matrix 检查收益是否真正集中在易混淆 intent pairs；可视化只作为辅助证据。

## 文档

- [项目介绍](docs/PROJECT.md)：研究问题、数据集、方法范围、实验设计与预期产出。
- [开发规划](docs/DEVELOPMENT_PLAN.md)：实施阶段、实验协议、验收标准与范围边界。
- [第一次基线实验初步报告](docs/FIRST_EXPERIMENT_REPORT.md)：首次运行结果、异常分析与下一轮解决方案。

## 当前状态

仓库已完成 BANKING77 EDA、可复现的训练/验证划分，以及 TF-IDF + SVM、BERT、SetFit 三组 baseline。TF-IDF 与 BERT 已完成四种数据预算的 `seed=42` 首次运行；其中 BERT 少样本结果出现预测塌缩，需要调整训练预算后重跑。SetFit 尚未生成完整指标。当前结果与处理建议见 [第一次基线实验初步报告](docs/FIRST_EXPERIMENT_REPORT.md)。

当前数据产物：

- `outputs/eda/eda_report.md`：EDA 结论与图表；
- `outputs/eda/banking77_eda.xlsx`：可筛选的数据分析工作簿；
- `outputs/datasets/train.csv`：9,000 条训练数据；
- `outputs/datasets/val.csv`：1,003 条验证数据；
- `outputs/datasets/split_manifest.json`：随机种子、源文件哈希和划分索引。

## 数据

本项目只使用 `data/banking_data/`：

- `train.csv`：10,003 条，77 类；
- `test.csv`：3,080 条，每类 40 条；
- `categories.json`：类别列表。

`data/` 下的 NLU++ 与 span extraction 文件来自上游数据仓库，不在当前实验范围内。

## 当前 TF-IDF 结果

固定 `seed=42` 的单次结果如下。它们用于验证流水线，不替代后续多随机种子的正式报告。

| Setting | Validation Macro-F1 | Test Macro-F1 |
|---|---:|---:|
| full | 0.8639 | 0.8879 |
| 5-shot | 0.5091 | 0.5519 |
| 10-shot | 0.6331 | 0.6649 |
| 20-shot | 0.7348 | 0.7567 |

## 安装

建议使用 Python 3.11 的独立环境：

```bash
conda create -n banking77 python=3.11 -y
conda activate banking77
pip install -r requirements.txt
```

本机已经缓存 `bert-base-uncased` 和 `sentence-transformers/all-MiniLM-L6-v2` 时，可以通过 `--local-files-only` 完全离线运行。

## 准备数据

```bash
PYTHONPATH=src python -m data.prepare_data
```

该命令固定使用 `seed=42`，生成 9,000 条训练数据、1,003 条验证数据，并保持官方 3,080 条测试数据不变。

## 运行 Baseline

三个 baseline 使用同一入口，支持 `full`、`5shot`、`10shot` 和 `20shot`：

```bash
# 最快，建议先用它检查完整流程
PYTHONPATH=src python -m run_baselines --method tfidf_svm --setting full

# BERT 微调；按验证集 Macro-F1 选择 checkpoint
PYTHONPATH=src python -m run_baselines --method bert --setting 5shot --local-files-only

# SetFit；few-shot 默认使用 20 次对比样本迭代
PYTHONPATH=src python -m run_baselines --method setfit --setting 5shot --local-files-only

# 运行全部方法与全部数据预算；耗时较长
PYTHONPATH=src python -m run_baselines --method all --setting all --local-files-only
```

常用选项：

```bash
# 指定随机种子、设备和 batch size
PYTHONPATH=src python -m run_baselines \
  --method bert --setting 10shot --seed 1 --device mps --batch-size 16

# 已存在 metrics.json 时默认跳过；需要重跑时显式覆盖
PYTHONPATH=src python -m run_baselines \
  --method tfidf_svm --setting full --overwrite

# 不保留模型权重，只保留配置、指标、预测和混淆矩阵
PYTHONPATH=src python -m run_baselines \
  --method setfit --setting 5shot --no-save-model
```

每次运行的结果写入：

```text
outputs/runs/<method>/<setting>/seed<seed>/
├── config.json
├── env.json
├── metrics.json
├── predictions.csv
├── confusion_matrix.csv
├── training_history.json   # 深度模型
└── model/                  # 默认保存，已被 gitignore
```

`metrics.json` 同时包含验证集和测试集的 Accuracy、Macro-F1、Micro-F1 与 per-class Precision/Recall/F1。TF-IDF 的 `top1_score` 是归一化 decision score，不是经过校准的概率。

## 验证代码

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## 可视化实验结果

结果可视化模块会自动发现包含 `metrics.json` 的完整运行，并输出总体指标对比、逐类 F1、归一化混淆矩阵、高频双向混淆类别对、预测类别分布和深度模型训练曲线。

```bash
# 汇总所有已完成实验，默认读取测试集结果
PYTHONPATH=src python -m analysis.visualize_results

# 只比较 full 设置下的 TF-IDF 与 BERT
PYTHONPATH=src python -m analysis.visualize_results \
  --method tfidf_svm --method bert \
  --setting full

# 查看验证集结果，并指定随机种子与输出目录
PYTHONPATH=src python -m analysis.visualize_results \
  --split validation --seed 42 \
  --output-dir outputs/figures/validation_results
```

默认输出到 `outputs/figures/model_results/`。其中 `test_metrics_summary.csv` 可用于复核跨运行指标，`top_confusions_*.csv` 保存每次运行最常见的双向混淆类别对，`visualization_manifest.json` 记录本次调用选择的运行和生成文件。
