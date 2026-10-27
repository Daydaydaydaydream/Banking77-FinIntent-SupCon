# 金融客服细粒度意图识别与少样本学习评估

> 历史 HN 结果不可用于评估难负例收益：复核发现旧实现计算了加权分母，却没有用它更新最终返回的损失，权重未参与优化。表中旧数值仅为历史记录，须使用修复后的实现重新运行。

面向 BANKING77 的 77 类英文金融客服意图分类项目，比较 full、5-shot、10-shot 和 20-shot 训练预算下的模型效果、稳定性与成本，并验证监督对比学习（SupCon）和难负例加权能否减少相似意图的误判。仓库名称保留 Banking77-FinIntent-SupCon，项目定位为可复现的 NLP 实验评估与求职展示项目。

评估以 Macro-F1 为主，同时使用 per-class F1 和 confusion matrix 检查收益是否真正集中在易混淆 intent pairs；可视化只作为辅助证据。

## RTX 5060 Ti / WSL2 求职收尾运行

已新增 GPU 配置、环境预检、离线推理和正式结果报告。完整安装、下载、失败恢复说明见 [WSL 运行手册](docs/WSL_RUNBOOK.md)。开发机已验证 CPU 流程；GPU 耗时、显存和正式效果待目标机器运行。

安装手册指定的环境并激活后，依次执行：

```bash
bash scripts/run_job_ready.sh prepare
bash scripts/run_job_ready.sh smoke
bash scripts/run_job_ready.sh train
bash scripts/run_job_ready.sh demo
```

默认正式范围为 5-shot × CE / CE+SupCon / CE+SupCon+HN × seeds 42/43/44，共 9 组，保存模型与逐条预测。报告位于 `outputs/job_ready_gpu_v2/delivery/REPORT.md`，演示模型按平均验证表现选方法、固定 seed42。完成后即可形成实验项目交付，其余 27 组为可选扩展。

## 文档

- [项目介绍](docs/PROJECT.md)：研究问题、数据集、方法范围、实验设计与预期产出。
- [求职项目概览](docs/PROJECT_OVERVIEW.md)：项目定位、已有证据和英文介绍。
- [本次开发验证](docs/DELIVERY_VALIDATION.md)：已验证范围与待 GPU 验证项。
- [2026-10-26 状态检查](docs/STATUS_2026-10-26.md)：代码、测试、实验产物及待完成项的本次核验。
- [后续开发规划](docs/DEVELOPMENT_PLAN.md)：受控实验、结果分析、求职交付与验收标准。
- [简历与面试说明](docs/CAREER_PROJECT.md)：当前可用表述、数字边界与讲解顺序。
- [当前实验结果](docs/RESULTS.md)：三种子统计与混淆分析。
- [Version 1 基线实验报告](docs/FIRST_EXPERIMENT_REPORT.md)：12 组基线结果、模型行为、局限与下一轮方案。

## 当前状态

基线已修复并重跑完成。少样本塌缩的根因是 float16 溢出，而非训练预算不足：仅把精度改为 bfloat16，5-shot 验证集 Macro-F1 就从 0.0077 升至 0.6159，预测类别数从 26 恢复到 77。

当前 `seed` 为 42/43/44 的三种子结果，测试集 Macro-F1（`mean ± std`）：

| Method | 5-shot | 10-shot | 20-shot | full |
|---|---:|---:|---:|---:|
| TF-IDF + Linear SVM | 0.5494 ± 0.0040 | 0.6622 ± 0.0027 | 0.7599 ± 0.0031 | 0.8879 ± 0.0000 |
| SetFit | **0.7500** ± 0.0229 | 0.7978 ± 0.0105 | 0.8219 ± 0.0057 | 0.8768 ± 0.0049 |
| BERT | 0.6968 ± 0.0111 | **0.8137** ± 0.0071 | **0.8629** ± 0.0057 | **0.9236** ± 0.0081 |
| BERT + SupCon | 0.6554 ± 0.0043 | 0.7954 ± 0.0061 | 0.8545 ± 0.0031 | **0.9299** ± 0.0048 |
| BERT + SupCon + HN | 0.6552 ± 0.0038 | 0.7951 ± 0.0044 | 0.8528 ± 0.0055 | 0.9276 ± 0.0023 |

超参数只在验证集上选择，冻结结果见 [`configs/bert_frozen.json`](configs/bert_frozen.json)。测试集不参与任何配置决策。

早期报告中的两组结论已被推翻：BERT 少样本塌缩使「SetFit 是最强少样本基线」不成立，full 设置下 BERT 也反超 TF-IDF，历史对照见 [`docs/FIRST_EXPERIMENT_REPORT.md`](docs/FIRST_EXPERIMENT_REPORT.md)。

包含基线、SupCon 和 SupCon+HN 的完整矩阵已完成 60 组运行，结论见下文「当前结论」。每次运行的配置、环境、指标、逐类指标与混淆矩阵位于 `outputs/runs/<method>/<setting>/seed<seed>/`，聚合表位于 `outputs/results/`。

## 当前结论

**当前两阶段 SupCon 没有显示跨三个种子一致的增益；旧 HN 实验因实现缺陷不能作方法结论。** 结论仅限当前实现与配置，不能外推为监督对比学习普遍无效。

BERT 为全参数微调，SupCon 为对比训练后冻结编码器、训练分类头；采样和计算成本也不同。因此这是训练方案比较，尚未隔离损失函数的独立贡献。5/10/20-shot 仅计算训练样本，另使用 1,003 条有标签验证数据。

三种子逐对配对差值（同一划分、同一随机种子下比较）：

| 对照 | 5-shot | 10-shot | 20-shot | full |
|---|---:|---:|---:|---:|
| SupCon − BERT | −0.0414 ± 0.0075 | −0.0182 ± 0.0125 | −0.0084 ± 0.0039 | +0.0063 ± 0.0123 |
| SupCon+HN − SupCon | −0.0002 ± 0.0010 | −0.0003 ± 0.0021 | −0.0017 ± 0.0036 | −0.0023 ± 0.0033 |

SupCon 在三个少样本设置上一致劣于 BERT，仅在 full 上以不足 1 分领先且最差种子仍为负。旧 HN 数值差异不代表有效的 HN 消融。

预声明的29 个易混淆类别对在三个种子上的一致性：

| Setting | 组改善 | 三种子均改善的类别对 | 三种子均恶化的类别对 |
|---|---:|---:|---:|
| 5-shot | 5/8 | 0/29 | 1/29 |
| 10-shot | 3/8 | 1/29 | 4/29 |
| 20-shot | 3/8 | 0/29 | 1/29 |
| full | 5/8 | 0/29 | 0/29 |

单一种子上 `verify_my_identity ↔ why_verify_identity` 从 24 降到 13 看似改善，但跨三个种子不复现，且 `top_up_state` 组从 39.0 升到 46.7。类别对在训练前依据标签语义声明，不存在选择性偏差。

对比损失下降及 full 设置的表现说明训练产生了变化，但不能单凭这些证据确定少样本退化的原因。训练阶段、编码器冻结方式、采样与预算差异均需通过受控实验进一步区分。当前三个种子中 SupCon 的指标波动较小，但这不等同于效果更好或普遍更稳定。

完整数据见 [`docs/RESULTS.md`](docs/RESULTS.md)，所有数字由 `outputs/runs/` 下的 `metrics.json` 重新生成。

## 产物复核

60 组配置和混淆矩阵复算通过；历史逐条预测及训练数据指纹缺失，无法完整追溯。重建划分无 train/validation 归一化文本重叠，但与官方 test 分别有 19/6 条唯一归一化文本交集。详见 [审计报告](docs/ARTIFACT_AUDIT.md)。

## 受控联合训练

已新增独立入口 `python -m run_controlled`，三组共享训练协议，历史运行不会被覆盖。完整命令、HN 修复说明与运行状态见 [受控实验说明](docs/CONTROLLED_EXPERIMENTS.md)。

## 后续规划

1. 复核运行产物与当前文档，统一训练预算、验证数据和方案比较口径。
2. CE / CE + SupCon / CE + SupCon + HN 统一流程已实现并通过三组 smoke，待执行正式 36 组实验。
3. 补齐效果与成本表、真实误判案例和训练异常排查说明。
4. 完成一页项目概览、简历描述与最小复现验证；按算法工程岗位需求补推理演示。
5. 独立探索 PathPT 启发的类别语义提示：先做同 backbone 的三组 5-shot 对照，当前尚未实现或训练。

详细范围、顺序和验收见 [后续开发规划](docs/DEVELOPMENT_PLAN.md)。各项完成状态以开发规划为准；正式受控实验结果尚未生成。

## 复现命令

```bash
# 1. 准备数据（固定 seed=42 的划分）
PYTHONPATH=src python -m data.prepare_data

# 2. 验证集超参搜索，冻结配置
PYTHONPATH=src python -m analysis.tune_bert --setting all --seed 42 --freeze

# 3. 基线矩阵（3 方法 × 4 设置 × 3 种子）
PYTHONPATH=src python -m run_baselines \
  --method tfidf_svm bert setfit --setting all --seed 42 43 44 \
  --config configs/bert_frozen.json --local-files-only --no-save-model

# 4. 消融（SupCon 与 SupCon+HN）
PYTHONPATH=src python -m run_baselines \
  --method supcon supcon_hn --setting all --seed 42 43 44 \
  --local-files-only --no-save-model --precision bf16 --supcon-epochs 10 \
  --hn-temperature 0.5

# 5. 汇总与报告
PYTHONPATH=src python -m analysis.aggregate --setting all
PYTHONPATH=src python -m analysis.confusion_pairs \
  --setting 5shot --seed 42 43 44 --compare bert supcon
PYTHONPATH=src python -m analysis.report --output docs/RESULTS.md

# 6. 测试
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 数据

本项目只使用 `data/banking_data/`：

- `train.csv`：10,003 条，77 类；
- `test.csv`：3,080 条，每类 40 条；
- `categories.json`：类别列表。

`data/` 下的 NLU++ 与 span extraction 文件来自上游数据仓库，不在当前实验范围内。

## Version 1 结果摘要

固定 `seed=42` 的 Test Macro-F1 如下。它们用于验证流水线和发现问题，不替代后续多随机种子的正式报告。

| Method | 5-shot | 10-shot | 20-shot | full |
|---|---:|---:|---:|---:|
| TF-IDF + SVM | 0.5519 | 0.6649 | 0.7567 | **0.8879** |
| BERT | 0.0077 | 0.0351 | 0.2676 | 0.8644 |
| SetFit | **0.7557** | **0.8062** | **0.8325** | 0.8778 |

BERT 的 few-shot 数值来自发生预测塌缩的异常运行，不应视为可信的模型能力比较。

## 历史结果（已被取代）

上表是 Version 1 的单种子结果，保留用于说明问题是如何被发现的。**这些数字不应再用于任何结论**：

- BERT 的三个 few-shot 数值源于 float16 梯度溢出导致的类别塌缩，已在当前版本修复；
- 当时的结论「SetFit 是最强少样本基线」建立在塌缩的 BERT 之上，修复后不成立；
- 当时的 full 设置结论「TF-IDF 胜过 Transformer」同样源于欠调的 BERT（验证集 0.8527），修复后为 0.9088。

当前有效结果见上文「当前状态」。

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

BERT few-shot 默认至少训练 300 optimizer steps，每 50 steps 在验证集上评估，验证 Macro-F1 连续无改善时早停，并以 1,000 steps 作为上限。`full` 仍使用 epoch 级训练与早停。这些阈值可以通过 `--min-steps`、`--max-steps`、`--eval-steps` 和 `--early-stopping-patience` 调整，但只应依据验证集选择。

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
