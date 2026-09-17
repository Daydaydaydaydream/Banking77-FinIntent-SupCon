# Banking77-FinIntent-SupCon

本项目针对金融客服意图**细粒度语义重叠、长尾样本稀缺**的难题，基于 BANKING77 数据集，融合监督对比学习（SupCon）与难负例挖掘优化特征表征，通过多基线对照与消融实验，提升少样本场景下金融细粒度意图分类的精度与鲁棒性。

详细方案见 [docs/dev-plan.md](docs/dev-plan.md)，完整项目文档见 [docs/PROJECT.md](docs/PROJECT.md) 后续实验规划见 [docs/experiment-plan.md](docs/experiment-plan.md)。

## 目录结构

```
.
├── data/
│   └── banking_data/          # BANKING77 原始数据 (train.csv / test.csv / categories.json)
├── docs/
│   ├── PROJECT.md             # 项目总体文档
│   ├── dev-plan.md            # 开发计划
│   └── experiment-plan.md     # 阶段 A–H 实验规划与判定门
├── src/
│   ├── config.py              # 路径、随机种子、数据集常量与阶段 A 划分预算
│   ├── data/prepare_data.py   # 分层 train/val 划分 + 少样本抽样 + --dedup 变体
│   ├── utils/
│   │   ├── seed.py            # 统一随机性控制（Python / NumPy / Torch / MPS）
│   │   ├── artifacts.py       # RunArtifacts：四类方法统一的产物契约
│   │   ├── data.py            # 数据加载（支持 seed 与 dedup 变体）
│   │   └── metrics.py         # Macro-F1 / per-class P-R-F1 / 分层 paired bootstrap
│   ├── baselines/
│   │   ├── tfidf_svm.py       # 基线 1：TF-IDF + Linear SVM
│   │   ├── bert_ft.py         # 基线 2：bert-base-uncased 微调（验证集选点）
│   │   └── setfit_baseline.py # 基线 3：原生 SetFit
│   ├── supcon/
│   │   ├── losses.py          # SupCon 损失 + 难负例相似度加权
│   │   ├── model.py           # BERT + 投影头 + 分类头
│   │   └── train.py           # 两阶段训练，阶段 2 按验证集选点
│   ├── analysis/
│   │   ├── visualize.py       # 混淆矩阵 / 置信度分布 / t-SNE（embedding 可选）
│   │   └── report.py          # 聚合多种子结果，生成 report.md
│   ├── run_experiments.py     # 主编排器：method × setting × seed
│   └── run_baselines.py       # 三条外部基线的便捷入口
├── outputs/                   # 生成的中间数据与结果
│   └── runs/<method>/<setting>/seed<k>/   # 统一产物目录（见下）
└── requirements.txt
```

## 实验契约（阶段 A）

**数据划分**：从官方 10,003 条训练集中**分层切出**训练池 9,000 与验证集 1,003；官方测试集 3,080 条冻结，仅用于最终评估。少样本的 k 条标注**只从训练池抽取**，验证集不计入标注预算。训练/验证划分对全部种子固定，仅少样本抽样随种子变化。

**模型选择**：BERT 与 SupCon 均以**验证集 Macro-F1** 选取 checkpoint，不使用测试集选点。

**产物目录**：每个 `(方法, 设置, 种子)` 组合写入同一结构，`metrics.json` 与 `predictions.csv` 纳入版本管理，权重与 embedding 不纳入。

```
outputs/runs/<method>/<setting>/seed<k>/
├── metrics.json        # macro_f1, micro_f1, accuracy, per-class P/R/F1
├── predictions.csv     # y_true, y_pred, top1~top3, confidence
├── embeddings.npy      # 测试集表示（可选；TF-IDF 存稀疏 .npz）
├── config.json         # 全部超参数 + method/setting/seed
├── env.json            # 依赖版本、设备、确定性状态
└── model/              # 权重与 tokenizer（gitignore）
```

## 环境

- Python 3.11（建议使用项目隔离的 conda 环境）
- Apple Silicon（MPS）或 CUDA；无 GPU 时回退 CPU

```bash
pip install -r requirements.txt
```

> **版本约束**：`setfit 1.1.3` 依赖 `transformers<5.0` 与 `sentence-transformers 3.x`，
> 升级前需同步升级 setfit，否则会出现 `default_logdir` 等导入错误。

> **国内网络**：若模型未缓存，运行前可设置 `export HF_ENDPOINT=https://hf-mirror.com`；
> 已在本地缓存的模型请用 `HF_HUB_OFFLINE=1` 运行以避免联网超时。

## 运行

```bash
cd src

# 1. 构建数据集（分层 train/val + 多种子少样本抽样，一次性）
python -m data.prepare_data
python -m data.prepare_data --dedup          # 可选的去重敏感性变体

# 2. 运行完整矩阵（自动跳过已完成项，可断点续跑）
python -m run_experiments --method all --setting all

# 3. 只跑部分：指定方法 / 设置 / 种子
python -m run_experiments --method supcon_hn --setting 5shot --seed 42
python -m run_baselines --baseline bert_ft --setting full    # 仅三条外部基线

# 4. 出图与报告
python -m analysis.visualize
python -m analysis.report
```

所有实验统一报告测试集 **Macro-F1**（多种子时给出 mean ± std），结果写入
`outputs/runs/**/metrics.json`，汇总报告写入 `report.md`。

## 数据集设置

| 数据设置 | 训练样本数 | 说明 |
|---|---|---|
| full | 9,000 | 分层切分后的训练池 |
| 5-shot | 385 | 每类 5 条标注 |
| 10-shot | 770 | 每类 10 条标注 |
| 20-shot | 1,540 | 每类 20 条标注 |

验证集固定 1,003 条，测试集固定 3,080 条。

## 方法

| 方法 | 说明 |
|---|---|
| TF-IDF + LinearSVC | 词面特征基线，训练最快 |
| BERT 微调 | bert-base-uncased 端到端微调，验证集选点 |
| SetFit | MiniLM 对比学习 + 逻辑回归头 |
| SupCon | 两阶段：对比学习预训练 + 冻结编码器训练线性头 |
| SupCon + 难负例 | 在上述损失上叠加相似度加权，重点推开易混淆类别 |
