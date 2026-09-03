# Banking77-FinIntent-SupCon

本项目针对金融客服意图**细粒度语义重叠、长尾样本稀缺**的难题，基于 BANKING77 数据集，融合监督对比学习（SupCon）与难负例挖掘优化特征表征，通过多基线对照与消融实验，提升少样本场景下金融细粒度意图分类的精度与鲁棒性。

详细方案见 [dev-plan.md](dev-plan.md)，完整项目文档见 [docs/PROJECT.md](docs/PROJECT.md)。

## 目录结构

```
.
├── data/
│   └── banking_data/          # BANKING77 原始数据 (train.csv / test.csv / categories.json)
├── src/
│   ├── config.py              # 路径、随机种子、数据集常量
│   ├── data/prepare_data.py   # 构建全量 + 5/10/20-shot 少样本数据集
│   ├── utils/data.py          # 数据加载
│   ├── utils/metrics.py       # Macro-F1 / per-class F1 / 混淆矩阵
│   ├── baselines/
│   │   ├── tfidf_svm.py       # 基线 1：TF-IDF + Linear SVM
│   │   ├── bert_ft.py         # 基线 2：bert-base-uncased 微调
│   │   └── setfit_baseline.py # 基线 3：原生 SetFit
│   └── run_baselines.py       # 编排入口，汇总 Macro-F1
├── outputs/                   # 生成的中间数据与结果（gitignore）
├── requirements.txt
└── dev-plan.md
```

## 环境

- Python 3.13（建议使用项目隔离的 venv）
- Apple Silicon（MPS）或 CUDA；无 GPU 时回退 CPU

```bash
pip install -r requirements.txt
```

> **版本约束**：`setfit 1.1.3` 依赖 `transformers<5.0` 与 `sentence-transformers 3.x`，
> 升级前需同步升级 setfit，否则会出现 `default_logdir` 等导入错误。

> **国内网络**：模型下载走 HuggingFace 镜像，运行前设置环境变量
> `export HF_ENDPOINT=https://hf-mirror.com`。

## 运行

```bash
cd src

# 1. 构建全量 + 5/10/20-shot 数据集（一次性）
python -m data.prepare_data

# 2. 运行全部基线（自动跳过已完成项，可断点续跑）
python -m run_baselines --baseline all --setting all

# 3. 只跑某个基线 / 某个数据设置
python -m run_baselines --baseline bert_ft --setting 5shot
python -m baselines.tfidf_svm full      # 单独运行
```

所有实验统一报告测试集 **Macro-F1**，结果写入 `outputs/results/results.json`。

## 基线设置

| 数据设置 | 训练样本数 | 说明 |
|---|---|---|
| full | 10,003 | 原始全量训练集 |
| 5-shot | 385 | 每类 5 条标注 |
| 10-shot | 770 | 每类 10 条标注 |
| 20-shot | 1,540 | 每类 20 条标注 |

三组基线：TF-IDF + Linear SVM、BERT 微调（bert-base-uncased）、原生 SetFit（all-MiniLM-L6-v2）。
