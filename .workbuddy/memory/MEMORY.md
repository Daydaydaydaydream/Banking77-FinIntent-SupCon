# Banking77-FinIntent-SupCon 项目长期笔记

## 环境与版本（重要，别乱升级）
- Python venv：`/Users/fengjingzhe/.workbuddy/binaries/python/envs/default`
- 钉死版本（setfit 1.1.3 强制三者对齐）：
  - `transformers==4.57.6`（**不能升 5.x**）
  - `sentence-transformers==3.4.1`（**不能升 6.x**）
  - `setfit==1.1.3`
  - `torch==2.14.0`、`scikit-learn==1.9.0`、`pandas==3.0.5`
- 模型下载必须 `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 直连不通）

## 关键 API 约定
- setfit 用 `Trainer`（非弃用的 `SetFitTrainer`）+ `SetFitModel.from_pretrained(backbone, use_differentiable_head=False)`
- 运行脚本：`cd src && python -m <module>`
- 结果统一报告测试集 Macro-F1，存 `outputs/results/results.json`（`run_baselines.py` 支持断点续跑）

## 基线设计
- 数据：full(10003) / 5-shot(385) / 10-shot(770) / 20-shot(1540)，seed=42，少样本用「shuffle+groupby.head(k)」
- 三基线：TF-IDF+LinearSVC、BERT 微调（bert-base-uncased，seq=64）、原生 SetFit（all-MiniLM-L6-v2，LR head）
- 下一步：SupCon + 难负例挖掘（dev-plan 步骤 2）
