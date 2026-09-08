# Banking77-FinIntent-SupCon 项目长期笔记

## 环境与版本（重要，别乱升级）
- **用户要求用 miniconda**，实际环境：`/Users/jingzhefeng/.conda/envs/banking77`（Python 3.13.15）
  - conda 本体：`/opt/homebrew/bin/conda`（25.1.1，osx-arm64，base 是 Python 3.12.9）
  - 激活：`conda activate banking77`；直接调用：`/Users/jingzhefeng/.conda/envs/banking77/bin/python`
- 钉死版本（setfit 1.1.3 强制三者对齐）：
  - `transformers==4.57.6`（**不能升 5.x**）
  - `sentence-transformers==3.4.1`（**不能升 6.x**）
  - `setfit==1.1.3`
  - `torch==2.14.0`、`scikit-learn==1.9.0`、`pandas==3.0.5`
- matplotlib 不在 requirements.txt 里，但 `analysis/visualize.py` 需要，装环境时手动补上

## 网络与模型下载（本机 huggingface.co 与 hf-mirror.com 均不通）
- 只有 pypi.org / 清华 / 阿里 PyPI 镜像可达；HF 全线 502 或超时
- 变通方案：**从 ModelScope 拉模型，按 HF 缓存布局落盘**，`from_pretrained("bert-base-uncased")`
  照常可用，无需改代码。已落盘：bert-base-uncased、sentence-transformers/all-MiniLM-L6-v2
- 跑实验时必须带 `HF_HUB_OFFLINE=1`，否则会卡在联网超时

## 关键 API 约定
- setfit 用 `Trainer`（非弃用的 `SetFitTrainer`）+ `SetFitModel.from_pretrained(backbone, use_differentiable_head=False)`
- 运行脚本：`cd src && python -m <module>`
- 结果统一报告测试集 Macro-F1，存 `outputs/results/results.json`（`run_baselines.py` 支持断点续跑）
- 设备：代码自动优先 MPS（Apple Silicon），已验证 `torch.backends.mps.is_available() == True`

## 基线设计
- 数据：full(10003) / 5-shot(385) / 10-shot(770) / 20-shot(1540)，seed=42，少样本用「shuffle+groupby.head(k)」
- 三基线：TF-IDF+LinearSVC、BERT 微调（bert-base-uncased，seq=64）、原生 SetFit（all-MiniLM-L6-v2，LR head）

## 已跑通的基线（2026-09-04，测试集 Macro-F1，MPS 约 1h20m）
| 方法 | full | 5shot | 10shot | 20shot |
|---|---|---|---|---|
| TF-IDF+LinearSVC | 0.8924 | 0.5126 | 0.6509 | 0.7597 |
| BERT 微调 | 0.8811 | 0.6308 | 0.7997 | 0.8692 |
| SetFit | 0.8707 | 0.7340 | 0.7982 | 0.8369 |

全量下 TF-IDF 略胜 BERT；少样本 SetFit 明显更强（5-shot 0.7340 vs 0.6308），20-shot 后 BERT 反超。

## WorkBuddy 沙箱 / macOS 签名坑（必读）
- **conda 创建环境时，沙箱会拦截 conda 的 post-link `codesign` 步骤**，导致 bin/python3.13 等二进制
  `codesign -v` 报 "invalid signature"，运行时被 macOS 直接 SIGKILL（exit 137）。表现为：
  `conda create`/`pip install` 退出码非 0（沙箱噪音），但环境其实已装好。
- **修复**：用 base conda 的 python 遍历环境目录，对 magic 为 Mach-O 且签名失效的文件执行
  `codesign -s - -f <file>` 重新 ad-hoc 签名（本环境共 28 个二进制需重签）。
- **运行/安装/下载必须关沙箱**：凡调用 `/Users/jingzhefeng/.conda/envs/banking77/bin/python`
  或 `conda`/`pip` 的命令，在 WorkBuddy 里都要带 `dangerouslyDisableSandbox=true`，
  否则即便在 workspace 内也会 SIGKILL。
- 教训：别被 "Exit Code: failed" 误导，先 `ls`+直接运行确认是否真失败。

## MPS 上的另一个坑：BERT 训练报 NotImplementedError
- 报错：`scaled_dot_product_attention for MPS does not support dropout`。
  BERT 在 train 模式下注意力会带 dropout，而 PyTorch 的 MPS 后端 SDPA 不支持 dropout。
- **修复**：加载编码器时指定 `attn_implementation="eager"`
  （`AutoModel.from_pretrained(..., attn_implementation="eager")` / `AutoModelForSequenceClassification`
  同理）。eager 注意力显式实现 dropout，在 Apple Silicon 上正常。
- 已改 `src/supcon/model.py` 与 `src/baselines/bert_ft.py`。eager 比 SDPA 慢，但正确。

## 运行命令（速查）
- 数据：`cd src && /Users/jingzhefeng/.conda/envs/banking77/bin/python -m data.prepare_data`
- 基线：`python -m run_baselines --baseline all --setting all`
- 消融：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m run_experiments --method all --setting all`
- 报告：`python -m analysis.report`（生成 `report.md`）；可视化：`python -m analysis.visualize`
- **所有涉及本 conda 环境的命令都加 `dangerouslyDisableSandbox=true`**

## 消融进度（SupCon，dev-plan 步骤 2）✅ 已完成
- `ablation_results.json`（2026-09-04 15:54）已含全部 12 项：
  baseline_bert / supcon / supcon_hn × full / 5shot / 10shot / 20shot
  （日志 `outputs/ablation_run.log`；首跑 `ablation.log` 因 MPS SDPA dropout 报错，改 eager 后重跑成功）
- 待办：`python -m analysis.report` 出 report.md + `analysis.visualize` 补图（尚未执行）

## BERT 全量收敛优化（2026-09-08）
- 原 full 配置 5 epochs / lr 2e-5 → 测试集 Macro-F1 0.8811（偏低，文献约 0.93）
- 优化：epochs 5→10、lr 2e-5→3e-5、warmup 0.06→0.1，新增 5% 分层验证集 +
  `load_best_model_at_end` 按验证 Macro-F1 选 best checkpoint（原为最后一个 epoch 直接评估）
- 结果：验证集最高 0.9422（epoch 5），**测试集 Macro-F1 0.8811 → 0.9262**（+4.5pt，达文献水平）
- 改动文件：`src/baselines/bert_ft.py`（仅 full 设置加 `val_ratio`，few-shot 设置不受影响）
- 注意：`run_experiments.py` 的 `baseline_bert` 复用 `bert_ft.run()`，ablation 表中
  `full / baseline_bert`（0.8894）为旧配置结果；如需公平对比，full 臂应重跑 baseline_bert

