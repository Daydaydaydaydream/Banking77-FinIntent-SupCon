# RTX 5060 Ti 16GB / WSL2 运行手册

更新：2026-10-27。目标是完成求职展示所需的 **5-shot × 3 方法 × 3 seeds = 9 组正式实验**，自动输出效果与成本、配对差值、真实错误案例，以及可离线推理的选中模型。PathPT 和新算法探索不在本次交付范围内。

开发机完成 CPU 测试与真实 BERT 小规模流程验证；未在 RTX 5060 Ti 上实测。GPU 兼容性、峰值显存及耗时以你的 preflight、smoke 和正式结果为准，不承诺具体训练时长。

## 1. 准备 WSL2 和项目

使用 WSL2 Ubuntu、Python 3.11，建议将项目放在 Linux 用户目录（如 `~/projects/Banking77-FinIntent-SupCon`）。通过 Git 同步本次代码提交，在 WSL 中切换到相应分支并拉取最新代码。

Windows 侧安装支持显卡的 NVIDIA 驱动；WSL 使用 Windows 提供的 GPU 驱动接口，不在 WSL 内安装 Linux NVIDIA 显卡驱动。参考 [NVIDIA WSL 官方说明](https://docs.nvidia.com/cuda/wsl-user-guide/)。在 WSL 中先确认：

```bash
nvidia-smi
python3.11 --version
```

安装缺少的 Python/venv 时，按所用 Ubuntu 版本的包源准备 Python 3.11；本项目不会替你修改系统或驱动。

### Windows 检出后的换行报错

若出现 `set: pipefail: invalid option name`，尤其报错前缀或行序异常，通常是 Windows 检出/编辑产生的 CRLF 回车。仓库已通过 `.gitattributes` 固定 shell 脚本为 LF；当前已有文件可在 WSL 项目根目录执行一次：

```bash
sed -i 's/\r$//' scripts/run_job_ready.sh
bash -n scripts/run_job_ready.sh
```

然后继续原来的步骤。运行时使用 `bash scripts/run_job_ready.sh prepare` 等命令，不要用 `sh`；该脚本使用 Bash 的数组参数和 `pipefail`。修复换行后不需要删除环境、模型或训练结果。

## 2. 安装隔离环境

以下命令在项目根目录执行。已有 Conda 环境也可使用，但保持 Python 3.11 并按下面固定核心版本安装。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-wsl.txt
python -m pip check
```

选用 PyTorch 2.10.0 的 CUDA 12.8 wheel；该组合见 [PyTorch 官方历史版本安装命令](https://pytorch.org/get-started/previous-versions/)。Blackwell 支持自 PyTorch 2.7 / CUDA 12.8 开始加入，不能沿用旧 CUDA wheel，见 [官方发布说明](https://pytorch.org/blog/pytorch-2-7/)。本项目不依赖 torchvision、torchaudio 或单独安装的 FlashAttention。

其余依赖不全部锁定：本次固定 torch/transformers/sentence-transformers/huggingface-hub 及已有 SetFit 版本；完整实际版本写入 `outputs/environment-wsl.txt`，每次运行另保存 `env.json`。正式矩阵期间不要升级依赖。

## 3. 数据、基础权重与 GPU 检查

```bash
bash scripts/run_job_ready.sh prepare
```

此命令：

1. 按现有固定规则准备训练池 9,000、验证 1,003、测试 3,080。
2. 若缺本地 BERT，则从 `google-bert/bert-base-uncased` 下载必要文件；先将 revision 解析为不可变 commit，写入 `models/bert-base-uncased/download_info.json`。
3. 检查 CUDA、BF16 的真实前向/反向算子及本地输入，保存 `outputs/preflight.json` 和实际依赖列表。

已有完整 `models/bert-base-uncased/` 则直接使用，不访问 Hub。也可以从另一台机器复制该目录；模型权重不提交到 Git。下载只发生在显式准备阶段，训练和推理均使用本地文件。

若已有模型目录不完整，程序拒绝覆盖；先保留/移走该目录，再重新准备。若首次下载失败，排查网络后重试。`nvidia-smi` 成功不等于 PyTorch wheel 能运行显卡，必须让 preflight 的真实 CUDA 运算通过。

可运行完整离线代码测试（缺少基础 BERT 时会跳过 1 项；缺少历史矩阵时，另跳过依赖它的 3 项检查）：

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/banking77-matplotlib python -m unittest discover -s tests -v
```

## 4. 先做三组 GPU smoke

```bash
bash scripts/run_job_ready.sh smoke
```

每组一步、5-shot seed42，验证/测试各每类两条；使用与正式实验相同的训练 batch。输出到 `outputs/job_ready_gpu_v2/smoke/`。它检查三组损失、梯度、显存和模型保存；指标不进入正式结论。三组都成功后再继续。

共享配置 `configs/controlled_gpu_v2.json`：

| 参数 | 值 |
|---|---|
| 编码器 | 本地 bert-base-uncased |
| 更新次数 | 300 |
| 周期验证 | 每 50 次更新 |
| 有效训练 batch | 77 类 × 每类 2 条 = 154 |
| 最大输入长度 | 64 tokens |
| 精度 | BF16；损失关键运算 FP32 |
| 注意力 | PyTorch SDPA |
| 梯度检查点 | 开启，use_reentrant=False |
| 学习率 | 5e-5 |
| SupCon 权重 / 温度 | 0.1 / 0.1 |
| HN 温度 | 0.5 |
| 选模 | 最高验证 Macro-F1，恢复完整模型 |

梯度检查点以额外计算换取显存；不需要安装外部注意力包。三组使用同一协议，历史 v1 输出保持原样。没有将大 batch 拆成梯度累积，因为这会改变 SupCon 看到的正负样本集合。

## 5. 运行 9 组正式实验

```bash
bash scripts/run_job_ready.sh train
```

按 setting → seed → method 顺序串行运行，保存每组最佳编码器、分类头、投影头、分词器和类别顺序。每 50 步打印验证进度；训练日志终端可以另存，原始历史也会写入各运行目录。

输出根目录：`outputs/job_ready_gpu_v2/`。建议预留至少 20GB 可用磁盘供环境、缓存与模型使用；9 组正式模型权重本身约 4GB，依赖及下载缓存会额外占用空间。

### 断点与失败

再次运行同一个 train 命令，会审计完整运行后跳过它们。**这是按运行跳过，不是从中断 optimizer step 恢复。** 不要同时启动两个进程写同一输出目录。

中断或失败的目录默认拒绝覆盖。确认没有旧训练进程后执行：

```bash
bash scripts/run_job_ready.sh train --retry-incomplete
```

程序将不完整目录移到 `_incomplete/<UTC时间>/` 保留证据，再从第 0 步重跑该组，完整组仍然跳过。smoke 中断后可用 `bash scripts/run_job_ready.sh smoke --retry-incomplete`。

协议、基础模型哈希、实现版本、运行设备或记录的环境变化时，拒绝混跑。重新实验使用新的根目录：

```bash
BANKING77_OUTPUT_DIR=outputs/job_ready_gpu_v2_retry bash scripts/run_job_ready.sh smoke
BANKING77_OUTPUT_DIR=outputs/job_ready_gpu_v2_retry bash scripts/run_job_ready.sh train
```

若 OOM，先关闭其他占 GPU 的程序。当前已启用 BF16 和检查点；如仍无法通过 smoke，保留 `failure.json` 并反馈，不要悄悄减少每类样本、改用 FP16 或只改变某一实验组。更改协议时另存配置并重新做全部对照。

### 训练完成后出现 Selected validation score mismatch

早期交付的导出逻辑使用非稳定排序，训练选模则使用 `argmax`。BF16 下两个类别的分数可能完全相同，导致保存预测与选模对并列类别的选择不一致。已统一为分数降序、并列时类别 id 升序。

若 9 组训练均已完成，但汇总报此错，在原 GPU 环境同步修复代码后执行：

```bash
bash scripts/run_job_ready.sh repair
```

该命令不训练、不重新选 checkpoint。它校验已保存权重、原运行环境和数据指纹，沿相同批大小、精度及预测路径重新推理。只有旧预测与新预测的差异全部可由**精确的最高分并列**解释，且修正后验证分数重现训练历史最佳值，才替换导出文件；否则停止并保留原文件，不能通过放宽校验容差掩盖问题。

替换前所有候选运行先在临时目录通过复核，旧预测、指标和混淆矩阵备份到各运行目录 `ranking_backup/<UTC时间>/`；模型、训练历史与配置保持原样。修复记录写入 metrics.json 的 ranking_repair，随后生成正式 delivery 报告。重复执行会校验并跳过已修复组。旧终端显示的测试分数属于修正前导出结果，最终分析使用修正后报告。

## 6. 自动报告与演示

train 成功后自动审计并生成报告。也可以单独执行：

```bash
bash scripts/run_job_ready.sh report
bash scripts/run_job_ready.sh demo
```

报告要求指定的 9 组全部完成，复算逐条预测、逐类指标和混淆矩阵，检查数据指纹、运行协议和最佳验证步数。缺组或证据不一致会失败，不生成新的完整报告。

| 文件 | 用途 |
|---|---|
| `delivery/REPORT.md` | 主结果、成本、配对差值、局限 |
| `delivery/runs.csv` / `summary.csv` | 逐运行结果与均值/样本标准差 |
| `delivery/paired_deltas.csv` | 同 seed 逐对 Macro-F1 差值 |
| `delivery/per_class_deltas.csv` | 逐类 F1 变化 |
| `delivery/confusion_pairs.csv` / `confusion_pair_summary.csv` | 29 对预定义混淆的逐种子与一致性 |
| `delivery/error_cases.json` | 改善、恶化和共同失败案例，附行号和来源 |
| `delivery/selected_model.json` | 按验证集选中的演示模型位置 |

演示选择规则：先按各方法三个种子的**平均验证 Macro-F1**选方法，再固定取 seed42；并列时顺序为 CE、SupCon、HN。它是一个演示 checkpoint，不代表多种子均值，不使用测试分数选方法或 seed。

自定义输入：

```bash
PYTHONPATH=src python -m predict \
  --selection outputs/job_ready_gpu_v2/delivery/selected_model.json \
  --text "Where is my card?" \
  --text "Why do I have to verify my identity?" \
  --device cuda --top-k 3
```

批量 CSV（UTF-8，包含 `text` 列）：

```bash
PYTHONPATH=src python -m predict \
  --selection outputs/job_ready_gpu_v2/delivery/selected_model.json \
  --input data/demo_inputs.csv --output outputs/demo_predictions.json \
  --device cuda --batch-size 32
```

也支持 `--run-dir outputs/job_ready_gpu_v2/ce/5shot/seed42`，以及 `--device cpu`。空输入会报错；长文本按训练长度截断，并返回 `truncated=true`。分数为未经校准的 softmax，不宣称可靠概率。运行目录和 `delivery` 保持相对布局时，可整体复制到其他机器离线推理。

## 7. 实验后如何用于求职

把 `delivery/`、运行 `config.json/env.json/metrics.json` 和终端异常记录带回进行分析。保留完整运行目录与选中模型；`predictions.csv` 是真实错误案例证据，不要只复制汇总分数。

本项目职责：少样本方案评估、训练排查和成本取舍。正式结果出来后再更新简历，涨分和负结果都如实报告。预先选择的三个种子证据有限；few-shot 另有 1,003 条验证标签；旧 HN 数字仍不可用于方法结论。

9 组通过报告验收后即可收尾，将主要精力转回医疗软件 Agent。其余设置共 27 组属于可选扩展，使用相同根目录和协议运行：

```bash
PYTHONPATH=src python -m run_controlled \
  --protocol configs/controlled_gpu_v2.json --output-dir outputs/job_ready_gpu_v2 \
  --device cuda --setting 10shot 20shot full --seed 42 43 44
PYTHONPATH=src python -m analysis.controlled_report \
  --root outputs/job_ready_gpu_v2 --setting 5shot 10shot 20shot full
```

扩展报告默认仍以第一个指定设置（上述为 5shot）选演示模型，避免跨训练标注预算选取混乱。
