# 第二阶段：CE / CE + SupCon / CE + SupCon + HN

## 实现与实验口径

新增 `run_controlled` 入口，独立保存在 `outputs/controlled_runs/`。三组均端到端更新 BERT 和同一个线性分类头；使用相同初始化种子、均值池化、投影头、类平衡采样、优化器和验证选择规则。

- CE：交叉熵。
- CE + SupCon：CE + 0.1 × SupCon。
- CE + SupCon + HN：相同联合损失，对 SupCon 分母中的异类项增加相似度权重。

共享协议见 `configs/controlled_v1.json`：每批每类 2 条、有效 batch 154，学习率 5e-5，固定 300 次更新，每 50 步验证，最大长度 64，SupCon 温度 0.1，HN 温度 0.5。不早停，完整模型按最高验证 Macro-F1 恢复；之后才计算测试指标。三组每轮使用相同的 seed+epoch 采样，既保证可复现，也避免每轮重复完全相同批次。

这些是预先声明的共享参数，不宣称已经验证集调优，也不宣称任何一组达到最佳能力。更新次数相同不等于计算成本相同，记录训练及验证总耗时。历史测试结果已经查看，新增对照不描述为从未接触测试集的确认性研究。

## HN 修复与历史结果失效

旧实现计算了加权 `log_prob`，但最终 `anchor_loss` 仍来自此前未加权的 `positive_log_prob`。因此旧 HN 权重没有参与优化，不能由旧表格推断 HN 无收益。

修复后：仅加权负样本，正样本保持权重 1；负样本权重按每行均值为 1 归一化并停止梯度，使用 log-sum-exp 计算分母。均匀权重严格退化为普通 SupCon，低温度避免直接指数溢出。回归测试检查损失和梯度均发生变化，而不仅检查是否有限。

历史 `outputs/runs/supcon_hn` 数值保持不动，标记为不可用于 HN 方法结论；正式新结果必须来自新入口。修复也影响后续重跑的两阶段 HN，不能把新旧实现混在同一统计中。

## 复现

先激活包含 requirements.txt 依赖的 Python 环境，并准备本地 BERT snapshot。

```bash
PYTHONPATH=src python -m data.prepare_data

# 三组真实 BERT 小规模流程验证：每组一步，验证/测试每类两条
# 输出到独立 smoke 子目录；这些分数不进入正式汇总
PYTHONPATH=src python -m run_controlled --smoke --no-save-model

# 正式矩阵：3 方法 × 4 设置 × 3 种子 = 36 组
PYTHONPATH=src python -m run_controlled --device cuda

# 或先完成一种训练预算，仍保持三组相同协议
PYTHONPATH=src python -m run_controlled --device cuda --setting 5shot

PYTHONPATH=src python -m unittest discover -s tests -v
```

默认读取本地 `models/bert-base-uncased`；可用 `--model-path` 指向其他本地 snapshot，禁止运行时自动下载。CPU 可用 `--device cpu --threads 4`。默认保存选中的编码器、任务头、tokenizer 和标签；`--no-save-model` 只省略权重。

固定协议及模型文件哈希记录在输出根目录 `protocol.json`。每组记录有序训练/验证/测试指纹、有效批次、实际步数、每步 CE/SupCon/总损失与梯度范数、验证记录、最佳步数，以及验证/测试逐条预测。完成的运行仅在协议、模型哈希和数据指纹匹配且必要产物存在时跳过；有失败记录或不完整输出的目录拒绝覆盖，排查后使用新的 `--output-dir`。失败返回非零退出码。

训练完成后自动生成 `runs.csv`、均值/样本标准差 `summary.csv`、相邻两组的配对差值 `paired_comparisons.json`。单种子标准差为空，不用零表示稳定；正式汇总排除 smoke。

## 当前验证状态

已实现三组联合训练、HN 修复和可追溯产物保存。71 项测试通过，包括 HN 损失/梯度生效、均匀权重退化、低温度数值稳定、固定更新次数、恢复完整最佳模型状态，以及真实小型 BERT 的保存加载与预测一致性。

三组本地 bert-base-uncased 的 CPU smoke 均完成，产物见 `outputs/controlled_runs/smoke/`：每组 1 次更新，5-shot、seed 42，验证/测试分别每类 2 条。三组数据/模型/协议指纹一致，验证及测试指标均已由逐条预测独立复算，通过断点跳过检查。单组训练及小规模验证约 8–10 秒；只是流程证据，不是收敛或性能证据。

**正式 36 组尚未运行。** 本机仅检测到 CPU，按小规模运行粗略估计需二十小时以上，实际随数据、验证及设备负载变化。已请求确定本机 CPU 或原 CUDA 环境；等待计算环境选择后执行正式矩阵。第二阶段目前为“代码与流程验证完成、正式实验待执行”，不得标记为完整实验完成。

## 2026-10-27 GPU 求职交付更新

目标环境已确定为 RTX 5060 Ti 16GB / WSL2。新增 `configs/controlled_gpu_v2.json`（BF16、SDPA、梯度检查点）和 `scripts/run_job_ready.sh`，默认先做 5-shot 的 9 组正式对照，完整 36 组为可选。旧 v1 配置和历史输出保持原样；GPU v2 使用新的输出根目录，不与旧运行混合汇总。

新增实现版本、运行环境一致性检查、完成产物复算、失败目录归档后整组重跑、模型文件校验、离线推理及自动交付报告。GPU 显存与吞吐通过目标机器记录，开发机的 CPU 验证不能代表 GPU 验证。

操作与失败恢复以 [WSL_RUNBOOK.md](WSL_RUNBOOK.md) 为准。上文“等待环境选择”为 10 月 12 日历史状态；目前为代码交付、等待目标机器正式运行。
