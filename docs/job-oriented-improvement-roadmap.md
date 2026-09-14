# Banking77-FinIntent-SupCon：项目完善与求职导向路线图

## 1. 核心判断

这个项目已经具备一份优秀 NLP/机器学习求职项目的基本骨架：

- 研究问题明确：金融客服细粒度意图分类；
- 同时覆盖全量数据与 5/10/20-shot 少样本场景；
- 包含 TF-IDF、BERT、SetFit 等不同类型的基线；
- 实现了监督对比学习（SupCon）和难负例加权；
- 规划了消融实验、混淆矩阵和特征可视化。

目前的主要问题不是方向不足，而是项目仍处于“实验进行中”的状态：核心 SupCon 结果尚未完成，实验可信度、工程质量和可部署性也没有形成完整闭环。

为了最大化求职价值，建议将项目升级为：

> 一个具有可靠实验结论、能够识别未知输入、支持安全回退，并可部署和监控的金融客服意图路由系统。

近期机器学习和 NLP 岗位通常同时重视数据处理、实验设计、误差分析、软件质量、部署和监控，而不仅是训练模型。可参考：

- [Apple Machine Learning Engineer - Speech](https://jobs.apple.com/en-us/details/200665627-0865/machine-learning-engineer-speech?team=MLAI)
- [Apple Software Engineer, Machine Learning & AI](https://jobs.apple.com/en-us/details/200605363/software-engineer-machine-learning-ai?team=SFTWR)
- [Microsoft 技术面试指南](https://careers.microsoft.com/v2/global/en/hiring-tips/technical-interviewing)

---

## 2. 第一优先级：建立可信的实验结论

这是项目用于求职展示之前最需要补齐的部分。

### 2.1 完成实验矩阵

当前应优先完成：

- SetFit 的 10-shot 和 20-shot 实验；
- SupCon 在 full、5-shot、10-shot、20-shot 下的实验；
- SupCon + Hard Negative 在相同设置下的实验；
- 所有方法的统一评估与误差分析。

项目的标题和核心贡献都围绕 SupCon 与难负例挖掘，因此在这些结果完成之前，项目最重要的技术主张还没有得到实证支持。

### 2.2 增加多随机种子实验

单次运行的结果可能受到以下因素影响：

- few-shot 样本抽取；
- 模型参数初始化；
- batch 顺序；
- dropout；
- GPU/MPS 的非确定性计算。

建议每个配置至少运行 3 个随机种子，条件允许时运行 5 个，并报告：

```text
Macro-F1 = mean ± standard deviation
```

最好同时保存每次独立运行的原始结果，避免只展示最佳结果。

### 2.3 划分验证集

当前流程没有独立验证集，模型训练完成后直接在测试集上评估。建议从原始训练集按类别分层划分训练集和验证集：

- 训练集：参数更新；
- 验证集：选择 epoch、学习率、温度系数和模型 checkpoint；
- 测试集：实验设计冻结后只进行最终评估。

few-shot 场景也应明确验证数据是否计入标注预算，并在报告中说明。

### 2.4 保证消融实验公平

为了证明提升来自 SupCon 或难负例模块，需要尽量统一：

- backbone；
- tokenizer 与最大序列长度；
- 数据划分；
- 随机种子；
- 最大训练步数或计算预算；
- checkpoint 选择方式；
- 分类头结构；
- 评估代码。

建议至少包含以下实验组：

| 实验组 | 作用 |
|---|---|
| Frozen BERT + Linear Probe | 判断原始表示质量 |
| BERT + Cross Entropy | 标准端到端微调基线 |
| SupCon + Linear Probe | 验证对比表征学习 |
| SupCon + Hard Negative + Linear Probe | 验证难负例的增量贡献 |
| Cross Entropy + λ·SupCon | 比较两阶段训练与联合训练 |

### 2.5 增加更完整的指标

除了 Macro-F1，建议增加：

- Accuracy；
- Micro-F1；
- per-class Precision、Recall、F1；
- 置信区间或显著性检验；
- 训练时间；
- 单条与批量推理延迟；
- 吞吐量；
- 峰值内存；
- 模型参数量和磁盘大小。

对主要方法可以使用 paired bootstrap，判断性能提升是否超出随机波动。

### 2.6 优先排查 BERT full 结果

当前 BERT full 的 Macro-F1 为 0.7964，明显低于 TF-IDF + LinearSVC 的 0.8924。传统模型超过深度模型并非不可能，但这个差距值得优先调查。

已有论文转述 BANKING77 原始 BERT full-data accuracy 约为 93.66%。Accuracy 与 Macro-F1 不能直接等同，但当前测试集每个类别均为 40 条样本，因此现有结果仍应作为异常信号进行排查。参考：[FinNLP 2023](https://aclanthology.org/2023.finnlp-1.7.pdf)。

建议检查：

- 学习率、epoch 和 warmup 是否合适；
- 是否应根据验证集保存最佳 checkpoint；
- 训练 loss 和验证指标是否正常收敛；
- label mapping 是否在所有阶段保持一致；
- 不同随机种子之间是否存在较大波动；
- MPS、CUDA 和 CPU 是否产生明显差异；
- 是否发生欠拟合、灾难性遗忘或梯度异常。

---

## 3. 当前实现中的具体缺口

### 3.1 分析流程不完整

`src/analysis/visualize.py` 的 `load_analysis` 默认要求 embedding 文件必须存在，但 BERT 当前只保存预测结果，没有保存 embedding。这会导致 BERT 被整个分析流程跳过，连本可生成的混淆矩阵也不会生成。

建议：

- 将 embedding 设为可选；
- 即使没有 embedding，也应生成预测相关图表；
- 为所有深度模型统一保存 encoder embedding；
- 为所有方法保存预测结果和样本级置信度。

### 3.2 SetFit 输出与其他方法不对齐

SetFit 当前只返回 Macro-F1，没有保存：

- per-class F1；
- 测试集预测；
- 置信度；
- embedding。

这会导致三种方法无法进行一致的混淆分析和样本级比较。

### 3.3 模型 checkpoint 没有形成可用产物

SupCon 训练结束后没有保存模型；BERT 也关闭了 checkpoint 保存。当前项目可以运行实验，但无法直接加载已训练模型进行演示、部署或复现预测。

建议保存：

- 模型权重；
- tokenizer；
- label map；
- 训练配置；
- 随机种子；
- 依赖版本；
- 验证集最优指标；
- 模型版本号或 commit hash。

### 3.4 随机性控制不完整

虽然 few-shot 数据抽样使用了固定 seed，但 SupCon 训练没有统一设置 PyTorch、NumPy 和 DataLoader 的随机性，因此结果不一定可复现。

建议实现统一的 `seed_everything`，并记录设备及确定性设置。

### 3.5 Hard Negative 仍局限于 batch 内

当前难负例权重完全由同一个 batch 内的余弦相似度决定。它可能有效，但还没有显式利用金融意图之间的真实混淆关系。

此外，应测试 hard-negative 权重是否需要阻断梯度，并使用更稳定的 log-sum-exp 实现，避免后续更换温度或混合精度时出现数值问题。

### 3.6 数据与文档一致性

当前数据检查得到：

- 训练集每类样本量为 35–187，约有 5.3 倍差异；
- 测试集每类恰好 40 条；
- train/test 存在 6 条标准化后完全相同且标签相同的文本。

建议：

- 把“严重长尾”调整为更严谨的“存在类别不均衡”；
- 在报告中披露重复数据；
- 增加一次去重敏感性实验；
- 保留官方划分结果，另行报告去重结果，不要偷偷改变 benchmark。

另外还存在以下小问题：

- TF-IDF 文件注释声称使用 word + character n-gram，实际只使用默认 word n-gram；
- README 中 `dev-plan.md` 链接位置错误，实际文件在 `docs/`；
- README 的目录树未完整展示 SupCon、analysis 和实验脚本；
- 缺少根目录项目许可证；
- 缺少自动化测试、持续集成和统一项目配置；
- `outputs/` 被全部忽略，公开仓库中看不到核心结果。

建议只忽略大型 checkpoint 和临时文件，将最终报告、小型 JSON 结果和关键图表纳入版本管理。

---

## 4. 最值得深挖的研究方向

## 4.1 Confusion-aware Hard Negative Mining

这是与现有项目结合最自然、最容易形成个人贡献的方向。

当前方法根据 batch 内相似度为负样本加权，可以进一步升级为：

1. 使用基线模型生成完整 confusion matrix；
2. 根据双向误判率建立 intent confusion graph；
3. 训练时优先将高混淆类别放入同一 batch；
4. 对高混淆类别对赋予更高损失权重；
5. 每隔若干 epoch 更新一次 confusion graph；
6. 对比随机负例、相似度负例和混淆图负例。

这样可以将项目贡献表述为：

> 从静态的 batch 内相似度加权，扩展到由历史误判证据驱动的动态难负例挖掘。

建议重点回答：

- 改进是否集中在高混淆 intent pair？
- 是否伤害原本容易分类的类别？
- 不同 hard-negative temperature 是否稳定？
- 动态混淆图是否优于一次性静态图？
- 增加的训练成本是多少？

难负例结合 SupCon 已有相关研究，因此不宜把“使用难负例”本身描述为全新算法。更合理的差异点是金融意图混淆图、动态采样策略以及完整的实证分析：

- [When Hard Negative Sampling Meets Supervised Contrastive Learning](https://arxiv.org/abs/2308.14893)
- [Supervised Contrastive Learning with Hard Negative Samples](https://arxiv.org/abs/2209.00078)

## 4.2 开放集意图识别与安全回退

真实客服系统不能假设每个输入都属于已有的 77 个类别。模型应该能够识别：

- 未知意图；
- 模糊输入；
- 多意图输入；
- 与银行业务无关的输入；
- 数据分布漂移后的输入。

可增加以下能力：

- Out-of-Scope/Unknown intent 检测；
- temperature scaling 等置信度校准；
- Top-3 intent 推荐；
- 高置信度自动路由；
- 低置信度转人工；
- OOS 数据与新意图漂移模拟；
- 人工审核反馈回流。

建议报告：

- AUROC；
- AUPR；
- FPR@95TPR；
- ECE；
- Coverage–Accuracy 曲线；
- 不同拒识阈值下的自动处理率和错误路由率。

这一方向能把项目从“benchmark 分类器”提升为“具备业务安全机制的路由系统”。

## 4.3 主动学习与标注成本

few-shot 实验只回答“给定固定数量样本后效果如何”，主动学习还可以回答“应该优先标注哪些样本”。

建议比较：

- Random Sampling；
- Entropy Sampling；
- Margin Sampling；
- Embedding Diversity；
- Uncertainty + Diversity；
- Confusion-aware Sampling。

最终输出标注量—Macro-F1 曲线，并计算：

- 达到目标性能需要多少标注；
- 相比随机采样节省多少标注；
- 哪些意图最需要人工标注；
- 模型在第几轮后出现收益递减。

这能把“少样本”与真实标注成本直接关联起来。

## 4.4 现代基线与高效训练

可以补充一到两个有代表性的现代基线，但不建议无目的堆模型：

- DeBERTa-v3；
- E5/BGE embedding + prototype/kNN；
- PEFT/LoRA；
- label-description zero-shot classification；
- 小型 LLM 作为低置信度 fallback。

SetFit 本身已经是一个重要的少样本高效基线。应重点比较：

- 模型效果；
- 训练时间；
- 参数量；
- 推理延迟；
- 显存或内存使用；
- 单位性能提升的计算成本。

参考：[Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)。

---

## 5. 工程化与部署闭环

如果目标岗位包含机器学习工程师、Applied AI Engineer 或 NLP Engineer，建议增加一个可运行的产品化闭环。

### 5.1 推理服务

提供 HTTP API，输入用户文本，返回：

```json
{
  "intent": "declined_card_payment",
  "confidence": 0.91,
  "top_k": [
    {"intent": "declined_card_payment", "score": 0.91},
    {"intent": "declined_transfer", "score": 0.06},
    {"intent": "cash_withdrawal", "score": 0.01}
  ],
  "route_to_human": false,
  "model_version": "v1.0.0"
}
```

至少支持：

- 单条预测；
- 批量预测；
- Top-K 结果；
- 置信度阈值；
- 未知意图回退；
- 模型健康检查。

### 5.2 可复现部署

建议增加：

- Dockerfile；
- 一键启动命令；
- 明确的模型下载或构建流程；
- 环境变量说明；
- 配置文件；
- 示例请求和响应。

### 5.3 性能测试

至少报告：

- CPU 单条推理 P50/P95 延迟；
- GPU 或 MPS 延迟；
- batch size 对吞吐量的影响；
- 模型加载时间；
- 峰值内存；
- 模型磁盘大小。

如果进一步做蒸馏、量化或 ONNX 导出，应同时报告性能变化和准确率损失。

### 5.4 监控与漂移

可以通过离线模拟展示：

- 输入长度变化；
- 类别分布变化；
- embedding drift；
- 低置信度比例变化；
- unknown intent 比例变化；
- 每类错误率变化。

项目不一定需要真正搭建大型监控平台，但需要说明监控指标、告警条件和人工处理流程。

### 5.5 自动测试与持续集成

建议覆盖：

- SupCon loss 数值和梯度测试；
- hard-negative 退化性质测试；
- balanced batch 正样本保证测试；
- label map 一致性测试；
- 数据划分无交叉测试；
- 推理 API schema 测试；
- 小样本 smoke test；
- README 中核心命令的可执行性检查。

---

## 6. README 与公开展示

招聘者通常只会先浏览项目首页，因此 README 应在几十秒内回答以下问题：

1. 解决了什么真实问题？
2. 为什么这个问题困难？
3. 做了哪些方法上的工作？
4. 最终效果如何？
5. 结果是否可靠？
6. 能否一键运行或查看 Demo？
7. 作者本人贡献了什么？

推荐的 README 结构：

```text
项目标题与一句话价值
最终结果表
系统架构图
核心方法
严格实验设置
误差分析与关键图表
在线或本地 Demo
快速开始
工程设计
局限与未来工作
```

首页应直接展示：

- 最重要的 Macro-F1 对比表；
- mean ± std；
- 一个最有代表性的混淆分析图；
- 一张系统架构图；
- 一个 API 或网页 Demo 截图；
- 可复现命令。

---

## 7. 推荐实施顺序

### 7.1 一周版本

1. 修复分析流程和文档问题；
2. 完成全部基础实验与 SupCon 消融；
3. 增加验证集和 3–5 个随机种子；
4. 排查 BERT full 结果异常；
5. 输出 mean ± std、混淆分析和关键图表；
6. 实现 confusion-aware hard negative 的最小版本；
7. 增加模型保存和基础推理 API；
8. 重写 README 首页。

### 7.2 两到三周版本

在上述基础上选择一个主方向深入：

- 开放集/OOS 意图检测；或
- 主动学习与标注成本优化。

然后补充：

- Docker；
- 自动测试与持续集成；
- 延迟、吞吐和内存 benchmark；
- 漂移监控模拟；
- 简单交互式 Demo。

不建议同时铺开过多模型和业务方向。一个深入且证据完整的扩展通常比多个浅层功能更有说服力。

---

## 8. 不同求职方向的包装重点

### 8.1 NLP/算法工程师

重点突出：

- SupCon 损失实现；
- 类平衡采样；
- confusion-aware hard negative；
- 严格消融实验；
- 多随机种子与显著性分析；
- 易混淆类别的误差分析。

### 8.2 Applied AI/NLP 应用工程师

重点突出：

- 金融客服意图路由；
- OOS 检测；
- 置信度校准；
- Top-K 建议；
- 低置信度转人工；
- 反馈数据回流。

### 8.3 机器学习工程师

重点突出：

- 可复现实验流水线；
- 配置与模型版本管理；
- 模型服务化；
- Docker 与自动测试；
- 延迟和吞吐 benchmark；
- 数据及模型漂移监控。

### 8.4 偏研究岗位

重点突出：

- 与相关工作的差异；
- 方法动机和数学定义；
- 公平对照；
- 多数据集或跨域验证；
- 稳定性和显著性；
- 失败案例与局限；
- 可复现实验附件。

---

## 9. 面试叙事框架

建议按照下面的顺序介绍项目：

1. **业务问题**：77 个金融意图高度相似，错误路由会产生业务成本；
2. **数据约束**：新意图和长尾意图难以获得大量专家标注；
3. **基线发现**：传统 TF-IDF 在全量数据下很强，SetFit 在 few-shot 下有优势；
4. **方法动机**：普通分类损失不能显式塑造类内和类间表征结构；
5. **核心方法**：SupCon + 类平衡 batch + confusion-aware hard negative；
6. **实验设计**：多随机种子、公平消融、显著性检验、样本级误差分析；
7. **工程闭环**：模型服务、置信度校准、未知意图回退和监控；
8. **局限与下一步**：跨域、多语言、意图持续新增和线上反馈。

面试时应特别准备以下问题：

- 为什么使用 Macro-F1？
- 为什么不用 triplet loss？
- SupCon 为什么需要同类正样本？
- batch 构造如何影响损失？
- hard negative 会不会带来 false negative？
- 为什么采用两阶段训练？联合训练表现如何？
- 如何证明提升不是随机波动？
- 为什么 TF-IDF 在 full-data 下很强？
- 如何处理未知意图和低置信度输入？
- 如果类别从 77 增加到 500，会发生什么？
- 如何在线监控模型失效？

---

## 10. 简历表述模板

在所有实验完成后，可以根据真实数据改写为：

> 构建面向 77 类金融客服意图的少样本分类系统，在 full、5/10/20-shot 设置下系统对比 TF-IDF、BERT、SetFit 与监督对比学习；设计基于类别混淆图的动态难负例采样，通过多随机种子消融和样本级误差分析验证其对高混淆意图的改善。

> 实现训练、评估、模型版本管理与推理服务闭环，支持 Top-K 预测、置信度校准、未知意图拒识和人工回退，并完成延迟、吞吐、内存及模型大小 benchmark。

完成实验后，应将上述模板中的描述替换为真实量化结果，例如：

- Macro-F1 提升多少个百分点；
- 高混淆类别平均 F1 提升多少；
- 达到相同性能节省多少标注；
- P95 推理延迟和吞吐量；
- OOS AUROC 与自动处理覆盖率；
- 模型压缩前后的大小和性能变化。

不要在简历中填写尚未完成或无法复现的数字。

---

## 11. 最终建议

这个项目最有价值的升级路线不是简单增加更多模型，也不是强行加入一个通用 RAG 或 Agent，而是形成三个相互支持的证据层：

1. **研究层**：SupCon 和 confusion-aware hard negative 是否真正改善高混淆意图；
2. **实验层**：结果是否经过公平对照、多随机种子和显著性验证；
3. **系统层**：模型是否可以部署、拒识未知输入、低置信度回退并接受监控。

完成这三个层次后，它会从一份课程实验升级为一份能够在面试中完整讲解 15–20 分钟、同时覆盖算法与工程能力的代表项目。
