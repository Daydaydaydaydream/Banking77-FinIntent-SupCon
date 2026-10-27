# 金融客服细粒度意图识别：求职项目概览

定位：Applied AI Engineer 求职的模型评估与问题排查辅助作品。医疗软件 Agent 展示复杂系统集成，本项目展示如何比较模型、识别异常并判断额外复杂度是否值得采用。

## 问题与方法

BANKING77 包含 77 类英文银行客服意图，相近类别存在细粒度语义差别。以 5/10/20-shot 和 full 比较 TF-IDF + SVM、BERT、SetFit 和两阶段 SupCon；以 Macro-F1 为主，结合逐类指标和 29 对预定义易混淆类别进行分析。

历史 60 组运行已完成并通过指标内部一致性复核。其中旧 HN 实现未将权重用于最终损失，其 12 组结果不支持方法收益结论。历史逐条预测及训练数据指纹缺失的限制保持披露。

## 已有可信发现

- 历史多种子结果中，5-shot SetFit 平均测试 Macro-F1 为 0.7500，full BERT 为 0.9236。
- 两阶段 SupCon 未显示跨三个种子一致的收益，训练阶段和冻结策略不同，不能把差异全部归因于损失。
- 早期 BERT 少样本塌缩经复核与 float16 溢出相关；修复训练精度的收益不称为算法创新。

数字与边界详见 [README](../README.md)、[历史审计](ARTIFACT_AUDIT.md)。本项目没有生产部署或真实银行业务收益证据。

## 本次交付与待运行项

已实现三组端到端受控训练、GPU 预检、模型及预测留存、严格产物复核、自动效果/成本报告，以及离线单条/批量 Top-k 推理。正式目标为 RTX 5060 Ti 16GB / WSL2 下的 5-shot × 三方法 × 三 seeds，共 9 组。

正式受控效果、GPU 性能和演示模型尚待用户运行产生；不预填涨分。按 [WSL 运行手册](WSL_RUNBOOK.md) 完成 prepare → smoke → train → demo。其余 27 组为可选扩展，PathPT 与新算法不是此次收尾前置条件。

## 面试讲解顺序

1. 为什么 77 类相近意图适合用 Macro-F1 和类别对分析。
2. 如何发现训练异常、验证损失和梯度确实生效。
3. 为什么旧训练方案比较不足以隔离 SupCon/HN 的贡献。
4. 新受控实验如何统一数据、采样、模型和更新预算。
5. 如何根据验证表现选择模型，保存并离线重载，同时说明效果、成本及局限。

介绍自己的实际工作时区分独立完成、协作完成和工具辅助的部分，并能够解释关键实现。

## English introduction

This project evaluates fine-grained intent classification on BANKING77 under limited training-label budgets. It compares classical and neural baselines, investigates training failures, and tests whether supervised contrastive learning and hard-negative weighting provide consistent benefits under a shared protocol. The delivery includes reproducible run artifacts, prediction-level audits, quality and cost reporting, and offline inference. Historical implementation and provenance limitations are documented; new GPU results will be reported only after the formal runs finish.
