"""Final experiment report generation.

Aggregates the stage-A artefact tree (``outputs/runs/<method>/<setting>/seed<k>``)
into ``report.md``:

* Macro-F1 as ``mean ± std`` across seeds (never a best-of selection);
* paired bootstrap significance against the strongest baseline;
* per-class F1 on the confusable intent pairs defined in the project plan;
* pointers to the generated figures.

The report is data-first: it states the setup and reports measured numbers
without interpretive embellishment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.visualize import CONFUSABLE_PAIRS
from config import PROJECT_ROOT, SEEDS, SETTINGS, ensure_dirs, run_dir
from utils.metrics import paired_bootstrap_macro_f1

OUTPUT_REPORT = PROJECT_ROOT / "report.md"

METHOD_LABELS: Dict[str, str] = {
    "tfidf_svm": "TF-IDF + LinearSVC",
    "bert_ft": "BERT 微调",
    "setfit": "SetFit",
    "supcon": "SupCon",
    "supcon_hn": "SupCon + 难负例",
}

BASELINE_FOR_SIGNIFICANCE = "bert_ft"


def _load_metrics(method: str, setting: str, seed: int) -> Optional[dict]:
    path = run_dir(method, setting, seed) / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_predictions(method: str, setting: str, seed: int) -> Optional[pd.DataFrame]:
    path = run_dir(method, setting, seed) / "predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _cell(method: str, setting: str, seeds: Sequence[int]) -> str:
    scores = [
        _load_metrics(method, setting, s)["macro_f1"]
        for s in seeds
        if _load_metrics(method, setting, s) is not None
    ]
    if not scores:
        return "—"
    if len(scores) == 1:
        return f"{scores[0]:.4f}"
    return f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"


def _main_table(methods: Sequence[str], settings: Sequence[str], seeds: Sequence[int]) -> List[str]:
    header = "| 方法 | " + " | ".join(settings) + " |"
    sep = "|---|" + "|".join(["---"] * len(settings)) + "|"
    rows = [header, sep]
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        rows.append(f"| {label} | " + " | ".join(_cell(method, s, seeds) for s in settings) + " |")
    return rows


def _significance_rows(
    methods: Sequence[str], settings: Sequence[str], seed: int, n_resamples: int
) -> List[str]:
    """Paired bootstrap of each method against the BERT baseline."""
    header = "| 方法 | 设置 | ΔMacro-F1 | 95% CI | p 值 |"
    rows = [header, "|---|---|---|---|---|"]
    any_result = False

    base_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for setting in settings:
        base_cache[setting] = _load_predictions(BASELINE_FOR_SIGNIFICANCE, setting, seed)

    for method in methods:
        if method == BASELINE_FOR_SIGNIFICANCE:
            continue
        for setting in settings:
            base = base_cache.get(setting)
            preds = _load_predictions(method, setting, seed)
            if base is None or preds is None:
                continue
            if not np.array_equal(base["y_true"].to_numpy(), preds["y_true"].to_numpy()):
                continue
            result = paired_bootstrap_macro_f1(
                base["y_true"].to_numpy(),
                preds["y_pred"].to_numpy(),
                base["y_pred"].to_numpy(),
                n_resamples=n_resamples,
                seed=seed,
            )
            any_result = True
            rows.append(
                f"| {METHOD_LABELS.get(method, method)} | {setting} | "
                f"{result['mean_delta']:+.4f} | "
                f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] | "
                f"{result['p_value']:.4f} |"
            )

    return rows if any_result else []


def _confusable_rows(
    methods: Sequence[str], setting: str, seed: int
) -> List[str]:
    """Per-class F1 for the confusable intent pairs, averaged over seeds."""
    intents: List[str] = []
    for a, b in CONFUSABLE_PAIRS:
        for intent in (a, b):
            if intent not in intents:
                intents.append(intent)

    header = "| 意图 | " + " | ".join(METHOD_LABELS.get(m, m) for m in methods) + " |"
    rows = [header, "|---|" + "|".join(["---"] * len(methods)) + "|"]

    for intent in intents:
        cells = []
        for method in methods:
            block = _load_metrics(method, setting, seed)
            if block is None:
                cells.append("—")
                continue
            entry = block.get("per_class", {}).get(intent)
            cells.append("—" if entry is None else f"{entry['f1']:.4f}")
        rows.append(f"| {intent} | " + " | ".join(cells) + " |")
    return rows


def generate_report(
    methods: Sequence[str] = tuple(METHOD_LABELS),
    settings: Sequence[str] = tuple(SETTINGS),
    seeds: Sequence[int] = tuple(SEEDS),
    n_resamples: int = 1000,
) -> Path:
    """Generate ``report.md`` and return its path."""
    ensure_dirs()
    seed0 = seeds[0]

    sections: List[str] = []
    sections.append("# Banking77 意图分类实验报告\n")

    sections.append("## 1. 实验设置\n")
    sections.append(
        "- 数据集：BANKING77（官方训练 10,003 / 测试 3,080，77 个意图类别）。\n"
        "- 数据划分：从官方训练集中**分层切出**训练池 9,000 与验证集 1,003；"
        "测试集保持官方划分并冻结，仅用于最终评估。\n"
        f"- 少样本设置：每类 5 / 10 / 20 条标注，抽样种子 {list(seeds)}。\n"
        "- 模型选择：统一以验证集 Macro-F1 选取 checkpoint，不使用测试集选点。\n"
        "- 主指标：Macro-F1；辅助指标：Accuracy、Micro-F1、per-class P/R/F1。\n"
        "- 骨干模型：bert-base-uncased（序列长度 64）；SetFit 使用 all-MiniLM-L6-v2。\n"
    )

    sections.append("## 2. 主结果（测试集 Macro-F1）\n")
    sections.append(
        f"每个单元格为 {len(seeds)} 个种子的 mean ± std；单种子时仅给出均值。\n"
    )
    sections.extend(_main_table(methods, settings, seeds))
    sections.append("")

    sections.append("## 3. 显著性检验\n")
    sections.append(
        f"以 {METHOD_LABELS[BASELINE_FOR_SIGNIFICANCE]} 为对照组，对测试集做分层 paired bootstrap"
        f"（{n_resamples} 次重采样，种子 {seed0}）。ΔMacro-F1 为「方法 − 对照」。\n"
    )
    sig_rows = _significance_rows(methods, settings, seed0, n_resamples)
    if sig_rows:
        sections.extend(sig_rows)
    else:
        sections.append("_暂无可用的预测文件，无法计算显著性。_")
    sections.append("")

    sections.append("## 4. 易混淆意图对的 per-class F1\n")
    sections.append(
        f"下表为 `{settings[0]}` 设置、种子 {seed0} 下，语义高度重叠的意图对的 per-class F1。\n"
    )
    sections.extend(_confusable_rows(methods, settings[0], seed0))
    sections.append("")

    sections.append("## 5. 可视化\n")
    sections.append(
        "图表由 `src/analysis/visualize.py` 生成，存放于 `outputs/figures/`：\n\n"
        "- 混淆矩阵：`confusion_<method>_<setting>_seed<k>.png`\n"
        "- 易混淆意图对子矩阵：`confusable_pairs_<setting>_seed<k>.png`\n"
        "- 置信度分布：`confidence_<setting>_seed<k>.png`\n"
        "- t-SNE 特征投影（仅对保存了 embedding 的方法）：`tsne_<method>_<setting>_seed<k>.png`\n"
    )

    sections.append("## 6. 结论与局限\n")
    sections.append(
        "结论应基于第 2、3、4 节的数值归纳。已知局限：\n\n"
        "- 少样本抽取方差通常大于训练方差，需在报告中单独说明；\n"
        "- 验证集（1,003 条）在 few-shot 设置下大于训练集，checkpoint 选择受此影响；\n"
        "- TF-IDF 的置信度为 decision_function 上的 softmax 近似，未做概率校准。\n"
    )

    report = "\n".join(sections)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    return OUTPUT_REPORT


if __name__ == "__main__":
    path = generate_report()
    print(f"[report] written to {path}")
