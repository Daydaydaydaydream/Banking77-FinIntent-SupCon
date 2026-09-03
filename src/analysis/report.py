"""Final experiment report generation.

Reads the persisted baseline and ablation results and writes a Markdown report
summarising Macro-F1, per-class F1 on the confusable intent pairs, and the
generated figures.  The report is objective and data-first: it states the
experimental setup and reports measured numbers without interpretive
embellishment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from config import PROJECT_ROOT, RESULT_DIR, ensure_dirs
from analysis.visualize import CONFUSABLE_PAIRS

OUTPUT_REPORT = PROJECT_ROOT / "report.md"

SETTINGS = ["full", "5shot", "10shot", "20shot"]
BASELINE_METHODS = [
    ("tfidf_svm", "TF-IDF + LinearSVC"),
    ("bert_ft", "BERT 微调"),
    ("setfit", "SetFit"),
]
ABLATION_METHODS = [
    ("baseline_bert", "Baseline (BERT 微调)"),
    ("supcon", "Baseline + SupCon"),
    ("supcon_hn", "Baseline + SupCon + 难负例挖掘"),
]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def _baseline_table(baseline: dict) -> List[str]:
    """Build the baseline Macro-F1 comparison table."""
    header = "| 方法 | " + " | ".join(SETTINGS) + " |"
    sep = "|---|" + "|".join(["---"] * len(SETTINGS)) + "|"
    rows = [header, sep]
    for key, label in BASELINE_METHODS:
        by_setting = baseline.get(key, {})
        cells = [_fmt(by_setting.get(s)) for s in SETTINGS]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def _ablation_table(ablation: dict) -> List[str]:
    """Build the ablation Macro-F1 comparison table."""
    header = "| 方法 | " + " | ".join(SETTINGS) + " |"
    sep = "|---|" + "|".join(["---"] * len(SETTINGS)) + "|"
    rows = [header, sep]
    for key, label in ABLATION_METHODS:
        cells = []
        for s in SETTINGS:
            entry = ablation.get(s, {}).get(key, {})
            cells.append(_fmt(entry.get("macro_f1")))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def _confusable_table(ablation: dict, setting: str = "full") -> List[str]:
    """Build a per-class F1 table over the confusable intent pairs.

    Each confusable pair contributes two rows (one per intent).  The table
    reports the per-class F1 of each method for those intents, highlighting
    whether SupCon / hard-negative weighting improves separation of
    semantically overlapping classes.
    """
    header = "| 意图 | " + " | ".join(label for _, label in ABLATION_METHODS) + " |"
    sep = "|---|" + "|".join(["---"] * len(ABLATION_METHODS)) + "|"
    rows = [header, sep]

    by_method = {
        key: ablation.get(setting, {}).get(key, {}).get("per_class_f1", {})
        for key, _ in ABLATION_METHODS
    }

    intents: List[str] = []
    for a, b in CONFUSABLE_PAIRS:
        if a not in intents:
            intents.append(a)
        if b not in intents:
            intents.append(b)

    for intent in intents:
        cells = [_fmt(by_method[key].get(intent)) for key, _ in ABLATION_METHODS]
        rows.append(f"| {intent} | " + " | ".join(cells) + " |")
    return rows


def generate_report() -> Path:
    """Generate ``report.md`` and return its path."""
    ensure_dirs()
    baseline = _load_json(RESULT_DIR / "results.json") or {}
    ablation = _load_json(RESULT_DIR / "ablation_results.json") or {}

    sections: List[str] = []

    sections.append("# Banking77 意图分类实验报告\n")

    sections.append("## 1. 实验设置\n")
    sections.append(
        "- 数据集：BANKING77（训练 10,003 / 测试 3,080，77 个意图类别）。\n"
        "- 少样本设置：每类 5 / 10 / 20 条标注（固定随机种子 42）。\n"
        "- 主指标：Macro-F1；辅助指标：per-class F1、混淆矩阵。\n"
        "- 骨干模型：bert-base-uncased（序列长度 64）。\n"
    )

    sections.append("## 2. Baseline 结果\n")
    sections.append("三组基线的测试集 Macro-F1 如下。\n")
    sections.extend(_baseline_table(baseline))
    sections.append("")

    sections.append("## 3. 消融实验结果\n")
    sections.append(
        "在 BERT 骨干上对比三组设置：直接微调（Baseline）、引入监督对比学习"
        "（SupCon）、进一步叠加难负例挖掘（SupCon + HN）。测试集 Macro-F1 如下。\n"
    )
    sections.extend(_ablation_table(ablation))
    sections.append("")

    sections.append("## 4. 易混淆意图对的 per-class F1\n")
    sections.append(
        "下表列出语义高度重叠的意图对（见项目计划）在三组方法下的 per-class F1，"
        "用于观察性能提升是否集中在易混淆类别上。\n"
    )
    sections.extend(_confusable_table(ablation))
    sections.append("")

    sections.append("## 5. 可视化\n")
    sections.append(
        "以下图表由分析脚本生成，存放于 `outputs/figures/`：\n\n"
        "- 混淆矩阵：`confusion_<method>_full.png`\n"
        "- 易混淆意图对子矩阵：`confusable_pairs_full.png`\n"
        "- t-SNE 特征投影：`tsne_<method>_full.png`\n"
    )

    sections.append("## 6. 结论与局限\n")
    sections.append(
        "结论与局限基于第 2、3、4 节的数值结果归纳，应随实验结果完成后补充。\n"
    )

    report = "\n".join(sections)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    return OUTPUT_REPORT


if __name__ == "__main__":
    path = generate_report()
    print(f"[report] written to {path}")
