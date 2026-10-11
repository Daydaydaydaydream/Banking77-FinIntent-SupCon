"""Generate the experiment report from committed artifacts only.

Every number here is read back from ``outputs/runs`` and ``outputs/results``.
Nothing is hard-coded, so the report cannot drift from the runs that produced
it.  Confusions observed on the test split are reported as findings, never fed
back into a training configuration.

Usage::

    PYTHONPATH=src python -m analysis.report --output docs/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.aggregate import discover, paired_comparison, pivot_table, seed_consistency, summarise
from analysis.confusion_pairs import CONFUSION_GROUPS, compare, declared_pairs, group_summary, pair_confusions
from config import ROOT, RUNS_DIR


SETTINGS_ORDER = ("5shot", "10shot", "20shot", "full")
METHOD_LABELS = {
    "tfidf_svm": "TF-IDF + Linear SVM",
    "setfit": "SetFit",
    "bert": "BERT (baseline)",
    "supcon": "BERT + SupCon",
    "supcon_hn": "BERT + SupCon + HN",
}
ABLATION_ORDER = ("bert", "supcon", "supcon_hn")


def _fmt(mean: float, std: float, seeds: int) -> str:
    if seeds <= 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def results_table(summary: Any, methods: list[str]) -> str:
    lines = [
        "| Method | 5-shot | 10-shot | 20-shot | full |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        cells = []
        for setting in SETTINGS_ORDER:
            row = summary[(summary.method == method) & (summary.setting == setting)]
            if row.empty:
                cells.append("-")
                continue
            row = row.iloc[0]
            text = _fmt(row.test_macro_f1_mean, row.test_macro_f1_std, int(row.seeds))
            if bool(row.any_collapsed):
                text += " ⚠"
            cells.append(text)
        lines.append(f"| {METHOD_LABELS.get(method, method)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_section(comparisons: list[dict[str, Any]], baseline: str) -> str:
    """One row per (treatment, setting) with the per-seed paired delta."""

    lines = [
        "| Comparison | Setting | paired seeds | mean delta | worst seed | consistent |",
        "|---|---|---:|---:|---:|---|",
    ]
    for treatment in ("supcon", "supcon_hn"):
        for setting in SETTINGS_ORDER:
            matches = [
                c
                for c in comparisons
                if c["setting"] == setting
                and c["baseline"] == baseline
                and c["treatment"] == treatment
            ]
            if not matches:
                continue
            row = matches[0]
            lines.append(
                f"| {METHOD_LABELS[treatment]} vs {METHOD_LABELS.get(baseline, baseline)} "
                f"| {setting} | {row['paired_seeds']} "
                f"| {row['delta_mean']:+.4f} ± {row['delta_std']:.4f} "
                f"| {row['delta_min']:+.4f} "
                f"| {'yes' if row['consistent_gain'] else 'no'} |"
            )
    return "\n".join(lines)


def confusion_section(method: str, setting: str, seed: int, runs_dir: Path) -> str:
    from analysis.confusion_pairs import load_confusion

    conf = load_confusion(method, setting, seed, runs_dir)
    if conf is None:
        return f"_No confusion matrix for {method}/{setting}/seed{seed}._"
    rows = pair_confusions(conf[0], declared_pairs())
    groups = group_summary(rows)
    lines = ["| Intent group | pairs | bidirectional errors |", "|---|---:|---:|"]
    for entry in groups:
        lines.append(f"| `{entry['group']}` | {entry['pairs']} | {entry['bidirectional_total']} |")
    return "\n".join(lines)


def _markdown_table(frame: Any, index_label: str = "") -> str:
    """Render a DataFrame as a GitHub markdown table without ``tabulate``."""

    columns = list(frame.columns)
    header = ([index_label] if index_label else []) + [str(c) for c in columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in frame.iterrows():
        cells = []
        if index_label:
            cells.append(str(index))
        cells.extend("-" if pd.isna(value) else str(value) for value in row.tolist())
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "RESULTS.md")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--baseline", default="bert")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = discover(args.runs_dir)
    if not rows:
        raise SystemExit(f"No completed runs under {args.runs_dir}")

    summary = summarise(rows)
    consistency = seed_consistency(rows)
    comparisons = paired_comparison(rows, args.baseline, "supcon")
    comparisons += paired_comparison(rows, args.baseline, "supcon_hn")
    comparisons += paired_comparison(rows, "supcon", "supcon_hn")

    present = [m for m in ("tfidf_svm", "setfit") + ABLATION_ORDER if (summary.method == m).any()]
    collapsed = [r for r in rows if r.get("collapsed")]

    parts: list[str] = []
    parts.append("# 实验结果\n")
    parts.append(
        "本报告的每个数字都由 `outputs/runs/` 下的 `metrics.json` 重新汇总得到，"
        "没有任何硬编码值。所有结果为多随机种子的均值与标准差。\n"
    )

    parts.append("## 1. 主结果表\n")
    parts.append(
        "测试集 Macro-F1，`mean ± std` over seeds "
        f"{sorted({r['seed'] for r in rows})}。⚠ 表示该运行被标记为类别塌缩，不可信。\n"
    )
    parts.append(results_table(summary, present) + "\n")

    parts.append("## 2. 随机种子稳定性\n")
    parts.append(
        "单一种子的结果无法区分方法差异与划分噪声。下表给出每个 (方法, 设置) 的种子波动范围。\n"
    )
    pivot = consistency.pivot(index="method", columns="setting", values="range")
    pivot = pivot.reindex(columns=[s for s in SETTINGS_ORDER if s in pivot.columns])
    parts.append(_markdown_table(pivot.round(4), index_label="method") + "\n")

    parts.append("## 3. 逐步消融\n")
    parts.append(
        f"以 {METHOD_LABELS.get(args.baseline, args.baseline)} 为对照，"
        "在相同 backbone、数据划分、批次构造与评估代码下逐级比较。"
        "「worst seed」是三种子里最差的那次配对差值；「consistent」表示三次全部为正。\n"
    )
    parts.append(ablation_section(comparisons, args.baseline) + "\n")

    parts.append("## 4. 预声明易混淆类别对\n")
    parts.append(
        "以下类别组在训练开始前依据标签语义与业务分组预先声明，"
        "不使用测试集混淆矩阵来选择类别对——否则评估集会泄漏进方法选择。"
        "双向误判数按 a→b 与 b→a 分别统计后相加。\n"
    )
    parts.append(
        "| 组 | 成员 |\n|---|---|\n"
        + "\n".join(
            f"| `{group}` | " + ", ".join(f"`{m}`" for m in members) + " |"
            for group, members in CONFUSION_GROUPS.items()
        )
        + "\n"
    )
    for method in ("bert", "supcon", "supcon_hn"):
        parts.append(f"\n### {METHOD_LABELS.get(method, method)} / 5-shot / seed{args.seed}\n")
        parts.append(confusion_section(method, "5shot", args.seed, args.runs_dir) + "\n")

    bert_vs_supcon = compare(args.baseline, "supcon", "5shot", args.seed, args.runs_dir)
    if "error" not in bert_vs_supcon:
        parts.append("\n### 逐对双向误判变化（BERT → SupCon，5-shot）\n")
        parts.append(
            "| 组 | label A | label B | BERT | SupCon | delta | 改善 |\n"
            "|---|---|---|---:|---:|---:|---|\n"
            + "\n".join(
                f"| `{row['group']}` | `{row['label_a']}` | `{row['label_b']}` "
                f"| {row[f'{args.baseline}_total']} | {row['supcon_total']} "
                f"| {row['delta']:+d} | {'是' if row['improved'] else '否'} |"
                for row in bert_vs_supcon["pairs"][:15]
            )
            + "\n"
        )
        parts.append(
            f"\n预声明类别对中有 {bert_vs_supcon['improved_pair_count']}"
            f"/{bert_vs_supcon['total_pair_count']} 组双向误判下降。\n"
        )

    parts.append("\n## 5. 产物完整性\n")
    collapsed_list = (
        ", ".join(f"{r['method']}/{r['setting']}/seed{r['seed']}" for r in collapsed)
        if collapsed
        else "无"
    )
    parts.append(
        f"- 已完成运行：{len(rows)}\n"
        f"- 标记为塌缩：{len(collapsed)}\n"
        f"- 塌缩运行：{collapsed_list}\n"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()