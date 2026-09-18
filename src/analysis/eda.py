"""Run BANKING77 exploratory data analysis and write reproducible outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.prepare_data import DEFAULT_OUTPUT_DIR, RAW_DIR, ROOT, normalize_text


DEFAULT_EDA_DIR = ROOT / "outputs" / "eda"


def gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(float))
    if values.size == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, values.size + 1)
    return float((2 * np.sum(index * values) / (values.size * values.sum())) - (values.size + 1) / values.size)


def normalized_entropy(values: np.ndarray) -> float:
    probabilities = values / values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(len(values)))


def length_summary(series: pd.Series) -> dict[str, float]:
    return {
        "min": int(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": int(series.max()),
    }


def duplicate_stats(frame: pd.DataFrame) -> dict[str, int]:
    normalized = frame["text"].map(normalize_text)
    grouped = frame.assign(_normalized=normalized).groupby("_normalized")["category"]
    sizes = grouped.size()
    label_counts = grouped.nunique()
    return {
        "exact_duplicate_rows": int(frame.duplicated(["text", "category"], keep=False).sum()),
        "normalized_duplicate_rows": int(sizes[sizes > 1].sum()),
        "normalized_duplicate_groups": int((sizes > 1).sum()),
        "normalized_conflicting_label_groups": int((label_counts > 1).sum()),
    }


def cross_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    left_norm = left.assign(_normalized=left["text"].map(normalize_text))
    right_norm = right.assign(_normalized=right["text"].map(normalize_text))
    exact = set(left["text"]).intersection(right["text"])
    merged = left_norm[["_normalized", "category"]].merge(
        right_norm[["_normalized", "category"]],
        on="_normalized",
        suffixes=("_left", "_right"),
    )
    return {
        "exact_text_groups": len(exact),
        "normalized_text_groups": int(merged["_normalized"].nunique()),
        "normalized_label_mismatches": int(
            merged.loc[merged["category_left"] != merged["category_right"], "_normalized"].nunique()
        ),
    }


def lexical_pairs(train: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(train["text"])
    labels = sorted(train["category"].unique())
    centroids = np.vstack(
        [np.asarray(matrix[train["category"].to_numpy() == label].mean(axis=0)).ravel() for label in labels]
    )
    similarities = cosine_similarity(centroids)
    pairs: list[tuple[str, str, float]] = []
    for i, label_a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            pairs.append((label_a, labels[j], float(similarities[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pd.DataFrame(pairs[:top_n], columns=["intent_a", "intent_b", "tfidf_centroid_cosine"])


def save_plots(
    raw_train: pd.DataFrame,
    class_distribution: pd.DataFrame,
    pairs: pd.DataFrame,
    split_distribution: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = class_distribution.sort_values("train_count")
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.barh(ordered["category"], ordered["train_count"], color="#4472C4")
    ax.set_title("BANKING77 training samples by intent")
    ax.set_xlabel("Samples")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figure_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    word_lengths = raw_train["text"].str.findall(r"\b\w+\b").str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(word_lengths, bins=range(0, int(word_lengths.max()) + 5, 5), color="#70AD47", edgecolor="white")
    ax.axvline(word_lengths.median(), color="#C00000", linestyle="--", label=f"Median: {word_lengths.median():.0f}")
    ax.set_title("Training utterance length distribution")
    ax.set_xlabel("Words per utterance")
    ax.set_ylabel("Samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "text_length_distribution.png", dpi=160)
    plt.close(fig)

    shown = pairs.head(12).sort_values("tfidf_centroid_cosine")
    labels = shown["intent_a"] + " / " + shown["intent_b"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, shown["tfidf_centroid_cosine"], color="#ED7D31")
    ax.set_title("Highest lexical-overlap intent pairs")
    ax.set_xlabel("TF-IDF centroid cosine similarity")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figure_dir / "lexical_overlap_pairs.png", dpi=160)
    plt.close(fig)

    deviation = split_distribution.sort_values("validation_share_minus_source_pp")
    fig, ax = plt.subplots(figsize=(10, 15))
    colors = np.where(deviation["validation_share_minus_source_pp"] >= 0, "#5B9BD5", "#A5A5A5")
    ax.barh(deviation["category"], deviation["validation_share_minus_source_pp"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Validation share deviation from source training data")
    ax.set_xlabel("Percentage points")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figure_dir / "split_distribution_deviation.png", dpi=160)
    plt.close(fig)


def run_eda(
    raw_dir: Path = RAW_DIR,
    split_dir: Path = DEFAULT_OUTPUT_DIR,
    output_dir: Path = DEFAULT_EDA_DIR,
) -> dict[str, object]:
    raw_train = pd.read_csv(raw_dir / "train.csv")
    raw_test = pd.read_csv(raw_dir / "test.csv")
    train = pd.read_csv(split_dir / "train.csv")
    validation = pd.read_csv(split_dir / "val.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_train_counts = raw_train["category"].value_counts().sort_index()
    raw_test_counts = raw_test["category"].value_counts().sort_index()
    train_counts = train["category"].value_counts().sort_index()
    val_counts = validation["category"].value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "category": raw_train_counts.index,
        "train_count": raw_train_counts.values,
        "test_count": raw_test_counts.reindex(raw_train_counts.index).values,
    })
    class_distribution["train_share_pct"] = class_distribution["train_count"] / len(raw_train) * 100
    class_distribution.to_csv(table_dir / "class_distribution.csv", index=False)

    split_distribution = pd.DataFrame({
        "category": raw_train_counts.index,
        "source_count": raw_train_counts.values,
        "train_count": train_counts.reindex(raw_train_counts.index).values,
        "validation_count": val_counts.reindex(raw_train_counts.index).values,
    })
    split_distribution["source_share_pct"] = split_distribution["source_count"] / len(raw_train) * 100
    split_distribution["validation_share_pct"] = split_distribution["validation_count"] / len(validation) * 100
    split_distribution["validation_share_minus_source_pp"] = (
        split_distribution["validation_share_pct"] - split_distribution["source_share_pct"]
    )
    split_distribution.to_csv(table_dir / "split_distribution.csv", index=False)

    pairs = lexical_pairs(raw_train)
    pairs.to_csv(table_dir / "lexical_overlap_pairs.csv", index=False)

    train_words = raw_train["text"].str.findall(r"\b\w+\b").str.len()
    test_words = raw_test["text"].str.findall(r"\b\w+\b").str.len()
    train_chars = raw_train["text"].str.len()
    test_chars = raw_test["text"].str.len()

    train_tokens = [token for text in raw_train["text"].map(normalize_text) for token in text.split()]
    test_tokens = [token for text in raw_test["text"].map(normalize_text) for token in text.split()]
    train_vocab = set(train_tokens)
    test_vocab = set(test_tokens)
    test_oov_occurrences = sum(token not in train_vocab for token in test_tokens)

    train_test_overlap = cross_overlap(raw_train, raw_test)
    train_val_overlap = cross_overlap(train, validation)

    distribution_values = raw_train_counts.to_numpy()
    stats: dict[str, object] = {
        "dataset": {
            "train_rows": len(raw_train),
            "test_rows": len(raw_test),
            "classes": int(raw_train["category"].nunique()),
            "missing_train_cells": int(raw_train.isna().sum().sum()),
            "missing_test_cells": int(raw_test.isna().sum().sum()),
        },
        "class_distribution": {
            "train_min": int(raw_train_counts.min()),
            "train_max": int(raw_train_counts.max()),
            "train_mean": float(raw_train_counts.mean()),
            "train_median": float(raw_train_counts.median()),
            "max_to_min_ratio": float(raw_train_counts.max() / raw_train_counts.min()),
            "gini": gini(distribution_values),
            "normalized_entropy": normalized_entropy(distribution_values),
            "test_min": int(raw_test_counts.min()),
            "test_max": int(raw_test_counts.max()),
        },
        "text_length_words": {"train": length_summary(train_words), "test": length_summary(test_words)},
        "text_length_characters": {"train": length_summary(train_chars), "test": length_summary(test_chars)},
        "vocabulary": {
            "train_unique_tokens": len(train_vocab),
            "test_unique_tokens": len(test_vocab),
            "test_unique_token_oov_rate": len(test_vocab - train_vocab) / len(test_vocab),
            "test_token_occurrence_oov_rate": test_oov_occurrences / len(test_tokens),
        },
        "duplicates": {
            "train": duplicate_stats(raw_train),
            "test": duplicate_stats(raw_test),
            "official_train_test_overlap": train_test_overlap,
        },
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_class_min": int(train_counts.min()),
            "train_class_max": int(train_counts.max()),
            "validation_class_min": int(val_counts.min()),
            "validation_class_max": int(val_counts.max()),
            "max_absolute_class_share_deviation_pp": float(
                split_distribution["validation_share_minus_source_pp"].abs().max()
            ),
            "train_validation_overlap": train_val_overlap,
        },
        "top_lexical_overlap_pairs": pairs.head(10).to_dict(orient="records"),
    }
    (output_dir / "eda_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    save_plots(raw_train, class_distribution, pairs, split_distribution, figure_dir)

    top_small = class_distribution.nsmallest(5, "train_count")[["category", "train_count"]]
    top_large = class_distribution.nlargest(5, "train_count")[["category", "train_count"]]
    pair_rows = "\n".join(
        f"| `{row.intent_a}` | `{row.intent_b}` | {row.tfidf_centroid_cosine:.3f} |"
        for row in pairs.head(10).itertuples()
    )
    small_rows = ", ".join(f"`{r.category}` ({r.train_count})" for r in top_small.itertuples())
    large_rows = ", ".join(f"`{r.category}` ({r.train_count})" for r in top_large.itertuples())
    report = f"""# BANKING77 EDA 报告

## 1. 数据概况

- 官方训练集：{len(raw_train):,} 条；官方测试集：{len(raw_test):,} 条；共 {raw_train['category'].nunique()} 类。
- 两个 CSV 均无缺失值，字段为 `text` 和 `category`。
- 测试集每类固定 {raw_test_counts.min()} 条，完全均衡；训练集每类 {raw_train_counts.min()}–{raw_train_counts.max()} 条。

## 2. 类别分布

训练集类别数量最大/最小比为 {raw_train_counts.max() / raw_train_counts.min():.2f}，Gini 系数为 {gini(distribution_values):.4f}，归一化熵为 {normalized_entropy(distribution_values):.4f}。数据存在温和不均衡，但不属于典型严重长尾。

- 样本最少的类别：{small_rows}。
- 样本最多的类别：{large_rows}。

![类别分布](figures/class_distribution.png)

## 3. 文本长度与词汇

- 训练文本平均 {train_words.mean():.2f} 词，中位数 {train_words.median():.0f} 词，95 分位 {train_words.quantile(0.95):.0f} 词，最大 {train_words.max()} 词。
- 测试文本平均 {test_words.mean():.2f} 词，中位数 {test_words.median():.0f} 词。
- 测试 token 出现次数口径的 OOV 率为 {test_oov_occurrences / len(test_tokens) * 100:.2f}%；测试独立 token 类型口径的 OOV 率为 {len(test_vocab - train_vocab) / len(test_vocab) * 100:.2f}%。

文本整体较短，`max_length=64` 可作为后续 BERT 实验的初始值，但最终应根据 tokenizer 后的长度再次确认截断比例。

![文本长度分布](figures/text_length_distribution.png)

## 4. 重复与潜在泄漏

- 官方训练集无完全相同的 `text + category` 重复行；宽松标准化后有 {stats['duplicates']['train']['normalized_duplicate_groups']} 个重复文本组、{stats['duplicates']['train']['normalized_duplicate_rows']} 行，包含 {stats['duplicates']['train']['normalized_conflicting_label_groups']} 个跨标签组。
- 官方测试集宽松标准化后有 {stats['duplicates']['test']['normalized_duplicate_groups']} 个重复文本组，其中 {stats['duplicates']['test']['normalized_conflicting_label_groups']} 个跨标签组。
- 官方 train/test 间有 {train_test_overlap['normalized_text_groups']} 个标准化文本重叠组，标签不一致组数为 {train_test_overlap['normalized_label_mismatches']}。这是官方 benchmark 的既有特征，主实验保留并在报告中披露。
- 新生成的 train/validation 之间标准化文本重叠为 {train_val_overlap['normalized_text_groups']}，没有由内部划分引入重复泄漏。

## 5. 高词面重叠类别

下表按每类 TF-IDF 质心余弦相似度排序，只代表词面重叠候选，不等同于模型 confusion matrix。训练基线后应以真实误判重新确定 hard-negative intent pairs。

| Intent A | Intent B | Cosine similarity |
|---|---|---:|
{pair_rows}

![词面重叠类别对](figures/lexical_overlap_pairs.png)

## 6. 训练/验证划分

- 随机种子：42。
- 划分方式：按 `category` 分层；初始划分后将标准化重复文本收拢到同一侧，并用同类 singleton 对换，保持每类数量不变。
- 训练集：{len(train):,} 条；验证集：{len(validation):,} 条；官方测试集 {len(raw_test):,} 条保持不变。
- 验证集每类 {val_counts.min()}–{val_counts.max()} 条；相对官方训练集的最大类别占比偏差仅 {split_distribution['validation_share_minus_source_pp'].abs().max():.3f} 个百分点。
- 划分文件位于 `outputs/datasets/`，完整源索引、源文件哈希和修复记录保存在 `split_manifest.json`。

![划分比例偏差](figures/split_distribution_deviation.png)

## 7. 对后续实验的建议

1. 固定本次验证集用于 checkpoint 与超参数选择，官方测试集只用于最终评估。
2. few-shot 样本仅从 9,000 条训练池抽取，验证集不计入标注预算。
3. 主指标使用 Macro-F1，并报告 per-class F1；训练基线后从 confusion matrix 识别真实易混淆类别对。
4. 由于存在少量标准化重复及跨标签近重复，最终报告应披露清洗口径，并可补充一次去重敏感性实验。
"""
    (output_dir / "eda_report.md").write_text(report, encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EDA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_eda(args.raw_dir, args.split_dir, args.output_dir)
    print(json.dumps({"dataset": stats["dataset"], "split": stats["split"]}, indent=2))


if __name__ == "__main__":
    main()
