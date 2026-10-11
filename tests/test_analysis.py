"""Tests for multi-seed aggregation and predefined confusion-pair analysis."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from analysis.aggregate import paired_comparison, pivot_table, seed_consistency, summarise
from analysis.confusion_pairs import CONFUSION_GROUPS, declared_pairs, group_summary, pair_confusions


def _run(method: str, setting: str, seed: int, test_f1: float, val_f1: float, collapsed: bool = False) -> dict:
    return {
        "method": method,
        "setting": setting,
        "seed": seed,
        "validation_macro_f1": val_f1,
        "test_macro_f1": test_f1,
        "test_accuracy": test_f1,
        "predicted_class_count": 77,
        "zero_f1_class_count": 0,
        "collapsed": collapsed,
        "non_finite_grad_events": 0,
        "train_seconds": 100.0,
        "path": f"{method}/{setting}/seed{seed}",
    }


class AggregateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _run("bert", "5shot", 42, 0.61, 0.60),
            _run("bert", "5shot", 43, 0.65, 0.64),
            _run("bert", "5shot", 44, 0.63, 0.62),
            _run("supcon", "5shot", 42, 0.66, 0.65),
            _run("supcon", "5shot", 43, 0.70, 0.69),
            _run("supcon", "5shot", 44, 0.68, 0.67),
        ]

    def test_mean_and_std_across_seeds(self) -> None:
        summary = summarise(self.rows)
        bert = summary[(summary.method == "bert") & (summary.setting == "5shot")].iloc[0]
        self.assertAlmostEqual(bert["test_macro_f1_mean"], 0.63, places=3)
        self.assertEqual(bert["seeds"], 3)
        self.assertGreater(bert["test_macro_f1_std"], 0.0)

    def test_single_seed_reports_zero_variance_but_no_std_label(self) -> None:
        rows = [_run("bert", "5shot", 42, 0.61, 0.60)]
        summary = summarise(rows)
        self.assertEqual(summary.iloc[0]["test_macro_f1_std"], 0.0)
        label = pivot_table(summary).loc["bert", "5shot"]
        self.assertIn("1 seed", label)
        self.assertNotIn("±", label)

    def test_collapsed_run_is_flagged_in_the_table(self) -> None:
        rows = [_run("bert", "5shot", 42, 0.01, 0.01, collapsed=True)] * 1
        summary = summarise(rows)
        self.assertTrue(bool(summary.iloc[0]["any_collapsed"]))
        self.assertIn("⚠", pivot_table(summary).loc["bert", "5shot"])

    def test_paired_comparison_uses_matching_seeds(self) -> None:
        result = paired_comparison(self.rows, "bert", "supcon")
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["paired_seeds"], 3)
        self.assertAlmostEqual(row["delta_mean"], 0.05, places=3)
        self.assertTrue(row["consistent_gain"])

    def test_paired_comparison_flags_inconsistent_gains(self) -> None:
        rows = [
            _run("bert", "5shot", 42, 0.60, 0.60),
            _run("supcon", "5shot", 42, 0.70, 0.70),
            _run("bert", "5shot", 43, 0.70, 0.70),
            _run("supcon", "5shot", 43, 0.60, 0.60),
        ]
        row = paired_comparison(rows, "bert", "supcon")[0]
        self.assertFalse(row["consistent_gain"])

    def test_seed_consistency_reports_the_range(self) -> None:
        table = seed_consistency(self.rows)
        bert = table[(table.method == "bert") & (table.setting == "5shot")].iloc[0]
        self.assertAlmostEqual(bert["range"], 0.04, places=3)

    def test_empty_input_returns_empty_frames(self) -> None:
        self.assertTrue(summarise([]).empty)
        self.assertEqual(paired_comparison([], "bert", "supcon"), [])


class ConfusionPairTest(unittest.TestCase):
    def test_pairs_are_derived_from_declared_groups(self) -> None:
        pairs = declared_pairs()
        expected = len(CONFUSION_GROUPS["balance_not_updated"])
        self.assertEqual(len(CONFUSION_GROUPS["balance_not_updated"]), expected)
        # 4-member group yields 6 unordered pairs
        identity = [p for p in pairs if p[0] == "identity_verification"]
        self.assertEqual(len(identity), 6)

    def test_every_declared_label_exists_in_the_dataset(self) -> None:
        categories = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "banking_data" / "categories.json").read_text("utf-8")
        )
        missing = [
            label
            for members in CONFUSION_GROUPS.values()
            for label in members
            if label not in categories
        ]
        self.assertEqual(missing, [], f"declared labels absent from BANKING77: {missing}")

    def test_bidirectional_directions_are_counted_separately(self) -> None:
        labels = ["a", "b", "c"]
        frame = pd.DataFrame(
            [
                [0, 7, 0],
                [3, 0, 0],
                [0, 0, 0],
            ],
            index=labels,
            columns=labels,
        )
        pairs = [("g", "a", "b")]
        rows = pair_confusions(frame, pairs)
        self.assertEqual(rows[0]["a->b"], 7)
        self.assertEqual(rows[0]["b->a"], 3)
        self.assertEqual(rows[0]["bidirectional_total"], 10)

    def test_missing_labels_are_skipped_rather_than_crashing(self) -> None:
        frame = pd.DataFrame([[0]], index=["a"], columns=["a"])
        self.assertEqual(pair_confusions(frame, [("g", "a", "nonexistent")]), [])

    def test_group_summary_aggregates_pair_counts(self) -> None:
        rows = [
            {"group": "g", "bidirectional_total": 4},
            {"group": "g", "bidirectional_total": 6},
            {"group": "h", "bidirectional_total": 1},
        ]
        summary = group_summary(rows)
        self.assertEqual(summary[0]["group"], "g")
        self.assertEqual(summary[0]["bidirectional_total"], 10)
        self.assertEqual(summary[0]["pairs"], 2)


if __name__ == "__main__":
    unittest.main()