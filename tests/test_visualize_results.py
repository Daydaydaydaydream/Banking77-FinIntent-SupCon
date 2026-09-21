from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.visualize_results import (
    discover_runs,
    load_confusion_matrix,
    top_confusion_pairs,
    visualize_results,
)


class VisualizeResultsTests(unittest.TestCase):
    def _write_run(self, root: Path, method: str = "demo") -> Path:
        run_dir = root / method / "5shot" / "seed42"
        run_dir.mkdir(parents=True)
        metrics = {
            "validation": {
                "accuracy": 0.5,
                "macro_f1": 0.45,
                "micro_f1": 0.5,
                "per_class": {
                    "class_a": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
                    "class_b": {"precision": 0.4, "recall": 0.4, "f1": 0.4, "support": 2},
                },
            },
            "test": {
                "accuracy": 0.75,
                "macro_f1": 0.7333,
                "micro_f1": 0.75,
                "per_class": {
                    "class_a": {"precision": 0.67, "recall": 1.0, "f1": 0.8, "support": 2},
                    "class_b": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2},
                },
            },
            "train_seconds": 1.5,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(
                [
                    ["true\\pred", "class_a", "class_b"],
                    ["class_a", 2, 0],
                    ["class_b", 1, 1],
                ]
            )
        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["y_pred"])
            writer.writeheader()
            writer.writerows([{"y_pred": "class_a"}] * 3 + [{"y_pred": "class_b"}])
        (run_dir / "training_history.json").write_text(
            json.dumps(
                [
                    {"epoch": 1.0, "eval_macro_f1": 0.3, "eval_loss": 1.2},
                    {"epoch": 2.0, "eval_macro_f1": 0.45, "eval_loss": 0.9},
                ]
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_discover_runs_ignores_incomplete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root)
            (root / "incomplete" / "full" / "seed42").mkdir(parents=True)
            runs = discover_runs(root)
            self.assertEqual([(run.method, run.setting, run.seed) for run in runs], [("demo", "5shot", 42)])

    def test_top_confusion_pairs_combines_both_directions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._write_run(Path(tmp))
            matrix = load_confusion_matrix(run_dir / "confusion_matrix.csv")
            pairs = top_confusion_pairs(matrix, top_k=1)
            self.assertEqual(int(pairs.iloc[0]["a_to_b"]), 0)
            self.assertEqual(int(pairs.iloc[0]["b_to_a"]), 1)
            self.assertEqual(int(pairs.iloc[0]["total"]), 1)

    def test_top_confusion_pairs_handles_perfect_predictions(self) -> None:
        matrix = pd.DataFrame([[2, 0], [0, 2]], index=["a", "b"], columns=["a", "b"])
        pairs = top_confusion_pairs(matrix)
        self.assertTrue(pairs.empty)
        self.assertEqual(
            list(pairs.columns),
            ["intent_a", "intent_b", "a_to_b", "b_to_a", "total"],
        )

    def test_visualize_results_generates_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            output_dir = root / "figures"
            self._write_run(runs_dir)
            generated = visualize_results(runs_dir=runs_dir, output_dir=output_dir, dpi=72)
            names = {path.name for path in generated}
            self.assertIn("test_macro_f1_comparison.png", names)
            self.assertIn("test_per_class_f1_5shot_seed42.png", names)
            self.assertIn("confusion_matrix_demo_5shot_seed42.png", names)
            self.assertIn("top_confusions_demo_5shot_seed42.csv", names)
            self.assertIn("prediction_distribution_demo_5shot_seed42.png", names)
            self.assertIn("training_history_demo_5shot_seed42.png", names)
            self.assertIn("visualization_manifest.json", names)
            self.assertTrue(all(path.exists() for path in generated))


if __name__ == "__main__":
    unittest.main()
