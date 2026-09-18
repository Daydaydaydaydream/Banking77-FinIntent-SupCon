from __future__ import annotations

import unittest

from utils.metrics import classification_metrics, confusion_rows


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], ["a", "b"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["per_class"]["a"]["support"], 2)

    def test_confusion_shape(self) -> None:
        rows = confusion_rows([0, 1, 1], [0, 0, 1], ["a", "b"])
        self.assertEqual(rows, [["a", 1, 0], ["b", 1, 1]])


if __name__ == "__main__":
    unittest.main()
