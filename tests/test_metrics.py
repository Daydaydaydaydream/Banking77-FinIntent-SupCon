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


class RankingTests(unittest.TestCase):
    def test_export_ties_match_training_argmax_for_all_class_pairs(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from utils.artifacts import RunArtifacts
        categories = [str(i) for i in range(77)]
        pairs = [(i,j) for i in range(77) for j in range(i+1,77)]
        scores = np.zeros((len(pairs),77),dtype=np.float32)
        for row,(a,b) in enumerate(pairs):
            scores[row,[a,b]] = .5
        with tempfile.TemporaryDirectory() as tmp:
            artifact = RunArtifacts('ce','5shot',42,root=Path(tmp))
            actual = artifact.write_predictions(['tie']*len(pairs),scores.argmax(1),scores,categories)
        self.assertTrue(np.array_equal(actual,scores.argmax(1)))

    def test_torch_inference_ranking_matches_numpy_on_ties(self):
        import numpy as np
        import torch
        scores = np.array([[0.,.4,.4,.2],[.25,.25,.25,.25]],dtype=np.float32)
        actual = torch.argsort(torch.tensor(scores),dim=1,descending=True,stable=True).numpy()
        self.assertTrue(np.array_equal(actual,np.argsort(-scores,axis=1,kind='stable')))


if __name__ == "__main__":
    unittest.main()
