from __future__ import annotations

import unittest

from utils.data import load_splits, sample_few_shot


class DataTests(unittest.TestCase):
    def test_full_split_sizes(self) -> None:
        train, validation, test, categories = load_splits("full", seed=42)
        self.assertEqual((len(train), len(validation), len(test), len(categories)), (9000, 1003, 3080, 77))

    def test_few_shot_is_balanced_and_reproducible(self) -> None:
        train, _, _, _ = load_splits("full", seed=42)
        first = sample_few_shot(train, shots=5, seed=42)
        second = sample_few_shot(train, shots=5, seed=42)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(first), 385)
        self.assertEqual(set(first["category"].value_counts()), {5})


if __name__ == "__main__":
    unittest.main()
