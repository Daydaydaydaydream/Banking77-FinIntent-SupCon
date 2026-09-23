from __future__ import annotations

import unittest

from baselines.bert_ft import resolve_training_schedule


class BertBaselineTests(unittest.TestCase):
    def test_few_shot_uses_minimum_and_maximum_step_budget(self) -> None:
        schedule = resolve_training_schedule(
            "5shot",
            min_steps=None,
            max_steps=None,
            eval_steps=None,
        )
        self.assertEqual(
            schedule,
            {
                "strategy": "steps",
                "min_steps": 300,
                "max_steps": 1000,
                "eval_steps": 50,
            },
        )

    def test_full_data_keeps_epoch_schedule(self) -> None:
        schedule = resolve_training_schedule(
            "full",
            min_steps=None,
            max_steps=None,
            eval_steps=None,
        )
        self.assertEqual(
            schedule,
            {
                "strategy": "epoch",
                "min_steps": 0,
                "max_steps": -1,
                "eval_steps": None,
            },
        )

    def test_rejects_minimum_above_maximum(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            resolve_training_schedule(
                "10shot",
                min_steps=500,
                max_steps=400,
                eval_steps=50,
            )

    def test_rejects_zero_maximum(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive or -1"):
            resolve_training_schedule(
                "5shot",
                min_steps=0,
                max_steps=0,
                eval_steps=50,
            )


if __name__ == "__main__":
    unittest.main()
