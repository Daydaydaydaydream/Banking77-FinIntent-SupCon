from __future__ import annotations

import unittest

from baselines.setfit_baseline import training_arguments_kwargs


class SetFitBaselineTests(unittest.TestCase):
    def test_embedding_loss_selects_best_embedding_checkpoint(self) -> None:
        values = training_arguments_kwargs(
            output_dir="checkpoints",
            batch_size=16,
            epochs=1,
            num_iterations=20,
            learning_rate=2e-5,
            max_length=64,
            seed=42,
        )
        self.assertEqual(values["metric_for_best_model"], "embedding_loss")
        self.assertFalse(values["greater_is_better"])
        self.assertTrue(values["load_best_model_at_end"])
        self.assertEqual(values["eval_strategy"], values["save_strategy"])


if __name__ == "__main__":
    unittest.main()
