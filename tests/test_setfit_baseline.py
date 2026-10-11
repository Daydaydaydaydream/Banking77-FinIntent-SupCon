from __future__ import annotations

import unittest

from baselines.setfit_baseline import extract_history, training_arguments_kwargs


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "output_dir": "checkpoints",
        "batch_size": 16,
        "epochs": 1,
        "num_iterations": 20,
        "learning_rate": 2e-5,
        "max_length": 64,
        "seed": 42,
    }
    base.update(overrides)
    return training_arguments_kwargs(**base)  # type: ignore[arg-type]


class _State:
    def __init__(self, log_history: list[dict]) -> None:
        self.log_history = log_history


class _STTrainer:
    """Mirrors ``setfit.Trainer.st_trainer``: the sentence-transformers trainer."""

    def __init__(self, log_history: list[dict] | None) -> None:
        if log_history is not None:
            self.state = _State(log_history)


class _Trainer:
    def __init__(self, outer: list[dict] | None = None, st: list[dict] | None = None) -> None:
        if outer is not None:
            self.state = _State(outer)
        self.st_trainer = _STTrainer(st)


class SetFitBaselineTests(unittest.TestCase):
    def test_embedding_loss_selects_best_embedding_checkpoint(self) -> None:
        values = _kwargs(save_checkpoints=True)
        self.assertEqual(values["metric_for_best_model"], "embedding_loss")
        self.assertFalse(values["greater_is_better"])
        self.assertTrue(values["load_best_model_at_end"])
        self.assertEqual(values["eval_strategy"], values["save_strategy"])

    def test_checkpoints_disabled_when_model_is_not_retained(self) -> None:
        """``save_strategy='epoch'`` wrote ~1GB of intermediates per run.

        With no model retained there is nothing to resume from, so the trainer
        must not write checkpoints at all.
        """

        values = _kwargs(save_checkpoints=False)
        self.assertEqual(values["save_strategy"], "no")
        self.assertFalse(values["load_best_model_at_end"])
        self.assertIsNone(values["metric_for_best_model"])
        # Evaluation and logging still happen so the run stays observable.
        self.assertEqual(values["eval_strategy"], "epoch")
        self.assertEqual(values["logging_strategy"], "epoch")

    def test_history_is_read_from_the_inner_sentence_transformers_trainer(self) -> None:
        """The archived runs recorded ``[]`` because the outer state is empty.

        ``setfit.Trainer`` keeps the real log on ``st_trainer.state``; that is
        the attribute the original implementation missed.
        """

        inner_log = [{"epoch": 0, "eval_embedding_loss": 0.4}]
        trainer = _Trainer(outer=[], st=inner_log)
        self.assertEqual(extract_history(trainer), inner_log)

    def test_outer_state_wins_when_it_has_content(self) -> None:
        outer_log = [{"epoch": 0, "outer": True}]
        trainer = _Trainer(outer=outer_log, st=[{"epoch": 0, "inner": True}])
        self.assertEqual(extract_history(trainer), outer_log)

    def test_history_returns_empty_list_when_nothing_recorded(self) -> None:
        self.assertEqual(extract_history(_Trainer(outer=[], st=None)), [])


if __name__ == "__main__":
    unittest.main()