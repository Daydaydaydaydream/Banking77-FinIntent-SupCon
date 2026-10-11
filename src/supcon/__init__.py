"""Supervised contrastive learning with optional hard-negative mining."""

from __future__ import annotations

from supcon.train import run, run_hard_negative


METHOD = "supcon"
HN_METHOD = "supcon_hn"

__all__ = ["run", "run_hard_negative", "METHOD", "HN_METHOD"]