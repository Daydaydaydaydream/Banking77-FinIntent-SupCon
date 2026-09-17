"""Unified randomness control for every training entry point.

Deep-learning runs depend on four independent randomness sources: Python's
``random``, NumPy, PyTorch's CPU/CUDA/MPS generators, and Python's string
hashing (``PYTHONHASHSEED``).  Seeding only one of them yields runs that look
reproducible but drift between processes.  :func:`seed_everything` seeds all of
them and reports what it actually managed to set, because deterministic
algorithms are not available for every operator and silently falling back is
common.

Torch is imported lazily so that data-preparation code can seed NumPy and
Python without pulling in a multi-hundred-megabyte dependency.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Sequence

import numpy as np

# Large prime used to mix a base seed with integer parts (epoch, fold, ...).
_MIX_PRIME = 1_000_003
_SEED_MODULUS = 2**31 - 1


def derive_seed(base: int, *parts: int) -> int:
    """Return a stable child seed derived from ``base`` and integer ``parts``.

    Replaces patterns such as ``SEED + epoch``, which couple the effective seed
    to loop indices and make two different (seed, epoch) pairs collide.
    """
    value = int(base) % _SEED_MODULUS
    for part in parts:
        value = (value * _MIX_PRIME + int(part)) % _SEED_MODULUS
    return int(value)


def seed_everything(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Seed every available randomness source and report the resulting state.

    Parameters
    ----------
    seed:
        Integer seed shared by all sources.
    deterministic:
        When ``True``, request ``torch.use_deterministic_algorithms(True)`` and
        disable cuDNN autotuning.  Operators without a deterministic
        implementation raise at runtime; the flag is recorded as ``False`` if it
        could not be applied.

    Returns
    -------
    dict
        Diagnostic record suitable for ``env.json``.  Keys reporting
        availability let a reader tell "seeded" apart from "not applicable".
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    state: Dict[str, Any] = {
        "seed": seed,
        "requested_deterministic": deterministic,
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
    }

    try:
        import torch
    except ImportError:
        state["torch_available"] = False
        return state

    state["torch_available"] = True
    state["torch_version"] = torch.__version__
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    state["cuda_available"] = bool(torch.cuda.is_available())

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if mps_available:
        torch.mps.manual_seed(seed)
    state["mps_available"] = mps_available

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            state["deterministic_algorithms"] = True
        except (RuntimeError, AttributeError) as exc:  # operator lacks a det. impl
            state["deterministic_algorithms"] = False
            state["deterministic_error"] = str(exc)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        state["deterministic_algorithms"] = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cudnn"):
        state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)

    return state


def torch_generator(seed: int):
    """Return a ``torch.Generator`` seeded from ``seed`` for DataLoader use."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed) % _SEED_MODULUS)
    return generator


def numpy_rng(seed: int) -> np.random.Generator:
    """Return a NumPy generator seeded from ``seed`` (never ``SEED + epoch``)."""
    return np.random.default_rng(int(seed) % _SEED_MODULUS)


def seed_report(state: Dict[str, Any]) -> str:
    """Render :func:`seed_everything` output as a single log line."""
    parts = [
        f"seed={state.get('seed')}",
        f"torch={state.get('torch_available', False)}",
        f"det={state.get('deterministic_algorithms', False)}",
    ]
    if state.get("mps_available"):
        parts.append("mps")
    if state.get("cuda_available"):
        parts.append("cuda")
    return "[" + ", ".join(parts) + "]"


def seeds_from_args(seeds: Sequence[int] | None) -> list:
    """Normalise a seed argument into a list of integers."""
    if seeds is None:
        return []
    return [int(s) for s in seeds]
