"""Reproducibility helpers.

Two rules keep runs reproducible:

1. ``set_global_seed`` is called once at the start of every script/experiment.
2. Components that draw random numbers (workload generator, thermal noise,
   epsilon-greedy exploration, ...) own a private ``numpy.random.Generator``
   created with ``make_rng`` instead of using the global ``np.random`` state.
   Two components seeded the same way therefore produce the same stream
   regardless of what else ran in between.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is only needed from Step 10 (DQN) onwards
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int | None) -> np.random.Generator:
    """Independent random generator for one component."""
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *labels: str | int) -> int:
    """Deterministically derive a sub-seed, e.g. ``derive_seed(42, "workloads")``.

    Gives each component its own stream from a single experiment seed without
    the streams overlapping. Uses NumPy's SeedSequence, so it is stable across
    Python processes (unlike ``hash()``).
    """
    entropy = [base_seed] + [
        label if isinstance(label, int) else int.from_bytes(label.encode(), "little") % (2**32)
        for label in labels
    ]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])
