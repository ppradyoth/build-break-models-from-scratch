"""Reproducible runs — attacks must land the same way every time (invariant I3)."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
