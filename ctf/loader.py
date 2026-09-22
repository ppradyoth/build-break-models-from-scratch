"""Load a level's implementation from either `starter/` or `solution/`.

Which one is chosen by the BBM_IMPL env var (default: "solution", so CI and
`make solve` are green). `make check` sets BBM_IMPL=starter to run YOUR code.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def load(level_dir: str | Path, module: str) -> ModuleType:
    which = os.environ.get("BBM_IMPL", "solution")
    path = Path(level_dir) / which / f"{module}.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} (BBM_IMPL={which!r})")
    spec = importlib.util.spec_from_file_location(f"{which}_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
