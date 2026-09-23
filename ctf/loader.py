"""Load a level's implementation from either `starter/` or `solution/`.

Which one is chosen by the BBM_IMPL env var (default: "solution", so CI and
`make solve` are green). `make check` sets BBM_IMPL=starter to run YOUR code.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def load(level_dir: str | Path, module: str, which: str | None = None) -> ModuleType:
    which = which or os.environ.get("BBM_IMPL", "solution")
    path = Path(level_dir) / which / f"{module}.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} (BBM_IMPL={which!r})")
    import sys

    name = f"{which}_{module}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # so @dataclass et al. can resolve the module
    spec.loader.exec_module(mod)
    return mod
