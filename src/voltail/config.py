"""Configuration loading.

All parameters live in YAML files under ``config/``. ``default.yaml`` is always
loaded first; an optional override file is deep-merged on top of it. A handful
of keys (asset lists, regime definitions, tuning grids) are *replaced* rather
than merged, because merging two asset lists would silently mix universes.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"

# Keys whose values replace (not merge with) the default when overridden.
REPLACE_KEYS = frozenset({"assets", "regimes", "tune_grid", "refit_every"})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            if key in REPLACE_KEYS:
                out[key] = copy.deepcopy(value)
            else:
                out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load ``default.yaml`` and deep-merge an optional override file.

    Parameters
    ----------
    path:
        Path to an override YAML (e.g. ``config/synthetic.yaml``). ``None`` or the
        default path itself returns the defaults. The environment variable
        ``VOLTAIL_CONFIG`` is used when ``path`` is ``None``.

    Returns
    -------
    dict
        The merged configuration. Relative paths under ``paths`` are resolved
        against the project root.
    """
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    path = path or os.environ.get("VOLTAIL_CONFIG") or None
    if path is not None and not Path(path).exists() and (PROJECT_ROOT / path).exists():
        path = PROJECT_ROOT / path  # allow repo-relative paths from any working directory
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, override)
    for key, rel in cfg["paths"].items():
        p = Path(rel)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def set_seeds(seed: int) -> None:
    """Fix every RNG we touch so reruns are bit-for-bit deterministic."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global state used by some libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create every output directory referenced in ``cfg['paths']``."""
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
