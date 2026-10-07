"""Stress / calm regime masks from the date ranges in the config."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regime_mask(index: pd.DatetimeIndex, spec: Any) -> np.ndarray:
    """Boolean mask for a regime given as [start, end] or a list of such ranges."""
    if isinstance(spec[0], (list, tuple)):
        mask = np.zeros(len(index), dtype=bool)
        for s in spec:
            mask |= regime_mask(index, s)
        return mask
    start, end = pd.Timestamp(spec[0]), pd.Timestamp(spec[1])
    return np.asarray((index >= start) & (index <= end))


def regime_masks(index: pd.DatetimeIndex, regimes: dict[str, Any]) -> dict[str, np.ndarray]:
    """``{"Full": all True, <regime>: mask, ...}`` restricted to regimes with any data."""
    out = {"Full": np.ones(len(index), dtype=bool)}
    for name, spec in regimes.items():
        m = regime_mask(index, spec)
        if m.any():
            out[name] = m
    return out
