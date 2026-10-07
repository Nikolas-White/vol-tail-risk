"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voltail.config import load_config
from voltail.data.synthetic import CLASS_PARAMS, ohlc_from_returns, simulate_gjr_t, trading_calendar


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def default_cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def synthetic_equity() -> pd.DataFrame:
    """~8 years of synthetic GJR-GARCH-t equity data in the *processed* schema."""
    from voltail.data.clean import clean_asset

    gen = np.random.default_rng(7)
    p = CLASS_PARAMS["equity"]
    cal = trading_calendar("2010-01-01", "2017-12-31", "equity")
    ret, h = simulate_gjr_t(len(cal), p, gen)
    raw = ohlc_from_returns(ret, h, p, gen, cal)
    cfg = load_config()
    df, _ = clean_asset(raw, "SYN", "equity", cfg["data"]["cleaning"])
    return df
