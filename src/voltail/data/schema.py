"""Canonical column schema shared by every stage of the pipeline.

Raw files (``data/raw/<ASSET>.parquet``) hold exactly :data:`RAW_COLUMNS` with a
tz-naive ``DatetimeIndex`` named ``date``. Processed files add returns and the
volatility proxies. Every downstream module is written against this schema, so
the pipeline works identically whether data came from yfinance, a manual CSV
download, or the synthetic generator.
"""

from __future__ import annotations

import pandas as pd

RAW_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

PROCESSED_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "ret",  # close-to-close log return, in percent (100 * ln(C_t / C_{t-1}))
    "range_ok",  # bool: OHLC passed consistency checks -> range proxies usable
    "sq_return",  # ret**2 (percent^2)
    "parkinson",  # Parkinson (1980) range variance (percent^2)
    "garman_klass",  # Garman-Klass (1980) OHLC variance (percent^2)
]


def validate_raw(df: pd.DataFrame, asset: str) -> None:
    """Raise ``ValueError`` if ``df`` does not follow the raw schema."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{asset}: raw data missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{asset}: index must be a DatetimeIndex")
    if df.index.tz is not None:
        raise ValueError(f"{asset}: index must be tz-naive")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{asset}: index must be sorted ascending")


def validate_processed(df: pd.DataFrame, asset: str) -> None:
    """Raise ``ValueError`` if ``df`` does not follow the processed schema."""
    missing = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{asset}: processed data missing columns {missing}")
    if df["ret"].isna().any():
        raise ValueError(f"{asset}: processed returns contain NaN")
    if not df.index.is_unique:
        raise ValueError(f"{asset}: duplicate dates in processed data")
