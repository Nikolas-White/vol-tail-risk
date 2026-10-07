"""Cleaning raw OHLC into the processed schema, logging every row that is dropped.

Each asset keeps its *own* trading calendar (Bitcoin trades on weekends, the TSX
and the S&P 500 have different holidays). We never forward-fill prices to invent
trading days: a holiday simply has no row, and the next return spans the gap.
Cross-asset alignment happens later, in the feature builder, with strictly
lagged as-of joins.

Steps (each logged with counts and example dates in the quality report):

1. drop rows with missing / non-positive close;
2. drop weekend rows for non-24/7 assets;
3. drop *stale* rows: close unchanged from the previous row **and** high == low
   (free FX feeds publish such rows on holidays; they are not real trading days);
4. drop isolated *spike-and-revert* bad ticks (see :func:`find_spikes`);
5. compute log returns and flag rows whose OHLC fails consistency checks or whose
   high-low range is an implausible "wick" (> 25x the median of the previous 20 days)
   (``range_ok = False``); those rows keep their return but get NaN range proxies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from voltail.data.schema import validate_processed, validate_raw
from voltail.features.proxies import add_proxies

_MAX_EXAMPLES = 8


def _examples(idx: pd.Index) -> list[str]:
    return [str(pd.Timestamp(d).date()) for d in idx[:_MAX_EXAMPLES]]


def find_spikes(close: pd.Series, mad_mult: float, reversal_frac: float, lookback: int) -> pd.Index:
    """Dates of isolated bad ticks.

    A row t is flagged when ``|r_t| > mad_mult * s_t`` and ``r_{t+1}`` reverses at
    least ``reversal_frac`` of ``r_t`` (opposite sign), where ``s_t`` is a robust
    scale (1.4826 x median absolute return) over the previous ``lookback`` days.

    Note: using r_{t+1} means this cleaning step looks one day ahead. That mirrors
    how data vendors correct bad prints after the fact; it only removes a handful
    of obviously erroneous prints and is disclosed in METHODOLOGY.md.
    """
    r = np.log(close).diff()
    scale = 1.4826 * r.abs().shift(1).rolling(lookback, min_periods=max(20, lookback // 5)).median()
    big = r.abs() > mad_mult * scale
    nxt = r.shift(-1)
    reverts = (np.sign(nxt) == -np.sign(r)) & (nxt.abs() >= reversal_frac * r.abs())
    return r.index[(big & reverts).fillna(False).to_numpy()]


def clean_asset(
    raw: pd.DataFrame, asset: str, asset_class: str, ccfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean one asset's raw OHLC frame.

    Returns
    -------
    (processed, report)
        ``processed`` follows :data:`voltail.data.schema.PROCESSED_COLUMNS`;
        ``report`` lists every drop/flag with counts and example dates.
    """
    validate_raw(raw, asset)
    df = raw.copy()
    rep: dict[str, Any] = {"asset": asset, "raw_rows": len(df)}

    bad_close = df["close"].isna() | (df["close"] <= 0)
    rep["dropped_missing_close"] = {
        "n": int(bad_close.sum()),
        "examples": _examples(df.index[bad_close]),
    }
    df = df[~bad_close]

    if asset_class != "crypto":
        wkend = df.index.dayofweek >= 5
        rep["dropped_weekend"] = {"n": int(wkend.sum()), "examples": _examples(df.index[wkend])}
        df = df[~wkend]
    else:
        rep["dropped_weekend"] = {"n": 0, "examples": []}

    # Fill missing O/H/L with close so the row survives, but mark range as unusable below.
    ohl_missing = df[["open", "high", "low"]].isna().any(axis=1)
    rep["missing_ohl_filled_from_close"] = {
        "n": int(ohl_missing.sum()),
        "examples": _examples(df.index[ohl_missing]),
    }
    for c in ["open", "high", "low"]:
        df[c] = df[c].fillna(df["close"])

    stale = (df["close"] == df["close"].shift(1)) & (df["high"] == df["low"])
    rep["dropped_stale"] = {"n": int(stale.sum()), "examples": _examples(df.index[stale])}
    df = df[~stale]

    spikes = find_spikes(
        df["close"], ccfg["spike_mad_mult"], ccfg["spike_reversal_frac"], ccfg["spike_lookback"]
    )
    rep["dropped_spike_reversal"] = {"n": len(spikes), "examples": _examples(spikes)}
    df = df.drop(index=spikes)

    df["ret"] = 100.0 * np.log(df["close"]).diff()
    df = df.iloc[1:]  # first row has no return

    eps = 1e-9
    hi_ok = df["high"] >= df[["open", "close"]].max(axis=1) * (1 - 1e-6) - eps
    lo_ok = df["low"] <= df[["open", "close"]].min(axis=1) * (1 + 1e-6) + eps
    rng_ok = (df["high"] - df["low"]) / df["close"] > ccfg["min_range_frac"]
    range_ok = hi_ok & lo_ok & rng_ok & ~ohl_missing.reindex(df.index, fill_value=False)
    # Bad high/low "wicks": a day whose log(H/L) is more than `wick_mult` times the median
    # range of the previous 20 valid days. Uses only day t and earlier, so no lookahead.
    hl = np.log(df["high"] / df["low"])
    med = hl.where(range_ok).shift(1).rolling(20, min_periods=10).median()
    bad_wick = (hl > ccfg.get("wick_mult", 25.0) * med).fillna(False)
    rep["range_flagged_bad_wick"] = {
        "n": int(bad_wick.sum()),
        "examples": _examples(df.index[bad_wick]),
    }
    range_ok &= ~bad_wick
    df["range_ok"] = range_ok.astype(bool)
    rep["range_flagged_unusable"] = {
        "n": int((~range_ok).sum()),
        "pct": round(100.0 * float((~range_ok).mean()), 2),
        "examples": _examples(df.index[~range_ok]),
    }

    df = add_proxies(df)
    df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "ret",
            "range_ok",
            "sq_return",
            "parkinson",
            "garman_klass",
        ]
    ]
    validate_processed(df, asset)
    rep.update(
        processed_rows=len(df),
        first=str(df.index[0].date()),
        last=str(df.index[-1].date()),
        total_dropped=rep["raw_rows"] - len(df) - 1,  # -1: first row consumed by differencing
    )
    return df, rep
