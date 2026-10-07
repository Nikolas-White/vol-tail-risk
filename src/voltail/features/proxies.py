"""Volatility proxies computable from free daily OHLC data.

True daily variance is unobservable. With intraday data one would use realized
variance (sum of squared 5-minute returns); free daily data does not allow that,
so we use three noisy *proxies*, all in percent^2 units:

* ``sq_return``   r_t^2. Conditionally unbiased for close-to-close variance but
                  extremely noisy (a single draw of a chi-square-like variable).
* ``parkinson``   (ln H/L)^2 / (4 ln 2). Uses the intraday high-low range;
                  roughly 5x more efficient than r^2 under Brownian motion.
* ``garman_klass``  0.5 (ln H/L)^2 - (2 ln 2 - 1)(ln C/O)^2. Adds open/close;
                  roughly 7x more efficient than r^2 under Brownian motion.

Range estimators measure the *open-to-close* (trading-hours) variance only, so
they miss the overnight gap and are biased low relative to close-to-close
variance. :func:`expanding_scale` estimates the ratio E[r^2] / E[range proxy]
using **only past data** so the range proxies can be put on the close-to-close
scale without lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


def squared_return(ret: pd.Series) -> pd.Series:
    """Squared percent log return."""
    return ret**2


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) variance estimator in percent^2."""
    hl = np.log(high / low)
    return 1e4 * hl**2 / (4.0 * LN2)


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass (1980) variance estimator in percent^2 (clipped at a tiny positive floor)."""
    hl = np.log(high / low)
    co = np.log(close / open_)
    gk = 1e4 * (0.5 * hl**2 - (2.0 * LN2 - 1.0) * co**2)
    # GK can be (rarely) non-positive when |C-O| is close to H-L; floor at a fraction of Parkinson.
    pk = parkinson(high, low)
    return gk.where(gk > 0.05 * pk, 0.05 * pk)


def expanding_scale(numerator: pd.Series, denominator: pd.Series, min_obs: int) -> pd.Series:
    """Ratio of expanding means sum(num)/sum(den), using data up to and including t.

    Rows where either input is NaN are ignored. Returns NaN until ``min_obs``
    valid pairs have been seen. The value at t uses only information available at
    the close of day t, so it can be applied to a day-t proxy without lookahead.
    """
    valid = numerator.notna() & denominator.notna()
    num = numerator.where(valid, 0.0).cumsum()
    den = denominator.where(valid, 0.0).cumsum()
    n = valid.cumsum()
    ratio = num / den
    return ratio.where(n >= min_obs)


def add_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sq_return``, ``parkinson`` and ``garman_klass`` columns.

    Range proxies are set to NaN on rows where ``range_ok`` is False.
    """
    out = df.copy()
    out["sq_return"] = squared_return(out["ret"])
    pk = parkinson(out["high"], out["low"])
    gk = garman_klass(out["open"], out["high"], out["low"], out["close"])
    out["parkinson"] = pk.where(out["range_ok"])
    out["garman_klass"] = gk.where(out["range_ok"])
    return out


def scaled_range_proxy(df: pd.DataFrame, column: str, min_obs: int) -> pd.Series:
    """Range proxy rescaled to close-to-close variance with an expanding, past-only ratio."""
    scale = expanding_scale(df["sq_return"], df[column], min_obs)
    return df[column] * scale
