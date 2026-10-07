"""Feature matrix for the gradient-boosting volatility model.

Every feature in row t is computable at the close of day t from the asset's own
history, plus *strictly earlier* observations of a cross-asset driver (the S&P 500
by default). Closing times differ across markets (US equities 16:00 ET, the free
FX feed's daily bar ends later, Bitcoin trades 24/7), so a same-date join could
leak information; ``merge_asof(..., allow_exact_matches=False)`` guarantees the
driver's value comes from an earlier calendar date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from voltail.models.baselines import ewma_variance

EPS = 1e-4  # floor inside logs (percent^2), avoids log(0) on zero-return days


def own_features(
    df: pd.DataFrame, target_col: str = "garman_klass", ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """Lagged volatility / return / range features of one asset, aligned at origin t."""
    v = df[target_col].ffill()
    r = df["ret"]
    r2 = df["sq_return"]
    pk = df["parkinson"].ffill()
    f = pd.DataFrame(index=df.index)
    f["lgk_1"] = np.log(v + EPS)
    f["lgk_5"] = np.log(v.rolling(5).mean() + EPS)
    f["lgk_22"] = np.log(v.rolling(22).mean() + EPS)
    f["lpk_1"] = np.log(pk + EPS)
    f["lr2_1"] = np.log(r2 + EPS)
    f["lr2_5"] = np.log(r2.rolling(5).mean() + EPS)
    f["lr2_22"] = np.log(r2.rolling(22).mean() + EPS)
    f["ret_1"] = r
    f["neg_ret_1"] = np.minimum(r, 0.0)
    f["ret_5"] = r.rolling(5).sum()
    f["l_ewma"] = np.log(ewma_variance(r.to_numpy(), ewma_lambda) + EPS)
    f["l_roll63"] = np.log(r.rolling(63).var() + EPS)
    return f


def cross_features(
    target_index: pd.DatetimeIndex, driver: pd.DataFrame, prefix: str = "x_"
) -> pd.DataFrame:
    """Driver features as of the last driver date *strictly before* each target date."""
    df_drv = own_features(driver)[["lgk_1", "lgk_5", "ret_1"]].dropna()
    left = pd.DataFrame({"date": target_index})
    right = (
        df_drv.add_prefix(prefix)
        .reset_index()
        .rename(columns={df_drv.index.name or "index": "date"})
    )
    right = right.rename(columns={right.columns[0]: "date"})
    merged = pd.merge_asof(left, right, on="date", direction="backward", allow_exact_matches=False)
    return merged.set_index("date").set_axis(target_index, axis=0)


def build_features(
    df: pd.DataFrame,
    driver: pd.DataFrame | None,
    target_col: str = "garman_klass",
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """Own features + optional cross-asset driver features."""
    f = own_features(df, target_col, ewma_lambda)
    if driver is not None:
        f = f.join(cross_features(df.index, driver))
    return f


def ml_target(v: pd.Series, horizon: int) -> pd.Series:
    """log of the mean proxy over t+1..t+h (NaN if any day missing or beyond the data)."""
    fwd = pd.concat([v.shift(-k) for k in range(1, horizon + 1)], axis=1).mean(axis=1, skipna=False)
    return np.log(fwd + EPS)
