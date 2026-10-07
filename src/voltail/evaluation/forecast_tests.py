"""Forecast-comparison statistics: Mincer-Zarnowitz, Diebold-Mariano, Model Confidence Set.

Mincer-Zarnowitz (1969)
    Regress the realised proxy on the forecast: proxy_t = a + b f_t + u_t. An unbiased,
    efficient forecast has a = 0 and b = 1; we Wald-test that jointly with Newey-West
    errors (residuals are autocorrelated, and for 5-day targets overlap by design).
    R^2 measures how much of the proxy's variation the forecast explains; with r^2 as
    the proxy it is *low by construction* (r^2 is very noisy), so compare R^2 across
    models, not to 1.

Diebold-Mariano (1995)
    d_t = L1_t - L2_t. H0: E[d_t] = 0 (equal accuracy). DM = mean(d) / sqrt(LRV/T) with a
    Newey-West long-run variance, plus the Harvey-Leybourne-Newbold (1997) small-sample
    correction and Student-t(T-1) reference distribution.

Model Confidence Set (Hansen, Lunde & Nason 2011)
    The set of models that contains the best model with probability >= 1 - size, using a
    block bootstrap. Models *inside* the MCS are statistically indistinguishable from the
    best; this is how we avoid over-claiming a winner. Implemented with ``arch.bootstrap.MCS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from voltail.models.ols import newey_west_lags, ols


@dataclass
class MZResult:
    a: float
    b: float
    se_a: float
    se_b: float
    r2: float
    wald: float
    p_value: float
    n: int


def mincer_zarnowitz(
    proxy: np.ndarray, forecast: np.ndarray, hac_lags: int | None = None
) -> MZResult:
    """MZ regression with a HAC Wald test of (a, b) = (0, 1)."""
    y = np.asarray(proxy, dtype=float)
    f = np.asarray(forecast, dtype=float)
    x = np.column_stack([np.ones_like(f), f])
    res = ols(y, x, hac_lags=hac_lags)
    diff = res.params - np.array([0.0, 1.0])
    wald = float(diff @ np.linalg.solve(res.cov_hac, diff))
    p = float(stats.chi2.sf(wald, 2))
    return MZResult(
        float(res.params[0]),
        float(res.params[1]),
        float(res.bse_hac[0]),
        float(res.bse_hac[1]),
        res.r2,
        wald,
        p,
        res.nobs,
    )


def newey_west_lrv(d: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of a series (Bartlett kernel)."""
    d = np.asarray(d, dtype=float) - np.mean(d)
    n = len(d)
    lrv = float(d @ d) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * w * float(d[lag:] @ d[:-lag]) / n
    return lrv


@dataclass
class DMResult:
    mean_diff: float
    stat: float
    p_value: float
    n: int


def diebold_mariano(
    loss1: np.ndarray, loss2: np.ndarray, horizon: int = 1, max_lag: int | None = None
) -> DMResult:
    """DM test of equal expected loss (two-sided), HLN-corrected.

    A *negative* statistic means model 1 has lower loss (is better).
    """
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    lags = max(horizon - 1, newey_west_lags(n)) if max_lag is None else max_lag
    lrv = newey_west_lrv(d, lags)
    if lrv <= 0:
        return DMResult(float(d.mean()), 0.0, 1.0, n)
    dm = d.mean() / np.sqrt(lrv / n)
    h = horizon
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(float(d.mean()), stat, p, n)


def dm_matrix(losses: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise DM statistics and p-values (row model vs column model)."""
    models = list(losses.columns)
    stat = pd.DataFrame(np.nan, index=models, columns=models)
    pval = stat.copy()
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            r = diebold_mariano(losses[a].to_numpy(), losses[b].to_numpy(), horizon)
            stat.loc[a, b], stat.loc[b, a] = r.stat, -r.stat
            pval.loc[a, b] = pval.loc[b, a] = r.p_value
    return stat, pval


def model_confidence_set(
    losses: pd.DataFrame, size: float, reps: int, block_size: int, seed: int
) -> pd.DataFrame:
    """MCS p-values and membership for each model column of ``losses``.

    Returns a frame with columns ``mcs_pvalue`` and ``in_mcs`` indexed by model.
    """
    from arch.bootstrap import MCS

    mcs = MCS(
        losses.to_numpy(),
        size=size,
        reps=reps,
        block_size=block_size,
        method="R",
        bootstrap="stationary",
        seed=seed,
    )
    mcs.compute()
    pv = mcs.pvalues["Pvalue"]
    pv.index = [losses.columns[int(i)] for i in pv.index]
    out = pd.DataFrame({"mcs_pvalue": pv})
    out["in_mcs"] = out["mcs_pvalue"] >= size
    return out.reindex(losses.columns)
