"""Baseline models that need no parameter estimation.

* ``roll21``             rolling sample variance of the last 21 returns (1 month)
* ``ewma``                 RiskMetrics: s2_{t+1} = lambda s2_t + (1 - lambda) r_t^2, lambda = 0.94,
                           with Gaussian VaR/ES (the 1996 RiskMetrics convention: zero mean)
* ``hs250``              historical simulation: VaR/ES are the empirical quantile / tail
                           mean of the last W returns, no model at all

All are computed causally: the value at origin t uses returns r_{t-W+1..t} only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from voltail.models.base import BlockOutput, ModelFamily
from voltail.models.distributions import alpha_label, normal_es, normal_quantile


def ewma_variance(r: np.ndarray, lam: float, init_obs: int = 30) -> np.ndarray:
    """EWMA variance at each origin t (forecast for t+1), using r_0..r_t.

    Initialised with the mean of the first ``init_obs`` squared returns (a fixed
    historical value, so the recursion stays causal).
    """
    r = np.asarray(r, dtype=float)
    init = float(np.mean(r[: min(init_obs, len(r))] ** 2))
    # s2_t = lam * s2_{t-1} + (1 - lam) r_t^2 is a first-order linear filter
    s2, _ = signal.lfilter([1.0 - lam], [1.0, -lam], r**2, zi=np.array([lam * init]))
    return s2


def hs_quantiles(r: np.ndarray, window: int, probs: np.ndarray) -> np.ndarray:
    """Empirical quantiles of the trailing ``window`` returns at each origin (NaN before)."""
    r = np.asarray(r, dtype=float)
    out = np.full((len(r), len(probs)), np.nan)
    if len(r) >= window:
        win = sliding_window_view(r, window)  # row j = r[j .. j+window-1] -> origin j+window-1
        out[window - 1 :] = np.quantile(win, probs, axis=1).T
    return out


def hs_es(r: np.ndarray, window: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """HS VaR (positive) and ES (positive) at each origin from the trailing window."""
    r = np.asarray(r, dtype=float)
    var = np.full(len(r), np.nan)
    es = np.full(len(r), np.nan)
    if len(r) >= window:
        win = sliding_window_view(r, window)
        q = np.quantile(win, alpha, axis=1)
        mask = win <= q[:, None]
        tail_mean = (win * mask).sum(axis=1) / mask.sum(axis=1)
        var[window - 1 :] = -q
        es[window - 1 :] = -tail_mean
    return var, es


class BaselineFamily(ModelFamily):
    """roll21, ewma, hs250 (no fitting, recomputed causally each block)."""

    name = "baseline"
    refit_key = None

    def run_block(self, hist: pd.DataFrame, train_end: int) -> BlockOutput:
        r = hist["ret"].to_numpy()
        n = len(r)
        sl = slice(train_end, n)
        out = BlockOutput()
        for w in (21,):
            v = pd.Series(r).rolling(w).var(ddof=1).to_numpy()
            out.cols[f"var1_roll{w}"] = v[sl]
            out.cols[f"var5_roll{w}"] = v[sl]
        lam = self.cfg["models"]["ewma_lambda"]
        s2 = ewma_variance(r, lam)[sl]
        out.cols["var1_ewma"] = s2
        out.cols["var5_ewma"] = s2  # EWMA's multi-step forecast is flat
        zero = np.zeros_like(s2)
        out.add_location_scale_risk(
            "ewma", zero, np.sqrt(s2), self.alphas, normal_quantile, normal_es, self.grid_u
        )
        for w in (250,):
            name = f"hs{w}"
            # only the windows ending at the block's origins are needed
            lo = max(0, train_end - w + 1)
            sub = r[lo:]
            keep = slice(train_end - lo, len(sub))
            for a in self.alphas:
                var, es = hs_es(sub, w, a)
                lab = alpha_label(a)
                out.cols[f"VaR_{lab}_{name}"] = var[keep]
                out.cols[f"ES_{lab}_{name}"] = es[keep]
            out.grids[name] = hs_quantiles(sub, w, self.grid_u)[keep]
        return out
