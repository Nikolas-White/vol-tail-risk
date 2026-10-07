"""Ordinary least squares from scratch (closed form) with classical and HAC errors.

Closed form::

    beta_hat = (X'X)^{-1} X'y

We never form the inverse explicitly for the point estimate; we solve the normal
equations ``(X'X) b = X'y`` with a Cholesky-based solver, which is the
numerically stable way to apply the same formula.

Standard errors:

* classical:   Var(b) = s^2 (X'X)^{-1},  s^2 = e'e / (n - k)
* Newey-West:  Var(b) = (X'X)^{-1} S (X'X)^{-1},  S = sum_l w_l (G_l + G_l'),
  Bartlett weights w_l = 1 - l / (L + 1). Needed for overlapping targets (e.g.
  5-day-ahead volatility), where residuals are autocorrelated by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass
class OLSResult:
    params: np.ndarray
    bse: np.ndarray
    bse_hac: np.ndarray
    resid: np.ndarray
    sigma2: float
    r2: float
    nobs: int
    hac_lags: int
    cov_hac: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.params


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb bandwidth floor(4 (n/100)^(2/9)) (Newey & West 1994)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_covariance(x: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West (Bartlett kernel) covariance of OLS coefficients, no small-sample correction."""
    u = x * resid[:, None]
    s = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = u[lag:].T @ u[:-lag]
        s += w * (g + g.T)
    return xtx_inv @ s @ xtx_inv


def ols(y: np.ndarray, x: np.ndarray, hac_lags: int | None = None) -> OLSResult:
    """Fit ``y = X b + e`` by OLS. ``x`` must already contain a constant column if wanted."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    if n <= k:
        raise ValueError(f"OLS needs n > k (got n={n}, k={k})")
    xtx = x.T @ x
    xty = x.T @ y
    c, low = linalg.cho_factor(xtx)
    beta = linalg.cho_solve((c, low), xty)
    resid = y - x @ beta
    sigma2 = float(resid @ resid / (n - k))
    xtx_inv = linalg.cho_solve((c, low), np.eye(k))
    bse = np.sqrt(np.diag(sigma2 * xtx_inv))
    lags = newey_west_lags(n) if hac_lags is None else hac_lags
    cov_hac = hac_covariance(x, resid, xtx_inv, lags)
    bse_hac = np.sqrt(np.diag(cov_hac))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return OLSResult(beta, bse, bse_hac, resid, sigma2, r2, n, lags, cov_hac)
