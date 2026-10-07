"""Loss functions for volatility and joint VaR/ES forecasts.

QLIKE
-----
We use Patton's (2011) QLIKE in the form::

    L(proxy, f) = proxy / f + ln f

It differs from the "normalised" version  proxy/f - ln(proxy/f) - 1  only by a
term that depends on the proxy alone (1 + ln proxy), so model *differences*,
rankings, Diebold-Mariano and MCS results are identical - but this form stays
finite on days with a zero return (proxy = 0), which do occur in daily data.

Why QLIKE rather than MSE: Patton (2011) shows both are "robust" - ranking models
by expected loss with a noisy-but-unbiased proxy (like r^2) gives the same
ranking as with the true variance - but MSE is dominated by a handful of extreme
days (its loss scales with variance squared), while QLIKE penalises
*under*-prediction more than over-prediction and is far less driven by outliers.

FZ0 (Fissler & Ziegel 2016; Patton, Ziegel & Chen 2019)
-------------------------------------------------------
VaR alone is "elicitable" (the pinball loss), ES alone is not, but the *pair*
(VaR, ES) is jointly elicitable. The FZ0 loss, with v = -VaR and e = -ES in
return space (both negative)::

    L(y, v, e) = -1/(alpha e) 1{y <= v} (v - y) + v/e + ln(-e) - 1

is a strictly consistent scoring function: in expectation it is minimised by the
true VaR and ES, so it can rank tail-risk models (and feed DM/MCS tests).
"""

from __future__ import annotations

import numpy as np


def qlike(proxy: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE loss ``proxy / f + ln f`` (lower is better)."""
    f = np.asarray(forecast, dtype=float)
    if np.any(f <= 0):
        raise ValueError("variance forecasts must be strictly positive for QLIKE")
    return np.asarray(proxy, dtype=float) / f + np.log(f)


def qlike_normalised(proxy: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """``proxy/f - ln(proxy/f) - 1`` (>= 0, = 0 iff f = proxy). Requires proxy > 0."""
    ratio = np.asarray(proxy, dtype=float) / np.asarray(forecast, dtype=float)
    return ratio - np.log(ratio) - 1.0


def mse(proxy: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-observation squared error on the variance scale."""
    return (np.asarray(proxy, dtype=float) - np.asarray(forecast, dtype=float)) ** 2


def fz0(ret: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """Per-observation FZ0 loss. ``var`` and ``es`` are *positive* loss numbers (our convention)."""
    y = np.asarray(ret, dtype=float)
    v = -np.asarray(var, dtype=float)
    e = -np.asarray(es, dtype=float)
    if np.any(e >= 0):
        raise ValueError("ES must be strictly positive (a loss) for FZ0")
    hit = (y <= v).astype(float)
    return -hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1.0
