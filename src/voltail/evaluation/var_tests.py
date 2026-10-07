"""VaR backtests: Kupiec POF, Christoffersen independence / conditional coverage, Basel zones.

Let I_t = 1{r_t < -VaR_t} be the "hit" (exceedance) sequence. A correct alpha-VaR
model has hits that are (i) i.i.d. Bernoulli(alpha): the right *number* of hits
and (ii) not clustered in time.

* Kupiec (1995) proportion of failures: likelihood ratio of Bernoulli(pi_hat) vs
  Bernoulli(alpha). Tests (i) only. LR ~ chi2(1).
* Christoffersen (1998) independence: fits a two-state Markov chain to the hits and
  tests whether P(hit | hit yesterday) = P(hit | no hit yesterday). Tests (ii). LR ~ chi2(1).
* Conditional coverage = POF + independence, LR ~ chi2(2).
* Basel traffic light (BCBS 1996): with 250 days of 99% VaR, 0-4 exceptions = green,
  5-9 = yellow, 10+ = red. Those cut-offs are where the Binomial(250, 1%) CDF crosses
  95% and 99.99%; :func:`traffic_light` applies the same rule to any sample size/level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import xlogy


def hits(ret: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Exceedance indicator: 1 when the loss exceeds VaR (VaR given as a positive loss)."""
    return (np.asarray(ret) < -np.asarray(var)).astype(int)


def _bern_loglik(n1: float, n0: float, p: float) -> float:
    return float(xlogy(n1, p) + xlogy(n0, 1.0 - p))


@dataclass
class VaRTestResult:
    n: int
    exceedances: int
    expected: float
    rate: float
    lr_pof: float
    p_pof: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float


def kupiec_pof(hit: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec POF likelihood ratio and p-value."""
    hit = np.asarray(hit)
    n, x = len(hit), int(hit.sum())
    pi = x / n
    lr = -2.0 * (_bern_loglik(x, n - x, alpha) - _bern_loglik(x, n - x, pi))
    lr = max(lr, 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hit: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence LR and p-value (first-order Markov alternative)."""
    h = np.asarray(hit).astype(int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _bern_loglik(n01 + n11, n00 + n10, pi)
    ll_alt = _bern_loglik(n01, n00, pi01) + _bern_loglik(n11, n10, pi11)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return lr, float(stats.chi2.sf(lr, 1))


def var_backtest(ret: np.ndarray, var: np.ndarray, alpha: float) -> VaRTestResult:
    """Run Kupiec, Christoffersen independence and conditional coverage on one VaR series."""
    hit = hits(ret, var)
    lr_pof, p_pof = kupiec_pof(hit, alpha)
    lr_ind, p_ind = christoffersen_ind(hit)
    lr_cc = lr_pof + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, 2))
    n = len(hit)
    return VaRTestResult(
        n, int(hit.sum()), alpha * n, hit.mean(), lr_pof, p_pof, lr_ind, p_ind, lr_cc, p_cc
    )


def traffic_light(exceedances: int, n: int, alpha: float) -> str:
    """Basel-style zone from the Binomial(n, alpha) CDF: green < 95% <= yellow < 99.99% <= red."""
    cdf = stats.binom.cdf(exceedances, n, alpha)
    if cdf < 0.95:
        return "green"
    if cdf < 0.9999:
        return "yellow"
    return "red"


def rolling_zones(hit: np.ndarray, window: int = 250, alpha: float = 0.01) -> dict[str, float]:
    """Share of rolling ``window``-day windows in each traffic-light zone."""
    h = np.asarray(hit)
    if len(h) < window:
        return {"green": np.nan, "yellow": np.nan, "red": np.nan, "max_exceed_250d": np.nan}
    counts = np.convolve(h, np.ones(window, dtype=int), mode="valid")
    zones = np.array([traffic_light(int(c), window, alpha) for c in counts])
    return {
        "green": float((zones == "green").mean()),
        "yellow": float((zones == "yellow").mean()),
        "red": float((zones == "red").mean()),
        "max_exceed_250d": int(counts.max()),
    }
