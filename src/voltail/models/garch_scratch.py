"""GARCH(1,1) with Gaussian innovations, estimated from scratch by maximum likelihood.

Model (returns in percent)::

    r_t      = mu + e_t
    e_t      = sigma_t * z_t,             z_t ~ N(0, 1)
    sigma2_t = omega + alpha * e_{t-1}^2 + beta * sigma2_{t-1}

Log-likelihood (what we maximise)::

    l(theta) = -1/2 * sum_t [ ln(2 pi) + ln sigma2_t + e_t^2 / sigma2_t ]

Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (covariance
stationarity). The recursion is started from the same "backcast" the ``arch``
package uses (an exponentially weighted average of the first 75 squared
residuals), so our estimates can be compared to ``arch`` like-for-like in
``tests/test_garch_scratch.py``.

Implementation note: the variance recursion is a first-order linear filter in
e_{t-1}^2, so we evaluate it with ``scipy.signal.lfilter`` instead of a Python
loop. That makes one likelihood evaluation O(n) in C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal

LOG_2PI = np.log(2.0 * np.pi)


def arch_style_backcast(resid: np.ndarray, tau: int = 75, lam: float = 0.94) -> float:
    """EWMA of the first ``tau`` squared residuals (identical to ``arch``'s default)."""
    tau = min(tau, len(resid))
    w = lam ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(resid[:tau] ** 2 * w))


def garch_variance(
    resid: np.ndarray, omega: float, alpha: float, beta: float, backcast: float
) -> np.ndarray:
    """Conditional variances sigma2_t for t = 0..n-1 given residuals.

    sigma2_0 = omega + (alpha + beta) * backcast, then the usual recursion. Only
    e_0..e_{t-1} enter sigma2_t, so the series is "predictable" (no lookahead).
    """
    e2_lag = np.empty_like(resid)
    e2_lag[0] = backcast
    e2_lag[1:] = resid[:-1] ** 2
    x = omega + alpha * e2_lag
    y, _ = signal.lfilter([1.0], [1.0, -beta], x, zi=np.array([beta * backcast]))
    return y


def gaussian_nll(params: np.ndarray, y: np.ndarray, backcast: float) -> float:
    """Negative Gaussian log-likelihood of GARCH(1,1) with constant mean."""
    mu, omega, alpha, beta = params
    resid = y - mu
    s2 = garch_variance(resid, omega, alpha, beta, backcast)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e12
    return 0.5 * float(np.sum(LOG_2PI + np.log(s2) + resid**2 / s2))


@dataclass
class GarchFit:
    """Result of :func:`fit_garch11`."""

    mu: float
    omega: float
    alpha: float
    beta: float
    loglik: float
    backcast: float
    converged: bool
    std_errors: dict[str, float] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "omega": self.omega, "alpha": self.alpha, "beta": self.beta}

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def uncond_var(self) -> float:
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay by half: ln(0.5) / ln(alpha + beta)."""
        return float(np.log(0.5) / np.log(self.persistence))

    def variance_path(self, y: np.ndarray) -> np.ndarray:
        """In-sample conditional variances for returns ``y``."""
        return garch_variance(
            np.asarray(y) - self.mu, self.omega, self.alpha, self.beta, self.backcast
        )

    def forecast(self, last_resid: float, last_var: float, horizon: int) -> np.ndarray:
        """h-step-ahead variance forecasts made at time t (given e_t and sigma2_t).

        sigma2_{t+1} = omega + alpha e_t^2 + beta sigma2_t
        sigma2_{t+h} = sigma2_bar + (alpha + beta)^(h-1) (sigma2_{t+1} - sigma2_bar)
        """
        s1 = self.omega + self.alpha * last_resid**2 + self.beta * last_var
        h = np.arange(horizon)
        return self.uncond_var + self.persistence**h * (s1 - self.uncond_var)


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian (relative step) of scalar function ``f`` at ``x``."""
    k = len(x)
    hess = np.zeros((k, k))
    step = eps * np.maximum(np.abs(x), 1e-2)
    for i in range(k):
        for j in range(i, k):
            x_pp, x_pm, x_mp, x_mm = x.copy(), x.copy(), x.copy(), x.copy()
            x_pp[i] += step[i]
            x_pp[j] += step[j]
            x_pm[i] += step[i]
            x_pm[j] -= step[j]
            x_mp[i] -= step[i]
            x_mp[j] += step[j]
            x_mm[i] -= step[i]
            x_mm[j] -= step[j]
            hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * step[i] * step[j])
            hess[j, i] = hess[i, j]
    return hess


def fit_garch11(y: np.ndarray, compute_se: bool = True) -> GarchFit:
    """Estimate GARCH(1,1)-N by maximum likelihood with SLSQP.

    Several starting points are tried (as ``arch`` does) and the best optimum kept.
    """
    y = np.asarray(y, dtype=float)
    var_y = float(np.var(y))
    backcast = arch_style_backcast(y - y.mean())
    bounds = [
        (-10 * abs(y.mean()) - 1.0, 10 * abs(y.mean()) + 1.0),
        (1e-8 * var_y, 10 * var_y),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    cons = [{"type": "ineq", "fun": lambda p: 0.99999 - p[2] - p[3]}]
    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95), (0.15, 0.75)]:
        x0 = np.array([y.mean(), var_y * (1 - a0 - b0), a0, b0])
        res = optimize.minimize(
            gaussian_nll,
            x0,
            args=(y, backcast),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    mu, omega, alpha, beta = best.x
    se: dict[str, float] = {}
    if compute_se:
        hess = _numerical_hessian(lambda p: gaussian_nll(p, y, backcast), best.x)
        try:
            cov = np.linalg.inv(hess)
            se = dict(
                zip(
                    ["mu", "omega", "alpha", "beta"],
                    np.sqrt(np.clip(np.diag(cov), 0, None)),
                    strict=True,
                )
            )
        except np.linalg.LinAlgError:
            se = {}
    return GarchFit(
        float(mu),
        float(omega),
        float(alpha),
        float(beta),
        -float(best.fun),
        backcast,
        bool(best.success),
        se,
    )
