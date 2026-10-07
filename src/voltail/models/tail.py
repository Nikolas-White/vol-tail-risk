"""Tail-risk models: filtered historical simulation (FHS) and Extreme Value Theory (EVT).

FHS (Barone-Adesi et al. 1999)
    Fit a GARCH model, compute standardized residuals z_t = (r_t - mu) / sigma_t on the
    estimation window, and use their *empirical* distribution instead of assuming
    normal/t:  VaR_{t+1} = -(mu + sigma_{t+1} q_alpha(z)).  GARCH handles the time-varying
    scale; history supplies the shape of the tail (skew, fat tails) without a parametric
    assumption.

Conditional EVT (McNeil & Frey 2000)
    Same GARCH filter, but the far tail of the standardized losses x = -z is modelled
    with a Generalized Pareto Distribution (GPD) fitted to exceedances over a high
    threshold u (peaks over threshold). Pickands-Balkema-de Haan: for large u, the
    excess x - u | x > u is approximately GPD(xi, beta) whatever the parent
    distribution. This extrapolates beyond the observed data more sensibly than raw
    empirical quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from voltail.models.base import BlockOutput
from voltail.models.distributions import alpha_label, gpd_tail_es, gpd_tail_quantile


@dataclass
class GPDFit:
    xi: float
    beta: float
    threshold: float
    frac_exceed: float
    n_exceed: int
    clipped: bool

    def loss_quantile(self, p: np.ndarray | float) -> np.ndarray:
        return gpd_tail_quantile(p, self.threshold, self.xi, self.beta, self.frac_exceed)

    def loss_es(self, p: float) -> float:
        return float(gpd_tail_es(self.loss_quantile(p), self.threshold, self.xi, self.beta))


def gpd_nll(theta: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of GPD(xi, beta) for excesses y >= 0."""
    xi, beta = theta
    if beta <= 0:
        return 1e12
    if abs(xi) < 1e-8:
        return len(y) * np.log(beta) + y.sum() / beta
    arg = 1.0 + xi * y / beta
    if np.any(arg <= 0):
        return 1e12
    return float(len(y) * np.log(beta) + (1.0 + 1.0 / xi) * np.log(arg).sum())


def fit_gpd(losses: np.ndarray, threshold_q: float, xi_max: float = 0.95) -> GPDFit:
    """Peaks-over-threshold GPD fit by maximum likelihood (from scratch, Nelder-Mead)."""
    losses = np.asarray(losses, dtype=float)
    u = float(np.quantile(losses, threshold_q))
    y = losses[losses > u] - u
    if len(y) < 15:
        raise ValueError(f"too few exceedances ({len(y)}) for a GPD fit")
    # method-of-moments start: xi0 = 0.5 (1 - m^2/s^2), beta0 = 0.5 m (1 + m^2/s^2)
    m, s2 = y.mean(), y.var()
    x0 = np.array([0.5 * (1 - m**2 / s2), 0.5 * m * (1 + m**2 / s2)])
    res = optimize.minimize(
        gpd_nll,
        x0,
        args=(y,),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000},
    )
    xi, beta = res.x
    clipped = xi > xi_max
    xi = min(xi, xi_max)  # xi >= 1 implies infinite ES; cap and record it
    return GPDFit(float(xi), float(beta), u, len(y) / len(losses), len(y), bool(clipped))


def fhs_risk(
    out: BlockOutput,
    model: str,
    z: np.ndarray,
    mu: float,
    sigma: np.ndarray,
    alphas: list[float],
    grid_u: np.ndarray,
) -> None:
    """Add FHS VaR/ES/grid to ``out`` given training standardized residuals ``z``."""
    for a in alphas:
        lab = alpha_label(a)
        q = np.quantile(z, a)
        tail = z[z <= q].mean()
        out.cols[f"VaR_{lab}_{model}"] = -(mu + sigma * q)
        out.cols[f"ES_{lab}_{model}"] = -(mu + sigma * tail)
    out.grids[model] = mu + sigma[:, None] * np.quantile(z, grid_u)[None, :]


def evt_risk(
    out: BlockOutput,
    model: str,
    z: np.ndarray,
    mu: float,
    sigma: np.ndarray,
    alphas: list[float],
    grid_u: np.ndarray,
    threshold_q: float,
) -> GPDFit:
    """Add conditional-EVT VaR/ES/grid to ``out`` (GPD on standardized losses -z)."""
    g = fit_gpd(-z, threshold_q)
    for a in alphas:
        lab = alpha_label(a)
        out.cols[f"VaR_{lab}_{model}"] = -mu + sigma * float(g.loss_quantile(a))
        out.cols[f"ES_{lab}_{model}"] = -mu + sigma * g.loss_es(a)
    out.grids[model] = mu - sigma[:, None] * g.loss_quantile(grid_u)[None, :]
    return g
