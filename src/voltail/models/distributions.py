"""Quantiles and Expected Shortfall of the standardized innovation distributions.

Conventions used everywhere in the project:

* returns ``r`` are in percent; a *loss* is ``-r``;
* ``VaR_alpha`` and ``ES_alpha`` are reported as **positive losses**:
  P(r < -VaR) = alpha and ES = -E[r | r < -VaR];
* "standardized" distributions have mean 0 and variance 1, so for a location-scale
  forecast r = mu + sigma z:  VaR = -(mu + sigma q_z(alpha)),  ES = -mu + sigma ES_z(alpha).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def alpha_label(alpha: float) -> str:
    """0.01 -> 'a010', 0.025 -> 'a025' (used in column names)."""
    return f"a{round(alpha * 1000):03d}"


def tail_grid(k: int, max_alpha: float = 0.025, min_u: float = 1e-4) -> np.ndarray:
    """Log-spaced probability levels in (0, max_alpha] at which tail quantiles are stored."""
    return np.geomspace(min_u, max_alpha, k)


# ----------------------------------------------------------------------------- normal
def normal_quantile(u: float | np.ndarray) -> np.ndarray:
    return stats.norm.ppf(u)


def normal_es(alpha: float) -> float:
    """ES of the standard normal as a positive number: phi(z_alpha) / alpha."""
    return float(stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)


# ----------------------------------------------------------------------------- Student t
def std_t_quantile(u: float | np.ndarray, nu: float) -> np.ndarray:
    """Quantile of a unit-variance Student-t with ``nu`` > 2 degrees of freedom."""
    return stats.t.ppf(u, nu) * np.sqrt((nu - 2.0) / nu)


def std_t_es(alpha: float, nu: float) -> float:
    """ES (positive) of a unit-variance Student-t.

    For the standard t: ES = (nu + t_a^2) / (nu - 1) * f_nu(t_a) / alpha, then rescale
    by sqrt((nu-2)/nu) to unit variance.
    """
    ta = stats.t.ppf(alpha, nu)
    es_std_t = (nu + ta**2) / (nu - 1.0) * stats.t.pdf(ta, nu) / alpha
    return float(es_std_t * np.sqrt((nu - 2.0) / nu))


# ----------------------------------------------------------------------------- Hansen skewed t
def skewt_quantile(u: float | np.ndarray, eta: float, lam: float) -> np.ndarray:
    """Quantile of Hansen's (1994) unit-variance skewed Student-t (as parameterised in ``arch``).

    ``eta`` > 2 is the degrees of freedom, ``lam`` in (-1, 1) the skewness; lam < 0 puts more
    mass in the left (loss) tail. lam = 0 gives the unit-variance Student-t.
    """
    from arch.univariate import SkewStudent

    return np.asarray(
        SkewStudent().ppf(np.atleast_1d(np.asarray(u, dtype=float)), [eta, lam])
    ).reshape(np.shape(u))


def skewt_es(alpha: float, eta: float, lam: float, n: int = 4000) -> float:
    """ES (positive) of the unit-variance skewed t: -(1/alpha) * integral_0^alpha q(u) du.

    No closed form is used; the integral is evaluated on a log-spaced grid
    (substituting u = e^s, so du = u ds), which handles the heavy left tail well.
    Verified against the closed-form Student-t ES at lam = 0 in the tests.
    """
    s = np.linspace(np.log(1e-12), np.log(alpha), n)
    u = np.exp(s)
    q = skewt_quantile(u, eta, lam)
    return float(-np.trapezoid(q * u, s) / alpha)


# ----------------------------------------------------------------------------- empirical
def empirical_var_es(z: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical alpha-quantile q (negative number) and tail mean E[z | z <= q] of a sample."""
    z = np.asarray(z)
    q = float(np.quantile(z, alpha))
    tail = z[z <= q]
    return q, float(tail.mean())


# ----------------------------------------------------------------------------- GPD / EVT
def gpd_tail_quantile(
    p: np.ndarray | float, u: float, xi: float, beta: float, frac_exceed: float
) -> np.ndarray:
    """Loss quantile at tail probability p from a peaks-over-threshold GPD fit.

    x_p = u + beta/xi * ((p / frac_exceed)^(-xi) - 1)   (xi != 0)
    x_p = u - beta * ln(p / frac_exceed)                 (xi == 0)
    """
    p = np.asarray(p, dtype=float)
    ratio = p / frac_exceed
    if abs(xi) < 1e-8:
        return u - beta * np.log(ratio)
    return u + beta / xi * (ratio ** (-xi) - 1.0)


def gpd_tail_es(x_p: np.ndarray | float, u: float, xi: float, beta: float) -> np.ndarray:
    """ES beyond the GPD quantile x_p: x_p/(1-xi) + (beta - xi u)/(1-xi), valid for xi < 1."""
    return np.asarray(x_p) / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
