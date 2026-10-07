"""Expected Shortfall backtest: Acerbi & Szekely (2014) "Z2" test with Monte Carlo p-values.

Statistic (losses L_t = -r_t > 0, ES_t > 0, hits I_t = 1{r_t < -VaR_t})::

    Z2 = 1 - (1 / (T alpha)) * sum_t  L_t I_t / ES_t

Under H0 (the predictive distributions are correct) E[Z2] = 0. Z2 < 0 means realised
tail losses are larger and/or more frequent than the model's ES implies (risk
under-estimated). The test is one-sided: reject when Z2 is too negative.

The null distribution depends on each day's predictive tail, so we simulate it.
For every model and day the walk-forward engine stores the predictive return
quantile q_t(u) on a log-spaced grid of tail probabilities u in [1e-4, 0.025].
A simulated day draws U ~ U(0, 1); if U < alpha it is a hit with loss -q_t(U)
(interpolated in log u; below the smallest grid point a power-law tail is
extrapolated from the two smallest grid points). This needs nothing but the
stored grid, so it works identically for parametric, historical-simulation, FHS
and EVT models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def z2_statistic(ret: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    ret, var, es = (np.asarray(x, dtype=float) for x in (ret, var, es))
    hit = ret < -var
    return float(1.0 - np.sum((-ret) * hit / es) / (len(ret) * alpha))


def _interp_loss(
    u: np.ndarray, rows: np.ndarray, log_grid: np.ndarray, grid_q: np.ndarray
) -> np.ndarray:
    """Loss -q_t(u) for draws ``u`` on days ``rows`` (log-u linear interpolation + power-law tail)."""
    lu = np.log(u)
    k = len(log_grid)
    j = np.clip(np.searchsorted(log_grid, lu) - 1, 0, k - 2)
    w = (lu - log_grid[j]) / (log_grid[j + 1] - log_grid[j])
    q_lo, q_hi = grid_q[rows, j], grid_q[rows, j + 1]
    loss = -(q_lo + w * (q_hi - q_lo))
    below = lu < log_grid[0]
    if below.any():
        l0, l1 = -grid_q[rows[below], 0], -grid_q[rows[below], 1]
        # power law L(u) = L0 (u/u0)^(-s), slope from the two smallest grid points
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.log(np.maximum(l0, 1e-12) / np.maximum(l1, 1e-12)) / (log_grid[1] - log_grid[0])
        s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0)  # s >= 1 would imply an infinite mean
        loss[below] = l0 * np.exp(-s * (lu[below] - log_grid[0]))
    return loss


@dataclass
class Z2Result:
    z2: float
    p_value: float
    n: int


def z2_test(
    ret: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    grid_q: np.ndarray,
    grid_u: np.ndarray,
    alpha: float,
    sims: int,
    seed: int,
    chunk: int = 500,
) -> Z2Result:
    """Z2 statistic and one-sided Monte Carlo p-value P_H0(Z2* <= Z2_obs)."""
    ret, var, es = (np.asarray(x, dtype=float) for x in (ret, var, es))
    grid_q = np.asarray(grid_q, dtype=float)
    t = len(ret)
    obs = z2_statistic(ret, var, es, alpha)
    rng = np.random.default_rng(seed)
    log_grid = np.log(np.asarray(grid_u, dtype=float))
    inv_es = 1.0 / es
    sim = np.empty(sims)
    done = 0
    while done < sims:
        m = min(chunk, sims - done)
        u = rng.random((m, t))
        hit_m, hit_t = np.nonzero(u < alpha)
        loss = _interp_loss(u[hit_m, hit_t], hit_t, log_grid, grid_q)
        contrib = np.bincount(hit_m, weights=loss * inv_es[hit_t], minlength=m)
        sim[done : done + m] = 1.0 - contrib / (t * alpha)
        done += m
    p = float((np.sum(sim <= obs) + 1) / (sims + 1))
    return Z2Result(obs, p, t)
