"""GARCH-family models estimated with the ``arch`` package, plus FHS and conditional EVT.

Models (all with a constant mean, returns in percent):

=========  =============================================================  ============
name       variance equation                                              innovations
=========  =============================================================  ============
garch_*    s2_t = w + a e_{t-1}^2 + b s2_{t-1}                            normal / t
gjr_*      s2_t = w + (a + g 1[e_{t-1} < 0]) e_{t-1}^2 + b s2_{t-1}       normal / t
gjr_skewt  same as gjr                                                    Hansen skewed t
egarch_t   ln s2_t = w + a(|z_{t-1}| - E|z|) + g z_{t-1} + b ln s2_{t-1}  t
=========  =============================================================  ============

GJR and EGARCH capture the *leverage effect*: negative returns raise future
volatility more than positive returns of the same size.

Walk-forward mechanics per block
--------------------------------
1. estimate parameters on returns up to ``train_end`` (warm-started from the previous
   refit's estimates; if the optimiser does not converge we keep the previous
   parameters and log it, rather than silently using a bad fit);
2. with parameters *fixed*, run the variance recursion over the data up to each
   origin and produce 1..5-step-ahead forecasts (``arch``'s recursion is causal);
3. EGARCH has no closed-form multi-step forecast, so its 5-day forecast uses
   ``arch``'s simulation method with a seeded RNG.

FHS and conditional EVT reuse the GJR-GARCH-t fit (``fhs_gjr``, ``evt_gjr``).
"""

from __future__ import annotations

import warnings
import zlib

import numpy as np
import pandas as pd
from arch import arch_model

from voltail.models.base import BlockOutput, ModelFamily
from voltail.models.distributions import (
    normal_es,
    normal_quantile,
    skewt_es,
    skewt_quantile,
    std_t_es,
    std_t_quantile,
)
from voltail.models.tail import evt_risk, fhs_risk

SPECS: dict[str, dict] = {
    "garch_n": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    "garch_t": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    "gjr_n": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "normal"},
    "gjr_t": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "gjr_skewt": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "skewt"},
    "egarch_t": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
}


# GARCH filters for the semi-parametric tail models: filter -> (FHS model, EVT model)
TAIL_FILTERS: dict[str, tuple[str, str]] = {
    "gjr_t": ("fhs_gjr", "evt_gjr"),
    "gjr_skewt": ("fhs_skewt", "evt_skewt"),
}


def make_model(y: pd.Series | np.ndarray, name: str):
    """Construct an (unfitted) ``arch`` model for one of :data:`SPECS`."""
    return arch_model(y, mean="Constant", rescale=False, **SPECS[name])


def fit_arch(y: pd.Series | np.ndarray, name: str, start: np.ndarray | None = None):
    """Fit; returns (result, converged). Warnings from the optimiser are captured, not printed."""
    am = make_model(y, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(
            disp="off", starting_values=start, show_warning=False, options={"maxiter": 500}
        )
        if res.convergence_flag != 0 and start is not None:
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 1000})
    return res, res.convergence_flag == 0


def _shock_sampler(dist: str, params: np.ndarray, seed: int):
    """Seeded sampler of unit-variance innovations with the signature arch expects."""
    gen = np.random.default_rng(seed)
    if dist == "t":
        nu = float(params[-1])
        scale = np.sqrt((nu - 2.0) / nu)
        return lambda size: gen.standard_t(nu, size) * scale
    return lambda size: gen.standard_normal(size)


class ArchFamily(ModelFamily):
    """All six arch GARCH variants + FHS/EVT on GJR-t, refit on the GARCH schedule."""

    name = "garch"
    refit_key = "garch"

    def __init__(self, cfg, asset, context=None, models: list[str] | None = None):
        super().__init__(cfg, asset, context)
        self.models = models or list(SPECS)

    def run_block(self, hist: pd.DataFrame, train_end: int) -> BlockOutput:
        y_all = hist["ret"]
        y_train = y_all.iloc[: train_end + 1]
        refit_date = hist.index[train_end]
        out = BlockOutput()
        for name in self.models:
            prev = self.state.get(name)
            res, ok = fit_arch(y_train, name, start=prev)
            if ok:
                params = res.params.to_numpy()
                self.state[name] = params
            elif prev is not None:
                params = prev
                out.notes.append(
                    f"{refit_date.date()}: {name} did not converge; kept previous parameters"
                )
            else:
                params = res.params.to_numpy()
                out.notes.append(
                    f"{refit_date.date()}: {name} did not converge on first fit; using last iterate"
                )
            pnames = list(res.params.index)
            for k, v in zip(pnames, params, strict=True):
                out.params.append(
                    {"refit_date": refit_date, "model": name, "param": k, "value": float(v)}
                )

            fixed = make_model(y_all, name).fix(params)
            kwargs: dict = {"start": train_end, "horizon": 5, "reindex": False}
            if SPECS[name]["vol"] == "EGARCH":
                seed = (
                    self.cfg["seed"]
                    + zlib.crc32(f"{self.asset}|{refit_date.date()}|{name}".encode())
                ) % 2**32
                # NOTE: arch's ``random_state`` argument does not seed the innovation draws for
                # method="simulation" (it uses the distribution's own unseeded generator), which
                # made reruns non-deterministic. Our no-lookahead test caught this; we pass an
                # explicitly seeded ``rng`` instead.
                kwargs.update(
                    method="simulation",
                    simulations=self.cfg["models"]["egarch_sim_draws"],
                    rng=_shock_sampler(SPECS[name]["dist"], params, seed),
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = fixed.forecast(**kwargs)
            var = fc.variance.to_numpy().copy()
            if SPECS[name]["vol"] == "EGARCH":
                # one-step variance is known exactly; take it from the analytic recursion
                var[:, 0] = fixed.forecast(
                    start=train_end, horizon=1, reindex=False
                ).variance.to_numpy()[:, 0]
            out.cols[f"var1_{name}"] = var[:, 0]
            out.cols[f"var5_{name}"] = var.mean(axis=1)

            p = dict(zip(pnames, params, strict=True))
            mu = np.full(var.shape[0], p["mu"])
            sigma = np.sqrt(var[:, 0])
            if SPECS[name]["dist"] == "t":
                nu = p["nu"]
                out.add_location_scale_risk(
                    name,
                    mu,
                    sigma,
                    self.alphas,
                    lambda u, nu=nu: std_t_quantile(u, nu),
                    lambda a, nu=nu: std_t_es(a, nu),
                    self.grid_u,
                )
            elif SPECS[name]["dist"] == "skewt":
                eta, lam = p["eta"], p["lambda"]
                es_cache = {a: skewt_es(a, eta, lam) for a in self.alphas}  # once per refit
                out.add_location_scale_risk(
                    name,
                    mu,
                    sigma,
                    self.alphas,
                    lambda u, eta=eta, lam=lam: skewt_quantile(u, eta, lam),
                    lambda a, c=es_cache: c[a],
                    self.grid_u,
                )
            else:
                out.add_location_scale_risk(
                    name, mu, sigma, self.alphas, normal_quantile, normal_es, self.grid_u
                )

            if name in TAIL_FILTERS:
                fhs_name, evt_name = TAIL_FILTERS[name]
                z = make_model(y_train, name).fix(params).std_resid
                z = np.asarray(z)[np.isfinite(z)]
                fhs_risk(out, fhs_name, z, p["mu"], sigma, self.alphas, self.grid_u)
                g = evt_risk(
                    out,
                    evt_name,
                    z,
                    p["mu"],
                    sigma,
                    self.alphas,
                    self.grid_u,
                    self.cfg["models"]["evt"]["threshold_quantile"],
                )
                for k, v in {"xi": g.xi, "beta": g.beta, "threshold": g.threshold}.items():
                    out.params.append(
                        {"refit_date": refit_date, "model": evt_name, "param": k, "value": v}
                    )
                if g.clipped:
                    out.notes.append(f"{refit_date.date()}: {evt_name} xi clipped at 0.95")
        return out
