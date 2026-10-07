"""Heterogeneous Autoregressive (HAR) volatility model, Corsi (2009), on a range-based proxy.

Idea: traders with daily, weekly and monthly horizons all react to volatility,
so tomorrow's variance depends on yesterday's, last week's average and last
month's average::

    log v_{t+1:t+h} = b0 + b_d log v_t + b_w log mean(v_{t-4..t}) + b_m log mean(v_{t-21..t}) + u

where ``v`` is the Garman-Klass variance proxy and ``v_{t+1:t+h}`` is its mean over
the next ``h`` days (h = 1 for next-day, h = 5 for next-week). It is estimated with
the from-scratch OLS in :mod:`voltail.models.ols`.

Working in logs keeps forecasts positive and makes residuals closer to Gaussian.
Because E[exp(u)] != 1, the back-transformed forecast is multiplied by the
Duan (1983) smearing factor mean(exp(residuals)) estimated in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from voltail.features.proxies import expanding_scale
from voltail.models.base import BlockOutput, ModelFamily
from voltail.models.ols import OLSResult, ols
from voltail.models.tail import fhs_risk


def har_features(
    v: pd.Series, lags: tuple[int, int, int] = (1, 5, 22), log: bool = True
) -> pd.DataFrame:
    """HAR regressors observable at the close of each day t (no future data).

    Missing proxy values (days with unusable OHLC) are forward-filled for the
    regressors only - this uses the most recent *past* value.
    """
    v = v.ffill()
    d, w, m = lags
    feats = pd.DataFrame(
        {
            "har_d": v.rolling(d).mean(),
            "har_w": v.rolling(w).mean(),
            "har_m": v.rolling(m).mean(),
        },
        index=v.index,
    )
    return np.log(feats) if log else feats


def har_target(v: pd.Series, horizon: int, log: bool = True) -> pd.Series:
    """Mean of ``v`` over days t+1..t+h, aligned at t. NaN if any of those days is missing."""
    fwd = pd.concat([v.shift(-k) for k in range(1, horizon + 1)], axis=1)
    tgt = fwd.mean(axis=1, skipna=False)
    return np.log(tgt) if log else tgt


@dataclass
class HARModel:
    """Fitted HAR model for one horizon."""

    result: OLSResult
    smear: float
    log: bool
    horizon: int
    columns: tuple[str, ...] = ("const", "har_d", "har_w", "har_m")

    @classmethod
    def fit(cls, v: pd.Series, horizon: int, lags=(1, 5, 22), log: bool = True) -> HARModel:
        """Estimate on proxy history ``v`` (all of which must be known at the fit date)."""
        x = har_features(v, lags, log)
        y = har_target(v, horizon, log)
        data = pd.concat([x, y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        xmat = np.column_stack([np.ones(len(data)), data[["har_d", "har_w", "har_m"]].to_numpy()])
        res = ols(data["y"].to_numpy(), xmat, hac_lags=max(horizon - 1, 0) + 5)
        smear = float(np.mean(np.exp(res.resid))) if log else 1.0
        return cls(res, smear, log, horizon)

    def predict_from_features(self, feats: pd.DataFrame) -> pd.Series:
        """Average daily variance forecast over the horizon, in proxy units."""
        xmat = np.column_stack([np.ones(len(feats)), feats[["har_d", "har_w", "har_m"]].to_numpy()])
        yhat = xmat @ self.result.params
        out = np.exp(yhat) * self.smear if self.log else np.maximum(yhat, 1e-8)
        return pd.Series(out, index=feats.index)

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.columns, self.result.params.tolist(), strict=True))


# ----------------------------------------------------------------------------- walk-forward family


class HARFamily(ModelFamily):
    """HAR on Garman-Klass, rescaled to close-to-close variance with a past-only ratio.

    VaR/ES use the empirical distribution of in-sample standardized returns
    r_{t+1} / sqrt(fitted variance_t) (FHS-style, zero mean).
    """

    name = "har"
    refit_key = "har"

    def run_block(self, hist: pd.DataFrame, train_end: int) -> BlockOutput:
        hcfg = self.cfg["models"]["har"]
        lags = tuple(hcfg["lags"])
        log = bool(hcfg["log"])
        target_col = self.cfg["proxies"]["model_target"]
        v = hist[target_col]
        scale = expanding_scale(
            hist["sq_return"], v, self.cfg["proxies"]["scale_min_obs"]
        ).to_numpy()
        v_train = v.iloc[: train_end + 1]
        feats = har_features(v, lags, log)  # causal: row t uses v up to t
        out = BlockOutput()
        refit_date = hist.index[train_end]
        fitted: dict[int, HARModel] = {}
        for h in (1, 5):
            m = HARModel.fit(v_train, h, lags, log)
            fitted[h] = m
            pred = m.predict_from_features(feats.iloc[train_end:]).to_numpy()
            out.cols[f"var{h}_har"] = pred * scale[train_end:]
            for k, val in m.params.items():
                out.params.append(
                    {"refit_date": refit_date, "model": f"har_h{h}", "param": k, "value": val}
                )
        # in-sample standardized returns for the VaR multiplier
        f_tr = feats.iloc[: train_end + 1].dropna()
        insample = (
            fitted[1].predict_from_features(f_tr)
            * pd.Series(scale, index=hist.index).loc[f_tr.index]
        )
        r_next = hist["ret"].shift(-1).loc[f_tr.index]
        ok = insample.notna() & r_next.notna() & (insample > 0)
        ok &= f_tr.index < hist.index[train_end]  # r_{t+1} must be inside the training window
        z = (r_next[ok] / np.sqrt(insample[ok])).to_numpy()
        sigma = np.sqrt(out.cols["var1_har"])
        fhs_risk(out, "har", z, 0.0, sigma, self.alphas, self.grid_u)
        return out
