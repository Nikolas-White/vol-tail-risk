"""Gradient-boosted trees (XGBoost) forecasting next-day and next-week log volatility.

Target   log mean Garman-Klass proxy over t+1..t+h (h = 1, 5)
Features lagged own volatility/return/range measures + strictly-lagged S&P 500
         features (see :mod:`voltail.features.ml_features`)

Honesty safeguards
------------------
* Hyperparameters are tuned **once per asset**, on the burn-in period that precedes
  the first out-of-sample forecast, with expanding-window time-series CV and a purge
  gap; they are then frozen for the whole backtest.
* Training rows are those whose *target window* ends on or before the refit date,
  so a 5-day target never peeks past the training cut-off.
* Tree models fit their training data closely, so in-sample residuals understate
  forecast error. The log-retransformation (smearing) factor and the standardized
  returns used for VaR are therefore computed from **out-of-fold** predictions
  (blocked K-fold inside the training window, with purge gaps).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from voltail.features.ml_features import EPS, build_features, ml_target
from voltail.features.proxies import expanding_scale
from voltail.models.base import BlockOutput, ModelFamily
from voltail.models.tail import fhs_risk


def qlike_level(proxy: np.ndarray, forecast: np.ndarray) -> float:
    """Mean QLIKE loss (see :mod:`voltail.evaluation.losses`), duplicated to avoid an import cycle."""
    ratio = proxy / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _make_model(params: dict, seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        verbosity=0,
        **params,
    )


def tune_xgb(
    x: pd.DataFrame, y: pd.Series, grid: dict, folds: int, purge: int, seed: int
) -> tuple[dict, list[dict]]:
    """Expanding-window CV over ``grid``; returns (best params, all scores)."""
    keys = sorted(grid)
    combos = [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))
    ]
    n = len(x)
    edges = np.linspace(0, n, folds + 2).astype(int)
    scores = []
    for params in combos:
        losses = []
        for k in range(1, folds + 1):
            tr_end, va_start, va_end = edges[k] - purge, edges[k], edges[k + 1]
            if tr_end < 100:
                continue
            m = _make_model(params, seed).fit(x.iloc[:tr_end], y.iloc[:tr_end])
            pred = np.exp(m.predict(x.iloc[va_start:va_end]))
            losses.append(qlike_level(np.exp(y.iloc[va_start:va_end].to_numpy()), pred))
        scores.append({**params, "cv_qlike": float(np.mean(losses)) if losses else np.inf})
    best = min(scores, key=lambda s: s["cv_qlike"])
    return {k: best[k] for k in keys}, scores


def oof_predictions(
    x: pd.DataFrame, y: pd.Series, params: dict, folds: int, purge: int, seed: int
) -> pd.Series:
    """Blocked K-fold out-of-fold predictions (log scale) with purge gaps around each test block."""
    n = len(x)
    edges = np.linspace(0, n, folds + 1).astype(int)
    oof = pd.Series(np.nan, index=x.index)
    for k in range(folds):
        a, b = edges[k], edges[k + 1]
        train_mask = np.ones(n, dtype=bool)
        train_mask[max(0, a - purge) : min(n, b + purge)] = False
        m = _make_model(params, seed).fit(x.iloc[train_mask], y.iloc[train_mask])
        oof.iloc[a:b] = m.predict(x.iloc[a:b])
    return oof


class XGBFamily(ModelFamily):
    """XGBoost volatility model (``xgb``), refit on the ML schedule."""

    name = "xgb"
    refit_key = "ml"

    def run_block(self, hist: pd.DataFrame, train_end: int) -> BlockOutput:
        mcfg = self.cfg["models"]["xgb"]
        seed = self.cfg["seed"]
        target_col = self.cfg["proxies"]["model_target"]
        driver = self.context.get("driver")
        if driver is not None:
            driver = driver.loc[: hist.index[-1]]  # never pass driver data beyond the block
        feats = build_features(hist, driver, target_col, self.cfg["models"]["ewma_lambda"])
        v = hist[target_col]
        v_train = v.iloc[: train_end + 1]
        scale = expanding_scale(
            hist["sq_return"], v, self.cfg["proxies"]["scale_min_obs"]
        ).to_numpy()
        refit_date = hist.index[train_end]
        purge = 22
        out = BlockOutput()
        oof1, smear1 = None, 1.0
        for h in (1, 5):
            y = ml_target(v_train, h)  # NaN where the target window passes train_end
            data = (
                feats.iloc[: train_end + 1]
                .join(y.rename("y"))
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            x_tr, y_tr = data.drop(columns="y"), data["y"]
            key = f"params_h{h}"
            if key not in self.state:  # tune once, on the pre-test window only
                best, scores = tune_xgb(
                    x_tr, y_tr, mcfg["tune_grid"], mcfg["cv_folds"], purge, seed
                )
                self.state[key] = best
                self.state[f"tuning_h{h}"] = scores
                for k, val in best.items():
                    out.params.append(
                        {
                            "refit_date": refit_date,
                            "model": f"xgb_h{h}_tuned",
                            "param": k,
                            "value": float(val),
                        }
                    )
            params = self.state[key]
            oof = oof_predictions(x_tr, y_tr, params, max(mcfg["cv_folds"], 3), purge, seed)
            resid = (y_tr - oof).dropna()
            # smearing in levels: E[v] = E[exp(log v)]; log target includes EPS -> subtract it back
            smear = float(np.mean(np.exp(resid)))
            model = _make_model(params, seed).fit(x_tr, y_tr)
            x_fc = feats.iloc[train_end:]
            pred = np.exp(model.predict(x_fc)) * smear - EPS
            pred = np.maximum(pred, 1e-6)
            out.cols[f"var{h}_xgb"] = pred * scale[train_end:]
            out.params.append(
                {"refit_date": refit_date, "model": f"xgb_h{h}", "param": "smear", "value": smear}
            )
            out.params.append(
                {
                    "refit_date": refit_date,
                    "model": f"xgb_h{h}",
                    "param": "n_train",
                    "value": float(len(x_tr)),
                }
            )
            if h == 1:
                oof1, smear1 = oof, smear
                gain = model.get_booster().get_score(importance_type="gain")
                tot = sum(gain.values()) or 1.0
                for k, val in gain.items():
                    out.params.append(
                        {
                            "refit_date": refit_date,
                            "model": "xgb_importance",
                            "param": k,
                            "value": val / tot,
                        }
                    )
        # VaR multiplier from out-of-fold standardized returns
        assert oof1 is not None
        s = pd.Series(scale, index=hist.index)
        fitted = (np.exp(oof1.dropna()) * smear1 - EPS).clip(lower=1e-6) * s.loc[
            oof1.dropna().index
        ]
        r_next = hist["ret"].shift(-1).loc[fitted.index]
        ok = fitted.notna() & r_next.notna() & (fitted.index < hist.index[train_end])
        z = (r_next[ok] / np.sqrt(fitted[ok])).to_numpy()
        fhs_risk(out, "xgb", z, 0.0, np.sqrt(out.cols["var1_xgb"]), self.alphas, self.grid_u)
        return out
