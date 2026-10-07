"""``make fit`` and ``make forecast``: full-sample fits and next-day production forecasts.

``fit``
    * GARCH(1,1)-normal estimated with the from-scratch MLE and with ``arch`` on each
      asset's full history (a standing reconciliation check, reported in RESULTS.md);
    * full-sample parameter tables (with standard errors) for every ``arch`` model and
      the HAR regressions, persistence and half-lives;
    * then runs ``forecast``.

``forecast``
    Re-estimates each model on all data up to the last close and produces the forecast
    for the *next* trading day, written to ``results/latest_forecasts.json`` - the file
    the API serves and the scheduled GitHub Action refreshes.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from voltail.backtest.runner import _driver_for
from voltail.data.store import load_processed
from voltail.logging_utils import get_logger
from voltail.models.distributions import alpha_label
from voltail.models.garch_arch import SPECS, make_model
from voltail.models.garch_scratch import fit_garch11
from voltail.models.har import HARModel
from voltail.models.registry import MODEL_LABELS, build_families

log = get_logger(__name__)


def next_trading_day(last: pd.Timestamp, asset_class: str) -> pd.Timestamp:
    """Next calendar day for crypto, next weekday otherwise (exchange holidays not modelled)."""
    if asset_class == "crypto":
        return last + pd.Timedelta(days=1)
    return last + pd.offsets.BDay(1)


def full_sample_fits(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data = load_processed(cfg)
    recon, params, har_rows = [], [], []
    for asset, df in data.items():
        y = df["ret"].to_numpy()
        mine = fit_garch11(y)
        ref = arch_model(
            y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
        ).fit(disp="off")
        for k_m, k_r in [
            ("mu", "mu"),
            ("omega", "omega"),
            ("alpha", "alpha[1]"),
            ("beta", "beta[1]"),
        ]:
            recon.append(
                {
                    "asset": asset,
                    "param": k_m,
                    "scratch": getattr(mine, k_m),
                    "arch": float(ref.params[k_r]),
                    "abs_diff": abs(getattr(mine, k_m) - float(ref.params[k_r])),
                    "scratch_se": mine.std_errors.get(k_m, np.nan),
                }
            )
        recon.append(
            {
                "asset": asset,
                "param": "loglik",
                "scratch": mine.loglik,
                "arch": float(ref.loglikelihood),
                "abs_diff": abs(mine.loglik - float(ref.loglikelihood)),
                "scratch_se": np.nan,
            }
        )
        for name in SPECS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = make_model(df["ret"], name).fit(disp="off", show_warning=False)
            p = res.params
            if name.startswith("egarch"):
                persistence = float(p["beta[1]"])
            else:
                persistence = float(p["alpha[1]"] + 0.5 * p.get("gamma[1]", 0.0) + p["beta[1]"])
            for k, v in p.items():
                params.append(
                    {
                        "asset": asset,
                        "model": name,
                        "param": k,
                        "value": float(v),
                        "se": float(res.std_err[k]),
                        "pvalue": float(res.pvalues[k]),
                    }
                )
            params.append(
                {
                    "asset": asset,
                    "model": name,
                    "param": "persistence",
                    "value": persistence,
                    "se": np.nan,
                    "pvalue": np.nan,
                }
            )
            hl = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf
            params.append(
                {
                    "asset": asset,
                    "model": name,
                    "param": "half_life_days",
                    "value": float(hl),
                    "se": np.nan,
                    "pvalue": np.nan,
                }
            )
            params.append(
                {
                    "asset": asset,
                    "model": name,
                    "param": "loglik",
                    "value": float(res.loglikelihood),
                    "se": np.nan,
                    "pvalue": np.nan,
                }
            )
            params.append(
                {
                    "asset": asset,
                    "model": name,
                    "param": "bic",
                    "value": float(res.bic),
                    "se": np.nan,
                    "pvalue": np.nan,
                }
            )
        for h in (1, 5):
            m = HARModel.fit(
                df[cfg["proxies"]["model_target"]],
                h,
                tuple(cfg["models"]["har"]["lags"]),
                cfg["models"]["har"]["log"],
            )
            for k, v, se in zip(m.columns, m.result.params, m.result.bse_hac, strict=True):
                har_rows.append(
                    {
                        "asset": asset,
                        "horizon": h,
                        "param": k,
                        "value": float(v),
                        "se_hac": float(se),
                    }
                )
            har_rows.append(
                {
                    "asset": asset,
                    "horizon": h,
                    "param": "r2",
                    "value": m.result.r2,
                    "se_hac": np.nan,
                }
            )
        log.info(
            "%s: full-sample fits done (scratch vs arch max |diff| = %.2e)",
            asset,
            max(r["abs_diff"] for r in recon if r["asset"] == asset and r["param"] != "loglik"),
        )
    out = {
        "garch_reconciliation": pd.DataFrame(recon),
        "fullsample_params": pd.DataFrame(params),
        "har_params": pd.DataFrame(har_rows),
    }
    d = Path(cfg["paths"]["results_dir"]) / "fit"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(d / f"{k}.parquet")
    return out


def _frozen_xgb_params(cfg: dict[str, Any]) -> dict[str, Any]:
    p = Path(cfg["paths"]["results_dir"]) / "xgb_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


def latest_forecasts(cfg: dict[str, Any]) -> dict[str, Any]:
    """Next-day forecasts from every model, re-estimated on all available data."""
    data = load_processed(cfg)
    alphas = cfg["risk"]["alphas"]
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_name": cfg["run_name"],
        "data_source": cfg["data"]["source"],
        "default_model": cfg["service"]["default_model"],
        "units": "percent log returns; vol = daily standard deviation in %; VaR/ES = positive 1-day loss in %",
        "assets": {},
    }
    for asset, df in data.items():
        spec = cfg["data"]["assets"][asset]
        fams = build_families(cfg, asset, _driver_for(asset, data, cfg))
        frozen = _frozen_xgb_params(cfg).get(asset)
        for fam in fams:
            if fam.name == "xgb":
                if frozen:
                    fam.state.update(frozen)  # serve the validated hyperparameters, do not re-tune
                else:
                    log.warning(
                        "%s: no frozen XGBoost hyperparameters found (run `make backtest`); tuning on full history",
                        asset,
                    )
        n = len(df)
        cols: dict[str, float] = {}
        for fam in fams:
            out = fam.run_block(df, n - 1)  # estimate on everything, forecast the next day
            cols.update({k: float(v[-1]) for k, v in out.cols.items()})
        models: dict[str, Any] = {}
        all_models = sorted(
            {c.split("_", 1)[1] for c in cols if c.startswith("var1_")}
            | {c.split("_", 2)[2] for c in cols if c.startswith("VaR_")}
        )
        for m in all_models:
            entry: dict[str, Any] = {"label": MODEL_LABELS.get(m, m)}
            if f"var1_{m}" in cols:
                entry["vol_1d"] = float(np.sqrt(cols[f"var1_{m}"]))
                entry["vol_5d_avg"] = float(np.sqrt(cols[f"var5_{m}"]))
                entry["vol_annualized"] = float(
                    np.sqrt(cols[f"var1_{m}"] * (365 if spec["asset_class"] == "crypto" else 252))
                )
            for a in alphas:
                lab = alpha_label(a)
                if f"VaR_{lab}_{m}" in cols:
                    entry[f"var_{round((1 - a) * 1000)}"] = cols[f"VaR_{lab}_{m}"]
                    entry[f"es_{round((1 - a) * 1000)}"] = cols[f"ES_{lab}_{m}"]
            models[m] = entry
        payload["assets"][asset] = {
            "name": ("SYNTHETIC stand-in for " if cfg["data"]["source"] == "synthetic" else "")
            + spec["name"],
            "ticker": (
                spec["ticker"] if cfg["data"]["source"] != "synthetic" else f"SYNTHETIC:{asset}"
            ),
            "asset_class": spec["asset_class"],
            "last_close_date": str(df.index[-1].date()),
            "forecast_for": str(next_trading_day(df.index[-1], spec["asset_class"]).date()),
            "last_return_pct": float(df["ret"].iloc[-1]),
            "models": models,
        }
        log.info(
            "%s: next-day forecasts for %s from %d models",
            asset,
            payload["assets"][asset]["forecast_for"],
            len(models),
        )
    return payload


def run_forecast_stage(cfg: dict[str, Any]) -> None:
    payload = latest_forecasts(cfg)
    path = Path(cfg["service"]["forecasts_file"])
    if not path.is_absolute():
        from voltail.config import PROJECT_ROOT

        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", path)


def run_fit_stage(cfg: dict[str, Any]) -> None:
    full_sample_fits(cfg)
    run_forecast_stage(cfg)
