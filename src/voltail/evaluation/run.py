"""Evaluate every stored walk-forward forecast and write tidy result tables.

Outputs (``results_dir/eval``):

vol_metrics.parquet   QLIKE/MSE per asset x horizon x proxy x regime x model (+ DM vs benchmark)
mz.parquet            Mincer-Zarnowitz regressions (full sample)
dm_pairs.parquet      pairwise Diebold-Mariano on QLIKE (primary proxy, full sample)
mcs_vol.parquet       Model Confidence Set on QLIKE (full sample)
var_backtests.parquet Kupiec, Christoffersen, Basel zones, Z2, FZ0 per asset x alpha x regime x model
mcs_fz0.parquet       Model Confidence Set on FZ0 (full sample)
ratings.parquet       rule-based model ratings (see ratings.py)
summary.json          headline facts used by the report generator

All comparisons use a *common sample*: the dates on which every model in the
comparison has a forecast and the proxy is observed, so no model is judged on
an easier period than another.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from voltail.backtest.runner import load_forecasts, load_grids
from voltail.evaluation.es_tests import z2_test
from voltail.evaluation.forecast_tests import (
    diebold_mariano,
    dm_matrix,
    mincer_zarnowitz,
    model_confidence_set,
)
from voltail.evaluation.losses import fz0, mse, qlike, qlike_normalised
from voltail.evaluation.ratings import rate_models
from voltail.evaluation.regimes import regime_masks
from voltail.evaluation.var_tests import rolling_zones, traffic_light, var_backtest
from voltail.logging_utils import get_logger
from voltail.models.distributions import alpha_label, tail_grid

log = get_logger(__name__)

PROXIES = {"sq": "squared return", "gk": "Garman-Klass (scaled)"}


def _seed(cfg: dict[str, Any], *parts: Any) -> int:
    return (cfg["seed"] + zlib.crc32("|".join(map(str, parts)).encode())) % 2**32


def evaluate_vol(asset: str, fc: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    models = [m for m in cfg["models"]["vol"] if f"var1_{m}" in fc]
    bench = cfg["evaluation"]["benchmark_vol_model"]
    ecfg = cfg["evaluation"]
    metrics, mz_rows, dm_rows, mcs_rows = [], [], [], []
    for h in cfg["walkforward"]["horizons"]:
        fcols = [f"var{h}_{m}" for m in models]
        for proxy in PROXIES:
            pcol = f"rv{h}_{proxy}"
            sample = fc[fcols + [pcol]].replace([np.inf, -np.inf], np.nan).dropna()
            sample = sample[(sample[fcols] > 0).all(axis=1)]
            if len(sample) < 100:
                continue
            y = sample[pcol].to_numpy()
            ql = pd.DataFrame(
                {m: qlike(y, sample[f"var{h}_{m}"].to_numpy()) for m in models}, index=sample.index
            )
            se = pd.DataFrame(
                {m: mse(y, sample[f"var{h}_{m}"].to_numpy()) for m in models}, index=sample.index
            )
            masks = regime_masks(sample.index, ecfg["regimes"])
            for reg, mask in masks.items():
                if mask.sum() < 20:
                    continue
                qmean = ql[mask].mean()
                ranks = qmean.rank()
                for m in models:
                    row = {
                        "asset": asset,
                        "horizon": h,
                        "proxy": proxy,
                        "regime": reg,
                        "model": m,
                        "n": int(mask.sum()),
                        "start": sample.index[mask][0],
                        "end": sample.index[mask][-1],
                        "qlike": float(qmean[m]),
                        "mse": float(se.loc[mask, m].mean()),
                        "rank_qlike": float(ranks[m]),
                        "dqlike_vs_bench": (
                            float(qmean[m] - qmean[bench]) if bench in models else np.nan
                        ),
                        "bias_ratio": float(
                            sample.loc[mask, f"var{h}_{m}"].mean() / sample.loc[mask, pcol].mean()
                        ),
                    }
                    if proxy == "gk":
                        row["qlike_norm"] = float(
                            qlike_normalised(
                                y[mask], sample.loc[mask, f"var{h}_{m}"].to_numpy()
                            ).mean()
                        )
                    if bench in models and m != bench:
                        d = diebold_mariano(
                            ql.loc[mask, m].to_numpy(), ql.loc[mask, bench].to_numpy(), h
                        )
                        row.update(dm_stat_vs_bench=d.stat, dm_p_vs_bench=d.p_value)
                    metrics.append(row)
            # full-sample-only statistics
            for m in models:
                r = mincer_zarnowitz(
                    y, sample[f"var{h}_{m}"].to_numpy(), hac_lags=max(h - 1, 0) + 10
                )
                mz_rows.append(
                    {"asset": asset, "horizon": h, "proxy": proxy, "model": m, **r.__dict__}
                )
            if proxy == ecfg["primary_vol_proxy"]:
                stat, pval = dm_matrix(ql, h)
                for a in models:
                    for b in models:
                        if a != b:
                            dm_rows.append(
                                {
                                    "asset": asset,
                                    "horizon": h,
                                    "model_a": a,
                                    "model_b": b,
                                    "stat": stat.loc[a, b],
                                    "p": pval.loc[a, b],
                                }
                            )
            mc = ecfg["mcs"]
            mres = model_confidence_set(
                ql,
                mc["size"],
                mc["reps"],
                mc["block_size"] + h - 1,
                _seed(cfg, asset, h, proxy, "mcs"),
            )
            for m, rr in mres.iterrows():
                mcs_rows.append(
                    {"asset": asset, "horizon": h, "proxy": proxy, "model": m, **rr.to_dict()}
                )
    return {
        "vol_metrics": pd.DataFrame(metrics),
        "mz": pd.DataFrame(mz_rows),
        "dm_pairs": pd.DataFrame(dm_rows),
        "mcs_vol": pd.DataFrame(mcs_rows),
    }


def evaluate_var(
    asset: str, fc: pd.DataFrame, grids: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    ecfg = cfg["evaluation"]
    alphas = cfg["risk"]["alphas"]
    models = [m for m in cfg["models"]["var"] if f"VaR_{alpha_label(alphas[0])}_{m}" in fc]
    grid_u = tail_grid(cfg["models"]["tail_quantile_grid"], max(alphas))
    rows, mcs_rows = [], []
    for a in alphas:
        lab = alpha_label(a)
        cols = [f"VaR_{lab}_{m}" for m in models] + [f"ES_{lab}_{m}" for m in models]
        sample = fc[cols + ["ret"]].replace([np.inf, -np.inf], np.nan).dropna()
        es_ok = (sample[[f"ES_{lab}_{m}" for m in models]] > 0).all(axis=1)
        sample = sample[es_ok]
        r = sample["ret"].to_numpy()
        fz = pd.DataFrame(
            {
                m: fz0(
                    r, sample[f"VaR_{lab}_{m}"].to_numpy(), sample[f"ES_{lab}_{m}"].to_numpy(), a
                )
                for m in models
            },
            index=sample.index,
        )
        masks = regime_masks(sample.index, ecfg["regimes"])
        for reg, mask in masks.items():
            if mask.sum() < 20:
                continue
            fz_mean = fz[mask].mean()
            fz_rank = fz_mean.rank()
            idx = sample.index[mask]
            for m in models:
                var = sample.loc[mask, f"VaR_{lab}_{m}"].to_numpy()
                es = sample.loc[mask, f"ES_{lab}_{m}"].to_numpy()
                rr = r[mask]
                bt = var_backtest(rr, var, a)
                g = grids[m].reindex(idx).to_numpy()
                z2 = z2_test(
                    rr, var, es, g, grid_u, a, ecfg["z2_sims"], _seed(cfg, asset, a, reg, m, "z2")
                )
                hit = (rr < -var).astype(int)
                # exceedances in the first 20 trading days of each contiguous regime window:
                # models that adapt slowly (e.g. historical simulation) fail exactly here
                if reg == "Full":
                    first20 = np.nan
                else:
                    pos = np.flatnonzero(mask)
                    starts = np.concatenate([[0], np.flatnonzero(np.diff(pos) > 1) + 1])
                    first20 = int(sum(hit[st : st + 20].sum() for st in starts))
                row = {
                    "asset": asset,
                    "alpha": a,
                    "level": 1 - a,
                    "regime": reg,
                    "model": m,
                    "n": bt.n,
                    "exceed": bt.exceedances,
                    "expected": bt.expected,
                    "rate": bt.rate,
                    "lr_pof": bt.lr_pof,
                    "p_pof": bt.p_pof,
                    "lr_ind": bt.lr_ind,
                    "p_ind": bt.p_ind,
                    "lr_cc": bt.lr_cc,
                    "p_cc": bt.p_cc,
                    "zone": traffic_light(bt.exceedances, bt.n, a),
                    "z2": z2.z2,
                    "p_z2": z2.p_value,
                    "fz0": float(fz_mean[m]),
                    "rank_fz0": float(fz_rank[m]),
                    "avg_var": float(var.mean()),
                    "avg_es": float(es.mean()),
                    "first20_exceed": first20,
                    "start": idx[0],
                    "end": idx[-1],
                }
                if reg == "Full":
                    row.update(
                        {
                            f"roll250_{k}": v
                            for k, v in rolling_zones(hit, ecfg["basel_window"], a).items()
                        }
                    )
                rows.append(row)
        mc = ecfg["mcs"]
        mres = model_confidence_set(
            fz, mc["size"], mc["reps"], mc["block_size"], _seed(cfg, asset, a, "mcs_fz0")
        )
        for m, rr in mres.iterrows():
            mcs_rows.append({"asset": asset, "alpha": a, "model": m, **rr.to_dict()})
    return {"var_backtests": pd.DataFrame(rows), "mcs_fz0": pd.DataFrame(mcs_rows)}


def _evaluate_asset(asset: str, fc: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    grids = load_grids(cfg, asset)
    out = evaluate_vol(asset, fc, cfg)
    out.update(evaluate_var(asset, fc, grids, cfg))
    log.info("%s: evaluation done", asset)
    return out


def overall_rankings(tables: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    """Average QLIKE rank across assets (primary proxy, full sample) and MCS membership counts."""
    vm = tables["vol_metrics"]
    primary = cfg["evaluation"]["primary_vol_proxy"]
    full = vm[(vm["regime"] == "Full")]
    rows = []
    for (h, proxy), g in full.groupby(["horizon", "proxy"]):
        avg_rank = g.groupby("model")["rank_qlike"].mean()
        mcs = tables["mcs_vol"]
        mcs = mcs[(mcs["horizon"] == h) & (mcs["proxy"] == proxy)]
        in_mcs = mcs.groupby("model")["in_mcs"].sum()
        n_assets = mcs.groupby("model")["in_mcs"].size()
        for m in avg_rank.index:
            rows.append(
                {
                    "horizon": h,
                    "proxy": proxy,
                    "model": m,
                    "avg_rank_qlike": float(avg_rank[m]),
                    "in_mcs": int(in_mcs.get(m, 0)),
                    "n_assets": int(n_assets.get(m, 0)),
                    "primary": proxy == primary,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "proxy", "avg_rank_qlike"])


def build_summary(
    tables: dict[str, pd.DataFrame], cfg: dict[str, Any], forecasts: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """Headline facts for the README / report / resume bullets (all computed, none typed)."""
    vm, vb, rk = tables["vol_metrics"], tables["var_backtests"], tables["rank_overall"]
    primary = cfg["evaluation"]["primary_vol_proxy"]
    s: dict[str, Any] = {
        "run_name": cfg["run_name"],
        "source": cfg["data"]["source"],
        "assets": list(forecasts),
    }
    s["oos_start"] = {a: str(f.index[0].date()) for a, f in forecasts.items()}
    s["oos_end"] = {a: str(f.index[-1].date()) for a, f in forecasts.items()}
    s["n_oos_days_total"] = int(sum(len(f) for f in forecasts.values()))
    n_models = len(set(cfg["models"]["vol"]) | set(cfg["models"]["var"]))
    s["n_models"] = n_models
    for h in cfg["walkforward"]["horizons"]:
        r = rk[(rk["horizon"] == h) & (rk["proxy"] == primary)].sort_values("avg_rank_qlike")
        s[f"best_vol_h{h}"] = r.iloc[0]["model"]
        s[f"best_vol_h{h}_avg_rank"] = float(r.iloc[0]["avg_rank_qlike"])
        s[f"vol_rank_h{h}"] = r[["model", "avg_rank_qlike", "in_mcs", "n_assets"]].to_dict(
            "records"
        )
        full = vm[(vm["horizon"] == h) & (vm["proxy"] == "gk") & (vm["regime"] == "Full")]
        if "qlike_norm" in full and len(full):
            piv = full.pivot(index="asset", columns="model", values="qlike_norm")
            bench = cfg["evaluation"]["benchmark_vol_model"]
            rel = (piv.div(piv[bench], axis=0) - 1.0) * 100
            s[f"qlike_gk_pct_vs_{bench}_h{h}"] = rel.median().round(2).to_dict()
    full_var = vb[vb["regime"] == "Full"]
    s["var_cells"] = int(len(full_var))
    s["var_pass_kupiec_share"] = float((full_var["p_pof"] >= 0.05).mean())
    s["var_pass_cc_share"] = float((full_var["p_cc"] >= 0.05).mean())
    by_model = full_var.groupby("model").agg(
        kupiec_pass=("p_pof", lambda p: int((p >= 0.05).sum())),
        cc_pass=("p_cc", lambda p: int((p >= 0.05).sum())),
        z2_pass=("p_z2", lambda p: int((p >= 0.05).sum())),
        red=("zone", lambda z: int((z == "red").sum())),
        cells=("zone", "size"),
        avg_rank_fz0=("rank_fz0", "mean"),
    )
    s["var_by_model"] = by_model.reset_index().to_dict("records")
    s["best_fz0_model"] = by_model["avg_rank_fz0"].idxmin()
    s["ratings"] = tables["ratings"].to_dict("records")
    s["n_tests"] = int(len(vb) * 4 + len(vm))
    return s


def run_evaluation(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    forecasts = load_forecasts(cfg)
    parts = Parallel(n_jobs=min(cfg["n_jobs"], len(forecasts)), backend="loky")(
        delayed(_evaluate_asset)(a, f, cfg) for a, f in forecasts.items()
    )
    tables: dict[str, pd.DataFrame] = {}
    for key in parts[0]:
        tables[key] = pd.concat([p[key] for p in parts if len(p[key])], ignore_index=True)
    tables["rank_overall"] = overall_rankings(tables, cfg)
    full = tables["var_backtests"]
    tables["ratings"] = rate_models(full[full["regime"] == "Full"])
    out = Path(cfg["paths"]["results_dir"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_parquet(out / f"{k}.parquet")
    summary = build_summary(tables, cfg, forecasts)
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("evaluation tables written to %s", out)
    return tables


def load_eval(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    d = Path(cfg["paths"]["results_dir"]) / "eval"
    tables = {p.stem: pd.read_parquet(p) for p in d.glob("*.parquet")}
    summary = json.loads((d / "summary.json").read_text())
    return tables, summary
