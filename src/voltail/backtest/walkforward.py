"""Walk-forward (out-of-sample) forecasting engine.

For each model family and asset:

1. The first ``min_train_obs`` rows are burn-in: used for estimation only.
2. Refit points R are the last trading day of each refit period (month/quarter/
   year, per family) plus the end of the burn-in. Parameters estimated at R use
   rows <= R only (expanding window by default, or the last ``rolling_obs`` rows).
3. Between refits, parameters are held fixed and the model is *filtered* forward
   through new data: the forecast made at the close of origin o (for day o+1)
   uses rows <= o.
4. The forecast for origin o is stored against the date of row o+1, together with
   the date of the last row used for estimation (``train_end_<family>``), which the
   no-lookahead test checks is strictly before the forecast date.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from voltail.logging_utils import get_logger
from voltail.models.base import ModelFamily

log = get_logger(__name__)


def refit_points(index: pd.DatetimeIndex, start_pos: int, freq: str | None) -> list[int]:
    """Positions of refit origins: ``start_pos - 1`` and every period end at or after ``start_pos``.

    ``freq`` is a pandas period alias ('M', 'Q', 'Y'); ``None`` means fit once.
    """
    first = start_pos - 1
    if freq is None:
        return [first]
    periods = index.to_period(freq)
    is_last = np.append(periods[1:] != periods[:-1], True)
    ends = np.flatnonzero(is_last)
    return sorted({first, *[int(p) for p in ends if p >= start_pos]})


def run_family(
    df: pd.DataFrame, family: ModelFamily, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict], list[str]]:
    """Walk one family forward through ``df``.

    Returns
    -------
    forecasts : DataFrame indexed by *target* date (the day being forecast)
    grids     : model -> DataFrame of tail return-quantiles (same index)
    params    : list of parameter records per refit
    notes     : convergence / clipping notes
    """
    wf = cfg["walkforward"]
    n = len(df)
    start_pos = int(wf["min_train_obs"])
    if n <= start_pos + 1:
        raise ValueError(f"{family.asset}: only {n} rows, need > {start_pos + 1}")
    freq = wf["refit_every"].get(family.refit_key) if family.refit_key else None
    points = refit_points(df.index, start_pos, freq)
    rolling = wf["window"] == "rolling" and family.refit_key is not None
    frames, grid_frames = [], {}
    params: list[dict] = []
    notes: list[str] = []
    for k, r in enumerate(points):
        last_origin = points[k + 1] - 1 if k + 1 < len(points) else n - 2
        if last_origin < r:
            continue
        w0 = max(0, r - int(wf["rolling_obs"]) + 1) if rolling else 0
        hist = df.iloc[w0 : last_origin + 1]  # nothing after the block's last origin is visible
        out = family.run_block(hist, r - w0)
        n_o = last_origin - r + 1
        target_dates = df.index[r + 1 : last_origin + 2]
        for c, arr in out.cols.items():
            if len(arr) != n_o:
                raise RuntimeError(f"{family.name}: column {c} has {len(arr)} rows, expected {n_o}")
        block = pd.DataFrame(out.cols, index=target_dates)
        block[f"train_end_{family.name}"] = df.index[r]
        frames.append(block)
        for m, g in out.grids.items():
            grid_frames.setdefault(m, []).append(pd.DataFrame(g, index=target_dates))
        params.extend(out.params)
        notes.extend(f"{family.asset}: {x}" for x in out.notes)
    fc = pd.concat(frames)
    grids = {m: pd.concat(gs) for m, gs in grid_frames.items()}
    return fc, grids, params, notes


def realized_targets(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Realized quantities aligned at the *target* date t (what the forecasts are scored against).

    * ``ret``         return on day t
    * ``rv1_sq``      r_t^2                                  (primary 1-day proxy)
    * ``rv1_gk``      GK_t x scale_{t-1}                      (range proxy on close-to-close scale)
    * ``rv5_sq`` / ``rv5_gk``  mean of the same over t..t+4   (next-week proxies)

    The scale factor uses data up to t-1 only, so even the *evaluation* proxy
    involves no future information beyond the day it describes.
    """
    from voltail.features.proxies import expanding_scale

    scale = expanding_scale(
        df["sq_return"], df["garman_klass"], cfg["proxies"]["scale_min_obs"]
    ).shift(1)
    out = pd.DataFrame(index=df.index)
    out["ret"] = df["ret"]
    out["rv1_sq"] = df["sq_return"]
    out["rv1_gk"] = df["garman_klass"] * scale
    out["rv1_pk"] = df["parkinson"] * scale
    for col in ["sq", "gk"]:
        s = out[f"rv1_{col}"]
        out[f"rv5_{col}"] = pd.concat([s.shift(-k) for k in range(5)], axis=1).mean(
            axis=1, skipna=False
        )
    out["gk_raw"] = df["garman_klass"]
    out["pk_raw"] = df["parkinson"]
    out["scale"] = scale
    return out


def run_asset(
    asset: str, df: pd.DataFrame, cfg: dict[str, Any], families: list[ModelFamily]
) -> dict[str, Any]:
    """Run every family for one asset and assemble the outputs."""
    t0 = time.time()
    fcs, grids, params, notes, timing = [], {}, [], [], {}
    for fam in families:
        t1 = time.time()
        fc, g, p, nt = run_family(df, fam, cfg)
        fcs.append(fc)
        grids.update(g)
        params.extend({"asset": asset, **x} for x in p)
        notes.extend(nt)
        timing[fam.name] = round(time.time() - t1, 1)
        log.info("%s: %-10s %5d forecasts in %6.1fs", asset, fam.name, len(fc), timing[fam.name])
    forecasts = pd.concat(fcs, axis=1)
    forecasts = forecasts.join(realized_targets(df, cfg), how="left")
    log.info("%s: done in %.1fs", asset, time.time() - t0)
    extra = {f.name: f.state.get("tuning_h1") for f in families if "tuning_h1" in f.state}
    frozen = {
        f.name: {k: v for k, v in f.state.items() if k.startswith("params_h")}
        for f in families
        if any(k.startswith("params_h") for k in f.state)
    }
    return {
        "asset": asset,
        "forecasts": forecasts,
        "grids": grids,
        "params": params,
        "notes": notes,
        "timing": timing,
        "tuning": extra,
        "frozen_hyperparams": frozen,
    }
