"""``make backtest``: walk-forward forecasts for every asset, then statistical evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from voltail.backtest.walkforward import run_asset
from voltail.data.store import load_processed
from voltail.logging_utils import get_logger
from voltail.models.registry import build_families

log = get_logger(__name__)


def _driver_for(
    asset: str, data: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> pd.DataFrame | None:
    drv = cfg["models"]["xgb"].get("cross_asset_driver")
    if drv and drv != asset and drv in data:
        return data[drv]
    return None


def _run_one(asset: str, data: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    from voltail.config import set_seeds

    set_seeds(cfg["seed"])
    fams = build_families(cfg, asset, _driver_for(asset, data, cfg))
    return run_asset(asset, data[asset], cfg, fams)


def save_asset_outputs(res: dict[str, Any], results_dir: Path) -> None:
    asset = res["asset"]
    (results_dir / "forecasts").mkdir(parents=True, exist_ok=True)
    (results_dir / "grids").mkdir(parents=True, exist_ok=True)
    res["forecasts"].to_parquet(results_dir / "forecasts" / f"{asset}.parquet")
    grid = pd.concat({m: g for m, g in res["grids"].items()}, axis=1)
    grid.columns = [f"{m}|{k}" for m, k in grid.columns]
    grid.to_parquet(results_dir / "grids" / f"{asset}.parquet")


def load_forecasts(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    d = Path(cfg["paths"]["results_dir"]) / "forecasts"
    return {
        a: pd.read_parquet(d / f"{a}.parquet")
        for a in cfg["data"]["assets"]
        if (d / f"{a}.parquet").exists()
    }


def load_grids(cfg: dict[str, Any], asset: str) -> dict[str, pd.DataFrame]:
    g = pd.read_parquet(Path(cfg["paths"]["results_dir"]) / "grids" / f"{asset}.parquet")
    out: dict[str, list[str]] = {}
    for c in g.columns:
        out.setdefault(c.split("|")[0], []).append(c)
    return {m: g[cols].set_axis(range(len(cols)), axis=1) for m, cols in out.items()}


def run_backtest_stage(cfg: dict[str, Any]) -> None:
    """Run all assets (in parallel), save forecasts/grids/params, then evaluate."""
    data = load_processed(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    assets = list(data)
    results = Parallel(n_jobs=min(cfg["n_jobs"], len(assets)), backend="loky")(
        delayed(_run_one)(a, data, cfg) for a in assets
    )
    params, diag = [], {"notes": [], "timing": {}, "xgb_tuning": {}, "run_name": cfg["run_name"]}
    frozen: dict[str, Any] = {}
    for res in results:
        save_asset_outputs(res, results_dir)
        params.extend(res["params"])
        diag["notes"].extend(res["notes"])
        diag["timing"][res["asset"]] = res["timing"]
        diag["xgb_tuning"][res["asset"]] = res["tuning"].get("xgb")
        if "xgb" in res["frozen_hyperparams"]:
            frozen[res["asset"]] = res["frozen_hyperparams"]["xgb"]
    pd.DataFrame(params).to_parquet(results_dir / "params.parquet")
    # hyperparameters tuned on each asset's burn-in window; production reuses them
    (results_dir / "xgb_params.json").write_text(json.dumps(frozen, indent=2))
    with open(results_dir / "diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, default=str)
    log.info("walk-forward complete for %s; %d notes", assets, len(diag["notes"]))

    from voltail.evaluation.run import run_evaluation

    run_evaluation(cfg)
