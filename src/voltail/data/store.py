"""Data stage orchestration and parquet I/O.

``run_data_stage`` = download (or simulate) -> clean -> write processed parquet
+ a JSON/Markdown data-quality report listing every dropped or flagged row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from voltail.data.clean import clean_asset
from voltail.data.schema import validate_processed
from voltail.logging_utils import get_logger

log = get_logger(__name__)


def raw_path(cfg: dict[str, Any], asset: str) -> Path:
    return Path(cfg["paths"]["raw_dir"]) / f"{asset}.parquet"


def processed_path(cfg: dict[str, Any], asset: str) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / f"{asset}.parquet"


def acquire_raw(cfg: dict[str, Any]) -> dict[str, Any]:
    """Download real data or simulate synthetic data into ``raw_dir``. Returns a manifest."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    if cfg["data"]["source"] == "synthetic":
        from voltail.data.synthetic import simulate_universe

        frames, truth = simulate_universe(cfg)
        true_var = truth.pop("true_variance")
        for asset, df in frames.items():
            df.to_parquet(raw_path(cfg, asset))
            true_var[asset].to_frame().to_parquet(raw_dir / f"truth_{asset}.parquet")
        manifest = {
            "source": "synthetic",
            "truth": truth,
            "assets": {
                a: {"used_ticker": f"SYNTHETIC:{a}", "rows": len(d)} for a, d in frames.items()
            },
        }
        with open(raw_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest
    from voltail.data.download import download_all

    return download_all(cfg)


def clean_all(cfg: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Clean every raw file and write processed parquet + quality report."""
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    quality: dict[str, Any] = {"source": manifest.get("source"), "assets": {}}
    for asset, spec in cfg["data"]["assets"].items():
        entry = manifest["assets"].get(asset, {})
        if not entry.get("used_ticker") or not raw_path(cfg, asset).exists():
            quality["assets"][asset] = {
                "skipped": True,
                "reason": entry.get("errors", "no raw file"),
            }
            log.warning("%s: no raw data, skipped", asset)
            continue
        raw = pd.read_parquet(raw_path(cfg, asset))
        processed, rep = clean_asset(raw, asset, spec["asset_class"], cfg["data"]["cleaning"])
        rep["ticker"] = entry.get("used_ticker")
        rep["substituted"] = bool(entry.get("substituted", False))
        processed.to_parquet(processed_path(cfg, asset))
        quality["assets"][asset] = rep
        log.info(
            "%s: %d raw -> %d processed rows (stale %d, spikes %d, weekend %d, range-flagged %d)",
            asset,
            rep["raw_rows"],
            rep["processed_rows"],
            rep["dropped_stale"]["n"],
            rep["dropped_spike_reversal"]["n"],
            rep["dropped_weekend"]["n"],
            rep["range_flagged_unusable"]["n"],
        )
    with open(out_dir / "data_quality.json", "w", encoding="utf-8") as fh:
        json.dump(quality, fh, indent=2)
    return quality


def run_data_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    """Acquire and clean all data. Returns the quality report."""
    manifest = acquire_raw(cfg)
    return clean_all(cfg, manifest)


def load_processed(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load every processed asset that exists, in config order."""
    out: dict[str, pd.DataFrame] = {}
    for asset in cfg["data"]["assets"]:
        p = processed_path(cfg, asset)
        if p.exists():
            df = pd.read_parquet(p)
            validate_processed(df, asset)
            out[asset] = df
    if not out:
        raise FileNotFoundError(
            f"No processed data in {cfg['paths']['processed_dir']}; run `make data` first."
        )
    return out


def load_quality(cfg: dict[str, Any]) -> dict[str, Any]:
    p = Path(cfg["paths"]["processed_dir"]) / "data_quality.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    p = Path(cfg["paths"]["raw_dir"]) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}
