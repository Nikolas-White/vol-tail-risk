"""Download and cache the market data locally (needs internet access to Yahoo Finance).

Usage
-----
    python scripts/download_data.py                  # uses config/default.yaml
    python scripts/download_data.py --config my.yaml

If Yahoo is unreachable you can drop CSVs manually into ``data/raw/`` instead:
one file per asset named ``<ASSET>.csv`` (e.g. ``SPX.csv``) with columns
``date,open,high,low,close,volume``, then run ``--from-csv``. The rest of the
pipeline is written against the cached parquet schema in
``src/voltail/data/schema.py`` and does not care where the data came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from voltail.config import load_config
from voltail.data.schema import RAW_COLUMNS, validate_raw
from voltail.data.store import clean_all, run_data_stage


def from_csv(cfg: dict, source: str = "manual_csv") -> None:
    """Build the raw parquet cache from ``data/raw/<ASSET>.csv`` and run the cleaning step.

    Duplicate dates are collapsed (last value kept) and a possibly-incomplete bar for
    today is dropped, exactly as in the yfinance path; both are recorded in the manifest.
    """
    from datetime import UTC, datetime

    from voltail.data.download import drop_incomplete_bar

    raw_dir = Path(cfg["paths"]["raw_dir"])
    manifest = {
        "source": source,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "start": cfg["data"]["start"],
        "auto_adjust": True,
        "assets": {},
    }
    for asset, spec in cfg["data"]["assets"].items():
        csv = raw_dir / f"{asset}.csv"
        if not csv.exists():
            manifest["assets"][asset] = {"used_ticker": None, "errors": [f"{csv} not found"]}
            continue
        df = pd.read_csv(csv, parse_dates=["date"], index_col="date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df[RAW_COLUMNS].astype(float)
        dups = df.index[df.index.duplicated(keep="last")]
        df = df[~df.index.duplicated(keep="last")]
        df = df[df.index >= pd.Timestamp(cfg["data"]["start"])]
        df, partial = drop_incomplete_bar(df, spec["asset_class"])
        validate_raw(df, asset)
        df.to_parquet(raw_dir / f"{asset}.parquet")
        manifest["assets"][asset] = {
            "used_ticker": spec["ticker"],
            "substituted": False,
            "file": csv.name,
            "rows": len(df),
            "first": str(df.index[0].date()),
            "last": str(df.index[-1].date()),
            "duplicate_dates_collapsed": [str(d.date()) for d in dups],
            "dropped_incomplete_bar": partial,
        }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    clean_all(cfg, manifest)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--from-csv", action="store_true", help="build the cache from data/raw/<ASSET>.csv files"
    )
    ap.add_argument("--source", default="manual_csv", help="label recorded in the manifest")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.from_csv:
        from_csv(cfg, args.source)
    else:
        run_data_stage(cfg)


if __name__ == "__main__":
    main()
