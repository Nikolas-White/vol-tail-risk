"""Download daily OHLC data from Yahoo Finance (via ``yfinance``) and cache to parquet.

Free data sources change and rate-limit without notice, so this module:

* retries each ticker with exponential back-off,
* falls back to alternative tickers listed in the config (e.g. ``GC=F`` for gold),
* records every substitution or failure in a manifest, which ends up in the
  data-quality report rather than being silently ignored.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from voltail.data.schema import RAW_COLUMNS, validate_raw
from voltail.logging_utils import get_logger

log = get_logger(__name__)


def _normalise_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance frame (possibly with MultiIndex columns) into the raw schema."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df[[c for c in RAW_COLUMNS if c in df.columns]].copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[RAW_COLUMNS].astype(float)


def download_ticker(
    ticker: str,
    start: str,
    end: str | None,
    auto_adjust: bool,
    retries: int = 4,
    backoff_s: float = 5.0,
) -> pd.DataFrame:
    """Download one ticker with retries. Raises ``RuntimeError`` if every attempt fails."""
    import yfinance as yf  # imported lazily so the rest of the package works offline

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                actions=False,
            )
            if df is not None and len(df) > 0:
                return _normalise_yf_frame(df)
            last_err = RuntimeError("empty frame returned")
        except Exception as exc:  # network / API errors are varied
            last_err = exc
        wait = backoff_s * 2 ** (attempt - 1)
        log.warning(
            "%s: attempt %d/%d failed (%s); retrying in %.0fs",
            ticker,
            attempt,
            retries,
            last_err,
            wait,
        )
        if attempt < retries:
            time.sleep(wait)
    raise RuntimeError(f"{ticker}: all {retries} download attempts failed: {last_err}")


def drop_incomplete_bar(
    df: pd.DataFrame, asset_class: str, now_utc: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Remove today's bar if it may still be forming (a run during market hours).

    * crypto: the daily bar is dated in UTC and closes at 00:00 UTC, so any row dated
      today (UTC) or later is partial;
    * everything else: a row dated today (New York) is dropped before 18:00 New York time.
    """
    now_utc = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if asset_class == "crypto":
        cutoff = now_utc.tz_localize(None).normalize()
    else:
        ny = now_utc.tz_convert("America/New_York")
        cutoff = ny.tz_localize(None).normalize() + (
            pd.Timedelta(days=1) if ny.hour >= 18 else pd.Timedelta(0)
        )
    partial = df.index >= cutoff
    return df[~partial], [str(d.date()) for d in df.index[partial]]


def download_all(cfg: dict[str, Any]) -> dict[str, Any]:
    """Download every configured asset (with fallbacks) into ``paths.raw_dir``.

    Returns the manifest dict, which is also written to ``raw_dir/manifest.json``.
    """
    dcfg = cfg["data"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "yfinance",
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "start": dcfg["start"],
        "end": dcfg["end"],
        "auto_adjust": dcfg["auto_adjust"],
        "assets": {},
    }
    for asset, spec in dcfg["assets"].items():
        candidates = [spec["ticker"], *spec.get("fallbacks", [])]
        entry: dict[str, Any] = {"requested_ticker": spec["ticker"], "errors": []}
        for ticker in candidates:
            try:
                df = download_ticker(
                    ticker,
                    dcfg["start"],
                    dcfg["end"],
                    dcfg["auto_adjust"],
                    dcfg.get("download_retries", 4),
                    dcfg.get("download_backoff_s", 5.0),
                )
                df, partial = drop_incomplete_bar(df, spec["asset_class"])
                if partial:
                    entry["dropped_incomplete_bar"] = partial
                    log.info("%s: dropped possibly-incomplete bar(s) %s", asset, partial)
                validate_raw(df, asset)
                df.to_parquet(raw_dir / f"{asset}.parquet")
                entry.update(
                    used_ticker=ticker,
                    substituted=ticker != spec["ticker"],
                    rows=len(df),
                    first=str(df.index[0].date()),
                    last=str(df.index[-1].date()),
                )
                if ticker != spec["ticker"]:
                    log.warning("%s: primary ticker failed, using fallback %s", asset, ticker)
                log.info(
                    "%s (%s): %d rows %s..%s", asset, ticker, len(df), entry["first"], entry["last"]
                )
                break
            except Exception as exc:
                entry["errors"].append(f"{ticker}: {exc}")
        else:
            entry["used_ticker"] = None
            log.error("%s: every ticker failed; asset will be skipped", asset)
        manifest["assets"][asset] = entry
    with open(raw_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    if not any(e.get("used_ticker") for e in manifest["assets"].values()):
        raise RuntimeError(
            "No asset could be downloaded. Check your internet connection / Yahoo rate limits, "
            "or place CSVs manually (see scripts/download_data.py --help)."
        )
    return manifest
