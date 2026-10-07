"""Download path (real-data branch) tested offline by mocking ``yfinance.download``."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from voltail.config import load_config
from voltail.data.store import clean_all, load_processed


def _fake_frame(n: int = 900, seed: int = 0) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, tz="America/New_York")
    close = 100 * np.exp(np.cumsum(gen.normal(0, 0.01, n)))
    open_ = close * np.exp(gen.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(gen.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(gen.normal(0, 0.004, n)))
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["X"]])
    df = pd.DataFrame(
        np.column_stack([open_, high, low, close, np.ones(n)]), index=idx, columns=cols
    )
    df.iloc[5, 3] = np.nan  # a missing close, as yfinance sometimes returns
    return df


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c["paths"]["raw_dir"] = str(tmp_path / "raw")
    c["paths"]["processed_dir"] = str(tmp_path / "processed")
    c["data"]["download_retries"] = 2
    c["data"]["download_backoff_s"] = 0.0
    c["data"]["assets"] = {
        "SPX": {"ticker": "^GSPC", "fallbacks": ["SPY"], "name": "S&P", "asset_class": "equity"},
        "GOLD": {
            "ticker": "GLD",
            "fallbacks": ["GC=F"],
            "name": "Gold",
            "asset_class": "commodity",
        },
        "DEAD": {"ticker": "NOPE", "fallbacks": [], "name": "Dead", "asset_class": "equity"},
    }
    return c


def test_fallbacks_and_skips_are_recorded(cfg, monkeypatch):
    calls: list[str] = []

    def fake_download(ticker, **kwargs):
        calls.append(ticker)
        if ticker in ("^GSPC", "GC=F"):
            return _fake_frame(seed=1)
        if ticker == "GLD":
            raise ConnectionError("rate limited")
        return pd.DataFrame()  # NOPE / SPY: empty

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    from voltail.data.download import download_all

    manifest = download_all(cfg)
    a = manifest["assets"]
    assert a["SPX"]["used_ticker"] == "^GSPC" and not a["SPX"]["substituted"]
    assert a["GOLD"]["used_ticker"] == "GC=F" and a["GOLD"]["substituted"]
    assert a["DEAD"]["used_ticker"] is None and a["DEAD"]["errors"]
    assert calls.count("GLD") == 2  # retried before falling back

    quality = clean_all(cfg, manifest)
    assert quality["assets"]["DEAD"]["skipped"]
    assert quality["assets"]["SPX"]["dropped_missing_close"]["n"] == 1
    data = load_processed(cfg)
    assert set(data) == {"SPX", "GOLD"}
    assert data["SPX"].index.tz is None
    with open(f"{cfg['paths']['raw_dir']}/manifest.json", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["assets"]["GOLD"]["used_ticker"] == "GC=F"


def test_all_assets_failing_raises(cfg, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(download=lambda t, **k: pd.DataFrame())
    )
    from voltail.data.download import download_all

    with pytest.raises(RuntimeError, match="No asset could be downloaded"):
        download_all(cfg)


def test_incomplete_bar_dropped():
    from voltail.data.download import drop_incomplete_bar

    idx = pd.DatetimeIndex(["2026-03-02", "2026-03-03", "2026-03-04"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    # 15:00 New York on 2026-03-04 (20:00 UTC): equity bar for 03-04 still forming
    during = pd.Timestamp("2026-03-04 20:00", tz="UTC")
    out, dropped = drop_incomplete_bar(df, "equity", during)
    assert dropped == ["2026-03-04"] and len(out) == 2
    # 19:00 New York (00:00 UTC next day): bar complete, keep it
    after = pd.Timestamp("2026-03-05 00:00", tz="UTC")
    assert len(drop_incomplete_bar(df, "equity", after)[0]) == 3
    # crypto bar dated 03-04 UTC is partial until 00:00 UTC on 03-05
    assert drop_incomplete_bar(df, "crypto", during)[1] == ["2026-03-04"]
    assert drop_incomplete_bar(df, "crypto", after)[1] == []
