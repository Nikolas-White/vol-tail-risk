"""Data pipeline tests: proxies, cleaning, schema, calendar handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voltail.data.clean import clean_asset, find_spikes
from voltail.data.download import _normalise_yf_frame
from voltail.data.schema import PROCESSED_COLUMNS, validate_raw
from voltail.data.synthetic import simulate_universe
from voltail.features.proxies import expanding_scale, garman_klass, parkinson


def test_parkinson_known_value():
    # H/L = e^0.02 -> (ln H/L)^2 = 4e-4 -> *1e4 / (4 ln2) = 4 / (4 ln 2) = 1/ln 2
    out = parkinson(pd.Series([np.exp(0.02)]), pd.Series([1.0]))
    assert out.iloc[0] == pytest.approx(1.0 / np.log(2.0))


def test_garman_klass_known_value():
    o, h, low, c = 1.0, np.exp(0.02), 1.0, np.exp(0.01)
    expected = 1e4 * (0.5 * 0.02**2 - (2 * np.log(2) - 1) * 0.01**2)
    out = garman_klass(pd.Series([o]), pd.Series([h]), pd.Series([low]), pd.Series([c]))
    assert out.iloc[0] == pytest.approx(expected)


def test_range_proxies_unbiased_for_brownian_motion():
    """Under driftless Brownian motion, Parkinson and GK are unbiased for the variance."""
    gen = np.random.default_rng(0)
    n_days, n_steps, sigma = 4000, 2000, 0.01
    paths = np.cumsum(gen.standard_normal((n_days, n_steps)) * sigma / np.sqrt(n_steps), axis=1)
    paths = np.concatenate([np.zeros((n_days, 1)), paths], axis=1)
    o = pd.Series(np.ones(n_days))
    h = pd.Series(np.exp(paths.max(axis=1)))
    low = pd.Series(np.exp(paths.min(axis=1)))
    c = pd.Series(np.exp(paths[:, -1]))
    true_var_pct2 = (100 * sigma) ** 2
    # discrete monitoring biases range estimators slightly downward; allow 6%
    assert parkinson(h, low).mean() == pytest.approx(true_var_pct2, rel=0.06)
    assert garman_klass(o, h, low, c).mean() == pytest.approx(true_var_pct2, rel=0.06)


def test_expanding_scale_uses_only_past():
    num = pd.Series(np.arange(1, 11, dtype=float))
    den = pd.Series(np.ones(10))
    s1 = expanding_scale(num, den, min_obs=3)
    num2 = num.copy()
    num2.iloc[7:] = 1e6  # perturb the "future"
    s2 = expanding_scale(num2, den, min_obs=3)
    pd.testing.assert_series_equal(s1.iloc[:7], s2.iloc[:7])
    assert np.isnan(s1.iloc[1]) and s1.iloc[2] == pytest.approx(2.0)


def test_find_spikes_detects_bad_tick():
    gen = np.random.default_rng(1)
    close = pd.Series(
        100 * np.exp(np.cumsum(gen.normal(0, 0.01, 600))),
        index=pd.bdate_range("2020-01-01", periods=600),
    )
    close.iloc[400] *= 1.3
    spikes = find_spikes(close, mad_mult=12, reversal_frac=0.8, lookback=250)
    assert list(spikes) == [close.index[400]]


def test_genuine_crash_not_flagged_as_spike():
    """A large move that does NOT revert (e.g. a real crash) must be kept."""
    gen = np.random.default_rng(2)
    r = gen.normal(0, 0.01, 600)
    r[400] = -0.20  # crash, no reversal
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2020-01-01", periods=600))
    assert len(find_spikes(close, 12, 0.8, 250)) == 0


def test_genuine_whipsaw_in_crisis_not_flagged():
    """COVID-style -10% / +9% days inside a high-volatility spell are real and must be kept."""
    gen = np.random.default_rng(5)
    r = gen.normal(0, 0.01, 600)
    r[380:400] = gen.normal(0, 0.04, 20)  # volatility already elevated (like early March 2020)
    r[400], r[401] = -0.10, 0.089
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2020-01-01", periods=600))
    from voltail.config import load_config

    cc = load_config()["data"]["cleaning"]
    assert (
        len(
            find_spikes(
                close, cc["spike_mad_mult"], cc["spike_reversal_frac"], cc["spike_lookback"]
            )
        )
        == 0
    )


def test_bad_wick_flagged_but_real_flash_crash_kept():
    from voltail.config import load_config

    gen = np.random.default_rng(9)
    n = 300
    close = 1.3 * np.exp(np.cumsum(gen.normal(0, 0.005, n)))
    idx = pd.bdate_range("2012-01-02", periods=n)
    high, low = close * 1.004, close * 0.996
    low[150] = close[150] * 0.58  # garbage low print, close unaffected
    low[200] = close[200] * 0.95  # big but plausible flash move (~12x normal range)
    raw = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx
    )
    raw.index.name = "date"
    df, rep = clean_asset(raw, "FX", "fx", load_config()["data"]["cleaning"])
    assert not df.loc[idx[150], "range_ok"] and np.isnan(df.loc[idx[150], "garman_klass"])
    assert df.loc[idx[200], "range_ok"]
    assert rep["range_flagged_bad_wick"]["n"] == 1


def test_cleaning_catches_injected_defects():
    from voltail.config import load_config

    cfg = load_config("config/synthetic.yaml")
    frames, truth = simulate_universe(cfg)
    asset = "EURUSD"
    df, rep = clean_asset(frames[asset], asset, "fx", cfg["data"]["cleaning"])
    defects = truth["injected_defects"][asset]
    spike_day = pd.Timestamp(defects["spike"][0])
    assert spike_day not in df.index
    assert rep["dropped_spike_reversal"]["n"] == 1
    for d in defects["zero_range"]:
        assert not df.loc[pd.Timestamp(d), "range_ok"]
        assert np.isnan(df.loc[pd.Timestamp(d), "garman_klass"])
    assert list(df.columns) == PROCESSED_COLUMNS
    assert df["ret"].notna().all()


def test_weekend_and_stale_rows_dropped():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")  # includes a weekend
    close = np.linspace(1.0, 1.1, 10)
    raw = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 0.0},
        index=idx,
    )
    raw.index.name = "date"
    raw.iloc[6, :4] = raw.iloc[5]["close"]  # stale: same close, high == low
    from voltail.config import load_config

    df, rep = clean_asset(raw, "X", "fx", load_config()["data"]["cleaning"])
    assert (df.index.dayofweek < 5).all()
    assert rep["dropped_stale"]["n"] == 1
    assert pd.Timestamp("2021-01-07") not in df.index
    assert rep["dropped_weekend"]["n"] == int((idx.dayofweek >= 5).sum())
    # crypto keeps weekends
    df_c, rep_c = clean_asset(raw, "X", "crypto", load_config()["data"]["cleaning"])
    assert rep_c["dropped_weekend"]["n"] == 0


def test_normalise_yf_multiindex_frame():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["^GSPC"]])
    df = pd.DataFrame(np.arange(10, dtype=float).reshape(2, 5), index=idx, columns=cols)
    out = _normalise_yf_frame(df)
    validate_raw(out, "SPX")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_validate_raw_rejects_bad_schema():
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
    with pytest.raises(ValueError):
        validate_raw(df, "BAD")


def test_synthetic_calendar_per_asset():
    from voltail.config import load_config

    frames, _ = simulate_universe(load_config("config/synthetic.yaml"))
    assert (frames["SPX"].index.dayofweek < 5).all()
    assert (frames["BTC"].index.dayofweek >= 5).any()
    assert (
        pd.Timestamp("2010-07-05") not in frames["SPX"].index
    )  # US holiday (Independence Day observed)
