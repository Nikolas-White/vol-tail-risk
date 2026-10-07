"""No-lookahead tests: the most important tests in the repository.

Strategy: run the walk-forward engine twice - once on the real series and once on a
copy where everything *after* a cut-off date c has been altered. A forecast for any
day up to and including the first trading day after c is made at the close of day
<= c, so it must be bit-for-bit identical in both runs. If any model peeked at
future data (through features, scaling, refits, smearing, CV folds...) this fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voltail.backtest.walkforward import refit_points, run_family
from voltail.config import load_config
from voltail.data.clean import clean_asset
from voltail.data.synthetic import simulate_universe
from voltail.models.baselines import BaselineFamily
from voltail.models.garch_arch import ArchFamily
from voltail.models.har import HARFamily
from voltail.models.ml import XGBFamily


@pytest.fixture(scope="module")
def smoke():
    cfg = load_config("config/smoke.yaml")
    frames, _ = simulate_universe(cfg)
    data = {
        a: clean_asset(frames[a], a, s["asset_class"], cfg["data"]["cleaning"])[0]
        for a, s in cfg["data"]["assets"].items()
    }
    return cfg, data


def perturb_after(
    df: pd.DataFrame, cut: pd.Timestamp, factor: float = 3.0, inclusive: bool = False
) -> pd.DataFrame:
    """Scale returns (x factor) and variance proxies (x factor^2) for rows after (or from) ``cut``."""
    out = df.copy()
    mask = out.index >= cut if inclusive else out.index > cut
    gen = np.random.default_rng(99)
    f = factor * gen.uniform(0.5, 1.5, mask.sum())
    out.loc[mask, "ret"] = out.loc[mask, "ret"].to_numpy() * f
    for c in ["sq_return", "parkinson", "garman_klass"]:
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * f**2
    for c in ["open", "high", "low", "close"]:
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * 1.1
    return out


def _cut_date(df: pd.DataFrame, cfg) -> pd.Timestamp:
    # a date in the middle of a refit block (not a period end) well inside the test sample
    pos = cfg["walkforward"]["min_train_obs"] + 150
    while (
        df.index[pos].to_period("M") != df.index[pos + 1].to_period("M") or df.index[pos].day < 10
    ):
        pos += 1
        if df.index[pos].day >= 10 and df.index[pos].to_period("M") == df.index[pos + 1].to_period(
            "M"
        ):
            break
    return df.index[pos]


FAMILIES = [
    ("baseline", lambda cfg, a, drv: BaselineFamily(cfg, a)),
    ("garch", lambda cfg, a, drv: ArchFamily(cfg, a)),
    ("har", lambda cfg, a, drv: HARFamily(cfg, a)),
    ("xgb", lambda cfg, a, drv: XGBFamily(cfg, a, context={"driver": drv})),
]


@pytest.mark.parametrize("fam_name,factory", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_future_data_does_not_change_past_forecasts(smoke, fam_name, factory):
    cfg, data = smoke
    df, drv = data["EURUSD"], data["SPX"]
    cut = _cut_date(df, cfg)
    next_day = df.index[df.index.get_loc(cut) + 1]

    fc_a, grid_a, _, _ = run_family(df, factory(cfg, "EURUSD", drv), cfg)
    fc_b, grid_b, _, _ = run_family(
        perturb_after(df, cut), factory(cfg, "EURUSD", perturb_after(drv, cut, inclusive=True)), cfg
    )

    keep = fc_a.index <= next_day
    assert keep.sum() > 50, "test window too small to be meaningful"
    num_cols = [c for c in fc_a.columns if not c.startswith("train_end")]
    pd.testing.assert_frame_equal(
        fc_a.loc[keep, num_cols],
        fc_b.loc[keep, num_cols],
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )
    for m in grid_a:
        np.testing.assert_allclose(
            grid_a[m].loc[keep].to_numpy(), grid_b[m].loc[keep].to_numpy(), rtol=1e-10, atol=1e-12
        )
    # sanity: the perturbation DOES change later forecasts (otherwise the test is vacuous)
    later = fc_a.index > next_day + pd.Timedelta(days=40)
    var_cols = [c for c in num_cols if c.startswith(("var1_", "VaR_"))]
    assert not np.allclose(
        fc_a.loc[later, var_cols].to_numpy(), fc_b.loc[later, var_cols].to_numpy()
    )


@pytest.mark.parametrize("fam_name,factory", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_parameters_estimated_strictly_before_forecast_date(smoke, fam_name, factory):
    cfg, data = smoke
    df = data["SPX"]
    fc, _, params, _ = run_family(df, factory(cfg, "SPX", None), cfg)
    te = fc[[c for c in fc.columns if c.startswith("train_end")][0]]
    assert (te < fc.index).all()
    for p in params:  # every refit date precedes every forecast it was used for
        used = te == p["refit_date"]
        if used.any():
            assert fc.index[used].min() > p["refit_date"]


def test_first_forecast_after_burn_in(smoke):
    cfg, data = smoke
    df = data["SPX"]
    fc, _, _, _ = run_family(df, BaselineFamily(cfg, "SPX"), cfg)
    assert fc.index[0] == df.index[cfg["walkforward"]["min_train_obs"]]


def test_refit_points_are_period_ends():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    pts = refit_points(idx, 50, "Q")
    assert pts[0] == 49
    ends = [idx[p] for p in pts[1:]]
    assert all(
        d.is_quarter_end or d == idx[-1] or idx[idx.get_loc(d) + 1].quarter != d.quarter
        for d in ends
    )


def test_cross_asset_driver_strictly_lagged():
    """A driver observation on the *same* date as the target must not be used."""
    from voltail.features.ml_features import cross_features

    dates = pd.bdate_range("2021-01-01", periods=60)
    drv = pd.DataFrame(
        {
            "ret": np.arange(60, dtype=float),
            "sq_return": np.arange(60, dtype=float) ** 2,
            "parkinson": 1.0,
            "garman_klass": 1.0,
        },
        index=dates,
    )
    cf = cross_features(dates, drv)
    # value at date d must equal the driver's return at the previous date (d-1)
    assert cf.loc[dates[30], "x_ret_1"] == drv.loc[dates[29], "ret"]


@pytest.mark.parametrize(
    "fam_name,factory",
    [f for f in FAMILIES if f[0] in ("garch", "har", "xgb")],
    ids=["garch", "har", "xgb"],
)
def test_no_lookahead_with_monthly_refits_and_rolling_window(smoke, fam_name, factory):
    """Same perturbation test under the *default* monthly schedule and a rolling estimation window."""
    import copy

    cfg, data = smoke
    cfg = copy.deepcopy(cfg)
    cfg["walkforward"]["refit_every"] = {"garch": "M", "har": "M", "ml": "Q"}
    cfg["walkforward"]["window"] = "rolling"
    cfg["walkforward"]["rolling_obs"] = 500
    cfg["models"]["xgb"]["tune_grid"] = {**cfg["models"]["xgb"]["tune_grid"], "max_depth": [2, 3]}
    df, drv = data["EURUSD"], data["SPX"]
    cut = _cut_date(df, cfg)
    next_day = df.index[df.index.get_loc(cut) + 1]
    fc_a, _, _, _ = run_family(df, factory(cfg, "EURUSD", drv), cfg)
    fc_b, _, _, _ = run_family(
        perturb_after(df, cut), factory(cfg, "EURUSD", perturb_after(drv, cut, inclusive=True)), cfg
    )
    keep = fc_a.index <= next_day
    num_cols = [c for c in fc_a.columns if not c.startswith("train_end")]
    pd.testing.assert_frame_equal(
        fc_a.loc[keep, num_cols],
        fc_b.loc[keep, num_cols],
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_forecast_and_realized_alignment(smoke):
    """The value stored at date t must be the forecast made at the close of t-1, scored against day t."""
    from voltail.backtest.walkforward import realized_targets
    from voltail.models.baselines import ewma_variance

    cfg, data = smoke
    df = data["SPX"]
    fc, _, _, _ = run_family(df, BaselineFamily(cfg, "SPX"), cfg)
    ew = pd.Series(
        ewma_variance(df["ret"].to_numpy(), cfg["models"]["ewma_lambda"]), index=df.index
    )
    t = fc.index[100]
    prev = df.index[df.index.get_loc(t) - 1]
    assert fc.loc[t, "var1_ewma"] == pytest.approx(ew.loc[prev])  # made with data up to t-1
    rt = realized_targets(df, cfg)
    pos = df.index.get_loc(t)
    assert rt.loc[t, "rv1_sq"] == pytest.approx(
        df["ret"].iloc[pos] ** 2
    )  # scored against day t itself
    assert rt.loc[t, "rv5_sq"] == pytest.approx(
        (df["ret"].iloc[pos : pos + 5] ** 2).mean()
    )  # t..t+4
    scale_prev = rt["scale"].loc[t]
    from voltail.features.proxies import expanding_scale

    expected_scale = expanding_scale(
        df["sq_return"], df["garman_klass"], cfg["proxies"]["scale_min_obs"]
    ).loc[prev]
    assert scale_prev == pytest.approx(expected_scale)  # proxy scaling uses data up to t-1 only
