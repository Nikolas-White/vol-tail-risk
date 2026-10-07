"""HAR / OLS from scratch vs statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from voltail.models.har import HARModel, har_features, har_target
from voltail.models.ols import ols


@pytest.fixture(scope="module")
def regression_data():
    gen = np.random.default_rng(0)
    n = 1500
    x = np.column_stack([np.ones(n), gen.normal(size=(n, 3))])
    # AR(1) errors so HAC and classical SEs differ
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + gen.normal()
    y = x @ np.array([0.5, 1.0, -2.0, 0.3]) + e
    return y, x


def test_ols_params_and_se_match_statsmodels(regression_data):
    y, x = regression_data
    mine = ols(y, x, hac_lags=8)
    ref = sm.OLS(y, x).fit()
    np.testing.assert_allclose(mine.params, ref.params, rtol=1e-10)
    np.testing.assert_allclose(mine.bse, ref.bse, rtol=1e-10)
    assert mine.r2 == pytest.approx(ref.rsquared, rel=1e-10)


def test_ols_hac_matches_statsmodels(regression_data):
    y, x = regression_data
    mine = ols(y, x, hac_lags=8)
    ref = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 8, "use_correction": False})
    np.testing.assert_allclose(mine.bse_hac, ref.bse, rtol=1e-8)


def test_har_matches_statsmodels_on_proxy(synthetic_equity):
    v = synthetic_equity["garman_klass"]
    model = HARModel.fit(v, horizon=1)
    x = har_features(v)
    y = har_target(v, 1)
    data = pd.concat([x, y.rename("y")], axis=1).dropna()
    ref = sm.OLS(data["y"], sm.add_constant(data[["har_d", "har_w", "har_m"]])).fit()
    np.testing.assert_allclose(model.result.params, ref.params.to_numpy(), rtol=1e-8)


def test_har_features_no_lookahead(synthetic_equity):
    v = synthetic_equity["garman_klass"]
    cut = v.index[1000]
    f1 = har_features(v)
    v2 = v.copy()
    v2.loc[v2.index > cut] *= 50.0
    f2 = har_features(v2)
    pd.testing.assert_frame_equal(f1.loc[:cut], f2.loc[:cut])


def test_har_target_is_forward_mean():
    v = pd.Series(np.arange(1.0, 11.0))
    t = har_target(v, 5, log=False)
    assert t.iloc[0] == pytest.approx(np.mean([2, 3, 4, 5, 6]))
    assert np.isnan(t.iloc[6])  # needs t+1..t+5, only 3 left


def test_har_coefficients_sensible(synthetic_equity):
    """On persistent GARCH data the HAR slopes are positive and sum to < 1."""
    m = HARModel.fit(synthetic_equity["garman_klass"], horizon=1)
    p = m.params
    assert p["har_d"] + p["har_w"] + p["har_m"] < 1.0
    assert p["har_w"] + p["har_m"] > 0
    assert m.smear > 1.0  # E[exp(u)] > 1 by Jensen
