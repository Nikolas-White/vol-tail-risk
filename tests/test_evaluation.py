"""Evaluation statistics on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from voltail.evaluation.es_tests import z2_statistic, z2_test
from voltail.evaluation.forecast_tests import (
    diebold_mariano,
    mincer_zarnowitz,
    model_confidence_set,
)
from voltail.evaluation.losses import fz0, mse, qlike, qlike_normalised
from voltail.evaluation.var_tests import (
    christoffersen_ind,
    hits,
    kupiec_pof,
    rolling_zones,
    traffic_light,
    var_backtest,
)
from voltail.models.distributions import normal_es, tail_grid

# ----------------------------------------------------------------------------- losses


def test_qlike_known_values():
    assert qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(1.0 + np.log(2.0))
    assert qlike(np.array([0.0]), np.array([1.0]))[0] == pytest.approx(
        0.0
    )  # finite on zero-return days
    assert qlike_normalised(np.array([3.0]), np.array([3.0]))[0] == pytest.approx(0.0)
    # forms differ only by a proxy-dependent constant
    p, f1, f2 = np.array([1.7]), np.array([1.0]), np.array([2.5])
    assert (qlike(p, f1) - qlike(p, f2))[0] == pytest.approx(
        (qlike_normalised(p, f1) - qlike_normalised(p, f2))[0]
    )


def test_qlike_minimised_at_true_variance_with_noisy_proxy():
    """Patton robustness: with an unbiased noisy proxy (r^2), expected QLIKE is minimised at the true variance."""
    gen = np.random.default_rng(0)
    true_var = 2.0
    proxy = true_var * gen.standard_normal(200_000) ** 2
    grid = np.linspace(1.0, 3.0, 41)
    losses = [qlike(proxy, np.full_like(proxy, g)).mean() for g in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(true_var, abs=0.06)


def test_qlike_penalises_underprediction_more():
    assert (
        qlike(np.array([1.0]), np.array([0.5]))[0] - qlike(np.array([1.0]), np.array([1.0]))[0]
        > qlike(np.array([1.0]), np.array([1.5]))[0] - qlike(np.array([1.0]), np.array([1.0]))[0]
    )


def test_mse_known_value():
    assert mse(np.array([3.0]), np.array([1.0]))[0] == 4.0


def test_fz0_known_value_and_consistency():
    # hand-computed: y=-3, VaR=2, ES=2.5, alpha=0.01 -> v=-2, e=-2.5
    v, e, y, a = -2.0, -2.5, -3.0, 0.01
    expected = -1 / (a * e) * (v - y) + v / e + np.log(-e) - 1
    assert fz0(np.array([y]), np.array([2.0]), np.array([2.5]), a)[0] == pytest.approx(expected)
    # strict consistency: true (VaR, ES) of N(0,1) beats perturbed ones on average
    gen = np.random.default_rng(1)
    r = gen.standard_normal(400_000)
    alpha = 0.025
    true_var, true_es = -stats.norm.ppf(alpha), normal_es(alpha)
    base = fz0(r, np.full_like(r, true_var), np.full_like(r, true_es), alpha).mean()
    for dv, de in [(0.3, 0.3), (-0.3, -0.3), (0.0, 0.4), (0.0, -0.3)]:
        alt = fz0(r, np.full_like(r, true_var + dv), np.full_like(r, true_es + de), alpha).mean()
        assert alt > base


# ----------------------------------------------------------------------------- VaR tests


def test_kupiec_hand_computed():
    hit = np.zeros(250, dtype=int)
    hit[:10] = 1
    lr, p = kupiec_pof(hit, 0.01)
    pi = 10 / 250
    expected = -2 * (
        (240 * np.log(0.99) + 10 * np.log(0.01)) - (240 * np.log(1 - pi) + 10 * np.log(pi))
    )
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(stats.chi2.sf(expected, 1))


def test_kupiec_zero_exceedances_handled():
    lr, p = kupiec_pof(np.zeros(1000, dtype=int), 0.01)
    assert lr == pytest.approx(-2 * 1000 * np.log(0.99))
    assert p < 0.01


@pytest.mark.parametrize("alpha", [0.01, 0.025])
def test_kupiec_and_christoffersen_size_and_power(alpha):
    """Correct VaR passes at about the nominal rate; a miscalibrated VaR fails almost always."""
    gen = np.random.default_rng(2)
    n, reps = 1000, 300
    q = -stats.norm.ppf(alpha)
    rej_good = rej_bad = rej_cc_good = 0
    for _ in range(reps):
        r = gen.standard_normal(n)
        good = var_backtest(r, np.full(n, q), alpha)
        bad = var_backtest(r, np.full(n, 0.75 * q), alpha)  # VaR 25% too small
        rej_good += good.p_pof < 0.05
        rej_cc_good += good.p_cc < 0.05
        rej_bad += bad.p_pof < 0.05
    assert rej_good / reps < 0.10  # nominal 5% (discrete test: allow slack)
    assert rej_cc_good / reps < 0.12
    assert rej_bad / reps > 0.90


def test_christoffersen_detects_clustering():
    """Hits with the right *average* rate but strong clustering fail independence; iid hits pass."""
    gen = np.random.default_rng(3)
    n = 3000
    # Markov chain: P(hit | no hit) = 0.007, P(hit | hit) = 0.30  -> stationary rate ~ 0.01
    h = np.zeros(n, dtype=int)
    for t in range(1, n):
        p = 0.30 if h[t - 1] else 0.007
        h[t] = gen.random() < p
    _, p_clustered = christoffersen_ind(h)
    iid = (gen.random(n) < 0.01).astype(int)
    _, p_iid = christoffersen_ind(iid)
    assert p_clustered < 0.01
    assert p_iid > 0.05


def test_unconditional_var_fails_on_garch_data_but_conditional_passes():
    """Constant VaR on volatility-clustered returns -> clustered hits; true conditional VaR -> passes."""
    from voltail.data.synthetic import CLASS_PARAMS, simulate_gjr_t

    gen = np.random.default_rng(4)
    p = CLASS_PARAMS["equity"]
    r, h = simulate_gjr_t(6000, p, gen)
    alpha = 0.01
    from voltail.models.distributions import std_t_quantile

    cond_var = -(p.mu + np.sqrt(h) * std_t_quantile(alpha, p.nu))
    uncond_var = np.full_like(
        r, -np.quantile(r, alpha)
    )  # even "knowing" the right unconditional level
    good = var_backtest(r, cond_var, alpha)
    bad = var_backtest(r, uncond_var, alpha)
    assert good.p_cc > 0.05
    assert bad.p_ind < 0.01


def test_basel_traffic_light_matches_bcbs_table():
    zones = [traffic_light(x, 250, 0.01) for x in range(15)]
    assert zones[:5] == ["green"] * 5
    assert zones[5:10] == ["yellow"] * 5
    assert zones[10:] == ["red"] * 5


def test_rolling_zones():
    h = np.zeros(600, dtype=int)
    h[100:112] = 1  # 12 exceptions within one 250d window -> red windows exist
    z = rolling_zones(h, 250, 0.01)
    assert z["red"] > 0 and z["max_exceed_250d"] == 12
    assert z["green"] + z["yellow"] + z["red"] == pytest.approx(1.0)


def test_hits_convention():
    assert list(hits(np.array([-3.0, -1.0, 2.0]), np.array([2.0, 2.0, 2.0]))) == [1, 0, 0]


# ----------------------------------------------------------------------------- ES test


def _normal_grid(n: int, sigma: np.ndarray, k: int = 40) -> tuple[np.ndarray, np.ndarray]:
    u = tail_grid(k, 0.025)
    return u, sigma[:, None] * stats.norm.ppf(u)[None, :]


def test_z2_correct_model_not_rejected_too_often_and_bad_model_rejected():
    gen = np.random.default_rng(5)
    n, alpha, reps = 1500, 0.025, 60
    rej_good = rej_bad = 0
    for i in range(reps):
        sigma = np.exp(gen.normal(0, 0.3, n))
        r = sigma * gen.standard_normal(n)
        u, grid = _normal_grid(n, sigma)
        var = -sigma * stats.norm.ppf(alpha)
        es = sigma * normal_es(alpha)
        good = z2_test(r, var, es, grid, u, alpha, sims=500, seed=i)
        # a model that thinks volatility is 30% lower than it is
        u_b, grid_b = _normal_grid(n, 0.7 * sigma)
        bad = z2_test(r, 0.7 * var, 0.7 * es, grid_b, u_b, alpha, sims=500, seed=i)
        rej_good += good.p_value < 0.05
        rej_bad += bad.p_value < 0.05
    assert rej_good / reps < 0.15
    assert rej_bad / reps > 0.9


def test_z2_zero_under_exact_expectation():
    # if every hit loss equals ES and the hit rate equals alpha, Z2 = 0 exactly
    n, alpha = 400, 0.025
    r = np.zeros(n)
    r[:10] = -3.0
    assert z2_statistic(r, np.full(n, 2.0), np.full(n, 3.0), alpha) == pytest.approx(0.0)


# ----------------------------------------------------------------------------- forecast comparison


def test_mincer_zarnowitz_unbiased_vs_biased():
    gen = np.random.default_rng(6)
    n = 40_000  # r^2-style proxies are very noisy; MZ needs a long sample to be precise
    f = np.exp(gen.normal(0, 0.5, n))
    proxy = f * gen.chisquare(1, n)  # unbiased noisy proxy
    good = mincer_zarnowitz(proxy, f)
    bad = mincer_zarnowitz(proxy, 2.0 * f)
    assert abs(good.b - 1.0) < 3 * good.se_b and abs(good.a) < 3 * good.se_a
    assert good.p_value > 0.01
    assert bad.p_value < 0.001 and bad.b < 0.7


def test_diebold_mariano():
    gen = np.random.default_rng(7)
    n = 2000
    l1 = gen.chisquare(1, n)
    same = diebold_mariano(l1, l1.copy())
    assert same.stat == 0.0 and same.p_value == 1.0
    better = diebold_mariano(l1, l1 + 0.2 + gen.normal(0, 0.1, n))
    assert better.stat < 0 and better.p_value < 1e-6


def test_model_confidence_set_excludes_bad_model():
    gen = np.random.default_rng(8)
    n = 1500
    base = gen.chisquare(1, n)
    losses = pd.DataFrame(
        {
            "good_a": base + gen.normal(0, 0.05, n),
            "good_b": base + gen.normal(0, 0.05, n),
            "bad": base + 0.5 + gen.normal(0, 0.05, n),
        }
    )
    res = model_confidence_set(losses, size=0.10, reps=300, block_size=10, seed=1)
    assert res.loc["good_a", "in_mcs"] and res.loc["good_b", "in_mcs"]
    assert not res.loc["bad", "in_mcs"]


def test_skew_research_verdict_rule():
    """The pre-registered decision rule is applied exactly as written."""
    from voltail.reporting.research import verdict

    lam = pd.DataFrame(
        {"lambda_full_sample": [-0.15, -0.12], "t_stat": [-5.0, -4.0]}, index=["SPX", "TSX"]
    )

    def cmp(p_kupiec, zone, diff, p_dm):
        return pd.DataFrame(
            [
                {
                    "asset": a,
                    "level": 0.99,
                    "rate_gjr_t": 0.017,
                    "rate_gjr_skewt": 0.011,
                    "p_kupiec_gjr_t": 0.001,
                    "p_kupiec_gjr_skewt": p_kupiec,
                    "zone_gjr_t": "red",
                    "zone_gjr_skewt": zone,
                    "fz0_diff": diff,
                    "dm_p_one_sided": p_dm,
                }
                for a in ("SPX", "TSX")
            ]
        )

    assert verdict(cmp(0.4, "green", -0.01, 0.01), lam)[0] == "SUPPORTED"
    assert verdict(cmp(0.4, "green", -0.01, 0.30), lam)[0] == "PARTIALLY SUPPORTED"
    assert verdict(cmp(0.01, "red", 0.01, 0.90), lam)[0] == "NOT SUPPORTED"
    lam_pos = lam.assign(lambda_full_sample=[0.05, -0.12])
    assert verdict(cmp(0.4, "green", -0.01, 0.01), lam_pos)[0] == "NOT SUPPORTED"


def test_h2_verdict_rule():
    from voltail.reporting.research import verdict_h2

    def rows(pair, pass_b, pass_n, better, worse):
        return [
            {
                "pair": pair,
                "pass_base": pb,
                "pass_new": pn,
                "dm_p_new_better": 0.01 if i < better else 0.5,
                "dm_p_new_worse": 0.01 if better <= i < better + worse else 0.5,
            }
            for i, (pb, pn) in enumerate(zip(pass_b, pass_n))
        ]

    up = rows("a", [0, 0, 1, 1], [1, 1, 1, 1], 2, 0) + rows("b", [0, 1, 1, 1], [1, 1, 1, 1], 1, 0)
    assert verdict_h2(pd.DataFrame(up))[0] == "SUPPORTED"
    mixed = rows("a", [1, 1, 1, 1], [1, 1, 1, 1], 2, 0) + rows(
        "b", [1, 1, 1, 1], [1, 1, 1, 1], 0, 0
    )
    assert verdict_h2(pd.DataFrame(mixed))[0] == "PARTIALLY SUPPORTED"
    worse = rows("a", [1, 1, 1, 1], [1, 0, 1, 1], 0, 2) + rows(
        "b", [1, 1, 1, 1], [1, 1, 1, 1], 0, 0
    )
    assert verdict_h2(pd.DataFrame(worse))[0] == "NOT SUPPORTED"
