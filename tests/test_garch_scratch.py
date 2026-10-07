"""From-scratch GARCH(1,1) MLE vs the ``arch`` package, and parameter recovery."""

from __future__ import annotations

import numpy as np
import pytest
from arch import arch_model

from voltail.models.garch_scratch import fit_garch11, garch_variance


def simulate_garch11(
    n: int, mu: float, omega: float, alpha: float, beta: float, seed: int
) -> np.ndarray:
    gen = np.random.default_rng(seed)
    z = gen.standard_normal(n + 500)
    s2 = omega / (1 - alpha - beta)
    e_prev = 0.0
    out = np.empty(n + 500)
    for t in range(n + 500):
        s2 = omega + alpha * e_prev**2 + beta * s2
        e_prev = np.sqrt(s2) * z[t]
        out[t] = mu + e_prev
    return out[500:]  # drop burn-in


TRUE = {"mu": 0.05, "omega": 0.02, "alpha": 0.08, "beta": 0.90}


@pytest.fixture(scope="module")
def garch_sample() -> np.ndarray:
    return simulate_garch11(6000, **TRUE, seed=42)


@pytest.fixture(scope="module")
def both_fits(garch_sample):
    mine = fit_garch11(garch_sample)
    ref = arch_model(
        garch_sample, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")
    return mine, ref


def test_params_match_arch(both_fits):
    mine, ref = both_fits
    p = ref.params
    assert mine.mu == pytest.approx(p["mu"], abs=2e-3)
    assert mine.omega == pytest.approx(p["omega"], rel=0.03)
    assert mine.alpha == pytest.approx(p["alpha[1]"], abs=2e-3)
    assert mine.beta == pytest.approx(p["beta[1]"], abs=2e-3)


def test_loglik_matches_arch(both_fits):
    mine, ref = both_fits
    # same likelihood, same backcast -> optimum log-likelihoods within numerical tolerance
    assert mine.loglik == pytest.approx(ref.loglikelihood, abs=0.05)


def test_variance_recursion_identical_to_arch(garch_sample, both_fits):
    """Plugging arch's parameters into our filter must reproduce arch's conditional variance exactly."""
    _, ref = both_fits
    p = ref.params
    resid = garch_sample - p["mu"]
    from voltail.models.garch_scratch import arch_style_backcast

    ours = garch_variance(
        resid,
        p["omega"],
        p["alpha[1]"],
        p["beta[1]"],
        arch_style_backcast(garch_sample - garch_sample.mean()),
    )
    np.testing.assert_allclose(ours, ref.conditional_volatility**2, rtol=1e-6)


def test_standard_errors_close_to_arch(both_fits):
    mine, ref = both_fits
    # arch reports robust (sandwich) SEs by default; compare to its classical (Hessian) ones
    ref_classic = arch_model(
        ref.model._y, mean="Constant", vol="GARCH", dist="normal", rescale=False
    ).fit(disp="off", cov_type="classic")
    for k_mine, k_ref in [("alpha", "alpha[1]"), ("beta", "beta[1]")]:
        assert mine.std_errors[k_mine] == pytest.approx(ref_classic.std_err[k_ref], rel=0.15)


def test_recovers_true_parameters(both_fits):
    """End-to-end: data simulated from a known GARCH process recovers its parameters (within 3 SE)."""
    mine, _ = both_fits
    for k in ["alpha", "beta", "mu"]:
        assert abs(getattr(mine, k) - TRUE[k]) < 3 * mine.std_errors[k], k
    assert mine.persistence == pytest.approx(TRUE["alpha"] + TRUE["beta"], abs=0.02)


def test_forecast_matches_arch(garch_sample, both_fits):
    mine, ref = both_fits
    # forecasts from our fitted params vs arch's analytic forecast using our params
    am = arch_model(garch_sample, mean="Constant", vol="GARCH", dist="normal", rescale=False)
    fixed = am.fix([mine.mu, mine.omega, mine.alpha, mine.beta])
    ref_fc = fixed.forecast(horizon=5, reindex=False).variance.to_numpy().ravel()
    s2 = mine.variance_path(garch_sample)
    ours = mine.forecast(garch_sample[-1] - mine.mu, s2[-1], 5)
    np.testing.assert_allclose(ours, ref_fc, rtol=1e-8)


def test_stationarity_constraint_respected():
    # near-integrated data (IGARCH-like) must still give alpha + beta < 1
    y = simulate_garch11(3000, 0.0, 0.002, 0.10, 0.899, seed=3)
    fit = fit_garch11(y, compute_se=False)
    assert fit.alpha + fit.beta < 1.0
    assert fit.omega > 0
