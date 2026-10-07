"""Quantile / ES formulas used for parametric VaR and ES."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from voltail.models.distributions import (
    normal_es,
    skewt_es,
    skewt_quantile,
    std_t_es,
    std_t_quantile,
)


def test_normal_es_closed_form():
    # ES of N(0,1) at 2.5% is phi(z)/alpha = 2.3378
    assert normal_es(0.025) == pytest.approx(2.3378, abs=1e-4)


def test_std_t_es_matches_simulation():
    gen = np.random.default_rng(0)
    nu = 5.0
    z = gen.standard_t(nu, 2_000_000) * np.sqrt((nu - 2) / nu)
    q = std_t_quantile(0.01, nu)
    assert np.mean(z <= q) == pytest.approx(0.01, abs=3e-4)
    assert -z[z <= q].mean() == pytest.approx(std_t_es(0.01, nu), rel=0.02)


def test_skewt_reduces_to_student_t_at_zero_skew():
    for nu in (4.0, 8.0):
        u = np.array([0.001, 0.01, 0.025])
        np.testing.assert_allclose(skewt_quantile(u, nu, 0.0), std_t_quantile(u, nu), rtol=1e-8)
        assert skewt_es(0.01, nu, 0.0) == pytest.approx(std_t_es(0.01, nu), rel=1e-4)


def test_negative_skew_fattens_the_loss_tail():
    """lam < 0 must raise both the 1% loss quantile and the ES versus the symmetric t."""
    nu = 6.0
    assert -skewt_quantile(0.01, nu, -0.2) > -std_t_quantile(0.01, nu)
    assert skewt_es(0.01, nu, -0.2) > std_t_es(0.01, nu)
    assert skewt_es(0.01, nu, 0.2) < std_t_es(0.01, nu)


def test_skewt_es_matches_simulation():
    """Numerical ES integral agrees with Monte Carlo draws from arch's skew-t."""
    from arch.univariate import SkewStudent

    eta, lam, alpha = 6.0, -0.2, 0.01
    gen = np.random.default_rng(1)
    sim = SkewStudent(seed=gen).simulate([eta, lam])
    z = sim(2_000_000)
    assert z.std() == pytest.approx(1.0, abs=0.01)  # unit variance parameterisation
    q = np.quantile(z, alpha)
    assert -q == pytest.approx(-float(skewt_quantile(alpha, eta, lam)), rel=0.02)
    assert -z[z <= q].mean() == pytest.approx(skewt_es(alpha, eta, lam), rel=0.02)
    assert stats.skew(z) < 0
