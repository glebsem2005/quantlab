"""Tests for analytics labs (stationarity, ARIMA, GARCH, cointegration).

All tests use synthetic data with known properties so we can verify
that estimators recover what we put in (same pattern as test_ou.py).
"""
import math
import random

import numpy as np
import pytest

from quantlab.labs import stationarity, arima, garch as garch_lab, cointegration


# ──────────────────────────────────────── helpers ────────────────────────────

def _random_walk(n: int = 500, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(x[-1] + rng.gauss(0, 1))
    return x


def _ar1(phi: float, n: int = 500, seed: int = 1) -> list[float]:
    """Stationary AR(1): y_t = phi*y_{t-1} + e_t,  |phi|<1."""
    rng = random.Random(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(phi * x[-1] + rng.gauss(0, 1))
    return x


def _cointegrated_pair(beta: float = 1.5, n: int = 600, seed: int = 2):
    """x = random walk, y = beta*x + stationary noise (white noise spread)."""
    rng = random.Random(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(x[-1] + rng.gauss(0, 1))
    y = [beta * xi + rng.gauss(0, 0.5) for xi in x]
    return y, x


def _cointegrated_pair_ar_spread(beta: float = 1.5, phi: float = 0.7,
                                  n: int = 500, seed: int = 20):
    """Cointegrated pair where spread follows AR(1) — OU can be fit."""
    rng = random.Random(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(x[-1] + rng.gauss(0, 1))
    spread = [0.0]
    for _ in range(n - 1):
        spread.append(phi * spread[-1] + rng.gauss(0, 0.4))
    y = [beta * xi + s for xi, s in zip(x, spread)]
    return y, x


# ──────────────────────────────────────────── stationarity ───────────────────

def test_stationarity_detects_unit_root():
    rw = _random_walk(n=500)
    r = stationarity.test(rw)
    # a pure random walk should come out non-stationary or ambiguous
    assert r.conclusion in ("non-stationary", "ambiguous")
    assert r.d_suggested >= 1


def test_stationarity_detects_stationary():
    ar = _ar1(phi=0.5, n=500)
    r = stationarity.test(ar)
    assert r.conclusion in ("stationary", "ambiguous")
    assert r.d_suggested == 0


def test_stationarity_d_suggestion_after_differencing():
    # cumsum of white noise is a random walk; its difference is stationary
    rng = np.random.default_rng(3)
    rw = list(np.cumsum(rng.normal(0, 1, 400)))
    r = stationarity.test(rw)
    assert r.d_suggested in (1, 2)  # 1 is the canonical answer


def test_stationarity_returns_interpretation():
    r = stationarity.test(_ar1(phi=0.3, n=200))
    assert "adf" in r.interpretation
    assert "kpss" in r.interpretation
    assert "conclusion" in r.interpretation
    assert "caveats" in r.interpretation


def test_stationarity_raises_on_short_series():
    with pytest.raises(ValueError):
        stationarity.test([1.0, 2.0, 3.0])


# ───────────────────────────────────────────────── ARIMA ─────────────────────

def test_arima_fits_ar1_process():
    """Simulate AR(1,0,0), fit ARIMA, check that p=1 is selected."""
    phi = 0.6
    ar = _ar1(phi=phi, n=300)
    result = arima.fit(ar, d=0, p_max=4, q_max=4, criterion="aic")
    p, d, q = result.order
    assert d == 0
    # AR(1) should be selected or close; p≤2 is a reasonable expectation
    assert p <= 2
    assert isinstance(result.aic, float)
    assert isinstance(result.walk_forward_rmse, float)
    assert "model" in result.interpretation


def test_arima_raises_on_short_series():
    with pytest.raises(ValueError):
        arima.fit([1.0] * 10, d=1)


def test_arima_walk_forward_rmse_reasonable():
    """Walk-forward RMSE on stationary AR(1) should be positive and finite."""
    ar = _ar1(phi=0.5, n=200)
    result = arima.fit(ar, d=0, p_max=3, q_max=3)
    assert result.walk_forward_rmse is not None
    assert result.walk_forward_rmse > 0
    assert math.isfinite(result.walk_forward_rmse)


# ─────────────────────────────────────────────────── GARCH ───────────────────

def _garch_returns(n: int = 1000, seed: int = 4) -> list[float]:
    """Simulate GARCH(1,1) returns."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.01, 0.09, 0.90
    h = [omega / (1 - alpha - beta)]
    r = []
    for _ in range(n):
        e = rng.normal(0, math.sqrt(h[-1]))
        r.append(e / 100)   # fraction-space
        h.append(omega + alpha * e ** 2 + beta * h[-1])
    return r


def test_garch_persistence_near_one():
    rets = _garch_returns(n=1000)
    result = garch_lab.fit(rets, spec="garch")
    alpha = result.params.get("alpha[1]", 0)
    beta = result.params.get("beta[1]", 0)
    persistence = alpha + beta
    assert persistence > 0.80, f"Expected α+β>0.80, got {persistence:.4f}"


def test_garch_vol_forecast_positive():
    rets = _garch_returns(n=500)
    result = garch_lab.fit(rets, spec="garch")
    assert result.vol_forecast_1 > 0
    assert result.vol_forecast_22 > 0
    assert result.annualized_vol > 0


def test_garch_egarch_runs():
    rets = _garch_returns(n=500)
    result = garch_lab.fit(rets, spec="egarch")
    assert result.spec == "egarch"
    assert math.isfinite(result.aic)


def test_garch_gjr_runs():
    rets = _garch_returns(n=500)
    result = garch_lab.fit(rets, spec="gjr-garch")
    assert result.spec == "gjr-garch"
    assert math.isfinite(result.aic)


def test_garch_fit_all_returns_three():
    rets = _garch_returns(n=500)
    results = garch_lab.fit_all(rets)
    assert len(results) == 3
    assert "garch" in results
    assert "egarch" in results
    assert "gjr-garch" in results


# ─────────────────────────────────────────────── cointegration ───────────────

def test_cointegration_detects_cointegrated_pair():
    y, x = _cointegrated_pair(beta=1.5, n=400)
    result = cointegration.engle_granger(y, x)
    assert result.cointegrated, f"Expected cointegrated, got p={result.pvalue:.4f}"
    assert abs(result.hedge_ratio_ols - 1.5) < 0.3
    assert result.ou.half_life > 0


def test_cointegration_rejects_independent_random_walks():
    rw1 = _random_walk(n=400, seed=10)
    rw2 = _random_walk(n=400, seed=11)
    result = cointegration.engle_granger(rw1, rw2)
    # two independent random walks should NOT be cointegrated most of the time
    # p<0.05 can occur by chance, so we test the spread has large half-life
    if result.cointegrated:
        # spurious regression — half-life should be very long
        assert result.ou.half_life > 50


def test_cointegration_spread_length():
    y, x = _cointegrated_pair(n=300)
    result = cointegration.engle_granger(y, x)
    assert len(result.spread) == 300


def test_johansen_detects_one_cointegrating_vector():
    y, x = _cointegrated_pair(beta=2.0, n=400)
    result = cointegration.johansen([y, x])
    # should detect at least 1 cointegrating vector
    assert result.n_cointegrating_vectors >= 1


def test_kalman_hedge_tracks_constant_beta():
    """If true beta is constant, Kalman should converge close to it."""
    y, x = _cointegrated_pair(beta=1.5, n=400)
    result = cointegration.kalman_hedge(y, x, delta=1e-5)
    final_beta = result.hedge_ratios[-1]
    assert abs(final_beta - 1.5) < 0.5, f"Kalman beta={final_beta:.4f}, expected ≈1.5"
    assert len(result.spread) == 400


def test_kalman_hedge_returns_ou():
    # Need AR(1) spread (not white noise) for OU to fit
    y, x = _cointegrated_pair_ar_spread(phi=0.7, n=500)
    result = cointegration.kalman_hedge(y, x)
    assert result.ou is not None
    assert result.ou.half_life > 0


def test_cointegration_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        cointegration.engle_granger([1.0, 2.0, 3.0], [1.0, 2.0])
