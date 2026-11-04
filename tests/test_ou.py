"""Simulate with known parameters, estimate, recover."""
from quantlab.labs import ou


def test_ou_recovers_parameters():
    true_theta, true_mu, true_sigma = 0.05, 100.0, 2.0
    x = ou.simulate(true_theta, true_mu, true_sigma, x0=100, n=20000, seed=7)
    fit = ou.estimate(x)
    assert abs(fit.theta - true_theta) < 0.01
    assert abs(fit.mu - true_mu) < 2.0
    assert abs(fit.sigma - true_sigma) < 0.1
    assert "half_life" in fit.interpretation


def test_ou_rejects_random_walk():
    import random, pytest, math
    rng = random.Random(1)
    x = [0.0]
    for _ in range(2000):
        x.append(x[-1] + rng.gauss(0, 1))
    # a pure random walk has phi ~ 1; estimator may or may not raise,
    # but if it fits, theta must be tiny and half-life enormous
    try:
        fit = ou.estimate(x)
        assert fit.half_life > 200
    except ValueError:
        pass
