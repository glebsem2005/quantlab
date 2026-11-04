"""Lab 01 — Ornstein–Uhlenbeck process.

    dX_t = θ(μ − X_t)dt + σ dW_t

Exact discretization over step Δ is an AR(1):
    X_{t+Δ} = c + φ X_t + ε,   φ = e^{−θΔ},  c = μ(1−φ),
    Var(ε) = σ²(1−φ²)/(2θ)

so MLE for (θ, μ, σ) reduces to OLS on the AR(1) plus residual variance.
This is the platform's signature exercise pattern: `simulate` with known
parameters, `estimate`, compare — feel estimator variance with your hands.

Every estimate returns an `interpretation` dict — the explanation-mode
contract from the design doc, here in embryonic form.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class OUFit:
    theta: float
    mu: float
    sigma: float
    half_life: float          # in units of dt (e.g. days if dt=1)
    phi: float                # AR(1) coefficient
    n: int
    interpretation: dict


def estimate(x: list[float], dt: float = 1.0) -> OUFit:
    n = len(x) - 1
    if n < 10:
        raise ValueError("need at least 11 observations")
    xs, ys = x[:-1], x[1:]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    phi = sxy / sxx
    c = my - phi * mx

    if not (0 < phi < 1):
        raise ValueError(
            f"AR(1) coefficient φ={phi:.4f} outside (0,1): the series does not "
            "look mean-reverting at this frequency (unit root or oscillation). "
            "Test stationarity before fitting OU.")

    resid = [b - (c + phi * a) for a, b in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)

    theta = -math.log(phi) / dt
    mu = c / (1 - phi)
    sigma = math.sqrt(s2 * 2 * theta / (1 - phi ** 2))
    half_life = math.log(2) / theta

    interpretation = {
        "theta": f"скорость возврата к среднему; θ={theta:.4f} в единицах 1/dt",
        "mu": f"долгосрочный уровень μ={mu:.4f}: процесс колеблется вокруг него",
        "half_life": (
            f"период полураспада отклонения ≈ {half_life:.1f} шагов: за это время "
            "ожидаемое отклонение от μ сокращается вдвое. Большое значение — "
            "слабая mean-reversion; стратегии на возврате будут медленными."),
        "sigma": f"мгновенная волатильность σ={sigma:.4f}",
        "caveats": [
            "OLS-оценка θ смещена вверх в малых выборках (Phillips–Yu); "
            "проверь на симуляции с известным θ",
            "линейный дрейф — предположение; для ставок сравни с CIS/CIR",
            "если φ близко к 1, доверительный интервал θ почти не отличим от 0",
        ],
    }
    return OUFit(theta=theta, mu=mu, sigma=sigma, half_life=half_life,
                 phi=phi, n=n, interpretation=interpretation)


def simulate(theta: float, mu: float, sigma: float, x0: float,
             n: int, dt: float = 1.0, seed: int | None = None) -> list[float]:
    """Exact-transition simulation (no Euler bias)."""
    rng = random.Random(seed)
    phi = math.exp(-theta * dt)
    sd = sigma * math.sqrt((1 - phi ** 2) / (2 * theta))
    x = [float(x0)]
    for _ in range(n):
        x.append(mu * (1 - phi) + phi * x[-1] + rng.gauss(0, sd))
    return x
