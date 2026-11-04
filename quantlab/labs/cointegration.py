"""Lab 05 — Cointegration and pairs-trading analytics.

Three tools:
  engle_granger  — for pairs; static OLS hedge ratio + OU fit on spread
  johansen       — for N≥2 series; trace statistic to count cointegrating vectors
  kalman_hedge   — time-varying [α_t, β_t] via Kalman filter; OU on dynamic spread

Kalman state model: θ_t = θ_{t-1} + w_t (random walk),
  w_t ~ N(0, δ/(1−δ)·I).  δ controls adaptation speed.
  Observation: y_t = F_t · θ_t + v_t,  F_t = [1, x_t],  v_t ~ N(0, Ve).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .ou import estimate as ou_estimate, OUFit


@dataclass
class EngleGrangerResult:
    stat: float
    pvalue: float
    hedge_ratio_ols: float
    spread: list[float]
    ou: OUFit
    cointegrated: bool
    interpretation: dict


@dataclass
class JohansenResult:
    n_cointegrating_vectors: int
    trace_stats: list[float]
    critical_5pct: list[float]
    eigenvectors: list[list[float]]
    interpretation: dict


@dataclass
class KalmanHedgeResult:
    hedge_ratios: list[float]
    spread: list[float]
    ou: OUFit | None
    interpretation: dict


# ------------------------------------------------------------------ public API

def engle_granger(y: Sequence[float], x: Sequence[float]) -> EngleGrangerResult:
    """Test cointegration between y and x, fit OU process on OLS spread.

    H₀: no cointegration (spread has unit root).
    Reject H₀ at p<0.05 → cointegrated.
    """
    ya = np.asarray(y, dtype=float)
    xa = np.asarray(x, dtype=float)
    if len(ya) != len(xa):
        raise ValueError("Series must have the same length")

    stat, pvalue, _ = coint(ya, xa)

    # static OLS hedge ratio: y = α + β·x + ε
    X = add_constant(xa)
    ols = OLS(ya, X).fit()
    beta = float(ols.params[1])
    alpha = float(ols.params[0])
    spread = list(ya - beta * xa - alpha)

    try:
        ou = ou_estimate(spread)
    except ValueError as e:
        raise ValueError(f"OU fit на спреде не удался: {e}") from e

    cointegrated = pvalue < 0.05
    return EngleGrangerResult(
        stat=float(stat), pvalue=float(pvalue),
        hedge_ratio_ols=beta, spread=spread, ou=ou,
        cointegrated=cointegrated,
        interpretation=_eg_interpretation(pvalue, beta, ou, cointegrated))


def johansen(
    series: list[Sequence[float]],
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Johansen trace test for N≥2 series.

    Parameters
    ----------
    series     : list of N equally-length time series
    det_order  : -1=no deterministic terms, 0=constant, 1=constant+trend
    k_ar_diff  : number of lagged differences in the VECM
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore

    X = np.column_stack([np.asarray(s, dtype=float) for s in series])
    if X.shape[0] < 20:
        raise ValueError("Нужно минимум 20 наблюдений")

    res = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)
    n = X.shape[1]

    trace_stats = [float(res.lr1[i]) for i in range(n)]
    # cvt columns: 0=10%, 1=5%, 2=1%
    crit_5 = [float(res.cvt[i, 1]) for i in range(n)]
    n_coint = sum(1 for ts, cv in zip(trace_stats, crit_5) if ts > cv)
    eigvecs = [list(map(float, res.evec[:, i])) for i in range(n)]

    return JohansenResult(
        n_cointegrating_vectors=n_coint,
        trace_stats=trace_stats, critical_5pct=crit_5,
        eigenvectors=eigvecs,
        interpretation=_johansen_interpretation(n_coint, n, trace_stats, crit_5))


def kalman_hedge(
    y: Sequence[float],
    x: Sequence[float],
    delta: float = 1e-5,
) -> KalmanHedgeResult:
    """Time-varying hedge ratio [α_t, β_t] via Kalman filter.

    Parameters
    ----------
    delta : state noise intensity; higher = faster adaptation.
            1e-5 typical for monthly data, 1e-4 for daily.
    """
    ya = np.asarray(y, dtype=float)
    xa = np.asarray(x, dtype=float)
    n = len(ya)

    Vw = (delta / (1.0 - delta)) * np.eye(2)      # state evolution covariance
    Ve = float(np.var(ya)) * 1e-3                   # observation noise

    theta = np.zeros(2)   # [alpha, beta]
    P = np.eye(2)

    alphas: list[float] = []
    betas: list[float] = []
    spreads: list[float] = []

    for t in range(n):
        F = np.array([1.0, xa[t]])
        # predict
        P = P + Vw
        # innovation
        inn = ya[t] - float(F @ theta)
        S = float(F @ P @ F) + Ve
        # Kalman gain
        K = (P @ F) / S
        # update
        theta = theta + K * inn
        P = (np.eye(2) - np.outer(K, F)) @ P

        alphas.append(float(theta[0]))
        betas.append(float(theta[1]))
        spreads.append(float(ya[t] - theta[1] * xa[t] - theta[0]))

    ou: OUFit | None = None
    try:
        ou = ou_estimate(spreads)
    except ValueError:
        pass

    return KalmanHedgeResult(
        hedge_ratios=betas, spread=spreads, ou=ou,
        interpretation=_kalman_interpretation(betas, spreads, ou, delta))


# ---------------------------------------------------------- interpretation helpers

def _eg_interpretation(pvalue: float, beta: float,
                        ou: OUFit, cointegrated: bool) -> dict:
    half_life_desc = ""
    if ou.half_life < 5:
        half_life_desc = "быстрый возврат (<5 дней) — высокий оборот стратегии"
    elif ou.half_life < 20:
        half_life_desc = "умеренный возврат (5–20 дней)"
    else:
        half_life_desc = "медленный возврат (>20 дней) — возможны длительные просадки"

    return {
        "coint_test": (
            f"Engle-Granger p={pvalue:.4f}. "
            + ("Коинтеграция подтверждена (p<0.05) ✓" if cointegrated
               else "Коинтеграция НЕ подтверждена (p≥0.05) — спред не mean-reverts.")),
        "hedge_ratio": (
            f"OLS hedge ratio β={beta:.4f}: "
            f"на 1 ед. X берём {beta:.4f} ед. Y в противоположную сторону. "
            "Нестабилен во времени — используй Kalman для торговли."),
        "mean_reversion": (
            f"Half-life спреда ≈ {ou.half_life:.1f} шагов: {half_life_desc}. "
            f"θ={ou.theta:.4f}, μ={ou.mu:.4f}, σ={ou.sigma:.4f}."),
        "trading_signal": (
            "Сигнал: long spread при z-score < −1, short при z-score > +1, выход при z=0."
            if cointegrated else
            "Спред не стационарен — торговать на возврат к среднему не рекомендуется."),
        "caveats": [
            "Статический hedge ratio нестабилен; проверь через kalman_hedge",
            "Коинтеграция на in-sample ≠ коинтеграция out-of-sample",
            "Проверь на подпериодах (первая / вторая половина истории)",
        ],
    }


def _johansen_interpretation(n_coint: int, n: int,
                               trace_stats: list[float],
                               crit_5: list[float]) -> dict:
    return {
        "result": (
            f"Johansen trace test: {n_coint} коинтегрирующих вектора из {n} рядов. "
            + ("Нет долгосрочных ограничений — система расходится." if n_coint == 0
               else f"{n_coint} долгосрочных равновесных соотношения в системе.")),
        "interpretation": (
            "Полная коинтеграция (n−1 векторов) означает, что вся система связана. "
            "Один вектор — один спред для торговли." if n_coint > 0 else
            "Рассмотри попарный Engle-Granger или более длинный ряд."),
        "caveats": [
            "Johansen чувствителен к выбору k_ar_diff: подбирай по AIC/BIC на VAR",
            "det_order=0 предполагает нет тренда в коинтегрирующем соотношении — проверь",
            "на коротких рядах (<200 наблюдений) мощность теста мала",
        ],
    }


def _kalman_interpretation(betas: list[float], spreads: list[float],
                             ou: OUFit | None, delta: float) -> dict:
    beta_mean = float(np.mean(betas))
    beta_std = float(np.std(betas))
    beta_stable = beta_std < 0.05

    body: dict = {
        "hedge_ratio": (
            f"Kalman β: среднее={beta_mean:.4f}, σ={beta_std:.4f}. "
            + ("Стабильный hedge ratio — статический OLS достаточен." if beta_stable
               else "Hedge ratio значимо меняется — статический OLS некорректен, Kalman необходим.")),
        "delta": (
            f"δ={delta}: скорость адаптации. "
            "Увеличь до 1e-4 если спред нестабилен на дневных данных."),
    }

    if ou is not None:
        body["spread"] = (
            f"OU на Kalman-спреде: θ={ou.theta:.4f}, "
            f"half-life≈{ou.half_life:.1f} шагов, σ={ou.sigma:.4f}.")
        body["trading_signal"] = (
            f"Сигнал по z-score спреда: "
            "long spread при z<−1, short при z>+1, выход при z=0. "
            f"Ожидаемая продолжительность сделки ≈ {ou.half_life:.0f} шагов.")
    else:
        body["spread"] = "OU fit на спреде не удался — спред не mean-reverts."
        body["trading_signal"] = "Торговля на основе этого спреда не рекомендуется."

    body["caveats"] = [
        "Kalman beta стабилизируется только после первых ~50 наблюдений",
        "Ve (observation noise) = var(y)*1e-3 — подбирай вручную при нестабильных результатах",
        "Проверь коинтеграцию через engle_granger перед торговлей",
    ]
    return body
