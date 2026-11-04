"""Lab 07 — Markov regime-switching model.

Two or three regimes (e.g., bull/bear, high-vol/low-vol) via
statsmodels MarkovRegression with switching variance.
Each regime has its own mean return and variance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RegimeResult:
    n_regimes: int
    regime_means: list[float]        # mean return per regime (fraction)
    regime_vols: list[float]         # std dev per regime (fraction)
    smoothed_probs: list[list[float]]  # [t][regime] — probability of being in regime
    current_probs: list[float]       # latest smoothed prob per regime
    expected_durations: list[float]  # average time spent in each regime
    aic: float
    bic: float
    interpretation: dict


def markov_switching(
    returns: Sequence[float],
    n_regimes: int = 2,
    k_ar_lags: int = 0,
) -> RegimeResult:
    """Fit Markov regime-switching model to return series.

    Parameters
    ----------
    returns   : daily log-returns
    n_regimes : number of regimes (2 = bull/bear, 3 = bull/neutral/bear)
    k_ar_lags : AR lags within each regime (0 = switching mean/var only)
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression  # type: ignore

    arr = np.asarray(returns, dtype=float) * 100   # %-space for stability
    if len(arr) < 50:
        raise ValueError("Нужно минимум 50 наблюдений для режимного анализа")

    mod = MarkovRegression(
        arr, k_regimes=n_regimes, trend="c", switching_variance=True)
    res = mod.fit(search_reps=20, disp=False)

    # extract regime parameters (back to fraction space)
    means = [float(res.params[f"const[{r}]"]) / 100 for r in range(n_regimes)]
    # variance params: sigma2[r] in %-squared space → std in fraction space
    vols = []
    for r in range(n_regimes):
        key = f"sigma2[{r}]"
        if key in res.params.index:
            vols.append(float(np.sqrt(res.params[key])) / 100)
        else:
            vols.append(float("nan"))

    smoothed = [[float(res.smoothed_marginal_probabilities[t, r])
                  for r in range(n_regimes)]
                 for t in range(len(arr))]
    current = [float(res.smoothed_marginal_probabilities[-1, r])
               for r in range(n_regimes)]

    # expected duration: 1/(1 - p_ii)
    trans = res.regime_transition
    durations = []
    for r in range(n_regimes):
        p_stay = float(trans[r, r, 0])  # P(stay in regime r)
        durations.append(1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf"))

    # sort regimes by mean return for labelling
    order = sorted(range(n_regimes), key=lambda r: means[r])

    return RegimeResult(
        n_regimes=n_regimes,
        regime_means=[means[r] for r in order],
        regime_vols=[vols[r] for r in order],
        smoothed_probs=[[p[r] for r in order] for p in smoothed],
        current_probs=[current[r] for r in order],
        expected_durations=[durations[r] for r in order],
        aic=float(res.aic), bic=float(res.bic),
        interpretation=_interpretation(
            n_regimes, [means[r] for r in order],
            [vols[r] for r in order],
            [durations[r] for r in order],
            [current[r] for r in order]))


def _regime_label(idx: int, n: int, mean: float, vol: float) -> str:
    if n == 2:
        return "медвежий" if idx == 0 else "бычий"
    labels = ["медвежий", "нейтральный", "бычий"]
    return labels[idx] if idx < len(labels) else f"режим {idx}"


def _interpretation(n: int, means: list[float], vols: list[float],
                     durations: list[float], current: list[float]) -> dict:
    lines = []
    for i in range(n):
        label = _regime_label(i, n, means[i], vols[i])
        lines.append(
            f"  Режим {i} ({label}): μ={means[i]*100:+.2f}%/день, "
            f"σ={vols[i]*100:.2f}%/день, "
            f"ожидаемая длительность ≈ {durations[i]:.0f} дней.")

    curr_regime = int(np.argmax(current))
    curr_label = _regime_label(curr_regime, n, means[curr_regime], vols[curr_regime])

    return {
        "regimes": "\n".join(lines),
        "current": (
            f"Текущий режим (по последнему наблюдению): "
            f"режим {curr_regime} ({curr_label}), "
            f"вероятность {current[curr_regime]*100:.1f}%."),
        "trading_use": (
            "Практическое применение: масштабируй размер позиции на вероятность "
            "бычьего режима. При высокой неопределённости (вероятности близки к 0.5) — "
            "уменьшай риск."),
        "caveats": [
            "Марковская модель предполагает фиксированные параметры внутри режима",
            "не торгуй сигнал напрямую — режимы определяются с запозданием",
            "search_reps=20 может не достичь глобального оптимума; перезапусти несколько раз",
        ],
    }
