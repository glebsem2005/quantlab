"""Lab 06 — Descriptive statistics and ACF/PACF.

Returns summary: annualised metrics (return, vol, Sharpe, drawdown, skew, kurtosis).
ACF/PACF: pre-computed arrays ready for terminal bar rendering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf


# ─────────────────────────────────────────── returns summary ─────────────────

@dataclass
class ReturnsSummary:
    n: int
    total_return: float          # total log-return (fraction)
    annual_return: float         # annualised geometric return
    daily_vol: float             # daily std of log-returns
    annual_vol: float            # daily_vol * sqrt(252)
    sharpe: float                # ann_return / ann_vol (rf=0)
    sortino: float               # ann_return / downside_vol
    max_drawdown: float          # peak-to-trough as fraction of peak price
    skewness: float
    kurtosis: float              # excess (0 = normal)
    best_day: float
    worst_day: float
    positive_days_pct: float
    interpretation: dict


def returns_summary(closes: Sequence[float], periods_per_year: int = 252) -> ReturnsSummary:
    """Compute return statistics from a price series.

    Parameters
    ----------
    closes             : price series (list or array)
    periods_per_year   : 252 for daily, 52 for weekly, 12 for monthly
    """
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    if n < 5:
        raise ValueError("Нужно минимум 5 цен")

    rets = np.diff(np.log(arr))   # log-returns

    total_ret = float(np.log(arr[-1] / arr[0]))
    ann_ret = float(total_ret / (len(rets) / periods_per_year))
    daily_vol = float(np.std(rets, ddof=1))
    ann_vol = daily_vol * math.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = float(np.std(rets[rets < 0], ddof=1)) if np.any(rets < 0) else 1e-9
    sortino = ann_ret / (downside * math.sqrt(periods_per_year))

    # max drawdown from prices
    peak = arr[0]
    max_dd = 0.0
    for p in arr:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd

    # higher moments
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1))
    skew = float(np.mean(((rets - mean_r) / std_r) ** 3)) if std_r > 0 else 0.0
    kurt = float(np.mean(((rets - mean_r) / std_r) ** 4)) - 3.0 if std_r > 0 else 0.0

    pos_pct = float(np.mean(rets > 0)) * 100

    return ReturnsSummary(
        n=n, total_return=total_ret, annual_return=ann_ret,
        daily_vol=daily_vol, annual_vol=ann_vol,
        sharpe=sharpe, sortino=sortino,
        max_drawdown=max_dd, skewness=skew, kurtosis=kurt,
        best_day=float(np.max(rets)), worst_day=float(np.min(rets)),
        positive_days_pct=pos_pct,
        interpretation=_returns_interpretation(
            ann_ret, ann_vol, sharpe, max_dd, skew, kurt))


def _returns_interpretation(ann_ret: float, ann_vol: float, sharpe: float,
                              max_dd: float, skew: float, kurt: float) -> dict:
    sharpe_desc = ("отличный" if sharpe > 2 else "хороший" if sharpe > 1
                   else "посредственный" if sharpe > 0 else "отрицательный")
    return {
        "return_vol": (
            f"Годовая доходность {ann_ret*100:.1f}%, волатильность {ann_vol*100:.1f}%. "
            f"Sharpe={sharpe:.2f} ({sharpe_desc})."),
        "drawdown": (
            f"Максимальная просадка {max_dd*100:.1f}%. "
            + ("Высокий риск просадки." if max_dd > 0.3 else
               "Умеренная просадка." if max_dd > 0.15 else "Низкая просадка.")),
        "distribution": (
            f"Асимметрия={skew:.2f}, эксцесс={kurt:.2f}. "
            + ("Левый хвост тяжелее нормального — риск больших потерь, чем ожидается."
               if skew < -0.5 else
               "Правый хвост тяжелее — редкие большие прибыли." if skew > 0.5
               else "Распределение близко к симметричному.")
            + (" Жирные хвосты (эксцесс>1) — нормальное допущение недооценивает риск."
               if kurt > 1 else "")),
        "caveats": [
            "rf=0 в Sharpe; для российского рынка используй ключевую ставку",
            "аннуализация предполагает constant risk — на практике vol кластеризуется",
            "max drawdown зависит от частоты данных (дневной vs. внутридневной)",
        ],
    }


# ──────────────────────────────────────────────────── ACF/PACF ───────────────

@dataclass
class ACFResult:
    lags: list[int]
    acf_values: list[float]
    pacf_values: list[float]
    ci_95: float       # ±1.96 / sqrt(n); approximate for ACF
    n: int
    interpretation: dict


def acf_pacf(x: Sequence[float], lags: int = 20) -> ACFResult:
    """Compute ACF and PACF for series x.

    Parameters
    ----------
    x    : time series (prices, returns, spread — caller decides)
    lags : number of lags to compute
    """
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < lags + 5:
        raise ValueError(f"Нужно минимум {lags + 5} наблюдений для {lags} лагов")

    acf_vals = sm_acf(arr, nlags=lags, fft=True, alpha=None)
    pacf_vals = sm_pacf(arr, nlags=lags, method="ywm", alpha=None)

    # take lags 1..lags (drop lag-0 = 1.0)
    lag_list = list(range(1, lags + 1))
    acf_list = [float(acf_vals[i]) for i in lag_list]
    pacf_list = [float(pacf_vals[i]) for i in lag_list]
    ci = 1.96 / math.sqrt(n)

    # count significant lags
    sig_acf = sum(1 for v in acf_list if abs(v) > ci)
    sig_pacf = sum(1 for v in pacf_list if abs(v) > ci)

    return ACFResult(
        lags=lag_list, acf_values=acf_list, pacf_values=pacf_list,
        ci_95=ci, n=n,
        interpretation=_acf_interpretation(acf_list, pacf_list, ci, sig_acf, sig_pacf))


def _acf_interpretation(acf: list[float], pacf: list[float],
                         ci: float, sig_acf: int, sig_pacf: int) -> dict:
    body: dict = {
        "summary": (
            f"{sig_acf} значимых лагов ACF и {sig_pacf} лагов PACF (за пределами 95% CI ±{ci:.3f})."),
    }
    # heuristic pattern recognition
    if sig_acf == 0:
        body["pattern"] = "Нет значимой автокорреляции → ряд близок к белому шуму."
    elif sig_acf >= 3 and all(v > 0 for v in acf[:3]):
        body["pattern"] = ("Постепенно убывающая положительная ACF → AR-процесс или "
                           "нестационарный ряд. Проверь стационарность (qlab lab stat).")
    elif sig_pacf <= 2 and sig_acf >= 2:
        body["pattern"] = (f"ACF убывает медленно, PACF обрывается на лаге {sig_pacf} → "
                           f"похоже на AR({sig_pacf}).")
    elif sig_acf <= 2 and sig_pacf >= 2:
        body["pattern"] = (f"ACF обрывается на лаге {sig_acf}, PACF убывает → "
                           f"похоже на MA({sig_acf}).")
    else:
        body["pattern"] = "Смешанная структура → возможно ARMA. Используй qlab lab arima."
    body["caveats"] = [
        "ACF на ценах практически всегда значима (unit root) — запускай на доходностях",
        "PACF-метод Yule-Walker (ywm) устойчив, но менее эффективен на малых выборках",
    ]
    return body
