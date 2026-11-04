"""Lab 03 — ARIMA grid search with walk-forward validation.

Grid search (p, d, q) by AIC/BIC on the in-sample window,
then walk-forward RMSE on the held-out tail to detect overfitting.
Walk-forward uses .apply(refit=False) + .append(refit=False) so the
originally-estimated parameters are used throughout — fast and honest.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


@dataclass
class ARIMAFit:
    order: tuple[int, int, int]
    aic: float
    bic: float
    log_likelihood: float
    params: dict                  # {param_name: value}
    ljung_box_p: float            # H0: no residual autocorrelation
    arch_lm_p: float              # H0: no ARCH effects in residuals
    walk_forward_rmse: float | None
    n_insample: int
    n_outsample: int
    interpretation: dict


def fit(
    x: Sequence[float],
    d: int = 1,
    p_max: int = 5,
    q_max: int = 5,
    criterion: str = "aic",
    wf_test_size: int | None = None,
) -> ARIMAFit:
    """Grid search ARIMA(p, d, q) and evaluate walk-forward out-of-sample.

    Parameters
    ----------
    x            : raw series (log prices or log returns depending on d)
    d            : differencing order from stationarity test
    p_max, q_max : search bounds (inclusive)
    criterion    : 'aic' or 'bic'
    wf_test_size : hold-out size; default = max(12, 20% of series)
    """
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < 20:
        raise ValueError("Нужно минимум 20 наблюдений")

    wf_size = wf_test_size if wf_test_size is not None else max(12, int(0.2 * n))
    wf_size = min(wf_size, n - 10)  # need at least 10 obs to fit
    n_insample = n - wf_size
    n_outsample = wf_size
    train = arr[:n_insample]

    best_ic = np.inf
    best_order = (1, d, 1)
    best_model = None

    for p, q in product(range(p_max + 1), range(q_max + 1)):
        if p + q == 0 and d == 0:
            continue   # constant-only model is degenerate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(train, order=(p, d, q)).fit()
            ic = m.aic if criterion == "aic" else m.bic
            if ic < best_ic:
                best_ic = ic
                best_order = (p, d, q)
                best_model = m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("ARIMA: ни одна модель не сошлась, попробуй меньший p_max/q_max")

    # residual diagnostics on in-sample fit
    resid = best_model.resid
    lb_df = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_p = float(lb_df["lb_pvalue"].iloc[0])

    try:
        _, arch_p, _, _ = het_arch(resid, nlags=5)
        arch_p = float(arch_p)
    except Exception:
        arch_p = float("nan")

    # walk-forward RMSE: fixed params from in-sample, expanding window
    wf_rmse = _walk_forward(best_model, arr, n_insample, n_outsample, best_order)

    # statsmodels ARIMA returns params as numpy array; names via .param_names
    _names = (list(best_model.param_names)
              if hasattr(best_model, "param_names")
              else [f"param_{i}" for i in range(len(np.atleast_1d(best_model.params)))])
    params = {k: float(v) for k, v in zip(_names, np.atleast_1d(best_model.params))}
    return ARIMAFit(
        order=best_order,
        aic=float(best_model.aic), bic=float(best_model.bic),
        log_likelihood=float(best_model.llf),
        params=params, ljung_box_p=lb_p, arch_lm_p=arch_p,
        walk_forward_rmse=wf_rmse,
        n_insample=n_insample, n_outsample=n_outsample,
        interpretation=_interpretation(best_order, best_model.aic, best_model.bic,
                                        lb_p, arch_p, wf_rmse))


def _walk_forward(fitted, arr: np.ndarray, n_insample: int,
                   n_outsample: int, order: tuple) -> float | None:
    """Expanding-window walk-forward with fixed parameters (no refit)."""
    errors: list[float] = []
    try:
        # apply the fitted parameters to the full in-sample window
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wf_model = fitted.apply(arr[:n_insample], refit=False)
        for i in range(n_outsample):
            fc = float(np.asarray(wf_model.forecast(1)).flat[0])
            actual = float(arr[n_insample + i])
            errors.append((fc - actual) ** 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wf_model = wf_model.append(arr[n_insample + i:n_insample + i + 1],
                                            refit=False)
    except Exception:
        return None
    return float(np.sqrt(np.mean(errors))) if errors else None


def _interpretation(order: tuple, aic: float, bic: float,
                     lb_p: float, arch_p: float,
                     wf_rmse: float | None) -> dict:
    p, d, q = order
    lb_ok = lb_p >= 0.05
    arch_flag = (not np.isnan(arch_p)) and arch_p < 0.05

    body: dict = {
        "model": (f"Лучшая модель: ARIMA({p},{d},{q}). "
                  f"AIC={aic:.1f}, BIC={bic:.1f}."),
        "residuals": (
            "Остатки " + (
                "не содержат значимой автокорреляции (Ljung-Box p={:.3f}) ✓".format(lb_p)
                if lb_ok else
                "содержат автокорреляцию (Ljung-Box p={:.3f}) — "
                "рассмотри бо́льший p или q".format(lb_p))),
        "arch_effect": (
            "В остатках " + (
                "есть ARCH-эффекты (гетероскедастичность, p={:.3f}) — "
                "рассмотри GARCH поверх ARIMA".format(arch_p)
                if arch_flag else
                "нет значимых ARCH-эффектов ✓")),
    }

    if wf_rmse is not None:
        body["walk_forward"] = (
            f"Walk-forward RMSE={wf_rmse:.5f} (out-of-sample, фиксированные параметры). "
            "Сравни с in-sample σ остатков: большое расхождение → переобучение.")
    else:
        body["walk_forward"] = "Walk-forward не рассчитан (слишком мало данных или ошибка)."

    body["caveats"] = [
        f"d={d} взято из аргумента — убедись, что совпадает с тестом стационарности",
        "AIC оптимизируется in-sample; сравни walk-forward RMSE для оценки переобучения",
        "ARIMA предполагает линейную условную среднюю; для fat-tails + ARCH используй GARCH",
    ]
    return body
