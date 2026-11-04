"""Lab 02 — Stationarity tests: ADF, KPSS, Phillips-Perron.

All three tests run together because they often disagree, and understanding
when they disagree is the key learning objective.

H₀ conventions (the critical distinction):
  ADF / PP  H₀: unit root present (non-stationary) — reject → evidence of stationarity
  KPSS      H₀: stationary                         — reject → evidence of non-stationarity

Ambiguous zone: ADF/PP fail to reject AND KPSS also fails to reject.
Both hypotheses are consistent with the data — near-unit-root or too-short series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class SingleTestResult:
    stat: float
    pvalue: float
    critical: dict[str, float]   # {"1%": ..., "5%": ..., "10%": ...}
    reject_h0: bool               # at 5% level
    h0_label: str


@dataclass
class StationarityResult:
    adf: SingleTestResult
    kpss: SingleTestResult
    pp: SingleTestResult
    n: int
    d_suggested: int              # 0=I(0), 1=I(1), 2=I(2)
    conclusion: str               # "stationary" | "non-stationary" | "ambiguous"
    interpretation: dict


def _pp_test(arr: np.ndarray, regression: str) -> SingleTestResult:
    try:
        from arch.unitroot import PhillipsPerron
        r = PhillipsPerron(arr, trend=("c" if regression == "c" else "ct"))
        stat = float(r.stat)
        pvalue = float(r.pvalue)
        crit = {k: float(v) for k, v in r.critical_values.items()}
        reject = pvalue < 0.05
    except Exception:
        # arch not available or old version — fall back to a second ADF
        res = adfuller(arr, autolag="AIC", regression="ct")
        stat, pvalue = float(res[0]), float(res[1])
        crit = {str(k): float(v) for k, v in res[4].items()}
        reject = pvalue < 0.05
    return SingleTestResult(
        stat=stat, pvalue=pvalue, critical=crit, reject_h0=reject,
        h0_label="единичный корень (нестационарен)")


def _conclude(adf_reject: bool, kpss_reject: bool) -> str:
    if adf_reject and not kpss_reject:
        return "stationary"
    if not adf_reject and kpss_reject:
        return "non-stationary"
    return "ambiguous"


def _is_stationary(arr: np.ndarray, regression: str) -> bool:
    adf_res = adfuller(arr, autolag="AIC", regression=regression)
    kpss_regression = "ct" if regression == "ct" else "c"
    kpss_res = kpss(arr, regression=kpss_regression, nlags="auto")
    return _conclude(adf_res[1] < 0.05, kpss_res[1] < 0.05) == "stationary"


def test(x: Sequence[float], *, regression: str = "c") -> StationarityResult:
    """Run ADF, KPSS, PP on series x.

    Parameters
    ----------
    x          : price or return series (list or 1-D array)
    regression : 'c' (intercept only) | 'ct' (trend + intercept) | 'n' (none)
    """
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    if n < 20:
        raise ValueError("Нужно минимум 20 наблюдений")

    kpss_regression = "ct" if regression == "ct" else "c"

    # ADF
    adf_res = adfuller(arr, autolag="AIC", regression=regression)
    adf = SingleTestResult(
        stat=float(adf_res[0]), pvalue=float(adf_res[1]),
        critical={str(k): float(v) for k, v in adf_res[4].items()},
        reject_h0=adf_res[1] < 0.05,
        h0_label="единичный корень (нестационарен)")

    # KPSS — opposite H₀
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_res = kpss(arr, regression=kpss_regression, nlags="auto")
    kpss_ = SingleTestResult(
        stat=float(kpss_res[0]), pvalue=float(kpss_res[1]),
        critical={str(k): float(v) for k, v in kpss_res[3].items()},
        reject_h0=kpss_res[1] < 0.05,
        h0_label="стационарность")

    # PP
    pp = _pp_test(arr, regression)

    conclusion = _conclude(adf.reject_h0, kpss_.reject_h0)

    # determine differencing order
    if conclusion == "stationary":
        d = 0
    else:
        diff1 = np.diff(arr)
        if _is_stationary(diff1, regression):
            d = 1
        else:
            d = 2

    return StationarityResult(
        adf=adf, kpss=kpss_, pp=pp, n=n,
        d_suggested=d, conclusion=conclusion,
        interpretation=_interpretation(adf, kpss_, pp, conclusion, d))


def _interpretation(adf: SingleTestResult, kpss_r: SingleTestResult,
                    pp: SingleTestResult, conclusion: str, d: int) -> dict:
    def _verdict(r: SingleTestResult) -> str:
        return "отверг H₀" if r.reject_h0 else "не отверг H₀"

    body: dict = {
        "adf": (f"ADF {_verdict(adf)} (p={adf.pvalue:.3f}). "
                "H₀ — наличие единичного корня. "
                "Отвержение → свидетельство стационарности."),
        "kpss": (f"KPSS {_verdict(kpss_r)} (p≈{kpss_r.pvalue:.3f}). "
                 "H₀ — стационарность (НАОБОРОТ от ADF!). "
                 "Отвержение → свидетельство нестационарности."),
        "pp": (f"PP {_verdict(pp)} (p={pp.pvalue:.3f}). "
               "Аналог ADF, устойчив к гетероскедастичным ошибкам."),
        "caveats": [
            "ADF маломощен на коротких рядах (<100 наблюдений)",
            "все три теста дают лишь асимптотически корректные p-values",
            "при φ≈0.97 тесты часто расходятся — это нормально, нужен более длинный ряд",
        ],
    }

    if conclusion == "stationary":
        body["conclusion"] = "Ряд, вероятно, стационарен I(0). d=0: используй ARIMA(p,0,q)."
    elif conclusion == "non-stationary":
        body["conclusion"] = (f"Ряд нестационарен I({d}). "
                               f"Рекомендуется d={d} для ARIMA или взять лог-доходности.")
    else:
        body["conclusion"] = (
            "Противоречивые результаты (ambiguous zone). Ряд может быть near-unit-root. "
            "Попробуй: (1) более длинный ряд, (2) regression='ct' с трендом, "
            "(3) работать с лог-доходностями вместо цен.")

    return body
