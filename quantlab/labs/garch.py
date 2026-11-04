"""Lab 04 — GARCH-family volatility models.

Three specifications via the `arch` package:
  garch     — GARCH(p,q): symmetric, the industry workhorse
  egarch    — EGARCH(p,q): log-variance, no sign constraints, captures asymmetry
  gjr-garch — GJR-GARCH(p,o,q): explicit leverage term γ for negative shocks

Input: daily log-returns (as fractions, e.g. 0.012 for +1.2%).
Internally multiplied by 100 for arch's numerical stability; vol forecasts
are divided back to fraction space before returning.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

GARCHSpec = str  # "garch" | "egarch" | "gjr-garch"


@dataclass
class GARCHFit:
    spec: str
    p: int
    q: int
    params: dict                  # parameter names → values (in %-space for omega/σ²)
    aic: float
    bic: float
    log_likelihood: float
    vol_forecast_1: float         # 1-step-ahead conditional σ (as fraction, not %)
    vol_forecast_5: float
    vol_forecast_22: float        # ≈ 1 trading month
    annualized_vol: float         # vol_1 * sqrt(252)
    has_leverage: bool            # GJR: γ>0 and significant at 5%
    interpretation: dict


def fit(
    returns: Sequence[float],
    spec: GARCHSpec = "garch",
    p: int = 1,
    q: int = 1,
) -> GARCHFit:
    """Fit GARCH-family model to daily log-returns.

    Parameters
    ----------
    returns : daily log-returns as fractions (not %)
    spec    : 'garch' | 'egarch' | 'gjr-garch'
    p, q    : ARCH / GARCH lag orders
    """
    from arch import arch_model  # type: ignore

    arr = np.asarray(returns, dtype=float) * 100  # %-space

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec == "gjr-garch":
            am = arch_model(arr, vol="GARCH", p=p, o=1, q=q, dist="Normal")
        elif spec == "egarch":
            am = arch_model(arr, vol="EGARCH", p=p, q=q, dist="Normal")
        else:
            am = arch_model(arr, vol="GARCH", p=p, q=q, dist="Normal")

        res = am.fit(disp="off", show_warning=False)

    # volatility forecasts (EGARCH only supports analytic for h=1; use simulation for h>1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc = res.forecast(horizon=22, reindex=False)
        except ValueError:
            fc = res.forecast(horizon=22, reindex=False, method="simulation", simulations=200)

    var_fc = fc.variance.iloc[-1]   # conditional variances in %-space
    # convert back to fraction space: σ% / 100
    vol_1 = float(np.sqrt(var_fc.iloc[0])) / 100
    vol_5 = float(np.sqrt(var_fc.iloc[4])) / 100
    vol_22 = float(np.sqrt(var_fc.iloc[21])) / 100
    ann_vol = vol_1 * np.sqrt(252)

    params = {k: float(v) for k, v in res.params.items()}
    pvalues = {k: float(v) for k, v in res.pvalues.items()}

    has_leverage = False
    if spec == "gjr-garch":
        gamma_key = next((k for k in params if k.startswith("gamma")), None)
        if gamma_key and params[gamma_key] > 0 and pvalues.get(gamma_key, 1) < 0.05:
            has_leverage = True

    return GARCHFit(
        spec=spec, p=p, q=q, params=params,
        aic=float(res.aic), bic=float(res.bic),
        log_likelihood=float(res.loglikelihood),
        vol_forecast_1=vol_1, vol_forecast_5=vol_5, vol_forecast_22=vol_22,
        annualized_vol=ann_vol, has_leverage=has_leverage,
        interpretation=_interpretation(spec, params, pvalues, vol_1, vol_22, ann_vol,
                                        has_leverage))


def fit_all(returns: Sequence[float]) -> dict[str, GARCHFit]:
    """Fit all three specifications; useful for AIC/BIC comparison."""
    results: dict[str, GARCHFit] = {}
    for spec in ("garch", "egarch", "gjr-garch"):
        try:
            results[spec] = fit(returns, spec=spec)
        except Exception:
            pass
    return results


def _interpretation(spec: str, params: dict, pvalues: dict,
                     vol_1: float, vol_22: float, ann_vol: float,
                     has_leverage: bool) -> dict:
    body: dict = {
        "model": f"Модель: {spec.upper()}(1,1). Входные данные: лог-доходности.",
        "volatility": (
            f"Прогноз волатильности: 1д={vol_1*100:.3f}%, "
            f"22д={vol_22*100:.3f}%, "
            f"годовая≈{ann_vol*100:.1f}%."),
    }

    alpha_key = next((k for k in params if k.startswith("alpha")), None)
    beta_key = next((k for k in params if k.startswith("beta")), None)
    if alpha_key and beta_key:
        alpha = params[alpha_key]
        beta = params[beta_key]
        persistence = alpha + beta
        body["persistence"] = (
            f"α+β={persistence:.4f} (устойчивость волатильности). "
            + ("Близко к 1 → волатильность долго помнит шоки (IGARCH-like), "
               "типично для акций." if persistence > 0.95 else
               "Умеренная устойчивость → волатильность быстро возвращается к среднему."))

    if has_leverage:
        body["leverage"] = (
            "Leverage-эффект подтверждён (γ>0, p<0.05): "
            "негативные шоки увеличивают волатильность сильнее, чем позитивные — "
            "классическое свойство акций.")
    elif spec == "gjr-garch":
        body["leverage"] = "Leverage-эффект не подтверждён статистически."

    target_daily = 0.01   # 1% daily target for example
    if vol_1 > 0:
        sizing_weight = min(1.0, target_daily / vol_1)
        body["volatility_targeting"] = (
            f"Volatility targeting при цели 1%/день: вес позиции ≈ {sizing_weight:.2f}. "
            "Масштабируй обратно пропорционально σ_t для constant-risk sizing.")

    body["caveats"] = [
        "GARCH предполагает стационарность доходностей (проверь ADF/KPSS)",
        "нормальное распределение ошибок часто недооценивает fat tails; попробуй dist='t'",
        f"spec={spec}: " + (
            "GARCH симметричен к знаку шока; используй GJR для проверки асимметрии"
            if spec == "garch" else
            "EGARCH не требует неотрицательных параметров, лучше для высокой волатильности"
            if spec == "egarch" else
            "GJR явно моделирует leverage; сравни AIC с GARCH"),
    ]
    return body
