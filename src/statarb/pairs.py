"""
Pairs-trading statistics: OLS hedge ratios, the Augmented Dickey-Fuller
test, the Engle-Granger two-step cointegration test, Ornstein-Uhlenbeck
half-life and a variance-ratio Hurst exponent.

Everything is implemented from scratch with numpy — no statsmodels at
runtime.  P-values for the ADF and Engle-Granger statistics are obtained by
monotone interpolation over published MacKinnon (1991/1994) response-surface
quantiles, and are therefore *approximate* (decisions at the usual 1/5/10%
levels, which rest on the tabulated critical values, are exact to the
table).  The approximation is documented rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares via lstsq, with standard errors."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return {"beta": beta, "resid": resid, "se": se, "sigma2": sigma2}


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """Regress y on [1, x]; return (alpha, beta, residual spread)."""
    X = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    fit = ols(X, y.to_numpy(float))
    alpha, beta = float(fit["beta"][0]), float(fit["beta"][1])
    spread = y - (alpha + beta * x)
    spread.name = "spread"
    return alpha, beta, spread


# ---------------------------------------------------------------------------
# ADF test
# ---------------------------------------------------------------------------

# MacKinnon asymptotic quantiles, ADF regression with constant, no trend.
_ADF_P = np.array([0.001, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
_ADF_S = np.array([-4.30, -3.43, -3.12, -2.86, -2.57, -2.08, -1.53, -1.02, -0.55, 0.35])
ADF_CRIT = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}

# MacKinnon (1991) quantiles for the Engle-Granger residual test,
# 2 series, constant in the cointegrating regression, no trend.
_EG_P = np.array([0.001, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
_EG_S = np.array([-4.95, -3.90, -3.60, -3.34, -3.04, -2.52, -1.95, -1.45, -1.02, -0.30])
EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


def _interp_pvalue(stat: float, stats_grid: np.ndarray, p_grid: np.ndarray) -> float:
    """Interpolate a p-value on a (statistic, p) quantile grid (log-p scale)."""
    if stat <= stats_grid[0]:
        return float(p_grid[0])
    if stat >= stats_grid[-1]:
        return float(p_grid[-1])
    # grids are increasing in stat and p
    return float(np.exp(np.interp(stat, stats_grid, np.log(p_grid))))


@dataclass
class ADFResult:
    stat: float
    pvalue: float
    lags: int
    nobs: int
    crit: dict

    @property
    def rejects_unit_root_5pct(self) -> bool:
        return self.stat < self.crit[0.05]


def adf_test(series: pd.Series | np.ndarray, maxlag: int | None = None) -> ADFResult:
    """
    Augmented Dickey-Fuller test (constant, no trend).

    Regression:  dy_t = c + gamma * y_{t-1} + sum_i delta_i * dy_{t-i} + e_t
    H0: gamma = 0 (unit root).  Lag order chosen by AIC up to a Schwert bound.
    """
    y = np.asarray(pd.Series(series).dropna(), float)
    n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    dy = np.diff(y)
    best = None
    for k in range(0, maxlag + 1):
        # build design: rows t = k .. n-2 (predicting dy[t] from y[t], dy[t-1..t-k])
        rows = n - 1 - k
        if rows <= k + 3:
            break
        Y = dy[k:]
        cols = [np.ones(rows), y[k:-1] if k == 0 else y[k:n - 1]]
        for i in range(1, k + 1):
            cols.append(dy[k - i:n - 1 - i])
        X = np.column_stack(cols)
        fit = ols(X, Y)
        sigma2 = max(fit["sigma2"], 1e-300)
        aic = rows * np.log(sigma2) + 2 * (k + 2)
        if best is None or aic < best[0]:
            best = (aic, k, fit)
    _, k, fit = best
    gamma = float(fit["beta"][1])
    se = float(fit["se"][1])
    stat = gamma / se if se > 0 else float("nan")
    return ADFResult(stat=stat, pvalue=_interp_pvalue(stat, _ADF_S, _ADF_P),
                     lags=k, nobs=int(len(fit["resid"])), crit=dict(ADF_CRIT))


# ---------------------------------------------------------------------------
# Engle-Granger cointegration
# ---------------------------------------------------------------------------

@dataclass
class EngleGrangerResult:
    alpha: float
    beta: float
    spread: pd.Series
    adf_stat: float
    pvalue: float
    crit: dict

    @property
    def cointegrated_5pct(self) -> bool:
        return self.adf_stat < self.crit[0.05]


def engle_granger(y: pd.Series, x: pd.Series) -> EngleGrangerResult:
    """
    Engle-Granger two-step test: OLS of y on x, then an ADF test (no
    constant, on the residuals) judged against Engle-Granger critical
    values, which are more negative than plain ADF values because the
    cointegrating vector was estimated.
    """
    alpha, beta, spread = hedge_ratio(y, x)
    r = spread.to_numpy(float)
    dr = np.diff(r)
    # residual ADF: dr_t = gamma * r_{t-1} + augmentation lags (no constant)
    n = len(r)
    maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    best = None
    for k in range(0, maxlag + 1):
        rows = n - 1 - k
        if rows <= k + 2:
            break
        Y = dr[k:]
        cols = [r[k:n - 1]]
        for i in range(1, k + 1):
            cols.append(dr[k - i:n - 1 - i])
        X = np.column_stack(cols)
        fit = ols(X, Y)
        sigma2 = max(fit["sigma2"], 1e-300)
        aic = rows * np.log(sigma2) + 2 * (k + 1)
        if best is None or aic < best[0]:
            best = (aic, fit)
    _, fit = best
    gamma = float(fit["beta"][0])
    se = float(fit["se"][0])
    stat = gamma / se if se > 0 else float("nan")
    return EngleGrangerResult(alpha=alpha, beta=beta, spread=spread,
                              adf_stat=stat,
                              pvalue=_interp_pvalue(stat, _EG_S, _EG_P),
                              crit=dict(EG_CRIT))


# ---------------------------------------------------------------------------
# Mean-reversion characterisation
# ---------------------------------------------------------------------------

def half_life(spread: pd.Series | np.ndarray) -> float:
    """
    Ornstein-Uhlenbeck half-life in observations.

    Regress  ds_t = lambda * s_{t-1} + c ;  half-life = -ln(2) / lambda.
    Returns inf when lambda >= 0 (no mean reversion detected).
    """
    s = np.asarray(pd.Series(spread).dropna(), float)
    ds = np.diff(s)
    X = np.column_stack([np.ones(len(ds)), s[:-1]])
    lam = float(ols(X, ds)["beta"][1])
    if lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


def hurst_exponent(series: pd.Series | np.ndarray,
                   lags: tuple[int, ...] = (2, 4, 8, 16, 32, 64)) -> float:
    """
    Hurst exponent by the variance-ratio method: for a process whose
    q-step increments satisfy Var(x_{t+q} - x_t) ~ q^(2H), H is half the
    slope of log Var vs log q.  H < 0.5 indicates mean reversion,
    H = 0.5 a random walk, H > 0.5 trending behaviour.
    """
    x = np.asarray(pd.Series(series).dropna(), float)
    taus, vars_ = [], []
    for q in lags:
        if len(x) <= q + 10:
            continue
        inc = x[q:] - x[:-q]
        v = float(np.var(inc, ddof=1))
        if v > 0:
            taus.append(q)
            vars_.append(v)
    if len(taus) < 3:
        return float("nan")
    slope = np.polyfit(np.log(taus), np.log(vars_), 1)[0]
    return float(slope / 2.0)


def rolling_zscore(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """Rolling z-score of the spread using a trailing window."""
    mu = spread.rolling(lookback).mean()
    sd = spread.rolling(lookback).std(ddof=1)
    return (spread - mu) / sd
