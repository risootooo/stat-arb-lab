"""Performance metrics shared by the backtests."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return (max drawdown as a negative fraction, peak date, trough date)."""
    eq = equity.dropna()
    running_peak = eq.cummax()
    dd = eq / running_peak - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def perf_stats(equity: pd.Series, name: str = "") -> dict:
    """Summary statistics for a daily equity curve (risk-free rate = 0)."""
    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) \
        if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0
    dd, peak, trough = max_drawdown(eq)
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) \
        if len(downside) > 1 and downside.std(ddof=1) > 0 else 0.0
    return {
        "name": name,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": dd,
        "dd_peak": peak,
        "dd_trough": trough,
        "calmar": float(cagr / abs(dd)) if dd < 0 else float("inf"),
    }


def format_stats(s: dict) -> str:
    return (
        f"{s['name']:<28} total {s['total_return']:+8.2%}  CAGR {s['cagr']:+7.2%}  "
        f"vol {s['ann_vol']:6.2%}  Sharpe {s['sharpe']:5.2f}  "
        f"MaxDD {s['max_drawdown']:7.2%} ({str(s['dd_peak'].date())} -> "
        f"{str(s['dd_trough'].date())})  Calmar {s['calmar']:5.2f}"
    )
