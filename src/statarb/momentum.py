"""
Cross-sectional + dual (absolute) momentum on a small ETF universe,
in the spirit of Antonacci's *Dual Momentum* and the classic 12-1
cross-sectional momentum literature (Jegadeesh & Titman 1993).

Rules, applied at each month-end close:
  * momentum of an asset = 12-1 return: P[t-1m] / P[t-12m] - 1
    (skip the most recent month, which is dominated by short-term reversal);
  * relative momentum: rank the risk assets by that momentum;
  * absolute momentum: keep only assets whose 12-1 momentum is positive
    (vs an approximate cash return of 0 — see README limitations);
  * hold the top ``top_k`` survivors equal-weighted; any unfilled weight,
    and the whole portfolio when nothing survives, goes to the defensive
    asset (TLT).  This is the "dual momentum" crash filter: in a broad
    downtrend the book rotates into duration instead of riding equities down.

Between rebalances weights are held fixed and the portfolio is marked to
market daily, so drawdowns are measured on a true daily equity curve.
Turnover costs are charged at each rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import perf_stats

RISK_ASSETS = ["SPY", "QQQ", "EFA", "EEM", "GLD", "VNQ"]
DEFENSIVE = "TLT"
BENCHMARK = "SPY"


def monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Close on the last *trading day* of each calendar month."""
    last_days = (prices.groupby([prices.index.year, prices.index.month])
                 .apply(lambda g: g.index[-1]))
    return prices.loc[sorted(last_days.tolist())].dropna(how="any")


def momentum_signal(mpx: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """12-1 momentum for every column of a monthly price frame."""
    return mpx.shift(skip) / mpx.shift(lookback) - 1.0


def select_weights(mom_row: pd.Series, risk_assets: list[str],
                   defensive: str, top_k: int = 3) -> dict[str, float]:
    """
    Dual-momentum selection for one rebalance date.

    ``mom_row`` holds 12-1 momentum for all assets on that date.
    Returns a weight dict summing to 1.
    """
    mom = mom_row[risk_assets].dropna()
    survivors = mom[mom > 0.0].sort_values(ascending=False)
    chosen = list(survivors.index[:top_k])
    w: dict[str, float] = {}
    if chosen:
        each = 1.0 / top_k
        for a in chosen:
            w[a] = each
        w[defensive] = 1.0 - each * len(chosen)
        if w[defensive] <= 0:
            del w[defensive]
    else:
        w[defensive] = 1.0
    return w


@dataclass
class MomentumResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    weights_history: pd.DataFrame
    stats: dict
    benchmark_stats: dict
    total_costs: float = 0.0
    params: dict = field(default_factory=dict)


def backtest_momentum(prices: pd.DataFrame, top_k: int = 3,
                      cost_bps: float = 10.0, capital: float = 100_000.0,
                      risk_assets: list[str] = RISK_ASSETS,
                      defensive: str = DEFENSIVE,
                      benchmark: str = BENCHMARK) -> MomentumResult:
    mpx = monthly_prices(prices)
    mom = momentum_signal(mpx)
    rebal_dates = [d for d in mpx.index if not mom.loc[d, risk_assets].isna().all()]
    rebal_set = {d: select_weights(mom.loc[d], risk_assets, defensive, top_k)
                 for d in rebal_dates}

    daily_rets = prices.pct_change().fillna(0.0)
    assets = list(prices.columns)
    cur_w = pd.Series(0.0, index=assets)
    equity = capital
    eq_series = {}
    weights_hist = []
    total_costs = 0.0
    rebal_day_set = set(rebal_dates)

    for date, row in daily_rets.iterrows():
        if cur_w.any():
            equity *= 1.0 + float((cur_w * row).sum())
        eq_series[date] = equity
        if date in rebal_day_set:
            target = pd.Series(0.0, index=assets)
            for a, w in rebal_set[date].items():
                target[a] = w
            turnover = float((target - cur_w).abs().sum())
            cost = cost_bps * 1e-4 * turnover * equity
            equity -= cost
            total_costs += cost
            eq_series[date] = equity
            cur_w = target
            weights_hist.append({"date": date, **{a: target[a] for a in assets}})

    eq = pd.Series(eq_series).sort_index()
    bench = capital * prices[benchmark] / prices[benchmark].iloc[0]
    bench = bench.loc[eq.index[0]:]
    stats = perf_stats(eq, name="dual_momentum")
    bstats = perf_stats(bench, name=f"benchmark_{benchmark}")
    return MomentumResult(
        equity=eq, benchmark_equity=bench,
        weights_history=pd.DataFrame(weights_hist).set_index("date")
        if weights_hist else pd.DataFrame(),
        stats=stats, benchmark_stats=bstats, total_costs=total_costs,
        params=dict(top_k=top_k, cost_bps=cost_bps, capital=capital))
