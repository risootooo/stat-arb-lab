"""
Event-driven (day-by-day) pairs backtest with transaction costs.

Signal: rolling z-score of the spread ``s = y - (alpha + beta * x)``.
  * enter long-spread  (long y, short beta*x) when z < -entry
  * enter short-spread (short y, long beta*x) when z > +entry
  * exit when |z| < exit_band (mean reversion complete)
  * stop out when |z| > stop (the "convergence trade" is blowing out)

At entry the y-leg gets ``notional`` dollars and the x-leg ``beta`` times
as many shares as the y-leg has, so the position is hedge-ratio neutral.
Costs are charged on both legs, on entry and on exit, at ``cost_bps`` of
traded notional.  The book is marked to market daily, so the accounting
identity  final_equity == capital + sum(net trade PnL)  holds exactly
(and is unit-tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import perf_stats
from .pairs import rolling_zscore


@dataclass
class PairBacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    zscore: pd.Series
    spread: pd.Series
    stats: dict
    exposure: float = 0.0
    total_costs: float = 0.0
    params: dict = field(default_factory=dict)


def backtest_pair(y: pd.Series, x: pd.Series, alpha: float, beta: float,
                  lookback: int = 60, entry: float = 2.0,
                  exit_band: float = 0.5, stop: float = 3.5,
                  cost_bps: float = 5.0, notional: float = 10_000.0,
                  capital: float = 100_000.0) -> PairBacktestResult:
    spread = (y - (alpha + beta * x)).rename("spread")
    z = rolling_zscore(spread, lookback)
    idx = y.index
    yv, xv = y.to_numpy(float), x.to_numpy(float)
    zv = z.to_numpy(float)

    equity = np.zeros(len(idx))
    trades: list[dict] = []
    realised = 0.0
    total_costs = 0.0
    days_in = 0
    pos = None  # dict with entry info

    def open_pos(t: int, direction: int) -> dict:
        qty_y = notional / yv[t]
        qty_x = beta * qty_y
        traded = notional + abs(qty_x) * xv[t]
        cost = cost_bps * 1e-4 * traded
        return {"dir": direction, "t0": t, "qty_y": qty_y, "qty_x": qty_x,
                "entry_spread": spread.iloc[t], "cost_open": cost}

    def close_pos(t: int, p: dict, reason: str) -> dict:
        traded = abs(p["qty_y"]) * yv[t] + abs(p["qty_x"]) * xv[t]
        cost_close = cost_bps * 1e-4 * traded
        gross = p["dir"] * (p["qty_y"] * (yv[t] - yv[p["t0"]])
                            - p["qty_x"] * (xv[t] - xv[p["t0"]]))
        total = p["cost_open"] + cost_close
        return {"entry_date": idx[p["t0"]], "exit_date": idx[t],
                "direction": "long_spread" if p["dir"] > 0 else "short_spread",
                "holding_days": t - p["t0"], "gross_pnl": gross,
                "costs": total, "net_pnl": gross - total, "exit_reason": reason}

    for t in range(len(idx)):
        if pos is not None:
            days_in += 1
            zt = zv[t]
            if not np.isnan(zt):
                if abs(zt) < exit_band:
                    tr = close_pos(t, pos, "mean_reversion")
                    realised += tr["net_pnl"]; total_costs += tr["costs"]
                    trades.append(tr); pos = None
                elif abs(zt) > stop:
                    tr = close_pos(t, pos, "stop")
                    realised += tr["net_pnl"]; total_costs += tr["costs"]
                    trades.append(tr); pos = None
        if pos is None and not np.isnan(zv[t]):
            if zv[t] < -entry:
                pos = open_pos(t, +1)
            elif zv[t] > entry:
                pos = open_pos(t, -1)
        # mark to market
        mtm = 0.0
        if pos is not None:
            mtm = pos["dir"] * (pos["qty_y"] * (yv[t] - yv[pos["t0"]])
                                - pos["qty_x"] * (xv[t] - xv[pos["t0"]])) \
                - pos["cost_open"]
        equity[t] = capital + realised + mtm
    if pos is not None:  # force-close on the last bar
        tr = close_pos(len(idx) - 1, pos, "end_of_sample")
        realised += tr["net_pnl"]; total_costs += tr["costs"]
        trades.append(tr)
        equity[-1] = capital + realised

    eq = pd.Series(equity, index=idx, name="equity")
    trades_df = pd.DataFrame(trades)
    stats = perf_stats(eq, name="pair")
    if len(trades_df):
        stats["n_trades"] = int(len(trades_df))
        stats["win_rate"] = float((trades_df["net_pnl"] > 0).mean())
        stats["avg_holding_days"] = float(trades_df["holding_days"].mean())
    else:
        stats["n_trades"] = 0
        stats["win_rate"] = float("nan")
        stats["avg_holding_days"] = float("nan")
    return PairBacktestResult(
        equity=eq, trades=trades_df, zscore=z, spread=spread, stats=stats,
        exposure=days_in / len(idx) if len(idx) else 0.0,
        total_costs=total_costs,
        params=dict(lookback=lookback, entry=entry, exit_band=exit_band,
                    stop=stop, cost_bps=cost_bps, notional=notional,
                    capital=capital))
