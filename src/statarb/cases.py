"""
The three real-data case studies of docs/CASES.md, computed end-to-end
from the cached prices in ``data/``.  Every number quoted in the README
and CASES.md comes out of these functions — nothing is typed by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest_pair
from .data import load_momentum, load_pairs
from .kalman import kalman_hedge_ratio
from .metrics import max_drawdown, perf_stats
from .momentum import backtest_momentum
from .pairs import (adf_test, engle_granger, half_life, hurst_exponent,
                    rolling_zscore)

PAIR_CANDIDATES = [("KO", "PEP"), ("XOM", "CVX")]


def _eg_row(y: pd.Series, x: pd.Series, yname: str, xname: str) -> dict:
    eg = engle_granger(y, x)
    return {"y": yname, "x": xname, "alpha": eg.alpha, "beta": eg.beta,
            "adf_stat": eg.adf_stat, "pvalue": eg.pvalue,
            "cointegrated_5pct": bool(eg.cointegrated_5pct),
            "half_life": half_life(eg.spread)}


def case_a() -> dict:
    """Cointegration hunt on the two classic pairs, both directions."""
    px = load_pairs()
    table = []
    for a, b in PAIR_CANDIDATES:
        table.append(_eg_row(px[a], px[b], a, b))
        table.append(_eg_row(px[b], px[a], b, a))
    # strongest full-sample cointegration = most negative EG statistic
    best = min(table, key=lambda r: r["adf_stat"])
    y, x = px[best["y"]], px[best["x"]]
    eg = engle_granger(y, x)
    hl = half_life(eg.spread)
    hurst = hurst_exponent(eg.spread)
    z = rolling_zscore(eg.spread, 60)
    bt = backtest_pair(y, x, eg.alpha, eg.beta)
    kf = kalman_hedge_ratio(y, x)

    mid = len(px) // 2
    halves = {}
    for lab, sl in (("first_half", slice(0, mid)), ("second_half", slice(mid, None))):
        sub = px.iloc[sl]
        e2 = engle_granger(sub[best["y"]], sub[best["x"]])
        halves[lab] = {"start": str(sub.index[0].date()),
                       "end": str(sub.index[-1].date()),
                       "alpha": e2.alpha, "beta": e2.beta,
                       "adf_stat": e2.adf_stat, "pvalue": e2.pvalue,
                       "cointegrated_5pct": bool(e2.cointegrated_5pct),
                       "half_life": half_life(e2.spread)}

    # the losing candidate, traded with its own (first-listed) direction
    other_name = [p for p in PAIR_CANDIDATES
                  if best["y"] not in p and best["x"] not in p][0]
    oy, ox = px[other_name[0]], px[other_name[1]]
    eg_o = engle_granger(oy, ox)
    bt_o = backtest_pair(oy, ox, eg_o.alpha, eg_o.beta)
    kf_o = kalman_hedge_ratio(oy, ox)

    # individual-leg ADF (is one leg near-stationary by itself?)
    leg_adf = {t: {"stat": adf_test(px[t]).stat, "pvalue": adf_test(px[t]).pvalue}
               for t in sorted({best["y"], best["x"]})}

    return {
        "table": table,
        "chosen": {"pair": f"{best['y']}/{best['x']}", "y": best["y"], "x": best["x"],
                   "alpha": eg.alpha, "beta": eg.beta,
                   "adf_stat": eg.adf_stat, "pvalue": eg.pvalue,
                   "cointegrated_5pct": bool(eg.cointegrated_5pct),
                   "half_life_days": hl, "hurst_spread": hurst,
                   "max_abs_z": float(z.abs().max()),
                   "max_abs_z_date": str(z.abs().idxmax().date())},
        "split_half": halves,
        "backtest": {**bt.stats, "exposure": bt.exposure,
                     "total_costs": bt.total_costs,
                     "n_stopouts": int((bt.trades["exit_reason"] == "stop").sum())
                     if len(bt.trades) else 0,
                     "final_equity": float(bt.equity.iloc[-1])},
        "kalman": {"beta_start_after_warmup": float(kf.beta.iloc[60]),
                   "beta_end": float(kf.beta.iloc[-1]),
                   "beta_min": float(kf.beta.min()), "beta_max": float(kf.beta.max())},
        "leg_adf": leg_adf,
        "other_pair": {"pair": f"{other_name[0]}/{other_name[1]}",
                       "adf_stat": eg_o.adf_stat, "pvalue": eg_o.pvalue,
                       "beta": eg_o.beta,
                       "backtest": {**bt_o.stats, "exposure": bt_o.exposure,
                                    "total_costs": bt_o.total_costs,
                                    "n_stopouts": int((bt_o.trades["exit_reason"] == "stop").sum())
                                    if len(bt_o.trades) else 0},
                       "kalman_beta_end": float(kf_o.beta.iloc[-1])},
        "_objects": {"bt": bt, "eg": eg},
    }


def case_b() -> dict:
    """Momentum vs buy-and-hold through the Feb-Apr 2020 COVID crash."""
    res = backtest_momentum(load_momentum())
    start = res.weights_history.index[0]
    eq = res.equity.loc[start:]
    bench = res.benchmark_equity.loc[start:]
    out = {"strategy_start": str(start.date())}
    for lab, series in (("strategy", eq), ("benchmark", bench)):
        win = series.loc["2020-01-31":"2020-04-30"]
        dd, peak, trough = max_drawdown(win)
        out[lab] = {
            "max_dd_window": dd, "dd_peak": str(peak.date()),
            "dd_trough": str(trough.date()),
            "ret_jan31_mar23": float(win.loc[: "2020-03-23"].iloc[-1] / win.iloc[0] - 1),
            "ret_jan31_apr30": float(win.iloc[-1] / win.iloc[0] - 1),
        }
    wh = res.weights_history.loc["2020-01-01":"2020-03-31"]
    out["holdings"] = {str(d.date()): {a: float(w) for a, w in row.items() if w > 0}
                       for d, row in wh.iterrows()}
    out["_objects"] = {"res": res, "eq": eq, "bench": bench}
    return out


def case_c() -> dict:
    """Full-period momentum record + its worst drawdown (the in-sample
    stand-in for a 'momentum crash', since the data starts in 2018)."""
    res = backtest_momentum(load_momentum())
    start = res.weights_history.index[0]
    eq = res.equity.loc[start:]
    bench = res.benchmark_equity.loc[start:]
    s = perf_stats(eq, "dual_momentum")
    b = perf_stats(bench, "SPY_buy_hold")
    return {"strategy_start": str(start.date()),
            "strategy": s, "benchmark": b,
            "total_costs": res.total_costs,
            "_objects": {"res": res, "eq": eq, "bench": bench}}


def run_all() -> dict:
    return {"case_a": case_a(), "case_b": case_b(), "case_c": case_c()}


def strip_objects(d):
    """Recursively drop private '_objects' entries (for JSON export)."""
    if isinstance(d, dict):
        return {k: strip_objects(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, (pd.Timestamp,)):
        return str(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d
