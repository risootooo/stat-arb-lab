#!/usr/bin/env python3
"""
StatArb Lab CLI — runs the three real-data case studies from the cached
prices in data/ (no network needed) and prints the summary tables.

    python main.py                # case-study tables
    python main.py --dashboard    # also (re)build docs/dashboard.html
    python main.py --json         # also write docs/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from statarb.cases import run_all, strip_objects          # noqa: E402
from statarb.metrics import format_stats                    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="StatArb Lab case studies")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = run_all()
    ca, cb, cc = r["case_a"], r["case_b"], r["case_c"]

    print("=" * 100)
    print("CASE A — Engle-Granger cointegration, both classic pairs, both directions")
    print("=" * 100)
    print(f"{'regression':<12}{'beta':>8}{'EG stat':>9}{'p approx':>10}{'coint@5%':>10}{'half-life':>11}")
    for row in ca["table"]:
        print(f"{row['y'] + '~' + row['x']:<12}{row['beta']:>8.3f}{row['adf_stat']:>9.3f}"
              f"{row['pvalue']:>10.3f}{str(row['cointegrated_5pct']):>10}"
              f"{row['half_life']:>10.1f}d")
    ch = ca["chosen"]
    print(f"\nStrongest: {ch['pair']} (dependent = {ch['y']})  alpha {ch['alpha']:.2f}, "
          f"beta {ch['beta']:.3f}, half-life {ch['half_life_days']:.1f}d, "
          f"Hurst(spread) {ch['hurst_spread']:.3f}, max |z| {ch['max_abs_z']:.2f} "
          f"on {ch['max_abs_z_date']}")
    for lab in ("first_half", "second_half"):
        h = ca["split_half"][lab]
        print(f"  split {lab} ({h['start']} -> {h['end']}): beta {h['beta']:+.3f}, "
              f"EG stat {h['adf_stat']:.3f}, p ~ {h['pvalue']:.3f}, "
              f"cointegrated@5% {h['cointegrated_5pct']}")
    bt = ca["backtest"]
    print(f"  backtest: total {bt['total_return']:+.2%}, Sharpe {bt['sharpe']:.2f}, "
          f"MaxDD {bt['max_drawdown']:.2%}, trades {bt['n_trades']:.0f}, "
          f"win {bt['win_rate']:.0%}, exposure {bt['exposure']:.0%}, "
          f"costs ${bt['total_costs']:,.0f}, stop-outs {bt['n_stopouts']}")
    op = ca["other_pair"]
    ob = op["backtest"]
    print(f"  other pair {op['pair']}: EG stat {op['adf_stat']:.3f} (not cointegrated), "
          f"backtest total {ob['total_return']:+.2%}, Sharpe {ob['sharpe']:.2f}, "
          f"stop-outs {ob['n_stopouts']} of {ob['n_trades']:.0f} trades")

    print()
    print("=" * 100)
    print("CASE B — COVID crash window, dual momentum vs SPY (31 Jan - 30 Apr 2020)")
    print("=" * 100)
    for lab in ("strategy", "benchmark"):
        w = cb[lab]
        print(f"{lab:<10} maxDD {w['max_dd_window']:.2%} ({w['dd_peak']} -> {w['dd_trough']}), "
              f"Jan31->Mar23 {w['ret_jan31_mar23']:+.2%}, Jan31->Apr30 {w['ret_jan31_apr30']:+.2%}")
    for d, h in cb["holdings"].items():
        print(f"  holdings {d}: " + ", ".join(f"{a} {w:.0%}" for a, w in h.items()))

    print()
    print("=" * 100)
    print(f"CASE C — full period from {cc['strategy_start']} (strategy live after 12m warm-up)")
    print("=" * 100)
    print(format_stats(cc["strategy"]))
    print(format_stats(cc["benchmark"]))
    s = cc["strategy"]
    print(f"  worst drawdown belongs to the strategy itself: {s['max_drawdown']:.2%} "
          f"({str(s['dd_peak'].date())} -> {str(s['dd_trough'].date())}); "
          f"total trading costs ${cc['total_costs']:,.0f}")
    print("\nDisclaimer: research/education only — not financial advice.")

    if args.dashboard:
        from statarb.report import build_dashboard
        out = Path(__file__).resolve().parent / "docs" / "dashboard.html"
        out.write_text(build_dashboard(r), encoding="utf-8")
        print(f"\nDashboard written to {out}")
    if args.json:
        out = Path(__file__).resolve().parent / "docs" / "results.json"
        clean = strip_objects({k: v for k, v in r.items()})
        # strip_objects leaves pandas objects inside stats dicts; coerce via json default
        out.write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
