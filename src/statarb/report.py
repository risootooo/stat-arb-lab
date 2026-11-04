"""Self-contained dark HTML dashboard built from the real case results.

Inline SVG only — no scripts, no external references, works offline.
"""
from __future__ import annotations

import pandas as pd


def _points(series: pd.Series, w: int, h: int, ymin: float, ymax: float,
            max_pts: int = 420) -> str:
    s = series.dropna()
    if len(s) > max_pts:
        step = len(s) / max_pts
        s = s.iloc[[int(i * step) for i in range(max_pts)]]
    rng = (ymax - ymin) or 1.0
    n = len(series.dropna())
    pos = {d: i for i, d in enumerate(series.dropna().index)}
    pts = []
    for d, v in s.items():
        px = pos[d] / max(n - 1, 1) * w
        py = h - (float(v) - ymin) / rng * h
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def _chart(series: pd.Series, w: int, h: int, color: str, title: str,
           markers: str = "", hline: float | None = None,
           ymin: float | None = None, ymax: float | None = None) -> str:
    s = series.dropna()
    lo = float(s.min()) if ymin is None else ymin
    hi = float(s.max()) if ymax is None else ymax
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    pts = _points(series, w, h, lo, hi)
    hl = ""
    if hline is not None:
        y = h - (hline - lo) / (hi - lo) * h
        hl = (f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" '
              f'stroke="#5b6575" stroke-dasharray="4 4" stroke-width="1"/>')
    return f"""<div class="chart"><h3>{title}</h3>
<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img">
<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>
{hl}{markers}</svg>
<div class="axis"><span>{s.index[0].date()}</span><span>min {lo:.2f} · max {hi:.2f}</span><span>{s.index[-1].date()}</span></div></div>"""


def _trade_markers(bt, z: pd.Series, w: int, h: int) -> str:
    zz = z.dropna()
    lo, hi = float(zz.min()), float(zz.max())
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    n = len(zz)
    pos = {d: i for i, d in enumerate(zz.index)}
    out = []
    for _, tr in bt.trades.iterrows():
        for d, col, sym in ((tr["entry_date"], "#f0b429", "▲"), (tr["exit_date"], "#8b93a3", "✕")):
            if d not in pos:
                continue
            cx = pos[d] / max(n - 1, 1) * w
            cy = h - (float(zz.loc[d]) - lo) / (hi - lo) * h
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{col}"><title>{sym} {str(d.date())}</title></circle>')
    return "".join(out)


def build_dashboard(results: dict) -> str:
    ca, cb, cc = results["case_a"], results["case_b"], results["case_c"]
    bt = ca["_objects"]["bt"]
    res = cc["_objects"]["res"]
    eq = (cc["_objects"]["eq"] / cc["_objects"]["eq"].iloc[0] * 100)
    bench = (cc["_objects"]["bench"] / cc["_objects"]["bench"].iloc[0] * 100)
    ch = ca["chosen"]
    markers = _trade_markers(bt, bt.zscore, 900, 190)

    both = pd.DataFrame({"strategy": eq, "bench": bench}).dropna()
    lo = float(both.min().min()) * 0.97
    hi = float(both.max().max()) * 1.03
    mom_chart = f"""<div class="chart"><h3>Dual momentum vs SPY buy-and-hold (rebased to 100, {cc['strategy_start']} → {str(eq.index[-1].date())})</h3>
<svg viewBox="0 0 900 240" preserveAspectRatio="none" role="img">
<polyline points="{_points(both['bench'], 900, 240, lo, hi)}" fill="none" stroke="#8b93a3" stroke-width="1.4"/>
<polyline points="{_points(both['strategy'], 900, 240, lo, hi)}" fill="none" stroke="#4cc38a" stroke-width="1.8"/></svg>
<div class="axis"><span style="color:#4cc38a">■ dual momentum</span><span style="color:#8b93a3">■ SPY</span><span>{str(eq.index[-1].date())}</span></div></div>"""

    s, b = cc["strategy"], cc["benchmark"]
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StatArb Lab — real-data case studies</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:32px 20px}}
.wrap{{max-width:960px;margin:0 auto}} h1{{font-size:30px;margin:0 0 4px}} h2{{margin:34px 0 10px;font-size:21px}}
.sub{{color:#8b93a3;margin:0 0 22px}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;flex:1;min-width:180px}}
.card .k{{color:#8b93a3;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}
.card .v{{font-size:24px;font-weight:650;margin-top:4px}} .card .n{{color:#8b93a3;font-size:12.5px;margin-top:2px}}
.chart{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 18px;margin:14px 0}}
.chart h3{{margin:0 0 10px;font-size:15.5px}} .chart svg{{width:100%;height:auto;display:block}}
.axis{{display:flex;justify-content:space-between;color:#8b93a3;font-size:12px;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid #262c36;padding:7px 9px;text-align:right}}
td:first-child,th:first-child{{text-align:left}} .warn{{color:#f0b429}} .bad{{color:#f47067}} .good{{color:#4cc38a}}
.verdict{{background:#2a1414;border:1px solid #6e2b25;border-radius:10px;padding:14px 18px}}
footer{{color:#8b93a3;font-size:12.5px;margin-top:30px}}
</style></head><body><div class="wrap">
<h1>StatArb Lab — case studies on real data</h1>
<p class="sub">Pairs: Yahoo Finance adjusted closes, Nov 2021 – Nov 2026 · Momentum universe: Jan 2018 – Nov 2026 · Downloaded 2026-11-04 · All figures computed by the code in this repo</p>

<div class="cards">
<div class="card"><div class="k">Case A pair</div><div class="v">{ch['pair']}</div><div class="n">Engle-Granger p ≈ {ch['pvalue']:.3f} · half-life {ch['half_life_days']:.0f} days — but see the verdict</div></div>
<div class="card"><div class="k">Pair backtest</div><div class="v">{ca['backtest']['total_return']:+.2%}</div><div class="n">Sharpe {ca['backtest']['sharpe']:.2f} · {ca['backtest']['n_trades']:.0f} trades · costs ${ca['backtest']['total_costs']:,.0f} · {ca['backtest']['n_stopouts']} stop-outs</div></div>
<div class="card"><div class="k">COVID window drawdown</div><div class="v">{cb['strategy']['max_dd_window']:.1%}</div><div class="n">momentum vs SPY {cb['benchmark']['max_dd_window']:.1%} (Feb–Apr 2020)</div></div>
<div class="card"><div class="k">Momentum vs SPY, full period</div><div class="v">{s['total_return']:+.0%} vs {b['total_return']:+.0%}</div><div class="n">Sharpe {s['sharpe']:.2f} vs {b['sharpe']:.2f} — the benchmark wins the bull market; honesty in docs/CASES.md</div></div>
</div>

<h2>Case A — {ch['pair']} spread and z-score (▲ entry · ✕ exit)</h2>
{_chart(bt.spread, 900, 190, '#58a6ff', f"Spread = {ch['y']} − ({ch['alpha']:.1f} {ch['beta']:+.3f} × {ch['x']})")}
{_chart(bt.zscore, 900, 190, '#f0b429', 'Rolling 60-day z-score with trade markers', markers=markers, hline=0.0)}
<div class="verdict"><strong class="bad">Verdict:</strong> the full-sample test passes (p ≈ {ch['pvalue']:.3f}), but neither half of the sample is cointegrated on its own and the hedge ratio flips sign between halves (β {ca['split_half']['first_half']['beta']:+.2f} → {ca['split_half']['second_half']['beta']:+.2f}). A textbook pass, a practical warning. Full analysis: docs/CASES.md.</div>

<h2>Momentum — the dual-momentum book vs its benchmark</h2>
{mom_chart}
<table><tr><th></th><th>Total</th><th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th></tr>
<tr><td>Dual momentum (net of costs)</td><td>{s['total_return']:+.2%}</td><td>{s['cagr']:+.2%}</td><td>{s['ann_vol']:.2%}</td><td>{s['sharpe']:.2f}</td><td>{s['max_drawdown']:.2%}</td></tr>
<tr><td>SPY buy-and-hold</td><td>{b['total_return']:+.2%}</td><td>{b['cagr']:+.2%}</td><td>{b['ann_vol']:.2%}</td><td>{b['sharpe']:.2f}</td><td>{b['max_drawdown']:.2%}</td></tr></table>

<h2>Case B — COVID crash window (31 Jan → 30 Apr 2020)</h2>
<table><tr><th></th><th>Max DD in window</th><th>31 Jan → 23 Mar</th><th>31 Jan → 30 Apr</th></tr>
<tr><td>Dual momentum</td><td>{cb['strategy']['max_dd_window']:.2%}</td><td>{cb['strategy']['ret_jan31_mar23']:+.2%}</td><td>{cb['strategy']['ret_jan31_apr30']:+.2%}</td></tr>
<tr><td>SPY buy-and-hold</td><td>{cb['benchmark']['max_dd_window']:.2%}</td><td>{cb['benchmark']['ret_jan31_mar23']:+.2%}</td><td>{cb['benchmark']['ret_jan31_apr30']:+.2%}</td></tr></table>
<p class="sub">The monthly filter reacted late — it was still long equities at the February rebalance — but shallower exposure and the late-February rotation into gold cut the drawdown materially. Details and holdings: docs/CASES.md.</p>

<footer>StatArb Lab · Research and education only — not financial advice. Past performance of any rule shown here does not predict future results; cointegration decays and momentum crashes are real, and both are documented rather than hidden in docs/CASES.md.</footer>
</div></body></html>"""
    return html
