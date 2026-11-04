# Case Studies — real data, real history, honest verdicts

All numbers below were computed by this repo's code (`python main.py`,
`src/statarb/cases.py`) on the cached Yahoo Finance data in `data/`
(downloaded 2026-11-04). Pairs: daily adjusted closes, 2021-11-04 →
2026-11-04. Momentum universe: 2018-01-02 → 2026-11-04. Nothing is
hand-typed; rerunning the code reproduces every figure.

---

## Case A — The classic pairs: Coke/Pepsi and Exxon/Chevron

Every pairs-trading tutorial starts with KO/PEP and XOM/CVX. We ran the
Engle-Granger two-step test on both, **in both regression directions**
(the test is not symmetric — the answer depends on which leg you call
"dependent", a fact most tutorials never mention):

| Regression | Hedge ratio β | EG statistic | p (approx) | Cointegrated @5%? | Half-life |
|---|---:|---:|---:|:---:|---:|
| KO ~ PEP | −0.375 | −1.402 | 0.765 | No | 137.5 d |
| **PEP ~ KO** | **−0.391** | **−3.554** | **0.028** | **Yes** | **34.4 d** |
| XOM ~ CVX | +1.008 | −2.411 | 0.285 | No | 73.7 d |
| CVX ~ XOM | +0.866 | −2.594 | 0.219 | No | 61.1 d |

So exactly one of four specifications passes: PEP regressed on KO
(spread = PEP − (172.53 − 0.391·KO), Hurst exponent of the spread 0.42,
consistent with mean reversion). Traded with a 60-day rolling z-score
(entry |z| > 2, exit |z| < 0.5, stop |z| > 3.5, 5 bps per leg per side):

| PEP/KO backtest, 2021-11 → 2026-11 | |
|---|---:|
| Total return (on $100k, $10k per leg) | **+3.35 %** |
| Sharpe / Sortino | 0.56 / 0.50 |
| Max drawdown | −1.62 % (2024-04-23 → 2025-01-10) |
| Trades / win rate / avg holding | 26 / 69 % / 17.3 days |
| Exposure / costs / stop-outs | 36 % / $305 / 4 |

A modest profit. **Now the part that matters — the relationship does not
survive its own sample:**

| Split sample (PEP ~ KO) | β | EG statistic | p | Cointegrated? |
|---|---:|---:|---:|:---:|
| First half (Nov 2021 – May 2024) | **+2.02** | −1.786 | 0.571 | No |
| Second half (May 2024 – Nov 2026) | −0.31 | −2.492 | 0.259 | No |
| Full sample | −0.39 | −3.554 | 0.028 | Yes |

Neither half is cointegrated on its own, and the hedge ratio *flips sign*
(+2.02 → −0.31) between halves. The Kalman-filter hedge ratio tells the
same story dynamically: it wanders between 0.55 and 3.04 over the sample
and ends at 1.47 — never anywhere near the static OLS value of −0.39.
Two further warnings: PEP *by itself* rejects a unit root at 5% in this
sample (ADF −2.91, p ≈ 0.04), which can manufacture spurious
"cointegration"; and the spread still reached |z| = 4.17 in Feb 2026.

**Verdict:** the full-sample pass is an artefact of stitching two regimes
together. This is cointegration decay in the wild — the same failure
mode, at kitchen-table scale, that destroyed LTCM's convergence book in
1998: a spread that "always came back" until the regime that made it come
back ended. The backtest's small profit is real but rests on a hedge
ratio nobody should have trusted.

### The other pair is worse

XOM/CVX fails the test in both directions (best statistic −2.59,
p ≈ 0.22), and its backtest shows what trading a non-cointegrated pair
costs: **−3.56 % total, Sharpe −0.47, max drawdown −5.54 %, and 18 of 39
trades stopped out** as the spread blew through the 3.5σ stop — including
a |z| of 5.18 on 2022-03-01, days after Russia's invasion of Ukraine
re-priced the two oil majors' relative value overnight. Convergence risk
is not a footnote; in this sample it was the dominant risk.

---

## Case B — Real historical case: dual momentum in the COVID crash, Feb–Apr 2020

The momentum book (`src/statarb/momentum.py`: top-3 of SPY, QQQ, EFA,
EEM, GLD, VNQ by 12-1 momentum, survivors must have positive absolute
momentum, unfilled weight parks in TLT, monthly rebalance, 10 bps costs)
was live from 2019-01-31, so it walked into the fastest crash in market
history fully invested. What actually happened:

| 31 Jan → 30 Apr 2020 | Max drawdown in window | 31 Jan → 23 Mar | 31 Jan → 30 Apr |
|---|---:|---:|---:|
| Dual momentum | **−25.20 %** (peak 14 Feb → trough 20 Mar) | −20.19 % | **−2.48 %** |
| SPY buy-and-hold | −33.72 % (19 Feb → 23 Mar) | −30.30 % | −9.18 % |

Holdings tell the honest story of *why the protection was only partial*:

| Rebalance | Portfolio |
|---|---|
| 2020-01-31 | SPY ⅓, QQQ ⅓, VNQ ⅓ |
| 2020-02-28 | SPY ⅓, QQQ ⅓, **GLD ⅓** (VNQ rotated out, gold in) |
| 2020-03-31 | SPY ⅓, QQQ ⅓, GLD ⅓ |

A monthly 12-1 filter *cannot* react inside February: at the 28 Feb
rebalance, 12-month momentum in equities was still positive (the crash
was eight days old against a 12-month lookback), so the book stayed long
equities through the worst of it. The drawdown saving (−25 % vs −34 %)
came from the gold rotation and from *not* being 100 % in the index; the
much better April number (−2.5 % vs −9.2 %) came from staying invested
for the rebound while buy-and-hold was still digging out. Dual momentum
here was crash *cushioning*, not crash *avoidance* — and any backtest
that shows a monthly rule sidestepping COVID entirely is probably
cheating on timestamps. (The pairs file starts in Nov 2021, so the
pairs' COVID behaviour cannot be shown from this cache; their in-sample
blow-out is documented in Case A via the 2022 XOM/CVX spread.)

---

## Case C — Momentum crashes: what this sample can and cannot say

The canonical momentum crash is 2009 (Daniel & Moskowitz 2016): factor
momentum lost over 40 % in months as the market rebounded and the
short/low-beta legs exploded. **Our data starts in 2018, so that event is
not in this repository, and we will not pretend otherwise.**

The worst drawdown the strategy *does* produce in-sample is instructive
in a different way — the 2022 rate-shock bear market:

| Full period, 2019-01-31 → 2026-11-04 | Dual momentum | SPY buy-and-hold |
|---|---:|---:|
| Total return | +144.57 % | **+219.07 %** |
| CAGR | +12.22 % | +16.13 % |
| Volatility | 17.36 % | 19.17 % |
| Sharpe / Sortino | 0.75 / 0.99 | **0.88** / 1.08 |
| Max drawdown | −33.46 % (2022-01-03 → 2022-10-24) | −33.72 % (2020-02-19 → 2020-03-23) |
| Calmar | 0.37 | 0.48 |

(Total strategy trading costs over the period: $4,769 on the $100k book.)

Two honest conclusions. First, over a decade dominated by a US equity
bull market, the benchmark wins outright — the filter's insurance
premium (time in TLT/GLD, whipsaw turnover) costs more than the crashes
it cushions. Second, the strategy's *worst* drawdown (−33.5 %) is as deep
as buy-and-hold's worst — it just happens in the slow 2022 grind, where a
12-1 signal keeps re-entering falling assets after brief rallies, rather
than in a fast crash. Momentum's risk did not disappear; it changed
shape. That is the finding, and it is worth more than a doctored equity
curve.

---

## Limitations

- **Survivorship/selection**: the pairs and ETFs are today's famous
  names, chosen *because* tutorials use them — itself a selection bias.
- **Full-sample hedge ratios** in Case A use information from the whole
  period (deliberately: the point of the case is that even the
  in-sample-optimal static hedge is unstable; a production system would
  re-estimate, e.g. with the Kalman filter provided).
- **P-values are approximate**: ADF/EG p-values are interpolated from
  MacKinnon response-surface quantiles; the 1/5/10 % critical-value
  decisions are exact to the published tables.
- **Costs are stylised**: flat bps per trade, no slippage, borrow,
  dividend, or tax modelling; ETF prices are dividend-adjusted closes.
- **Absolute-momentum hurdle is cash ≈ 0**: no T-bill series is bundled,
  which flatters the filter slightly in high-rate years (2022-23).
- Nothing here is a recommendation. Cointegration decays (Case A),
  monthly filters react late (Case B), and momentum changes its risk
  shape rather than removing it (Case C).
