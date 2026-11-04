"""
Download and cache the real price data used by StatArb Lab.

Data is fetched once with yfinance (per-ticker ``history`` calls — the bulk
``download`` endpoint silently dropped TLT on 2026-11-04) and cached as small
CSVs under ``data/`` so every analysis in the repo then runs fully offline.

Two files are produced:

* ``data/pairs_prices.csv``    — KO, PEP, XOM, CVX, last 5 years of daily
  split/dividend-adjusted closes.
* ``data/momentum_prices.csv`` — SPY, QQQ, EFA, EEM, TLT, GLD, VNQ from
  2018-01-01.  The momentum universe deliberately uses a longer history than
  the pairs file: a plain 5-year window starting in Nov 2021 would exclude
  the Feb-Apr 2020 COVID crash, which is Case B of docs/CASES.md.

Run with a Python that has yfinance installed, e.g.:

    python examples/download_data.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PAIRS_TICKERS = ["KO", "PEP", "XOM", "CVX"]
MOMENTUM_TICKERS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "VNQ"]
MOMENTUM_START = "2018-01-01"


def fetch(tickers: list[str], start: str | None) -> pd.DataFrame:
    """Fetch adjusted daily closes one ticker at a time."""
    closes = {}
    for t in tickers:
        hist = yf.Ticker(t).history(start=start, period=None if start else "5y",
                                    auto_adjust=True)
        if hist.empty:
            raise RuntimeError(f"no data returned for {t}")
        s = hist["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        closes[t] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(closes).sort_index()
    df.index.name = "Date"
    return df.dropna(how="any")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    pairs = fetch(PAIRS_TICKERS, start=None)  # trailing 5y via history(period=5y)
    # history(period="5y") is handled inside fetch via period kwarg when start=None
    pairs.to_csv(DATA / "pairs_prices.csv")
    mom = fetch(MOMENTUM_TICKERS, start=MOMENTUM_START)
    mom.to_csv(DATA / "momentum_prices.csv")
    today = dt.date.today().isoformat()
    for name, df in [("pairs", pairs), ("momentum", mom)]:
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols, "
              f"{df.index.min().date()} -> {df.index.max().date()} (downloaded {today})")


if __name__ == "__main__":
    main()
