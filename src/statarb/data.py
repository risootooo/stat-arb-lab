"""Load the cached real price data shipped in ``data/`` (fully offline)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_pairs() -> pd.DataFrame:
    """Daily adjusted closes for KO, PEP, XOM, CVX (Nov 2021 - Nov 2026)."""
    return pd.read_csv(DATA_DIR / "pairs_prices.csv",
                       parse_dates=["Date"], index_col="Date")


def load_momentum() -> pd.DataFrame:
    """Daily adjusted closes for the momentum universe (Jan 2018 - Nov 2026)."""
    return pd.read_csv(DATA_DIR / "momentum_prices.csv",
                       parse_dates=["Date"], index_col="Date")
