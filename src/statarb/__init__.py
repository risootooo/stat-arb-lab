"""StatArb Lab: pairs trading and dual-momentum research on real data."""

from .backtest import backtest_pair
from .kalman import kalman_hedge_ratio
from .momentum import backtest_momentum, momentum_signal, select_weights
from .pairs import (adf_test, engle_granger, half_life, hedge_ratio,
                    hurst_exponent, rolling_zscore)

__all__ = [
    "adf_test", "engle_granger", "half_life", "hedge_ratio", "hurst_exponent",
    "rolling_zscore", "kalman_hedge_ratio", "backtest_pair",
    "backtest_momentum", "momentum_signal", "select_weights",
]
__version__ = "1.0.0"
