"""Library-style use: analyse one pair end-to-end from the cached data.

    python examples/run_pair.py            # PEP vs KO (Case A's pair)
    python examples/run_pair.py XOM CVX
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from statarb import (backtest_pair, engle_granger, half_life,  # noqa: E402
                     hurst_exponent, kalman_hedge_ratio)
from statarb.data import load_pairs                          # noqa: E402
from statarb.metrics import format_stats                     # noqa: E402


def main() -> None:
    y_name, x_name = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("PEP", "KO")
    px = load_pairs()
    y, x = px[y_name], px[x_name]
    eg = engle_granger(y, x)
    print(f"{y_name} ~ {x_name}: alpha {eg.alpha:.2f}, beta {eg.beta:.3f}, "
          f"EG stat {eg.adf_stat:.3f} (p ~ {eg.pvalue:.3f}), "
          f"cointegrated@5% {eg.cointegrated_5pct}")
    print(f"half-life {half_life(eg.spread):.1f} days, "
          f"Hurst(spread) {hurst_exponent(eg.spread):.3f}")
    kf = kalman_hedge_ratio(y, x)
    print(f"Kalman beta: {kf.beta.iloc[60]:.3f} -> {kf.beta.iloc[-1]:.3f} "
          f"(static OLS beta {eg.beta:.3f})")
    bt = backtest_pair(y, x, eg.alpha, eg.beta)
    print(format_stats(bt.stats))
    print(f"trades {bt.stats['n_trades']}, win rate {bt.stats['win_rate']:.0%}, "
          f"exposure {bt.exposure:.0%}, costs ${bt.total_costs:,.0f}")


if __name__ == "__main__":
    main()
