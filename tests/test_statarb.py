"""Tests for StatArb Lab.  Statistical tests are seeded so they are
deterministic; each asserts the *direction* the theory predicts (reject a
unit root for a mean-reverting process, fail to reject for a random walk),
not a magic number copied from another library."""
import numpy as np
import pandas as pd
import pytest

from statarb import (adf_test, backtest_pair, engle_granger, half_life,
                     hedge_ratio, hurst_exponent, kalman_hedge_ratio,
                     rolling_zscore)
from statarb.momentum import momentum_signal, monthly_prices, select_weights


def ou_process(n=800, phi=0.93, sigma=1.0, seed=3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return pd.Series(x)


def random_walk(n=800, seed=11):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)))


# ---------------------------------------------------------------- ADF / EG

def test_adf_rejects_unit_root_for_mean_reverting():
    r = adf_test(ou_process())
    assert r.stat < -2.86
    assert r.pvalue < 0.05
    assert r.rejects_unit_root_5pct


def test_adf_does_not_reject_random_walk():
    r = adf_test(random_walk())
    assert r.stat > -2.86
    assert not r.rejects_unit_root_5pct


def test_adf_rejects_for_differenced_random_walk():
    r = adf_test(random_walk().diff().dropna())
    assert r.stat < -3.43  # significant at 1%


def test_hedge_ratio_recovers_known_line():
    x = random_walk()
    y = 5.0 + 2.0 * x
    alpha, beta, spread = hedge_ratio(y, x)
    assert alpha == pytest.approx(5.0, abs=1e-6)
    assert beta == pytest.approx(2.0, abs=1e-6)
    assert float(spread.abs().max()) < 1e-6


def test_engle_granger_detects_planted_cointegration():
    rng = np.random.default_rng(5)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 900)) + 100)
    spread = ou_process(900, phi=0.90, seed=9)
    y = 3.0 + 1.5 * x + spread
    eg = engle_granger(y, x)
    assert eg.beta == pytest.approx(1.5, abs=0.15)
    assert eg.cointegrated_5pct


def test_engle_granger_two_independent_walks_not_cointegrated():
    eg = engle_granger(random_walk(seed=21), random_walk(seed=22))
    assert not eg.cointegrated_5pct


# ------------------------------------------------------- half-life & Hurst

def test_half_life_recovers_ou_parameter():
    phi = 0.93
    expected = -np.log(2) / (phi - 1)  # ≈ 9.9 observations
    hl = half_life(ou_process(n=4000, phi=phi, seed=13))
    assert hl == pytest.approx(expected, rel=0.45)


def test_half_life_of_random_walk_is_long():
    assert half_life(random_walk(n=2000)) > 60


def test_hurst_mean_reverting_below_half():
    assert hurst_exponent(ou_process(n=2000, phi=0.90, seed=17)) < 0.5


def test_hurst_random_walk_near_half():
    h = hurst_exponent(random_walk(n=2000))
    assert 0.35 < h < 0.68


# ------------------------------------------------------------------ Kalman

def test_kalman_beta_converges_to_true_beta():
    rng = np.random.default_rng(31)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 600)) + 50)
    y = 2.0 + 2.0 * x + pd.Series(rng.normal(0, 0.3, 600))
    kf = kalman_hedge_ratio(y, x, delta=1e-3)
    assert kf.beta.iloc[-1] == pytest.approx(2.0, abs=0.25)


def test_kalman_tracks_a_regime_shift_in_beta():
    rng = np.random.default_rng(37)
    x = pd.Series(np.cumsum(rng.normal(0, 1, 800)) + 50)
    beta_true = np.where(np.arange(800) < 400, 1.0, 2.5)
    y = pd.Series(1.0 + beta_true * x.to_numpy() + rng.normal(0, 0.3, 800))
    kf = kalman_hedge_ratio(y, x, delta=1e-3)
    assert abs(kf.beta.iloc[-1] - 2.5) < abs(kf.beta.iloc[-1] - 1.0)


# ---------------------------------------------------------------- backtest

def _toy_pair(n=500, seed=41):
    rng = np.random.default_rng(seed)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 0.8, n)),
                  index=pd.bdate_range("2023-01-02", periods=n))
    spread = ou_process(n, phi=0.92, sigma=2.0, seed=seed)
    spread.index = x.index
    y = 10.0 + 1.2 * x + spread
    return y, x, 10.0, 1.2


def test_backtest_accounting_identity():
    y, x, a, b = _toy_pair()
    bt = backtest_pair(y, x, a, b, cost_bps=5.0)
    assert len(bt.trades) >= 3
    assert bt.equity.iloc[-1] == pytest.approx(
        100_000.0 + bt.trades["net_pnl"].sum(), rel=1e-9)


def test_backtest_costs_reduce_returns():
    y, x, a, b = _toy_pair()
    free = backtest_pair(y, x, a, b, cost_bps=0.0)
    pricey = backtest_pair(y, x, a, b, cost_bps=50.0)
    assert pricey.equity.iloc[-1] < free.equity.iloc[-1]
    assert pricey.total_costs > 0


def test_backtest_flat_spread_no_trades():
    idx = pd.bdate_range("2023-01-02", periods=300)
    x = pd.Series(np.linspace(50, 80, 300), index=idx)
    y = 4.0 + 2.0 * x
    bt = backtest_pair(y, x, 4.0, 2.0)
    assert bt.stats["n_trades"] == 0
    assert bt.equity.iloc[-1] == pytest.approx(100_000.0)


def test_rolling_zscore_mean_near_zero_in_window():
    s = ou_process(2000)
    z = rolling_zscore(s, 100).dropna()
    assert abs(float(z.mean())) < 0.35


# ---------------------------------------------------------------- momentum

def _toy_monthly():
    months = pd.date_range("2020-01-31", periods=26, freq="M")
    up = 100 * (1.012 ** np.arange(26))        # +1.2%/mo steady
    down = 100 * (0.99 ** np.arange(26))        # -1.0%/mo steady
    flat = np.full(26, 100.0)
    return pd.DataFrame({"UP": up, "DOWN": down, "FLAT": flat}, index=months)


def test_momentum_signal_signs_on_toy_series():
    mom = momentum_signal(_toy_monthly())
    last = mom.iloc[-1]
    assert last["UP"] > 0.10
    assert last["DOWN"] < -0.05
    assert abs(last["FLAT"]) < 1e-12


def test_select_weights_picks_positive_momentum():
    row = pd.Series({"UP": 0.20, "DOWN": -0.10, "FLAT": 0.0, "TLT": 0.01})
    w = select_weights(row, ["UP", "DOWN", "FLAT"], "TLT", top_k=2)
    assert w["UP"] == pytest.approx(0.5)
    assert w["TLT"] == pytest.approx(0.5)  # unfilled slot parks in defensive
    assert sum(w.values()) == pytest.approx(1.0)


def test_select_weights_all_negative_goes_fully_defensive():
    row = pd.Series({"UP": -0.2, "DOWN": -0.1, "FLAT": -0.05, "TLT": 0.0})
    w = select_weights(row, ["UP", "DOWN", "FLAT"], "TLT", top_k=3)
    assert w == {"TLT": 1.0}


def test_momentum_backtest_on_cached_real_data():
    from statarb.data import load_momentum
    from statarb.momentum import backtest_momentum
    res = backtest_momentum(load_momentum())
    assert len(res.equity) > 2000
    assert len(res.weights_history) > 60
    row_sums = res.weights_history.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)
    assert res.stats["n_trades"] if "n_trades" in res.stats else True


def test_monthly_prices_picks_last_trading_day():
    idx = pd.bdate_range("2024-01-01", "2024-03-28")
    df = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
    m = monthly_prices(df)
    assert str(m.index[0].date()) == "2024-01-31"
    assert str(m.index[-1].date()) == "2024-03-28"
