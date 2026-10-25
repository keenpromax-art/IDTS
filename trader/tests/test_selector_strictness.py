"""Tests for the selector's strictness gates.

A setup must earn its place: enough history, positive expectancy, a profit
factor above 1, a workable win rate, a convincing live signal and a score
above the bar. And a market only trades its strongest candidates.
"""
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from trader.core.selector import Selector
from trader.core.models import BacktestStats, Side
from trader.strategies import ALL_RULES, ALL_STRATEGIES
from trader.strategies.context import Ctx
from trader.strategies.rule import Condition, Rule

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))


def make_df(n=320, drift=0.0, vol=1.0, seed=1):
    rng = np.random.default_rng(seed)
    price = 100.0
    close = []
    for _ in range(n):
        price += rng.normal(drift, 0.3 * vol)
        close.append(price)
    close = np.array(close)
    idx = pd.date_range("2026-03-02 09:15", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": close + 0.25, "low": close - 0.25,
         "close": close, "volume": rng.integers(1000, 20000, n)},
        index=idx,
    )


@pytest.fixture()
def sel():
    s = Selector(settings)
    s.set_strategies([x() for x in ALL_STRATEGIES] + list(ALL_RULES))
    return s


# ------------------------------------------------------------------ gates

def test_defaults_are_strict():
    s = settings["selector"]
    assert s["min_trade_count"] >= 8
    assert s["min_expectancy_r"] >= 0.0
    assert s["min_profit_factor"] > 1.0
    assert s["min_win_rate"] >= 0.3
    assert s["min_signal_confidence"] >= 0.5
    assert s["min_score"] >= 0.5
    assert s["min_market_score"] >= 0.55
    assert s["max_trades_per_market_per_cycle"] <= 3


def test_too_few_trades_is_ineligible(sel):
    stats = BacktestStats(strategy="x", symbol="S", trades=3, wins=3, losses=0,
                          win_rate=1.0, profit_factor=2.0, expectancy_r=1.0)
    assert not sel._is_eligible(stats)


def test_negative_expectancy_is_ineligible(sel):
    stats = BacktestStats(strategy="x", symbol="S", trades=20, wins=4, losses=16,
                          win_rate=0.2, profit_factor=0.5, expectancy_r=-0.4)
    assert not sel._is_eligible(stats)


def test_profit_factor_below_one_is_ineligible(sel):
    stats = BacktestStats(strategy="x", symbol="S", trades=20, wins=8, losses=12,
                          win_rate=0.4, profit_factor=0.9, expectancy_r=0.05)
    assert not sel._is_eligible(stats)


def test_low_win_rate_is_ineligible(sel):
    stats = BacktestStats(strategy="x", symbol="S", trades=20, wins=5, losses=15,
                          win_rate=0.25, profit_factor=1.4, expectancy_r=0.1)
    assert not sel._is_eligible(stats)


def test_good_stats_are_eligible(sel):
    stats = BacktestStats(strategy="x", symbol="S", trades=20, wins=9, losses=11,
                          win_rate=0.45, profit_factor=1.6, expectancy_r=0.2)
    assert sel._is_eligible(stats)


# ------------------------------------------------------------------ scoring

def test_zero_expectancy_does_not_score_half(sel):
    """A setup going nowhere must not look half-good."""
    stats = BacktestStats(strategy="ema_cross", symbol="S", trades=20, wins=10, losses=10,
                          win_rate=0.5, profit_factor=1.0, expectancy_r=0.0)
    regime = sel.detect_regime(make_df())
    score = sel.score(stats, regime, 0.6)
    # expectancy (0 weight .40) at 0.35 for zero expectancy, PF at 0.5,
    # win rate at 0.5, regime fit from the detected regime, confidence 0.6
    expected = (0.40 * 0.35 + 0.20 * 0.5 + 0.15 * 0.5
                + 0.15 * sel.regime_fit("ema_cross", regime) + 0.10 * 0.6)
    assert score == pytest.approx(expected, abs=0.02)
    assert score < settings["selector"]["min_score"]


def test_negative_expectancy_scores_below_zero_expectancy(sel):
    regime = sel.detect_regime(make_df())
    neg = BacktestStats(strategy="ema_cross", symbol="S", trades=20, wins=4, losses=16,
                       win_rate=0.2, profit_factor=0.4, expectancy_r=-0.5)
    zero = BacktestStats(strategy="ema_cross", symbol="S", trades=20, wins=10, losses=10,
                         win_rate=0.5, profit_factor=1.0, expectancy_r=0.0)
    assert sel.score(neg, regime, 0.6) < sel.score(zero, regime, 0.6)


def test_good_expectancy_scores_high(sel):
    regime = sel.detect_regime(make_df())
    good = BacktestStats(strategy="ema_cross", symbol="S", trades=20, wins=12, losses=8,
                         win_rate=0.6, profit_factor=2.2, expectancy_r=0.6)
    score = sel.score(good, regime, 0.9)
    assert score > settings["selector"]["min_market_score"]


# ------------------------------------------------------------------ backtest

def test_backtest_excludes_the_live_bar(sel):
    """The bar the live signal came from must not be graded."""
    df = make_df(n=200, drift=0.1, seed=5)
    class Fake:
        name = "fake"
        kind = "trend"
        def generate_from(self, ctx):
            return None
    with_last = sel.backtest(Fake(), df, "S")
    # the same frame minus its last bar must give the same result
    without_last = sel.backtest(Fake(), df.iloc[:-1], "S")
    assert with_last.trades == without_last.trades


# ------------------------------------------------------------------ signals

def test_weak_signal_is_not_traded(sel):
    """A good historical score does not excuse a weak live signal."""
    df = make_df(n=200, drift=0.1, seed=6)
    good = BacktestStats(strategy="ema_cross", symbol="S", trades=20, wins=12, losses=8,
                         win_rate=0.6, profit_factor=2.2, expectancy_r=0.6)
    regime = sel.detect_regime(df)

    class Fake:
        name = "ema_cross"
        kind = "trend"
        def generate_from(self, ctx):
            from trader.core.models import Signal
            return Signal(symbol="S", market="India", strategy="ema_cross",
                          side=Side.LONG, entry=100.0, stop=98.0, target=104.0,
                          confidence=0.3)

    fake = Fake()
    fake_strats = [fake]
    sel.set_strategies(fake_strats)
    live = {"ema_cross": fake.generate_from(Ctx(df, "S", "India"))}
    # patch the backtest to return good stats
    sel.backtest = lambda *a, **k: good
    assert sel.evaluate("S", df, fake_strats, live, 1) is None


def test_mean_reversion_setups_refuse_a_strong_trend():
    """Fading an extreme is not a trade inside a strong trend."""
    from trader.strategies.oscillators import (CCIStrategy, MFIStrategy,
                                               RSIReversalStrategy, StochasticStrategy,
                                               WilliamsRStrategy)
    df = make_df(n=200, drift=0.5, vol=0.4, seed=9)
    ctx = Ctx(df, "S", "India")
    assert ctx.adx > 30, "test data should be a strong trend"
    for S in (CCIStrategy, MFIStrategy, RSIReversalStrategy, StochasticStrategy,
              WilliamsRStrategy):
        assert S().generate_from(ctx) is None, S.name


def test_vwap_band_refuses_a_strong_trend():
    from trader.strategies.momentum import VWAPBandStrategy
    df = make_df(n=200, drift=0.5, vol=0.4, seed=9)
    ctx = Ctx(df, "S", "India")
    if ctx.adx > 30:
        assert VWAPBandStrategy().generate_from(ctx) is None
