"""Deterministic tests for Phase 5 ranking and realistic execution."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trader.core.models import BacktestStats, Side, Signal
from trader.core.selector import Selector
from trader.scanner.ranking import (
    ExecutionModel, QualityRanker, RankingContext, compute_performance_metrics,
    walk_forward_folds,
)

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))


def test_performance_metrics_hand_computed():
    m = compute_performance_metrics([1.0, -0.5, 2.0, -1.0])
    assert m.trades == 4 and m.sample_size == 4
    assert m.expectancy_r == pytest.approx(0.375)
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(3.0 / 1.5)
    assert m.max_drawdown > 0
    assert 0.0 <= m.win_rate_lower_bound <= 0.5


def test_net_expectancy_subtracts_costs():
    m = compute_performance_metrics([1.0, 1.0], costs_r=[0.2, 0.2])
    assert m.net_expectancy_r == pytest.approx(0.8)
    assert m.expectancy_r == pytest.approx(1.0)


def test_sharpe_sortino_and_empty_input():
    m = compute_performance_metrics([1, 1, 1, 1])
    assert m.sortino > 0 and m.sharpe > 0 and m.max_drawdown == 0
    empty = compute_performance_metrics([])
    assert empty.trades == 0 and empty.sharpe == 0


def test_walk_forward_folds_are_chronological_and_non_overlapping():
    folds = walk_forward_folds(600, train_bars=120, test_bars=60, folds=3)
    assert len(folds) == 3
    for a, b, c, d in folds:
        assert a < b <= c < d
        assert b <= c
    assert folds[0][2] < folds[1][2]
    assert walk_forward_folds(10, 120, 60, 3) == []


def test_execution_model_applies_costs_and_direction():
    m = ExecutionModel(mode="next_bar", slippage_bps=5, spread_bps=2, commission_bps=3)
    assert m.cost_fraction == pytest.approx(10 / 10000)
    assert m.entry_price(100, True) > 100
    assert m.entry_price(100, False) < 100
    assert m.exit_price(100, True) < 100
    assert m.exit_price(100, False) > 100


class OneShot:
    name = "one_shot"
    kind = "trend"

    def __init__(self, fire_bar=10, stop=99.0, target=102.0):
        self.fire_bar = fire_bar
        self.stop = stop
        self.target = target

    def generate_from(self, ctx):
        if ctx._bar != self.fire_bar:
            return None
        px = float(ctx.px)
        return Signal(symbol="S", market="", strategy=self.name, side=Side.LONG,
                      entry=px, stop=self.stop, target=self.target, confidence=0.9)


def gap_df(n=120, entry_bar=10):
    idx = pd.date_range(datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
                        periods=n, freq="5min")
    close = np.full(n, 100.0)
    opn = np.full(n, 100.0)
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    # bar 10 closes at 100; bar 11 opens higher and runs to the target
    opn[entry_bar + 1] = 101.0
    high[entry_bar + 1] = 105.0
    low[entry_bar + 1] = 100.5
    close[entry_bar + 1] = 104.0
    return pd.DataFrame({"open": opn, "high": high, "low": low,
                         "close": close, "volume": np.full(n, 1000.0)}, index=idx)


def test_backtest_uses_next_bar_open_not_signal_close():
    cfg = copy.deepcopy(settings)
    cfg["selector"]["execution_model"]["mode"] = "next_bar"
    cfg["selector"]["execution_model"]["spread_bps"] = 0
    cfg["selector"]["execution_model"]["slippage_bps"] = 0
    cfg["selector"]["execution_model"]["commission_bps"] = 0
    sel = Selector(cfg)
    stats = sel.backtest(OneShot(), gap_df(), "S")
    assert stats.trades == 1
    # Filled at bar 11's 101 open (not the 100 signal close), the target at
    # 102 yields 1R of 2 points risk = +0.5R.
    assert stats.expectancy_r == pytest.approx(0.5)
    assert stats.net_expectancy_r == stats.expectancy_r


def test_gap_through_stop_invalidates_the_trade():
    df = gap_df()
    df.iloc[11, df.columns.get_loc("open")] = 95.0
    df.iloc[11, df.columns.get_loc("high")] = 95.5
    df.iloc[11, df.columns.get_loc("low")] = 94.5
    df.iloc[11, df.columns.get_loc("close")] = 95.0
    stats = Selector(copy.deepcopy(settings)).backtest(OneShot(), df, "S")
    assert stats.trades == 0


def test_backtest_mode_change_produces_different_result():
    cfg = copy.deepcopy(settings)
    cfg["selector"]["execution_model"] = {"mode": "same_bar", "slippage_bps": 0,
                                          "spread_bps": 0, "commission_bps": 0}
    legacy = Selector(cfg).backtest(OneShot(), gap_df(), "S")
    cfg["selector"]["execution_model"]["mode"] = "next_bar"
    realistic = Selector(cfg).backtest(OneShot(), gap_df(), "S")
    assert legacy.trades == realistic.trades == 1
    assert legacy.expectancy_r != realistic.expectancy_r


def test_backtest_populates_enhanced_metrics():
    stats = Selector(copy.deepcopy(settings)).backtest(OneShot(), gap_df(), "S")
    assert stats.sharpe != 0 or stats.sortino != 0
    assert stats.sample_size == stats.trades == 1
    assert 0 <= stats.win_rate_lower_bound <= 1


def test_score_unchanged_without_context_backward_compatible():
    cfg = copy.deepcopy(settings)
    sel = Selector(cfg)
    stats = BacktestStats(strategy="ema_cross", symbol="S", trades=20, wins=10,
                          losses=10, win_rate=0.5, profit_factor=1.0, expectancy_r=0.0)
    from datetime import datetime, timezone
    from trader.core.models import Regime
    regime = Regime(symbol="S")
    base = sel.score(stats, regime, 0.6)
    assert base == pytest.approx(0.40 * 0.35 + 0.20 * 0.5 + 0.15 * 0.5
                                 + 0.15 * sel.regime_fit("ema_cross", regime) + 0.10 * 0.6)


def test_enhanced_ranking_uses_context_when_enabled():
    cfg = copy.deepcopy(settings)
    cfg["selector"]["ranking"]["enabled"] = True
    cfg["selector"]["ranking"]["blend"] = 1.0
    sel = Selector(cfg)
    stats = BacktestStats(strategy="ema_cross", symbol="S", trades=40, wins=26, losses=14,
                          win_rate=0.65, profit_factor=2.0, expectancy_r=0.5,
                          net_expectancy_r=0.4, sharpe=1.5, sortino=2.0,
                          max_drawdown=2.0, sample_size=40)
    from trader.core.models import Regime
    regime = Regime(symbol="S")
    good = RankingContext(relative_strength=1.5, liquidity_quality=0.9,
                          data_freshness=1.0, walk_forward_score=0.3)
    bad = RankingContext(relative_strength=-1.5, liquidity_quality=0.1,
                         data_freshness=0.1, walk_forward_score=-0.3)
    assert sel.score(stats, regime, 0.8, context=good) > sel.score(stats, regime, 0.8, context=bad)


def test_quality_ranker_weights_are_configurable():
    cfg = {"enabled": True, "weights": {"net_expectancy": 1.0}}
    r = QualityRanker(cfg)
    stats = BacktestStats(strategy="s", symbol="S", net_expectancy_r=0.6, trades=10)
    hi = r.enhanced_score(stats, 1.0, 1.0, RankingContext())
    stats.net_expectancy_r = -0.3
    lo = r.enhanced_score(stats, 1.0, 1.0, RankingContext())
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.0)
