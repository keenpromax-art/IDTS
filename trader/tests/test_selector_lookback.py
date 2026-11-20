"""Scoring lookback window.

Scoring gates (expectancy, profit factor, win rate) grade setups over a
longer lookback (``scoring_lookback_days``, default 20) than live signal
generation, which keeps the short ``backtest_sessions`` window.
"""
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from trader.core.selector import Selector
from trader.core.models import Side

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))


def make_sel(selector_overrides=None):
    cfg = dict(settings)
    sel = dict(settings["selector"])
    if selector_overrides:
        sel.update(selector_overrides)
    cfg["selector"] = sel
    return Selector(cfg)


def make_df(n_days=30, bars_per_day=75, seed=1):
    """Frame with exactly `bars_per_day` bars on every date."""
    rng = np.random.default_rng(seed)
    price = 100.0
    closes = []
    for _ in range(n_days * bars_per_day):
        price += rng.normal(0.0, 0.3)
        closes.append(price)
    closes = np.array(closes)
    index = []
    for day in pd.date_range("2026-03-02", periods=n_days, freq="B"):
        index.extend(pd.date_range(day, periods=bars_per_day, freq="5min"))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.25,
            "low": closes - 0.25,
            "close": closes,
            "volume": rng.integers(1000, 20000, len(closes)),
        },
        index=pd.DatetimeIndex(index),
    )


class Fake:
    name = "fake"
    kind = "trend"

    def generate_from(self, ctx):
        return None


# ------------------------------------------------------------------ config

def test_scoring_lookback_defaults_to_sessions_when_not_configured():
    cfg = dict(settings)
    sel = {k: v for k, v in settings["selector"].items()
           if k != "scoring_lookback_days"}
    cfg["selector"] = sel
    s = Selector(cfg)
    assert s.scoring_lookback_days == s.sessions == 5


def test_scoring_lookback_uses_configured_days():
    s = make_sel({"scoring_lookback_days": 20})
    assert s.scoring_lookback_days == 20
    # live path keeps the short window
    assert s.sessions == 5


# ------------------------------------------------------------------ window sizing

def test_backtest_window_longer_with_scoring_lookback():
    s = make_sel({"scoring_lookback_days": 20, "backtest_max_bars": 10000})
    df = make_df(n_days=140)   # 10500 bars, above the max_bars cap
    live = s._backtest_window(df)                    # backtest_sessions = 5
    scoring = s._backtest_window(df, lookback_days=20)
    assert len(live) == 5 * 75
    assert len(scoring) == 20 * 75
    assert len(scoring) == 4 * len(live)


def test_live_signal_window_unchanged():
    s = make_sel({"scoring_lookback_days": 20, "backtest_max_bars": 10000})
    df = make_df(n_days=140)
    # backtest() with no lookback_days still uses the 5-session live window
    calls = []
    orig = s._backtest_window

    def spy(frame, lookback_days=None):
        calls.append(lookback_days)
        return orig(frame, lookback_days=lookback_days)

    s._backtest_window = spy
    s.backtest(Fake(), df, "S")
    s._backtest_window = orig
    assert calls[-1] is None
    assert len(s._backtest_window(df)) == 5 * 75
