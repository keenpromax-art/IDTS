"""Deterministic tests for the Phase 2 pre-screening layer."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from unittest.mock import patch

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.scanner.prescreen import PreScreener, ScreenConfig

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def frame(n=60, close=100.0, vol=1000.0, last_vol=None, last_open=None, last_high=None,
          last_low=None, last_close=None, trend=False):
    idx = pd.date_range(datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
                        - timedelta(minutes=5 * (n - 1)), periods=n, freq="5min")
    closes = np.linspace(close - 5, close, n) if trend else np.full(n, close)
    highs = closes + 0.5
    lows = closes - 0.5
    opens = np.r_[closes[0], closes[:-1]]
    volumes = np.full(n, vol)
    if last_vol is not None:
        volumes[-1] = last_vol
    if last_open is not None:
        opens[-1] = last_open
    if last_high is not None:
        highs[-1] = last_high
    if last_low is not None:
        lows[-1] = last_low
    if last_close is not None:
        closes[-1] = last_close
        highs[-1] = max(highs[-1], last_close)
        lows[-1] = min(lows[-1], last_close)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)


def test_disabled_screener_passes_everything():
    s = PreScreener({"enabled": False, "min_relative_volume": 999})
    assert s.screen(frame()).passed
    assert not s.active


def test_relative_volume_filter_detects_spike():
    df = frame(last_vol=5000)
    f = PreScreener({}).features(df)
    assert f["relative_volume"] == pytest.approx(5.0)
    assert PreScreener({"enabled": True, "min_relative_volume": 3}).screen(df).passed
    assert not PreScreener({"enabled": True, "min_relative_volume": 10}).screen(df).passed


def test_volume_acceleration_and_traded_value_filters():
    df = frame(last_vol=5000)
    assert PreScreener({"enabled": True, "min_volume_acceleration": 3}).screen(df).passed
    assert not PreScreener({"enabled": True, "min_traded_value": 1e9}).screen(df).passed
    assert PreScreener({"enabled": True, "min_traded_value": 1e4}).screen(df).passed


def test_pct_move_and_gap_filters():
    df = frame(close=100, last_open=105, last_high=106, last_low=104)
    f = PreScreener({}).features(df)
    assert f["abs_gap_pct"] == pytest.approx(5.0)
    assert PreScreener({"enabled": True, "min_abs_gap_pct": 4}).screen(df).passed
    trend = frame(close=100, trend=True)
    assert PreScreener({"enabled": True, "min_abs_pct_move": 1.0}).screen(trend).passed
    assert not PreScreener({"enabled": True, "max_abs_pct_move": 1}).screen(trend).passed


def test_atr_expansion_vwap_range_and_trend_persistence_filters():
    df = frame(last_high=120, trend=True)
    f = PreScreener({}).features(df)
    assert np.isfinite(f["atr_expansion"])
    assert np.isfinite(f["vwap_deviation_atr"])
    assert 0.0 <= f["trend_persistence"] <= 1.0
    assert f["range_pct"] > 10
    res = PreScreener({"enabled": True, "max_range_pct": 5}).screen(df)
    assert not res.passed and any("range_pct" in r for r in res.reasons)


def test_orb_and_combine_semantics():
    df = frame(last_close=102, last_high=102)
    assert PreScreener({"enabled": True, "orb_breakout": True}).screen(df).passed
    weak = frame()
    r_all = PreScreener({"enabled": True, "min_relative_volume": 50,
                         "min_abs_pct_move": 50, "combine": "all"}).screen(weak)
    r_any = PreScreener({"enabled": True, "min_relative_volume": 50,
                         "min_abs_pct_move": 50, "combine": "any"}).screen(weak)
    assert not r_all.passed and not r_any.passed


def test_orb_any_with_one_passing_filter():
    df = frame(last_vol=9000)
    r = PreScreener({"enabled": True, "orb_breakout": True,
                     "min_relative_volume": 5, "combine": "any"}).screen(df)
    assert r.passed


def test_relative_strength_requires_benchmark_data():
    df = frame(close=100, trend=True)
    bench_up = frame(close=100, trend=True)
    bench_flat = frame(close=100)
    up = PreScreener({"enabled": True, "min_relative_strength_pct": 0.0}).screen(df, bench_flat)
    down = PreScreener({"enabled": True, "min_relative_strength_pct": 1.0}).screen(df, bench_up)
    missing = PreScreener({"enabled": True, "min_relative_strength_pct": 1.0}).screen(df)
    assert up.passed
    assert not down.passed
    assert "relative_strength_pct" in missing.unavailable


def test_engine_skips_strategies_for_screened_symbols(tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["live_quotes"] = False
    cfg["data"]["prescreen"] = {"enabled": True, "min_relative_volume": 1000}
    eng = Engine(cfg, markets, {m: ["A"] for m in markets}, "paper", False)
    eng.fx_last_update = 1e12
    df = frame()
    with patch.object(MarketClock, "status", return_value={"status": "OPEN", "next_open": None}), \
         patch.object(eng.providers["India"], "batch_download",
                      return_value={"A": df}), \
         patch.object(eng.strategies[0], "generate_from", side_effect=AssertionError("must not run")):
        rows, cands, pmap, stats = eng._scan_market("India")
    assert stats["prescreened"] == 1
    assert cands == []
    assert rows[0]["signal"] == "NONE"
    assert "pre-screen" in rows[0]["gate"]


def test_engine_runs_strategies_when_filter_passes(tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["live_quotes"] = False
    cfg["data"]["prescreen"] = {"enabled": True, "min_relative_volume": 0.1}
    eng = Engine(cfg, markets, {m: ["A"] for m in markets}, "paper", False)
    eng.fx_last_update = 1e12
    with patch.object(MarketClock, "status", return_value={"status": "OPEN", "next_open": None}), \
         patch.object(eng.providers["India"], "batch_download",
                      return_value={"A": frame()}):
        rows, cands, pmap, stats = eng._scan_market("India")
    assert stats["prescreened"] == 0
    assert stats["scanned"] == 1
