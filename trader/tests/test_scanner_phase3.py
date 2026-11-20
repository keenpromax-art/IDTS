"""Deterministic tests for Phase 3 multi-timeframe analysis."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.core.models import BacktestStats, Regime, Side, Signal
from trader.scanner.timeframes import (
    MTFResult, TF_ORDER, MultiTimeframeAnalyzer, available_frames,
    resample_candles,
)

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def frame(n=120, start=100.0, drift=0.08, vol=1000.0, freq="5min"):
    idx = pd.date_range(datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
                        periods=n, freq=freq)
    close = start + drift * np.arange(n)
    spread = 0.5 + 0.01 * np.arange(n)
    return pd.DataFrame({
        "open": close - drift, "high": close + spread,
        "low": close - spread, "close": close, "volume": np.full(n, vol),
    }, index=idx)


def test_supported_timeframes_and_resampling_correctness():
    assert list(TF_ORDER) == ["1m", "5m", "15m", "30m", "1h", "1d"]
    df = frame(n=120, freq="5min", vol=10.0)
    h1 = resample_candles(df, "1h")
    assert h1 is not None and len(h1) == 10
    assert h1["open"].iloc[0] == df["open"].iloc[0]
    assert h1["close"].iloc[0] == df["close"].iloc[11]
    assert h1["volume"].iloc[0] == pytest.approx(120.0)
    assert resample_candles(df, "1m") is None


def test_available_frames_derives_higher_and_keeps_supplied_lower():
    frames = {"5m": frame(n=120), "1m": frame(n=600, freq="1min")}
    out = available_frames(frames, "5m", ["1m", "5m", "15m", "1h", "1d"])
    assert set(out) == {"1m", "5m", "15m", "1h", "1d"}
    assert len(out["1m"]) == 600


def test_snapshot_extracts_one_vote_per_correlated_family():
    an = MultiTimeframeAnalyzer()
    snap = an.snapshot("5m", frame(n=120, drift=0.08))
    assert set(snap.family_votes) == {"trend", "momentum", "volatility", "volume"}
    assert snap.trend_dir == 1
    assert snap.momentum_dir == 1


def test_analyzer_detects_trend_alignment_and_volatility_expansion():
    an = MultiTimeframeAnalyzer({"enabled": True, "timeframes": ["5m", "15m", "1h"]})
    result = an.analyze("A", {"5m": frame(n=120, drift=0.08)})
    assert result.direction == 1
    assert result.alignment == pytest.approx(1.0)
    assert result.momentum_confirmed
    assert len(result.per_timeframe) == 3
    assert result.families_used


def test_conflicts_are_reported_not_silently_averaged():
    an = MultiTimeframeAnalyzer({"enabled": True, "timeframes": ["5m", "1h"]})
    up = frame(n=120, drift=0.08)
    down = frame(n=120, drift=-0.08, start=200)
    result = an.analyze("A", {"5m": up, "1h": down})
    assert result.conflicts
    assert any("5m" in c and "1h" in c for c in result.conflicts)


def test_signal_against_higher_trend_is_a_conflict():
    an = MultiTimeframeAnalyzer({"enabled": True, "timeframes": ["5m", "1h"]})
    result = an.analyze("A", {"5m": frame(n=120, drift=0.08),
                              "1h": frame(n=120, freq="1h", drift=0.08)},
                        signal_side=Side.SHORT)
    assert any("signal -1 against 1h trend" in c for c in result.conflicts)


def test_families_are_capped_and_independent_of_indicator_count():
    an = MultiTimeframeAnalyzer({"enabled": True, "timeframes": ["5m"]})
    result = an.analyze("A", {"5m": frame(n=120, drift=0.08)})
    assert len(result.families_used) <= 4
    assert abs(result.score) <= 1.0


def test_disabled_analyzer_returns_empty_result():
    an = MultiTimeframeAnalyzer({"enabled": False})
    result = an.analyze("A", {"5m": frame()})
    assert not an.active and result.per_timeframe == {}


def _engine(tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["live_quotes"] = False
    cfg["data"]["mtf"] = {"enabled": True, "timeframes": ["5m", "15m"],
                          "block_on_conflict": True}
    eng = Engine(cfg, markets, {m: ["A"] for m in markets}, "paper", False)
    eng.fx_last_update = 1e12
    return eng


def test_engine_attaches_mtf_result_to_scan_row(tmp_path):
    eng = _engine(tmp_path)
    sig = Signal(symbol="A", market="India", strategy="s", side=Side.LONG,
                 entry=100, stop=98, target=104, confidence=0.9)
    bt = BacktestStats(strategy="s", symbol="A", trades=20, win_rate=0.6,
                       profit_factor=2, expectancy_r=0.5, eligible=True)
    with patch.object(MarketClock, "status", return_value={"status": "OPEN", "next_open": None}), \
         patch.object(eng.providers["India"], "batch_download",
                      return_value={"A": frame(n=120)}), \
         patch.object(eng.selector, "evaluate",
                      return_value=("s", 0.9, bt, Regime(symbol="A"), sig)), \
         patch.object(eng, "_trade_candidates"):
        rows, candidates, _, stats = eng._scan_market("India")
    assert rows[0]["mtf"] is not None
    assert "families_used" in rows[0]["mtf"]


def test_engine_blocks_candidate_on_timeframe_conflict(tmp_path):
    eng = _engine(tmp_path)
    sig = Signal(symbol="A", market="India", strategy="s", side=Side.LONG,
                 entry=100, stop=98, target=104, confidence=0.9)
    bt = BacktestStats(strategy="s", symbol="A", trades=20, win_rate=0.6,
                       profit_factor=2, expectancy_r=0.5, eligible=True)
    conflict = MTFResult(direction=-1, conflicts=["5m +1 vs 1h -1"])
    with patch.object(MarketClock, "status", return_value={"status": "OPEN", "next_open": None}), \
         patch.object(eng.providers["India"], "batch_download",
                      return_value={"A": frame(n=120)}), \
         patch.object(eng.selector, "evaluate",
                      return_value=("s", 0.9, bt, Regime(symbol="A"), sig)), \
         patch.object(eng.mtf, "analyze", return_value=conflict), \
         patch.object(eng, "_trade_candidates"):
        _, candidates, _, _ = eng._scan_market("India")
    assert candidates == []
