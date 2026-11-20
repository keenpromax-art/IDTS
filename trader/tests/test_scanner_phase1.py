"""Deterministic regression tests for Phase 1 scanner fixes.

No network, no broker, no wall-clock dependence beyond injected clocks.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from zoneinfo import ZoneInfo

from trader.core.engine import Engine, drop_unfinished
from trader.core.market_clock import MarketClock
from trader.core.models import BacktestStats, Side, Signal, Regime
from trader.core.selector import Selector
from trader.data.base import DataProvider
from trader.data.factory import FallbackProvider, ValidatedProvider
from trader.scanner.candles import (
    completed_candles, interval_seconds, is_completed, normalize_candles,
    resolve_execution_price, validate_candles,
)
from trader.scanner.resilience import (
    CircuitBreaker, NegativeCache, ProviderResilience, RateLimiter, RetryPolicy,
)

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def frame(n=60, close=100.0, *, bad_ohlc=False, forming_last=False, volume=1000.0):
    end = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    idx = pd.date_range(end - timedelta(minutes=5 * (n - 1)), periods=n, freq="5min")
    if forming_last:
        idx = idx[:-1].append(pd.DatetimeIndex([datetime(2026, 1, 5, 12, 3, tzinfo=timezone.utc)]))
    df = pd.DataFrame({
        "open": [close] * n, "high": [close + 0.5] * n,
        "low": [close - 0.5] * n, "close": [close] * n,
        "volume": [volume] * n,
    }, index=idx)
    if bad_ohlc:
        df.iloc[-1, df.columns.get_loc("high")] = 1.0
    return df


class Stub(DataProvider):
    def __init__(self, data=None, name="stub", batch=True, available=True):
        self.data = data or {}
        self.name = name
        self.available = available
        self.calls = []
        self._batch = batch

    def get_candles(self, symbol, interval="5m", lookback_days=5):
        self.calls.append(symbol)
        return self.data.get(symbol)

    def batch_download(self, symbols, interval="5m", lookback_days=5):
        self.calls.append("BATCH")
        return {s: self.data.get(s) for s in symbols}


# --- 1.1 Angel / batch fallback ------------------------------------------------

def test_fallback_batch_uses_primary_and_falls_back_per_symbol():
    good = frame()
    primary = Stub({"A": good}, name="angel")
    fallback = Stub({"B": good}, name="yfinance")
    fb = FallbackProvider(primary, fallback, max_workers=2)
    out = fb.batch_download(["A", "B", "C"], "5m", 5)
    assert "BATCH" in primary.calls
    assert out["A"] is not None and out["B"] is not None and out["C"] is None
    assert primary_used(fb) == 1 and fallback_used(fb) == 1


def primary_used(fb):
    return fb.primary_used


def fallback_used(fb):
    return fb.fallback_used


def test_fallback_batch_works_when_primary_has_no_batch_api():
    good = frame()

    class NoBatch:
        name = "angel"
        available = True
        def __init__(self, data):
            self.data = data
            self.calls = []
        def get_candles(self, symbol, interval="5m", lookback_days=5):
            self.calls.append(symbol)
            return self.data.get(symbol)

    primary = NoBatch({"A": good})
    fallback = Stub({"B": good}, name="yfinance")
    fb = FallbackProvider(primary, fallback, max_workers=2)
    out = fb.batch_download(["A", "B"], "5m", 5)
    assert out["A"] is not None and out["B"] is not None
    assert "A" in primary.calls and "B" in fallback.calls


def test_fallback_batch_rejects_invalid_primary_frame_and_uses_fallback():
    fallback = Stub({"A": frame()}, name="yfinance")
    fb = FallbackProvider(Stub({"A": frame(bad_ohlc=True)}, name="angel"), fallback)
    out = fb.batch_download(["A"])
    assert out["A"] is not None
    assert fallback_used(fb) == 1


# --- 1.2 common validation -----------------------------------------------------

def test_validated_provider_rejects_bad_ohlc_from_any_provider():
    vp = ValidatedProvider(Stub({"A": frame(bad_ohlc=True)}))
    assert vp.get_candles("A") is None
    assert vp.batch_download(["A"])["A"] is None


def test_validation_rejects_duplicate_and_unsorted_timestamps():
    df = frame(n=10)
    dup = pd.concat([df.iloc[:3], df.iloc[1:]])
    assert not validate_candles(dup).ok
    assert not validate_candles(df.iloc[::-1]).ok


def test_normalize_rejects_nonfinite_and_missing_columns():
    assert normalize_candles(None) is None
    assert normalize_candles(pd.DataFrame({"open": [1]})) is None
    df = frame(n=5)
    df.loc[df.index[2], "close"] = np.inf
    # invalid rows are dropped, never filled or interpolated
    assert len(normalize_candles(df)) == 4


# --- 1.3 / 1.5 completed candle logic ----------------------------------------

def test_interval_seconds_and_completion():
    assert interval_seconds("1m") == 60 and interval_seconds("1h") == 3600
    ts = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    assert not is_completed(ts, "5m", now=ts + timedelta(minutes=4))
    assert is_completed(ts, "5m", now=ts + timedelta(minutes=5, seconds=10))


def test_completed_candles_keeps_finished_last_bar_and_drops_forming_one():
    finished = frame(n=10)
    forming = frame(n=10, forming_last=True)
    now = datetime(2026, 1, 5, 12, 5, 10, tzinfo=timezone.utc)
    assert len(completed_candles(finished, "5m", now=now)) == 10
    assert len(completed_candles(forming, "5m", now=now)) == 9
    with patch("trader.core.engine.completed_candles", side_effect=completed_candles):
        assert len(drop_unfinished(finished, "5m", now=now)) == 10


def test_single_forming_bar_never_generates_candidate():
    forming = frame(n=1, forming_last=True)
    assert completed_candles(forming, "5m", now=datetime(2026, 1, 5, 12, 1, tzinfo=timezone.utc)).empty


# --- 1.4 executable price separation ------------------------------------------

def test_execution_prefers_fresh_quote_and_rejects_stale():
    sig = 100.0
    now = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    fresh = resolve_execution_price(sig, live_price=101.0, live_ts=now,
                                    max_quote_age_seconds=30, now=now)
    assert fresh.executable and fresh.price == 101.0 and fresh.source == "live"
    stale = resolve_execution_price(sig, live_price=101.0,
                                    live_ts=now - timedelta(minutes=5),
                                    max_quote_age_seconds=30, now=now)
    assert stale.source == "candle" and stale.price == 100.0


def test_execution_rejects_price_that_moved_beyond_limit():
    now = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    out = resolve_execution_price(100.0, live_price=103.0, live_ts=now,
                                  max_deviation_pct=1.0, now=now)
    assert out is None


# --- 1.8 resilience ------------------------------------------------------------

def test_negative_cache_short_lived_and_bounded():
    clock = {"t": 100.0}
    nc = NegativeCache(ttl_seconds=10, clock=lambda: clock["t"])
    nc.record("X")
    assert nc.is_blocked("X")
    clock["t"] = 111.0
    assert not nc.is_blocked("X")


def test_circuit_breaker_opens_recovers_and_half_opens():
    clock = {"t": 0.0}
    cb = CircuitBreaker("src", failure_threshold=2, recovery_seconds=10,
                        clock=lambda: clock["t"])
    assert cb.allow()
    cb.record_failure(); cb.record_failure()
    assert not cb.allow() and cb.state.value == "open"
    clock["t"] = 11.0
    assert cb.allow() and cb.state.value == "half_open"
    cb.record_success()
    assert cb.state.value == "closed"


def test_rate_limiter_spaces_calls():
    clock = {"t": 0.0}
    slept = []
    rl = RateLimiter(0.5, clock=lambda: clock["t"],
                     sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
    rl.acquire("a"); rl.acquire("a")
    assert slept and slept[0] > 0


def test_resilience_retries_then_short_circuits_via_negative_cache():
    clock = {"t": 0.0}
    slept = []
    r = ProviderResilience("p", negative_ttl_seconds=5, failure_threshold=1,
                           retry=RetryPolicy(retries=2, base_delay=0.1, jitter=False),
                           clock=lambda: clock["t"],
                           sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise RuntimeError("down")

    out1 = r.fetch("sym", failing)
    assert not out1.ok and calls["n"] == 3 and len(slept) == 2
    out2 = r.fetch("sym", failing)
    assert not out2.ok and out2.cache_hit and calls["n"] == 3
    assert r.health["breaker_trips"] >= 1


# --- 1.6 / 1.3 engine dedup and stale gate -----------------------------------

@pytest.fixture
def engine(tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["live_quotes"] = False
    eng = Engine(cfg, markets, {m: ["A", "B"] for m in markets}, "paper", False)
    eng.fx_last_update = 1e12
    with patch.object(MarketClock, "status", return_value={"status": "OPEN", "next_open": None}):
        yield eng


def make_sig(strategy="ema_cross", side=Side.LONG):
    return Signal(symbol="A", market="India", strategy=strategy, side=side,
                  entry=100.0, stop=98.0, target=104.0, confidence=0.9,
                  ts=datetime.now(timezone.utc))


def test_stale_candidate_never_reaches_risk_gate(engine):
    with patch.object(engine.risk, "can_open", return_value=(True, "")) as gate:
        engine._open_candidate("India", {
            "symbol": "A", "price": 100.0, "score": 0.9, "sig": make_sig(),
            "name": "ema_cross", "label": "x", "regime": Regime(symbol="A"),
            "stale": True, "candle_ts": "t1",
        })
    gate.assert_not_called()


def test_same_signal_same_candle_is_not_entered_twice(engine):
    cand = {"symbol": "A", "price": 100.0, "score": 0.9, "sig": make_sig(),
            "name": "ema_cross", "label": "x", "regime": Regime(symbol="A"),
            "stale": False, "candle_ts": "t1"}
    with patch.object(engine.risk, "can_open", return_value=(True, "")), \
         patch.object(engine.risk, "position_size", return_value=1.0):
        engine._open_candidate("India", cand)
        engine._open_candidate("India", copy.deepcopy(cand))
    assert len(engine.portfolio.positions) == 1


def test_new_candle_allows_same_signal_again(engine):
    cand = {"symbol": "A", "price": 100.0, "score": 0.9, "sig": make_sig(),
            "name": "ema_cross", "label": "x", "regime": Regime(symbol="A"),
            "stale": False, "candle_ts": "t1"}
    with patch.object(engine.risk, "can_open", return_value=(True, "")), \
         patch.object(engine.risk, "position_size", return_value=1.0):
        engine._open_candidate("India", cand)
        cand2 = copy.deepcopy(cand); cand2["candle_ts"] = "t2"
        engine._open_candidate("India", cand2)
    assert len(engine.portfolio.positions) == 2


# --- 1.7 selector cache -------------------------------------------------------

class FakeStrategy:
    def __init__(self, name, kind="trend"):
        self.name = name
        self.kind = kind


def signal(conf=0.9):
    return Signal(symbol="A", market="India", strategy="x", side=Side.LONG,
                  entry=1.0, stop=0.9, target=1.2, confidence=conf)


def test_selector_grades_signal_appearing_on_cached_cycle(monkeypatch):
    cfg = {"selector": {"score_refresh_cycles": 5, "max_backtest_setups": 4,
                        "min_score": 0.0, "min_signal_confidence": 0.0,
                        "min_trade_count": 1, "min_expectancy_r": 0.0,
                        "min_profit_factor": 1.0, "min_win_rate": 0.0,
                        "weights": {"expectancy": 0.4, "profit_factor": 0.2,
                                    "win_rate": 0.15, "regime_fit": 0.15,
                                    "confidence": 0.1}}}
    sel = Selector(cfg)
    strategies = [FakeStrategy("A"), FakeStrategy("B")]
    sel.set_strategies(strategies)
    monkeypatch.setattr(sel, "backtest", lambda *a, **k: BacktestStats(
        strategy="A", symbol="A", trades=20, wins=12, losses=8, win_rate=0.6,
        profit_factor=2.0, expectancy_r=0.5, eligible=True))
    df = frame(n=60, close=100.0)
    first = sel.evaluate("A", df, strategies, {"A": signal()}, cycle=1)
    assert first[0] == "A"
    second = sel.evaluate("A", df, strategies, {"A": signal(), "B": signal()}, cycle=2)
    assert second[0] in ("A", "B")
    cached = sel._cache["A"][1]
    assert cached["B"][1].eligible is True
    assert cached["B"][3] is not None


def test_selector_does_not_require_signal_at_grading_time():
    cfg = {"selector": {"score_refresh_cycles": 5, "max_backtest_setups": 4,
                        "min_score": 0.0, "min_signal_confidence": 0.0,
                        "min_trade_count": 1, "weights": {"expectancy": 1.0}}}
    sel = Selector(cfg)
    st = FakeStrategy("A")
    sel.set_strategies([st])
    graded = []
    def bt(strategy, df, symbol, **kw):
        graded.append(strategy.name)
        return BacktestStats(strategy=strategy.name, symbol=symbol, trades=10,
                             win_rate=0.5, profit_factor=1.5, expectancy_r=0.3,
                             eligible=True)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sel, "backtest", bt)
    try:
        sel.evaluate("A", frame(), [st], {}, cycle=1)
        assert graded == []
        out = sel.evaluate("A", frame(), [st], {"A": signal()}, cycle=2)
        assert out is not None and graded == ["A"]
    finally:
        monkeypatch.undo()
