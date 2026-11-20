"""Deterministic tests for Phase 4 provider abstraction and streaming."""
from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.data.base import DataProvider
from trader.scanner.providers import (
    AngelStreamingQuoteProvider, IncrementalCandleStore, MarketDataGateway,
    Quote, QuoteProvider, SourceHealth, StreamingHandle, YahooQuoteProvider,
)

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def frame(n=30, close=100.0):
    idx = pd.date_range(datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
                        - timedelta(minutes=5 * (n - 1)), periods=n, freq="5min")
    return pd.DataFrame({"open": [close] * n, "high": [close + 1] * n,
                         "low": [close - 1] * n, "close": [close] * n,
                         "volume": [1000.0] * n}, index=idx)


class StubQuoteProvider(QuoteProvider):
    name = "stub"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
        self.lock = threading.Lock()

    def get_quote(self, symbol):
        with self.lock:
            self.calls += 1
        if self.fail:
            return None
        return Quote(symbol=symbol, price=100.0 + self.calls,
                     ts=datetime.now(timezone.utc), source=self.name)


class StubCandles(DataProvider):
    name = "stub-candles"
    available = True

    def get_candles(self, symbol, interval="5m", lookback_days=5):
        return frame()


# --- quote timestamps ----------------------------------------------------------

def test_quote_age_and_serialization():
    now = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    q = Quote("A", 100.0, now, "test", "poll")
    assert q.age_seconds(now + timedelta(seconds=5)) == 5.0
    assert q.to_dict()["source"] == "test"
    naive = Quote("B", 1.0, now.replace(tzinfo=None), "test")
    assert naive.age_seconds(now) == 0.0


def test_quote_provider_batch_returns_only_valid_quotes():
    p = StubQuoteProvider()
    out = p.batch_quotes(["A", "B"])
    assert set(out) == {"A", "B"}
    assert p.batch_quotes([]) == {}


# --- streaming handle ----------------------------------------------------------

def test_streaming_handle_polls_and_stops_cleanly():
    p = StubQuoteProvider()
    seen = []
    h = StreamingHandle(p, ["A", "B"], seen.append, interval_seconds=0.01)
    assert h.mode == "poll"
    h.start().start()
    deadline = time.monotonic() + 2
    while len(seen) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    h.subscribe(["C"]); h.unsubscribe(["A"])
    h.stop(); h.stop()
    assert len(seen) >= 4
    assert set(h.latest) <= {"A", "B", "C"}
    assert not h.running


def test_streaming_handle_marks_websocket_mode_when_supported():
    class WS(QuoteProvider):
        supports_streaming = True
        name = "ws"
        def get_quote(self, symbol):
            return Quote(symbol, 1.0, datetime.now(timezone.utc), "ws")
    h = StreamingHandle(WS(), ["A"], lambda q: None)
    assert h.mode == "websocket"


# --- Angel quote adapter -------------------------------------------------------

def test_angel_quote_provider_uses_candle_close_and_detects_no_stream():
    class FakeAngel:
        name = "angel"
        available = True
        _smart = None
        def get_candles(self, symbol, interval="5m", lookback_days=5):
            return frame()
    p = AngelStreamingQuoteProvider(FakeAngel())
    q = p.get_quote("A")
    assert q is not None and q.price == 100.0 and q.source == "angel"
    assert not p.supports_streaming and p.health().streaming is False
    assert p.batch_quotes(["A", "B"])["B"].symbol == "B"


def test_angel_quote_provider_detects_websocket_methods():
    class WS:
        def connect(self): pass
        def subscribe(self): pass
        def liveEquityData(self): pass
    class Smart:
        ws = WS()
    class FakeAngel:
        name = "angel"
        available = True
        _smart = Smart()
        def get_candles(self, symbol, interval="5m", lookback_days=5):
            return frame()
    p = AngelStreamingQuoteProvider(FakeAngel())
    assert p.supports_streaming and p.get_quote("A").mode == "websocket"


# --- incremental store ---------------------------------------------------------

def test_incremental_store_upserts_without_duplicates_and_keeps_newest():
    store = IncrementalCandleStore(max_bars=5)
    old = frame(n=5)
    store.upsert("A", old)
    fresh = old.iloc[-2:].copy()
    fresh["close"] = 999.0
    merged = store.upsert("A", fresh)
    assert len(merged) == 5
    assert merged.index.is_monotonic_increasing
    assert merged["close"].iloc[-1] == 999.0
    assert store.upsert("A", None) is not None
    assert store.symbols() == ["A"]


# --- gateway -------------------------------------------------------------------

def test_gateway_bounded_batch_and_health():
    gw = MarketDataGateway("India", StubCandles(), StubQuoteProvider(),
                           max_workers=2, rate_limit_seconds=0.0)
    out = gw.batch_candles(["A", "B", "C"])
    assert set(out) == {"A", "B", "C"}
    q = gw.get_quote("A")
    assert q is not None and q.age_seconds() < 5
    h = gw.health()
    assert h["market"] == "India" and h["quotes"]["name"] == "stub"
    gw.close()


def test_gateway_incremental_updates():
    gw = MarketDataGateway("India", StubCandles(), None)
    first = gw.get_candles("A", incremental=True)
    second = gw.get_candles("A", incremental=True)
    assert first is not None and second is not None
    assert gw.health()["incremental_symbols"] == 1


def test_yahoo_quote_provider_has_no_streaming_claim():
    p = YahooQuoteProvider(max_workers=1)
    assert not p.supports_streaming and not p.health().streaming
    p.close()


# --- engine integration --------------------------------------------------------

def _engine(tmp_path, gateway=False):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["data"]["gateway_enabled"] = gateway
    cfg["data"]["streaming_enabled"] = False
    eng = Engine(cfg, markets, {m: ["A"] for m in markets}, "paper", False)
    eng.fx_last_update = 1e12
    return eng


def test_engine_has_no_gateways_by_default(tmp_path):
    eng = _engine(tmp_path)
    assert eng.gateways == {}
    assert eng.start_streaming_quotes() == set()


def test_engine_builds_gateway_and_stream_when_enabled(tmp_path):
    eng = _engine(tmp_path, gateway=True)
    assert set(eng.gateways) == set(eng.markets_cfg)
    for gw in eng.gateways.values():
        gw.quotes = StubQuoteProvider()
    started = eng.start_streaming_quotes(interval_seconds=0.01)
    assert started == set(eng.markets_cfg)
    eng.stop_streaming_quotes()
    assert eng.stream_handles == {}
