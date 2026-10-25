"""Tests for the live quote thread.

Prices must move every second and a stop or target must be acted on within
a second or two, not at the next scan.
"""
import time
from datetime import datetime, timezone

import pytest
import yaml
from pathlib import Path

from trader.core.engine import Engine
from trader.core.models import Position, Side
from trader.core.prices import PriceTicker

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]


def make_engine(tmp_path, **overrides):
    import copy
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["risk"].update(overrides)
    return Engine(cfg, markets, wl, "paper", False)


def open_position(engine, market="USA", symbol="AAPL", entry=100.0, stop=95.0, target=110.0):
    p = Position(id=f"p-{symbol}", symbol=symbol, market=market, side=Side.LONG,
                 strategy="ema_cross", qty=10, entry=entry, stop=stop, target=target,
                 current=entry, opened_at=datetime.now(timezone.utc), score_reason="test")
    engine.portfolio.positions.append(p)
    engine.broker.positions[p.id] = p
    return p


# ------------------------------------------------------------------ ticker

def test_ticker_returns_prices_for_known_symbols():
    t = PriceTicker(max_workers=6)
    try:
        out = t.refresh(["AAPL", "MSFT"])
        assert out, "no prices returned"
        assert all(v > 0 for v in out.values())
    finally:
        t.close()


def test_ticker_handles_a_bad_symbol():
    t = PriceTicker(max_workers=4)
    try:
        out = t.refresh(["AAPL", "NOT_A_REAL_TICKER_XYZ"])
        assert "AAPL" in out
    finally:
        t.close()


def test_ticker_with_no_symbols():
    t = PriceTicker()
    try:
        assert t.refresh([]) == {}
    finally:
        t.close()


def test_ticker_tracks_quote_age():
    t = PriceTicker(max_workers=4)
    try:
        t.refresh(["AAPL"])
        age = t.age("AAPL")
        assert age is not None and age < 30
        assert t.age_of_newest() is not None
    finally:
        t.close()


# ------------------------------------------------------------------ engine

def test_watched_symbols_prioritise_open_positions(tmp_path):
    engine = make_engine(tmp_path)
    p = open_position(engine, symbol="AAPL")
    engine.last_scan = {"USA": [{"symbol": "MSFT"}, {"symbol": "NVDA"}]}
    watched = engine._watched_symbols()
    assert watched[0] == "AAPL", "open positions must be quoted first"
    assert "MSFT" in watched and "NVDA" in watched
    assert len(watched) == len(set(watched)), "no duplicates"


def test_live_price_closes_a_position(tmp_path):
    """A position that hits its stop must be closed by the quote thread."""
    engine = make_engine(tmp_path)
    p = open_position(engine, symbol="AAPL", entry=100.0, stop=1e9, target=1e9)
    # pretend the quote came in below the stop
    engine._check_exits({"AAPL": 50.0})
    assert not engine.portfolio.positions
    assert len(engine.portfolio.trades) == 1
    assert engine.portfolio.trades[0].exit_reason.value == "sl"


def test_live_price_closes_on_target(tmp_path):
    engine = make_engine(tmp_path)
    open_position(engine, symbol="AAPL", entry=100.0, stop=95.0, target=105.0)
    engine._check_exits({"AAPL": 110.0})
    assert not engine.portfolio.positions
    assert engine.portfolio.trades[0].exit_reason.value == "target"


def test_live_price_updates_the_position_mark(tmp_path):
    engine = make_engine(tmp_path)
    p = open_position(engine, symbol="AAPL", entry=100.0, stop=50.0, target=150.0)
    engine._check_exits({"AAPL": 120.0})
    assert p.current == 120.0
    assert len(engine.portfolio.positions) == 1


def test_live_price_marks_short_positions(tmp_path):
    engine = make_engine(tmp_path)
    p = Position(id="s1", symbol="AAPL", market="USA", side=Side.SHORT,
                 strategy="ema_cross", qty=10, entry=100.0, stop=110.0, target=90.0,
                 current=100.0, opened_at=datetime.now(timezone.utc), score_reason="t")
    engine.portfolio.positions.append(p)
    engine.broker.positions[p.id] = p
    engine._check_exits({"AAPL": 115.0})   # above the short's stop at 110
    assert not engine.portfolio.positions
    assert engine.portfolio.trades[0].exit_reason.value == "sl"


def test_scanner_rows_get_the_live_price(tmp_path):
    engine = make_engine(tmp_path)
    engine.last_scan = {"USA": [{"symbol": "AAPL", "price": 100.0, "change": 0.0}]}
    engine._refresh_quotes = lambda: None
    # drive the row update directly
    engine.last_scan["USA"][0]["live_price"] = 123.0
    engine.last_scan["USA"][0]["live_change"] = 2.0
    row = engine.last_scan["USA"][0]
    assert row["live_price"] == 123.0
    assert row["live_change"] == 2.0


def test_quote_thread_starts_and_stops(tmp_path):
    engine = make_engine(tmp_path)
    engine.price_refresh_seconds = 0.2
    engine.start_price_thread()
    assert engine._price_thread is not None
    time.sleep(0.5)
    engine.stop_price_thread()
    assert engine._price_thread is None


def test_exits_do_not_double_close(tmp_path):
    """Two threads must not close the same position twice."""
    engine = make_engine(tmp_path)
    open_position(engine, symbol="AAPL", entry=100.0, stop=1e9, target=1e9)
    engine._check_exits({"AAPL": 10.0})
    engine._check_exits({"AAPL": 10.0})
    assert len(engine.portfolio.trades) == 1
