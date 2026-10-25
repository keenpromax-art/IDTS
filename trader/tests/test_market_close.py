"""Tests for the market-close behaviour.

A market we are not scanning cannot protect its positions, so every position
in a closed market must be closed automatically.
"""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest
import yaml
from pathlib import Path

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.core.models import Position, Side, ExitReason

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]

# a tiny watchlist keeps the test fast (no network, few symbols)
WATCHLISTS = {m: [f"{m}_SYM{i}" for i in range(2)] for m in markets}


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    """Force every market to a known status and a mid-session local time."""
    state = {"statuses": {}, "offset": timedelta(0)}

    def status(self, market, now=None):
        st = {
            "market": market,
            "status": state["statuses"].get(market, "OPEN"),
            "reason": None,
            "local_time": None,
            "open_dt": None,
            "close_dt": None,
            "next_open": None,
            "next_close": None,
            "countdown_seconds": 3600,
            "in_lunch": False,
        }
        tz = ZoneInfo(markets[market]["timezone"])
        base = datetime.now(tz)
        open_t = base.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = base.replace(hour=14, minute=0, second=0, microsecond=0)
        st["open_dt"] = open_t
        st["close_dt"] = close_t
        if st["status"] == "PRE-CLOSE":
            # ten minutes before the close
            st["local_time"] = close_t - timedelta(minutes=10)
        else:
            st["local_time"] = open_t + (close_t - open_t) / 2
        return st

    monkeypatch.setattr(MarketClock, "status", status)
    return state


def make_engine(tmp_path, state, **overrides):
    import copy
    import pandas as pd
    from trader.data.base import DataProvider
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["risk"].update(overrides)
    engine = Engine(cfg, markets, WATCHLISTS, "paper", False)
    # no network in tests
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []},
                         index=pd.DatetimeIndex([]))
    for prov in engine.providers.values():
        prov.get_candles = lambda *a, **k: empty.copy()
        prov.batch_download = lambda syms, *a, **k: {s: empty.copy() for s in syms}
    return engine


def open_position(engine, market="India", symbol="RELIANCE.NS", price=2500.0):
    p = Position(id=f"p-{market}-{symbol}", symbol=symbol, market=market, side=Side.LONG,
                 strategy="ema_cross", qty=10, entry=price, stop=price - 50,
                 target=price + 100, current=price,
                 opened_at=datetime.now(timezone.utc),
                 score_reason="test")
    engine.portfolio.positions.append(p)
    engine.broker.positions[p.id] = p
    return p


# ------------------------------------------------------------------ tests

def test_positions_closed_when_market_closes(tmp_path, fixed_clock):
    fixed_clock["statuses"] = {"India": "CLOSED"}
    engine = make_engine(tmp_path, fixed_clock)
    p = open_position(engine)
    engine.scan_once()
    assert not any(x.symbol == p.symbol for x in engine.portfolio.positions)
    assert len(engine.portfolio.trades) == 1
    assert engine.portfolio.trades[0].exit_reason == ExitReason.SQUAREOFF
    # the position must be gone from the broker too
    assert p.id not in engine.broker.positions


def test_positions_closed_on_lunch_break(tmp_path, fixed_clock):
    fixed_clock["statuses"] = {"Japan": "LUNCH"}
    engine = make_engine(tmp_path, fixed_clock)
    p = open_position(engine, market="Japan", symbol="7203.T")
    engine.scan_once()
    assert not any(x.symbol == p.symbol for x in engine.portfolio.positions)
    assert len(engine.portfolio.trades) == 1


def test_open_market_keeps_position(tmp_path, fixed_clock):
    fixed_clock["statuses"] = {"India": "OPEN"}
    engine = make_engine(tmp_path, fixed_clock)
    p = open_position(engine)
    engine.scan_once()
    assert any(x.symbol == p.symbol for x in engine.portfolio.positions)
    assert len(engine.portfolio.trades) == 0


def test_only_the_closed_market_is_flattened(tmp_path, fixed_clock):
    fixed_clock["statuses"] = {"India": "CLOSED", "USA": "OPEN"}
    engine = make_engine(tmp_path, fixed_clock)
    a = open_position(engine, market="India", symbol="RELIANCE.NS")
    b = open_position(engine, market="USA", symbol="AAPL")
    engine.scan_once()
    symbols = [x.symbol for x in engine.portfolio.positions]
    assert a.symbol not in symbols
    assert b.symbol in symbols


def test_restored_position_in_closed_market_is_closed(tmp_path, fixed_clock):
    """A position carried over from a previous session must not survive."""
    fixed_clock["statuses"] = {"India": "CLOSED"}
    engine = make_engine(tmp_path, fixed_clock)
    p = open_position(engine)
    engine.storage.save_position(p)
    # simulate a restart: a fresh engine reads the position back
    engine2 = make_engine(tmp_path, fixed_clock)
    assert not any(x.symbol == p.symbol for x in engine2.portfolio.positions)
    assert len(engine2.portfolio.trades) == 1


def test_restored_position_in_open_market_survives(tmp_path, fixed_clock):
    fixed_clock["statuses"] = {"India": "OPEN"}
    engine = make_engine(tmp_path, fixed_clock)
    p = open_position(engine)
    engine.storage.save_position(p)
    engine2 = make_engine(tmp_path, fixed_clock)
    assert any(x.symbol == p.symbol for x in engine2.portfolio.positions)
    assert len(engine2.portfolio.trades) == 0


def test_close_on_market_closed_can_be_disabled(tmp_path, fixed_clock):
    fixed_clock["statuses"] = {"India": "CLOSED"}
    engine = make_engine(tmp_path, fixed_clock, close_on_market_closed=False)
    p = open_position(engine)
    engine.scan_once()
    assert any(x.symbol == p.symbol for x in engine.portfolio.positions)


def test_preclose_forces_flat(tmp_path, fixed_clock):
    """Positions are already flat before the close, not at it."""
    fixed_clock["statuses"] = {"India": "PRE-CLOSE"}
    engine = make_engine(tmp_path, fixed_clock, square_off_before_close_minutes=12)
    p = open_position(engine)
    engine.scan_once()
    assert not any(x.symbol == p.symbol for x in engine.portfolio.positions)
    assert len(engine.portfolio.trades) == 1
