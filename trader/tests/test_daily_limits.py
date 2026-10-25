"""Tests for the daily limits and the continuous scan.

The kill switch is a *daily* circuit breaker: it must stop trading for the
rest of the day and then release itself, not latch for ever.
"""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest
import yaml
from pathlib import Path

from trader.core.engine import Engine
from trader.core.models import Position, Side

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]


@pytest.fixture(autouse=True)
def open_market(monkeypatch):
    """Force every market to a mid-session time.

    Several of these checks depend on the trading window, so the clock is
    pinned rather than left to whatever time the suite happens to run at.
    """
    from trader.core.market_clock import MarketClock

    def status(self, market, now=None):
        from datetime import datetime as dt
        tz = ZoneInfo(markets[market]["timezone"])
        base = dt.now(tz)
        open_t = base.replace(hour=10, minute=0, second=0, microsecond=0)
        close_t = base.replace(hour=15, minute=0, second=0, microsecond=0)
        return {"market": market, "status": "OPEN", "reason": None,
                "local_time": open_t + (close_t - open_t) / 2,
                "open_dt": open_t, "close_dt": close_t, "next_open": None,
                "next_close": close_t, "countdown_seconds": 3600, "in_lunch": False}

    monkeypatch.setattr(MarketClock, "status", status)


def make_engine(tmp_path, **overrides):
    import copy
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["risk"].update(overrides)
    return Engine(cfg, markets, wl, "paper", False)


def open_position(engine, market="USA", symbol="AAPL", price=100.0):
    p = Position(id=f"p-{symbol}", symbol=symbol, market=market, side=Side.LONG,
                 strategy="ema_cross", qty=10, entry=price, stop=price - 5,
                 target=price + 10, current=price,
                 opened_at=datetime.now(timezone.utc), score_reason="test")
    engine.portfolio.positions.append(p)
    engine.broker.positions[p.id] = p
    return p


# ------------------------------------------------------------------ kill switch

def test_kill_switch_blocks_entries(tmp_path):
    engine = make_engine(tmp_path)
    engine.risk.kill_switch = True
    ok, reason = engine.risk.can_open("USA", "AAPL", Side.LONG, [], 10000.0, 0.0)
    assert not ok
    assert reason == "kill switch"


def test_kill_switch_releases_on_a_new_day(tmp_path):
    """The kill switch must not stay latched for ever."""
    engine = make_engine(tmp_path)
    engine.risk.kill_switch = True
    engine._kill_flattened = True
    # pretend the trading day rolled over
    engine._session_date = "1999-01-01"
    engine._roll_over_day()
    assert engine.risk.kill_switch is False
    assert engine._kill_flattened is False
    ok, _ = engine.risk.can_open("USA", "AAPL", Side.LONG, [], 10000.0, 0.0)
    assert ok


def test_day_rollover_resets_the_daily_counters(tmp_path):
    engine = make_engine(tmp_path)
    engine.risk.trades_today["AAPL"] = 2
    engine.risk.last_loss_time["AAPL"] = datetime.now(timezone.utc)
    engine._session_date = "1999-01-01"
    engine._roll_over_day()
    assert engine.risk.trades_today == {}
    assert engine.risk.last_loss_time == {}


def test_day_pnl_resets_on_a_new_day(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.realized_pnl["USA"] = -500.0
    engine.portfolio.update_prices({})
    assert engine.portfolio.day_pnl_local("USA") == pytest.approx(-500.0)
    engine._session_date = "1999-01-01"
    engine._roll_over_day()
    assert engine.portfolio.day_pnl_local("USA") == pytest.approx(0.0)


def test_daily_loss_uses_the_market_own_currency(tmp_path):
    """The limit is a share of the market's equity, so both sides are local."""
    engine = make_engine(tmp_path, max_daily_loss_pct=1.5)
    # the USA trades in USD: a $100 loss on $10,000 equity is 1%, under the limit
    engine.portfolio.realized_pnl["USA"] = -100.0
    engine.portfolio.update_prices({})
    ok, reason = engine.risk.can_open("USA", "AAPL", Side.LONG, [],
                                      engine.portfolio.equity_local("USA"),
                                      engine.portfolio.day_pnl_local("USA"))
    assert ok, f"a 1% loss must stay under a 1.5% limit ({reason})"
    assert not engine.risk.kill_switch


def test_daily_loss_trips_the_kill_switch(tmp_path):
    engine = make_engine(tmp_path, max_daily_loss_pct=1.0)
    engine.portfolio.realized_pnl["USA"] = -200.0   # 2% of 10,000
    engine.portfolio.update_prices({})
    ok, reason = engine.risk.can_open("USA", "AAPL", Side.LONG, [],
                                      engine.portfolio.equity_local("USA"),
                                      engine.portfolio.day_pnl_local("USA"))
    assert not ok
    assert reason == "max daily loss"
    assert engine.risk.kill_switch is True


# ------------------------------------------------------------------ log noise

def test_blocked_entries_are_summarised_not_logged_per_symbol(tmp_path):
    """A blocked candidate must not flood the log with one line per symbol."""
    from trader.core.models import Signal
    engine = make_engine(tmp_path)
    engine.risk.kill_switch = True
    sig = Signal(symbol="S", market="USA", strategy="ema_cross", side=Side.LONG,
                 entry=100.0, stop=98.0, target=104.0, confidence=0.6)
    for i in range(20):
        cand = {"symbol": f"S{i}", "price": 100.0, "score": 0.9,
                "sig": sig, "name": "x", "label": "x", "reason": "", "regime": None}
        engine._open_candidate("USA", cand)
    assert engine._skipped.get("kill switch") == 20
    # nothing per-symbol was written to the log
    assert not any("skip USA S" in x for x in engine.logs)


def test_square_off_all_only_logs_when_something_closed(tmp_path):
    engine = make_engine(tmp_path)
    engine.square_off_all()
    assert not any("square-off" in x for x in engine.logs), \
        "an empty square-off should be silent"
    open_position(engine)
    engine.square_off_all()
    assert any("square-off all" in x for x in engine.logs)


# ------------------------------------------------------------------ cadence

def test_scan_progress_is_cleared_when_a_cycle_ends(tmp_path):
    engine = make_engine(tmp_path)
    engine.scan_progress = "USA 100/100"
    engine._update_fx = lambda: None
    # stub the data providers so the test does not hit the network
    import pandas as pd
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []},
                         index=pd.DatetimeIndex([]))
    for prov in engine.providers.values():
        prov.get_candles = lambda *a, **k: empty.copy()
        prov.batch_download = lambda syms, *a, **k: {s: empty.copy() for s in syms}
    engine.scan_once()
    assert engine.scan_progress == ""


# ------------------------------------------------------------------ sizing

def test_position_size_is_capped_by_notional(tmp_path):
    """A very tight stop must not ask for a position worth many times the account."""
    engine = make_engine(tmp_path, max_position_pct_of_equity=20)
    equity = 10000.0
    # a stop 0.01 wide on a 1000 price would ask for a huge position
    qty = engine.risk.position_size(equity, 1000.0, 999.99)
    assert qty * 1000.0 <= equity * 0.20 + 1000.0
    # and without the cap it would have been much larger
    engine.risk.risk["max_position_pct_of_equity"] = 0
    uncapped = engine.risk.position_size(equity, 1000.0, 999.99)
    assert uncapped > qty


def test_position_size_respects_the_risk_fraction(tmp_path):
    engine = make_engine(tmp_path, risk_per_trade_pct=1.0)
    engine.risk.risk["max_position_pct_of_equity"] = 0   # cap off for this check
    equity = 10000.0
    qty = engine.risk.position_size(equity, 100.0, 98.0)   # 2 per share of risk
    assert qty == 50
    assert qty * 2 == pytest.approx(100.0)   # 1% of 10,000
