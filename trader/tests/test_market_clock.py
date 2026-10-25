import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from trader.core.market_clock import MarketClock

PKG = Path(__file__).resolve().parents[1]
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def test_india_closed_weekend():
    clock = MarketClock(markets)
    # Saturday
    sat = datetime(2026, 10, 24, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    st = clock.status("India", sat)
    assert st["status"] == "CLOSED"


def test_india_open():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 22, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    st = clock.status("India", dt)
    assert st["status"] == "OPEN"


def test_india_preclose():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 22, 15, 20, tzinfo=ZoneInfo("Asia/Kolkata"))
    st = clock.status("India", dt)
    assert st["status"] == "PRE-CLOSE"


def test_countdown():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 22, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    st = clock.status("India", dt)
    assert st["countdown_seconds"] is not None and st["countdown_seconds"] > 0


# --- the markets added later -------------------------------------------

def test_hongkong_session():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 22, 10, 30, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    st = clock.status("HongKong", dt)
    assert st["status"] == "OPEN"


def test_hongkong_preclose():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 22, 15, 50, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    st = clock.status("HongKong", dt)
    assert st["status"] == "PRE-CLOSE"


def test_australia_session():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 22, 11, 0, tzinfo=ZoneInfo("Australia/Sydney"))
    st = clock.status("Australia", dt)
    assert st["status"] == "OPEN"


def test_australia_closed_on_weekend():
    clock = MarketClock(markets)
    dt = datetime(2026, 10, 24, 11, 0, tzinfo=ZoneInfo("Australia/Sydney"))
    st = clock.status("Australia", dt)
    assert st["status"] == "CLOSED"
    assert st["reason"] == "weekend"


def test_every_market_has_a_code_and_currency():
    for name, cfg in markets.items():
        assert cfg.get("code"), f"{name} has no code"
        assert cfg.get("currency"), f"{name} has no currency"
        assert len(cfg["code"]) == 2, f"{name} code should be 2 characters"


def test_all_markets_have_watchlists():
    import yaml
    wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]
    for name in markets:
        assert name in wl, f"{name} has no watchlist"
        # some markets (Brazil, Europe) have fewer because yfinance does not
        # publish intraday history for every ticker there
        assert len(wl[name]) >= 40, f"{name} watchlist is too small"
