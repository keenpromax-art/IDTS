"""Tests for the parallel per-market scan in Engine.scan_once.

The scan is parallelised across markets: one ThreadPoolExecutor task per open
market, max_workers = min(open_markets, 4), with the per-market body extracted
into Engine._scan_market. Sequential parts (FX update, day rollover, kill
switch, close/flatten of closed markets, exit checks, price_map merge,
persistence) are unchanged.

The clock is locked to known states and providers are stubbed so no network
is hit. Covers:
  * _scan_market returns well-shaped rows, a price-map slice and stats
  * open markets really download concurrently (overlapping batch_download windows)
  * a parallel scan equals a sequential scan on deterministic data
  * closed markets are still flattened and every market keeps a last_scan entry
  * ThreadPoolExecutor is capped at max_workers = 4
"""
import copy
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.core.models import ExitReason, Position, Side

PKG = Path(__file__).resolve().parents[1]
import yaml  # noqa: E402  (loaded after PKG is defined)

settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]

# two symbols per market keeps the test fast and deterministic
WATCHLISTS = {m: [f"{m}_SYM0", f"{m}_SYM1"] for m in markets}

EMPTY = pd.DataFrame(
    {"open": [], "high": [], "low": [], "close": [], "volume": []},
    index=pd.DatetimeIndex([], tz="UTC"),
)


def flat_df(close: float = 100.0, n: int = 60) -> pd.DataFrame:
    """Deterministic flat OHLCV history ending essentially now."""
    idx = pd.date_range(
        datetime.now(timezone.utc) - timedelta(minutes=5 * (n - 1)),
        periods=n,
        freq="5min",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": [close] * n,
            "high": [close + 0.5] * n,
            "low": [close - 0.5] * n,
            "close": [close] * n,
            "volume": [1000] * n,
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    """Force every market to a known status (defaults to OPEN for all)."""
    state = {"statuses": {}}

    def status(self, market, now=None):
        tz = ZoneInfo(markets[market]["timezone"])
        base = datetime.now(tz)
        open_t = base.replace(hour=9, minute=0, second=0, microsecond=0)
        close_t = base.replace(hour=17, minute=0, second=0, microsecond=0)
        return {
            "market": market,
            "status": state["statuses"].get(market, "OPEN"),
            "reason": None,
            "local_time": base,
            "open_dt": open_t,
            "close_dt": close_t,
            "next_open": None,
            "next_close": None,
            "countdown_seconds": 3600,
            "in_lunch": False,
        }

    monkeypatch.setattr(MarketClock, "status", status)
    return state


def make_engine(tmp_path, fixed_clock):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    engine = Engine(cfg, markets, WATCHLISTS, "paper", False)
    # no network in tests: skip the FX update (it calls yfinance)
    engine.fx_last_update = time.time()
    for prov in engine.providers.values():
        prov.get_candles = lambda *a, **k: EMPTY.copy()
        prov.batch_download = lambda syms, *a, **k: {s: EMPTY.copy() for s in syms}
    return engine


def open_position(engine, market, symbol, price=2500.0):
    p = Position(
        id=f"p-{market}-{symbol}", symbol=symbol, market=market, side=Side.LONG,
        strategy="ema_cross", qty=10, entry=price, stop=price - 50,
        target=price + 100, current=price,
        opened_at=datetime.now(timezone.utc), score_reason="test",
    )
    engine.portfolio.positions.append(p)
    engine.broker.positions[p.id] = p
    return p


# ------------------------------------------------------------------ tests


def test_scan_market_returns_rows_price_map_and_stats(tmp_path, fixed_clock):
    engine = make_engine(tmp_path, fixed_clock)
    syms = WATCHLISTS["India"]
    engine.providers["India"].batch_download = (
        lambda syms, *a, **k: {s: flat_df() for s in syms}
    )

    rows, candidates, pmap, stats = engine._scan_market("India")

    assert [r["symbol"] for r in rows] == syms
    assert all(r["market"] == "India" for r in rows)
    assert all(r["price"] == 100.0 and r["change"] == 0.0 for r in rows)
    # flat data can still fire setups on some strategies - just validate shape
    assert all(r["signal"] in ("NONE", "LONG", "SHORT") for r in rows)
    assert all(isinstance(r["setups"], int) and r["setups"] >= 0 for r in rows)
    assert all(r["regime"] != "no data" for r in rows)
    assert pmap == {s: 100.0 for s in syms}
    assert stats["scanned"] == 2
    assert stats["stale"] == 0  # flat_df ends essentially now
    assert stats["signals"] >= 0
    assert stats["blocked"] >= 0
    assert stats["candidates"] >= 0
    assert candidates == []
    assert engine.last_scan["India"] == rows


def test_open_markets_download_concurrently(tmp_path, fixed_clock):
    engine = make_engine(tmp_path, fixed_clock)
    events = {"starts": {}, "ends": {}, "order": []}
    lock = threading.Lock()

    def make_dl(market):
        def dl(symbols, *a, **k):
            with lock:
                events["starts"].setdefault(market, []).append(time.monotonic())
                events["order"].append(market)
            time.sleep(0.3)  # simulate a slow download
            with lock:
                events["ends"].setdefault(market, []).append(time.monotonic())
            return {s: EMPTY.copy() for s in symbols}

        return dl

    for m, prov in engine.providers.items():
        prov.batch_download = make_dl(m)

    engine.scan_once()

    # every open market was downloaded exactly once
    assert set(events["starts"]) == set(engine.markets_cfg)
    assert all(len(v) == 1 for v in events["starts"].values())

    # two markets submitted to the same 4-worker pool must have overlapping
    # download windows - that is the parallelism under test
    a_start, a_end = events["starts"]["India"][0], events["ends"]["India"][0]
    j_start, j_end = events["starts"]["Japan"][0], events["ends"]["Japan"][0]
    assert a_start < j_end and j_start < a_end

    # scan completed for every market
    assert set(engine.last_scan) == set(engine.markets_cfg)


def test_parallel_scan_matches_sequential_scan(tmp_path, fixed_clock):
    from concurrent.futures import Future

    class ImmediateFuture(Future):
        def __init__(self, fn, *a, **k):
            super().__init__()
            try:
                self.set_result(fn(*a, **k))
            except Exception as e:
                self.set_exception(e)

        def result(self, timeout=None):
            return super().result(timeout)

    class ImmediateExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def submit(self, fn, *a, **k):
            return ImmediateFuture(fn, *a, **k)

    def data_for(market):
        return lambda syms, *a, **k: {s: flat_df() for s in syms}

    def run(parallel: bool):
        engine = make_engine(tmp_path, fixed_clock)
        for m in ("India", "Japan"):
            engine.providers[m].batch_download = data_for(m)
        if not parallel:
            with patch("trader.core.engine.ThreadPoolExecutor", ImmediateExecutor):
                engine.scan_once()
        else:
            engine.scan_once()
        return engine

    def norm(rows):
        return [{k: v for k, v in r.items() if k != "updated"} for r in rows]

    par = run(True)
    seq = run(False)

    assert set(par.last_scan) == set(seq.last_scan)
    for market in par.last_scan:
        assert norm(par.last_scan[market]) == norm(seq.last_scan[market]), market
    assert par.last_cycle_stats == seq.last_cycle_stats


def test_max_workers_capped_at_four(tmp_path, fixed_clock):
    seen = {}

    class RecordingExecutor(RealThreadPoolExecutor):
        def __init__(self, *a, **k):
            seen["max_workers"] = k.get("max_workers")
            super().__init__(*a, **k)

    engine = make_engine(tmp_path, fixed_clock)
    with patch("trader.core.engine.ThreadPoolExecutor", RecordingExecutor):
        engine.scan_once()
    # 9 open markets -> min(9, 4) == 4
    assert seen["max_workers"] == 4


def test_closed_market_still_flattened(tmp_path, fixed_clock):
    fixed_clock["statuses"]["India"] = "CLOSED"
    engine = make_engine(tmp_path, fixed_clock)
    p = open_position(engine, "India", "INDIA_SYM0")
    engine.scan_once()

    assert not any(x.symbol == "INDIA_SYM0" for x in engine.portfolio.positions)
    assert len(engine.portfolio.trades) == 1
    assert engine.portfolio.trades[0].exit_reason == ExitReason.SQUAREOFF
    assert p.id not in engine.broker.positions

    # an open market is still scanned, not left empty
    assert engine.last_scan["Japan"] != []
    assert engine.last_scan["India"] == []


def test_every_market_keeps_a_scan_entry(tmp_path, fixed_clock):
    engine = make_engine(tmp_path, fixed_clock)
    engine.scan_once()
    assert set(engine.last_scan) == set(engine.markets_cfg)
    for rows in engine.last_scan.values():
        assert isinstance(rows, list)
