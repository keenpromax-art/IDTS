"""Fractional coin quantities for crypto markets.

Crypto pairs trade in coin fractions, so the RiskManager must size in
fractions of the market's qty_step instead of whole units. Legacy
equity/futures markets keep integer sizing.
"""
import copy
from datetime import datetime, timezone

import yaml
from pathlib import Path

from trader.core.engine import Engine
from trader.core.models import Position, Side

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]


def make_engine(tmp_path, **overrides):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["risk"].update(overrides)
    return Engine(cfg, markets, wl, "paper", False)


# ------------------------------------------------------------------ sizing

def test_crypto_sizes_fractional(tmp_path):
    engine = make_engine(tmp_path, risk_per_trade_pct=1.0)
    engine.risk.risk["max_position_pct_of_equity"] = 0
    equity = 10000.0
    # BTC at 50k with a 1k risk distance asks for 0.1 BTC
    qty = engine.risk.position_size(equity, 50000.0, 49000.0, 0.0001)
    assert qty == 0.1
    assert round(qty * 1000.0, 6) == 100.0   # 1% of 10,000


def test_crypto_floors_to_the_step(tmp_path):
    engine = make_engine(tmp_path, risk_per_trade_pct=1.0)
    engine.risk.risk["max_position_pct_of_equity"] = 0
    equity = 10000.0
    # 100 units of risk at a 551.5 stop distance -> 0.18132 coins.  The step
    # floor must not round that up: 0.1813 coins, never 0.1814.
    qty = engine.risk.position_size(equity, 50000.0, 49448.5, 0.0001)
    assert qty == 0.1813
    assert abs(qty / 0.0001 - round(qty / 0.0001)) < 1e-6   # an exact step
    assert qty * 551.5 <= 100.0   # never more risk than the 1% budget


def test_crypto_notional_cap_still_bites(tmp_path):
    engine = make_engine(tmp_path, risk_per_trade_pct=1.0,
                         max_position_pct_of_equity=20)
    equity = 10000.0
    # a very tight stop would ask for 5 BTC (250k) - the 20% cap must bind
    qty = engine.risk.position_size(equity, 50000.0, 49990.0, 0.0001)
    assert qty * 50000.0 <= equity * 0.20 + 50000.0


def test_legacy_markets_stay_integer(tmp_path):
    engine = make_engine(tmp_path, risk_per_trade_pct=1.0)
    engine.risk.risk["max_position_pct_of_equity"] = 0
    qty = engine.risk.position_size(10000.0, 100.0, 98.0)   # no step -> int
    assert qty == 50 and not isinstance(qty, float)


def test_zero_viable_fraction_returns_zero(tmp_path):
    engine = make_engine(tmp_path, risk_per_trade_pct=1.0)
    engine.risk.risk["max_position_pct_of_equity"] = 0
    # a 1 EUR-equity account budgets 0.01 of risk: 0.01 / 500 = 0.00002,
    # under a single 0.0001 step -> no position at all
    assert engine.risk.position_size(1.0, 50000.0, 49500.0, 0.0001) == 0


# ------------------------------------------------------------------ integration

def test_paper_position_keeps_fractional_qty(tmp_path):
    engine = make_engine(tmp_path)
    p = Position(id="p-btc", symbol="BTCUSDT", market="Crypto", side=Side.LONG,
                 strategy="ema_cross", qty=0.1, entry=50000.0, stop=49000.0,
                 target=52000.0, current=50500.0,
                 opened_at=datetime.now(timezone.utc), score_reason="test")
    engine.portfolio.positions.append(p)
    engine.broker.positions[p.id] = p
    # positions keep the fraction; matched against a float block instead of
    # an exact key lookup so legacy integer positions keep working too
    assert engine.broker.positions["p-btc"].qty == 0.1
    find = engine.portfolio.find_position("BTCUSDT", "Crypto")
    assert find is not None and find.qty == 0.1


def test_fractional_pnl_math(tmp_path):
    from trader.core.models import Trade
    from trader.core.models import ExitReason
    t = Trade(id="t1", symbol="BTCUSDT", market="Crypto", side=Side.LONG,
              strategy="ema_cross", qty=0.1, entry=50000.0, exit=50500.0,
              opened_at=datetime.now(timezone.utc),
              closed_at=datetime.now(timezone.utc), exit_reason=ExitReason.TARGET)
    assert t.pnl_gross == 50.0   # 0.1 BTC * 500 USDT move
    t2 = Trade(id="t2", symbol="BTCUSDT", market="Crypto", side=Side.LONG,
               strategy="ema_cross", qty=0.75, entry=50000.0, exit=50100.0,
               opened_at=datetime.now(timezone.utc),
               closed_at=datetime.now(timezone.utc), exit_reason=ExitReason.TARGET,
               charges=7.5)
    assert t2.pnl == t2.pnl_gross - 7.5 and t2.pnl_gross == 75.0


def test_storage_roundtrip_fractional(tmp_path):
    from trader.core.storage import Storage
    st = Storage(str(tmp_path / "t.db"))
    p = Position(id="p-btc", symbol="BTCUSDT", market="Crypto", side=Side.LONG,
                 strategy="ema_cross", qty=0.1, entry=50000.0, stop=49000.0,
                 target=52000.0, current=50500.0,
                 opened_at=datetime.now(timezone.utc), score_reason="test")
    st.save_position(p)
    restored = st.load_open_positions()
    assert len(restored) == 1 and restored[0].qty == 0.1


def test_validate_markets_accepts_qty_step():
    from trader.core.config import validate_markets
    raw = {"markets": {"Crypto": {"code": "CR", "currency": "USDT",
                                  "timezone": "UTC", "open": "00:00",
                                  "close": "23:59", "qty_step": 0.0001}}}
    assert validate_markets(raw)["Crypto"]["qty_step"] == 0.0001


def test_validate_markets_rejects_bad_qty_step():
    from trader.core.config import validate_markets
    for bad in (-0.001, "abc"):
        raw = {"markets": {"Crypto": {"code": "CR", "currency": "USDT",
                                      "timezone": "UTC", "open": "00:00",
                                      "close": "23:59", "qty_step": bad}}}
        try:
            validate_markets(raw)
            raise AssertionError(f"qty_step {bad!r} must be rejected")
        except ValueError:
            pass


def test_backtest_sizes_fractional():
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    from trader.backtest.engine import BacktestEngine, BacktestConfig
    from trader.core.models import Signal

    n = 60
    idx = [datetime.now(timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
    base = 500.0 + np.arange(n) * 10.0
    df = pd.DataFrame(index=idx, data={
        "open": base, "high": base + 50, "low": base - 50,
        "close": base, "volume": [1000] * n})
    exec_price = float(df["open"].iloc[1])

    # a strategy that fires once on bar 0
    class OneShot:
        name = "oneshot"
        def generate_from(self, ctx):
            if ctx.bar == 0:
                return Signal(symbol="BT", market="BT", strategy="oneshot",
                              side=Side.LONG, entry=exec_price,
                              stop=exec_price - 100.0,
                              target=exec_price + 200.0)
            return None

    cfg = BacktestConfig(initial_cash=10000.0, qty_step=0.0001,
                         allow_short=False)
    res = BacktestEngine(OneShot(), df, cfg).run()
    # risk 0.75% of 10000 = 75; distance 100 -> 0.75 coins, not 0
    assert len(res.trades) == 1
    assert res.trades[0].qty == 0.75
