"""Rolling 24h loss limit for always_open (Crypto) markets.

Crypto-style markets never close, so the daily loss limit is not enough: a
session can bleed across calendar days. This adds a rolling window loss
limit that trips the kill switch for always_open markets.
"""

from datetime import datetime, timezone, timedelta

from trader.core.models import Side
from trader.core.risk import RiskManager

RISK_CFG = {
    "max_daily_loss_pct": 1.0,
    "rolling_loss_window_hours": 24,
    "rolling_loss_pct": 1.5,
    "max_concurrent_positions": 4,
    "max_positions_per_market": 2,
    "max_trades_per_symbol_per_day": 1,
    "cooldown_minutes_after_loss": 45,
    "no_entry_first_minutes": 10,
    "no_entry_last_minutes": 45,
    "lot_size": 1,
}


def fake_clock():
    """Clock stub: cfg with always_open flags, status() returns OPEN only."""
    return type(
        "FC",
        (),
        {
            "cfg": {
                "Crypto": {"always_open": True},
                "India": {"always_open": False},
            },
            "status": lambda m: {"status": "OPEN"},
        },
    )


def new_risk(**overrides):
    cfg = dict(RISK_CFG)
    cfg.update(overrides)
    return RiskManager(cfg={"risk": cfg}, clock=fake_clock())


def test_rolling_loss_trips_kill_switch_for_crypto():
    risk = new_risk()
    now = datetime.now(timezone.utc)
    equity = 100000.0
    # three -0.6% losses within 24h => -1.8%, beyond the -1.5% rolling limit
    for _ in range(3):
        risk.record_trade_result("BTC-USDT", "Crypto", -equity * 0.6 / 100.0, at=now)
    ok, reason = risk.can_open("Crypto", "BTC-USDT", Side.LONG, [], equity, 0.0)
    assert reason == "rolling loss"
    assert not ok
    assert risk.kill_switch is True


def test_no_rolling_loss_when_within_threshold():
    risk = new_risk()
    now = datetime.now(timezone.utc)
    equity = 100000.0
    risk.record_trade_result("BTC-USDT", "Crypto", -equity * 0.5 / 100.0, at=now)
    ok, reason = risk.can_open("Crypto", "BTC-USDT", Side.LONG, [], equity, 0.0)
    # -0.5% is above the -1.5% threshold: rolling loss must not trip
    assert reason != "rolling loss"
    assert risk.kill_switch is False


def test_old_trades_pruned_outside_window():
    risk = new_risk()
    now = datetime.now(timezone.utc)
    equity = 100000.0
    # a -2% loss from 25h ago falls out of the 24h window
    risk.record_trade_result("BTC-USDT", "Crypto", -equity * 2.0 / 100.0,
                             at=now - timedelta(hours=25))
    risk.record_trade_result("BTC-USDT", "Crypto", -equity * 0.6 / 100.0, at=now)
    assert len(risk._rolling_trades["Crypto"]) == 1
    ok, reason = risk.can_open("Crypto", "BTC-USDT", Side.LONG, [], equity, 0.0)
    assert reason != "rolling loss"
    assert risk.kill_switch is False


def test_session_market_not_affected_by_rolling_loss():
    risk = new_risk()
    now = datetime.now(timezone.utc)
    equity = 1000000.0
    # India is not always_open: a -5% rolling loss must not block entries
    risk.record_trade_result("RELIANCE", "India", -equity * 5.0 / 100.0, at=now)
    ok, reason = risk.can_open("India", "RELIANCE", Side.LONG, [], equity, 0.0)
    assert reason != "rolling loss"
    assert risk.kill_switch is False
