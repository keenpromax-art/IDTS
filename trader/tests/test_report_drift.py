"""Per-strategy rolling expectancy drift in the session report bundle.

A strategy is flagged when its trailing expectancy (last N trades,
N = min(10, n)) has decayed vs its all-time expectancy:

* overall_r > 0  -> drift when trailing_r < overall_r * 0.7  (> 30% decay)
* overall_r <= 0 -> drift when trailing_r < overall_r

Only strategies with at least 3 trades are evaluated.
"""

from datetime import datetime, timezone, timedelta

from rich.console import Console

from trader.core.models import ExitReason, Side, Trade
from trader.tests.test_dashboard import _demo_engine
from trader.ui.dashboard import Dashboard
from trader.ui.theme import build_theme

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _trade(tid, strategy, r, minute, market="India", side=Side.LONG):
    """Trade with an exact r_multiple: qty=1, risk=10, pnl_gross = r*10."""
    entry, stop = (100.0, 90.0) if side == Side.LONG else (100.0, 110.0)
    risk = abs(entry - stop)
    exit_ = entry + r * risk if side == Side.LONG else entry - r * risk
    closed = BASE + timedelta(minutes=minute)
    return Trade(
        id=tid, symbol=f"{strategy.upper()}.NS", market=market, side=side,
        strategy=strategy, qty=1, entry=entry, exit=exit_, stop=stop,
        opened_at=closed - timedelta(minutes=10), closed_at=closed,
        exit_reason=ExitReason.TARGET if r > 0 else ExitReason.STOPLOSS,
        charges=0.0,
    )


def _bundle(engine, trades):
    engine.portfolio.trades = trades
    dash = Dashboard(engine)
    dash.console = Console(width=240, height=60, theme=build_theme("default"),
                           record=True)
    return dash, dash._report_bundle()


def _panel_text(dash):
    dash.console.print(dash._report_summary_panel())
    return dash.console.export_text()


def test_drift_flagged_when_trailing_decays_past_30pct(tmp_path):
    # orb: two old +2R wins, then ten recent +0.5R wins.
    # overall_r = (4 + 5) / 12 = 0.75 ; trailing(10) = 0.50 < 0.75 * 0.7 = 0.525
    trades = [_trade(f"o{i}", "orb", 2.0, i) for i in range(2)]
    trades += [_trade(f"o{i}", "orb", 0.5, 100 + i) for i in range(2, 12)]
    engine = _demo_engine(tmp_path)
    dash, b = _bundle(engine, trades)

    d = b["strategy_drift"]["orb"]
    assert d["drift_flag"] is True
    assert abs(d["overall_r"] - 0.75) < 1e-9
    assert abs(d["trailing_r"] - 0.50) < 1e-9
    assert abs(d["drift_pct"] - (-33.33)) < 0.01

    out = _panel_text(dash)
    assert "Drift: orb +0.50R vs +0.75R (flag: -33%)" in out


def test_no_drift_when_expectancy_is_consistent(tmp_path):
    trades = [_trade(f"e{i}", "ema_cross", 0.5, i) for i in range(12)]
    engine = _demo_engine(tmp_path)
    dash, b = _bundle(engine, trades)

    d = b["strategy_drift"]["ema_cross"]
    assert d["drift_flag"] is False
    assert abs(d["overall_r"] - 0.5) < 1e-9
    assert abs(d["trailing_r"] - 0.5) < 1e-9
    assert abs(d["drift_pct"]) < 1e-9

    assert "Drift: none detected" in _panel_text(dash)


def test_strategies_with_fewer_than_three_trades_are_not_evaluated(tmp_path):
    trades = [_trade(f"s{i}", "orb", 1.0, i) for i in range(2)]
    engine = _demo_engine(tmp_path)
    dash, b = _bundle(engine, trades)

    assert "orb" not in b["strategy_drift"]
    assert "Drift: none detected" in _panel_text(dash)


def test_exactly_three_trades_trailing_equals_overall_is_not_flagged(tmp_path):
    # n == 3 <= 10, so the trailing window is the whole history.
    trades = [_trade(f"t{i}", "orb", 1.0, i) for i in range(2)]
    trades.append(_trade("t2", "orb", -2.0, 50))   # overall_r == trailing_r == 0
    engine = _demo_engine(tmp_path)
    dash, b = _bundle(engine, trades)

    d = b["strategy_drift"]["orb"]
    assert abs(d["overall_r"]) < 1e-9
    assert d["drift_flag"] is False
    assert "Drift: none detected" in _panel_text(dash)


def test_drift_flagged_with_negative_overall_expectancy(tmp_path):
    # overall_r <= 0 branch: trailing must be strictly worse than overall.
    trades = [_trade(f"n{i}", "orb", -0.5, i) for i in range(2)]
    trades += [_trade(f"n{i}", "orb", -1.0, 100 + i) for i in range(2, 12)]
    # overall_r = -11/12 ≈ -0.917 ; trailing(10) = -1.0 < -0.917 -> flagged
    engine = _demo_engine(tmp_path)
    dash, b = _bundle(engine, trades)

    d = b["strategy_drift"]["orb"]
    assert d["drift_flag"] is True
    assert abs(d["overall_r"] - (-11 / 12)) < 1e-9
    assert abs(d["trailing_r"] - (-1.0)) < 1e-9

    out = _panel_text(dash)
    assert "Drift: orb -1.00R vs -0.92R (flag: -9%)" in out


def test_drift_line_lists_every_flagged_strategy(tmp_path):
    trades = [_trade(f"a{i}", "orb", 2.0, i) for i in range(2)]
    trades += [_trade(f"a{i}", "orb", 0.5, 100 + i) for i in range(2, 12)]
    trades += [_trade(f"b{i}", "ema_cross", -0.5, i) for i in range(2)]
    trades += [_trade(f"b{i}", "ema_cross", -1.0, 100 + i) for i in range(2, 12)]
    engine = _demo_engine(tmp_path)
    dash, b = _bundle(engine, trades)

    assert b["strategy_drift"]["orb"]["drift_flag"] is True
    assert b["strategy_drift"]["ema_cross"]["drift_flag"] is True

    out = _panel_text(dash)
    assert out.count("Drift:") == 1
    assert "orb -1.00R" not in out   # orb's trailing is 0.50R here
    assert "ema_cross -1.00R vs -0.92R" in out
    assert "orb +0.50R vs +0.75R" in out
