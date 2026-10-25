"""Tests for currency handling.

Money is tracked per market in that market's own currency. Anything that
adds numbers up - the header, the totals, the statistics - has to convert
to a single currency first, or an AUD profit gets added to an HKD loss.
"""
from datetime import datetime, timezone

import pytest
import yaml
from pathlib import Path

from trader.core.engine import Engine
from trader.core.models import Position, Side, Trade

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def make_engine(tmp_path, **overrides):
    import copy
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["risk"].update(overrides)
    return Engine(cfg, markets, yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"],
                  "paper", False)


def test_each_market_has_its_own_currency():
    seen = set()
    for name, cfg in markets.items():
        assert cfg["currency"], f"{name} has no currency"
        seen.add(cfg["currency"])
    # no two markets should share a currency
    assert len(seen) == len(markets), "two markets share a currency"


def test_fx_rates_cover_every_currency(tmp_path):
    engine = make_engine(tmp_path)
    for name, cfg in markets.items():
        cur = cfg["currency"]
        assert cur in engine.portfolio.fx_rates, f"no FX rate for {cur} ({name})"


def test_stats_convert_to_inr(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.fx_rates.update({"AUD": 54.0, "HKD": 10.6, "INR": 1.0})
    # a +100 AUD trade and a -10 HKD trade
    t1 = Trade(id="1", symbol="CBA.AX", market="Australia", side=Side.LONG,
               strategy="x", qty=1, entry=100.0, exit=200.0,
               opened_at=datetime.now(timezone.utc), closed_at=datetime.now(timezone.utc),
               exit_reason="target", charges=0.0, stop=90.0)
    t2 = Trade(id="2", symbol="0001.HK", market="HongKong", side=Side.LONG,
               strategy="x", qty=1, entry=100.0, exit=90.0,
               opened_at=datetime.now(timezone.utc), closed_at=datetime.now(timezone.utc),
               exit_reason="sl", charges=0.0, stop=95.0)
    engine.portfolio.trades.extend([t1, t2])
    st = engine.portfolio.stats()
    # 100 AUD = 5400 INR, -10 HKD = -106 INR
    assert st["total_pnl"] == pytest.approx(5400.0 - 106.0)
    assert st["currency"] == "INR"
    # the raw sum would have been 90 - adding different currencies
    assert st["total_pnl"] != pytest.approx(90.0)


def test_profit_factor_is_computed_in_inr(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.fx_rates.update({"AUD": 54.0})
    t1 = Trade(id="1", symbol="CBA.AX", market="Australia", side=Side.LONG,
               strategy="x", qty=1, entry=100.0, exit=110.0,
               opened_at=datetime.now(timezone.utc), closed_at=datetime.now(timezone.utc),
               exit_reason="target", charges=0.0, stop=90.0)
    t2 = Trade(id="2", symbol="CBA.AX", market="Australia", side=Side.LONG,
               strategy="x", qty=1, entry=100.0, exit=95.0,
               opened_at=datetime.now(timezone.utc), closed_at=datetime.now(timezone.utc),
               exit_reason="sl", charges=0.0, stop=90.0)
    engine.portfolio.trades.extend([t1, t2])
    st = engine.portfolio.stats()
    assert st["profit_factor"] == pytest.approx(10.0 / 5.0)
    assert st["wins"] == 1 and st["losses"] == 1
    assert st["win_rate"] == pytest.approx(50.0)


def test_unrealized_total_is_converted(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.fx_rates.update({"AUD": 54.0, "HKD": 10.6})
    p1 = Position(id="1", symbol="CBA.AX", market="Australia", side=Side.LONG,
                  strategy="x", qty=1, entry=100.0, stop=90.0, target=120.0, current=110.0,
                  opened_at=datetime.now(timezone.utc), score_reason="t")
    p2 = Position(id="2", symbol="0001.HK", market="HongKong", side=Side.LONG,
                  strategy="x", qty=1, entry=100.0, stop=90.0, target=120.0, current=105.0,
                  opened_at=datetime.now(timezone.utc), score_reason="t")
    engine.portfolio.positions.extend([p1, p2])
    # 10 AUD = 540 INR, 5 HKD = 53 INR
    assert engine.portfolio.unrealized_inr() == pytest.approx(540.0 + 53.0)


def test_per_market_pnl_stays_in_local_currency(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.fx_rates.update({"AUD": 54.0})
    engine.portfolio.realized_pnl["Australia"] = 25.0
    engine.portfolio.unrealized_pnl["Australia"] = 5.0
    per = engine.portfolio.per_market_pnl()
    assert per["Australia"] == pytest.approx(30.0), "per-market P&L must stay local"
    # but the day total is converted
    assert engine.portfolio.day_pnl_inr() == pytest.approx(30.0 * 54.0)


def test_equity_is_converted(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.fx_rates.update({"AUD": 54.0, "INR": 1.0})
    engine.portfolio.realized_pnl["Australia"] = 100.0
    eq = engine.portfolio.total_equity_inr()
    # each market's capital is converted before it is added up
    expected = sum((engine.portfolio.initial_capital[m]
                    + engine.portfolio.realized_pnl.get(m, 0.0)
                    + engine.portfolio.unrealized_pnl.get(m, 0.0))
                   * engine.portfolio.fx_rate(m)
                   for m in engine.portfolio.capital)
    assert eq == pytest.approx(expected)


def test_to_inr_helper(tmp_path):
    engine = make_engine(tmp_path)
    engine.portfolio.fx_rates.update({"JPY": 0.55})
    assert engine.portfolio.to_inr(100.0, "Japan") == pytest.approx(55.0)
    assert engine.portfolio.to_inr(100.0, "India") == pytest.approx(100.0)
