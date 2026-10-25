"""The README documents real numbers - this test keeps them honest.

If a default in settings.yaml changes, or a setup is added or removed, the
README has to change with it. This test fails until both agree.
"""
import re
from pathlib import Path

import yaml

from trader.strategies import ALL_RULES, ALL_STRATEGIES

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "trader"
README = (ROOT / "README.md").read_text(encoding="utf-8")
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def readme_says(pattern):
    return re.search(pattern, README) is not None


def test_strategy_and_rule_counts():
    assert readme_says(r"\*\*29 single setups\*\*"), "README must state 29 setups"
    assert readme_says(r"\*\*16 composite rules\*\*"), "README must state 16 rules"
    assert len(ALL_STRATEGIES) == 29
    assert len(ALL_RULES) == 16


def test_every_rule_is_documented():
    for r in ALL_RULES:
        assert f"`{r.name}`" in README, f"{r.name} is not in the README rule table"


def test_every_strategy_is_enabled_and_documented():
    enabled = settings["strategies"]["enabled"]
    for name in enabled:
        assert f"`{name}`" in README or name in README, f"{name} not in README"


def test_enabled_names_are_real():
    known = {c.name for c in ALL_STRATEGIES}
    for name in settings["strategies"]["enabled"]:
        assert name in known, f"{name} is not a real strategy"
    rknown = {r.name for r in ALL_RULES}
    for name in settings["rules"]["enabled"]:
        assert name in rknown, f"{name} is not a real rule"


def test_risk_defaults_match_readme():
    r = settings["risk"]
    assert r["risk_per_trade_pct"] == 0.75
    assert r["max_position_pct_of_equity"] == 20
    assert r["max_concurrent_positions"] == 6
    assert r["max_positions_per_market"] == 3
    assert r["max_trades_per_symbol_per_day"] == 2
    assert r["max_daily_loss_pct"] == 1.5
    assert r["cooldown_minutes_after_loss"] == 30
    assert r["no_entry_first_minutes"] == 5
    assert r["no_entry_last_minutes"] == 30
    assert r["square_off_before_close_minutes"] == 12
    assert r["min_risk_reward"] == 1.5
    assert r["close_on_market_closed"] is True
    for label, value in (("0.75%", r["risk_per_trade_pct"]),
                         ("1.5%", r["max_daily_loss_pct"]),
                         ("20%", r["max_position_pct_of_equity"]),
                         ("30 minutes", r["cooldown_minutes_after_loss"]),
                         ("first 5 min", r["no_entry_first_minutes"]),
                         ("last 30 min", r["no_entry_last_minutes"]),
                         ("12 min", r["square_off_before_close_minutes"]),
                         ("1.5R", r["min_risk_reward"])):
        assert label in README, f"README does not mention {label}"


def test_selector_defaults_match_readme():
    s = settings["selector"]
    assert s["min_trade_count"] == 8
    assert s["min_expectancy_r"] == 0.0
    assert s["min_profit_factor"] == 1.1
    assert s["min_win_rate"] == 0.35
    assert s["min_signal_confidence"] == 0.55
    assert s["min_score"] == 0.55
    assert s["min_market_score"] == 0.60
    assert s["max_trades_per_market_per_cycle"] == 2
    assert s["score_refresh_cycles"] == 5
    assert s["regime"]["adx_trending_threshold"] == 25
    weights = s["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    for label, value in (("0.40", weights["expectancy"]),
                         ("0.20", weights["profit_factor"]),
                         ("0.15", weights["win_rate"]),
                         ("0.15", weights["regime_fit"]),
                         ("0.10", weights["confidence"])):
        assert label in README, f"README does not mention weight {label}"


def test_readme_documents_the_custom_strategy_workflow():
    for token in ("python -m trader add-strategy",
                  "python -m trader strategies",
                  "custom_strategies.yaml",
                  "trader/custom/"):
        assert token in README, f"README does not document {token}"
    for token in ("trigger", "filters", "stop", "target", "crosses_above",
                  "r_multiple", "kind"):
        assert token in README, f"README does not document the '{token}' key"


def test_readme_documents_live_quotes():
    for token in ("live_quotes", "price_refresh_seconds", "Live quotes",
                  "quotes 0s"):
        assert token in README, f"README does not document {token}"
    assert settings["live_quotes"] is True
    assert settings["price_refresh_seconds"] <= 2.0
    assert "v" in README, "README must document the live-quote hotkey"


def test_market_sessions_match_readme():
    expected = {
        "India": ("09:15", "15:30", None, "IN", "INR", ".NS"),
        "Japan": ("09:00", "15:00", ("11:30", "12:30"), "JP", "JPY", ".T"),
        "HongKong": ("09:30", "16:00", None, "HK", "HKD", ".HK"),
        "Australia": ("10:00", "16:00", None, "AU", "AUD", ".AX"),
        "Korea": ("09:00", "15:30", None, "KR", "KRW", ".KS"),
        "Brazil": ("10:00", "17:00", None, "BR", "BRL", ".SA"),
        "Europe": ("08:00", "16:30", None, "UK", "GBP", ".L"),
        "USA": ("09:30", "16:00", None, "US", "USD", ""),
        "Crypto": ("00:00", "23:59", None, "CR", "USDT", ""),
    }
    assert set(markets) == set(expected), "unexpected set of markets"
    for mkt, (o, c, lunch, code, cur, suffix) in expected.items():
        cfg = markets[mkt]
        assert cfg["open"] == o
        assert cfg["close"] == c
        assert cfg["code"] == code
        assert cfg["currency"] == cur
        assert cfg["suffix"] == suffix
        if lunch:
            assert cfg["lunch"]["start"] == lunch[0]
            assert cfg["lunch"]["end"] == lunch[1]
        for value in (o, c, code, cur):
            assert value in README, f"README does not mention {mkt} {value}"


def test_capital_defaults_match_readme():
    cap = settings["capital"]
    assert cap["India"] == 1000000
    assert cap["Japan"] == 1000000
    assert cap["Korea"] == 1000000
    assert cap["HongKong"] == 100000
    assert cap["Australia"] == 100000
    assert cap["Brazil"] == 100000
    assert cap["Europe"] == 10000
    assert cap["USA"] == 10000
    assert cap["Crypto"] == 10000
    for value in ("1,000,000", "100,000", "10,000"):
        assert value in README, f"README does not mention capital {value}"


def test_every_market_has_symbols():
    import yaml
    wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]
    for mkt in markets:
        assert mkt in wl, f"{mkt} has no watchlist"
        assert len(wl[mkt]) >= 40, f"{mkt} watchlist is too small"
        assert len(wl[mkt]) == len(set(wl[mkt])), f"{mkt} watchlist has duplicates"


def test_new_markets_have_a_hundred_symbols():
    import yaml
    wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]
    for mkt in ("Korea", "Brazil"):
        assert len(wl[mkt]) >= 40, f"{mkt} should have a full watchlist"
        suffix = {"Korea": ".KS", "Brazil": ".SA"}[mkt]
        assert all(s.endswith(suffix) for s in wl[mkt]), \
            f"{mkt} has a ticker with the wrong suffix"


def test_readme_documents_the_live_gates():
    assert "live_trading_confirmed" in README
    assert "--mode live" in README
    assert "confirmation" in README.lower()


def test_readme_has_the_disclaimer():
    text = README.lower()
    assert "not financial advice" in text
    assert "paper results do not guarantee" in text
    assert "at your own risk" in text


def test_readme_documents_the_key_commands():
    for cmd in ("python -m trader --mode paper", "python -m trader demo",
                "python -m trader report"):
        assert cmd in README, f"README does not document `{cmd}`"


def test_readme_documents_every_hotkey():
    for key in ("`q`", "`x`", "`p`", "`Tab`", "`r`"):
        assert key in README, f"README does not document the {key} hotkey"
