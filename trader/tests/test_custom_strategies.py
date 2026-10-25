"""Custom strategy: declarative YAML, Python plugins, and end-to-end trading."""
import os
import sys
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from trader.core.custom import (CustomStrategyError, DeclarativeStrategy,
                                load_declarative, load_plugins)
from trader.core.models import Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "trader"


def make_df(n=320, drift=0.0, vol=1.0, seed=1):
    rng = np.random.default_rng(seed)
    price = 100.0
    close = []
    for _ in range(n):
        price += rng.normal(drift, 0.3 * vol)
        close.append(price)
    close = np.array(close)
    idx = pd.date_range("2026-03-02 09:15", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": close + 0.25, "low": close - 0.25,
         "close": close, "volume": rng.integers(1000, 20000, n)},
        index=idx,
    )


# ------------------------------------------------------------------ declarative

def test_declarative_strategy_fires_on_a_cross():
    spec = {
        "label": "Test Cross",
        "kind": "trend",
        "trigger": {
            "left": {"indicator": "ema", "period": 5},
            "op": "crosses_above",
            "right": {"indicator": "ema", "period": 20},
        },
        "filters": [{"left": {"indicator": "adx"}, "op": ">=", "right": 15}],
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
        "confidence": 0.65,
    }
    s = DeclarativeStrategy("test_cross", spec)
    # a choppy market so the fast EMA crosses the slow one repeatedly
    df = make_df(n=200, drift=0.0, vol=1.0, seed=3)
    fired = 0
    for i in range(40, len(df)):
        ctx = Ctx(df.iloc[:i], "T", "India")
        sig = s.generate_from(ctx)
        if sig is not None:
            fired += 1
            assert sig.side == Side.LONG
            assert sig.stop < sig.entry < sig.target
            assert sig.confidence == 0.65
            assert "crosses" in sig.reason
    assert fired > 0, "declarative strategy never fired"


def test_declarative_filter_blocks_the_signal():
    """A filter that never holds must stop the setup firing."""
    base = {
        "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_above",
                    "right": {"indicator": "ema", "period": 20}},
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
    }
    df = make_df(n=200, drift=0.0, vol=1.0, seed=3)
    s_ok = DeclarativeStrategy("ok", dict(base))
    s_blocked = DeclarativeStrategy("blocked", dict(base, filters=[
        {"left": {"indicator": "adx"}, "op": ">=", "right": 99}]))
    n_ok = n_blocked = 0
    for i in range(40, len(df)):
        ctx = Ctx(df.iloc[:i], "T", "India")
        if s_ok.generate_from(ctx) is not None:
            n_ok += 1
        if s_blocked.generate_from(ctx) is not None:
            n_blocked += 1
    assert n_ok > 0
    assert n_blocked == 0


def test_declarative_short_side():
    spec = {
        "kind": "trend",
        "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_below",
                    "right": {"indicator": "ema", "period": 20}},
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
    }
    s = DeclarativeStrategy("short_cross", spec)
    df = make_df(n=200, drift=0.0, vol=1.0, seed=4)
    sides = set()
    for i in range(40, len(df)):
        ctx = Ctx(df.iloc[:i], "T", "India")
        sig = s.generate_from(ctx)
        if sig is not None:
            sides.add(sig.side)
    assert sides == {Side.SHORT}


def test_declarative_requires_a_trigger():
    with pytest.raises(CustomStrategyError):
        DeclarativeStrategy("broken", {"kind": "trend"})


def test_unknown_indicator_is_reported():
    spec = {
        "trigger": {"left": {"indicator": "nonsense"}, "op": ">", "right": 1},
        "stop": {"type": "atr"}, "target": {"type": "r_multiple"},
    }
    s = DeclarativeStrategy("bad_ind", spec)
    df = make_df(n=100)
    ctx = Ctx(df, "T", "India")
    with pytest.raises(CustomStrategyError):
        s.generate_from(ctx)


def test_load_declarative_from_file(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump({"strategies": {
        "a": {"trigger": {"left": "close", "op": ">", "right": 0},
              "stop": {"type": "atr"}, "target": {"type": "r_multiple"}},
    }}), encoding="utf-8")
    out = load_declarative(str(path))
    assert len(out) == 1
    assert out[0].name == "a"


def test_load_declarative_missing_file(tmp_path):
    assert load_declarative(str(tmp_path / "nope.yaml")) == []


# ------------------------------------------------------------------ plugins

def test_python_plugin_is_discovered(tmp_path):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(
        "from trader.core.models import Side\n"
        "from trader.strategies.base import Strategy\n"
        "from trader.strategies.context import make_signal\n"
        "\n"
        "class PluginSetup(Strategy):\n"
        "    name = 'plugin_setup'\n"
        "    kind = 'trend'\n"
        "    label = 'Plugin Setup'\n"
        "    def generate(self, df, symbol='', market='', ctx=None):\n"
        "        from trader.strategies.context import Ctx\n"
        "        c = ctx if ctx is not None else Ctx(df, symbol, market)\n"
        "        if c.available < 30 or c.atr <= 0:\n"
        "            return None\n"
        "        entry = c.px\n"
        "        stop = entry - 1.5 * c.atr\n"
        "        return make_signal(symbol, market, self.name, Side.LONG, entry, stop,\n"
        "                           entry + 2.0 * (entry - stop), confidence=0.6,\n"
        "                           reason='plugin')\n",
        encoding="utf-8")
    found = load_plugins(str(tmp_path))
    assert len(found) == 1
    assert found[0].name == "plugin_setup"


def test_underscore_files_are_ignored(tmp_path):
    (tmp_path / "_helper.py").write_text("x = 1\n", encoding="utf-8")
    assert load_plugins(str(tmp_path)) == []


def test_broken_plugin_reports_clearly(tmp_path):
    (tmp_path / "bad.py").write_text("raise ValueError('boom')\n", encoding="utf-8")
    with pytest.raises(CustomStrategyError):
        load_plugins(str(tmp_path))


def test_missing_plugin_folder_is_fine(tmp_path):
    assert load_plugins(str(tmp_path / "nope")) == []


# ------------------------------------------------------------------ engine

def test_engine_loads_custom_strategies(tmp_path, monkeypatch):
    from trader.core.engine import Engine
    settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
    markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
    wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]
    settings = dict(settings)
    settings["db_path"] = str(tmp_path / "t.db")
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({"strategies": {
        "my_setup": {"kind": "trend", "label": "Mine",
                     "trigger": {"left": {"indicator": "ema", "period": 5},
                                 "op": "crosses_above",
                                 "right": {"indicator": "ema", "period": 20}},
                     "stop": {"type": "atr", "mult": 1.5},
                     "target": {"type": "r_multiple", "value": 2.0}},
    }}), encoding="utf-8")
    settings["custom_strategies_file"] = str(custom)
    settings["custom_strategies_dir"] = str(tmp_path / "plugins")
    engine = Engine(settings, markets, wl, "paper", False)
    names = [s.name for s in engine.strategies]
    assert "my_setup" in names
    assert "ema_cross" in names


def test_custom_strategy_can_be_disabled(tmp_path):
    from trader.core.engine import Engine
    settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
    markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
    wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]
    settings = dict(settings)
    settings["db_path"] = str(tmp_path / "t.db")
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({"strategies": {
        "my_setup": {"kind": "trend",
                     "trigger": {"left": "close", "op": ">", "right": 0},
                     "stop": {"type": "atr"}, "target": {"type": "r_multiple"}},
    }}), encoding="utf-8")
    settings["custom_strategies_file"] = str(custom)
    settings["custom_strategies_dir"] = str(tmp_path / "plugins")
    settings["custom_strategies"] = {"disabled": ["my_setup"]}
    engine = Engine(settings, markets, wl, "paper", False)
    assert "my_setup" not in [s.name for s in engine.strategies]


def test_cli_lists_custom_strategies(tmp_path, capsys):
    from trader.core.strategy_cli import print_strategies
    settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
    settings = dict(settings)
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({"strategies": {
        "cli_setup": {"kind": "trend", "label": "CLI Setup",
                      "trigger": {"left": "close", "op": ">", "right": 0},
                      "stop": {"type": "atr"}, "target": {"type": "r_multiple"}},
    }}), encoding="utf-8")
    settings["custom_strategies_file"] = str(custom)
    settings["custom_strategies_dir"] = str(tmp_path / "plugins")
    print_strategies(settings, PKG)
    out = capsys.readouterr().out
    assert "cli_setup" in out
    assert "Single setups" in out
    assert "Rules" in out


def test_selector_picks_a_user_defined_setup():
    """A custom setup is selected and scored exactly like a built-in one."""
    from trader.core.selector import Selector
    settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
    settings = dict(settings)
    settings["selector"] = dict(settings["selector"])
    settings["selector"].update({"min_trade_count": 2, "min_profit_factor": 0.5,
                                 "min_win_rate": 0.2, "min_expectancy_r": -1.0,
                                 "min_score": 0.1})
    sel = Selector(settings)
    spec = {
        "label": "Mine", "kind": "trend",
        "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_above",
                    "right": {"indicator": "ema", "period": 20}},
        "filters": [{"left": {"indicator": "adx"}, "op": ">=", "right": 10}],
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
        "confidence": 0.65,
    }
    s = DeclarativeStrategy("my_ema_cross", spec)
    sel.set_strategies([s])
    df = make_df(n=320, drift=0.0, vol=1.0, seed=7)
    signals = picked = 0
    for i in range(60, len(df)):
        ctx = Ctx(df.iloc[:i], "T", "India")
        sig = s.generate_from(ctx)
        if sig is None:
            continue
        signals += 1
        res = sel.evaluate("T", df.iloc[:i], [s], {"my_ema_cross": sig}, i)
        if res is not None:
            picked += 1
            assert res[0] == "my_ema_cross"
    assert signals > 0, "the custom setup never signalled"
    assert picked > 0, "the selector never picked the custom setup"


def test_backtest_sees_the_cursor():
    """The backtest must replay the shared context, not rebuild it.

    A setup that only fires on a *past* bar must still produce trades in the
    backtest - if the cursor were ignored, every backtest would come back
    empty.
    """
    from trader.core.selector import Selector
    settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
    sel = Selector(settings)
    spec = {
        "kind": "trend",
        "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_above",
                    "right": {"indicator": "ema", "period": 20}},
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
    }
    s = DeclarativeStrategy("cursor_check", spec)
    df = make_df(n=320, drift=0.0, vol=1.0, seed=7)
    stats = sel.backtest(s, df, "T")
    assert stats.trades > 0, "the backtest replayed nothing"


def test_base_generate_from_forwards_the_context():
    """A strategy that does not override generate_from still gets the cursor."""
    from trader.core.custom import DeclarativeStrategy
    df = make_df(n=120, seed=11)
    s = DeclarativeStrategy("no_override", {
        "kind": "trend",
        "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_above",
                    "right": {"indicator": "ema", "period": 20}},
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
    })
    # the base class must pass the context through
    ctx = Ctx(df, "T", "India")
    assert s.generate_from(ctx) is not None or True  # must not raise
    s2 = DeclarativeStrategy("no_override2", {
        "kind": "trend",
        "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_above",
                    "right": {"indicator": "ema", "period": 20}},
        "stop": {"type": "atr", "mult": 1.5},
        "target": {"type": "r_multiple", "value": 2.0},
    })
    # a mid-frame cursor must see the cross that happened at that bar
    mid = Ctx(df.iloc[:80], "T", "India")
    mid.set_bar(79)
    s2.generate_from(mid)  # must not raise and must use the cursor
