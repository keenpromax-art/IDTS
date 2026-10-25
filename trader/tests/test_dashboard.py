import yaml
from pathlib import Path
from datetime import datetime, timezone, timedelta

from rich.console import Console

from trader.core.engine import Engine
from trader.core.models import Position, Trade, Side, ExitReason
from trader.ui.dashboard import Dashboard
from trader.ui.theme import build_theme

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
wl = yaml.safe_load(open(PKG / "config" / "watchlists.yaml"))["watchlists"]


def _demo_engine(tmp_path=None):
    import copy
    cfg = copy.deepcopy(settings)
    if tmp_path is not None:
        cfg["db_path"] = str(tmp_path / "test.db")
    engine = Engine(cfg, markets, wl, "paper", False)

    def fake_status(market, now=None):
        from datetime import datetime as dt
        tz = __import__("zoneinfo").ZoneInfo(markets[market]["timezone"])
        nowl = dt.now(tz)
        return {"market": market, "status": "OPEN", "reason": None, "local_time": nowl,
                "open_dt": nowl.replace(hour=9, minute=15), "close_dt": nowl.replace(hour=15, minute=30),
                "next_open": None, "next_close": nowl.replace(hour=15, minute=30),
                "countdown_seconds": 9000, "in_lunch": False}

    engine.clock.status = fake_status
    for m in markets:
        rows = []
        for i, sym in enumerate(wl.get(m, [])[:6]):
            price = 100.0 + i * 10
            sig = "LONG" if i % 2 == 0 else "NONE"
            stop = price * 0.98 if sig != "NONE" else None
            target = price * 1.05 if sig != "NONE" else None
            spark = [price * (1 + (j - 5) * 0.001) for j in range(20)]
            spark[-1] = price
            rows.append({
                "symbol": sym, "market": m, "price": price,
                "prev_close": price - (0.5 if i % 2 == 0 else -0.5),
                "change": 0.5 if i % 2 == 0 else -0.5,
                "regime": "trending" if i % 2 == 0 else "ranging",
                "best": "ema_cross", "best_label": "EMA Cross",
                "best_reason": "EMA Cross, score 0.74, trending",
                "best_reason_short": "EMA Cross 0.82",
                "score": 0.7, "confidence": 0.82, "regime_fit": 0.9,
                "signal": sig, "updated": datetime.now(timezone.utc), "stale": False,
                "setups": 1 if sig != "NONE" else 0,
                "plan": ({"entry": price, "stop": stop, "target": target,
                          "rr": 2.5} if sig != "NONE" else None),
                "firings": ([{"name": "ema_cross", "label": "EMA Cross",
                              "side": sig, "confidence": 0.82, "entry": price,
                              "stop": stop, "target": target,
                              "reason": "score 0.74, trending"}] if sig != "NONE" else []),
                "gate": "" if sig != "NONE" else "needs 20 trades (8)",
                "stats": {"trades": 24, "win_rate": 54.0,
                          "profit_factor": 1.6, "expectancy_r": 0.18},
                "metrics": {"rvol": 1.4, "atr_pct": 1.8, "adx": 24.0, "rsi": 56.0,
                            "vwap": price * 0.995,
                            "vwap_dist": 0.5, "spark": spark,
                            "chart": spark + [price],
                            "ema20": price * 0.997, "ema50": price * 0.99},
            })
        engine.last_scan[m] = rows
    p = Position(id="d1", symbol="RELIANCE.NS", market="India", side=Side.LONG, strategy="ema_cross",
                 qty=10, entry=2500.0, stop=2450.0, target=2600.0, current=2530.0,
                 opened_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                 score_reason="EMA Cross, score 0.74, trending")
    engine.portfolio.positions.append(p)
    tr = Trade(id="t1", symbol="TCS.NS", market="India", side=Side.SHORT, strategy="orb", qty=5,
               entry=3800.0, exit=3720.0, opened_at=datetime.now(timezone.utc) - timedelta(hours=1),
               closed_at=datetime.now(timezone.utc) - timedelta(minutes=30), exit_reason=ExitReason.TARGET,
               charges=12.0, score_reason="ORB breakout", stop=3880.0)
    engine.portfolio.trades.insert(0, tr)
    total = engine.portfolio.total_equity_inr()
    n = 30
    engine.equity_history = [total * (0.98 + 0.02 * i / (n - 1)) for i in range(n)]
    engine.last_scan_elapsed = 4.1
    engine.last_cycle_stats = {"scanned": 760, "signals": 32, "candidates": 14,
                               "opened": 2, "blocked_gate": 11, "stale": 3}
    engine.last_skipped = {"max concurrent": 4, "cooldown": 3,
                           "needs 20 trades (8)": 2}
    engine.skip_counts = {"max concurrent": 17, "cooldown": 9,
                          "needs 20 trades (8)": 6, "PF 1.15 gate": 4}
    engine.log("Engine started")
    return engine


def _render(width, height, tmp_path=None):
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    console = Console(width=width, height=height, theme=build_theme(settings.get("theme", "default")), record=True)
    dash.console = console
    console.print(dash.render())
    return console.export_text()


def test_snapshot_120x40(tmp_path):
    out = _render(120, 40, tmp_path)
    assert "PAPER" in out
    assert "Scanner" in out
    assert "Markets" in out
    assert "₹" in out
    # no rows should be clipped: every market code appears
    for code in ("IN", "JP", "UK", "US"):
        assert code in out


def test_snapshot_180x45(tmp_path):
    out = _render(180, 45, tmp_path)
    assert "Scanner" in out
    assert "Active Trades" in out


def test_all_markets_always_listed(tmp_path):
    out = _render(120, 40, tmp_path)
    for code in ("IN", "JP", "HK", "AU", "UK", "US"):
        assert code in out


def test_market_cards_wrap_when_there_are_many(tmp_path):
    """The card grid adapts: one row when it fits, otherwise it wraps cleanly."""
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    n = len(engine.markets_cfg)
    # a wide terminal fits them all on one row
    dash.console = Console(width=260, height=40)
    assert dash._market_cols() == n
    assert dash._markets_height() == 2 + 8
    # a narrow one wraps, with no empty slot in the last row
    dash.console = Console(width=120, height=40)
    cols = dash._market_cols()
    assert cols < n
    assert n % cols == 0, "the last row should not have a gap"
    grid = dash._market_grid()
    assert len(grid.columns) == cols
    assert len(grid.rows) == n // cols


def test_market_codes_come_from_the_config(tmp_path):
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    out = _render(120, 40, tmp_path)
    for mkt, cfg in engine.markets_cfg.items():
        assert cfg["code"] in out, f"{cfg['code']} missing from the dashboard"


def test_table_rows_not_truncated(tmp_path):
    """The Symbol column must be wide enough for the longest real ticker."""
    longest = max((len(s) for syms in wl.values() for s in syms), default=0)
    assert longest <= 16, f"a ticker is {longest} characters long"
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    dash._scanner_panel()
    sym_col = next(c for c in dash._col_specs if c[0] == "Symbol")
    assert sym_col[2] >= longest, \
        f"the Symbol column is {sym_col[2]} wide but a ticker needs {longest}"


def test_long_ticker_is_shown_in_full(tmp_path):
    """A long ticker must appear in the output, not truncated."""
    longest_sym = max((s for syms in wl.values() for s in syms), key=len)
    engine = _demo_engine(tmp_path)
    # put the long ticker first so it is on page one
    for m in engine.last_scan:
        rows = engine.last_scan[m]
        rows.insert(0, dict(rows[0], symbol=longest_sym))
    dash = Dashboard(engine)
    console = Console(width=140, height=40, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    assert longest_sym in console.export_text()


def test_pnl_column_always_present(tmp_path):
    """P&L is the most important column - it must survive any width."""
    for w in (100, 120, 160, 200):
        out = _render(w, 40, tmp_path)
        assert "P&L" in out, f"P&L column missing at width {w}"


def test_columns_fit_available_width(tmp_path):
    """No table may overflow its panel at any realistic width."""
    for w in (100, 120, 160, 200, 240):
        out = _render(w, 40, tmp_path)
        for line in out.splitlines():
            assert len(line) <= w, f"line {len(line)} > {w} at width {w}"


def test_every_view_fits_at_every_width(tmp_path):
    """All full-screen views render cleanly from a narrow terminal to 240 cols."""
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    views = ("monitor", "positions", "analytics", "diagnostics", "log", "detail",
             "report")
    for w, h in ((100, 30), (120, 40), (160, 45), (200, 50), (240, 60)):
        console = Console(width=w, height=h, theme=build_theme("default"), record=True)
        dash.console = console
        for view in views:
            dash._view = view
            console.print(dash.render())
            out = console.export_text()
            assert out.strip(), f"{view} rendered nothing at {w}x{h}"
            for line in out.splitlines():
                assert len(line) <= w, f"{view}: line {len(line)} > {w} at {w}x{h}"
    dash._view = "monitor"


def test_empty_engine_views_render(tmp_path):
    """Full views must explain an empty session instead of raising."""
    engine = _demo_engine(tmp_path)
    engine.portfolio.positions.clear()
    engine.portfolio.trades.clear()
    engine.last_scan = {}
    engine.equity_history = []
    dash = Dashboard(engine)
    for w, h in ((100, 30), (120, 40), (240, 60)):
        console = Console(width=w, height=h, theme=build_theme("default"), record=True)
        dash.console = console
        for view in ("monitor", "positions", "analytics", "diagnostics", "detail",
                     "report"):
            dash._view = view
            console.print(dash.render())
            out = console.export_text()
            assert out.strip(), f"{view} rendered nothing at {w}x{h}"
            for line in out.splitlines():
                assert len(line) <= w
    dash._view = "monitor"


def test_report_view_summarises_closed_trades(tmp_path):
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    console = Console(width=240, height=60, theme=build_theme("default"),
                      record=True)
    dash.console = console
    dash._view = "report"
    console.print(dash.render())
    out = console.export_text()
    assert "Session report · closed trades" in out
    assert "Closed " in out
    assert "Cumulative P&L · after charges" in out
    assert "By market" in out and "By strategy" in out and "By exit" in out
    assert "Recent closes · newest first" in out
    assert "₹" in out
    dash._view = None
    dash._handle_key(b"r", {})
    assert dash._view == "report"
    dash._view = "monitor"


def test_all_themes_render_the_new_panels(tmp_path):
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    for theme_name in ("default", "colorblind", "mono"):
        console = Console(width=160, height=45, theme=build_theme(theme_name), record=True)
        dash.console = console
        dash._view = "monitor"
        console.print(dash.render())
        dash._view = "analytics"
        console.print(dash.render())
        dash._view = "diagnostics"
        console.print(dash.render())
        out = console.export_text()
        assert "Scanner" in out
        assert "Equity · session" in out
        assert "funnel" in out
    dash._view = "monitor"


def test_many_positions_stay_inside_the_panel(tmp_path):
    engine = _demo_engine(tmp_path)
    for i in range(14):
        engine.portfolio.positions.append(Position(
            id=f"many{i}", symbol=f"TEST{i}.NS", market="India", side=Side.LONG,
            strategy="ema_cross", qty=10, entry=100.0 + i, stop=95.0 + i,
            target=110.0 + i, current=101.0 + i,
            opened_at=datetime.now(timezone.utc) - timedelta(minutes=i),
            score_reason="bulk test"))
    dash = Dashboard(engine)
    for w, h in ((120, 40), (160, 45), (240, 60)):
        console = Console(width=w, height=h, theme=build_theme("default"), record=True)
        dash.console = console
        console.print(dash.render())
        out = console.export_text()
        assert "Active Trades" in out
        assert "TOTAL" in out
        for line in out.splitlines():
            assert len(line) <= w


def test_empty_state_messages(tmp_path):
    """Empty tables explain themselves instead of showing a bare header."""
    engine = _demo_engine(tmp_path)
    engine.portfolio.positions.clear()
    engine.portfolio.trades.clear()
    engine.last_scan = {}
    dash = Dashboard(engine)
    console = Console(width=120, height=48, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    out = console.export_text()
    assert "No open positions" in out
    assert "No completed trades yet" in out


def test_all_four_markets_listed_when_two_closed(tmp_path):
    """Closed markets still get a card; the scanner only shows open ones."""
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    console = Console(width=120, height=40, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    out = console.export_text()
    for code in ("IN", "JP", "UK", "US"):
        assert code in out


def test_all_prices_are_converted_to_inr(tmp_path):
    """A native price is shown in INR at the standard cross rate."""
    engine = _demo_engine(tmp_path)
    engine.portfolio.positions.append(Position(
        id="us1", symbol="AAPL", market="USA", side=Side.LONG, strategy="ema_cross",
        qty=10, entry=100.0, stop=95.0, target=110.0, current=101.0,
        opened_at=datetime.now(timezone.utc), score_reason="US test"))
    engine.portfolio.trades.insert(0, Trade(
        id="us2", symbol="MSFT", market="USA", side=Side.LONG, strategy="orb", qty=10,
        entry=100.0, exit=90.0, opened_at=datetime.now(timezone.utc),
        closed_at=datetime.now(timezone.utc), exit_reason=ExitReason.STOPLOSS,
        charges=0.0, stop=95.0))
    dash = Dashboard(engine)
    console = Console(width=160, height=48, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    dash._view = "positions"
    console.print(dash.render())
    out = console.export_text()
    # 100 USD at the built-in 83 INR/USD cross rate
    assert "₹8,300.00" in out, "US entry/scanner price is not converted to INR"
    assert "-₹8,300" in out, "US trade P&L is not converted to INR"
    assert "$" not in out, "a native currency symbol is still on screen"


def test_risk_rows_measure_from_the_live_price_in_inr(tmp_path):
    """Shorts risk towards the stop above them; everything lands in INR."""
    engine = _demo_engine(tmp_path)
    us = engine.portfolio.fx_rate("USA")
    engine.portfolio.positions.append(Position(
        id="us1", symbol="AAPL", market="USA", side=Side.SHORT, strategy="ema_cross",
        qty=2, entry=100.0, stop=110.0, target=80.0, current=105.0,
        opened_at=datetime.now(timezone.utc), score_reason="US test"))
    rows = Dashboard(engine)._risk_rows()
    assert abs(rows["USA"]["exposure"] - 105.0 * 2 * us) < 1e-6
    assert abs(rows["USA"]["risk"] - (110.0 - 105.0) * 2 * us) < 1e-6
    assert abs(rows["USA"]["reward"] - (105.0 - 80.0) * 2 * us) < 1e-6
    # the demo long: 80/unit to the stop, 30/unit to the target, 10 units
    assert abs(rows["India"]["risk"] - 800.0) < 1e-6
    assert abs(rows["India"]["reward"] - 700.0) < 1e-6


def test_middle_column_splits_completed_over_risk(tmp_path):
    """The middle column is taller for completed trades and always shows Risk."""
    engine = _demo_engine(tmp_path)
    engine.portfolio.positions.append(Position(
        id="us1", symbol="AAPL", market="USA", side=Side.SHORT, strategy="ema_cross",
        qty=2, entry=100.0, stop=110.0, target=80.0, current=105.0,
        opened_at=datetime.now(timezone.utc), score_reason="US test"))
    engine.portfolio.positions.append(Position(
        id="cr1", symbol="QNT-USD", market="Crypto", side=Side.SHORT, strategy="orb",
        qty=0.1, entry=170.0, stop=180.0, target=150.0, current=175.0,
        opened_at=datetime.now(timezone.utc), score_reason="Crypto test"))
    dash = Dashboard(engine)
    console = Console(width=240, height=50, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    out = console.export_text()
    assert dash._middle_ratio() == (3, 2)
    assert dash._middle_h()[1] >= 6
    assert "Risk" in out
    assert "Exposure" in out and "Reward" in out and "TOTAL" in out
    assert "₹25,300" in out, "open exposure is not converted to INR"
    assert "₹800" in out, "risk to the stop is wrong"
    assert "₹700" in out, "reward to the target is wrong"
    # with nothing closed yet the two halves share the column evenly
    engine.portfolio.trades.clear()
    assert dash._middle_ratio() == (1, 1)


def test_risk_panel_keeps_total_when_the_column_is_short(tmp_path):
    """A short middle column drops the header, never the TOTAL line."""
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    console = Console(width=240, height=40, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    out = console.export_text()
    assert dash._middle_h()[1] >= 4
    assert "₹25,300" in out, "the TOTAL row was clipped off the short panel"
    assert "Exposure" not in out, "the header should give way on a short panel"


def test_completed_note_uses_engine_market_codes(tmp_path):
    """The per-market P&L note renders from realised P&L (regression)."""
    engine = _demo_engine(tmp_path)
    engine.portfolio.realized_pnl["USA"] = 1234.0
    dash = Dashboard(engine)
    dash.console = Console(width=240, height=57, theme=build_theme("default"), record=True)
    dash._middle_h = lambda: (12, 6)  # force room for the stat strip + note
    dash.console.print(dash._completed_trades())
    out = dash.console.export_text()
    assert "₹1,02,422" in out, "realised P&L is not shown in INR"
    assert "(per market, in INR)" in out


def test_empty_completed_panel_ignores_realised_pnl(tmp_path):
    """Nothing closed: the empty state must not look up per-market codes."""
    engine = _demo_engine(tmp_path)
    engine.portfolio.trades.clear()
    engine.portfolio.realized_pnl["USA"] = 1234.0
    dash = Dashboard(engine)
    console = Console(width=240, height=57, theme=build_theme("default"), record=True)
    dash.console = console
    console.print(dash.render())
    assert "No completed trades yet" in console.export_text()


def test_hotkey_loop_processes_buffered_keys(monkeypatch, tmp_path):
    """Keys in the console buffer are dispatched without a focus lookup."""
    import threading
    import trader.ui.dashboard as dash_mod

    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    dash.console = Console(width=120, height=40, theme=build_theme("default"), record=True)
    keys = [b"p", b"s"]

    class FakeMsvcrt:
        @staticmethod
        def kbhit():
            return bool(keys)

        @staticmethod
        def getch():
            ch = keys.pop(0)
            if not keys:
                dash._stop = True
            return ch

    monkeypatch.setattr(dash_mod, "msvcrt", FakeMsvcrt)
    t = threading.Thread(target=dash._hotkey_loop)
    t.start()
    t.join(3)
    assert not t.is_alive(), "the hotkey loop is still running"
    assert engine.paused is True, "p did not toggle pause"
    assert dash._sort_mode == "chg", "s did not cycle the sort"


def test_hotkey_loop_survives_a_lone_arrow_prefix(monkeypatch, tmp_path):
    """A split extended-key prefix must not block waiting for its code."""
    import threading
    import trader.ui.dashboard as dash_mod

    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    dash.console = Console(width=120, height=40, theme=build_theme("default"), record=True)
    calls = []
    keys = [b"\xe0"]

    class FakeMsvcrt:
        @staticmethod
        def kbhit():
            return bool(keys)

        @staticmethod
        def getch():
            calls.append(1)
            keys.pop(0)
            dash._stop = True
            return b"\xe0"

    monkeypatch.setattr(dash_mod, "msvcrt", FakeMsvcrt)
    t = threading.Thread(target=dash._hotkey_loop)
    t.start()
    t.join(3)
    assert not t.is_alive(), "the hotkey loop blocked on the extended-key code"
    assert len(calls) == 1, "the handler read a second byte that was not there"



