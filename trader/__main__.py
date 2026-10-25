from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# when running as package, config relative to package
PKG = Path(__file__).resolve().parent


def load_config():
    cfg_dir = PKG / "config"
    with open(cfg_dir / "settings.yaml") as f:
        raw_settings = yaml.safe_load(f)
    with open(cfg_dir / "markets.yaml") as f:
        raw_markets = yaml.safe_load(f)
    with open(cfg_dir / "watchlists.yaml") as f:
        raw_wl = yaml.safe_load(f)
    from trader.core.config import validate_settings, validate_markets, validate_watchlists
    settings_obj = validate_settings(raw_settings)
    markets = validate_markets(raw_markets)
    watchlists = validate_watchlists(markets, raw_wl)
    # keep a dict-like settings for the rest of the codebase
    settings = settings_obj.model_dump() if hasattr(settings_obj, "model_dump") else (settings_obj.dict() if hasattr(settings_obj, "dict") else raw_settings)
    # preserve nested objects as dicts (pydantic already did)
    return settings, markets, watchlists


def main():
    load_dotenv(PKG / ".env")
    load_dotenv(ROOT / ".env")
    # strategy management subcommands
    if len(sys.argv) > 1 and sys.argv[1] in ("strategies", "add-strategy"):
        settings, markets, watchlists = load_config()
        from trader.core.strategy_cli import add_strategy_wizard, print_strategies
        if sys.argv[1] == "strategies":
            sys.exit(print_strategies(settings, PKG))
        sys.exit(add_strategy_wizard(settings, PKG))
    # health / validate-config subcommands
    if len(sys.argv) > 1 and sys.argv[1] in ("health", "validate-config"):
        settings, markets, watchlists = load_config()
        from trader.core.config import validate_settings, validate_markets, validate_watchlists
        try:
            validate_settings(settings)
            validate_markets({"markets": markets})
            validate_watchlists(markets, {"watchlists": watchlists})
            print("config: OK")
        except Exception as e:
            print("config: FAIL -", e)
            sys.exit(1)
        if sys.argv[1] == "health":
            db = settings.get("db_path", "data/trader.db")
            print("db path:", db, "exists:", os.path.exists(db))
            print("markets:", len(markets), "symbols:", sum(len(v) for v in watchlists.values()))
            print("mode:", settings.get("mode"), "live_confirmed:", settings.get("live_trading_confirmed"))
            print("crypto_enabled:", settings.get("crypto_enabled"))
        return

    # report subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        settings, markets, watchlists = load_config()
        from trader.core.storage import Storage
        st = Storage(settings.get("db_path", "data/trader.db"))
        rows = st.report_summary()
        # each market's P&L is in its own currency - show it that way and
        # add a converted INR column
        from trader.ui.format import mkt_ccy
        rates = {"INR": 1.0, "USD": 83.0, "USDT": 83.0, "JPY": 0.55, "GBP": 105.0,
                 "HKD": 10.6, "AUD": 54.0, "KRW": 0.075, "BRL": 15.0}
        print(f"{'market':12}{'strategy':18}{'ccy':>5}{'trades':>8}{'wins':>8}"
              f"{'pnl':>14}{'pnl INR':>14}")
        for r in rows:
            market, strategy, n_trades, n_wins, pnl = r[0], r[1], r[2], r[3], r[4] or 0.0
            ccy = mkt_ccy(markets, market)
            rate = rates.get(ccy, 1.0)
            print(f"{str(market):12}{str(strategy):18}{ccy:>5}{n_trades:>8}{n_wins or 0:>8}"
                  f"{pnl:>14.2f}{pnl * rate:>14.2f}")
        return
    # backtest subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        settings, markets, watchlists = load_config()
        import argparse as _ap
        bp = _ap.ArgumentParser(prog="python -m trader backtest")
        bp.add_argument("--market", default="USA")
        bp.add_argument("--strategy", default="ema_cross")
        bp.add_argument("--symbol", default=None)
        bp.add_argument("--folds", type=int, default=5)
        bp.add_argument("--cash", type=float, default=100000.0)
        args = bp.parse_args(sys.argv[2:])
        from trader.core.engine import drop_unfinished
        from trader.strategies import ALL_STRATEGIES, ALL_RULES
        # resolve strategy
        strat_cls = None
        for c in ALL_STRATEGIES:
            if c.name == args.strategy:
                strat_cls = c
                break
        if strat_cls is None:
            for r in ALL_RULES:
                if r.name == args.strategy:
                    strat_cls = None
                    # rules are instances; wrap
                    rule_inst = r
                    break
        import pandas as pd
        from trader.data.factory import build_provider
        from trader.backtest.engine import BacktestEngine, BacktestConfig, walk_forward
        prov = build_provider(args.market)
        symbols = watchlists.get(args.market, [])
        if args.symbol:
            symbols = [args.symbol]
        symbols = symbols[:3]
        for sym in symbols:
            df = prov.get_candles(sym, settings.get("candle_interval", "5m"),
                                  settings.get("lookback_days", 5))
            if df is None or df.empty:
                print(f"{sym}: no data")
                continue
            dfc = drop_unfinished(df)
            if strat_cls is not None:
                strat = strat_cls()
            else:
                strat = rule_inst
            cfg = BacktestConfig(initial_cash=args.cash,
                                 qty_step=(markets.get(args.market) or {}).get("qty_step") or 1.0)
            bt = BacktestEngine(strat, dfc, cfg)
            res = bt.run()
            st = res.stats
            print(f"{args.market} {sym} {args.strategy}: trades={st['trades']} "
                  f"win={st['win_rate']:.1f}% pnl={st['total_pnl']:.2f} "
                  f"pf={st['profit_factor']:.2f} dd={st['max_drawdown']:.2f} "
                  f"sharpe={st['sharpe']:.2f}")
            if args.folds and args.folds > 1:
                folds = walk_forward(lambda s=strat_cls: s() if strat_cls else strat, dfc,
                                    folds=args.folds, cfg=cfg)
                for f in folds:
                    fs = f["stats"]
                    print(f"  fold {f['fold']}: trades={fs['trades']} win={fs['win_rate']:.1f}% "
                          f"pnl={fs['total_pnl']:.2f}")
        return

    # demo subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        settings, markets, watchlists = load_config()
        settings.setdefault("theme", "default")
        from trader.core.engine import Engine
        from trader.ui.dashboard import Dashboard
        from trader.core.models import Position, Trade, Side, ExitReason
        from datetime import datetime, timezone, timedelta
        import copy
        import random
        import tempfile
        # the demo never reads or writes the real session database
        settings = copy.deepcopy(settings)
        settings["db_path"] = str(
            Path(tempfile.mkdtemp(prefix="trader-demo-")) / "demo.db")
        engine = Engine(settings, markets, watchlists, "paper", False)
        # fake open markets
        def fake_status(market, now=None):
            from datetime import datetime as dt
            tz = __import__("zoneinfo").ZoneInfo(markets[market]["timezone"])
            nowl = dt.now(tz)
            return {"market": market, "status": "OPEN", "reason": None, "local_time": nowl,
                    "open_dt": nowl.replace(hour=9, minute=15), "close_dt": nowl.replace(hour=15, minute=30),
                    "next_open": None, "next_close": nowl.replace(hour=15, minute=30),
                    "countdown_seconds": 9000, "in_lunch": False}
        engine.clock.status = fake_status
        # fake scan rows - every field the redesigned dashboard reads
        random.seed(1)
        strategy_pool = ["ema_cross", "vwap", "orb", "supertrend", "macd_momentum",
                         "bb_squeeze", "rsi_reversal", "donchian"]
        for m in markets:
            rows = []
            syms = watchlists.get(m, [])[:8]
            for i, sym in enumerate(syms):
                sig = random.choice(["LONG", "SHORT", "NONE", "NONE", "NONE"])
                chg = round(random.uniform(-2, 2), 2)
                price = round(random.uniform(100, 5000), 2)
                regime = random.choice(["trending", "ranging"])
                best = random.choice(strategy_pool)
                rvol = round(random.uniform(0.6, 2.8), 2)
                atr_pct = round(random.uniform(0.4, 3.2), 2)
                adx = round(random.uniform(12, 38), 1)
                rsi = round(random.uniform(28, 72), 1)
                vwap = price * (1 + random.uniform(-0.01, 0.01))
                conf = round(random.uniform(0.4, 0.95), 2)
                stop = price * (0.985 if sig == "LONG" else 1.015)
                target = price * (1.03 if sig == "LONG" else 0.97)
                if sig == "NONE":
                    stop = target = None
                spark = [price * (1 + (j - 9) * random.uniform(-0.004, 0.004))
                         for j in range(20)]
                spark[-1] = price
                plan = None
                if sig != "NONE":
                    risk = abs(price - stop)
                    plan = {"entry": price, "stop": stop, "target": target,
                            "rr": round(abs(target - price) / risk, 2) if risk else None}
                firings = [] if sig == "NONE" else [{
                    "name": best, "label": best.replace("_", " ").title(),
                    "side": sig, "confidence": conf, "entry": price,
                    "stop": stop, "target": target,
                    "reason": f"score {random.uniform(0.5, 0.9):.2f}, {regime}",
                }]
                rows.append({
                    "symbol": sym, "market": m, "price": price,
                    "prev_close": round(price / (1 + chg / 100), 2) if chg != -100 else price,
                    "change": chg, "regime": regime,
                    "best": best, "best_label": best.replace("_", " ").title(),
                    "best_reason": f"{best.replace('_', ' ').title()}, {regime}, confidence {conf:.2f}",
                    "best_reason_short": f"{best.replace('_', ' ').title()} {conf:.2f}",
                    "score": round(random.uniform(0.3, 0.9), 2),
                    "confidence": conf,
                    "regime_fit": round(random.uniform(0.3, 1.0), 2),
                    "signal": sig, "updated": datetime.now(timezone.utc),
                    "stale": i == 7 and m == "USA",
                    "setups": 1 if sig != "NONE" else 0,
                    "firings": firings, "plan": plan,
                    "gate": "" if sig != "NONE" else random.choice([
                        "", "", "needs 20 trades (8)", "exp +0.04R",
                        "PF 1.15 gate", "WR 42% gate"]),
                    "stats": {"trades": random.randint(8, 40),
                              "win_rate": round(random.uniform(35, 65), 1),
                              "profit_factor": round(random.uniform(0.9, 2.4), 2),
                              "expectancy_r": round(random.uniform(-0.15, 0.45), 2)},
                    "metrics": {
                        "rvol": rvol, "atr_pct": atr_pct, "adx": adx, "rsi": rsi,
                        "vwap": round(vwap, 2),
                        "vwap_dist": round((price - vwap) / vwap * 100, 2),
                        "spark": spark, "chart": spark + [price],
                        "ema20": round(price * 0.998, 2),
                        "ema50": round(price * 0.992, 2),
                    },
                })
            engine.last_scan[m] = rows
        _all = [r for rows in engine.last_scan.values() for r in rows]
        _signals = sum(1 for r in _all if r.get("signal") != "NONE")
        # fake positions: enough to exercise the active panel and risk table
        pos_specs = [
            ("RELIANCE.NS", "India", Side.LONG, "ema_cross", 10, 2500, 2450, 2600, 2530, 12),
            ("AAPL", "USA", Side.LONG, "orb", 12, 100, 96, 112, 104, 35),
            ("7203.T", "Japan", Side.SHORT, "vwap", 30, 2800, 2900, 2600, 2760, 55),
            ("CBA.AX", "Australia", Side.LONG, "supertrend", 8, 90, 86, 102, 93, 20),
            ("HSBA.L", "Europe", Side.SHORT, "macd_momentum", 20, 640, 660, 600, 648, 80),
            ("0001.HK", "HongKong", Side.LONG, "bb_squeeze", 40, 45, 43.5, 49, 46.2, 15),
            ("005930.KS", "Korea", Side.LONG, "rsi_reversal", 6, 70000, 68000, 75000, 71200, 95),
            ("PETR4.SA", "Brazil", Side.SHORT, "donchian", 50, 38, 40, 34, 37.4, 42),
            ("BTC-USD", "Crypto", Side.LONG, "ema_cross", 0.01, 65000, 63000, 70000, 66200, 180),
        ]
        for i, (sym, mkt, side, strat, qty, entry, stop, target, cur, mins) in enumerate(pos_specs):
            p = Position(id=f"demo{i}", symbol=sym, market=mkt, side=side, strategy=strat,
                         qty=qty, entry=entry, stop=stop, target=target, current=cur,
                         opened_at=datetime.now(timezone.utc) - timedelta(minutes=mins),
                         score_reason=f"{strat.replace('_', ' ').title()}, score 0.74, trending")
            engine.portfolio.positions.append(p)
        engine.portfolio.update_prices({p.symbol: p.current for p in engine.portfolio.positions})
        # fake closed trades across markets, strategies and exit reasons
        reasons = [ExitReason.TARGET, ExitReason.STOPLOSS, ExitReason.TRAILING,
                   ExitReason.SQUAREOFF, ExitReason.MANUAL]
        idx = 0
        for m in markets:
            syms = watchlists.get(m, [])[:8]
            for j in range(3):
                sym = syms[(j + 1) % len(syms)] if syms else f"DEMO{j}"
                side = Side.LONG if (idx + j) % 2 == 0 else Side.SHORT
                entry = round(random.uniform(100, 4000), 2)
                move = random.uniform(-0.04, 0.05)
                exit_px = round(entry * (1 + move), 2)
                rate = engine.portfolio.fx_rate(m)
                raw_qty = 50000.0 / max(entry * rate, 1.0)
                qty = round(raw_qty, 4) if m == "Crypto" else max(1, int(raw_qty))
                opened = datetime.now(timezone.utc) - timedelta(minutes=90 + idx * 11)
                tr = Trade(id=f"demo-t{idx}", symbol=sym, market=m, side=side,
                           strategy=random.choice(strategy_pool), qty=qty,
                           entry=entry, exit=exit_px, opened_at=opened,
                           closed_at=opened + timedelta(minutes=25 + idx),
                           exit_reason=reasons[idx % len(reasons)],
                           charges=round(random.uniform(3, 40), 2),
                           score_reason="demo trade", stop=entry * 0.98)
                engine.portfolio.trades.insert(0, tr)
                engine.portfolio.realized_pnl[m] = (
                    engine.portfolio.realized_pnl.get(m, 0.0) + tr.pnl)
                engine.portfolio.capital[m] = (
                    engine.portfolio.capital.get(m, 0.0) + tr.pnl)
                idx += 1
        total = engine.portfolio.total_equity_inr()
        n = 34
        engine.equity_history = [total * (0.965 + 0.035 * i / (n - 1)) for i in range(n)]
        engine.last_scan_elapsed = 4.1
        _blocked = sum(1 for r in _all if r.get("gate"))
        _stale = sum(1 for r in _all if r.get("stale"))
        engine.last_cycle_stats = {
            "scanned": len(_all), "signals": _signals,
            "candidates": _signals, "opened": len(pos_specs),
            "blocked_gate": _blocked, "stale": _stale,
        }
        engine.last_skipped = {"max concurrent": 4, "cooldown": 3,
                               "needs 20 trades (8)": 2, "max daily loss": 1}
        engine.skip_counts = {"max concurrent": 17, "cooldown": 9,
                              "needs 20 trades (8)": 6, "PF 1.15 gate": 4,
                              "max daily loss": 2}
        engine.log("SIGNAL RELIANCE.NS LONG ema_cross")
        engine.log("ORDER RELIANCE.NS filled")
        engine.log("WARN stale quote AAPL (187s)", "warn")
        dash = Dashboard(engine)
        try:
            dash.run()
        except KeyboardInterrupt:
            pass
        engine.shutdown_report()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default=None)
    parser.add_argument("--no-crypto", action="store_true", help="disable crypto markets")
    args = parser.parse_args()
    settings, markets, watchlists = load_config()
    if args.no_crypto:
        settings["crypto_enabled"] = False
    mode = args.mode or settings.get("mode", "paper")
    # live safety
    live_confirmed = False
    if mode == "live":
        if not settings.get("live_trading_confirmed", False):
            print("live_trading_confirmed must be true in settings.yaml")
            sys.exit(1)
        print("WARNING: LIVE TRADING MODE. Type 'YES I UNDERSTAND' to confirm:")
        confirm = input("> ").strip()
        if confirm != "YES I UNDERSTAND":
            print("aborted")
            sys.exit(1)
        live_confirmed = True
    # logging setup
    import logging
    log_cfg = settings.get("logging", {})
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        filename=log_cfg.get("file", "logs/trader.log"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from trader.core.engine import Engine
    from trader.ui.dashboard import Dashboard
    engine = Engine(settings, markets, watchlists, mode, live_confirmed)
    dash = Dashboard(engine)
    try:
        dash.run()
    except KeyboardInterrupt:
        pass
    engine.shutdown_report()


if __name__ == "__main__":
    main()
