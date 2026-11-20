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
        from trader.demo import seed_demo_data
        import copy
        import tempfile
        # the demo never reads or writes the real session database
        settings = copy.deepcopy(settings)
        settings["db_path"] = str(
            Path(tempfile.mkdtemp(prefix="trader-demo-")) / "demo.db")
        engine = Engine(settings, markets, watchlists, "paper", False)
        seed_demo_data(engine, markets, watchlists)
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
