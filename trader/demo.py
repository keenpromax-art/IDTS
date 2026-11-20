"""Shared demo seeding for the CLI demo and the Streamlit app.

Fills an engine with deterministic synthetic market rows, open positions,
closed trades and equity history so the dashboard can be explored without
a live data feed.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta

from trader.core.models import Position, Trade, Side, ExitReason


def seed_demo_data(engine, markets, watchlists):
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
                "name": best, "setup": best, "label": best.replace("_", " ").title(),
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
    return engine
