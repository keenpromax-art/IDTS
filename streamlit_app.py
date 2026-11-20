"""Streamlit UI for the IDTS trading engine.

A browser-based port of the terminal Rich dashboard (`python -m trader`).
Reads the same engine state: markets, scanner rows, positions, closed
trades, equity history, logs and diagnostics.

Run:
    streamlit run streamlit_app.py
    IDTS_DEMO=1 streamlit run streamlit_app.py     # synthetic demo data
    IDTS_DB_PATH=/path/to/trader.db streamlit run streamlit_app.py
    IDTS_MODE=live streamlit run streamlit_app.py  # requires live_trading_confirmed
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from trader.ui.format import (  # noqa: E402
    fmt_money, fmt_price, fmt_qty, fmt_pnl, fmt_pct_always, fmt_r, fmt_age,
    fmt_winrate, fmt_pf, fmt_score, fmt_pnl_compact, gauge,
    mkt_code, level_tag, strip_level, SHORT_LABELS,
)

PKG = ROOT / "trader"
IST = ZoneInfo("Asia/Kolkata")

from trader.scanner.views import ScannerViews, VIEW_NAMES  # noqa: E402

# --------------------------------------------------------------------------- config


@st.cache_data(ttl=30, show_spinner=False)
def _raw_config():
    with open(PKG / "config" / "settings.yaml") as f:
        settings_raw = yaml.safe_load(f) or {}
    return settings_raw


def _requested_mode():
    raw = _raw_config()
    return (os.environ.get("IDTS_MODE") or raw.get("mode") or "paper").lower()


# --------------------------------------------------------------------------- engine


def _scan_loop(engine):
    interval = max(1, int(engine.cfg.get("scan_interval_seconds", 60)))
    while engine.running:
        cycle = time.time()
        engine.next_scan_at = cycle + interval
        if not engine.paused:
            try:
                engine.scan_once()
            except Exception as e:
                engine.log(f"scan loop error: {e}", "error")
        remaining = interval - (time.time() - cycle)
        while remaining > 0 and engine.running and not engine.scan_now.is_set():
            time.sleep(min(0.25, remaining))
            remaining = interval - (time.time() - cycle)
        engine.scan_now.clear()
        engine.next_scan_at = time.time() + interval


@st.cache_resource(show_spinner="Starting trading engine…")
def get_engine(mode: str, live_confirmed: bool, demo: bool):
    from trader.__main__ import load_config
    from trader.core.engine import Engine

    settings, markets, watchlists = load_config()
    if os.environ.get("IDTS_DB_PATH"):
        settings["db_path"] = os.environ["IDTS_DB_PATH"]
    if demo:
        settings = copy.deepcopy(settings)
        settings["db_path"] = str(Path(tempfile.mkdtemp(prefix="idts-demo-")) / "demo.db")
        settings["live_quotes"] = False
        settings["scan_interval_seconds"] = 3600
    engine = Engine(settings, markets, watchlists, mode, live_confirmed)
    if demo:
        from trader.demo import seed_demo_data
        seed_demo_data(engine, markets, watchlists)
        engine.paused = True
        engine.risk.paused = True
        return engine
    threading.Thread(target=_scan_loop, args=(engine,), daemon=True,
                     name="scan").start()
    try:
        engine.start_price_thread()
    except Exception as e:
        engine.log(f"live quotes unavailable: {e}", "warn")
    return engine


# --------------------------------------------------------------------------- helpers


def _inr_pnl(engine, v, market):
    return engine.portfolio.to_inr(v, market)


def _now_ist():
    return datetime.now(timezone.utc).astimezone(IST)


def equity_series(engine, limit=240):
    points = []
    try:
        rows = engine.storage.load_equity_history(limit)
        today = datetime.now().date()
        points = [float(eq) for ts, eq, _ in rows if ts.date() == today]
    except Exception:
        points = []
    points.extend(float(v) for v in engine.equity_history if v is not None)
    current = engine.portfolio.total_equity_inr()
    if not points or abs(points[-1] - current) > 1e-9:
        points.append(current)
    return points[-limit:]


def strategy_label(engine, name):
    if not name:
        return ""
    labels = getattr(engine, "_st_label_cache", None)
    if labels is None:
        labels = {}
        for s in getattr(engine, "strategies", []):
            try:
                labels[s.name] = s.display_name()
            except Exception:
                labels[s.name] = getattr(s, "name", str(s))
        engine._st_label_cache = labels
    return labels.get(name, name)


def short_label(engine, name):
    if not name:
        return "—"
    short = SHORT_LABELS.get(name)
    if short:
        return short
    label = strategy_label(engine, name) or name
    if len(label) <= 10:
        return label
    for a, b in (("Breakout", "BO"), ("Breakdown", "BD"), ("Momentum", "Mom"),
                 ("Reversal", "Rev"), ("Continuation", "Cont"), ("Pullback", "PB"),
                 ("Expansion", "Exp"), ("Squeeze", "Sqz"), ("Volume", "Vol"),
                 ("Trend", "Tr"), ("Cross", "X"), ("Short", "S")):
        label = label.replace(a, b)
    return label[:10]


def risk_rows(engine):
    out = {}
    for p in engine.portfolio.positions:
        fx = engine.portfolio.fx_rate(p.market)
        cur = p.current or p.entry
        qty = p.qty or 0.0
        if p.side.value == "LONG":
            risk, reward = cur - p.stop, p.target - cur
        else:
            risk, reward = p.stop - cur, cur - p.target
        agg = out.setdefault(p.market, {"exposure": 0.0, "risk": 0.0, "reward": 0.0})
        agg["exposure"] += cur * qty * fx
        agg["risk"] += max(0.0, risk) * qty * fx
        agg["reward"] += max(0.0, reward) * qty * fx
    return out


def position_r(engine, p):
    ss = st.session_state
    track = ss.setdefault("_risk_track", {})
    rexc = ss.setdefault("_excursion", {})
    risk_per_unit = track.get(p.id)
    if risk_per_unit is None:
        risk_per_unit = abs(p.entry - p.stop)
        if risk_per_unit > 0:
            track[p.id] = risk_per_unit
    if not risk_per_unit or not p.qty:
        return None, None, None
    upnl = ((p.current - p.entry) * p.qty if p.side.value == "LONG"
            else (p.entry - p.current) * p.qty)
    r = upnl / (risk_per_unit * p.qty)
    rec = rexc.get(p.id)
    if rec is None:
        rec = [r, r]
        rexc[p.id] = rec
    else:
        rec[0], rec[1] = max(rec[0], r), min(rec[1], r)
    return r, rec[0], rec[1]


def upnl_inr(engine, p):
    upnl = ((p.current - p.entry) * p.qty if p.side.value == "LONG"
            else (p.entry - p.current) * p.qty)
    return _inr_pnl(engine, upnl, p.market), upnl


def exit_reason_text(reason):
    r = str(getattr(reason, "value", reason))
    m = {"target": "TARGET", "sl": "SL", "trailing": "TRAIL", "square-off": "SQ-OFF",
         "manual": "MANUAL", "kill-switch": "KILL"}
    return m.get(r.lower(), r.upper())


def scanner_rows(engine):
    ss = st.session_state
    open_markets = [m for m in engine.markets_cfg
                    if engine.clock.status(m)["status"] in ("OPEN", "PRE-CLOSE", "LUNCH")]
    markets = open_markets or list(engine.markets_cfg)
    if ss.get("market_filter"):
        markets = [ss["market_filter"]]
    rows, seen = [], set()
    for m in markets:
        for r in engine.last_scan.get(m, []):
            row = dict(r)
            row.setdefault("market", m)
            rows.append(row)
            seen.add(row.get("symbol"))
    if engine.live_quotes and hasattr(engine, "ticker"):
        try:
            prices = engine.ticker.snapshot()
        except Exception:
            prices = {}
        for m in markets:
            for sym in engine.watchlists.get(m, []):
                if sym in seen or sym not in prices:
                    continue
                age = None
                try:
                    age = engine.ticker.age(sym)
                except Exception:
                    pass
                rows.append({
                    "symbol": sym, "market": m, "price": None,
                    "live_price": prices[sym], "change": None,
                    "regime": "scanning", "best": None, "best_label": None,
                    "best_reason": "", "best_reason_short": "", "gate": "",
                    "score": None, "signal": "NONE", "setups": 0, "firings": [],
                    "metrics": {}, "plan": None, "stats": None,
                    "updated": datetime.now(timezone.utc),
                    "stale": age is not None and age > 120, "prefill": True,
                })
    if ss.get("text_filter"):
        flt = ss["text_filter"].lower()
        rows = [r for r in rows
                if flt in str(r.get("symbol", "")).lower()
                or flt in str(r.get("best", "")).lower()
                or flt in str(r.get("best_label", "")).lower()]
    if ss.get("signal_filter"):
        rows = [r for r in rows if r.get("signal") == ss["signal_filter"]]

    def row_score(r):
        sc = r.get("score")
        return sc if sc is not None else -1.0

    def row_chg(r):
        c = r.get("live_change")
        if c is None:
            c = r.get("change")
        return c if c is not None else float("-inf")

    sort_mode = ss.get("sort_mode", "score")
    if sort_mode == "chg":
        rows.sort(key=lambda r: -row_chg(r))
    elif sort_mode == "signal":
        rank = {"LONG": 0, "SHORT": 1}
        rows.sort(key=lambda r: (rank.get(r.get("signal", "NONE"), 2), -row_score(r)))
    elif sort_mode == "symbol":
        rows.sort(key=lambda r: str(r.get("symbol", "")).lower())
    else:
        rows.sort(key=lambda r: (0 if r.get("signal") in ("LONG", "SHORT") else 1,
                                 -row_score(r)))
    return rows


def find_scan_row(engine, symbol, market):
    for m, rows in engine.last_scan.items():
        if market and m != market:
            continue
        for r in rows:
            if r.get("symbol") == symbol:
                return r
    return None


def market_session_fraction(st_status):
    local, open_dt, close_dt = st_status["local_time"], st_status["open_dt"], st_status["close_dt"]
    if local is None or open_dt is None or close_dt is None:
        return None
    total = (close_dt - open_dt).total_seconds()
    if total <= 0:
        return None
    return min(1.0, max(0.0, (local - open_dt).total_seconds() / total))


def report_bundle(engine):
    p = engine.portfolio
    st_ = p.stats()
    trades = sorted(p.trades, key=lambda t: t.closed_at)
    pnls = [_inr_pnl(engine, t.pnl, t.market) for t in trades]
    cum, acc = [0.0], 0.0
    for pnl in pnls:
        acc += pnl
        cum.append(acc)
    charges = sum(_inr_pnl(engine, t.charges, t.market) for t in trades)
    holds = [max(0.0, (t.closed_at - t.opened_at).total_seconds()) for t in trades]
    avg_hold = sum(holds) / len(holds) if holds else 0.0
    pairs = list(zip(trades, pnls))
    best = max(pairs, key=lambda x: x[1]) if pairs else None
    worst = min(pairs, key=lambda x: x[1]) if pairs else None

    def blank():
        return {"n": 0, "w": 0, "pnl": 0.0, "r": 0.0, "gw": 0.0, "gl": 0.0}

    markets, strategies, exits = {}, {}, {}
    for t, pnl in pairs:
        exit_key = str(getattr(t.exit_reason, "value", t.exit_reason))
        for store, key in ((markets, t.market), (strategies, t.strategy), (exits, exit_key)):
            g = store.setdefault(key, blank())
            g["n"] += 1
            g["w"] += 1 if pnl > 0 else 0
            g["pnl"] += pnl
            g["r"] += t.r_multiple
            if pnl > 0:
                g["gw"] += pnl
            else:
                g["gl"] += abs(pnl)
    for store in (markets, strategies, exits):
        for g in store.values():
            g["win_rate"] = g["w"] / g["n"] * 100 if g["n"] else 0.0
            g["pf"] = g["gw"] / g["gl"] if g["gl"] else (float("inf") if g["gw"] else 0.0)
            g["avg_r"] = g["r"] / g["n"] if g["n"] else 0.0

    by_strategy = {}
    for t in trades:
        by_strategy.setdefault(t.strategy, []).append(t)
    strategy_drift = {}
    for name, tlist in by_strategy.items():
        n = len(tlist)
        if n < 3:
            continue
        overall_r = sum(t.r_multiple for t in tlist) / n
        k = min(10, n)
        trailing_r = sum(t.r_multiple for t in tlist[-k:]) / k
        base = abs(overall_r)
        drift_pct = ((trailing_r - overall_r) / base * 100.0) if base else 0.0
        flag = trailing_r < overall_r * 0.7 if overall_r > 0 else trailing_r < overall_r
        strategy_drift[name] = {"overall_r": overall_r, "trailing_r": trailing_r,
                                "drift_pct": drift_pct, "drift_flag": flag}

    sides = {}
    for t, pnl in pairs:
        s = sides.setdefault(t.side.value, {"n": 0, "w": 0, "pnl": 0.0})
        s["n"] += 1
        s["w"] += 1 if pnl > 0 else 0
        s["pnl"] += pnl

    counts = [0] * 5
    for t in trades:
        r = t.r_multiple
        idx = 0 if r <= -1 else (1 if r <= 0 else (2 if r <= 1 else (3 if r <= 2 else 4)))
        counts[idx] += 1
    buckets = list(zip(["<= -1R", "-1..0", "0..1", "1..2", "> 2R"], counts))

    metric_summary = {"avg_rsi": "—", "avg_adx": "—", "avg_atr_pct": "—", "avg_rvol": "—"}
    scanner_metrics = {}
    for rows_ in engine.last_scan.values():
        for r in rows_:
            m = r.get("metrics") or {}
            sym = r.get("symbol", "")
            if m and sym not in scanner_metrics:
                scanner_metrics[sym] = dict(m)
    if scanner_metrics:
        for key, out_key, fmt_ in (("rsi", "avg_rsi", "{:.1f}"), ("adx", "avg_adx", "{:.1f}"),
                                   ("atr_pct", "avg_atr_pct", "{:.2f}%"), ("rvol", "avg_rvol", "{:.2f}x")):
            vals = [m.get(key) for m in scanner_metrics.values() if m.get(key) is not None]
            if vals:
                metric_summary[out_key] = fmt_.format(sum(vals) / len(vals))

    return {"trades": trades, "pnls": pnls, "cum": cum, "stats": st_, "charges": charges,
            "avg_hold": avg_hold, "best": best, "worst": worst, "markets": markets,
            "strategies": strategies, "exits": exits, "sides": sides, "buckets": buckets,
            "metric_summary": metric_summary, "strategy_drift": strategy_drift}


CSS = """
<style>
.idts-badge { display:inline-block; padding:2px 12px; border-radius:12px;
  font-weight:700; font-size:0.8rem; letter-spacing:0.5px; }
.idts-paper { background:rgba(52,199,89,0.15); color:#34c759; border:1px solid rgba(52,199,89,0.4); }
.idts-live  { background:rgba(255,59,48,0.15); color:#ff3b30; border:1px solid rgba(255,59,48,0.4); }
.idts-pill  { display:inline-block; padding:0 8px; border-radius:8px; font-size:0.72rem; font-weight:600; }
.idts-open  { background:rgba(52,199,89,0.18); color:#34c759; }
.idts-pre   { background:rgba(255,149,0,0.18); color:#ff9500; }
.idts-closed{ background:rgba(138,138,149,0.18); color:#8a8a95; }
.idts-mono  { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; }
.idts-muted { color:#8a8a95; }
.idts-log   { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size:0.78rem; line-height:1.5; white-space:pre-wrap; }
.idts-log .ts { color:#8a8a95; }
.idts-err { color:#ff4d4d; } .idts-warn { color:#ff9500; }
.idts-sig { color:#00d97d; } .idts-ord { color:#00d4ff; }
.stMetric [data-testid="stMetricValue"] { font-size:1.28rem; white-space:nowrap;
  font-variant-numeric:tabular-nums; }
.stMetric [data-testid="stMetricLabel"] { font-size:0.72rem; }
.idts-meta { display:flex; flex-wrap:wrap; align-items:center; row-gap:2px;
  font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-numeric:tabular-nums; font-size:0.75rem; color:#8a8a95;
  margin:2px 0 14px 0; }
.idts-meta-group { display:inline-flex; align-items:center; gap:9px;
  padding:2px 12px 2px 0; margin-right:12px; border-right:1px solid rgba(138,138,149,0.28); }
.idts-meta-group:last-child { border-right:0; margin-right:0; padding-right:0; }
.idts-mkt-line { white-space:nowrap; font-variant-numeric:tabular-nums; }
.idts-section { margin-top:6px; }
</style>
"""


def _pnl_color(v):
    return "#00d97d" if v > 0 else ("#ff4d4d" if v < 0 else "#8a8a95")


def render_header(engine):
    portfolio = engine.portfolio
    total_eq = portfolio.total_equity_inr()
    day_pnl = portfolio.day_pnl_inr()
    day_pct = (day_pnl / (total_eq - day_pnl) * 100) if (total_eq - day_pnl) else 0.0
    st_ = portfolio.stats()
    trades, wr = st_["trades"], st_["win_rate"]
    mode = engine.mode.upper()

    left, *metrics = st.columns([1.15, 1.95, 1.75, 1.15, 0.8, 0.8, 0.9, 1.15])
    with left:
        cls = "idts-live" if mode == "LIVE" else "idts-paper"
        st.markdown(f"<div style='padding-top:6px'><span class='idts-badge {cls}'>{mode}</span>"
                    f"<div class='idts-mono' style='font-size:1.02rem;font-weight:700;white-space:nowrap'>IDTS</div>"
                    f"<div class='idts-muted' style='font-size:0.62rem;letter-spacing:0.08em;white-space:nowrap'>INTRADAY TRADER</div></div>",
                    unsafe_allow_html=True)
    with metrics[0]:
        st.metric("Equity", fmt_money(total_eq, "INR"))
    with metrics[1]:
        st.metric("Day P&L", fmt_pnl(day_pnl, "INR"),
                  delta=f"{fmt_pct_always(day_pct)} today", delta_color="normal")
    with metrics[2]:
        st.metric("Win rate", fmt_winrate(wr, trades))
    with metrics[3]:
        st.metric("PF", fmt_pf(st_["profit_factor"] if trades else None))
    with metrics[4]:
        st.metric("Open", len(portfolio.positions))
    with metrics[5]:
        st.metric("Trades", trades)
    with metrics[6]:
        st.markdown(f"<div style='padding-top:2px'><div class='idts-muted' style='font-size:0.72rem'>IST</div>"
                    f"<div class='idts-mono' style='font-size:1.08rem;font-weight:600;white-space:nowrap'>{_now_ist():%H:%M:%S}</div></div>",
                    unsafe_allow_html=True)

    # second line: realised / unrealised / exposure / risk / hi-lo / drawdown
    series = equity_series(engine)
    exposure = sum(v["exposure"] for v in risk_rows(engine).values())
    risk = sum(v["risk"] for v in risk_rows(engine).values())
    exp_pct = exposure / total_eq * 100 if total_eq else 0.0
    risk_pct = risk / total_eq * 100 if total_eq else 0.0
    peak = max(series) if series else total_eq
    dd = (peak - total_eq) / peak * 100 if peak else 0.0
    realised = portfolio.day_realized_inr()
    unrealised = portfolio.day_unrealized_inr()
    avg_wl = (f"{fmt_money(st_['avg_win'], 'INR')} / {fmt_money(st_['avg_loss'], 'INR')}"
              if trades else "—")
    exp_r = fmt_r(st_["expectancy_r"]) if trades else "—"
    kill = "ARMED" if getattr(engine.risk, "kill_switch", False) else "clear"
    kill_html = f"<span style='color:#ff9500'>{kill}</span>" if kill == "ARMED" else kill
    q_age = None
    if hasattr(engine, "ticker"):
        try:
            q_age = engine.ticker.age_of_newest()
        except Exception:
            q_age = None
    feed = fmt_age(q_age) if engine.live_quotes else "off"
    hi = fmt_money(max(series) if series else total_eq, "INR")
    lo = fmt_money(min(series) if series else total_eq, "INR")
    st.markdown(
        f"<div class='idts-meta'>"
        f"<span class='idts-meta-group'>"
        f"<span>Realised <span style='color:{_pnl_color(realised)}'>{fmt_pnl(realised, 'INR')}</span></span>"
        f"<span>Unrealised <span style='color:{_pnl_color(unrealised)}'>{fmt_pnl(unrealised, 'INR')}</span></span>"
        f"</span>"
        f"<span class='idts-meta-group'>"
        f"<span>Exp {exp_pct:.1f}%</span><span>Risk {risk_pct:.1f}%</span><span>DD {dd:.2f}%</span>"
        f"</span>"
        f"<span class='idts-meta-group'>"
        f"<span>Hi {hi}</span><span>Lo {lo}</span>"
        f"</span>"
        f"<span class='idts-meta-group'>"
        f"<span>{exp_r} exp</span><span>avg W/L {avg_wl}</span>"
        f"</span>"
        f"<span class='idts-meta-group'>"
        f"<span>Kill {kill_html}</span><span>feed {feed}</span><span>scan {engine.last_scan_elapsed:.1f}s</span>"
        f"</span>"
        f"</div>",
        unsafe_allow_html=True)


VIEW_LABELS = {"Monitor": "monitor", "Scanner": "scanner", "Positions": "positions",
               "Analytics": "analytics", "Diagnostics": "diagnostics", "Report": "report",
               "Log": "log", "Detail": "detail", "Help": "help"}
SORT_MODES = ("score", "chg", "signal", "symbol")


def _arm(key, ttl=6.0):
    ss = st.session_state
    now = time.time()
    if ss.get("_arm_key") == key and now - ss.get("_arm_at", 0) <= ttl:
        ss["_arm_key"] = None
        return True
    ss["_arm_key"] = key
    ss["_arm_at"] = now
    return False


def render_sidebar(engine):
    ss = st.session_state
    with st.sidebar:
        st.markdown("### IDTS Trader")
        demo = os.environ.get("IDTS_DEMO") == "1"
        if demo:
            st.caption("Demo data (IDTS_DEMO=1) — synthetic, paused, temp DB")
        else:
            n = sum(len(v) for v in engine.watchlists.values())
            st.caption(f"{len(engine.markets_cfg)} markets · {n} symbols · broker {engine.broker_name}")

        view = st.radio("View", list(VIEW_LABELS),
                        index=list(VIEW_LABELS.values()).index(ss.get("view", "monitor")))
        ss["view"] = VIEW_LABELS[view]
        st.divider()

        # ---------------- scanner filters
        st.markdown("**Scanner**")
        markets = ["ALL OPEN"] + list(engine.markets_cfg)
        mf = st.selectbox("Market", markets, index=0 if not ss.get("market_filter")
                          else markets.index(ss["market_filter"]))
        ss["market_filter"] = None if mf == "ALL OPEN" else mf
        sf = st.selectbox("Signal", ["All", "LONG", "SHORT"],
                          index=["All", "LONG", "SHORT"].index(ss.get("signal_filter") or "All"))
        ss["signal_filter"] = None if sf == "All" else sf
        ss["text_filter"] = st.text_input("Filter symbol / strategy", ss.get("text_filter", ""))
        ss["sort_mode"] = st.selectbox("Sort", SORT_MODES,
                                       index=SORT_MODES.index(ss.get("sort_mode", "score")))
        st.divider()

        # ---------------- engine controls
        st.markdown("**Engine**")
        c1, c2 = st.columns(2)
        with c1:
            label = "Resume" if engine.paused else "Pause"
            if st.button(label, width="stretch"):
                engine.paused = not engine.paused
                engine.risk.paused = engine.paused
                engine.log("scan " + ("paused" if engine.paused else "resumed"))
                st.rerun()
        with c2:
            if st.button("Force scan", width="stretch"):
                engine.scan_now.set()
                engine.log("scan forced")
                st.toast("Scan forced")
        c1, c2 = st.columns(2)
        with c1:
            lq = st.toggle("Live quotes", value=bool(engine.live_quotes))
            if lq != bool(engine.live_quotes):
                engine.live_quotes = lq
                engine.log("live quotes " + ("on" if lq else "off"))
        with c2:
            crypto_on = "Crypto" not in engine.disabled_markets
            nc = st.toggle("Crypto", value=crypto_on)
            if nc != crypto_on:
                try:
                    engine.set_market_enabled("Crypto", nc)
                except Exception as e:
                    engine.log(f"crypto toggle failed: {e}", "error")
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("KILL switch", width="stretch", type="secondary"):
                if _arm("kill"):
                    engine.risk.trigger_kill()
                    engine.log("kill switch armed", "warn")
                    st.rerun()
        with c2:
            if st.button("Flatten ALL", width="stretch", type="secondary"):
                if _arm("flat"):
                    n = engine.square_off_all()
                    st.toast(f"Squared off {n} position(s)")
                    st.rerun()
        if ss.get("_arm_key") in ("kill", "flat"):
            st.caption(f":orange[Press **{ss['_arm_key'].upper()}** again within 6s to confirm]")
        st.divider()

        # ---------------- refresh
        ss["refresh_s"] = st.select_slider(
            "Auto-refresh", options=[0, 1, 2, 5, 10], value=ss.get("refresh_s", 2),
            format_func=lambda v: "off" if v == 0 else f"{v}s")

        # ---------------- command bar
        with st.form("cmd", clear_on_submit=True):
            cmd = st.text_input("Command  (e.g. `filter reliance`, `sort chg`, `close 1`, `help`)",
                                placeholder="type a command and press Enter")
            submitted = st.form_submit_button("Run", width="stretch")
        if submitted and cmd.strip():
            _run_command(engine, cmd.strip())
            st.rerun()

        # ---------------- demo seed
        if not demo:
            if st.button("Seed demo data (in-memory)", width="stretch"):
                from trader.demo import seed_demo_data
                seed_demo_data(engine, engine.markets_cfg, engine.watchlists)
                engine.paused = True
                engine.risk.paused = True
                st.toast("Demo rows loaded — in-memory only")
                st.rerun()


def _run_command(engine, cmd):
    ss = st.session_state
    low = cmd.strip().lower()
    parts = low.split()
    head, rest = parts[0], "".join(parts[1:])
    tail = cmd[len(parts[0]):].strip()
    if head == "help":
        ss["view"] = "help"
    elif head == "filter":
        if tail in ("", "off", "clear"):
            ss["text_filter"] = ""
            engine.log("scanner filter cleared")
        else:
            ss["text_filter"] = tail.strip("'\"")
            engine.log(f"scanner filter '{ss['text_filter']}'")
    elif head == "sort":
        if rest in SORT_MODES:
            ss["sort_mode"] = rest
            engine.log(f"scanner sort {rest}")
        else:
            engine.log("usage: :sort score|chg|signal|symbol")
    elif head == "longs":
        ss["signal_filter"] = "LONG"
    elif head == "shorts":
        ss["signal_filter"] = "SHORT"
    elif head == "all":
        ss["signal_filter"] = None
    elif head == "close":
        _close_by_arg(engine, tail or rest)
    elif head == "flat":
        engine.square_off_all()
    elif head in ("pause", "resume"):
        want = head == "pause"
        if engine.paused != want:
            engine.paused = want
            engine.risk.paused = want
            engine.log("scan " + ("paused" if want else "resumed"))
    elif head == "crypto":
        on, off = rest in ("on", "1", "true"), rest in ("off", "0", "false")
        if on or off:
            try:
                engine.set_market_enabled("Crypto", on)
            except Exception as e:
                engine.log(f"crypto failed: {e}", "error")
        else:
            engine.log("usage: :crypto on|off")
    elif head == "kill":
        engine.risk.trigger_kill()
        engine.log("kill switch armed by command", "warn")
    elif head == "scan":
        engine.scan_now.set()
        engine.log("scan forced")
    elif head == "report":
        ss["view"] = "report"
    elif head == "view":
        if rest in ("monitor", ""):
            ss["view"] = "monitor"
        elif rest in VIEW_LABELS.values():
            ss["view"] = rest
        else:
            engine.log("usage: :view monitor|positions|analytics|diagnostics|report|log|detail|help")
    elif head == "log":
        if rest in ("all", ""):
            ss["log_level"] = None
        elif rest in ("warn", "warning"):
            ss["log_level"] = "WARN"
        elif rest in ("error", "err"):
            ss["log_level"] = "ERROR"
        else:
            engine.log("usage: :log all|warn|error")
    else:
        ss["text_filter"] = low
        engine.log(f"scanner filter '{low}' (type help for commands)")


def _close_by_arg(engine, arg):
    positions = engine.portfolio.positions
    if not positions:
        engine.log("nothing to close")
        return
    pos = None
    if arg.isdigit():
        i = int(arg) - 1
        if 0 <= i < len(positions):
            pos = positions[i]
    else:
        for p in positions:
            if arg and (p.symbol.lower() == arg or arg in p.symbol.lower()):
                pos = p
                break
    if pos is None:
        engine.log(f"no position matches '{arg}'")
    else:
        engine.close_one(pos)


# --------------------------------------------------------------------------- monitor view


def _status_pill(status):
    cls = {"OPEN": "idts-open", "PRE-CLOSE": "idts-pre", "LUNCH": "idts-pre", "PRE-OPEN": "idts-pre"}.get(status, "idts-closed")
    return f"<span class='idts-pill {cls}'>● {status}</span>"


def render_markets_strip(engine):
    open_m, closed_m = [], []
    for m in engine.markets_cfg:
        status = engine.clock.status(m)["status"]
        (closed_m if status == "CLOSED" else open_m).append(m)

    if open_m:
        per_row = 5 if len(open_m) > 5 else len(open_m)
        for row_start in range(0, len(open_m), per_row):
            batch = open_m[row_start:row_start + per_row]
            cols = st.columns(per_row)
            for i, m in enumerate(batch):
                with cols[i]:
                    _market_card(engine, m)
    if closed_m:
        chips = []
        for m in closed_m:
            code = mkt_code(engine.markets_cfg, m)
            st_ = engine.clock.status(m)
            txt = f"**{code}** CLOSED"
            if st_.get("next_open") is not None:
                txt += f" opens {st_['next_open']:%H:%M}"
            npos = sum(1 for p in engine.portfolio.positions if p.market == m)
            if npos:
                txt += f" · {npos}pos"
            chips.append(txt)
        st.caption("  ·  ".join(chips))


def _market_card(engine, m):
    st_ = engine.clock.status(m)
    status = st_["status"]
    cd = engine.clock.fmt_countdown(st_.get("countdown_seconds"))
    rows = engine.last_scan.get(m, [])
    scanned = sum(1 for r in rows if r.get("price") is not None)
    sigs = sum(1 for r in rows if r.get("signal") in ("LONG", "SHORT"))
    total = len(engine.watchlists.get(m, []))
    changes = [r.get("live_change", r.get("change")) for r in rows]
    changes = [c for c in changes if c is not None]
    breadth = sum(1 for c in changes if c >= 0) / len(changes) * 100 if changes else None
    today = datetime.now(timezone.utc).date()
    closed_today = [t for t in engine.portfolio.trades
                    if t.market == m and t.closed_at.date() == today]
    wins = sum(1 for t in closed_today if t.pnl > 0)
    day_pnl = engine.portfolio.day_pnl_local(m)
    day_inr = _inr_pnl(engine, day_pnl, m)
    eq = engine.portfolio.equity_local(m)
    day_pct = (day_pnl / eq * 100.0) if eq else 0.0
    max_loss = float((engine.risk.risk.get("max_daily_loss_pct", 1.5) or 1.5))
    loss_used = (max(0.0, -day_pnl) / (eq * max_loss / 100.0) * 100.0) if eq and max_loss else 0.0
    npos = sum(1 for p in engine.portfolio.positions if p.market == m)

    when = st_["local_time"].strftime("%H:%M") if st_.get("local_time") else "—"
    second = (f"closes {cd}" if status in ("OPEN", "PRE-CLOSE", "LUNCH")
              else (f"opens {st_['next_open']:%H:%M}" if st_.get("next_open") else f"opens {cd}"))
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>"
            f"<b class='idts-mono' style='font-size:1.0rem'>{mkt_code(engine.markets_cfg, m)}</b>"
            f"{_status_pill(status)}</div>"
            f"<div class='idts-muted idts-mkt-line' style='font-size:0.72rem'>{when} · {second}</div>",
            unsafe_allow_html=True)
        frac = market_session_fraction(st_)
        if frac is not None:
            st.progress(frac)
        b = "—" if breadth is None else f"{breadth:.0f}%"
        st.markdown(
            f"<div class='idts-mono idts-mkt-line' style='font-size:0.8rem;color:{_pnl_color(day_inr)};font-weight:600'>"
            f"{fmt_pnl_compact(day_inr, 'INR')} ({fmt_pct_always(day_pct)})</div>"
            f"<div class='idts-muted idts-mkt-line' style='font-size:0.7rem'>↕{b} · {scanned}/{total} · {sigs}sig · {npos}pos</div>"
            f"<div class='idts-muted idts-mkt-line' style='font-size:0.7rem'>{wins}W/{len(closed_today) - wins}L · loss {loss_used:.0f}/{max_loss:.1f}%</div>",
            unsafe_allow_html=True)


def render_scanner(engine):
    ss = st.session_state
    rows = scanner_rows(engine)
    st.markdown(f"**Scanner** · {len(rows)} rows"
                + (f" · :orange[{ss['text_filter']}]" if ss.get("text_filter") else "")
                + (f" · {ss['signal_filter']}" if ss.get("signal_filter") else "")
                + (f" · sort:{ss.get('sort_mode')}" if ss.get("sort_mode", "score") != "score" else ""))

    if not rows:
        if engine.paused:
            st.info("Scanning paused — press Resume in the sidebar.")
        elif not engine.last_scan:
            st.info("Waiting for the first scan… (or seed demo data from the sidebar)")
        else:
            st.info("No rows match the current filters.")
        return

    table_rows = []
    for r in rows:
        mkt = r.get("market", "")
        price = r.get("live_price") or r.get("price")
        if price is None and hasattr(engine, "ticker"):
            try:
                price = engine.ticker.price(r.get("symbol"))
            except Exception:
                price = None
        chg = r.get("live_change")
        if chg is None:
            chg = r.get("change")
        metrics = r.get("metrics") or {}
        plan = r.get("plan") or {}
        upd = r.get("updated")
        age_txt = "—"
        if hasattr(engine, "ticker"):
            try:
                q = engine.ticker.age(r.get("symbol"))
                if q is not None:
                    age_txt = fmt_age(q)
            except Exception:
                pass
        if age_txt == "—" and upd is not None:
            try:
                age_txt = fmt_age((datetime.now(timezone.utc) - upd).total_seconds())
            except Exception:
                pass
        sig = r.get("signal", "NONE")
        sig_txt = f"▲ {sig}" if sig == "LONG" else (f"▼ {sig}" if sig == "SHORT" else sig)
        why = "quote thread · waiting for scan" if r.get("prefill") else (
            r.get("best_reason_short") or r.get("best_reason") or r.get("gate") or "")
        table_rows.append({
            "Mkt": mkt_code(engine.markets_cfg, mkt),
            "Symbol": ("› " if (r.get("symbol"), mkt) == tuple(ss.get("selected") or ()) else "") + str(r.get("symbol", "")),
            "Last": _inr_pnl(engine, price, mkt) if price is not None else None,
            "%Chg": chg,
            "RVOL": metrics.get("rvol"),
            "ATR%": metrics.get("atr_pct"),
            "ADX": metrics.get("adx"),
            "RSI": metrics.get("rsi"),
            "VWAP%": metrics.get("vwap_dist"),
            "Regime": r.get("regime", "—"),
            "Strategy": short_label(engine, r.get("best")) if r.get("best") else "—",
            "R:R": (plan.get("rr") if plan else None),
            "Setups": r.get("setups") or 0,
            "Score": r.get("score"),
            "Signal": sig_txt,
            "Age": age_txt,
            "Why": str(why)[:48],
        })
    df = pd.DataFrame(table_rows)

    def _chg(v):
        if v is None or pd.isna(v):
            return "—"
        return f"{'▲' if v >= 0 else '▼'}{abs(v):.2f}%"

    styler = (df.style
              .format({"Last": lambda v: "—" if v is None or pd.isna(v) else fmt_money(v, "INR"),
                       "%Chg": _chg,
                       "RVOL": lambda v: "—" if v is None or pd.isna(v) else f"{v:.1f}x",
                       "ATR%": lambda v: "—" if v is None or pd.isna(v) else f"{v:.1f}%",
                       "ADX": lambda v: "—" if v is None or pd.isna(v) else f"{v:.0f}",
                       "RSI": lambda v: "—" if v is None or pd.isna(v) else f"{v:.0f}",
                       "VWAP%": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.1f}%",
                       "R:R": lambda v: "—" if v is None or pd.isna(v) else f"{v:.1f}",
                       "Score": lambda v: "—" if v is None or pd.isna(v) else fmt_score(v)},
                      na_rep="—")
              .map(lambda v: f"color:{'#00d97d' if isinstance(v, (int, float)) and v > 0 else ('#ff4d4d' if isinstance(v, (int, float)) and v < 0 else '')}",
                   subset=["%Chg"])
              .map(lambda v: "color:#00d97d" if v == "▲ LONG" else ("color:#ff4d4d" if v == "▼ SHORT" else "color:#8a8a95"),
                   subset=["Signal"]))

    event = st.dataframe(
        styler, key="scanner_df", height=470, width="stretch",
        hide_index=True, on_select="rerun", selection_mode="single-row",
        column_config={
            "Why": st.column_config.TextColumn("Why", width="large"),
            "Score": st.column_config.TextColumn("Score", width="small"),
        })
    sel = event.selection.rows if event and event.selection else []
    if sel:
        r = df.iloc[sel[0]]
        sym = r["Symbol"].lstrip("› ").strip()
        # recover real (symbol, market) from the raw rows
        match = [row for row in rows
                 if row.get("symbol") == sym
                 and mkt_code(engine.markets_cfg, row.get("market", "")) == r["Mkt"]]
        if match:
            ss["selected"] = (match[0].get("symbol"), match[0].get("market"))


def render_active_trades(engine):
    positions = engine.portfolio.positions
    st.markdown(f"**Active trades** · {len(positions)} open")
    if not positions:
        st.caption("No open positions · a fill appears here the moment a setup signals")
        return
    table_rows = []
    for p in positions:
        upnl, upnl_local = upnl_inr(engine, p)
        pct = (upnl_local / (p.entry * p.qty) * 100) if p.entry and p.qty else 0.0
        r, mfe, mae = position_r(engine, p)
        age = (datetime.now(timezone.utc) - p.opened_at).total_seconds()
        state = "BE" if getattr(p, "moved_to_be", False) else (
            "TRAIL" if ((p.side.value == "LONG" and p.stop > p.entry) or
                        (p.side.value == "SHORT" and p.stop < p.entry)) else "SL")
        mfe_mae = f"{mfe:+.1f}/{mae:+.1f}" if mfe is not None else "—"
        table_rows.append({
            "Mkt": mkt_code(engine.markets_cfg, p.market),
            "Symbol": p.symbol,
            "Side": ("▲ LONG" if p.side.value == "LONG" else "▼ SHORT"),
            "Qty": fmt_qty(p.qty),
            "Strategy": strategy_label(engine, p.strategy),
            "Entry": fmt_price(_inr_pnl(engine, p.entry, p.market), "INR"),
            "Last": fmt_price(_inr_pnl(engine, p.current, p.market), "INR"),
            "Stop": fmt_price(_inr_pnl(engine, p.stop, p.market), "INR"),
            "Target": fmt_price(_inr_pnl(engine, p.target, p.market), "INR"),
            "P&L": fmt_pnl(upnl, "INR"),
            "%": fmt_pct_always(pct),
            "R": fmt_r(r) if r is not None else "—",
            "Age": fmt_age(age),
            "State": state,
            "Gauge": gauge(p.stop, p.current, p.target, 10),
            "MFE/MAE": mfe_mae,
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch",
                 height=min(300, 60 + 36 * len(positions) + 36))
    total_inr = engine.portfolio.unrealized_inr()
    st.caption(f"Total unrealised: **{fmt_pnl(total_inr, 'INR')}** (INR)")

    # close flow: pick a position, arm, confirm
    opts = {f"#{i + 1} {p.symbol} · {p.side.value} · {fmt_qty(p.qty)}": i
            for i, p in enumerate(positions)}
    choice = st.selectbox("Close a position", list(opts), key="close_sel")
    key = f"close:{opts[choice]}"
    if st.button("Close selected", key="close_btn", type="secondary"):
        if _arm(key):
            engine.close_one(positions[opts[choice]])
            st.toast(f"Closed {choice}")
            st.rerun()
        else:
            st.caption(":orange[Armed — press Close selected again within 6s to confirm]")


def render_completed_trades(engine, height=260):
    trades = engine.portfolio.trades
    st.markdown(f"**Completed trades** · {len(trades)}")
    if not trades:
        st.caption("No completed trades yet · stops, targets and square-offs land here")
        return
    table_rows = []
    for t in trades:
        pnl = _inr_pnl(engine, t.pnl, t.market)
        dur = (t.closed_at - t.opened_at).total_seconds()
        table_rows.append({
            "Time": t.closed_at.astimezone().strftime("%H:%M"),
            "Mkt": mkt_code(engine.markets_cfg, t.market),
            "Symbol": t.symbol,
            "Strategy": short_label(engine, t.strategy),
            "Side": ("▲ LONG" if t.side.value == "LONG" else "▼ SHORT"),
            "Qty": fmt_qty(t.qty),
            "Entry": fmt_price(_inr_pnl(engine, t.entry, t.market), "INR"),
            "Exit": fmt_price(_inr_pnl(engine, t.exit, t.market), "INR"),
            "Rsn": exit_reason_text(t.exit_reason),
            "P&L": fmt_pnl(pnl, "INR"),
            "R": fmt_r(t.r_multiple),
            "Dur": (f"{int(dur // 60)}m" if dur < 3600 else f"{dur / 3600:.1f}h"),
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch",
                 height=height)
    st_ = engine.portfolio.stats()
    trades_n = st_["trades"]
    charges = sum(_inr_pnl(engine, t.charges, t.market) for t in trades)
    gross = st_["total_pnl"] + charges
    st.caption(
        f"N {trades_n} · win {fmt_winrate(st_['win_rate'], trades_n)} · "
        f"gross {fmt_pnl(gross, 'INR')} · fees {fmt_money(charges, 'INR')} · "
        f"net **{fmt_pnl(st_['total_pnl'], 'INR')}** · PF {fmt_pf(st_['profit_factor'] if trades_n else None)} · "
        f"exp {fmt_r(st_['expectancy_r']) if trades_n else '—'}")
    pm = {m: v for m, v in engine.portfolio.per_market_pnl().items() if abs(v) >= 1e-9}
    if pm:
        chips = [f"{mkt_code(engine.markets_cfg, m)} {fmt_pnl(_inr_pnl(engine, v, m), 'INR')}"
                 for m, v in sorted(pm.items(), key=lambda kv: -abs(_inr_pnl(engine, kv[1], kv[0])))]
        st.caption("  ·  ".join(chips) + "  (per market, INR)")


def render_risk_panel(engine):
    rows = risk_rows(engine)
    st.markdown("**Risk**")
    if not rows:
        st.caption("No open positions · nothing at risk")
        return
    portfolio = engine.portfolio
    eq = portfolio.total_equity_inr()
    exposure = sum(a["exposure"] for a in rows.values())
    risk = sum(a["risk"] for a in rows.values())
    reward = sum(a["reward"] for a in rows.values())
    day = portfolio.day_pnl_inr()
    max_loss_pct = float(engine.risk.risk.get("max_daily_loss_pct", 0) or 0)
    limit = eq * max_loss_pct / 100.0
    used = max(0.0, -day)
    frac = (used / limit) if limit > 0 else 0.0
    sym_exp = {}
    for p in portfolio.positions:
        sym_exp[p.symbol] = sym_exp.get(p.symbol, 0.0) + engine.portfolio.to_inr(p.current * p.qty, p.market)
    top_sym, top_val = (max(sym_exp.items(), key=lambda kv: kv[1]) if sym_exp else ("—", 0.0))
    conc = top_val / exposure * 100.0 if exposure else 0.0
    st.progress(min(1.0, frac), text=f"Day loss {used / eq * 100 if eq else 0:.2f}% / {max_loss_pct:.1f}% · top {top_sym} {conc:.0f}%")
    table_rows = []
    for m, agg in sorted(rows.items(), key=lambda kv: (-kv[1]["risk"], -kv[1]["exposure"])):
        rr = agg["reward"] / agg["risk"] if agg["risk"] > 0 else None
        table_rows.append({
            "Mkt": mkt_code(engine.markets_cfg, m),
            "Exposure": fmt_money(agg["exposure"], "INR"),
            "Risk": fmt_money(agg["risk"], "INR"),
            "Reward": fmt_money(agg["reward"], "INR"),
            "%Eq": f"{agg['risk'] / eq * 100:.1f}" if eq else "—",
            "R:R": f"{rr:.1f}" if rr is not None else "—",
        })
    rr_total = reward / risk if risk > 0 else None
    table_rows.append({
        "Mkt": "TOTAL", "Exposure": fmt_money(exposure, "INR"), "Risk": fmt_money(risk, "INR"),
        "Reward": fmt_money(reward, "INR"),
        "%Eq": f"{risk / eq * 100:.1f}" if eq else "—",
        "R:R": f"{rr_total:.1f}" if rr_total is not None else "—",
    })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch",
                 height=min(280, 44 + 35 * len(table_rows) + 20))


def render_logs(engine, limit=None, full=False):
    ss = st.session_state
    logs = engine.logs
    if ss.get("log_level") == "ERROR":
        logs = [l for l in logs if "ERROR" in l.upper()]
    elif ss.get("log_level") == "WARN":
        logs = [l for l in logs if "WARN" in l.upper() or "ERROR" in l.upper()]
    lines = []
    for msg in (logs[:limit] if limit else logs):
        tag, _style = level_tag(msg)
        body = strip_level(msg)
        if body.startswith("["):
            try:
                ts_end = body.index("]")
                ts, rest = body[1:ts_end], body[ts_end + 2:]
            except ValueError:
                ts, rest = "", body
        else:
            ts, rest = "", body
        cls = {"ERROR": "idts-err", "WARN": "idts-warn", "SIGNAL": "idts-sig", "ORDER": "idts-ord"}.get(tag.strip(), "")
        lines.append(f"<span class='ts'>{ts}</span> <span class='{cls}'>{tag:<6}</span> {rest}")
    label = "Log"
    if ss.get("log_level"):
        label += f" · {ss['log_level']}+"
    st.markdown(f"**{label}** · {len(logs)}")
    if not lines:
        st.caption("No events yet")
        return
    with st.container(height=(520 if full else 210)):
        st.markdown("<div class='idts-log'>" + "<br>".join(lines) + "</div>",
                    unsafe_allow_html=True)


def render_monitor(engine):
    render_markets_strip(engine)
    render_scanner(engine)
    st.divider()
    left, mid, right = st.columns([4.4, 3.2, 2.4])
    with left:
        render_active_trades(engine)
    with mid:
        render_completed_trades(engine, height=170)
        st.divider()
        render_risk_panel(engine)
    with right:
        lvl = st.selectbox("Log filter", ["All", "WARN+", "ERROR"], key="log_level_sb")
        st.session_state["log_level"] = {"All": None, "WARN+": "WARN", "ERROR": "ERROR"}[lvl]
        render_logs(engine, limit=40)


# --------------------------------------------------------------------------- positions view


def render_positions_view(engine):
    positions = engine.portfolio.positions
    eq = engine.portfolio.total_equity_inr()
    risk = sum(v["risk"] for v in risk_rows(engine).values())
    unreal = engine.portfolio.unrealized_inr()
    st_ = engine.portfolio.stats()
    st.markdown(f"### Positions · {len(positions)} open")
    summary = (f"Equity **{fmt_money(eq, 'INR')}** · unrealised **{fmt_pnl(unreal, 'INR')}** · "
               f"open risk **{fmt_money(risk, 'INR')}** ({risk / eq * 100:.1f}% eq) · "
               f"closed **{st_['trades']}**")
    st.caption(summary)
    render_active_trades(engine)


# --------------------------------------------------------------------------- analytics view


def render_analytics_view(engine):
    st.markdown("### Analytics")
    series = equity_series(engine, 240)
    if series:
        df = pd.DataFrame({"equity": series})
        st.line_chart(df, height=260)
        st.caption(f"start {fmt_money(series[0], 'INR')} · now {fmt_money(series[-1], 'INR')} · "
                   f"change {fmt_pnl(series[-1] - series[0], 'INR')} · "
                   f"high {fmt_money(max(series), 'INR')} · low {fmt_money(min(series), 'INR')}")
    else:
        st.caption("Waiting for the first scan — the equity curve fills in as snapshots arrive.")

    c1, c2 = st.columns([3, 2])
    with c1:
        stats = engine.portfolio.strategy_stats()
        ordered = sorted(stats.items(), key=lambda kv: -kv[1]["pnl"])
        if ordered:
            rs = {}
            for tr in engine.portfolio.trades:
                rs.setdefault(tr.strategy, []).append(tr.r_multiple)
            rows = []
            for name, s in ordered[:20]:
                avg = s["pnl"] / s["trades"] if s["trades"] else 0.0
                rr = sum(rs.get(name, [])) / len(rs[name]) if rs.get(name) else 0.0
                rows.append({"Strategy": strategy_label(engine, name), "N": s["trades"],
                             "Win": fmt_winrate(s["win_rate"], s["trades"]),
                             "P&L": fmt_pnl(s["pnl"], "INR"),
                             "Avg": fmt_pnl(avg, "INR"), "AvgR": fmt_r(rr)})
            st.markdown("**Strategy leaderboard**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         height=min(420, 60 + 35 * len(rows)))
        else:
            st.markdown("**Strategy leaderboard**")
            st.caption("No closed trades yet")
    with c2:
        counts, pnl_map = {}, {}
        for tr in engine.portfolio.trades:
            reason = str(getattr(tr.exit_reason, "value", tr.exit_reason))
            counts[reason] = counts.get(reason, 0) + 1
            pnl_map[reason] = pnl_map.get(reason, 0.0) + _inr_pnl(engine, tr.pnl, tr.market)
        if counts:
            total = sum(counts.values())
            rows = []
            for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                avg = pnl_map[reason] / n if n else 0.0
                rows.append({"Exit": exit_reason_text(reason), "N": n,
                             "Share": f"{n / total * 100:.0f}%",
                             "P&L": fmt_pnl(pnl_map[reason], "INR"),
                             "Avg": fmt_pnl(avg, "INR")})
            st.markdown("**Exit reasons**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         height=min(320, 60 + 35 * len(rows)))
        else:
            st.markdown("**Exit reasons**")
            st.caption("No exits yet")
        buckets = {}
        for tr in engine.portfolio.trades:
            h = tr.closed_at.astimezone(IST).hour
            b = buckets.setdefault(h, [0, 0.0])
            b[0] += 1
            b[1] += _inr_pnl(engine, tr.pnl, tr.market)
        if buckets:
            dfh = pd.DataFrame(
                {"hour": [f"{h:02d}" for h in sorted(buckets)],
                 "trades": [buckets[h][0] for h in sorted(buckets)]}).set_index("hour")
            st.markdown("**Trades by hour (IST)**")
            st.bar_chart(dfh, height=200)


# --------------------------------------------------------------------------- report view


def _group_table(engine, title, groups, labeler, key_label):
    if not groups:
        st.caption(f"{title}: no closed trades yet")
        return
    rows = []
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["pnl"]):
        pf = g["pf"]
        pf_txt = "∞" if pf == float("inf") else f"{pf:.2f}"
        rows.append({key_label: labeler(key) if labeler else str(key),
                     "N": g["n"], "Win": fmt_winrate(g["win_rate"], g["n"]),
                     "Net": fmt_pnl(g["pnl"], "INR"), "PF": pf_txt, "AvgR": fmt_r(g["avg_r"])})
    st.markdown(f"**{title}**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_report_view(engine):
    st.markdown("### Session report · closed trades")
    b = report_bundle(engine)
    if not b["trades"]:
        st.caption("No closed trades yet — the report fills in as trades close.")
        return
    st_ = b["stats"]
    st.caption(
        f"Metric averages — RSI {b['metric_summary']['avg_rsi']} · ADX {b['metric_summary']['avg_adx']} · "
        f"ATR {b['metric_summary']['avg_atr_pct']} · RVOL {b['metric_summary']['avg_rvol']}")

    cols = st.columns(6)
    metrics = [
        ("Closed", str(st_["trades"])),
        ("Wins / Losses", f"{st_['wins']} / {st_['losses']}"),
        ("Win rate", fmt_winrate(st_["win_rate"], st_["trades"])),
        ("Net P&L", fmt_pnl(st_["total_pnl"], "INR")),
        ("PF", fmt_pf(st_["profit_factor"])),
        ("Expectancy", fmt_r(st_["expectancy_r"])),
        ("Avg win", fmt_pnl(st_["avg_win"], "INR")),
        ("Avg loss", fmt_pnl(st_["avg_loss"], "INR")),
        ("Charges", fmt_money(b["charges"], "INR")),
        ("Avg trade", fmt_pnl(st_["total_pnl"] / st_["trades"] if st_["trades"] else 0.0, "INR")),
        ("Best", (fmt_pnl(b["best"][1], "INR") + f" {b['best'][0].symbol}") if b["best"] else "—"),
        ("Worst", (fmt_pnl(b["worst"][1], "INR") + f" {b['worst'][0].symbol}") if b["worst"] else "—"),
    ]
    for i, (label, value) in enumerate(metrics):
        with cols[i % 6]:
            st.metric(label, value)

    sides = b["sides"]
    side_bits = []
    for side, label in (("LONG", "Long"), ("SHORT", "Short")):
        s = sides.get(side)
        if s:
            wr = s["w"] / s["n"] * 100 if s["n"] else 0.0
            side_bits.append(f"{label} {s['n']} · {wr:.0f}% · {fmt_pnl_compact(s['pnl'], 'INR')}")
        else:
            side_bits.append(f"{label} 0")
    side_bits.append(f"avg hold {fmt_age(b['avg_hold'])}")
    st.caption("  ·  ".join(side_bits))

    flagged = [(n, d) for n, d in sorted(b["strategy_drift"].items()) if d["drift_flag"]]
    if flagged:
        bits = [f"{n} {d['trailing_r']:+.2f}R vs {d['overall_r']:+.2f}R ({d['drift_pct']:+.0f}%)"
                for n, d in flagged]
        st.warning("Expectancy drift: " + " · ".join(bits))
    else:
        st.caption("Expectancy drift: none detected")

    c1, c2 = st.columns([3, 2])
    with c1:
        if len(b["cum"]) > 1:
            st.line_chart(pd.DataFrame({"cum_pnl": b["cum"][1:]}), height=240,
                          y_label="Cumulative P&L (INR)")
            st.caption(f"peak {fmt_pnl_compact(max(b['cum']), 'INR')} · low {fmt_pnl_compact(min(b['cum']), 'INR')} · "
                       f"now {fmt_pnl_compact(b['cum'][-1], 'INR')}")
    with c2:
        dfb = pd.DataFrame(b["buckets"], columns=["R", "trades"]).set_index("R")
        st.bar_chart(dfb, height=240)
        st.caption(f"R distribution · avg {fmt_r(st_['expectancy_r'])}")

    c1, c2, c3 = st.columns(3)
    with c1:
        _group_table(engine, "By market", b["markets"],
                     lambda m: mkt_code(engine.markets_cfg, m), "Mkt")
    with c2:
        _group_table(engine, "By strategy", b["strategies"],
                     lambda s: strategy_label(engine, s), "Strategy")
    with c3:
        _group_table(engine, "By exit", b["exits"],
                     lambda e: exit_reason_text(e), "Exit")

    st.markdown("**Recent closes · newest first**")
    rows = []
    for t in list(reversed(b["trades"]))[:20]:
        pnl = _inr_pnl(engine, t.pnl, t.market)
        dur = (t.closed_at - t.opened_at).total_seconds()
        rows.append({"Time": t.closed_at.astimezone().strftime("%H:%M"),
                     "Mkt": mkt_code(engine.markets_cfg, t.market),
                     "Symbol": t.symbol,
                     "Strategy": short_label(engine, t.strategy),
                     "Side": ("▲ LONG" if t.side.value == "LONG" else "▼ SHORT"),
                     "Exit": exit_reason_text(t.exit_reason),
                     "Qty": fmt_qty(t.qty),
                     "Held": fmt_age(dur),
                     "Net": fmt_pnl(pnl, "INR"), "R": fmt_r(t.r_multiple)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- diagnostics view


def render_diagnostics_view(engine):
    st.markdown("### Diagnostics")
    cs = getattr(engine, "last_cycle_stats", {}) or {}
    scanned = int(cs.get("scanned", 0))
    base = max(1, scanned)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Signal funnel · last cycle**")
        stages = [("Scanned", scanned), ("Signals", int(cs.get("signals", 0))),
                  ("Candidates", int(cs.get("candidates", 0))),
                  ("Blocked (gate)", int(cs.get("blocked_gate", 0))),
                  ("Opened", int(cs.get("opened", 0))), ("Stale", int(cs.get("stale", 0)))]
        if not cs:
            st.caption("Waiting for the first scan")
        for name, value in stages:
            st.progress(min(1.0, value / base if base else 0.0), text=f"{name}: {value}")
    with c2:
        st.markdown("**Timings**")
        st.metric("Last scan", f"{engine.last_scan_elapsed:.2f}s")
        age = None
        quotes = 0
        if hasattr(engine, "ticker"):
            try:
                age = engine.ticker.age_of_newest()
                quotes = len(engine.ticker.snapshot())
            except Exception:
                pass
        feed = fmt_age(age) if age is not None else ("off" if not engine.live_quotes else "—")
        threads = sum(1 for t in threading.enumerate() if t.is_alive())
        st.metric("Quote feed", feed, delta=(f"{quotes} quotes" if engine.live_quotes else None),
                  delta_color="off")
        st.metric("Threads alive", threads)
        st.caption(", ".join(t.name for t in threading.enumerate() if t.is_alive()))

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Rejections · session**")
        counts = dict(getattr(engine, "skip_counts", {}) or {})
        if not counts:
            st.caption("Nothing skipped this session")
        else:
            mx = max(counts.values())
            rows = []
            for reason, n in sorted(counts.items(), key=lambda kv: -kv[1])[:14]:
                rows.append({"Reason": reason, "N": n, "Bar": "▇" * max(1, int(n / mx * 16))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         height=min(420, 60 + 35 * len(rows)))
    with c2:
        st.markdown("**Data health · all markets**")
        rows = []
        for m in engine.markets_cfg:
            status = engine.clock.status(m)["status"]
            mrows = engine.last_scan.get(m, [])
            stale = sum(1 for r in mrows if r.get("stale"))
            sigs = sum(1 for r in mrows if r.get("signal") in ("LONG", "SHORT"))
            ages = []
            for r in mrows:
                up = r.get("updated")
                if up is not None:
                    try:
                        ages.append((datetime.now(timezone.utc) - up).total_seconds())
                    except Exception:
                        pass
            prov = engine.providers.get(m)
            src = getattr(prov, "primary", prov).name if prov is not None and hasattr(prov, "primary") else (getattr(prov, "name", "?") if prov else "?")
            rows.append({"Mkt": mkt_code(engine.markets_cfg, m), "Status": status,
                         "Rows": len(mrows), "Signals": sigs, "Stale": stale,
                         "Age": fmt_age(min(ages)) if ages else "—",
                         "Source": str(src)[:12],
                         "Live": "live" if engine.live_quotes else "off"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- detail view


def score_bar_safe(v):
    if v is None:
        return "▱▱▱▱▱"
    filled = max(0, min(5, int(round(v * 5))))
    return "▰" * filled + "▱" * (5 - filled)


def render_detail_view(engine):
    ss = st.session_state
    selected = ss.get("selected")
    row = None
    if selected:
        row = find_scan_row(engine, selected[0], selected[1])
    if row is None:
        # fall back: first actionable row, else first row of any market
        for m in engine.markets_cfg:
            rows_ = engine.last_scan.get(m, [])
            act = [r for r in rows_ if r.get("signal") in ("LONG", "SHORT")]
            if act:
                row = act[0]
                break
            if rows_ and row is None:
                row = rows_[0]
    if row is None:
        st.markdown("### Detail")
        st.caption("No symbol selected. Pick a row in the Scanner (or seed demo data) — the selection is remembered.")
        return

    sym, mkt = row.get("symbol", ""), row.get("market", "")
    price = row.get("live_price") or row.get("price")
    if price is None and hasattr(engine, "ticker"):
        try:
            price = engine.ticker.price(sym)
        except Exception:
            price = None
    chg = row.get("live_change")
    if chg is None:
        chg = row.get("change")
    ss["selected"] = (sym, mkt)

    price_txt = fmt_price(_inr_pnl(engine, price, mkt), "INR") if price is not None else "—"
    chg_txt = (f"▲{chg:+.2f}%" if chg is not None and chg >= 0 else
               (f"▼{abs(chg):.2f}%" if chg is not None else "—"))
    sig = row.get("signal", "NONE")
    st.markdown(f"### {sym} · {mkt_code(engine.markets_cfg, mkt)}")
    sig_html = (f"<span style='color:#00d97d'>▲ {sig}</span>" if sig == "LONG" else
                (f"<span style='color:#ff4d4d'>▼ {sig}</span>" if sig == "SHORT" else
                 f"<span class='idts-muted'>{sig}</span>"))
    st.markdown(f"<div class='idts-mono' style='font-size:1.1rem'>{price_txt} "
                f"<span style='color:{_pnl_color(chg or 0)}'>{chg_txt}</span> &nbsp; {sig_html} &nbsp; "
                f"<span class='idts-muted'>regime {row.get('regime', '—')}</span> · "
                f"score {fmt_score(row.get('score'))} {score_bar_safe(row.get('score'))}</div>",
                unsafe_allow_html=True)

    plan = row.get("plan") or {}
    if plan:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", fmt_price(_inr_pnl(engine, plan.get("entry"), mkt), "INR") if plan.get("entry") is not None else "—")
        c2.metric("Stop", fmt_price(_inr_pnl(engine, plan.get("stop"), mkt), "INR") if plan.get("stop") is not None else "—")
        c3.metric("Target", fmt_price(_inr_pnl(engine, plan.get("target"), mkt), "INR") if plan.get("target") is not None else "—")
        c4.metric("R:R", f"{plan['rr']:.1f}" if plan.get("rr") is not None else "—")
    else:
        st.caption("Plan: no qualifying setup right now")
    why = str(row.get("best_reason") or row.get("gate") or "—")
    st.caption(f"Why: {why}" + (f"  ·  gate {row.get('gate')}" if row.get("gate") and row.get("best_reason") else ""))

    metrics = row.get("metrics") or {}
    tiles = [
        ("ATR%", metrics.get("atr_pct"), "{:.2f}%"), ("RVOL", metrics.get("rvol"), "{:.2f}x"),
        ("ADX", metrics.get("adx"), "{:.1f}"), ("RSI", metrics.get("rsi"), "{:.1f}"),
        ("VWAP dist", metrics.get("vwap_dist"), "{:+.2f}%"),
        ("EMA20", metrics.get("ema20"), "{:.2f}"), ("EMA50", metrics.get("ema50"), "{:.2f}"),
        ("MACD hist", metrics.get("macd_hist"), "{:.2f}"),
        ("Stoch K", metrics.get("stoch_k"), "{:.1f}"), ("Williams %R", metrics.get("willr"), "{:.1f}"),
        ("CCI", metrics.get("cci"), "{:.1f}"), ("BB pos", metrics.get("bb_pos"), "{:.1f}%"),
    ]
    tiles = [t for t in tiles if t[1] is not None]
    if tiles:
        cols = st.columns(6)
        for i, (label, value, fmt_) in enumerate(tiles):
            with cols[i % 6]:
                st.metric(label, fmt_.format(value))
        if metrics.get("vwap") is not None:
            st.caption(f"VWAP {fmt_price(_inr_pnl(engine, metrics['vwap'], mkt), 'INR')}" +
                       (f" · Supertrend {'UP' if (metrics.get('supertrend') or 0) > 0 else 'DN'}"
                        if metrics.get("supertrend") is not None else ""))

    st_ = row.get("stats") or {}
    if st_:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", st_.get("trades", 0))
        c2.metric("Win rate", f"{st_.get('win_rate', 0):.0f}%")
        c3.metric("PF", fmt_pf(st_.get("profit_factor")))
        c4.metric("Expectancy", fmt_r(st_["expectancy_r"]) if st_.get("expectancy_r") is not None else "—")

    firings = list(row.get("firings") or [])
    firings.sort(key=lambda f: f.get("confidence", 0) or 0, reverse=True)
    if firings:
        st.markdown("**Setups firing**")
        rows_ = []
        for f_ in firings[:3]:
            rows_.append({
                "Setup": f_.get("setup") or f_.get("name") or "—",
                "Side": f_.get("side", "—"),
                "Conf": f"{f_.get('confidence', 0):.2f}" if f_.get("confidence") is not None else "—",
                "Entry": (fmt_price(_inr_pnl(engine, f_["entry"], mkt), "INR") if f_.get("entry") is not None else "—"),
                "Stop": (fmt_price(_inr_pnl(engine, f_["stop"], mkt), "INR") if f_.get("stop") is not None else "—"),
                "Target": (fmt_price(_inr_pnl(engine, f_["target"], mkt), "INR") if f_.get("target") is not None else "—"),
                "Reason": str(f_.get("reason", ""))[:60],
            })
        st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch")

    chart_vals = list(metrics.get("chart") or [])
    if not chart_vals:
        chart_vals = list(getattr(engine, "price_history", {}).get(sym, []) or [])[-60:]
    if price is not None:
        chart_vals.append(price)
    if len(chart_vals) >= 2:
        st.line_chart(pd.DataFrame({"price": chart_vals}), height=180)
    else:
        st.caption("collecting quotes…")


# --------------------------------------------------------------------------- help view


def render_help_view(engine):
    st.markdown("### Keys → Controls")
    st.caption("The terminal dashboard is keyboard-driven; every action has a control here instead.")
    rows = [
        ("q", "Close the browser tab / Ctrl+C the `streamlit` process"),
        ("x — Flat", "Flatten ALL button in the sidebar (arm → confirm)"),
        ("1..9 — quick close", "Close-a-position picker + 2-step Close button"),
        ("p — Pause / resume", "Pause / Resume button in the sidebar"),
        ("Tab — cycle market", "Market dropdown in the sidebar"),
        ("←→ — page", "Tables scroll; no pagination needed"),
        ("↑↓ — select row", "Row selection in the Scanner table"),
        ("d / Enter — detail", "Detail view in the sidebar navigation"),
        ("f — force scan", "Force scan button / `scan` command"),
        ("s — cycle sort", "Sort dropdown / `sort score|chg|signal|symbol`"),
        ("g — cycle signals", "Signal dropdown / `longs` `shorts` `all`"),
        ("v — live quotes", "Live quotes toggle"),
        ("c — crypto on/off", "Crypto toggle / `crypto on|off`"),
        ("m / i / a / y / r / l", "View radio: Monitor / Positions / Analytics / Diagnostics / Report / Log"),
        ("e — log filter", "Filter dropdown above the log / `log all|warn|error`"),
        (": or / — command bar", "Command box at the bottom of the sidebar"),
        ("h / ? — help", "This page"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Key", "In this UI"]),
                 hide_index=True, width="stretch",
                 height=min(640, 60 + 35 * len(rows)))
    st.markdown("**Commands**")
    cmds = [
        ("filter <text>", "filter scanner by symbol/strategy; `filter off` clears"),
        ("sort score|chg|signal|symbol", "change the scanner sort"),
        ("longs / shorts / all", "show only long / short / all signals"),
        ("close N or SYMBOL", "close one position (paper), e.g. `close 1`"),
        ("flat", "square-off every position"),
        ("pause / resume", "pause or resume scanning"),
        ("crypto on|off", "turn the crypto market on/off"),
        ("kill", "trip the daily-kill switch"),
        ("scan", "force a scan"),
        ("report", "open the session report"),
        ("view monitor|positions|analytics|diagnostics|report|log|detail", "switch view"),
        ("log all|warn|error", "filter the log"),
    ]
    st.dataframe(pd.DataFrame(cmds, columns=["Command", "What it does"]),
                 hide_index=True, width="stretch",
                 height=min(520, 60 + 35 * len(cmds)))


# --------------------------------------------------------------------------- body & main


def render_footer(engine):
    bits = [f"scan {engine.last_scan_elapsed:.1f}s"]
    if engine.scan_progress:
        bits.append(f"scanning {engine.scan_progress}")
    elif not engine.paused:
        nxt = getattr(engine, "next_scan_at", 0)
        if nxt:
            bits.append(f"next scan in {max(0, int(nxt - time.time()))}s")
    if engine.paused:
        bits.append("PAUSED")
    ages = []
    for mrows in engine.last_scan.values():
        for r in mrows:
            up = r.get("updated")
            if up is not None:
                try:
                    ages.append((datetime.now(timezone.utc) - up).total_seconds())
                except Exception:
                    pass
    if ages:
        bits.append(f"data age {int(min(ages))}s")
    bits.append("paper" if engine.mode != "live" else "LIVE")
    st.caption(" · ".join(bits))


def render_scanner_view(engine):
    """Nine dedicated scanner views over the latest completed-candle scan."""
    st.subheader("Scanner")
    st.caption("Ranked from the last completed-candle scan. Quotes are labelled "
               "by source and age; signal prices and executable quotes are kept separate.")
    choice = st.selectbox("View", list(VIEW_NAMES), key="scanner_view_select")
    limit = st.number_input("Rows", min_value=5, max_value=200, value=25, step=5,
                            key="scanner_view_limit")
    rows = ScannerViews(engine).build(choice, int(limit))
    if not rows:
        st.info(f"No candidates in {choice}.")
        return
    st.dataframe(pd.DataFrame([r.to_dict() for r in rows]), hide_index=True,
                 width="stretch")


def render_body(engine):
    render_header(engine)
    view = st.session_state.get("view", "monitor")
    if view == "monitor":
        render_monitor(engine)
    elif view == "scanner":
        render_scanner_view(engine)
    elif view == "positions":
        render_positions_view(engine)
    elif view == "analytics":
        render_analytics_view(engine)
    elif view == "diagnostics":
        render_diagnostics_view(engine)
    elif view == "report":
        render_report_view(engine)
    elif view == "log":
        render_logs(engine, full=True)
    elif view == "detail":
        render_detail_view(engine)
    elif view == "help":
        render_help_view(engine)
    st.divider()
    render_footer(engine)


def main():
    st.set_page_config(page_title="IDTS Trader", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    raw_mode = _requested_mode()
    live_requested = raw_mode == "live"
    raw = _raw_config()

    if live_requested and not bool(raw.get("live_trading_confirmed", False)):
        st.error("Live trading requested (IDTS_MODE=live) but `live_trading_confirmed: true` "
                 "is not set in trader/config/settings.yaml. Refusing to start — "
                 "the terminal dashboard enforces the same gate.")
        st.stop()

    if live_requested and not st.session_state.get("live_ok"):
        st.warning("LIVE TRADING MODE")
        st.write("Orders will be sent to the real broker. Type **YES I UNDERSTAND** to arm this session.")
        phrase = st.text_input("Confirmation phrase", key="live_phrase")
        if phrase.strip() == "YES I UNDERSTAND":
            st.session_state["live_ok"] = True
            st.rerun()
        st.stop()

    mode = "live" if live_requested else "paper"
    demo = os.environ.get("IDTS_DEMO") == "1"
    engine = get_engine(mode, live_requested, demo)

    ss = st.session_state
    ss.setdefault("view", "monitor")
    ss.setdefault("market_filter", None)
    ss.setdefault("signal_filter", None)
    ss.setdefault("text_filter", "")
    ss.setdefault("sort_mode", "score")
    ss.setdefault("selected", None)
    ss.setdefault("log_level", None)
    ss.setdefault("refresh_s", 2)

    render_sidebar(engine)

    refresh = int(ss.get("refresh_s") or 0)
    if refresh:
        @st.fragment(run_every=refresh)
        def _live():
            render_body(engine)
        _live()
    else:
        render_body(engine)


if __name__ == "__main__":
    main()
