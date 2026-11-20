"""IDTS HTML Dashboard Server.

Serves a browser-based dashboard that reads from the same SQLite database
as the terminal dashboard. Zero external dependencies beyond PyYAML (already
a project requirement).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yaml

from trader.core.market_clock import MarketClock
from trader.core.storage import Storage

log = logging.getLogger("idts.dashboard")

DB_PATH = os.environ.get("IDTS_DB_PATH", "data/trader.db")
MARKETS_FILE = os.environ.get("IDTS_MARKETS_FILE", "trader/config/markets.yaml")
SETTINGS_FILE = os.environ.get("IDTS_SETTINGS_FILE", "trader/config/settings.yaml")
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.environ.get("IDTS_DASHBOARD_PORT", "8765"))

_markets_cfg_cache = None
_settings_cache = None


def get_markets_cfg() -> dict:
    global _markets_cfg_cache
    if _markets_cfg_cache is None:
        with open(MARKETS_FILE) as f:
            _markets_cfg_cache = yaml.safe_load(f).get("markets", {})
    return _markets_cfg_cache


def get_settings() -> dict:
    global _settings_cache
    if _settings_cache is None:
        with open(SETTINGS_FILE) as f:
            _settings_cache = yaml.safe_load(f) or {}
    return _settings_cache


def get_storage() -> Storage:
    return Storage(DB_PATH)


def get_clock() -> MarketClock:
    return MarketClock(get_markets_cfg())


def fmt_money(value: float, ccy: str = "INR") -> str:
    sym = {"INR": "\u20b9", "USD": "$", "JPY": "\u00a5", "GBP": "\u00a3",
           "HKD": "HK$", "AUD": "A$", "KRW": "\u20a9", "BRL": "R$",
           "USDT": "US$", "EUR": "\u20ac"}.get(ccy, ccy)
    if value < 0:
        return f"-{sym}{abs(value):,.2f}"
    return f"{sym}{value:,.2f}"


def fmt_age(seconds) -> str:
    if seconds is None:
        return "\u2014"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}"


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "IDTS-Dashboard/1.0"

    def log_message(self, format, *args):
        log.debug("%s - %s", self.address_string(), format % args)

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, status=200):
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/markets":
            self._handle_markets()
        elif path == "/api/scanner":
            self._handle_scanner()
        elif path == "/api/positions":
            self._handle_positions()
        elif path == "/api/trades":
            self._handle_trades()
        elif path == "/api/equity":
            self._handle_equity()
        elif path == "/api/signals":
            self._handle_signals()
        elif path == "/api/stats":
            self._handle_stats()
        elif path == "/api/logs":
            self._handle_logs()
        elif path == "/api/config":
            self._handle_config()
        elif path == "/api/scan" and self.command == "POST":
            self._handle_scan()
        elif path == "/" or path == "":
            self._send_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            self._send_file(STATIC_DIR / rel)
        else:
            self._send_json({"error": "not found"}, 404)

    # ------------------------------------------------------------------ markets

    def _handle_markets(self):
        clock = get_clock()
        markets_cfg = get_markets_cfg()
        pos_count = self._positions_count()
        day_pnl = self._day_pnl_by_market()

        result = []
        for market, cfg in markets_cfg.items():
            st = clock.status(market)
            status = st["status"]
            local_time = st.get("local_time")
            countdown = st.get("countdown_seconds")

            result.append({
                "market": market,
                "code": cfg.get("code", market[:2].upper()),
                "exchange": cfg.get("exchange", ""),
                "status": status,
                "local_time": local_time.isoformat() if local_time else None,
                "countdown_seconds": countdown,
                "countdown_text": self._format_countdown(countdown),
                "positions": pos_count.get(market, 0),
                "day_pnl": day_pnl.get(market, 0),
                "currency": cfg.get("currency", "INR"),
                "is_open": status in ("OPEN", "PRE-CLOSE"),
                "always_open": bool(cfg.get("always_open")),
            })

        self._send_json(result)

    def _format_countdown(self, seconds):
        if seconds is None:
            return "\u2014"
        s = max(0, int(seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{sec:02d}s"
        return f"{sec}s"

    # ------------------------------------------------------------------ scanner

    def _handle_scanner(self):
        storage = get_storage()
        rows = []
        if hasattr(storage, "load_scan_cache"):
            rows = storage.load_scan_cache() or []

        # Fall back to signals table if no scan cache
        if not rows:
            signals = self._get_recent_signals(100)
            rows = [{
                "symbol": s.get("symbol", ""),
                "market": s.get("market", ""),
                "price": s.get("entry", 0),
                "change": None,
                "signal": s.get("side", "NONE"),
                "score": s.get("confidence", 0),
                "strategy": s.get("strategy", ""),
                "regime": s.get("regime", ""),
                "reason": s.get("reason", ""),
                "updated": s.get("ts", ""),
                "stale": False,
            } for s in signals]

        self._send_json(rows)

    # ------------------------------------------------------------------ positions

    def _handle_positions(self):
        storage = get_storage()
        positions = storage.load_open_positions()

        result = []
        for p in positions:
            upnl = (p.current - p.entry) * p.qty if p.side.value == "LONG" else (p.entry - p.current) * p.qty
            pct = (upnl / (p.entry * p.qty) * 100) if p.entry and p.qty else 0
            age = (datetime.now(timezone.utc) - p.opened_at).total_seconds()

            result.append({
                "id": p.id,
                "symbol": p.symbol,
                "market": p.market,
                "side": p.side.value,
                "strategy": p.strategy,
                "qty": p.qty,
                "entry": p.entry,
                "current": p.current,
                "stop": p.stop,
                "target": p.target,
                "upnl": upnl,
                "pct": pct,
                "age": age,
                "age_text": fmt_age(age),
                "trailing_stop": getattr(p, "trailing_stop", None),
                "moved_to_be": bool(getattr(p, "moved_to_be", False)),
            })

        self._send_json(result)

    def _positions_count(self) -> dict:
        storage = get_storage()
        positions = storage.load_open_positions()
        counts = {}
        for p in positions:
            counts[p.market] = counts.get(p.market, 0) + 1
        return counts

    def _day_pnl_by_market(self) -> dict:
        trades = self._get_recent_trades(500)
        day = {}
        for t in trades:
            day[t["market"]] = day.get(t["market"], 0) + t.get("pnl", 0)
        return day

    # ------------------------------------------------------------------ trades

    def _get_recent_trades(self, limit=100):
        storage = get_storage()
        with storage._conn() as c:
            rows = c.execute(
                "SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            cols = [d[0] for d in c.description]
        return [dict(zip(cols, r)) for r in rows]

    def _handle_trades(self):
        limit = int(parse_qs(urlparse(self.path).query).get("limit", ["100"])[0])
        trades = self._get_recent_trades(limit)
        result = [{
            "id": t["id"],
            "symbol": t["symbol"],
            "market": t["market"],
            "side": t["side"],
            "strategy": t["strategy"],
            "qty": t["qty"],
            "entry": t["entry"],
            "exit": t["exit"],
            "pnl": t["pnl"],
            "exit_reason": t["exit_reason"],
            "opened_at": t["opened_at"],
            "closed_at": t["closed_at"],
            "r_multiple": t.get("r_multiple", 0),
        } for t in trades]
        self._send_json(result)

    # ------------------------------------------------------------------ equity

    def _handle_equity(self):
        storage = get_storage()
        rows = storage.load_equity_history(240)
        result = [{
            "ts": ts.isoformat(),
            "equity": equity,
            "day_pnl": day_pnl,
        } for ts, equity, day_pnl in rows]
        self._send_json(result)

    # ------------------------------------------------------------------ signals

    def _get_recent_signals(self, limit=100):
        storage = get_storage()
        with storage._conn() as c:
            rows = c.execute(
                "SELECT * FROM signals ORDER BY ts DESC LIMIT ?",
                (limit,)
            ).fetchall()
            cols = [d[0] for d in c.description]
        return [dict(zip(cols, r)) for r in rows]

    def _handle_signals(self):
        limit = int(parse_qs(urlparse(self.path).query).get("limit", ["50"])[0])
        self._send_json(self._get_recent_signals(limit))

    # ------------------------------------------------------------------ stats

    def _handle_stats(self):
        trades = self._get_recent_trades(1000)
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
        total = len(trades)
        win_rate = (wins / total * 100) if total else 0

        storage = get_storage()
        positions = storage.load_open_positions()
        unrealized = sum(
            (p.current - p.entry) * p.qty if p.side.value == "LONG" else (p.entry - p.current) * p.qty
            for p in positions
        )

        markets = {}
        for t in trades:
            m = t["market"]
            if m not in markets:
                markets[m] = {"pnl": 0.0, "wins": 0, "trades": 0}
            markets[m]["pnl"] += t.get("pnl", 0)
            markets[m]["trades"] += 1
            if t.get("pnl", 0) > 0:
                markets[m]["wins"] += 1

        self._send_json({
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "open_positions": len(positions),
            "unrealized_pnl": unrealized,
            "markets": markets,
        })

    # ------------------------------------------------------------------ logs / config

    def _handle_logs(self):
        self._send_json([])

    def _handle_config(self):
        cfg = get_settings()
        self._send_json({
            "mode": cfg.get("mode", "paper"),
            "live_trading_confirmed": bool(cfg.get("live_trading_confirmed", False)),
            "scan_interval_seconds": cfg.get("scan_interval_seconds", 60),
            "live_quotes": bool(cfg.get("live_quotes", True)),
            "risk_per_trade_pct": (cfg.get("risk", {}) or {}).get("risk_per_trade_pct", 0.5),
            "max_daily_loss_pct": (cfg.get("risk", {}) or {}).get("max_daily_loss_pct", 1.0),
        })

    def _handle_scan(self):
        """Placeholder scan trigger. The HTML dashboard is a read-only client;
        actual scanning is driven by the engine (python -m trader). This
        endpoint acknowledges the request so the frontend can show feedback."""
        self._send_json({"status": "accepted", "message": "Start the engine with `python -m trader` to run scans"})


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    log.info("IDTS HTML Dashboard on http://0.0.0.0:%d", PORT)
    log.info("DB: %s", DB_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
