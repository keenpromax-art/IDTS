from __future__ import annotations

import sqlite3
import os
from datetime import datetime
from typing import List, Optional

from trader.core.models import Trade, Signal, Order, Position, Side, ExitReason


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init()

    def _conn(self):
        # the scan thread and the quote thread can both write, so give the
        # connection room and put the database in WAL mode
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return conn

    def _init(self):
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY, symbol TEXT, market TEXT, side TEXT, strategy TEXT,
                qty REAL, entry REAL, exit REAL, opened_at TEXT, closed_at TEXT,
                exit_reason TEXT, charges REAL, score_reason TEXT, pnl REAL, stop REAL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, market TEXT, strategy TEXT,
                side TEXT, entry REAL, stop REAL, target REAL, confidence REAL, ts TEXT, reason TEXT
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY, symbol TEXT, market TEXT, side TEXT, qty REAL,
                order_type TEXT, price REAL, status TEXT, filled_price REAL, ts TEXT, charges REAL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, equity REAL, day_pnl REAL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY, symbol TEXT, market TEXT, side TEXT, strategy TEXT,
                qty REAL, entry REAL, stop REAL, target REAL, current REAL, opened_at TEXT,
                charges REAL, score_reason TEXT, trailing_stop REAL, moved_to_be INTEGER
            )""")
            try:
                c.execute("ALTER TABLE trades ADD COLUMN stop REAL")
            except Exception:
                pass
            for col in ("trailing_stop REAL", "moved_to_be INTEGER"):
                try:
                    c.execute(f"ALTER TABLE positions ADD COLUMN {col}")
                except Exception:
                    pass
            # indexes for the hot read paths
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market)",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at)",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts)",
                "CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market)",
                "CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts)",
            ):
                try:
                    c.execute(stmt)
                except Exception:
                    pass

    def save_trade(self, t: Trade):
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (t.id, t.symbol, t.market, t.side.value, t.strategy, t.qty, t.entry, t.exit,
                       t.opened_at.isoformat(), t.closed_at.isoformat(), t.exit_reason.value,
                       t.charges, t.score_reason, t.pnl, t.stop))

    def save_signal(self, symbol, market, sig: Signal):
        with self._conn() as c:
            c.execute("""INSERT INTO signals (symbol,market,strategy,side,entry,stop,target,confidence,ts,reason)
                         VALUES (?,?,?,?,?,?,?,?,?,?)""",
                      (symbol, market, sig.strategy, sig.side.value, sig.entry, sig.stop, sig.target,
                       sig.confidence, sig.ts.isoformat(), sig.reason))

    def save_order(self, o: Order):
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                      (o.id, o.symbol, o.market, o.side.value, o.qty,
                       o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type),
                       o.price, o.status.value if hasattr(o.status, "value") else str(o.status),
                       o.filled_price, o.ts.isoformat(), o.charges))

    def save_equity_snapshot(self, equity, day_pnl):
        with self._conn() as c:
            c.execute("INSERT INTO equity_snapshots (ts,equity,day_pnl) VALUES (?,?,?)",
                      (datetime.now().isoformat(), equity, day_pnl))

    def load_equity_history(self, limit: int = 240):
        """Newest equity snapshots, oldest first.

        The dashboard uses this to seed its sparkline, day hi/lo and
        drawdown before the live history has accumulated.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts,equity,day_pnl FROM equity_snapshots ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),)).fetchall()
        out = []
        for ts, equity, day_pnl in reversed(rows):
            try:
                when = datetime.fromisoformat(ts)
            except Exception:
                continue
            out.append((when, float(equity), float(day_pnl)))
        return out

    def save_position(self, p: Position):
        with self._conn() as c:
            c.execute("""INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (p.id, p.symbol, p.market, p.side.value, p.strategy, p.qty, p.entry, p.stop,
                       p.target, p.current, p.opened_at.isoformat(), p.charges, p.score_reason,
                       getattr(p, "trailing_stop", None), 1 if getattr(p, "moved_to_be", False) else 0))

    def delete_position(self, pid: str):
        with self._conn() as c:
            c.execute("DELETE FROM positions WHERE id=?", (pid,))

    def load_open_positions(self) -> List[Position]:
        out = []
        with self._conn() as c:
            rows = c.execute("SELECT id,symbol,market,side,strategy,qty,entry,stop,target,current,opened_at,charges,score_reason,trailing_stop,moved_to_be FROM positions").fetchall()
        for r in rows:
            p = Position(
                id=r[0], symbol=r[1], market=r[2], side=Side(r[3]), strategy=r[4], qty=r[5],
                entry=r[6], stop=r[7], target=r[8], current=r[9],
                opened_at=datetime.fromisoformat(r[10]), charges=r[11] or 0.0, score_reason=r[12] or "",
                trailing_stop=r[13], moved_to_be=bool(r[14]),
            )
            out.append(p)
        return out

    def report_summary(self):
        with self._conn() as c:
            rows = c.execute("SELECT market,strategy,COUNT(*),SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END),SUM(pnl) FROM trades GROUP BY market,strategy").fetchall()
        return rows
