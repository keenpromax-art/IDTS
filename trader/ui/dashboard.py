from __future__ import annotations

import threading
import time
import msvcrt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box

from trader.ui.theme import build_theme
from trader.ui.format import (
    fmt_money, fmt_price, fmt_qty, fmt_pnl, pnl_style, fmt_pct, fmt_pct_always, fmt_score, score_style, score_bar,
    status_style, status_glyph, sparkline, gauge, truncate, exit_chip, level_tag, strip_level,
    fmt_pf, QUIET_BOX, QUIET_TABLE_CHROME, fmt_winrate, mkt_code,
    pct_arrow, TIGHT_TABLE_CHROME, fmt_age, fmt_r, fmt_compact, fmt_pnl_compact,
    braille_chart,
)
from trader.core.market_clock import MarketClock

# the single animation in the app: a scan in progress shows a fast spinner.
# Perceived speed lives here - a quick spinner makes the wait feel shorter.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# short labels for every built-in setup/rule, so a column never shows "…"
# while a meaningful three-to-ten character name exists.
SHORT_LABELS = {
    "ema_cross": "EMA-X", "ema_20_50": "20/50", "ema_50_200": "50/200",
    "ema_8_21": "8/21", "ema_stack": "Stack", "macd_momentum": "MACD-V",
    "macd_zero": "MACD-0", "macd_divergence": "MACD-D", "stochastic": "STOCH",
    "cci": "CCI", "roc": "ROC", "mfi": "MFI", "williams_r": "W%R",
    "rsi_reversal": "RSI-Rev", "donchian": "DONCH", "keltner": "KELT",
    "bb_expansion": "BB-Exp", "volume_breakout": "Vol-BO",
    "range_extreme": "Range-X", "psar": "PSAR", "adx_di": "ADX-DI",
    "ichimoku": "ICHI", "triple_screen": "3-Scr", "vwap": "VWAP",
    "vwap_band": "VWAP-B", "orb": "ORB", "supertrend": "ST",
    "bb_squeeze": "BB-Sqz", "pivot": "Pivot",
    "rule_trend_stack": "TrendPB", "rule_trend_stack_short": "TrendPB-S",
    "rule_golden_cross": "Golden+", "rule_macd_ema": "MACD-EMA",
    "rule_macd_ema_short": "MACD-EMA-S", "rule_breakout_volume": "BO-Vol",
    "rule_breakout_volume_short": "BD-Vol", "rule_squeeze_release": "Sqz-Rel",
    "rule_vwap_reversion": "VWAP-Rev", "rule_vwap_reversion_short": "VWAP-Rev-S",
    "rule_supertrend_volume": "ST-Vol", "rule_rsi_mfi": "RSI-MFI",
    "rule_rsi_mfi_short": "RSI-MFI-S", "rule_ema20_50_cross": "20/50-ADX",
    "rule_keltner_trend": "Kelt-Stack", "rule_obv_momentum": "OBV-Mom",
}

VIEWS = ("monitor", "positions", "analytics", "diagnostics", "report",
         "log", "help", "detail")


def _block_gauge(fraction, width=8, full="█", empty="░"):
    """A tiny text gauge; colour is applied by the caller."""
    try:
        f = max(0.0, min(1.0, float(fraction)))
    except (TypeError, ValueError):
        f = 0.0
    filled = int(round(f * width))
    return full * filled + empty * (width - filled)


class Dashboard:
    def __init__(self, engine):
        self.engine = engine
        theme_name = engine.cfg.get("theme", "default")
        self.console = Console(theme=build_theme(theme_name))
        density = str((engine.cfg.get("ui") or {}).get("density", "auto")).lower()
        self._density = density if density in ("auto", "compact", "normal", "wide") else "auto"
        self.market_filter = None  # None = all open markets
        self.markets = list(engine.markets_cfg.keys())
        self._stop = False
        self._prev_signals = {}
        self._changed = set()
        self._page = 0
        # interactivity state
        self._view = None            # None/monitor | positions | analytics | diagnostics | log | help | detail
        self._sort_mode = "score"    # score | chg | signal | symbol
        self._sort_modes = ("score", "chg", "signal", "symbol")
        self._signal_filter = None   # None | "LONG" | "SHORT"
        self._text_filter = ""
        self._selected = None        # (symbol, market) of the selected row
        self._input_mode = False     # command bar active
        self._input_buf = ""
        self._pending_close = None   # index into portfolio.positions
        self._pending_close_at = 0.0
        self._spin = 0               # spinner frame for scanning states
        self._log_level = None       # None=all | "WARN" | "ERROR"
        self._log_wrap = False
        self._done_page = 0
        self._excursion = {}         # position id -> [mfe_r, mae_r]
        self._initial_risk = {}      # position id -> risk per unit at first sight
        self._label_cache = {}
        self._equity_seed = self._load_equity_seed()
        self._session_peak = 0.0
        self._render_cache = None    # memoisation window for one render pass

    def _rc(self, key, fn):
        """Memoise *fn* for the duration of one render pass."""
        cache = self._render_cache
        if cache is None:
            return fn()
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    # ------------------------------------------------------------ density
    def _wide_threshold(self) -> int:
        if self._density == "compact":
            return 200
        if self._density == "wide":
            return 140
        return 159

    def _wide(self) -> bool:
        return (self.console.width or 120) >= self._wide_threshold()

    def _layout_name(self) -> str:
        return "wide" if self._wide() else "narrow"

    def _header_h(self) -> int:
        return 4

    def _load_equity_seed(self):
        try:
            rows = self.engine.storage.load_equity_history(240)
        except Exception:
            return []
        today = datetime.now().date()
        return [(ts, eq) for ts, eq, _pnl in rows if ts.date() == today]

    def _equity_series(self, limit=60):
        """Equity points for the sparkline, hi/lo, drawdown and analytics."""
        e = self.engine
        points = [eq for _ts, eq in self._equity_seed]
        points.extend(float(v) for v in e.equity_history if v is not None)
        current = e.portfolio.total_equity_inr()
        if not points or abs(points[-1] - current) > 1e-9:
            points.append(current)
        return points[-limit:]

    def _spinner(self) -> str:
        self._spin = (self._spin + 1) % len(SPINNER_FRAMES)
        return SPINNER_FRAMES[self._spin]

    def _header(self) -> Panel:
        e = self.engine
        mode = e.mode.upper()
        badge = Text(" LIVE ", style="badge.live") if mode == "LIVE" else Text(" PAPER ", style="badge.paper")
        total_eq = e.portfolio.total_equity_inr()
        day_pnl = e.portfolio.day_pnl_inr()
        day_pct = (day_pnl / (total_eq - day_pnl) * 100) if (total_eq - day_pnl) else 0.0
        st = e.portfolio.stats()
        trades = st["trades"]
        wr_txt = fmt_winrate(st["win_rate"], trades)
        pf_txt = fmt_pf(st["profit_factor"] if trades else None)
        now_ist = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
        series = self._equity_series()
        spark = sparkline(series, 18)
        day_lo = min(series) if series else total_eq
        day_hi = max(series) if series else total_eq
        peak = max(series) if series else total_eq
        dd = ((peak - total_eq) / peak * 100.0) if peak else 0.0
        risk_rows = self._risk_rows()
        exposure = sum(v.get("exposure", 0.0) for v in risk_rows.values())
        risk = sum(v.get("risk", 0.0) for v in risk_rows.values())
        exp_pct = (exposure / total_eq * 100.0) if total_eq else 0.0
        risk_pct = (risk / total_eq * 100.0) if total_eq else 0.0
        w = self.console.width or 120
        line1 = Text()
        line1.append_text(badge)
        line1.append("  Equity ", style="muted")
        line1.append(fmt_money(total_eq, "INR"), style="text")
        line1.append(" ", style="muted")
        line1.append(spark, style="status.open" if day_pnl >= 0 else "status.warn")
        line1.append("  Day ", style="muted")
        line1.append(fmt_pnl(day_pnl, "INR"), style=pnl_style(day_pnl))
        line1.append(f" ({fmt_pct_always(day_pct)})", style=pnl_style(day_pnl))
        line1.append("   ", style="muted")
        line1.append(now_ist.strftime("%H:%M:%S"), style="text")
        line1.append(" IST", style="muted")
        if w >= 130:
            line1.append("   Win ", style="muted")
            line1.append(f"{wr_txt} · {trades} trades", style="text")
            line1.append(" · PF ", style="muted")
            line1.append(pf_txt, style="text")

        line2 = Text()
        line2.append("Realised ", style="muted")
        line2.append(fmt_pnl(e.portfolio.day_realized_inr(), "INR"),
                     style=pnl_style(e.portfolio.day_realized_inr()))
        line2.append(" · Unrealised ", style="muted")
        line2.append(fmt_pnl(e.portfolio.day_unrealized_inr(), "INR"),
                     style=pnl_style(e.portfolio.day_unrealized_inr()))
        line2.append(" · Exp ", style="muted")
        line2.append(f"{exp_pct:.1f}%", style="text")
        line2.append(" · Risk ", style="muted")
        line2.append(f"{risk_pct:.1f}%", style="text")
        if w >= 150:
            line2.append(" · Hi ", style="muted")
            line2.append(fmt_money(day_hi, "INR"), style="text")
            line2.append(" Lo ", style="muted")
            line2.append(fmt_money(day_lo, "INR"), style="text")
            line2.append(" · DD ", style="muted")
            line2.append(f"{dd:.2f}%", style="pnl.neg" if dd > 0 else "muted")
            line2.append(" · ", style="muted")
            line2.append(fmt_r(st["expectancy_r"]) if trades else "—", style="text")
            line2.append(" avg W/L ", style="muted")
            line2.append(fmt_pnl(st["avg_win"], "INR") if st["wins"] else "—", style="pnl.pos")
            line2.append("/", style="muted")
            line2.append(fmt_pnl(st["avg_loss"], "INR") if st["losses"] else "—", style="pnl.neg")
        if w >= 190:
            line2.append(" · Kill ", style="muted")
            line2.append("ARMED" if e.risk.kill_switch else "clear",
                         style="status.warn" if e.risk.kill_switch else "muted")
            q_age = e.ticker.age_of_newest() if hasattr(e, "ticker") else None
            line2.append(" · feed ", style="muted")
            line2.append(fmt_age(q_age) if e.live_quotes else "off",
                         style="text" if e.live_quotes else "dim")
            line2.append(" · scan ", style="muted")
            line2.append(f"{e.last_scan_elapsed:.1f}s", style="text")
        line1.truncate(max(20, w - 4))
        line2.truncate(max(20, w - 4))
        if mode == "LIVE":
            return Panel(Group(line1, line2), box=box.ROUNDED, border_style="#DA3633")
        return Panel(Group(line1, line2), box=box.ROUNDED, border_style="border")

    def _inr(self, value: float, market: str) -> float:
        """Every price is shown in INR at the standard cross rate.

        Execution still happens at the exchange's native price - this only
        converts what the dashboard displays, so a US entry, an Australian
        stop and an Indian P&L are all read in the same currency.
        """
        return self.engine.portfolio.to_inr(value, market)

    CARD_MIN_WIDTH = 24
    CARD_HEIGHT = 8   # 6 content lines + borders

    def _market_lists(self):
        """(open, closed) markets. Closed ones collapse to chips."""
        return self._rc("market_lists", self._market_lists_uncached)

    def _market_lists_uncached(self):
        open_m, closed_m = [], []
        for m in self.markets:
            status = self.engine.clock.status(m)["status"]
            (closed_m if status == "CLOSED" else open_m).append(m)
        return open_m, closed_m

    def _market_stats(self, m):
        return self._rc(("market_stats", m), lambda: self._market_stats_uncached(m))

    def _market_stats_uncached(self, m):
        e = self.engine
        rows = e.last_scan.get(m, [])
        total = len(e.watchlists.get(m, []))
        scanned = sum(1 for r in rows if r.get("price") is not None)
        signals = sum(1 for r in rows if r.get("signal") in ("LONG", "SHORT"))
        blocked = sum(1 for r in rows if r.get("gate"))
        changes = []
        for r in rows:
            ch = r.get("live_change")
            if ch is None:
                ch = r.get("change")
            if ch is not None:
                changes.append(ch)
        breadth = (sum(1 for c in changes if c >= 0) / len(changes) * 100) if changes else None
        today = datetime.now(timezone.utc).date()
        closed_today = [t for t in e.portfolio.trades
                        if t.market == m and t.closed_at.date() == today]
        wins = sum(1 for t in closed_today if t.pnl > 0)
        losses = len(closed_today) - wins
        day_pnl = e.portfolio.day_pnl_local(m)
        eq = e.portfolio.equity_local(m)
        day_pct = (day_pnl / eq * 100.0) if eq else 0.0
        max_loss = float(e.risk.risk.get("max_daily_loss_pct", 1.5) or 1.5)
        loss_used = (max(0.0, -day_pnl) / (eq * max_loss / 100.0) * 100.0) if eq and max_loss else 0.0
        ot = sum(1 for p in e.portfolio.positions if p.market == m)
        return {
            "scanned": scanned, "total": total, "signals": signals,
            "blocked": blocked, "breadth": breadth, "positions": ot,
            "wins": wins, "losses": losses, "day_pnl": day_pnl,
            "day_pct": day_pct, "loss_used": loss_used, "max_loss": max_loss,
            "rows": rows, "trades": len(closed_today),
        }

    def _card_inner_width(self, cols=None):
        cols = cols or self._market_cols()
        return max(10, (self.console.width or 120) // max(1, cols) - 4)

    def _market_card(self, m, inner=None):
        e = self.engine
        clock = e.clock
        st = clock.status(m)
        status = st["status"]
        sstyle = status_style(status)
        cd = clock.fmt_countdown(st["countdown_seconds"])
        open_dt = st["open_dt"]
        close_dt = st["close_dt"]
        local_dt = st["local_time"]
        if local_dt < open_dt:
            prog = 0.0
        elif local_dt >= close_dt:
            prog = 1.0
        elif status == "CLOSED":
            prog = 0.0
        else:
            total = (close_dt - open_dt).total_seconds()
            prog = (local_dt - open_dt).total_seconds() / total if total > 0 else 0.0
        filled = int(round(prog * 8))
        bar = "▰" * filled + "▱" * (8 - filled)
        stats = self._market_stats(m)
        prov = e.providers[m]
        src = getattr(prov, "primary", prov).name if hasattr(prov, "primary") else prov.name
        src_short = {"yfinance": "Y!", "yahoo": "Y!"}.get(str(src).lower(), str(src)[:5])
        rows = stats["rows"]
        age = None
        if rows:
            upd = rows[0].get("updated")
            if upd is not None:
                try:
                    age = (datetime.now(timezone.utc) - upd).total_seconds()
                except Exception:
                    age = None
        age_txt = fmt_age(age)
        pnl = stats["day_pnl"]
        inner = inner or self._card_inner_width()
        head = f"{mkt_code(self.engine.markets_cfg, m)} {status_glyph(status)} {status}"
        when = f"{st['local_time'].strftime('%H:%M')}"
        if status == "CLOSED":
            second = f"opens {st['next_open'].strftime('%H:%M')}" if st.get("next_open") else f"opens {cd}"
        elif status == "PRE-CLOSE":
            second = f"closes {cd}"
        else:
            second = f"closes {cd} · {src_short} {age_txt}" if inner >= 28 else f"close {cd} · {src_short} {age_txt}"
        if inner >= 28:
            head += f"  {when}"
        bar_line = f"{bar} "
        breadth = "—" if stats["breadth"] is None else f"{stats['breadth']:.0f}%"
        if inner >= 26:
            scan_txt = f"scan {stats['scanned']}/{stats['total']} · {stats['signals']} sig · {stats['positions']} pos"
        else:
            scan_txt = f"{stats['scanned']}/{stats['total']} {stats['signals']}s {stats['positions']}p"
        day_txt = f"Day {fmt_pnl_compact(self._inr(pnl, m), 'INR')} ({fmt_pct_always(stats['day_pct'])})"
        wl = f"{stats['wins']}W/{stats['losses']}L"
        loss_txt = f"loss {stats['loss_used']:.0f}/{stats['max_loss']:.1f}%"
        lines = Text()
        lines.append(truncate(head, inner), style="text")
        lines.append("\n")
        lines.append(truncate(second, inner), style="status.preclose" if status == "PRE-CLOSE" else "muted")
        lines.append("\n")
        lines.append(bar_line, style=sstyle)
        lines.append(f"↕ {breadth}", style="muted")
        lines.append("\n")
        lines.append(truncate(scan_txt, inner), style="muted")
        lines.append("\n")
        lines.append(truncate(day_txt, inner), style=pnl_style(pnl))
        lines.append("\n")
        lines.append(truncate(f"{wl} · {loss_txt}", inner), style="muted")
        border = "status.open" if status == "OPEN" else ("status.preclose" if status in ("PRE-CLOSE", "LUNCH") else "border")
        return Panel(lines, border_style=border, box=box.ROUNDED, padding=(0, 1))

    def _market_cols(self):
        """How many open-market cards fit on one row."""
        w = self.console.width or 120
        return self._rc(("market_cols", w), self._market_cols_uncached)

    def _market_cols_uncached(self):
        n = len(self._market_lists()[0])
        if n <= 1:
            return 1
        width = self.console.width or 120
        best = None
        for cols in range(n, 0, -1):
            if width // cols < self.CARD_MIN_WIDTH:
                continue
            rows = (n + cols - 1) // cols
            gaps = rows * cols - n
            key = (gaps, -cols)
            if best is None or key < best[0]:
                best = (key, cols)
        return best[1] if best else 1

    def _market_grid(self):
        open_m, _closed = self._market_lists()
        cols = self._market_cols()
        inner = self._card_inner_width(cols)
        grid = Table.grid(expand=True)
        for _ in range(cols):
            grid.add_column(ratio=1)
        for i in range(0, len(open_m), cols):
            chunk = open_m[i:i + cols]
            cells = [self._market_card(m, inner) for m in chunk]
            while len(cells) < cols:
                cells.append("")
            grid.add_row(*cells)
        return grid

    def _closed_chips(self, avail=None):
        e = self.engine
        _open, closed = self._market_lists()
        if not closed:
            return []
        avail = avail or max(20, (self.console.width or 120) - 4)
        parts = []
        for m in closed:
            code = mkt_code(e.markets_cfg, m)
            st = e.clock.status(m)
            txt = f"{code} ● CLOSED"
            if st.get("next_open") is not None:
                txt += f" opens {st['next_open'].strftime('%H:%M')}"
            ot = sum(1 for p in e.portfolio.positions if p.market == m)
            if ot:
                txt += f" · {ot}pos"
            parts.append(txt)
        lines, cur = [], ""
        for part in parts:
            candidate = (cur + "  │  " + part) if cur else part
            if len(candidate) > avail and cur:
                lines.append(cur)
                cur = part
            else:
                cur = candidate
        if cur:
            lines.append(cur)
        return [Text(x, style="muted") for x in lines]

    def _compact_market_line(self, m):
        e = self.engine
        st = e.clock.status(m)
        status = st["status"]
        stats = self._market_stats(m)
        code = mkt_code(e.markets_cfg, m)
        return Text.assemble(
            (code, "text"), (" ", "muted"),
            (status_glyph(status), status_style(status)),
            (f"{status[:4]}", status_style(status)),
            (f" {stats['scanned']}/{stats['total']}", "muted"),
            (f" {stats['signals']}sig", "muted"),
            (f" {stats['positions']}p", "muted"),
            (f" {fmt_pnl_compact(self._inr(stats['day_pnl'], m), 'INR')}",
             pnl_style(stats["day_pnl"])),
        )

    def _compact_market_row(self, m):
        """One information-dense line per market, used when cards do not fit."""
        e = self.engine
        st = e.clock.status(m)
        status = st["status"]
        stats = self._market_stats(m)
        code = mkt_code(e.markets_cfg, m)
        breadth = "—" if stats["breadth"] is None else f"{stats['breadth']:.0f}%"
        close = e.clock.fmt_countdown(st["countdown_seconds"])
        txt = Text.assemble(
            (f"{code:<3}", "text"), (" ", "muted"),
            (status_glyph(status), status_style(status)),
            (f"{status[:5]:<6}", status_style(status)),
            (f" {stats['scanned']:>3}/{stats['total']:<4}", "muted"),
            (f" {stats['signals']}sig", "muted"),
            (f" {stats['positions']}pos", "muted"),
            (f"  ↕{breadth}", "muted"),
            (f"  {fmt_pnl_compact(self._inr(stats['day_pnl'], m), 'INR')}",
             pnl_style(stats["day_pnl"])),
            (f" ({fmt_pct_always(stats['day_pct'])})", pnl_style(stats["day_pnl"])),
            (f"  {stats['wins']}W/{stats['losses']}L", "muted"),
            (f"  loss {stats['loss_used']:.0f}/{stats['max_loss']:.1f}%", "muted"),
            (f"  {close}", "dim"),
        )
        return txt

    def _compact_market_lines(self, avail, max_lines):
        open_m, _closed = self._market_lists()
        if max_lines >= len(open_m):
            rows = []
            for m in open_m[:max_lines]:
                row = self._compact_market_row(m)
                row.truncate(avail)
                rows.append(row)
            return rows
        # not enough lines for one per market: pack as many codes as fit
        lines, cur, cur_len = [], Text(), 0
        for m in open_m:
            seg = self._compact_market_line(m)
            seg_len = len(seg.plain)
            need = seg_len if cur_len == 0 else seg_len + 3
            if cur_len and cur_len + need > avail:
                lines.append(cur)
                cur, cur_len = Text(), 0
                if len(lines) >= max_lines:
                    break
            if cur_len:
                cur.append("   ", style="dim")
            cur.append_text(seg)
            cur_len += need
        if cur_len and len(lines) < max_lines:
            lines.append(cur)
        return lines[:max_lines]

    def _markets_height(self):
        """Rows the markets panel needs at its natural (full-card) size."""
        return self._rc(("markets_height", self.console.width),
                        self._markets_height_uncached)

    def _markets_height_uncached(self):
        open_m, closed_m = self._market_lists()
        cols = self._market_cols()
        rows = (len(open_m) + cols - 1) // cols
        chips = len(self._closed_chips()) if closed_m else 0
        if not open_m:
            return 2 + max(1, chips)
        return 2 + rows * self.CARD_HEIGHT + chips

    def _markets_compact_height(self):
        """Rows for the one-line-per-market fallback."""
        return self._rc(("markets_compact_height", self.console.width),
                        self._markets_compact_height_uncached)

    def _markets_compact_height_uncached(self):
        open_m, closed_m = self._market_lists()
        chips = len(self._closed_chips()) if closed_m else 0
        return 2 + max(1, len(open_m)) + chips

    def _markets_panel(self, height: int = 0) -> Panel:
        e = self.engine
        open_m, closed_m = self._market_lists()
        if not open_m and not closed_m:
            return Panel(Text("No markets configured", style="muted"), title="Markets",
                         box=box.ROUNDED, border_style="border")
        inner = max(20, (self.console.width or 120) - 4)
        chips = self._closed_chips(inner)
        full_h = self._markets_height()
        if height and height >= full_h:
            body = Group(self._market_grid(), *chips) if chips else self._market_grid()
            return Panel(body, title="Markets", box=box.ROUNDED, border_style="border")
        room = max(1, (height - 2) if height else 6)
        lines = self._compact_market_lines(inner, max(1, room - len(chips)))
        lines.extend(chips[:max(0, room - len(lines))])
        if not lines:
            lines = [Text("  ".join(mkt_code(e.markets_cfg, m) for m in self.markets), style="accent")]
        return Panel(Group(*lines[:room]), title="Markets", box=box.ROUNDED,
                     border_style="border")

    def _open_markets(self):
        e = self.engine
        return [m for m in self.markets
                if e.clock.status(m)["status"] in ("OPEN", "PRE-CLOSE", "LUNCH")]

    def _scanner_rows(self):
        e = self.engine
        # determine markets to show
        if self.market_filter is None:
            markets = self._open_markets()
            if not markets:
                markets = self.markets
        else:
            markets = [self.market_filter]
        rows = []
        seen = set()
        for m in markets:
            for r in e.last_scan.get(m, []):
                rows.append(r)
                seen.add(r.get("symbol"))
        # First scan in progress: the quote thread already has prices for a
        # growing set of symbols, so show those instead of a blank table.
        if e.live_quotes and hasattr(e, "ticker"):
            try:
                prices = e.ticker.snapshot()
            except Exception:
                prices = {}
            if prices:
                for m in markets:
                    for sym in e.watchlists.get(m, []):
                        if sym in seen or sym not in prices:
                            continue
                        age = e.ticker.age(sym)
                        rows.append({
                            "symbol": sym, "market": m, "price": None,
                            "live_price": prices[sym], "change": None,
                            "regime": "scanning", "best": None, "best_label": None,
                            "best_reason": "", "best_reason_short": "",
                            "gate": "", "score": None, "signal": "NONE",
                            "setups": 0, "firings": [], "metrics": {},
                            "plan": None, "stats": None,
                            "updated": datetime.now(timezone.utc),
                            "stale": age is not None and age > 120,
                            "prefill": True,
                        })
        # filters: text (symbol or strategy substring) and signal side
        if self._text_filter:
            flt = self._text_filter.lower()
            rows = [r for r in rows
                    if flt in str(r.get("symbol", "")).lower()
                    or flt in str(r.get("best", "")).lower()
                    or flt in str(r.get("best_label", "")).lower()]
        if self._signal_filter:
            rows = [r for r in rows if r.get("signal") == self._signal_filter]

        # sort per the selected mode
        def row_score(r):
            sc = r.get("score")
            return sc if sc is not None else -1.0

        def row_chg(r):
            c = r.get("live_change")
            if c is None:
                c = r.get("change")
            return c if c is not None else float("-inf")

        if self._sort_mode == "score":
            def key(r):
                sig = r.get("signal", "NONE")
                signalling = 0 if sig in ("LONG", "SHORT") else 1
                return (signalling, -row_score(r))
        elif self._sort_mode == "chg":
            def key(r):
                return -row_chg(r)
        elif self._sort_mode == "signal":
            def key(r):
                sig = r.get("signal", "NONE")
                rank = {"LONG": 0, "SHORT": 1}.get(sig, 2)
                return (rank, -row_score(r))
        else:  # symbol
            def key(r):
                return str(r.get("symbol", "")).lower()
        rows.sort(key=key)
        # detect signal changes
        new_changed = set()
        for r in rows:
            sym = r.get("symbol")
            sig = r.get("signal", "NONE")
            if sym in self._prev_signals and self._prev_signals[sym] != sig and sig in ("LONG", "SHORT"):
                new_changed.add(sym)
            self._prev_signals[sym] = sig
        self._changed = new_changed
        return rows, markets

    def _find_selected(self, rows):
        """Index of the selected row in the current (filtered, sorted) rows."""
        if not rows:
            return None, None
        if self._selected is None:
            return 0, 0
        for i, r in enumerate(rows):
            if (r.get("symbol"), r.get("market")) == self._selected:
                return i, i
        self._selected = (rows[0].get("symbol"), rows[0].get("market"))
        return 0, 0

    def _get_selected_row(self):
        rows, _ = self._scanner_rows()
        idx, _ = self._find_selected(rows)
        if idx is None or not rows:
            return None, None
        self._selected = (rows[idx].get("symbol"), rows[idx].get("market"))
        return rows[idx], self._selected

    def _fit_columns(self, ordered, avail):
        """Pick and size columns so the table fits in `avail` chars.

        Each column is (name, min_width, max_width, justify, priority). Columns
        are added highest-priority-first while they fit; if the set is still
        too wide, the widest flexible columns are shrunk towards their min
        width before anything is dropped. Priority order (not insertion order)
        decides what survives.
        """
        def cost(cols):
            return sum(c[2] for c in cols) + 2 * len(cols) + 1

        # work on mutable copies so the widths can be shrunk
        pool = [list(c) for c in ordered]
        chosen = []
        for c in sorted(pool, key=lambda c: -c[4]):
            if cost(chosen + [c]) <= avail:
                chosen.append(c)
        # shrink the widest columns until the table fits
        while cost(chosen) > avail:
            shrinkable = [c for c in chosen if c[2] > c[1]]
            if not shrinkable:
                break
            victim = max(shrinkable, key=lambda c: c[2] - c[1])
            victim[2] = max(victim[1], victim[2] - 1)
        # display order = original spec order
        order = {c[0]: i for i, c in enumerate(pool)}
        chosen.sort(key=lambda c: order[c[0]])
        self._col_specs = chosen
        return chosen

    def _panel_width(self, which="active"):
        """Approximate inner width available to a panel at the current console size."""
        w = self.console.width or 120
        # a Rich panel uses 2 chars for its border and 1 for padding each side
        if which == "active":
            if self._wide():
                return int(w * 0.44) - 4
            return w - 4
        if which == "completed":
            if self._wide():
                return int(w * 0.32) - 4
            return int(w * 0.6) - 4
        if which == "log":
            if self._wide():
                return int(w * 0.24) - 4
            return int(w * 0.4) - 4
        return w - 4

    def _page_size(self):
        """Rows available for data in the scanner panel at the current height."""
        panel_h = max(4, self._space()[1])
        # panel borders(2) + table chrome (QUIET_TABLE_CHROME, or TIGHT_TABLE_
        # CHROME below 8 rows) + optional empty-state line(2)
        chrome = QUIET_TABLE_CHROME if panel_h >= 2 + QUIET_TABLE_CHROME + 1 else TIGHT_TABLE_CHROME
        extra = 2 if not self._scanner_rows()[0] else 0
        return max(1, panel_h - 2 - chrome - extra)

    def _plan_price(self, value, market):
        if value is None:
            return "—"
        return fmt_money(self._inr(value, market), "INR")

    def _plan_triplet(self, plan, market):
        vals = [plan.get("entry"), plan.get("stop"), plan.get("target")]
        if any(v is None for v in vals):
            return "—"
        inr = [self._inr(v, market) for v in vals]
        if any(abs(v) >= 1e5 for v in inr):
            parts = [fmt_compact(v, "INR") for v in inr]
        else:
            parts = [fmt_money(v, "INR") for v in inr]
        return "/".join(parts)

    def _scanner_cell(self, value, width, justify):
        if isinstance(value, Text):
            cell = value.copy()
        else:
            cell = Text(str(value))
        if len(cell.plain) > width:
            cell.truncate(width, overflow="ellipsis")
        pad = max(0, width - len(cell.plain))
        if justify == "right":
            left = pad
        elif justify == "center":
            left = pad // 2
        else:
            left = 0
        right = pad - left
        out = Text()
        if left:
            out.append(" " * left)
        out.append_text(cell)
        if right:
            out.append(" " * right)
        return out

    def _scanner_line(self, cols, values, style=None):
        line = Text(style=style or "")
        line.append(" ")
        for i, col in enumerate(cols):
            if i:
                line.append("  ")
            line.append_text(self._scanner_cell(
                values[i], col[2], col[3]))
        line.append("  ")
        return line

    def _scanner_panel(self) -> Panel:
        e = self.engine
        rows, markets = self._scanner_rows()
        mf = "ALL OPEN" if self.market_filter is None else self.market_filter
        page_size = self._page_size()
        total = len(rows)
        pages = max(1, (total + page_size - 1) // page_size)
        self._page = min(self._page, pages - 1)
        start = self._page * page_size
        page_rows = rows[start:start + page_size]
        spec = [
            ("Mkt", 3, 3, "left", 9),
            ("Symbol", 8, 16, "left", 10),
            ("Last", 8, 12, "right", 8),
            ("%Chg", 7, 8, "right", 7),
            ("RVOL", 4, 5, "right", 4),
            ("ATR%", 5, 6, "right", 4),
            ("ADX", 3, 4, "right", 3),
            ("RSI", 3, 4, "right", 3),
            ("VWAP%", 6, 7, "right", 4),
            ("Regime", 7, 9, "left", 4),
            ("Strategy", 7, 14, "left", 6),
            ("Plan", 11, 24, "left", 3),
            ("R:R", 4, 5, "right", 4),
            ("Setups", 4, 7, "right", 3),
            ("Score", 8, 11, "right", 7),
            ("Signal", 7, 8, "center", 9),
            ("Age", 3, 4, "right", 2),
            ("Spark", 6, 10, "left", 2),
            ("Why", 10, 34, "left", 5),
        ]
        avail = self._panel_width("scanner") - 1
        cols = self._fit_columns(spec, avail)
        spare = avail - (sum(c[2] for c in cols) + 2 * len(cols) + 1)
        flex = [c for c in cols if c[0] in ("Why", "Symbol", "Strategy",
                                            "Plan", "Regime")]
        for i in range(max(0, spare)):
            if flex:
                flex[i % len(flex)][2] += 1
        widths = {c[0]: c[2] for c in cols}
        _, scanner_h, _, _ = self._space()
        tight = scanner_h < 2 + QUIET_TABLE_CHROME + 1
        data_rows = []
        empty_msg = None
        if not page_rows:
            if self._text_filter or self._signal_filter:
                conds = []
                if self._text_filter:
                    conds.append(f"text '{self._text_filter}'")
                if self._signal_filter:
                    conds.append(self._signal_filter.lower())
                empty_msg = f"No symbols match filter ({', '.join(conds)}) - :filter off to reset"
            else:
                open_mk = [m for m in self.markets if e.clock.status(m)["status"] in ("OPEN", "PRE-CLOSE")]
                if e.paused:
                    empty_msg = "Scanning paused — press p to resume"
                elif e.scan_progress:
                    empty_msg = f"{self._spinner()} Scanning {e.scan_progress}…"
                elif not open_mk:
                    m = self.markets[0]
                    st = e.clock.status(m)
                    cd = e.clock.fmt_countdown(st["countdown_seconds"])
                    empty_msg = f"All markets closed — next open in {cd}."
                if not open_mk:
                    m = self.markets[0]
                    st = e.clock.status(m)
                    cd = e.clock.fmt_countdown(st["countdown_seconds"])
                    empty_msg = f"All markets closed — next open in {cd}."
                elif not e.last_scan:
                    empty_msg = f"{self._spinner()} First scan running…"
                else:
                    closed = [m for m in self.markets if m not in open_mk]
                    if closed:
                        m = closed[0]
                        st = e.clock.status(m)
                        cd = e.clock.fmt_countdown(st["countdown_seconds"])
                        empty_msg = f"{m} closed, opens in {cd}."
                    else:
                        # Show scanning progress with data age info
                        age_info = ""
                        if e.last_scan:
                            ages = []
                            for market, rows in e.last_scan.items():
                                for r in rows:
                                    if r.get("price") is not None:
                                        age = e.ticker.age(r["symbol"]) if hasattr(e, "ticker") else None
                                        if age is not None:
                                            ages.append(age)
                            if ages:
                                avg_age = sum(ages) / len(ages)
                                age_info = f" · avg age {fmt_age(avg_age)}"
                        empty_msg = f"{self._spinner()} Scanning…{age_info}"
        for r in page_rows:
            sym = r.get("symbol", "")
            mkt = r.get("market", "")
            price = r.get("live_price") or r.get("price")
            if price is None and hasattr(e, "ticker"):
                price = e.ticker.price(sym)
            if price is not None:
                last_cell = Text(fmt_price(self._inr(price, mkt), "INR"), style="text")
                if r.get("stale"):
                    last_cell.append(" *", style="stale")
            else:
                last_cell = Text("—", style="muted")
            chg = r.get("live_change")
            if chg is None:
                chg = r.get("change")
            chg_txt = pct_arrow(chg)
            metrics = r.get("metrics") or {}
            rvol = metrics.get("rvol")
            atr_pct = metrics.get("atr_pct")
            adx = metrics.get("adx")
            rsi = metrics.get("rsi")
            vwap_dist = metrics.get("vwap_dist")
            regime = r.get("regime", "—")
            regime_style = "regime.trending" if "trend" in str(regime).lower() else ("regime.ranging" if "rang" in str(regime).lower() else "muted")
            plan = r.get("plan")
            plan_txt = self._plan_triplet(plan, mkt) if plan else "—"
            score = r.get("score")
            score_txt = fmt_score(score)
            sig = r.get("signal", "NONE")
            if sig == "LONG":
                sig_txt = Text("▲ LONG", style="pnl.pos")
            elif sig == "SHORT":
                sig_txt = Text("▼ SHORT", style="pnl.neg")
            else:
                sig_txt = Text("NONE", style="dim")
            q_age = e.ticker.age(sym) if hasattr(e, "ticker") else None
            upd = r.get("updated")
            if q_age is not None:
                age_txt = fmt_age(q_age)
            elif upd is not None:
                try:
                    age_txt = fmt_age((datetime.now(timezone.utc) - upd).total_seconds())
                except Exception:
                    age_txt = "—"
            else:
                age_txt = "—"
            spark_vals = list(metrics.get("spark") or [])
            if price is not None:
                spark_vals.append(price)
            spark_txt = sparkline(spark_vals, 8) if spark_vals else "—"
            why = ""
            why_style = "muted"
            if r.get("prefill"):
                why = "quote thread · waiting for scan"
            elif sig in ("LONG", "SHORT"):
                why = r.get("best_reason_short") or r.get("best_reason") or ""
            elif r.get("gate"):
                why = r.get("gate")
                why_style = "stale"
            else:
                why = r.get("best_reason_short") or ""
            is_sel = (sym, mkt) == self._selected
            if is_sel:
                sym = "›" + sym
                row_style = "sel"
            else:
                row_style = "bold" if sym in self._changed else None
            if r.get("prefill"):
                row_style = "dim"
            elif sig == "NONE" and not is_sel:
                row_style = (row_style + " dim") if row_style else "dim"
            score_w = widths.get("Score", 9)
            if score is None:
                score_cell = Text("—", style="muted")
            elif score_w >= 9:
                score_cell = Text(score_bar(score, 4) + " " + score_txt, style=score_style(score))
            elif score_w >= 7:
                score_cell = Text(score_bar(score, 2) + " " + score_txt, style=score_style(score))
            else:
                score_cell = Text(score_txt, style=score_style(score))
            report = r.get("stats") or {}
            cells = {
                "Mkt": mkt_code(self.engine.markets_cfg, mkt),
                "Symbol": sym,
                "Last": last_cell,
                "%Chg": chg_txt,
                "RVOL": f"{rvol:.1f}x" if rvol is not None else ("—" if not r.get("prefill") else "scanning"),
                "ATR%": f"{atr_pct:.1f}%" if atr_pct is not None else ("—" if not r.get("prefill") else "wait"),
                "ADX": f"{adx:.0f}" if adx is not None else ("—" if not r.get("prefill") else "—"),
                "RSI": f"{rsi:.0f}" if rsi is not None else ("—" if not r.get("prefill") else "—"),
                "VWAP%": pct_arrow(vwap_dist) if vwap_dist is not None else (Text("—", style="muted") if not r.get("prefill") else Text("wait", style="dim")),
                "Regime": Text(regime, style=regime_style),
                "Strategy": self._short_label(r.get("best")) if r.get("best") else ("—" if not r.get("prefill") else "—"),
                "Plan": plan_txt,
                "R:R": f"{plan['rr']:.1f}" if plan and plan.get("rr") is not None else ("—" if not r.get("prefill") else "—"),
                "Setups": Text(str(r.get("setups") or 0),
                               style="accent" if (r.get("setups") or 0) else "muted"),
                "Score": score_cell,
                "Signal": sig_txt,
                "Age": age_txt,
                "Spark": Text(spark_txt, style="status.open" if chg is not None and chg >= 0 else "pnl.neg"),
                "Why": Text(truncate(why, 34), style=why_style),
            }
            data_rows.append((row_style, [cells[c[0]] for c in cols]))
        rule_w = avail
        lines = []
        if not tight:
            lines.append(Text(" " + "─" * (rule_w - 1), style="border"))
        lines.append(self._scanner_line(
            cols, [Text(c[0], style="bold") for c in cols]))
        if not tight:
            lines.append(Text(" " + "─" * (rule_w - 1), style="border"))
        for row_style, values in data_rows:
            lines.append(self._scanner_line(cols, values, row_style))
        if empty_msg:
            if not tight:
                lines.append(Text(""))
            lines.append(Text(empty_msg, style="muted"))
        if not tight:
            lines.append(Text(""))
        body = Text()
        for i, line in enumerate(lines):
            if i:
                body.append("\n")
            body.append_text(line)
        parts = [mf]
        if self._text_filter:
            parts.append(f"'{self._text_filter}'")
        if self._signal_filter:
            parts.append(self._signal_filter.lower())
        if self._sort_mode != "score":
            parts.append(f"sort:{self._sort_mode}")
        title = f"Scanner · {' · '.join(parts)}"
        subtitle = Text(f"updated {datetime.now().strftime('%H:%M:%S')} · page {self._page + 1}/{pages}",
                        style="muted")
        return Panel(body, title=title, subtitle=subtitle, box=box.ROUNDED,
                     border_style="border")

    # ------------------------------------------------------------ overlays

    def _overlay(self) -> Optional[Panel]:
        """The active modal overlay, if any (help or symbol detail)."""
        if self._view == "help":
            return self._help_panel()
        if self._view == "detail":
            return self._detail_panel()
        return None

    HV = "border"

    def _help_panel(self) -> Panel:
        t = Table.grid(expand=False, padding=(0, 2, 0, 0))
        t.add_column(justify="right")
        t.add_column(justify="left")
        rows = [
            ("q", "quit"),
            ("x", "square-off all"),
            ("p", "pause / resume scanning"),
            ("Tab", "cycle market filter (all open → each market)"),
            ("Backspace", "cycle market filter backwards"),
            ("←→", "scanner page back / forward"),
            ("↑↓", "select a scanner row"),
            ("d", "detail view of selected row (Esc closes)"),
            ("Enter", "open detail for the selected row"),
            ("f", "force a scan right now"),
            ("s", "cycle sort: score → chg → signal → symbol"),
            ("g", "cycle signals: all → longs → shorts"),
            ("v", "toggle live quotes"),
            ("c", "toggle crypto market on/off"),
            ("m", "monitor view (the default)"),
            ("i", "positions view"),
            ("a", "analytics view"),
            ("y", "diagnostics view"),
            ("r", "session report (closed trades)"),
            ("l", "full-screen log view"),
            ("n / b", "next / previous completed-trades page"),
            ("e", "cycle log filter: all → WARN → ERROR"),
            ("w", "toggle log line wrap"),
            ("1..9", "press once to arm, again to close that position"),
            (": or /", "command bar (below)"),
            ("h / ?", "this help"),
        ]
        for k, v in rows:
            t.add_row(Text(k, style="accent"), Text(v, style="text"))
        cmd = Table.grid(expand=False, padding=(0, 1, 0, 0))
        cmd.add_column()
        cmd.add_column()
        cmds = [
            (":filter <text>", "filter scanner by symbol/strategy; `:filter off` clears"),
            (":sort score|chg|signal|symbol", "change the scanner sort"),
            (":longs / :shorts / :all", "show only long / short / all signals"),
            (":close N or SYMBOL", "close one position (paper)"),
            (":flat", "square-off every position"),
            (":pause / :resume", "pause or resume scanning"),
            (":crypto on|off", "turn the crypto market on/off"),
            (":kill", "trip the daily-kill switch now"),
            (":scan", "force a scan"),
            (":report", "open the session report"),
            (":view monitor|positions|analytics|diagnostics|report|log", "switch view"),
            (":log all|warn|error", "filter the log"),
            (":log wrap", "toggle log wrapping"),
            (":density auto|compact|normal|wide", "layout density"),
            (":help", "open this help"),
        ]
        cmd.add_row(Text("Commands", style="accent"), Text("", style="text"))
        for k, v in cmds:
            cmd.add_row(Text(k, style="muted"), Text(v, style="text"))
        body = Group(t, Text(""), cmd, Text(""), Text("Esc closes overlays.", style="dim"))
        return Panel(body, title="Dashboard keys", box=box.ROUNDED, border_style=self.HV)

    def _detail_panel(self) -> Panel:
        e = self.engine
        row, key = self._get_selected_row()
        if row is None:
            body = Group(Text("No row selected.", style="muted"),
                         Text("Use ↑/↓ in the scanner, then press d.", style="dim"))
            return Panel(body, title="Detail", box=box.ROUNDED, border_style=self.HV)
        sym = row.get("symbol", "")
        mkt = row.get("market", "")
        price = row.get("live_price") or row.get("price")
        if price is None and hasattr(e, "ticker"):
            price = e.ticker.price(sym)
        chg = row.get("live_change")
        if chg is None:
            chg = row.get("change")
        metrics = row.get("metrics") or {}
        sc = row.get("score")
        info = Text()
        info.append(f"{sym}", style="text")
        info.append(f"   {mkt}", style="muted")
        info.append("   ", style="muted")
        info.append(fmt_price(self._inr(price, mkt), "INR") if price is not None else "—",
                    style="text")
        if chg is not None:
            info.append("   ", style="muted")
            info.append_text(pct_arrow(chg))
        info.append("   regime ", style="muted")
        info.append(str(row.get("regime", "—")), style="text")
        info.append("   score ", style="muted")
        info.append(fmt_score(sc), style=score_style(sc))
        info.append(" ", style="text")
        info.append(score_bar(sc, 10), style=score_style(sc))
        plan = row.get("plan") or {}
        info.append("\nplan   ", style="muted")
        if plan:
            info.append(f"entry {self._plan_price(plan.get('entry'), mkt)}  ", style="text")
            info.append(f"stop {self._plan_price(plan.get('stop'), mkt)}  ", style="pnl.neg")
            info.append(f"target {self._plan_price(plan.get('target'), mkt)}  ", style="pnl.pos")
            info.append(f"R:R {plan.get('rr'):.1f}", style="text")
        else:
            info.append("no qualifying setup right now", style="dim")
        info.append("\nwhy    ", style="muted")
        why_text = str(row.get("best_reason") or row.get("gate") or "—")
        # Truncate long why text
        if len(why_text) > 40:
            why_text = why_text[:37] + "..."
        info.append(why_text, style="text")
        if row.get("gate") and row.get("best_reason"):
            info.append(f"   gate {row.get('gate')}", style="stale")

        # Compact metrics bar at bottom of info section
        am = metrics or {}
        metrics_parts = []
        if am.get("high_pct") is not None:
            metrics_parts.append(f"H:{am['high_pct']:+.1f}%")
        if am.get("low_pct") is not None:
            metrics_parts.append(f"L:{am['low_pct']:+.1f}%")
        if am.get("volume_ratio") is not None and am["volume_ratio"] != 1.0:
            metrics_parts.append(f"V:{am['volume_ratio']:.1f}x")
        if am.get("price_vs_vwap") is not None:
            metrics_parts.append(f"VWAP:{am['price_vs_vwap']:+.1f}%")
        if metrics_parts:
            info.append("  ", style="muted")
            info.append_text(Text(" · ".join(metrics_parts), style="dim"))

        # ── enhanced technical indicators ──────────────────────────────
        atr_pct = metrics.get("atr_pct")
        rvol = metrics.get("rvol")
        adx = metrics.get("adx")
        rsi = metrics.get("rsi")
        vwap_dist = metrics.get("vwap_dist")
        ema20 = metrics.get("ema20")
        ema50 = metrics.get("ema50")
        macd_line = metrics.get("macd_line")
        macd_signal = metrics.get("macd_signal")
        macd_hist = metrics.get("macd_hist")
        bb_mid = metrics.get("bb_mid")
        bb_upper = metrics.get("bb_upper")
        bb_lower = metrics.get("bb_lower")
        bb_pos = metrics.get("bb_pos")
        stoch_k = metrics.get("stoch_k")
        stoch_d = metrics.get("stoch_d")
        willr = metrics.get("willr")
        cci = metrics.get("cci")
        obv = metrics.get("obv")
        supertrend = metrics.get("supertrend")
        regime = row.get("regime", "—")

        # Enhanced indicator panel
        ind = Table.grid(padding=(0, 2))
        ind.add_column(justify="right")
        ind.add_column(justify="left")
        ind.add_column(justify="right")
        ind.add_column(justify="left")

        def _num(v, fmt):
            return fmt.format(v) if v is not None else "—"

        # Indicator rows
        ind_rows = [
            ("ATR%", _num(atr_pct, "{:.2f}%"), "RVOL", _num(rvol, "{:.2f}x")),
            ("ADX", _num(adx, "{:.1f}"), "RSI", _num(rsi, "{:.1f}")),
            ("VWAP", self._plan_price(metrics.get("vwap"), mkt), "dist", _num(vwap_dist, "{:+.2f}%")),
            ("EMA20", self._plan_price(ema20, mkt), "EMA50", self._plan_price(ema50, mkt)),
        ]

        # Add MACD if available
        if macd_line is not None or macd_signal is not None or macd_hist is not None:
            macd_str = "—"
            if macd_hist is not None:
                macd_str = f"{macd_hist:.2f}"
            elif macd_line is not None and macd_signal is not None:
                macd_str = f"{macd_line - macd_signal:.2f}"
            ind_rows.append(("MACD", macd_str, "Signal", _num(macd_signal, "{:.2f}")))

        # Add Bollinger Bands if available
        if bb_mid is not None or bb_upper is not None or bb_lower is not None:
            bb_pos_str = _num(bb_pos, "{:.1f}%") if bb_pos is not None else "—"
            ind_rows.append(("BB Pos", bb_pos_str, "Mid", self._plan_price(bb_mid, mkt)))

        # Add Stochastic if available
        if stoch_k is not None or stoch_d is not None:
            ind_rows.append(("Stoch", f"K:{_num(stoch_k, '{:.1f}')} D:{_num(stoch_d, '{:.1f}')}", "—", "—"))

        # Add Williams %R if available
        if willr is not None:
            ind_rows.append(("Williams %R", _num(willr, "{:.1f}"), "—", "—"))

        # Add CCI if available
        if cci is not None:
            ind_rows.append(("CCI", _num(cci, "{:.1f}"), "—", "—"))

        # Add OBV if available
        if obv is not None:
            ind_rows.append(("OBV", _num(obv, "{:.0f}"), "—", "—"))

        # Add Supertrend if available
        if supertrend is not None:
            st_dir = "UP" if supertrend > 0 else "DN"
            ind_rows.append(("Supertrend", st_dir, "—", "—"))

        # Additional market metrics
        am_rows = [
            ("High %", _num(metrics.get("high_pct"), "{:+.1f}%"), "Low %", _num(metrics.get("low_pct"), "{:+.1f}%")),
            ("Close% Open", _num(metrics.get("close_pct_from_open"), "{:+.2f}%"), "Vol Ratio", _num(metrics.get("volume_ratio"), "{:.2f}x")),
            ("Price VWAP", _num(metrics.get("price_vs_vwap"), "{:+.2f}%"), "—", "—"),
        ]
        am_filtered = [r for r in am_rows if r[1] != "—" or r[3] != "—"]

        for a, b, c, d in ind_rows:
            ind.add_row(Text(a, style="muted"), Text(str(b), style="text"),
                        Text(c, style="muted"), Text(str(d), style="text"))

        # Additional market metrics
        for a, b, c, d in am_filtered:
            ind.add_row(Text(a, style="muted"), Text(str(b), style="text"),
                        Text(c, style="muted"), Text(str(d), style="text"))

        # Enhanced stats panel
        st = row.get("stats") or {}
        if st:
            ind.add_row(Text("trades", style="muted"), Text(str(st.get("trades", 0)), style="text"),
                        Text("win", style="muted"), Text(fmt_winrate(st.get("win_rate", 0), st.get("trades", 0)), style="text"))
            ind.add_row(Text("PF", style="muted"), Text(fmt_pf(st.get("profit_factor")), style="text"),
                        Text("exp", style="muted"), Text(fmt_r(st.get("expectancy_r")), style="text"))

        # Firings / setups table - show only top 3 best setups
        firings = list(row.get("firings") or [])
        # Sort by confidence and filter out empty setups
        firings_sorted = sorted(firings, key=lambda f: f.get("confidence", 0) or 0, reverse=True)
        ft = Table(box=QUIET_BOX, expand=True, pad_edge=False)
        ft.add_column("Setup", min_width=8, max_width=14)
        ft.add_column("Side", width=6)
        ft.add_column("Conf", justify="right", width=5)
        ft.add_column("Entry", width=10)
        ft.add_column("Stop", width=10)
        ft.add_column("Target", width=10)
        ft.add_column("PnL", width=8)

        # Show top 3 setups with valid data
        for firing in firings_sorted[:3]:
            setup = firing.get("setup", "")
            if not setup or setup == "—":
                continue
            side = firing.get("side", "—")
            conf = firing.get("confidence", "—")
            entry = self._plan_price(firing.get("entry"), mkt) if firing.get("entry") else "—"
            stop = self._plan_price(firing.get("stop"), mkt) if firing.get("stop") else "—"
            target = self._plan_price(firing.get("target"), mkt) if firing.get("target") else "—"
            pnl = self._plan_price(firing.get("pnl"), mkt) if firing.get("pnl") is not None else "—"
            ft.add_row(
                Text(setup, style="text"),
                Text(side, style="pnl.pos" if side == "LONG" else "pnl.neg" if side == "SHORT" else "muted"),
                Text(str(conf)[:4], style="muted"),
                Text(entry, style="text"),
                Text(stop, style="pnl.neg"),
                Text(target, style="pnl.pos"),
                Text(pnl, style="pnl.pos" if firing.get("pnl", 0) > 0 else "pnl.neg"),
            )

        if ft.rows.__len__() == 0:
            ft.add_row("—", "—", "—", "—", "—", "—", "—")

        # Chart - braille chart of price history
        metrics = row.get("metrics") or {}
        price = row.get("live_price") or row.get("price")
        if price is None and hasattr(e, "ticker"):
            price = e.ticker.price(sym)
        chart_vals = list(metrics.get("chart") or [])
        if not chart_vals:
            chart_vals = list(getattr(e, "price_history", {}).get(sym, []) or [])[-60:]
        if price is not None:
            chart_vals.append(price)
        chart_w = max(20, (self.console.width or 120) - 12)
        from trader.ui.format import braille_chart
        chart_lines = braille_chart(chart_vals, chart_w, 5)
        chart_style = "status.open" if len(chart_vals) < 2 or chart_vals[-1] >= chart_vals[0] else "pnl.neg"
        chart = Text("\n".join(chart_lines), style=chart_style)
        if not chart_vals:
            chart = Text("collecting quotes…", style="dim")

        # Layout: info + indicators + setups + chart
        body = Group(info, ind, ft, chart)
        title = f"Detail · {sym} · {mkt}"
        return Panel(body, title=title, box=box.ROUNDED, border_style=self.HV)

    def _view_panel(self, view):
        if view == "positions":
            return self._positions_view()
        if view == "analytics":
            return self._analytics_view()
        if view == "diagnostics":
            return self._diagnostics_view()
        if view == "report":
            return self._report_view()
        return self._log_panel(full=True)

    def _positions_view(self):
        e = self.engine
        rows = e.portfolio.positions
        st = e.portfolio.stats()
        total_inr = e.portfolio.unrealized_inr()
        eq = e.portfolio.total_equity_inr()
        risk_rows = self._risk_rows()
        risk = sum(v["risk"] for v in risk_rows.values())
        summary = Text()
        summary.append("Equity ", style="muted")
        summary.append(fmt_money(eq, "INR"), style="text")
        summary.append(" · unrealised ", style="muted")
        summary.append(fmt_pnl(total_inr, "INR"), style=pnl_style(total_inr))
        summary.append(" · open risk ", style="muted")
        summary.append(fmt_money(risk, "INR"), style="text")
        summary.append(f" ({risk / eq * 100:.1f}% eq)" if eq else "", style="muted")
        summary.append(" · closed ", style="muted")
        summary.append(f"{st['trades']}", style="text")
        spec = [
            ("Mkt", 3, 3, "left", 9),
            ("Symbol", 8, 18, "left", 10),
            ("Side", 4, 7, "center", 8),
            ("Qty", 4, 8, "right", 4),
            ("Strategy", 6, 22, "left", 6),
            ("Entry", 5, 11, "right", 9),
            ("Last", 5, 11, "right", 6),
            ("Stop", 5, 11, "right", 7),
            ("Target", 5, 11, "right", 8),
            ("P&L", 7, 11, "right", 11),
            ("%", 5, 7, "right", 5),
            ("R", 4, 6, "right", 7),
            ("Age", 3, 5, "right", 4),
            ("State", 3, 5, "center", 4),
            ("Gauge", 8, 14, "center", 6),
        ]
        cols = self._fit_columns(spec, self.console.width - 4)
        t = Table(box=QUIET_BOX, expand=True, pad_edge=False, show_header=True)
        for name, mn, mx, just, _ in cols:
            kw = dict(min_width=mn, max_width=mx, justify=just, no_wrap=True)
            if name in ("Symbol", "Strategy"):
                kw["overflow"] = "ellipsis"
            t.add_column(name, **kw)
        if not rows:
            t.add_row(*[Text("—", style="muted") for _ in cols])
        for p in rows:
            upnl = (p.current - p.entry) * p.qty if p.side.value == "LONG" else (p.entry - p.current) * p.qty
            pct = (upnl / (p.entry * p.qty) * 100) if p.entry and p.qty else 0.0
            r, mfe, mae = self._position_r(p)
            age = (datetime.now(timezone.utc) - p.opened_at).total_seconds()
            cells = {
                "Mkt": mkt_code(e.markets_cfg, p.market),
                "Symbol": p.symbol,
                "Side": Text(f"▲ {p.side.value}" if p.side.value == "LONG" else f"▼ {p.side.value}",
                             style="pnl.pos" if p.side.value == "LONG" else "pnl.neg"),
                "Qty": fmt_qty(p.qty),
                "Strategy": self._strategy_label(p.strategy),
                "Entry": fmt_price(self._inr(p.entry, p.market), "INR"),
                "Last": fmt_price(self._inr(p.current, p.market), "INR"),
                "Stop": fmt_price(self._inr(p.stop, p.market), "INR"),
                "Target": fmt_price(self._inr(p.target, p.market), "INR"),
                "P&L": Text(fmt_pnl(self._inr(upnl, p.market), "INR"), style=pnl_style(upnl)),
                "%": Text(fmt_pct(pct, 1), style=pnl_style(upnl)),
                "R": Text(fmt_r(r), style=pnl_style(r if r is not None else 0)),
                "Age": fmt_age(age),
                "State": self._stop_state(p),
                "Gauge": gauge(p.stop, p.current, p.target, 12),
            }
            t.add_row(*[cells[c[0]] for c in cols])
        if rows:
            cells = {
                "Mkt": "", "Symbol": Text("TOTAL", style="accent"), "Side": "", "Qty": "",
                "Strategy": "", "Entry": "", "Last": "", "Stop": "", "Target": "",
                "P&L": Text(fmt_pnl(total_inr, "INR"), style=pnl_style(total_inr)),
                "%": "", "R": "", "Age": "", "State": "", "Gauge": "",
            }
            t.add_row(*[cells[c[0]] for c in cols], style="muted")
        body = Group(summary, Text(""), t)
        return Panel(body, title=f"Positions · {len(rows)} open", box=box.ROUNDED,
                     border_style="border")

    def _equity_curve_panel(self, height):
        series = self._equity_series(240)
        width = max(10, self.console.width - 12)
        lines = braille_chart(series, width, max(3, height - 5))
        style = "status.open" if len(series) < 2 or series[-1] >= series[0] else "pnl.neg"
        body = Text("\n".join(lines), style=style)
        summary = Text()
        if series:
            summary.append("start ", style="muted")
            summary.append(fmt_money(series[0], "INR"), style="text")
            summary.append(" · now ", style="muted")
            summary.append(fmt_money(series[-1], "INR"), style="text")
            change = series[-1] - series[0]
            summary.append(" · change ", style="muted")
            summary.append(fmt_pnl(change, "INR"), style=pnl_style(change))
            summary.append(" · high ", style="muted")
            summary.append(fmt_money(max(series), "INR"), style="text")
            summary.append(" · low ", style="muted")
            summary.append(fmt_money(min(series), "INR"), style="text")
        else:
            summary.append("Waiting for the first scan", style="muted")
        return Panel(Group(summary, body), title="Equity · session", box=box.ROUNDED,
                     border_style="border")

    def _strategy_panel(self):
        stats = self.engine.portfolio.strategy_stats()
        ordered = sorted(stats.items(), key=lambda kv: -kv[1]["pnl"])
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column("Strategy", min_width=8, max_width=24, overflow="ellipsis")
        t.add_column("N", justify="right", width=4)
        t.add_column("Win", justify="right", width=6)
        t.add_column("P&L", justify="right", min_width=8, max_width=12)
        t.add_column("Avg", justify="right", min_width=8, max_width=12)
        t.add_column("AvgR", justify="right", width=5)
        if not ordered:
            return Panel(Text("No closed trades yet", style="muted"),
                         title="Strategy leaderboard", box=box.ROUNDED,
                         border_style="border")
        rs = {}
        for tr in self.engine.portfolio.trades:
            rs.setdefault(tr.strategy, []).append(tr.r_multiple)
        for name, s in ordered[:20]:
            avg = s["pnl"] / s["trades"] if s["trades"] else 0.0
            rr = sum(rs.get(name, [])) / len(rs[name]) if rs.get(name) else 0.0
            t.add_row(self._strategy_label(name), str(s["trades"]),
                      fmt_winrate(s["win_rate"], s["trades"]),
                      Text(fmt_pnl(s["pnl"], "INR"), style=pnl_style(s["pnl"])),
                      Text(fmt_pnl(avg, "INR"), style=pnl_style(avg)),
                      Text(fmt_r(rr), style=pnl_style(rr)))
        return Panel(t, title="Strategy leaderboard", box=box.ROUNDED,
                     border_style="border")

    def _exit_panel(self):
        e = self.engine
        counts = {}
        pnl = {}
        for tr in e.portfolio.trades:
            reason = tr.exit_reason.value
            counts[reason] = counts.get(reason, 0) + 1
            pnl[reason] = pnl.get(reason, 0.0) + self._inr(tr.pnl, tr.market)
        total = sum(counts.values())
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column("Exit", min_width=6, max_width=10)
        t.add_column("N", justify="right", width=4)
        t.add_column("Share", justify="left", min_width=8, max_width=14)
        t.add_column("P&L", justify="right", min_width=8, max_width=12)
        t.add_column("Avg", justify="right", min_width=8, max_width=12)
        if not total:
            return Panel(Text("No exits yet", style="muted"), title="Exit reasons",
                         box=box.ROUNDED, border_style="border")
        max_n = max(counts.values())
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            bar = "▇" * max(1, int(n / max_n * 10))
            avg = pnl[reason] / n if n else 0.0
            t.add_row(reason.replace("STOPLOSS", "STOP").replace("SQUAREOFF", "SQUARE"),
                      str(n), Text(bar, style="accent"),
                      Text(fmt_pnl(pnl[reason], "INR"), style=pnl_style(pnl[reason])),
                      Text(fmt_pnl(avg, "INR"), style=pnl_style(avg)))
        return Panel(t, title="Exit reasons", box=box.ROUNDED, border_style="border")

    def _hour_panel(self):
        e = self.engine
        ist = ZoneInfo("Asia/Kolkata")
        buckets = {}
        for tr in e.portfolio.trades:
            h = tr.closed_at.astimezone(ist).hour
            b = buckets.setdefault(h, [0, 0.0])
            b[0] += 1
            b[1] += self._inr(tr.pnl, tr.market)
        if not buckets:
            return Panel(Text("No closed trades yet", style="muted"),
                         title="P&L by hour (IST)", box=box.ROUNDED,
                         border_style="border")
        max_n = max(b[0] for b in buckets.values())
        lines = Text()
        for h in sorted(buckets):
            n, pnl = buckets[h]
            lines.append(f"{h:02d}:00 ", style="muted")
            lines.append("▇" * max(1, int(n / max_n * 12)), style="accent")
            lines.append(f" {n:>2}  ", style="text")
            lines.append(fmt_pnl(pnl, "INR"), style=pnl_style(pnl))
            lines.append("\n")
        return Panel(lines, title="P&L by hour (IST)", box=box.ROUNDED,
                     border_style="border")

    def _report_bundle(self):
        return self._rc("report_bundle", self._report_bundle_uncached)

    def _report_bundle_uncached(self):
        p = self.engine.portfolio
        st = p.stats()
        trades = sorted(p.trades, key=lambda t: t.closed_at)
        pnls = [self._inr(t.pnl, t.market) for t in trades]
        cum = [0.0]
        for pnl in pnls:
            cum.append(cum[-1] + pnl)
        charges = sum(self._inr(t.charges, t.market) for t in trades)
        holds = [max(0.0, (t.closed_at - t.opened_at).total_seconds())
                 for t in trades]
        avg_hold = (sum(holds) / len(holds)) if holds else 0.0
        best = max(zip(trades, pnls), key=lambda x: x[1]) if trades else None
        worst = min(zip(trades, pnls), key=lambda x: x[1]) if trades else None

        def blank():
            return {"n": 0, "w": 0, "pnl": 0.0, "r": 0.0,
                    "gw": 0.0, "gl": 0.0}

        markets, strategies, exits = {}, {}, {}
        for t, pnl in zip(trades, pnls):
            exit_key = str(getattr(t.exit_reason, "value", t.exit_reason))
            for store, key in ((markets, t.market),
                               (strategies, t.strategy),
                               (exits, exit_key)):
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
                g["win_rate"] = (g["w"] / g["n"] * 100) if g["n"] else 0.0
                g["pf"] = (g["gw"] / g["gl"]) if g["gl"] else (
                    float("inf") if g["gw"] else 0.0)
                g["avg_r"] = (g["r"] / g["n"]) if g["n"] else 0.0

        sides = {}
        for t, pnl in zip(trades, pnls):
            s = sides.setdefault(t.side.value,
                                 {"n": 0, "w": 0, "pnl": 0.0})
            s["n"] += 1
            s["w"] += 1 if pnl > 0 else 0
            s["pnl"] += pnl

        buckets = [("<= -1R", 0), ("-1..0", 0), ("0..1", 0),
                   ("1..2", 0), ("> 2R", 0)]
        counts = [0] * 5
        for t in trades:
            r = t.r_multiple
            idx = 0 if r <= -1 else (1 if r <= 0 else (
                2 if r <= 1 else (3 if r <= 2 else 4)))
            counts[idx] += 1
        buckets = [(label, counts[i]) for i, (label, _) in enumerate(buckets)]

        # ── add scanner metrics summary ──────────────────────────────
        # Collect metric data from last scan across all markets
        scanner_metrics = {}
        for market, rows in self.engine.last_scan.items():
            for r in rows:
                m = r.get("metrics", {})
                if not m:
                    continue
                sym = r.get("symbol", "")
                # Only keep first occurrence per symbol, or merge
                if sym not in scanner_metrics:
                    scanner_metrics[sym] = m
                # Merge additional metrics
                for k, v in m.items():
                    if k not in scanner_metrics[sym] or scanner_metrics[sym][k] is None:
                        scanner_metrics[sym][k] = v

        # Aggregate metric summaries
        metric_summary = {
            "avg_rsi": "—",
            "avg_adx": "—",
            "avg_atr_pct": "—",
            "avg_rvol": "—",
            "score_dist": "—",
        }
        if scanner_metrics:
            rsi_vals = [m.get("rsi") for m in scanner_metrics.values() if m.get("rsi") is not None]
            adx_vals = [m.get("adx") for m in scanner_metrics.values() if m.get("adx") is not None]
            atr_vals = [m.get("atr_pct") for m in scanner_metrics.values() if m.get("atr_pct") is not None]
            rvol_vals = [m.get("rvol") for m in scanner_metrics.values() if m.get("rvol") is not None]
            if rsi_vals:
                metric_summary["avg_rsi"] = f"{sum(rsi_vals)/len(rsi_vals):.1f}"
            if adx_vals:
                metric_summary["avg_adx"] = f"{sum(adx_vals)/len(adx_vals):.1f}"
            if atr_vals:
                metric_summary["avg_atr_pct"] = f"{sum(atr_vals)/len(atr_vals):.2f}%"
            if rvol_vals:
                metric_summary["avg_rvol"] = f"{sum(rvol_vals)/len(rvol_vals):.2f}x"

        return {"trades": trades, "pnls": pnls, "cum": cum, "stats": st,
                "charges": charges, "avg_hold": avg_hold, "best": best,
                "worst": worst, "markets": markets,
                "strategies": strategies, "exits": exits,
                "sides": sides, "buckets": buckets,
                "scanner_metrics": scanner_metrics,
                "metric_summary": metric_summary}

    def _report_summary_panel(self):
        b = self._report_bundle()
        inner = max(20, (self.console.width or 120) - 4)
        if not b["trades"]:
            return Panel(Text("No closed trades yet — the report fills as trades close.",
                              style="muted"),
                         title="Session report · closed trades",
                         box=box.ROUNDED, border_style="border")
        st = b["stats"]
        rate_style = "status.open" if st["win_rate"] >= 50 else "pnl.neg"

        l1 = Text()
        l1.append("Closed ", style="muted")
        l1.append(str(st["trades"]), style="text")
        l1.append(f"  ·  {st['wins']}W / {st['losses']}L  ·  win ", style="muted")
        l1.append(_block_gauge(st["win_rate"] / 100.0, 10), style=rate_style)
        l1.append(" " + fmt_winrate(st["win_rate"], st["trades"]), style=rate_style)

        l2 = Text()
        l2.append("Net ", style="muted")
        l2.append(fmt_pnl(st["total_pnl"], "INR"), style=pnl_style(st["total_pnl"]))
        l2.append("  ·  PF ", style="muted")
        l2.append(fmt_pf(st["profit_factor"]), style=pnl_style(st["profit_factor"] - 1))
        l2.append("  ·  expectancy ", style="muted")
        l2.append(fmt_r(st["expectancy_r"]),
                  style=pnl_style(st["expectancy_r"]))

        l3 = Text()
        l3.append("Gross ", style="muted")
        l3.append(fmt_pnl(st["gross_win"], "INR"), style="pnl.pos")
        l3.append(" / ", style="muted")
        l3.append(fmt_pnl(-st["gross_loss"], "INR"), style="pnl.neg")
        l3.append("  ·  charges ", style="muted")
        l3.append(fmt_money(b["charges"], "INR"), style="muted")
        l3.append("  ·  avg trade ", style="muted")
        avg_trade = st["total_pnl"] / st["trades"] if st["trades"] else 0.0
        l3.append(fmt_pnl(avg_trade, "INR"), style=pnl_style(avg_trade))

        l4 = Text()
        l4.append("Avg win ", style="muted")
        l4.append(fmt_pnl(st["avg_win"], "INR"), style="pnl.pos")
        l4.append("  ·  avg loss ", style="muted")
        l4.append(fmt_pnl(st["avg_loss"], "INR"), style="pnl.neg")
        l4.append("  ·  best ", style="muted")
        if b["best"]:
            bt, bp = b["best"]
            l4.append(fmt_pnl(bp, "INR"), style="pnl.pos")
            l4.append(f" {bt.symbol}", style="muted")
        else:
            l4.append("—", style="muted")
        l4.append("  ·  worst ", style="muted")
        if b["worst"]:
            wt, wp = b["worst"]
            l4.append(fmt_pnl(wp, "INR"), style="pnl.neg")
            l4.append(f" {wt.symbol}", style="muted")
        else:
            l4.append("—", style="muted")
        l4.append("  ·  metric avg ", style="muted")
        ms = b.get("metric_summary", {})
        l4.append(f"RSI{ms['avg_rsi']} ADX{ms['avg_adx']} ATR{ms['avg_atr_pct']} Rvol{ms['avg_rvol']}", style="dim")

        l5 = Text()
        for side, label in (("LONG", "Long"), ("SHORT", "Short")):
            s = b["sides"].get(side)
            l5.append(f"{label} ", style="muted")
            if not s:
                l5.append("0", style="text")
            else:
                wr = s["w"] / s["n"] * 100 if s["n"] else 0.0
                l5.append(f"{s['n']} · {wr:.0f}% · ", style="text")
                l5.append(fmt_pnl_compact(s["pnl"], "INR"),
                          style=pnl_style(s["pnl"]))
            l5.append("   ", style="muted")
        l5.append("hold ", style="muted")
        l5.append(fmt_age(b["avg_hold"]), style="text")

        l6 = Text()
        l6.append("Exits ", style="muted")
        labels = {"target": "target", "sl": "stop", "trailing": "trail",
                  "square-off": "square", "manual": "manual",
                  "kill-switch": "kill"}
        parts = []
        for key, g in sorted(b["exits"].items(), key=lambda kv: -kv[1]["n"]):
            parts.append(f"{labels.get(key, key)} {g['n']}")
        l6.append(" · ".join(parts) if parts else "—", style="text")

        # Scanner metrics overview
        sm = b.get("scanner_metrics", {})
        l7 = Text()
        l7.append("Scanner metrics ", style="muted")
        top5 = sorted(sm.keys())[:5] if sm else []
        for sym in top5:
            m = sm.get(sym, {})
            rsi = m.get("rsi", "—")
            adx = m.get("adx", "—")
            rvol = m.get("rvol", "—")
            l7.append(f" {sym}:R{rsi} A{adx} R{rvol} ", style="dim")
        l7.append("\n", style="dim")

        lines = [l1, l2, l3, l4, l5, l6]
        body = Text()
        for i, line in enumerate(lines):
            line.truncate(inner)
            if i:
                body.append("\n")
            body.append_text(line)
        return Panel(body, title="Session report · closed trades",
                     box=box.ROUNDED, border_style="border")

    def _report_curve_panel(self, height):
        b = self._report_bundle()
        if not b["trades"]:
            return Panel(Text("No closed trades yet", style="muted"),
                         title="Cumulative P&L · after charges",
                         box=box.ROUNDED, border_style="border")
        width = max(16, int((self.console.width or 120) * 0.6) - 4)
        chart_h = max(1, height - 3)
        lines = braille_chart(b["cum"], width, chart_h)
        current = b["cum"][-1]
        label = Text()
        label.append("peak ", style="muted")
        label.append(fmt_pnl_compact(max(b["cum"]), "INR"), style="pnl.pos")
        label.append("  low ", style="muted")
        label.append(fmt_pnl_compact(min(b["cum"]), "INR"), style="pnl.neg")
        label.append("  now ", style="muted")
        label.append(fmt_pnl_compact(current, "INR"), style=pnl_style(current))
        chart = Text("\n".join(lines), style="pnl.pos" if current >= 0 else "pnl.neg")
        return Panel(Group(label, chart), title="Cumulative P&L · after charges",
                     box=box.ROUNDED, border_style="border")

    def _report_r_panel(self, height):
        b = self._report_bundle()
        if not b["trades"]:
            return Panel(Text("No R history yet", style="muted"),
                         title="R distribution", box=box.ROUNDED,
                         border_style="border")
        counts = b["buckets"]
        most = max((n for _label, n in counts), default=0) or 1
        width = max(6, min(36, int((self.console.width or 120) * 0.18)))
        lines = Text()
        for label, n in counts:
            style = "pnl.neg" if label.startswith(("<", "-")) else "pnl.pos"
            lines.append(f"{label:<6}", style="muted")
            lines.append(f"{n:>3} ", style="text")
            lines.append("█" * max(1, int(n / most * width)) if n else "",
                         style=style)
            lines.append("\n")
        lines.append("avg ", style="muted")
        lines.append(fmt_r(b["stats"]["expectancy_r"]),
                     style=pnl_style(b["stats"]["expectancy_r"]))
        return Panel(lines, title=f"R distribution · {len(b['trades'])}",
                     box=box.ROUNDED, border_style="border")

    def _report_group_panel(self, title, groups, height, key_label,
                            labeler=None, compact=False, key_w=None):
        if not groups:
            return Panel(Text("No closed trades yet", style="muted"),
                         title=title, box=box.ROUNDED,
                         border_style="border")
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column(key_label, width=key_w or (10 if compact else 14),
                     overflow="ellipsis", no_wrap=True)
        t.add_column("N", justify="right", width=3)
        t.add_column("Win", justify="right", width=6)
        t.add_column("Net", justify="right", width=9)
        if not compact:
            t.add_column("PF", justify="right", width=4)
        t.add_column("AvgR", justify="right", width=5)
        rows = sorted(groups.items(), key=lambda kv: -kv[1]["pnl"])
        room = max(1, height - QUIET_TABLE_CHROME - 2 - (1 if not compact else 0))
        for key, g in rows[:room]:
            name = labeler(key) if labeler else key
            cells = [name, str(g["n"]),
                     fmt_winrate(g["win_rate"], g["n"]),
                     Text(fmt_pnl(g["pnl"], "INR"),
                          style=pnl_style(g["pnl"]))]
            if not compact:
                cells.append(Text(fmt_pf(g["pf"]),
                                  style=pnl_style(g["pf"] - 1)))
            cells.append(Text(fmt_r(g["avg_r"]),
                              style=pnl_style(g["avg_r"])))
            t.add_row(*cells)
        return Panel(t, title=title, box=box.ROUNDED, border_style="border")

    def _report_market_panel(self, height):
        compact = (self.console.width or 120) < 110
        return self._report_group_panel(
            "By market", self._report_bundle()["markets"], height, "Mkt",
            lambda m: mkt_code(self.engine.markets_cfg, m), compact, 4)

    def _report_strategy_panel(self, height):
        compact = (self.console.width or 120) < 110
        return self._report_group_panel(
            "By strategy", self._report_bundle()["strategies"], height,
            "Strategy", self._strategy_label, compact, 8 if compact else 14)

    def _report_exit_panel(self, height):
        b = self._report_bundle()
        groups = b["exits"]
        if not groups:
            return Panel(Text("No exits yet", style="muted"),
                         title="By exit", box=box.ROUNDED,
                         border_style="border")
        labels = {"target": "TARGET", "sl": "SL", "trailing": "TRAIL",
                  "square-off": "SQUARE", "manual": "MANUAL",
                  "kill-switch": "KILL"}
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column("Exit", width=8)
        t.add_column("N", justify="right", width=3)
        t.add_column("Net", justify="right", width=10)
        t.add_column("Avg", justify="right", width=10)
        rows = sorted(groups.items(), key=lambda kv: -kv[1]["n"])
        room = max(1, height - QUIET_TABLE_CHROME - 2)
        for key, g in rows[:room]:
            avg = g["pnl"] / g["n"] if g["n"] else 0.0
            t.add_row(labels.get(key, key.upper()), str(g["n"]),
                      Text(fmt_pnl(g["pnl"], "INR"),
                           style=pnl_style(g["pnl"])),
                      Text(fmt_pnl(avg, "INR"), style=pnl_style(avg)))
        return Panel(t, title="By exit", box=box.ROUNDED,
                     border_style="border")

    def _report_recent_panel(self, height):
        b = self._report_bundle()
        if not b["trades"]:
            return Panel(Text("No closed trades yet", style="muted"),
                         title="Recent closes", box=box.ROUNDED,
                         border_style="border")
        labels = {"target": "TARGET", "sl": "SL", "trailing": "TRAIL",
                  "square-off": "SQUARE", "manual": "MANUAL",
                  "kill-switch": "KILL"}
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column("Time", width=5)
        t.add_column("Mkt", width=4)
        t.add_column("Symbol", width=16, overflow="ellipsis", no_wrap=True)
        t.add_column("Strategy", width=14, overflow="ellipsis", no_wrap=True)
        t.add_column("Side", width=7)
        t.add_column("Exit", width=7)
        t.add_column("Qty", justify="right", width=7)
        t.add_column("Held", justify="right", width=5)
        t.add_column("Net", justify="right", width=10)
        t.add_column("R", justify="right", width=5)
        room = max(1, height - QUIET_TABLE_CHROME - 2)
        for tr in reversed(b["trades"][-room:]):
            pnl = self._inr(tr.pnl, tr.market)
            side = "▲ LONG" if tr.side.value == "LONG" else "▼ SHORT"
            t.add_row(
                tr.closed_at.astimezone().strftime("%H:%M"),
                mkt_code(self.engine.markets_cfg, tr.market),
                tr.symbol, self._strategy_label(tr.strategy),
                Text(side, style="pnl.pos" if tr.side.value == "LONG" else "pnl.neg"),
                labels.get(str(getattr(tr.exit_reason, "value", tr.exit_reason)),
                           str(getattr(tr.exit_reason, "value", tr.exit_reason)).upper()),
                fmt_qty(tr.qty),
                fmt_age((tr.closed_at - tr.opened_at).total_seconds()),
                Text(fmt_pnl(pnl, "INR"), style=pnl_style(pnl)),
                Text(fmt_r(tr.r_multiple), style=pnl_style(tr.r_multiple)))
        return Panel(t, title="Recent closes · newest first",
                     box=box.ROUNDED, border_style="border")

    def _report_view(self):
        h = self.console.height or 40
        w = self.console.width or 120
        budget = max(10, h - self._header_h() - 1)
        top_h = 8
        mid_h = max(7, min(16, budget // 3))
        low_h = max(6, budget - top_h - mid_h)
        recent_h = 0
        if w >= 200 and budget >= 50:
            recent_h = min(14, max(8, low_h - 18))
            low_h -= recent_h
        layout = Layout()
        mid = Layout()
        mid.split_row(Layout(self._report_curve_panel(mid_h), ratio=3),
                      Layout(self._report_r_panel(mid_h), ratio=2))
        low = Layout()
        if w >= 200:
            low.split_row(Layout(self._report_market_panel(low_h), ratio=1),
                          Layout(self._report_strategy_panel(low_h), ratio=1),
                          Layout(self._report_exit_panel(low_h), ratio=1))
        else:
            low.split_row(Layout(self._report_market_panel(low_h), ratio=1),
                          Layout(self._report_strategy_panel(low_h), ratio=1))
        parts = [Layout(self._report_summary_panel(), size=top_h),
                 Layout(mid, size=mid_h), Layout(low, size=low_h)]
        if recent_h:
            parts.append(Layout(self._report_recent_panel(recent_h),
                                size=recent_h))
        layout.split_column(*parts)
        return layout

    def _analytics_view(self):
        h = self.console.height or 40
        budget = max(10, h - self._header_h() - 1)
        top_h = max(8, min(13, budget // 3))
        layout = Layout()
        if self._wide():
            right = Layout()
            right.split_column(Layout(self._exit_panel()),
                               Layout(self._hour_panel()))
            low = Layout()
            low.split_row(Layout(self._strategy_panel(), ratio=3),
                          Layout(right, ratio=2))
            layout.split_column(Layout(self._equity_curve_panel(top_h), size=top_h),
                                low)
        else:
            rest = max(12, budget - top_h)
            layout.split_column(
                Layout(self._equity_curve_panel(top_h), size=top_h),
                Layout(self._strategy_panel(), size=rest // 2),
                Layout(self._exit_panel(), size=rest // 4),
                Layout(self._hour_panel()),
            )
        return layout

    def _funnel_panel(self):
        cs = getattr(self.engine, "last_cycle_stats", {}) or {}
        scanned = int(cs.get("scanned", 0))
        stages = [
            ("Scanned", scanned),
            ("Signals", int(cs.get("signals", 0))),
            ("Candidates", int(cs.get("candidates", 0))),
            ("Blocked", int(cs.get("blocked_gate", 0))),
            ("Opened", int(cs.get("opened", 0))),
            ("Stale", int(cs.get("stale", 0))),
        ]
        lines = Text()
        base = max(1, scanned)
        for name, value in stages:
            lines.append(f"{name:<10}", style="muted")
            lines.append(f"{value:>5} ", style="text")
            lines.append(_block_gauge(value / base if base else 0, 12), style="accent")
            lines.append("\n")
        if not cs:
            lines.append("Waiting for the first scan", style="muted")
        return Panel(lines, title="Signal funnel · last cycle", box=box.ROUNDED,
                     border_style="border")

    def _timing_panel(self):
        e = self.engine
        lines = Text()
        lines.append("last scan   ", style="muted")
        lines.append(f"{e.last_scan_elapsed:.2f}s", style="text")
        lines.append("\nquote feed  ", style="muted")
        age = e.ticker.age_of_newest() if hasattr(e, "ticker") else None
        lines.append(fmt_age(age) if age is not None else ("off" if not e.live_quotes else "—"),
                     style="text")
        lines.append("\nquotes      ", style="muted")
        try:
            lines.append(str(len(e.ticker.snapshot())), style="text")
        except Exception:
            lines.append("—", style="muted")
        lines.append("\nthreads     ", style="muted")
        names = [t.name for t in threading.enumerate() if t.is_alive()]
        lines.append(f"{len(names)} alive", style="text")
        lines.append("\n" + ", ".join(names[:6]), style="dim")
        return Panel(lines, title="Timings", box=box.ROUNDED, border_style="border")

    def _skip_panel(self):
        counts = dict(getattr(self.engine, "skip_counts", {}) or {})
        if not counts:
            return Panel(Text("Nothing skipped this session", style="muted"),
                         title="Rejections · session", box=box.ROUNDED,
                         border_style="border")
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column("Reason", min_width=8, max_width=22)
        t.add_column("N", justify="right", width=5)
        t.add_column("", justify="left")
        max_n = max(counts.values())
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1])[:14]:
            t.add_row(reason, str(n),
                      Text("▇" * max(1, int(n / max_n * 16)), style="stale"))
        return Panel(t, title="Rejections · session", box=box.ROUNDED,
                     border_style="border")

    def _health_panel(self):
        e = self.engine
        t = Table(box=QUIET_BOX, expand=False, pad_edge=False)
        t.add_column("Mkt", width=4)
        t.add_column("Status", min_width=7, max_width=10)
        t.add_column("Rows", justify="right", width=5)
        t.add_column("Signals", justify="right", width=7)
        t.add_column("Stale", justify="right", width=5)
        t.add_column("Age", justify="right", width=5)
        t.add_column("Source", min_width=6, max_width=8)
        for m in self.markets:
            status = e.clock.status(m)
            rows = e.last_scan.get(m, [])
            stale = sum(1 for r in rows if r.get("stale"))
            sigs = sum(1 for r in rows if r.get("signal") in ("LONG", "SHORT"))
            ages = []
            for r in rows:
                up = r.get("updated")
                if up is not None:
                    try:
                        ages.append((datetime.now(timezone.utc) - up).total_seconds())
                    except Exception:
                        pass
            t.add_row(
                mkt_code(e.markets_cfg, m),
                Text(status["status"], style=status_style(status["status"])),
                str(len(rows)), str(sigs), str(stale),
                fmt_age(min(ages)) if ages else "—",
                "live" if e.live_quotes else "off",
            )
        return Panel(t, title="Data health · all markets", box=box.ROUNDED,
                     border_style="border")

    def _diagnostics_view(self):
        h = self.console.height or 40
        budget = max(10, h - self._header_h() - 1)
        top_h = max(9, min(12, budget // 3))
        layout = Layout()
        top = Layout()
        top.split_row(Layout(self._funnel_panel(), ratio=3),
                      Layout(self._timing_panel(), ratio=2))
        if self._wide():
            low = Layout()
            low.split_row(Layout(self._skip_panel(), ratio=2),
                          Layout(self._health_panel(), ratio=3))
            layout.split_column(Layout(top, size=top_h), low)
        else:
            layout.split_column(Layout(top, size=top_h),
                                Layout(self._skip_panel()),
                                Layout(self._health_panel()))
        return layout

    def _strategy_label(self, name: str) -> str:
        """Friendly name for a setup or rule (rules are named rule_xxx)."""
        if not name:
            return ""
        labels = self._label_cache
        if not labels:
            for s in getattr(self.engine, "strategies", []):
                try:
                    labels[s.name] = s.display_name()
                except Exception:
                    labels[s.name] = s.name
        return labels.get(name, name)

    def _short_label(self, name: str) -> str:
        """A three-to-ten character label that still says what the setup is."""
        if not name:
            return "—"
        short = SHORT_LABELS.get(name)
        if short:
            return short
        label = self._strategy_label(name) or name
        if len(label) <= 10:
            return label
        swaps = (
            ("Breakout", "BO"), ("Breakdown", "BD"), ("Momentum", "Mom"),
            ("Reversal", "Rev"), ("Confirmation", "Conf"), ("Continuation", "Cont"),
            ("Pullback", "PB"), ("Deviation", "Dev"), ("Expansion", "Exp"),
            ("Squeeze", "Sqz"), ("Oversold", "OS"), ("Overbought", "OB"),
            ("Volume", "Vol"), ("Trend", "Tr"), ("Short", "S"), ("Cross", "X"),
        )
        for a, b in swaps:
            label = label.replace(a, b)
        return truncate(label, 10)

    def _position_r(self, p):
        """Live R multiple and MFE/MAE, in units of the original risk."""
        risk_per_unit = self._initial_risk.get(p.id)
        if risk_per_unit is None:
            risk_per_unit = abs(p.entry - p.stop)
            if risk_per_unit > 0:
                self._initial_risk[p.id] = risk_per_unit
        if not risk_per_unit or not p.qty:
            return None, None, None
        upnl = (p.current - p.entry) * p.qty if p.side.value == "LONG" else (p.entry - p.current) * p.qty
        r = upnl / (risk_per_unit * p.qty)
        rec = self._excursion.get(p.id)
        if rec is None:
            rec = [r, r]
            self._excursion[p.id] = rec
        else:
            rec[0] = max(rec[0], r)
            rec[1] = min(rec[1], r)
        return r, rec[0], rec[1]

    def _stop_state(self, p):
        if getattr(p, "moved_to_be", False):
            return Text("BE", style="status.preclose")
        if (p.side.value == "LONG" and p.stop > p.entry) or \
           (p.side.value == "SHORT" and p.stop < p.entry):
            return Text("TRAIL", style="accent")
        return Text("SL", style="muted")

    def _active_trades(self, compact=False) -> Panel:
        e = self.engine
        avail = self._panel_width("active")
        spec = [
            ("Mkt", 3, 3, "left", 9),
            ("Symbol", 8, 16, "left", 10),
            ("Side", 4, 7, "center", 8),
            ("Qty", 4, 7, "right", 4),
            ("Strategy", 6, 20, "left", 6),
            ("Entry", 5, 10, "right", 5),
            ("Last", 5, 10, "right", 6),
            ("P&L", 7, 10, "right", 11),
            ("R", 4, 6, "right", 7),
            ("Age", 4, 5, "right", 5),
            ("State", 3, 5, "center", 4),
            ("Gauge", 8, 10, "center", 9),
            ("Exc", 7, 9, "right", 3),
        ]
        cols = self._fit_columns(spec, avail)
        _, _, bottom_h, active_h = self._space()
        allocated = active_h if compact else bottom_h
        need = 2 + QUIET_TABLE_CHROME + max(1, len(e.portfolio.positions)) + 1
        tight = allocated < need
        room = max(1, allocated - 2)
        show_header = room >= 3
        slots = max(1, room - (1 if show_header else 0) - 1)
        t = Table(box=None if tight else QUIET_BOX, expand=True, pad_edge=False,
                  show_header=show_header)
        for name, mn, mx, just, _ in cols:
            kw = dict(min_width=mn, max_width=mx, justify=just, no_wrap=True)
            if name in ("Symbol", "Strategy"):
                kw["overflow"] = "ellipsis"
            t.add_column(name, **kw)
        if not e.portfolio.positions:
            self._excursion.clear()
            self._initial_risk.clear()
            body = Text("No open positions · a fill appears here the moment a setup signals",
                        style="muted")
            return Panel(Align(body, align="left", vertical="middle"),
                         title="Active Trades", box=box.ROUNDED, border_style="border")
        live_ids = set()
        for p in e.portfolio.positions:
            live_ids.add(p.id)
        shown = 0
        for p in e.portfolio.positions:
            if shown >= slots:
                break
            shown += 1
            upnl = (p.current - p.entry) * p.qty if p.side.value == "LONG" else (p.entry - p.current) * p.qty
            pct = (upnl / (p.entry * p.qty) * 100) if p.entry and p.qty else 0.0
            r, mfe, mae = self._position_r(p)
            age = (datetime.now(timezone.utc) - p.opened_at).total_seconds()
            side_txt = Text(f"▲ {p.side.value}" if p.side.value == "LONG" else f"▼ {p.side.value}",
                            style="pnl.pos" if p.side.value == "LONG" else "pnl.neg")
            cells = {
                "Mkt": mkt_code(self.engine.markets_cfg, p.market),
                "Symbol": p.symbol,
                "Side": side_txt,
                "Qty": fmt_qty(p.qty),
                "Strategy": self._strategy_label(p.strategy),
                "Entry": fmt_price(self._inr(p.entry, p.market), "INR"),
                "Last": fmt_price(self._inr(p.current, p.market), "INR"),
                "P&L": Text(fmt_pnl(self._inr(upnl, p.market), "INR"), style=pnl_style(upnl)),
                "R": Text(fmt_r(r), style=pnl_style(r if r is not None else 0)),
                "Age": fmt_age(age),
                "State": self._stop_state(p),
                "Gauge": gauge(p.stop, p.current, p.target, 10),
                "Exc": Text(f"{mfe:+.1f}/{mae:+.1f}" if mfe is not None else "—",
                            style="muted"),
            }
            t.add_row(*[cells[c[0]] for c in cols])
        for pid in list(self._excursion):
            if pid not in live_ids:
                self._excursion.pop(pid, None)
                self._initial_risk.pop(pid, None)
        total_inr = e.portfolio.unrealized_inr()
        cells = {
            "Mkt": "", "Symbol": Text("TOTAL", style="accent"), "Side": "", "Qty": "",
            "Strategy": "", "Entry": "", "Last": "",
            "P&L": Text(fmt_pnl(total_inr, "INR"), style=pnl_style(total_inr)),
            "R": "", "Age": "", "State": "", "Gauge": "", "Exc": "",
        }
        t.add_row(*[cells[c[0]] for c in cols], style="muted")
        # full setup label and why for the selected (or first) position
        focus = None
        if self._selected:
            focus = next((p for p in e.portfolio.positions
                          if p.symbol == self._selected[0]), None)
        if focus is None:
            focus = e.portfolio.positions[0]
        body = t
        if focus is not None and not tight and allocated >= need + 1:
            label = self._strategy_label(focus.strategy) or focus.strategy
            why = f"{focus.symbol}: {label}"
            if focus.score_reason:
                why += f" — {focus.score_reason}"
            body = Group(t, Text(truncate(why, max(20, avail - 2)), style="muted"))
        total_pos = len(e.portfolio.positions)
        title = (f"Active Trades · {shown}/{total_pos}" if shown < total_pos
                 else f"Active Trades · {total_pos}")
        return Panel(body, title=title, box=box.ROUNDED, border_style="border")

    def _completed_trades(self, compact=False) -> Panel:
        e = self.engine
        avail = self._panel_width("completed")
        spec = [
            ("Mkt", 3, 3, "left", 9),
            ("Symbol", 8, 16, "left", 10),
            ("Side", 4, 7, "center", 8),
            ("Qty", 4, 7, "right", 4),
            ("Strategy", 6, 18, "left", 5),
            ("Entry", 5, 10, "right", 4),
            ("Exit", 5, 10, "right", 6),
            ("Rsn", 4, 7, "center", 5),
            ("P&L", 7, 10, "right", 11),
            ("R", 4, 6, "right", 10),
            ("Dur", 3, 5, "right", 3),
            ("Open", 5, 5, "right", 2),
            ("Age", 2, 4, "right", 1),
        ]
        cols = self._fit_columns(spec, avail)
        height = self._space()[2] if compact else self._middle_h()[0]
        nt = len(e.portfolio.trades)
        tight = bool(nt) and height < 2 + QUIET_TABLE_CHROME + 2
        chrome = TIGHT_TABLE_CHROME if tight else QUIET_TABLE_CHROME
        room = max(1, height - 2 - chrome)
        show_strip = bool(nt)
        note_parts = []
        all_parts = []
        for m, pnl in sorted(e.portfolio.per_market_pnl().items(),
                             key=lambda kv: -abs(self._inr(kv[1], kv[0]))):
            if abs(pnl) >= 1e-9:
                all_parts.append((m, f"{mkt_code(e.markets_cfg, m)} "
                                      f"{fmt_pnl(self._inr(pnl, m), 'INR')}"))
        limit = 3 if avail >= 70 else 2
        note_parts = [txt for _m, txt in all_parts[:limit]]
        if len(all_parts) > limit:
            note_parts.append(f"+{len(all_parts) - limit}")
        show_note = show_strip and room >= 4 and bool(note_parts)
        max_rows = max(1, room - (1 if show_strip else 0) - (1 if show_note else 0))
        t = Table(box=None if tight else QUIET_BOX, expand=True, pad_edge=False)
        for name, mn, mx, just, _ in cols:
            kw = dict(min_width=mn, max_width=mx, justify=just, no_wrap=True)
            if name in ("Symbol", "Strategy"):
                kw["overflow"] = "ellipsis"
            t.add_column(name, **kw)
        if not e.portfolio.trades:
            body = Text("No completed trades yet · stops, targets and square-offs land here",
                        style="muted")
            return Panel(Align(body, align="left", vertical="middle"),
                         title="Completed Trades", box=box.ROUNDED, border_style="border")
        pages = max(1, (nt + max_rows - 1) // max_rows)
        self._done_page = min(max(0, self._done_page), pages - 1)
        start = self._done_page * max_rows
        for tr in e.portfolio.trades[start:start + max_rows]:
            side_txt = Text(f"▲ {tr.side.value}" if tr.side.value == "LONG" else f"▼ {tr.side.value}",
                            style="pnl.pos" if tr.side.value == "LONG" else "pnl.neg")
            dur = (tr.closed_at - tr.opened_at).total_seconds()
            dur_txt = f"{int(dur // 60)}m" if dur < 3600 else f"{dur / 3600:.1f}h"
            r_txt = f"{tr.r_multiple:+.1f}R"
            r_style = "pnl.pos" if tr.r_multiple >= 0 else "pnl.neg"
            opened = tr.opened_at
            open_txt = opened.strftime("%H:%M")
            age = (datetime.now(timezone.utc) - tr.closed_at).total_seconds()
            cells = {
                "Mkt": mkt_code(self.engine.markets_cfg, tr.market),
                "Symbol": tr.symbol,
                "Side": side_txt,
                "Qty": fmt_qty(tr.qty),
                "Strategy": self._strategy_label(tr.strategy),
                "Entry": fmt_price(self._inr(tr.entry, tr.market), "INR"),
                "Exit": fmt_price(self._inr(tr.exit, tr.market), "INR"),
                "Rsn": exit_chip(tr.exit_reason),
                "P&L": Text(fmt_pnl(self._inr(tr.pnl, tr.market), "INR"), style=pnl_style(tr.pnl)),
                "R": Text(r_txt, style=r_style),
                "Dur": dur_txt,
                "Open": open_txt,
                "Age": fmt_age(age),
            }
            t.add_row(*[cells[c[0]] for c in cols])
        st = e.portfolio.stats()
        trades = st["trades"]
        pf_txt = fmt_pf(st["profit_factor"] if trades else None)
        charges = sum(self._inr(t.charges, t.market) for t in e.portfolio.trades)
        gross = st["total_pnl"] + charges
        strip = Text()
        strip.append("N ", style="muted")
        strip.append(f"{trades}", style="text")
        if avail < 110:
            strip.append(" · W ", style="muted")
            strip.append(fmt_winrate(st["win_rate"], trades), style="text")
            strip.append(" · net ", style="muted")
            strip.append(fmt_pnl(st["total_pnl"], "INR"), style=pnl_style(st["total_pnl"]))
            strip.append(" · PF ", style="muted")
            strip.append(pf_txt, style="text")
            strip.append(" · exp ", style="muted")
            strip.append(fmt_r(st["expectancy_r"]) if trades else "—", style="text")
        else:
            strip.append(" · win ", style="muted")
            strip.append(fmt_winrate(st["win_rate"], trades), style="text")
            strip.append(" · gross ", style="muted")
            strip.append(fmt_pnl(gross, "INR"), style=pnl_style(gross))
            strip.append(" · fees ", style="muted")
            strip.append(fmt_money(charges, "INR"), style="muted")
            strip.append(" · net ", style="muted")
            strip.append(fmt_pnl(st["total_pnl"], "INR"), style=pnl_style(st["total_pnl"]))
            strip.append(" · PF ", style="muted")
            strip.append(pf_txt, style="text")
            strip.append(" · exp ", style="muted")
            strip.append(fmt_r(st["expectancy_r"]) if trades else "—", style="text")
            strip.append(" · avg W/L ", style="muted")
            strip.append(f"{fmt_money(st['avg_win'], 'INR')}/{fmt_money(st['avg_loss'], 'INR')}",
                         style="text")
            strip.append("  (INR)", style="dim")
        title = f"Completed Trades · {nt} · page {self._done_page + 1}/{pages}"
        if not show_strip and nt == 0:
            return Panel(t, title=title, box=box.ROUNDED,
                         border_style="border")
        body = Group(t, strip)
        if show_note:
            body = Group(t, strip, Text("  ".join(note_parts) +
                                        "   (per market, in INR)", style="dim"))
        return Panel(body, title=title, box=box.ROUNDED, border_style="border")

    def _risk_rows(self):
        """Per-market open exposure, risk to stop and reward to target.

        Everything is converted to INR at the same cross rate the rest of
        the dashboard uses. The distances are measured from the live price,
        so the numbers shrink as a trade approaches its stop or target.
        """
        return self._rc("risk_rows", self._risk_rows_uncached)

    def _risk_rows_uncached(self):
        e = self.engine
        out = {}
        for p in e.portfolio.positions:
            fx = e.portfolio.fx_rate(p.market)
            qty = p.qty or 0.0
            cur = p.current or p.entry
            if p.side.value == "LONG":
                risk = cur - p.stop
                reward = p.target - cur
            else:
                risk = p.stop - cur
                reward = cur - p.target
            agg = out.setdefault(p.market, {"exposure": 0.0, "risk": 0.0, "reward": 0.0})
            agg["exposure"] += cur * qty * fx
            agg["risk"] += max(0.0, risk) * qty * fx
            agg["reward"] += max(0.0, reward) * qty * fx
        return out

    def _risk_panel(self) -> Panel:
        e = self.engine
        rows = self._risk_rows()
        if not rows:
            body = Text("No open positions · nothing at risk", style="muted")
            return Panel(Align(body, align="left", vertical="middle"),
                         title="Risk", box=box.ROUNDED, border_style="border")
        eq = e.portfolio.total_equity_inr()
        exposure = sum(a["exposure"] for a in rows.values())
        risk = sum(a["risk"] for a in rows.values())
        reward = sum(a["reward"] for a in rows.values())
        day = e.portfolio.day_pnl_inr()
        max_loss_pct = float(e.risk.risk.get("max_daily_loss_pct", 0) or 0)
        limit = eq * max_loss_pct / 100.0
        used = max(0.0, -day)
        frac = (used / limit) if limit > 0 else 0.0
        sym_exp = {}
        for p in e.portfolio.positions:
            sym_exp[p.symbol] = sym_exp.get(p.symbol, 0.0) + self._inr(p.current * p.qty, p.market)
        top_sym, top_val = max(sym_exp.items(), key=lambda kv: kv[1]) if sym_exp else ("—", 0.0)
        conc = (top_val / exposure * 100.0) if exposure else 0.0
        height = self._middle_h()[1]
        summary = Text(no_wrap=True, overflow="ellipsis")
        summary.append("Day loss ", style="muted")
        summary.append(f"{used / eq * 100.0:.2f}%" if eq else "—", style=pnl_style(-used))
        summary.append(f" / {max_loss_pct:.1f}% ", style="muted")
        summary.append(_block_gauge(frac, 8), style="pnl.neg" if frac >= 0.8 else "status.open")
        summary.append(" · at risk ", style="muted")
        summary.append(f"{risk / eq * 100.0:.2f}%" if eq else "—", style="text")
        summary.append(f" · top {top_sym} ", style="muted")
        summary.append(f"{conc:.0f}%", style="text")
        summary.truncate(self._panel_width("completed"))
        spec = [
            ("Mkt", 5, 5, "left", 9),
            ("Exposure", 7, 12, "right", 6),
            ("Risk", 6, 11, "right", 7),
            ("Reward", 6, 11, "right", 5),
            ("%Eq", 4, 5, "right", 3),
            ("R:R", 4, 5, "right", 4),
        ]
        cols = self._fit_columns(spec, self._panel_width("completed"))
        n = len(rows)
        # room for a full frame plus the summary line?
        show_summary = height >= 2 + 1 + 1 + QUIET_TABLE_CHROME + n + 1
        quiet = height >= 2 + QUIET_TABLE_CHROME + n + 1
        if quiet:
            slots = max(1, height - 2 - QUIET_TABLE_CHROME - 1 - (1 if show_summary else 0))
            show_header = True
        else:
            room = max(1, height - 2)
            show_header = room >= 4
            slots = max(1, room - (1 if show_header else 0) - 1)
        t = Table(box=QUIET_BOX if quiet else None, show_header=show_header,
                  expand=True, pad_edge=False)
        for name, mn, mx, just, _ in cols:
            t.add_column(name, min_width=mn, max_width=mx, justify=just, no_wrap=True)
        ordered = sorted(rows.items(), key=lambda kv: (-kv[1]["risk"], -kv[1]["exposure"]))
        for m, agg in ordered[:slots]:
            rr = agg["reward"] / agg["risk"] if agg["risk"] > 0 else None
            cells = {
                "Mkt": mkt_code(e.markets_cfg, m),
                "Exposure": fmt_money(agg["exposure"], "INR"),
                "Risk": fmt_money(agg["risk"], "INR"),
                "Reward": fmt_money(agg["reward"], "INR"),
                "%Eq": f"{agg['risk'] / eq * 100.0:.1f}" if eq else "—",
                "R:R": f"{rr:.1f}" if rr is not None else "—",
            }
            t.add_row(*[cells[c[0]] for c in cols])
        rr = reward / risk if risk > 0 else None
        cells = {
            "Mkt": Text("TOTAL", style="accent"),
            "Exposure": fmt_money(exposure, "INR"),
            "Risk": fmt_money(risk, "INR"),
            "Reward": fmt_money(reward, "INR"),
            "%Eq": f"{risk / eq * 100.0:.1f}" if eq else "—",
            "R:R": f"{rr:.1f}" if rr is not None else "—",
        }
        t.add_row(*[cells[c[0]] for c in cols], style="muted")
        body = Group(summary, t) if show_summary else t
        shown = min(slots, n)
        title = f"Risk · top {shown}/{n}" if shown < n else f"Risk · {n}"
        return Panel(body, title=title, box=box.ROUNDED, border_style="border")

    def _visible_logs(self):
        out = []
        for msg in self.engine.logs:
            _tag, _style = level_tag(msg)
            if self._log_level == "ERROR" and _tag.strip() != "ERROR":
                continue
            if self._log_level == "WARN" and _tag.strip() not in ("WARN", "ERROR"):
                continue
            out.append(msg)
        return out

    def _log_panel(self, full=False) -> Panel:
        e = self.engine
        logs = self._visible_logs()
        lines = Text(overflow="fold" if self._log_wrap else "ellipsis",
                     no_wrap=not self._log_wrap)
        if full:
            h = self.console.height or 24
            body_w = max(12, (self.console.width or 120) - 18)
        else:
            h = self._space()[2]
            body_w = max(12, self._panel_width("log") - 16)
        slots = max(6, h - 2)
        shown = logs[:slots]
        # keep the newest visible when wrapping needs more lines than fit
        for msg in shown:
            tag, style = level_tag(msg)
            if msg.startswith("["):
                try:
                    ts_end = msg.index("]")
                    ts = msg[1:ts_end]
                    body = msg[ts_end + 2:]
                except ValueError:
                    ts = ""
                    body = msg
            else:
                ts = ""
                body = msg
            body = strip_level(body)
            if not self._log_wrap:
                body = truncate(body, body_w)
            lines.append(f"{ts} ", style="dim")
            lines.append(f"{tag:<6} ", style=style)
            lines.append(body + "\n", style="text")
        if not logs:
            filt = f" at {self._log_level}" if self._log_level else ""
            lines.append(f"No events{filt} yet", style="muted")
        title = "Log"
        if self._log_level:
            title += f" · {self._log_level}+"
        if self._log_wrap:
            title += " · wrap"
        title += f" · {len(logs)}"
        return Panel(lines, title=title, box=box.ROUNDED, border_style="border")

    def _content_height(self, rows, header_rows=4, min_h=4, max_h=40):
        """Height a table panel needs: borders + rules + header + data rows."""
        return max(min_h, min(max_h, header_rows + rows))

    def _middle_ratio(self):
        """How the wide layout's middle column splits: Completed over Risk."""
        if self._space()[2] < 8:
            # too short to give both panels a body; keep the history only
            return (1, 0)
        return (3, 2) if self.engine.portfolio.trades else (1, 1)

    def _middle_h(self):
        """Predicted heights of (completed, risk) inside the middle column."""
        return self._rc("middle_h", self._middle_h_uncached)

    def _middle_h_uncached(self):
        bottom_h = self._space()[2]
        d, r = self._middle_ratio()
        if not r:
            return bottom_h, 0
        e = self.engine
        nt = len(e.portfolio.trades)
        nm = len(self._risk_rows())
        done_want = (2 + QUIET_TABLE_CHROME + min(16, nt) + 1) if nt else 4
        risk_want = (2 + 1 + QUIET_TABLE_CHROME + nm + 1) if nm else 4
        if done_want + risk_want <= bottom_h:
            extra = bottom_h - done_want - risk_want
            done = done_want + extra * d // (d + r)
        else:
            done = bottom_h * d // (d + r)
        risk = bottom_h - done
        if risk < 4:
            risk = 4
            done = max(0, bottom_h - risk)
        return done, risk

    def _active_h(self):
        """Natural height of the standalone (narrow) active-trades panel."""
        n = len(self.engine.portfolio.positions)
        if not n:
            return 4
        need = 2 + QUIET_TABLE_CHROME + n + 1 + 1
        return max(5, min(12, need))

    def _bottom_h(self):
        """Natural height of the wide bottom row (active | history/risk | log)."""
        e = self.engine
        n = len(e.portfolio.positions)
        active = (2 + QUIET_TABLE_CHROME + n + 1 + 1) if n else 4
        nt = len(e.portfolio.trades)
        note = 1 if any(abs(v) > 1e-9 for v in e.portfolio.per_market_pnl().values()) else 0
        done = (2 + QUIET_TABLE_CHROME + min(16, nt) + 1 + note) if nt else 4
        nm = len(self._risk_rows())
        risk = (2 + 1 + QUIET_TABLE_CHROME + nm + 1) if nm else 4
        nl = min(14, len(e.logs)) if e.logs else 1
        log = 2 + max(6, nl)
        return max(active, done + risk, log)

    def _space(self):
        """Height allocation that always fits the terminal.

        Returns (mk_h, scanner_h, bottom_h, active_h). The header and footer
        are fixed; content-driven wants are capped at roughly 45% for the
        scanner and 40% for the bottom row, and the markets keep their cards
        while the budget allows. Narrow terminals give active trades their
        own band; wide ones put active inside the bottom row.
        """
        key = ("space", self.console.width, self.console.height, self._density)
        return self._rc(key, self._space_uncached)

    def _space_uncached(self):
        wide = self._wide()
        h = self.console.height or 40
        budget = max(4, h - self._header_h() - 1)

        e = self.engine
        nrows = sum(len(v) for v in e.last_scan.values())
        if not nrows:
            nrows = 1
        scanner_want = 2 + QUIET_TABLE_CHROME + min(18, max(4, nrows))
        mk_full = self._markets_height()
        mk_want = min(mk_full, self._markets_compact_height())
        bottom_want = self._bottom_h()

        if wide:
            active_want = 0
            active_min = 0
            mk_min = min(mk_want, 5)
            bottom_min = 5
        else:
            active_want = self._active_h()
            active_min = min(active_want, 5)
            mk_min = min(mk_want, 4)
            bottom_min = 4
        scanner_min = 6
        # floors degrade rather than overflow on a very short terminal
        while active_min + bottom_min + mk_min + scanner_min > budget:
            if mk_min:
                mk_min -= 1
            elif bottom_min:
                bottom_min -= 1
            elif scanner_min > 4:
                scanner_min -= 1
            else:
                break
        if active_min + bottom_min + scanner_min > budget:
            active_min = 0

        scan_cap = max(scanner_min, int(budget * 0.45))
        bottom_cap = max(bottom_min, int(budget * 0.40))

        active = active_want
        bottom = min(bottom_want, max(bottom_min, int(budget * 0.40)))
        scanner = min(scanner_want, max(scanner_min, int(budget * 0.45)))
        mk = mk_want
        # shrink (active first, then the history, then markets) until it fits
        while active + bottom + scanner + mk > budget:
            if active > active_min:
                active -= 1
            elif bottom > bottom_min:
                bottom -= 1
            elif mk > 0:
                mk -= 1
            elif scanner > 3:
                scanner -= 1
            else:
                break
        # spend the leftover where it buys information
        left = budget - (active + bottom + scanner + mk)
        if left > 0:
            grow = min(left, bottom_cap - bottom)
            bottom += max(0, grow)
            left -= max(0, grow)
        if left > 0:
            grow = min(left, scan_cap - scanner)
            scanner += max(0, grow)
            left -= max(0, grow)
        if left > 0:
            grow = min(left, mk_want - mk)
            mk += max(0, grow)
            left -= max(0, grow)
        if left > 0:
            # no panel is short of content now; rows are still the most
            # useful use of a leftover line
            scanner += left
        return mk, scanner, bottom, active

    def render(self) -> Layout:
        """Build one frame; helper results are memoised for this pass only."""
        self._render_cache = {}
        try:
            return self._render_frame()
        finally:
            self._render_cache = None

    def _render_frame(self) -> Layout:
        layout = Layout()
        footer = self._footer()
        wide = self._wide()
        header = Layout(self._header(), size=self._header_h())

        # modal overlays (help / detail) replace the main content but keep
        # the header and the command bar visible
        overlay = self._overlay()
        if overlay is not None:
            layout.split_column(header, Layout(overlay), Layout(footer, size=1))
            return layout

        view = self._view or "monitor"
        if view in ("positions", "analytics", "diagnostics", "report", "log"):
            layout.split_column(header, Layout(self._view_panel(view)),
                                Layout(footer, size=1))
            return layout

        mk_h, scanner_h, bottom_h, active_h = self._space()

        if wide:
            bottom = Layout()
            middle = Layout(ratio=32)
            d, r = self._middle_ratio()
            done_h, risk_h = self._middle_h()
            if r:
                middle.split_column(
                    Layout(self._completed_trades(), size=done_h),
                    Layout(self._risk_panel(), size=risk_h),
                )
            else:
                middle.split_column(Layout(self._completed_trades()))
            bottom.split_row(
                Layout(self._active_trades(compact=False), ratio=44),
                middle,
                Layout(self._log_panel(), ratio=24),
            )
            layout.split_column(
                header,
                Layout(self._markets_panel(mk_h), size=mk_h),
                Layout(self._scanner_panel(), size=scanner_h),
                Layout(bottom, size=bottom_h),
                Layout(footer, size=1),
            )
        else:
            bottom = Layout()
            bottom.split_row(
                Layout(self._completed_trades(compact=True), ratio=3),
                Layout(self._log_panel(), ratio=2),
            )
            layout.split_column(
                header,
                Layout(self._markets_panel(mk_h), size=mk_h),
                Layout(self._scanner_panel(), size=scanner_h),
                Layout(self._active_trades(compact=True), size=active_h),
                Layout(bottom, size=bottom_h),
                Layout(footer, size=1),
            )
        return layout

    def _footer(self):
        e = self.engine
        txt = Text()
        if self._input_mode:
            txt.append(":" + (self._input_buf or ""), style="accent")
            txt.append("▌", style="accent")
            txt.append("   enter run · esc cancel · type help", style="muted")
            return txt
        # key / label hierarchy: pressed keys are bright, their names quiet,
        # separators invisible - the accent stays reserved for live states
        keys = [("q", "Quit"), ("x", "Flat"), ("p", "Pause"), ("Tab", "Mkt"),
                ("←→", "Pg"), ("↑↓", "Sel"), ("d", "Detail"), ("f", "Scan"),
                ("s", "Sort"), ("g", "Sigs"), ("m/i/a/y/l/r", "View"),
                ("e/w", "Log"), (":", "Cmd"), ("?", "Help")]
        for k, label in keys:
            txt.append(k, style="bold")
            txt.append(f" {label}", style="dim")
            txt.append(" ", style="dim")
        # armed close: a second press inside 6s actually closes the position
        if self._pending_close is not None:
            positions = e.portfolio.positions
            if 0 <= self._pending_close < len(positions):
                age = time.time() - self._pending_close_at
                if age > 6 or self._pending_close >= len(positions):
                    self._pending_close = None
                else:
                    p = positions[self._pending_close]
                    left = max(1, int(6 - age))
                    txt.append(f" · press {self._pending_close + 1} again to CLOSE "
                               f"{p.market} {p.symbol} ({left}s)", style="status.preclose")
            else:
                self._pending_close = None
        if self._selected:
            txt.append(f" · sel {self._selected[0]}", style="muted")
        txt.append(f"  scan {e.last_scan_elapsed:.1f}s", style="muted")
        if e.scan_progress:
            txt.append(f" · {self._spinner()} scanning {e.scan_progress}", style="accent")
        elif not e.paused:
            nxt = getattr(e, "next_scan_at", 0)
            if nxt:
                left = max(0, int(nxt - time.time()))
                txt.append(f" · next scan in {left}s", style="muted")
        # data age
        ages = []
        for m, rows in e.last_scan.items():
            for r in rows:
                if r.get("updated") is not None:
                    try:
                        ages.append((datetime.now(timezone.utc) - r["updated"]).total_seconds())
                    except Exception:
                        pass
        if ages:
            txt.append(f" · data age {int(min(ages))}s", style="muted")
        # live quote status
        if hasattr(e, "ticker"):
            q_age = e.ticker.age_of_newest()
            if e.live_quotes:
                if q_age is not None:
                    txt.append(f" · quotes {int(q_age)}s", style="status.open")
                else:
                    txt.append(" · quotes starting…", style="muted")
                # how far through the live markets' symbol queue we are
                try:
                    total = len(e._fast_queue)
                    done = min(e._fast_cursor, total)
                    if total:
                        txt.append(f" · scan {done}/{total} ({e.fast_scan_seconds:.0f}s)",
                                   style="muted")
                except Exception:
                    pass
            else:
                txt.append(" · quotes off", style="dim")
        if e.paused:
            txt.append(" · PAUSED", style="status.preclose")
        txt.truncate(max(20, (self.console.width or 120) - 1))
        return txt

    def _hotkey_loop(self):
        last_key = {}
        while not self._stop:
            try:
                # msvcrt's buffer belongs to this console, so anything in it
                # was typed here; drain it without consulting window focus
                # (focus detection is unreliable under ConPTY terminals).
                for _ in range(16):
                    if not msvcrt.kbhit():
                        break
                    self._handle_key(msvcrt.getch(), last_key)
            except Exception as e:
                # a dead hotkey loop looks exactly like "keys do nothing"
                # - surface the failure in the log panel instead
                try:
                    self.engine.log(f"hotkey error: {e}", "error")
                except Exception:
                    pass
            # expire an armed close nobody confirmed
            if self._pending_close is not None and time.time() - self._pending_close_at > 6:
                self._pending_close = None
            time.sleep(0.1)

    def _handle_key(self, ch: bytes, last_key: dict):
        """Dispatch one physical key press."""
        now = time.time()
        # ---------------- command bar ----------------
        if self._input_mode:
            try:
                raw = ch.decode()
            except Exception:
                raw = ""
            if ch in (b"\r", b"\n"):
                cmd = self._input_buf.strip()
                self._input_mode = False
                self._input_buf = ""
                if cmd:
                    self._run_command(cmd)
                return
            elif ch == b"\x1b":
                self._input_mode = False
                self._input_buf = ""
                return
            elif ch in (b"\x08", b"\x7f"):
                self._input_buf = self._input_buf[:-1]
                return
            elif raw and 0x20 <= ord(raw[0]) < 0x7f:
                self._input_buf += raw.lower()
                return
            else:
                return
        # debounced char class for the normal (hotkey) mode
        try:
            c = ch.decode().lower()
        except Exception:
            c = ""
        # debounce key repeat (Windows sends repeats while held)
        if c and now - last_key.get(c, 0) < 0.35:
            time.sleep(0.1)
            return
        last_key[c] = now
        # ---------------- quit always works ----------------
        if c == "q":
            self.engine.running = False
            self._stop = True
            return
        # ---------------- full-screen views ----------------
        if self._view in ("positions", "analytics", "diagnostics", "report", "log"):
            if ch == b"\x1b":
                self._view = None
            elif c in ("m", "i", "a", "y", "r", "l"):
                self._switch_view(c)
            elif self._view == "log" and c == "e":
                self._cycle_log_level()
            elif self._view == "log" and c == "w":
                self._log_wrap = not self._log_wrap
            return
        # ---------------- overlays ----------------
        if self._view is not None:
            if ch == b"\x1b" or c in ("d", "h", "?", " "):
                self._view = None
            return
        # ---------------- normal hotkeys ----------------
        if c in ("m", "i", "a", "y", "l"):
            self._switch_view(c)
        elif c == "n":
            self._done_page += 1
        elif c == "b":
            self._done_page = max(0, self._done_page - 1)
        elif c == "e":
            self._cycle_log_level()
        elif c == "w":
            self._log_wrap = not self._log_wrap
        if ch == b"\x1b":  # esc: clear selection
            self._selected = None
        elif c == "x":
            self.engine.square_off_all()
        elif c == "p":
            self.engine.paused = not self.engine.paused
            self.engine.risk.paused = self.engine.paused
            self.engine.log("scan " + ("paused" if self.engine.paused else "resumed"))
        elif c == "\t":
            self._cycle_market(+1)
        elif ch == b"\x08":  # backspace: cycle markets backwards
            self._cycle_market(-1)
        elif c == "r":
            self._view = "report"
        elif c == "v":
            self.engine.live_quotes = not self.engine.live_quotes
            self.engine.log("live quotes " + ("on" if self.engine.live_quotes else "off"))
        elif c == "c":
            self._set_crypto("Crypto" in self.engine.disabled_markets)
        elif c == "f":
            self.engine.scan_now.set()
            self.engine.log("scan forced")
        elif c == "s":
            self._cycle_sort()
        elif c == "g":
            self._cycle_signal()
        elif c in (":", "/"):
            self._input_mode = True
            self._input_buf = ""
        elif c in ("h", "?"):
            self._view = "help"
        elif c == "d" or ch == b"\r" or ch == b"\n":
            row, _key = self._get_selected_row()
            if row is not None:
                self._view = "detail"
        elif c.isdigit() and c != "0":
            self._quick_close(int(c))
        elif ch in (b"\xe0", b"\x00"):
            # extended key prefix: only read the code when it is already
            # buffered, or a lone prefix would block the hotkey thread
            if not msvcrt.kbhit():
                return
            code = msvcrt.getch()
            if code == b"K":
                self._page = max(0, self._page - 1)
            elif code == b"M":
                self._page = self._page + 1
            elif code == b"H":  # up
                self._move_selection(-1)
            elif code == b"P":  # down
                self._move_selection(+1)

    # ---------------------------------------------------------------- helpers

    def _switch_view(self, key: str):
        mapping = {"m": None, "i": "positions", "a": "analytics",
                   "y": "diagnostics", "r": "report", "l": "log"}
        self._view = mapping.get(key)

    def _cycle_log_level(self):
        order = (None, "WARN", "ERROR")
        self._log_level = order[(order.index(self._log_level) + 1) % len(order)]
        self.engine.log("log filter "
                        + ("all" if self._log_level is None else self._log_level))

    def _cycle_market(self, step: int):
        """Cycle the market filter; markets in step order, wrapping to all."""
        if not self.markets:
            return
        if self.market_filter is None:
            self.market_filter = self.markets[0] if step > 0 else self.markets[-1]
        else:
            try:
                idx = self.markets.index(self.market_filter)
            except ValueError:
                self.market_filter = None
                return
            new = idx + step
            if new < 0 or new >= len(self.markets):
                self.market_filter = None
            else:
                self.market_filter = self.markets[new]
            self._page = 0

    def _cycle_sort(self):
        mapping = ("score", "chg", "signal", "symbol")
        self._sort_mode = mapping[(mapping.index(self._sort_mode) + 1) % len(mapping)]
        self.engine.log(f"scanner sort {self._sort_mode}")

    def _cycle_signal(self):
        order = (None, "LONG", "SHORT")
        self._signal_filter = order[(order.index(self._signal_filter) + 1) % len(order)]
        self._page = 0
        self.engine.log("scanner signals "
                        + ("all" if self._signal_filter is None else self._signal_filter.lower()))

    def _move_selection(self, delta: int):
        rows, _ = self._scanner_rows()
        if not rows:
            self._selected = None
            return
        idx, _ = self._find_selected(rows)
        if idx is None:
            idx = 0
        idx = max(0, min(len(rows) - 1, idx + delta))
        self._selected = (rows[idx].get("symbol"), rows[idx].get("market"))
        # keep the selection visible on the current page
        page_size = self._page_size()
        self._page = idx // page_size

    def _quick_close(self, n: int):
        """Number key 1..9: first press arms, second press within 6s closes."""
        positions = self.engine.portfolio.positions
        idx = n - 1
        if idx >= len(positions):
            self.engine.log(f"no position #{n}")
            return
        if self._pending_close == idx and time.time() - self._pending_close_at <= 6:
            pos = positions[idx]
            self._pending_close = None
            self.engine.close_one(pos)
        else:
            self._pending_close = idx
            self._pending_close_at = time.time()

    def _set_crypto(self, on: bool):
        e = self.engine
        if on:
            e.disabled_markets.discard("Crypto")
            if "Crypto" not in self.markets and "Crypto" in e.markets_cfg:
                self.markets.append("Crypto")
            e.log("crypto enabled")
        else:
            e.disabled_markets.add("Crypto")
            self.markets = [m for m in self.markets if m != "Crypto"]
            try:
                from trader.core.models import ExitReason
                e._close_market_positions("Crypto", ExitReason.SQUAREOFF, note="disabled")
            except Exception:
                pass
            e.log("crypto disabled")

    def _run_command(self, cmd: str):
        """Run one command from the `:` command bar."""
        e = self.engine
        low = cmd.strip().lower()
        if not low:
            return
        parts = low.split()
        head, rest = parts[0], "".join(parts[1:])
        if head == "help":
            self._view = "help"
        elif head == "filter":
            arg = cmd[6:].strip() if len(cmd) > 6 else ""
            if arg in ("", "off", "clear"):
                self._text_filter = ""
                self.engine.log("scanner filter cleared")
            else:
                self._text_filter = arg
                self._page = 0
                self.engine.log(f"scanner filter '{arg}'")
        elif head == "sort":
            if rest in ("score", "chg", "signal", "symbol"):
                self._sort_mode = rest
                self.engine.log(f"scanner sort {rest}")
            else:
                self.engine.log("usage: :sort score|chg|signal|symbol")
        elif head == "longs":
            self._signal_filter = "LONG"
            self._page = 0
        elif head == "shorts":
            self._signal_filter = "SHORT"
            self._page = 0
        elif head == "all":
            self._signal_filter = None
            self._page = 0
        elif head == "close":
            self._close_by_arg(cmd[5:].strip())
        elif head == "flat":
            e.square_off_all()
        elif head in ("pause", "resume"):
            want = head == "pause"
            if e.paused != want:
                e.paused = want
                e.risk.paused = want
                e.log("scan " + ("paused" if want else "resumed"))
        elif head == "crypto":
            on = rest in ("on", "1", "true")
            off = rest in ("off", "0", "false")
            if on or off:
                self._set_crypto(on)
            else:
                self.engine.log("usage: :crypto on|off")
        elif head == "kill":
            e.risk.trigger_kill()
            e.log("kill switch armed by command", "warn")
        elif head == "scan":
            e.scan_now.set()
            self.engine.log("scan forced")
        elif head == "report":
            self._view = "report"
        elif head == "view":
            if rest in ("monitor", ""):
                self._view = None
            elif rest in VIEWS:
                self._view = rest
            else:
                self.engine.log("usage: :view monitor|positions|analytics|diagnostics|report|log|help|detail")
        elif head == "log":
            if rest in ("all", ""):
                self._log_level = None
            elif rest in ("warn", "warning"):
                self._log_level = "WARN"
            elif rest in ("error", "err"):
                self._log_level = "ERROR"
            elif rest == "wrap":
                self._log_wrap = not self._log_wrap
            else:
                self.engine.log("usage: :log all|warn|error|wrap")
        elif head == "density":
            if rest in ("auto", "compact", "normal", "wide"):
                self._density = rest
                self.engine.log(f"density {rest}")
            else:
                self.engine.log("usage: :density auto|compact|normal|wide")
        else:
            # a bare word filters the scanner on it
            self._text_filter = low
            self._page = 0
            self.engine.log(f"scanner filter '{low}' (type :help for commands)")

    def _close_by_arg(self, arg: str):
        e = self.engine
        positions = e.portfolio.positions
        if not positions:
            self.engine.log("nothing to close")
            return
        pos = None
        if arg.isdigit():
            i = int(arg) - 1
            if 0 <= i < len(positions):
                pos = positions[i]
        else:
            for p in positions:
                if p.symbol.lower() == arg or arg in p.symbol.lower():
                    pos = p
                    break
        if pos is None:
            self.engine.log(f"no position matches '{arg}'")
        else:
            e.close_one(pos)

    def run(self):
        def scan_loop():
            interval = max(1, int(self.engine.cfg.get("scan_interval_seconds", 60)))
            while self.engine.running:
                cycle_started = time.time()
                self.engine.next_scan_at = cycle_started + interval
                if not self.engine.paused:
                    try:
                        self.engine.scan_once()
                    except Exception as e:
                        self.engine.log(f"scan loop error: {e}", "error")
                # keep a steady cadence: sleep whatever is left of the
                # interval, so the cycle is exactly `scan_interval_seconds`
                # and not scan-time + interval — unless a scan was forced
                remaining = interval - (time.time() - cycle_started)
                while remaining > 0 and self.engine.running and not self.engine.scan_now.is_set():
                    time.sleep(min(0.25, remaining))
                    remaining = interval - (time.time() - cycle_started)
                self.engine.scan_now.clear()
                self.engine.next_scan_at = time.time() + interval

        scan_thread = threading.Thread(target=scan_loop, daemon=True)
        scan_thread.start()
        hk_thread = threading.Thread(target=self._hotkey_loop, daemon=True)
        hk_thread.start()
        # the quote thread keeps prices and exits moving between scans
        try:
            self.engine.start_price_thread()
        except Exception as e:
            self.engine.log(f"live quotes unavailable: {e}", "warn")
        try:
            with Live(self.render(), console=self.console, refresh_per_second=4, screen=True) as live:
                while self.engine.running:
                    live.update(self.render())
                    time.sleep(0.25)
        finally:
            self._stop = True
            self.engine.stop_price_thread()
            self.engine.shutdown_report()
