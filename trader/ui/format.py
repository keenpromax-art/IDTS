from rich.text import Text
from rich import box

# Quiet table frame: a rule at the top, a rule under the header, nothing else.
# Line count for a table with N data rows: 4 + N  (top rule, head, head rule, bottom)
QUIET_BOX = box.Box(
    " ── \n"   # top
    "    \n"   # head
    " ── \n"   # head_row
    "    \n"   # mid
    "    \n"   # row
    "    \n"   # foot_row
    "    \n"   # foot
    "    \n"   # bottom
)

QUIET_TABLE_CHROME = 4  # lines a quiet table uses before any data row

# Tight tables render with box=None: header line + data rows only, no
# rules at all (Rich renders nothing for the rules). 1 + N lines total.
TIGHT_TABLE_CHROME = 1  # lines a tight table uses before any data row

CCY_SYMBOL = {"INR": "₹", "USD": "$", "JPY": "¥", "GBP": "£",
              "HKD": "HK$", "AUD": "A$", "KRW": "₩", "BRL": "R$",
              "USDT": "US$", "EUR": "€"}
# currencies quoted without decimals
CCY_NO_DECIMAL = {"JPY", "KRW"}


def _indian_group(n):
    neg = n < 0
    s = str(int(round(abs(n))))
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ",".join(groups + [last3])
    return ("-" if neg else "") + grouped


def fmt_money(value, ccy="INR", decimals=0):
    sym = CCY_SYMBOL.get(ccy, ccy + " ")
    if ccy == "INR" and not decimals:
        num = _indian_group(value)
    else:
        num = f"{value:,.{decimals}f}"
    neg = num.startswith("-")
    if neg:
        num = num[1:]
        return f"-{sym}{num}"
    return f"{sym}{num}"


def fmt_price(value, ccy="INR"):
    """A traded price: always in the market's convention, never rounded away.

    Quote prices keep their decimals (2 for most currencies, whole numbers
    for the ones that trade that way); very large notionals drop the cents
    instead of overflowing or being clipped mid-number, and the fill log
    still carries the exact fill.
    """
    neg = value < 0
    sym = CCY_SYMBOL.get(ccy, ccy + " ")
    if ccy in CCY_NO_DECIMAL:
        d = 0
    elif abs(value) >= 10000:
        d = 0
    else:
        d = 2
    if ccy == "INR":
        s = f"{abs(value):.{d}f}"
        ip, _, dp = s.partition(".")
        s = _indian_group(int(ip)) + (f".{dp}" if dp else "")
    else:
        s = f"{abs(value):,.{d}f}"
    return ("-" if neg and abs(value) >= 0.005 else "") + sym + s


def fmt_qty(q):
    """Quantity without float noise: 10 -> 10, 0.125 -> 0.125."""
    return f"{q:g}"


def pct_arrow(chg):
    """A consistent ▲/▼ change cell for the scanner, detail and watchlists."""
    if chg is None:
        return Text("—", style="muted")
    arrow = "▲" if chg >= 0 else "▼"
    style = "pnl.pos" if chg >= 0 else "pnl.neg"
    return Text(f"{arrow}{abs(chg):.2f}%", style=style)


def fmt_pnl(value, ccy="INR"):
    if value == 0:
        return fmt_money(0, ccy)
    sign = "+" if value > 0 else ""
    return f"{sign}{fmt_money(value, ccy)}"


def pnl_style(value):
    if value > 0:
        return "pnl.pos"
    if value < 0:
        return "pnl.neg"
    return "pnl.zero"


def fmt_pct(value, trades=0):
    if trades == 0:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def fmt_winrate(value, trades=0):
    if trades == 0:
        return "—"
    return f"{value:.1f}%"


def fmt_pct_always(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def fmt_score(value):
    if value is None:
        return "—"
    return f"{value:.2f}"


def score_style(value):
    if value is None:
        return "score.low"
    if value >= 0.7:
        return "score.high"
    if value >= 0.5:
        return "score.mid"
    return "score.low"


def score_bar(value, width=5):
    if value is None:
        return "▱" * width
    filled = int(round(value * width))
    filled = max(0, min(width, filled))
    return "▰" * filled + "▱" * (width - filled)


def status_style(status):
    if status == "OPEN":
        return "status.open"
    if status in ("PRE-CLOSE", "PRE-OPEN", "LUNCH"):
        return "status.preclose"
    return "status.closed"


def status_glyph(status):
    return "●"


def sparkline(values, width=12):
    if not values:
        return "▁" * width
    vals = values[-width:]
    lo = min(vals)
    hi = max(vals)
    blocks = "▁▂▃▄▅▆▇█"
    if hi == lo:
        return blocks[0] * len(vals)
    out = ""
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out += blocks[idx]
    return out.rjust(width, blocks[0])


def gauge(sl, price, tp, width=8):
    if sl is None or tp is None or tp == sl:
        return "─" * width
    ratio = (price - sl) / (tp - sl)
    ratio = max(0.0, min(1.0, ratio))
    pos = int(round(ratio * (width - 1)))
    chars = ["─"] * width
    chars[pos] = "●"
    left = "".join(chars[:pos])
    right = "".join(chars[pos + 1:])
    mid = chars[pos]
    return f"{left}{mid}{right}"


def truncate(text, n=28):
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def exit_chip(reason):
    r = str(reason).lower()
    if "target" in r:
        return Text("TARGET", style="chip.target")
    if "sl" in r or "stop" in r:
        return Text("SL", style="chip.sl")
    if "trail" in r:
        return Text("TRAIL", style="chip.trail")
    if "square" in r:
        return Text("SQ-OFF", style="chip.sqoff")
    if "kill" in r:
        return Text("KILL", style="chip.kill")
    return Text(str(reason).upper(), style="muted")


def level_tag(msg):
    """Derive (tag, style) from an explicit level prefix, else guess from text."""
    m = str(msg).lstrip()
    for tag, style in (("ERROR", "log.error"), ("WARN", "log.warn"),
                        ("SIGNAL", "log.signal"), ("ORDER", "log.order"),
                        ("INFO", "log.info")):
        if m.upper().startswith(tag):
            return tag, style
    low = m.lower()
    if "error" in low:
        return "ERROR", "log.error"
    if "warn" in low or "stale" in low:
        return "WARN", "log.warn"
    if "closed" in low or "sleeping" in low:
        return "INFO", "log.info"
    if "open " in low or "exit" in low or "signal" in low:
        return "SIGNAL", "log.signal"
    if "order" in low or "square" in low:
        return "ORDER", "log.order"
    return "INFO", "log.info"


def strip_level(msg):
    """Remove a leading level word from the message body."""
    for w in ("WARN ", "ERROR ", "SIGNAL ", "ORDER ", "INFO "):
        if msg.startswith(w):
            return msg[len(w):]
    return msg


def fmt_pf(value):
    if value is None:
        return "—"
    if value == float("inf"):
        return "∞"
    return f"{value:.2f}"


def fmt_age(seconds):
    """Compact age: 8s, 4m, 1h05, 2d3h."""
    if seconds is None:
        return "—"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


def fmt_r(value):
    if value is None:
        return "—"
    return f"{value:+.1f}R"


def fmt_compact(value, ccy="INR"):
    """A short money string for market cards: ₹1.2L, ₹3.4Cr, $12.1k."""
    sym = CCY_SYMBOL.get(ccy, ccy + " ")
    a = abs(value)
    sign = "-" if value < 0 else ""
    if ccy == "INR":
        if a >= 1e7:
            return f"{sign}{sym}{a / 1e7:.1f}Cr"
        if a >= 1e5:
            return f"{sign}{sym}{a / 1e5:.1f}L"
        if a >= 1e3:
            return f"{sign}{sym}{a / 1e3:.1f}k"
        return f"{sign}{sym}{a:.0f}"
    if a >= 1e9:
        return f"{sign}{sym}{a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}{sym}{a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}{sym}{a / 1e3:.1f}k"
    return f"{sign}{sym}{a:.0f}"


def fmt_pnl_compact(value, ccy="INR"):
    if value == 0:
        return fmt_compact(0, ccy)
    sign = "+" if value > 0 else ""
    return sign + fmt_compact(value, ccy)


_BRAILLE_BITS = (
    (0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80),
)


def braille_chart(values, width=40, height=4):
    """A braille line chart of *values*, oldest on the left.

    Braille gives a 2x4 dot grid per cell, so a small panel can draw a
    recognisable 5m price line without pulling in a chart library.
    """
    rows = max(1, int(height))
    cols = max(4, int(width))
    if not values:
        return [" " * cols for _ in range(rows)]
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return [" " * cols for _ in range(rows)]
        if len(vals) < cols:
            src = vals
            out = []
            last = len(src) - 1
            for j in range(cols):
                pos = j * last / max(1, cols - 1)
                i0 = min(int(pos), last)
                frac = pos - i0
                i1 = min(i0 + 1, last)
                out.append(src[i0] + (src[i1] - src[i0]) * frac)
        vals = out
    else:
        vals = vals[-cols:]
    lo, hi = min(vals), max(vals)
    px_w, px_h = cols * 2, rows * 4
    if hi == lo:
        hi = lo + 1.0
    cells = [[0] * cols for _ in range(rows)]

    def point(i, v):
        x = int(round(i / max(1, len(vals) - 1) * (px_w - 1)))
        y = int(round((hi - v) / (hi - lo) * (px_h - 1)))
        return x, y

    prev = None
    for i, v in enumerate(vals):
        cur = point(i, v)
        if prev is not None:
            x0, y0 = prev
            x1, y1 = cur
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                x = int(round(x0 + (x1 - x0) * s / steps))
                y = int(round(y0 + (y1 - y0) * s / steps))
                cells[y // 4][x // 2] |= _BRAILLE_BITS[y % 4][x % 2]
        prev = cur
    return ["".join(chr(0x2800 + c) for c in row) for row in cells]


MKT_CODE = {"India": "IN", "Japan": "JP", "Europe": "UK", "USA": "US",
            "HongKong": "HK", "Australia": "AU"}
MKT_CCY = {"India": "INR", "Japan": "JPY", "Europe": "GBP", "USA": "USD",
           "HongKong": "HKD", "Australia": "AUD"}


def mkt_code(markets_cfg: dict, market: str) -> str:
    """Short code for a market, from the market config."""
    cfg = (markets_cfg or {}).get(market) or {}
    return cfg.get("code") or MKT_CODE.get(market, market[:2].upper())


def mkt_ccy(markets_cfg: dict, market: str) -> str:
    """Trading currency for a market, from the market config."""
    cfg = (markets_cfg or {}).get(market) or {}
    return cfg.get("currency") or MKT_CCY.get(market, "INR")
