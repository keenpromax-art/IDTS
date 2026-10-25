from __future__ import annotations

from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple
import yaml
from pathlib import Path


class MarketClock:
    def __init__(self, markets_cfg: Dict[str, Any]):
        self.cfg = markets_cfg

    def _tz(self, market: str) -> ZoneInfo:
        return ZoneInfo(self.cfg[market]["timezone"])

    def _parse(self, s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _is_weekend(self, market: str, dt: datetime) -> bool:
        # Saturday=5, Sunday=6
        return dt.weekday() >= 5

    def _is_holiday(self, market: str, d: date) -> bool:
        holidays = self.cfg[market].get("holidays", []) or []
        return str(d) in [str(h) for h in holidays]

    def _in_lunch(self, market: str, t: time) -> bool:
        lunch = self.cfg[market].get("lunch")
        if not lunch:
            return False
        s = self._parse(lunch["start"])
        e = self._parse(lunch["end"])
        return s <= t < e

    def now_local(self, market: str) -> datetime:
        return datetime.now(self._tz(market))

    def status(self, market: str, now: Optional[datetime] = None) -> Dict[str, Any]:
        tz = self._tz(market)
        now = now or datetime.now(tz)
        local = now.astimezone(tz) if now.tzinfo else now.replace(tzinfo=tz)
        # ensure tz-aware local
        if local.tzinfo is None:
            local = local.replace(tzinfo=tz)
        cfg = self.cfg[market]
        if cfg.get("always_open"):
            d = local.date()
            open_dt = datetime.combine(d, time(0, 0), tzinfo=tz)
            close_dt = datetime.combine(d, time(23, 59), tzinfo=tz)
            return {
                "market": market,
                "status": "OPEN",
                "reason": None,
                "local_time": local,
                "open_dt": open_dt,
                "close_dt": close_dt,
                "next_open": None,
                "next_close": None,
                "countdown_seconds": None,
                "in_lunch": False,
            }
        open_t = self._parse(cfg["open"])
        close_t = self._parse(cfg["close"])
        pre_close_min = cfg.get("pre_close_minutes", 15)
        d = local.date()
        weekend = self._is_weekend(market, local)
        holiday = self._is_holiday(market, d)
        t = local.timetz().replace(tzinfo=None) if False else local.time()

        # build open/close datetimes for today
        open_dt = datetime.combine(d, open_t, tzinfo=tz)
        close_dt = datetime.combine(d, close_t, tzinfo=tz)
        pre_close_dt = close_dt - timedelta(minutes=pre_close_min)

        closed_reason = None
        if weekend:
            closed_reason = "weekend"
        elif holiday:
            closed_reason = "holiday"

        if closed_reason:
            # next open: next business day open
            next_open = self._next_open(market, local)
            return {
                "market": market,
                "status": "CLOSED",
                "reason": closed_reason,
                "local_time": local,
                "open_dt": open_dt,
                "close_dt": close_dt,
                "next_open": next_open,
                "next_close": None,
                "countdown_seconds": (next_open - local).total_seconds() if next_open else None,
                "in_lunch": False,
            }

        # normal day
        if local < open_dt:
            status = "CLOSED"
        elif local >= close_dt:
            status = "CLOSED"
        elif local >= pre_close_dt:
            status = "PRE-CLOSE"
        elif self._in_lunch(market, t):
            status = "LUNCH"
        else:
            status = "OPEN"

        if status == "CLOSED":
            if local < open_dt:
                next_open = open_dt
                next_close = close_dt
            else:
                next_open = self._next_open(market, local)
                next_close = None
        else:
            next_open = None
            next_close = close_dt

        countdown = None
        if status == "CLOSED" and next_open is not None:
            countdown = (next_open - local).total_seconds()
        elif status in ("OPEN", "PRE-CLOSE", "LUNCH") and next_close is not None:
            countdown = (next_close - local).total_seconds()

        return {
            "market": market,
            "status": status,
            "reason": None,
            "local_time": local,
            "open_dt": open_dt,
            "close_dt": close_dt,
            "next_open": next_open,
            "next_close": next_close,
            "countdown_seconds": countdown,
            "in_lunch": self._in_lunch(market, t),
        }

    def is_open(self, market: str, now: Optional[datetime] = None) -> bool:
        st = self.status(market, now)
        return st["status"] in ("OPEN", "PRE-CLOSE", "LUNCH") and st["status"] != "LUNCH"

    def is_trading_allowed(self, market: str, now: Optional[datetime] = None) -> bool:
        # open and not lunch and not pre-close? pre-close allows management but new entries blocked by risk; clock just says open
        st = self.status(market, now)
        return st["status"] in ("OPEN", "PRE-CLOSE")

    def _next_open(self, market: str, local: datetime) -> datetime:
        tz = self._tz(market)
        cfg = self.cfg[market]
        open_t = self._parse(cfg["open"])
        d = local.date() + timedelta(days=1)
        while True:
            if d.weekday() < 5 and str(d) not in [str(h) for h in (cfg.get("holidays") or [])]:
                return datetime.combine(d, open_t, tzinfo=tz)
            d += timedelta(days=1)

    def time_to_close_seconds(self, market: str, now: Optional[datetime] = None) -> Optional[float]:
        st = self.status(market, now)
        if st["next_close"] is None:
            return None
        local = st["local_time"]
        return (st["next_close"] - local).total_seconds()

    def fmt_countdown(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "-"
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"
