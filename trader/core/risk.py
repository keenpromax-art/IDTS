from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from trader.core.models import Position, Side
from trader.core.market_clock import MarketClock


class RiskManager:
    def __init__(self, cfg: dict, clock: MarketClock):
        self.cfg = cfg
        self.risk = cfg["risk"]
        self.clock = clock
        self.kill_switch = False
        self.daily_pnl: Dict[str, float] = {}
        self.trades_today: Dict[str, int] = {}  # symbol -> count
        self.last_loss_time: Dict[str, datetime] = {}
        self.paused = False
        self.kill_switch_at = None

    def reset_day(self):
        self.daily_pnl = {}
        self.trades_today = {}
        self.last_loss_time = {}
        self.kill_switch = False

    def position_size(self, equity: float, entry: float, stop: float,
                      qty_step: Optional[float] = None):
        risk_pct = self.risk["risk_per_trade_pct"] / 100.0
        risk_amt = equity * risk_pct
        per_share_risk = abs(entry - stop)
        if per_share_risk <= 0:
            return 0
        qty = risk_amt / per_share_risk
        # cap the notional: a very tight stop would otherwise ask for a
        # position worth many times the account
        max_pct = self.risk.get("max_position_pct_of_equity")
        if max_pct:
            max_notional = equity * float(max_pct) / 100.0
            if entry > 0:
                qty = min(qty, max_notional / entry)
        # fractional markets (crypto): coins trade in fractions, so the
        # quantity is floored to the market's qty_step (e.g. 0.0001) instead
        # of whole units. Legacy markets keep integer sizing.
        import math
        step = float(qty_step) if qty_step else None
        if step and 0 < step < 1:
            q = min(qty, math.floor(qty / step) * step)
            q = round(q, 8)
            # never size into a non-positive or NaN distance
            if not (q > 0) or q != q:
                return 0
            return q
        # legacy markets: whole units, with integer lot rounding (default 1)
        q = int(qty)
        lot = int(self.risk.get("lot_size", 1) or 1)
        if lot > 1:
            q = (q // lot) * lot
        # never size into a non-positive or NaN distance
        if not (q > 0) or q != q:
            return 0
        return max(0, q)

    def total_positions(self, positions: List[Position]) -> int:
        return len(positions)

    def market_positions(self, positions: List[Position], market: str) -> int:
        return sum(1 for p in positions if p.market == market)

    def can_open(self, market: str, symbol: str, side: Side, positions: List[Position],
                equity: float, daily_pnl_total: float, stale_data: bool = False) -> tuple:
        if self.paused:
            return False, "paused"
        if self.kill_switch:
            return False, "kill switch"
        if stale_data:
            return False, "stale data"
        # daily loss limit
        max_loss_pct = self.risk["max_daily_loss_pct"] / 100.0
        if equity > 0 and daily_pnl_total <= -equity * max_loss_pct:
            self.kill_switch = True
            return False, "max daily loss"
        # concurrent limits
        if len(positions) >= self.risk["max_concurrent_positions"]:
            return False, "max concurrent"
        if self.market_positions(positions, market) >= self.risk["max_positions_per_market"]:
            return False, "max per market"
        if any(p.symbol == symbol for p in positions):
            return False, "already in position"
        # per symbol daily trades
        if self.trades_today.get(symbol, 0) >= self.risk["max_trades_per_symbol_per_day"]:
            return False, "max trades/symbol"
        # cooldown
        if symbol in self.last_loss_time:
            cd = self.risk["cooldown_minutes_after_loss"]
            if datetime.now(timezone.utc) - self.last_loss_time[symbol] < timedelta(minutes=cd):
                return False, "cooldown"
        # time of day
        cfg = self.clock.cfg.get(market, {}) or {}
        st = self.clock.status(market)
        local = st["local_time"]
        open_dt = st["open_dt"]
        close_dt = st["close_dt"]
        if not cfg.get("always_open"):
            if open_dt and local < open_dt + timedelta(minutes=self.risk["no_entry_first_minutes"]):
                return False, "too early"
            if close_dt and local > close_dt - timedelta(minutes=self.risk["no_entry_last_minutes"]):
                return False, "too late"
        if st["status"] not in ("OPEN",):
            # allow pre-close? no new entries in pre-close
            return False, "market not open"
        return True, "ok"

    def should_square_off(self, market: str) -> bool:
        cfg = self.clock.cfg.get(market, {}) or {}
        if cfg.get("always_open"):
            return False
        st = self.clock.status(market)
        close_dt = st["close_dt"]
        local = st["local_time"]
        if close_dt is None:
            return False
        mins = self.risk["square_off_before_close_minutes"]
        return local >= close_dt - timedelta(minutes=mins)

    def record_trade_result(self, symbol: str, pnl: float):
        self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
        if pnl < 0:
            self.last_loss_time[symbol] = datetime.now(timezone.utc)

    def trigger_kill(self):
        self.kill_switch = True
        self.kill_switch_at = datetime.now(timezone.utc)

    def is_killed(self) -> bool:
        return bool(self.kill_switch)
