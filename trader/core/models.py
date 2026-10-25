from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal, Dict, Any
from enum import Enum


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ExitReason(str, Enum):
    TARGET = "target"
    STOPLOSS = "sl"
    TRAILING = "trailing"
    SQUAREOFF = "square-off"
    KILLSWITCH = "kill-switch"
    MANUAL = "manual"


@dataclass
class Candle:
    ts: datetime  # UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    interval: str = "5m"


@dataclass
class Signal:
    symbol: str
    market: str
    strategy: str
    side: Side
    entry: float
    stop: float
    target: float
    confidence: float = 0.5
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    regime: str = ""
    score: float = 0.0


@dataclass
class Order:
    id: str
    symbol: str
    market: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    broker_order_id: Optional[str] = None
    charges: float = 0.0
    stop: Optional[float] = None
    target: Optional[float] = None
    strategy: str = ""
    score_reason: str = ""


@dataclass
class Position:
    id: str
    symbol: str
    market: str
    side: Side
    strategy: str
    qty: float
    entry: float
    stop: float
    target: float
    current: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trailing_stop: Optional[float] = None
    moved_to_be: bool = False
    charges: float = 0.0
    score_reason: str = ""


@dataclass
class Trade:
    id: str
    symbol: str
    market: str
    side: Side
    strategy: str
    qty: float
    entry: float
    exit: float
    opened_at: datetime
    closed_at: datetime
    exit_reason: ExitReason
    charges: float = 0.0
    score_reason: str = ""
    stop: float = 0.0

    @property
    def pnl_gross(self) -> float:
        if self.side == Side.LONG:
            return (self.exit - self.entry) * self.qty
        return (self.entry - self.exit) * self.qty

    @property
    def pnl(self) -> float:
        return self.pnl_gross - self.charges

    @property
    def r_multiple(self) -> float:
        risk = abs(self.entry - self.stop)
        if risk <= 0 or self.qty <= 0:
            return 0.0
        return self.pnl_gross / (risk * self.qty)


@dataclass
class BacktestStats:
    strategy: str
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    max_drawdown: float = 0.0
    eligible: bool = False


@dataclass
class Regime:
    symbol: str
    adx: float = 0.0
    atr_percentile: float = 50.0
    trending: bool = False
    ranging: bool = False
    label: str = "unknown"
