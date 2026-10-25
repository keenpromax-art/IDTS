from __future__ import annotations

import uuid
import time
import logging
from typing import Optional, List, Callable, Dict
from datetime import datetime, timezone

from trader.core.models import Order, Position, Trade, Side, OrderType, OrderStatus, ExitReason
from trader.core.models import Signal

log = logging.getLogger(__name__)


def calc_charges(market: str, side, qty: float, price: float, cfg: dict) -> float:
    bcfg = cfg.get("paper", {}).get("brokerage", {}).get(market, {})
    brokerage_flat = float(bcfg.get("brokerage_flat", 0))
    stt_pct = float(bcfg.get("stt_pct", 0)) / 100.0
    exch_pct = float(bcfg.get("exchange_pct", 0)) / 100.0
    gst_pct = float(bcfg.get("gst_pct", 0)) / 100.0
    turnover = qty * price
    stt = turnover * stt_pct
    exch = turnover * exch_pct
    gst = brokerage_flat * gst_pct
    return brokerage_flat + stt + exch + gst


from trader.brokers.base import BrokerCapability

class PaperBroker:
    name = "paper"
    capabilities = BrokerCapability(name="paper", supports_short=True,
        supports_stop_orders=False, supports_partial_fills=True, live_allowed=False)

    def __init__(self, cfg: dict, slippage_bps: float = 5.0):
        self.cfg = cfg
        self.slippage_bps = slippage_bps
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []

    def _slip(self, price: float, side: Side, is_buy: bool) -> float:
        bps = self.slippage_bps / 10000.0
        if is_buy:
            return price * (1 + bps)
        return price * (1 - bps)

    def place_order(self, order: Order, price_fn=None) -> Order:
        price = order.price
        if price is None and price_fn is not None:
            price = price_fn(order.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            return order
        is_buy = order.side == Side.LONG
        fill = self._slip(price, order.side, is_buy)
        order.filled_price = fill
        order.status = OrderStatus.FILLED
        order.charges = calc_charges(order.market, order.side, order.qty, fill, self.cfg)
        self.orders.append(order)
        # create position
        pos = Position(
            id=order.id,
            symbol=order.symbol,
            market=order.market,
            side=order.side,
            strategy=getattr(order, "strategy", "unknown"),
            qty=order.qty,
            entry=fill,
            stop=order.stop if hasattr(order, "stop") else fill,
            target=order.target if hasattr(order, "target") else fill,
            current=fill,
            charges=order.charges,
            score_reason=getattr(order, "score_reason", ""),
        )
        # need stop/target from order? Order doesn't carry. We'll set via attributes if present
        if hasattr(order, "stop"):
            pos.stop = order.stop
        if hasattr(order, "target"):
            pos.target = order.target
        if hasattr(order, "strategy"):
            pos.strategy = order.strategy
        if hasattr(order, "score_reason"):
            pos.score_reason = order.score_reason
        self.positions[order.id] = pos
        return order

    def cancel(self, order_id: str) -> bool:
        return False

    def is_connected(self) -> bool:
        return True

    def positions(self) -> List[Position]:
        return list(self.positions.values())

    def exit_position(self, pos: Position, price: float, reason: ExitReason) -> float:
        is_buy = pos.side == Side.SHORT  # closing short = buy
        fill = self._slip(price, pos.side, is_buy)
        ch = calc_charges(pos.market, pos.side, pos.qty, fill, self.cfg)
        # the position may not be in our book (restored, or already closed)
        self.positions.pop(pos.id, None)
        return fill, ch

    def square_off_all(self, price_fn=None) -> List:
        out = []
        for pid, pos in list(self.positions.items()):
            price = pos.current
            if price_fn is not None:
                p = price_fn(pos.symbol)
                if p is not None:
                    price = p
            fill, ch = self.exit_position(pos, price, ExitReason.SQUAREOFF)
            out.append((pos, fill, ch))
        return out
