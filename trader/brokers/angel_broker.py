from __future__ import annotations

import os
import uuid
import logging
from typing import Optional, List, Callable, Dict
from datetime import datetime, timezone

from trader.core.models import Order, Position, Side, OrderType, OrderStatus, ExitReason
from trader.brokers.base import BrokerCapability

log = logging.getLogger(__name__)


class AngelBroker:
    name = "angel"
    capabilities = BrokerCapability(name="angel", supports_short=True,
        supports_stop_orders=False, supports_partial_fills=True, live_allowed=True)

    def __init__(self, angel_provider):
        self.ap = angel_provider
        self._smart = angel_provider._smart if angel_provider else None
        self.positions: Dict[str, Position] = {}

    def place_order(self, order: Order, price_fn=None) -> Order:
        if self._smart is None:
            order.status = OrderStatus.REJECTED
            return order
        try:
            token = self.ap._token_for(order.symbol)
            txn = "BUY" if order.side == Side.LONG else "SELL"
            otype = "MARKET" if order.order_type == OrderType.MARKET else "LIMIT"
            params = {
                "variety": "NORMAL",
                "tradingsymbol": order.symbol.replace(".NS", ""),
                "symboltoken": token,
                "transactiontype": txn,
                "exchange": "NSE",
                "ordertype": otype,
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(order.price or 0),
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(order.qty),
            }
            resp = self._smart.placeOrder(params)
            order.broker_order_id = str(resp)
            order.status = OrderStatus.FILLED
            # approximate fill price
            price = order.price
            if price is None and price_fn is not None:
                price = price_fn(order.symbol)
            order.filled_price = price
            return order
        except Exception as e:
            log.error("angel place_order failed: %s", e)
            order.status = OrderStatus.REJECTED
            return order

    def is_connected(self) -> bool:
        try:
            return bool(self._smart is not None)
        except Exception:
            return False

    def cancel(self, order_id: str) -> bool:
        try:
            if self._smart:
                self._smart.cancelOrder(order_id, "NORMAL")
                return True
        except Exception as e:
            log.error("cancel failed: %s", e)
        return False

    def positions(self) -> List[Position]:
        try:
            if self._smart:
                resp = self._smart.position()
                # not fully mapped; return cached
                pass
        except Exception as e:
            log.error("positions failed: %s", e)
        return list(self.positions.values())

    def exit_position(self, pos: Position, price: float, reason: ExitReason):
        order = Order(id=str(uuid.uuid4()), symbol=pos.symbol, market=pos.market,
                      side=Side.SHORT if pos.side == Side.LONG else Side.LONG,
                      qty=pos.qty, order_type=OrderType.MARKET, price=price,
                      strategy=pos.strategy, score_reason=pos.score_reason)
        self.place_order(order, lambda s: price)
        if pos.id in self.positions:
            del self.positions[pos.id]
        return price, 0.0

    def square_off_all(self, price_fn=None):
        out = []
        for pid, pos in list(self.positions.items()):
            price = pos.current
            if price_fn is not None:
                p = price_fn(pos.symbol)
                if p is not None:
                    price = p
            self.exit_position(pos, price, ExitReason.SQUAREOFF)
            out.append((pos, price, 0.0))
        return out

    def reconcile(self, price_fn=None):
        return self.positions()
