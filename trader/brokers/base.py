from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Callable
import pandas as pd

from trader.core.models import Order, Position, Side


class BrokerCapability:
    """Declared capabilities of a broker. Live gating checks these."""
    def __init__(self, name="base", supports_short=True, supports_stop_orders=False,
                 supports_partial_fills=True, live_allowed=False):
        self.name = name
        self.supports_short = supports_short
        self.supports_stop_orders = supports_stop_orders
        self.supports_partial_fills = supports_partial_fills
        self.live_allowed = live_allowed

    def as_dict(self):
        return {
            "name": self.name, "supports_short": self.supports_short,
            "supports_stop_orders": self.supports_stop_orders,
            "supports_partial_fills": self.supports_partial_fills,
            "live_allowed": self.live_allowed,
        }


class Broker(ABC):
    name: str = "base"
    capabilities: "BrokerCapability" = BrokerCapability()

    @abstractmethod
    def place_order(self, order: Order, price_fn: Optional[Callable[[str], Optional[float]]] = None) -> Order:
        ...

    @abstractmethod
    def cancel(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def positions(self) -> List[Position]:
        ...

    @abstractmethod
    def square_off_all(self, price_fn=None) -> List[Order]:
        ...

    def reconcile(self, price_fn=None) -> List[Position]:
        return self.positions()

    def is_connected(self) -> bool:
        return True
