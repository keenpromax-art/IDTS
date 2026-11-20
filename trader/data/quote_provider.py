"""Abstract interface for quote providers.

PriceTicker depends on this interface rather than on yfinance directly, so a
second provider can be swapped in later. Stub interface only -- no provider
is wired yet (YAGNI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class QuoteProvider(ABC):
    """Fetch the latest quote for a single symbol."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[float]:
        """Return the last price for *symbol*, or None if unavailable."""
        raise NotImplementedError
