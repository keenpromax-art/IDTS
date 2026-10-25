"""Near-real-time price quotes.

yfinance has no websocket feed, but the quote metadata that backs
``Ticker.fast_info`` is refreshed continuously by the exchange and is
*fresher* than the last completed intraday bar. That is enough to watch
prices move and to react to a stop or target being hit within a second or
two rather than at the next scan.

The ticker is deliberately simple: a thread pool fetches the last price for
a set of symbols, and the caller decides which symbols matter right now.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

# a quote older than this is not worth showing
STALE_QUOTE_SECONDS = 120


class PriceTicker:
    """Fetches the latest price for a set of symbols."""

    def __init__(self, max_workers: int = 12):
        self.max_workers = max(1, int(max_workers))
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers,
                                        thread_name_prefix="quote")
        self._lock = threading.Lock()
        self.prices: Dict[str, float] = {}
        self.quoted_at: Dict[str, float] = {}
        self.last_refresh: float = 0.0
        self.errors: int = 0

    def close(self):
        self._pool.shutdown(wait=False)

    # -- fetching --------------------------------------------------------
    def _quote(self, symbol: str) -> Optional[float]:
        import yfinance as yf
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                price = ticker.fast_info.last_price
                if price is None:
                    return None
                return float(price)
            except Exception as e:
                if attempt < 2:
                    import time
                    import random
                    backoff = min(2 ** attempt, 30) + random.random()
                    time.sleep(backoff)
                else:
                    return None

    def refresh(self, symbols: Iterable[str]) -> Dict[str, float]:
        """Fetch the latest price for every symbol. Returns the fresh quotes."""
        syms = [s for s in dict.fromkeys(symbols) if s]
        if not syms:
            return {}
        out: Dict[str, float] = {}
        try:
            for sym, price in zip(syms, self._pool.map(self._quote, syms)):
                if price is not None and price > 0:
                    out[sym] = price
        except Exception as e:
            log.debug("quote refresh failed: %s", e)
        now = time.time()
        with self._lock:
            self.prices.update(out)
            for s in out:
                self.quoted_at[s] = now
            self.last_refresh = now
            self.errors += len(syms) - len(out)
        return out

    # -- reading ---------------------------------------------------------
    def price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self.prices.get(symbol)

    def age(self, symbol: str) -> Optional[float]:
        """Seconds since this symbol was last quoted."""
        with self._lock:
            ts = self.quoted_at.get(symbol)
        if ts is None:
            return None
        return time.time() - ts

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self.prices)

    def age_of_newest(self) -> Optional[float]:
        with self._lock:
            if not self.quoted_at:
                return None
            return time.time() - max(self.quoted_at.values())
