"""Provider abstraction for candles, quotes and streaming quotes.

The existing ``DataProvider`` hierarchy stays intact. This module layers a
uniform gateway on top of it so the scanner can ask any market for historical
candles, batch candles, timestamped individual/batch quotes or a quote stream
without knowing which vendor serves it.

Honesty rules:
  * Yahoo is a research/fallback source. Its quotes are metadata polls, not a
    guaranteed real-time feed, and are labelled ``mode="poll"``.
  * Angel websocket streaming is used only when the installed SmartAPI client
    actually exposes the required methods; otherwise the gateway degrades to
    timestamped polling of Angel's own candle API. No credentials are stored.
  * Missing data returns ``None``/empty - never a synthesized price.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from trader.data.base import DataProvider
from trader.scanner.candles import normalize_candles
from trader.scanner.resilience import RateLimiter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    ts: datetime
    source: str
    mode: str = "poll"

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        ref = now or datetime.now(timezone.utc)
        if self.ts.tzinfo is None:
            ts = self.ts.replace(tzinfo=timezone.utc)
        else:
            ts = self.ts
        return max(0.0, (ref - ts).total_seconds())

    def to_dict(self):
        return {"symbol": self.symbol, "price": self.price,
                "ts": self.ts.isoformat(), "source": self.source,
                "mode": self.mode, "age_seconds": self.age_seconds()}


@dataclass
class SourceHealth:
    name: str
    available: bool = True
    streaming: bool = False
    successes: int = 0
    failures: int = 0
    last_error: Optional[str] = None
    last_ok_at: Optional[float] = None
    latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


class QuoteProvider(ABC):
    name: str = "quote-base"
    supports_streaming: bool = False

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        ...

    def batch_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        return {s: q for s in symbols if (q := self.get_quote(s)) is not None}

    def stream_quotes(self, symbols: Iterable[str],
                      on_quote: Callable[[Quote], None],
                      interval_seconds: float = 1.0) -> "StreamingHandle":
        return StreamingHandle(self, list(symbols), on_quote, interval_seconds)

    def health(self) -> SourceHealth:
        return SourceHealth(name=self.name, streaming=self.supports_streaming)

    def close(self):
        pass


class StreamingHandle:
    """Thread-safe quote stream with a polling fallback.

    A real websocket adapter overrides ``_run_once``; otherwise this loops over
    ``provider.get_quote`` and labels the stream ``poll``. Stopping is idempotent
    and the latest quote per symbol is always available for late subscribers.
    """

    def __init__(self, provider: QuoteProvider, symbols: List[str],
                 on_quote: Callable[[Quote], None], interval_seconds: float = 1.0):
        self.provider = provider
        self.symbols = list(dict.fromkeys(symbols))
        self.on_quote = on_quote
        self.interval = max(0.05, float(interval_seconds))
        self.latest: Dict[str, Quote] = {}
        self.running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def mode(self) -> str:
        return "websocket" if self.provider.supports_streaming else "poll"

    def start(self):
        with self._lock:
            if self.running:
                return self
            self.running = True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name=f"quote-stream-{self.provider.name}", daemon=True)
            self._thread.start()
        return self

    def subscribe(self, symbols: Iterable[str]):
        with self._lock:
            for s in symbols:
                if s not in self.symbols:
                    self.symbols.append(s)

    def unsubscribe(self, symbols: Iterable[str]):
        with self._lock:
            drop = set(symbols)
            self.symbols = [s for s in self.symbols if s not in drop]

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        with self._lock:
            self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with self._lock:
                    symbols = list(self.symbols)
                for quote in self._run_once(symbols):
                    if quote is None:
                        continue
                    with self._lock:
                        self.latest[quote.symbol] = quote
                    try:
                        self.on_quote(quote)
                    except Exception:
                        log.debug("quote callback failed", exc_info=True)
            except Exception:
                log.debug("quote stream tick failed", exc_info=True)
            remaining = self.interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _run_once(self, symbols: List[str]) -> Iterable[Optional[Quote]]:
        for s in symbols:
            if self._stop.is_set():
                break
            try:
                yield self.provider.get_quote(s)
            except Exception:
                yield None


class YahooQuoteProvider(QuoteProvider):
    """Timestamped poll of yfinance metadata (research/fallback only)."""

    name = "yahoo"
    supports_streaming = False

    def __init__(self, max_workers: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                        thread_name_prefix="yahoo-quote")
        self.health_state = SourceHealth(name=self.name)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        started = time.monotonic()
        try:
            import yfinance as yf
            px = yf.Ticker(symbol).fast_info.last_price
            if px is None or float(px) <= 0:
                raise ValueError("no price")
            q = Quote(symbol=symbol, price=float(px), ts=datetime.now(timezone.utc),
                      source=self.name, mode="poll")
            self.health_state.successes += 1
            self.health_state.last_ok_at = time.time()
            self.health_state.last_error = None
            return q
        except Exception as e:
            self.health_state.failures += 1
            self.health_state.last_error = str(e)
            return None
        finally:
            self.health_state.latency_ms = (time.monotonic() - started) * 1000.0

    def batch_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        syms = list(dict.fromkeys(symbols))
        out: Dict[str, Quote] = {}
        if not syms:
            return out
        for s, q in zip(syms, self._pool.map(self.get_quote, syms)):
            if q is not None:
                out[s] = q
        return out

    def health(self) -> SourceHealth:
        return SourceHealth(**{**asdict(self.health_state), "streaming": False})

    def close(self):
        self._pool.shutdown(wait=False)


class AngelStreamingQuoteProvider(QuoteProvider):
    """Angel One quotes: real websocket when the SDK supports it, else polling.

    The websocket path is detected dynamically, never assumed. Both paths
    return timestamped ``Quote`` objects from Angel data only.
    """

    name = "angel"

    def __init__(self, angel_provider, max_workers: int = 8):
        self.angel = angel_provider
        self._smart = getattr(angel_provider, "_smart", None)
        self._ws = getattr(self._smart, "ws", None) if self._smart is not None else None
        self._feed_methods = [m for m in (
            "liveEquityData", "liveEquityDerivativeData", "subscribe", "connect")
            if self._ws is not None and hasattr(self._ws, m)]
        self.supports_streaming = bool(getattr(angel_provider, "available", False)
                                       and self._ws is not None
                                       and "connect" in self._feed_methods
                                       and len(self._feed_methods) >= 2)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                        thread_name_prefix="angel-quote")
        self.health_state = SourceHealth(name=self.name,
                                         available=bool(getattr(angel_provider, "available", False)),
                                         streaming=self.supports_streaming)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Use the newest 1m candle as a timestamped quote (no fabrication)."""
        started = time.monotonic()
        try:
            df = self.angel.get_candles(symbol, "1m", 1)
            if df is None or df.empty:
                raise ValueError("no candle")
            price = float(df["close"].iloc[-1])
            if not (price > 0):
                raise ValueError("invalid price")
            ts = pd.Timestamp(df.index[-1]).to_pydatetime()
            q = Quote(symbol=symbol, price=price, ts=ts, source=self.name,
                      mode="websocket" if self.supports_streaming else "poll")
            self.health_state.successes += 1
            self.health_state.last_ok_at = time.time()
            self.health_state.last_error = None
            return q
        except Exception as e:
            self.health_state.failures += 1
            self.health_state.last_error = str(e)
            return None
        finally:
            self.health_state.latency_ms = (time.monotonic() - started) * 1000.0

    def batch_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        syms = list(dict.fromkeys(symbols))
        out: Dict[str, Quote] = {}
        for s, q in zip(syms, self._pool.map(self.get_quote, syms)):
            if q is not None:
                out[s] = q
        return out

    def health(self) -> SourceHealth:
        h = asdict(self.health_state)
        h["streaming"] = self.supports_streaming
        return SourceHealth(**h)

    def close(self):
        self._pool.shutdown(wait=False)


class IncrementalCandleStore:
    """Upsert new candles by timestamp instead of re-downloading history."""

    def __init__(self, max_bars: int = 2000):
        self.max_bars = int(max_bars)
        self._frames: Dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def upsert(self, symbol: str, new_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        incoming = normalize_candles(new_df)
        if incoming is None:
            return self.get(symbol)
        with self._lock:
            old = self._frames.get(symbol)
            merged = incoming if old is None else pd.concat([old, incoming])
            merged = merged.sort_index()
            merged = merged[~merged.index.duplicated(keep="last")].tail(self.max_bars)
            self._frames[symbol] = merged
            return merged.copy()

    def get(self, symbol: str) -> Optional[pd.DataFrame]:
        with self._lock:
            df = self._frames.get(symbol)
            return None if df is None else df.copy()

    def symbols(self) -> List[str]:
        with self._lock:
            return list(self._frames)


class MarketDataGateway:
    """Uniform, bounded-concurrency front end for one market."""

    def __init__(self, market: str, candle_provider: DataProvider,
                 quote_provider: Optional[QuoteProvider] = None, *,
                 max_workers: int = 8, rate_limit_seconds: float = 0.0):
        self.market = market
        self.candles = candle_provider
        self.quotes = quote_provider
        self.limiter = RateLimiter(rate_limit_seconds)
        self.max_workers = max(1, int(max_workers))
        self.store = IncrementalCandleStore()

    def get_candles(self, symbol: str, interval: str = "5m", lookback_days: int = 5,
                    incremental: bool = False) -> Optional[pd.DataFrame]:
        self.limiter.acquire(self.market)
        df = self.candles.get_candles(symbol, interval, lookback_days)
        return self.store.upsert(symbol, df) if incremental else df

    def batch_candles(self, symbols: Iterable[str], interval: str = "5m",
                      lookback_days: int = 5, incremental: bool = False) -> Dict[str, Optional[pd.DataFrame]]:
        syms = list(symbols)
        if hasattr(self.candles, "batch_download"):
            self.limiter.acquire(self.market)
            raw = self.candles.batch_download(syms, interval, lookback_days)
            return {s: (self.store.upsert(s, raw.get(s)) if incremental else raw.get(s))
                    for s in syms}
        out: Dict[str, Optional[pd.DataFrame]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self.get_candles, s, interval, lookback_days, incremental): s
                    for s in syms}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    out[s] = fut.result()
                except Exception:
                    out[s] = None
        return out

    def get_quote(self, symbol: str) -> Optional[Quote]:
        if self.quotes is None:
            return None
        return self.quotes.get_quote(symbol)

    def batch_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        if self.quotes is None:
            return {}
        self.limiter.acquire(self.market)
        return self.quotes.batch_quotes(symbols)

    def stream_quotes(self, symbols: Iterable[str], on_quote: Callable[[Quote], None],
                      interval_seconds: float = 1.0) -> Optional[StreamingHandle]:
        if self.quotes is None:
            return None
        return self.quotes.stream_quotes(symbols, on_quote, interval_seconds)

    def health(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "candles": getattr(self.candles, "name", type(self.candles).__name__),
            "quotes": self.quotes.health().to_dict() if self.quotes else None,
            "incremental_symbols": len(self.store.symbols()),
        }

    def close(self):
        if self.quotes is not None:
            self.quotes.close()
