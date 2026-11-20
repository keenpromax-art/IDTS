from __future__ import annotations

import random
import time
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf

from yf_proxy import ProxyPool, YahooFetcher, load_proxies, start_background_validation
from trader.scanner.resilience import ProviderResilience, RetryPolicy
from trader.data.base import DataProvider

log = logging.getLogger(__name__)


def fetch_with_retry(
    fn,
    *,
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
):
    """Call *fn* with jittered exponential backoff on transient failure.

    ``fn`` is invoked up to ``retries + 1`` times. After each failed attempt
    the delay grows as ``base_delay * 2**attempt`` (capped at ``max_delay``)
    plus uniform jitter in ``[0, 1)``. On success the result is returned
    immediately. If every attempt fails, the last exception is re-raised.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # retry on any transient failure
            last_exc = exc
            if attempt == retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay += random.random()
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


class YFinanceProvider(DataProvider):
    name = "yfinance"

    TTL_BY_INTERVAL = {
        "1m": 45,
        "2m": 90,
        "5m": 240,
        "15m": 600,
        "30m": 1200,
        "1h": 1800,
        "1d": 3600,
    }

    _shared_fetcher: Optional[YahooFetcher] = None
    _shared_pool: Optional[ProxyPool] = None
    _initialized = False

    def __init__(self, cache_ttl: int = 55):
        self.cache_ttl = cache_ttl
        if not YFinanceProvider._initialized:
            proxies_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "proxies.txt")
            if not os.path.exists(proxies_path):
                proxies_path = "proxies.txt"
            pool = ProxyPool(load_proxies(proxies_path))
            try:
                pool.validate()
            except Exception as e:
                log.warning("Initial proxy validation failed: %s", e)
            start_background_validation(pool, interval_seconds=1800)
            YFinanceProvider._shared_pool = pool
            YFinanceProvider._shared_fetcher = YahooFetcher(pool, cache_ttl=cache_ttl)
            YFinanceProvider._initialized = True

        self.fetcher = YFinanceProvider._shared_fetcher
        # Short negative cache + breaker + one bounded retry on transport
        # failure. Empty-but-valid frames are NOT cached (a symbol can return
        # to the feed), so no market data is ever fabricated.
        self.resilience = ProviderResilience(
            "yfinance", negative_ttl_seconds=float(os.getenv("IDTS_YF_NEG_TTL", "10")),
            failure_threshold=5, recovery_seconds=30.0,
            retry=RetryPolicy(retries=1, base_delay=0.3, max_delay=1.5))

    @property
    def health(self):
        return self.resilience.health

    def _ttl(self, interval: str) -> float:
        ttl = self.TTL_BY_INTERVAL.get(interval)
        if ttl is None:
            return float(self.cache_ttl)
        return max(float(self.cache_ttl), float(ttl))

    def get_candles(self, symbol, interval="5m", lookback_days=5) -> Optional[pd.DataFrame]:
        try:
            period = f"{min(lookback_days, 59)}d" if interval in ("5m", "15m", "30m", "1h") else f"{min(lookback_days, 7)}d"
            if interval == "1m":
                period = "7d"
            
            fetched = self.resilience.fetch(
                f"{symbol}|{interval}|{period}",
                lambda: self.fetcher.download([symbol], period=period, interval=interval),
                is_valid=lambda d: d is not None,
            )
            if not fetched.ok:
                return None
            data = fetched.value
            if data is None or (hasattr(data, "empty") and data.empty):
                return None
            
            if isinstance(data, pd.DataFrame) and "Open" in data.columns:
                df = data.copy()
            elif isinstance(data, pd.DataFrame) and symbol in data.columns.get_level_values(0):
                df = data[symbol].copy()
            else:
                if len(data.columns) > 1 and isinstance(data.columns, pd.MultiIndex):
                    try:
                        df = data[symbol].copy()
                    except Exception:
                        df = data.copy()
                else:
                    df = data.copy()

            if df is None or df.empty:
                return None

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            if df.empty:
                return None

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index.name = "ts"

            return self.finalize_candles(df, symbol=symbol, interval=interval)
        except Exception as e:
            log.warning("YahooFetcher error for %s: %s (skipping bar)", symbol, e)
            return None

    def batch_download(self, symbols, interval="5m", lookback_days=5) -> dict:
        result = {}
        if not symbols:
            return result
        try:
            period = f"{min(lookback_days, 59)}d" if interval != "1m" else "7d"
            data = self.fetcher.download(list(symbols), period=period, interval=interval)
            for sym in symbols:
                try:
                    if data is None or (hasattr(data, "empty") and data.empty):
                        result[sym] = None
                        continue
                    
                    if len(symbols) == 1:
                        df = data.copy()
                    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex) and sym in data.columns.get_level_values(0):
                        df = data[sym].copy()
                    elif isinstance(data, pd.DataFrame) and sym in data.columns:
                        df = data.copy()
                    else:
                        try:
                            df = data[sym].copy()
                        except Exception:
                            df = data.copy()

                    if df is None or df.empty:
                        result[sym] = None
                        continue

                    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    df = df[["open", "high", "low", "close", "volume"]].dropna()
                    if df.empty:
                        result[sym] = None
                        continue

                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    df.index.name = "ts"

                    result[sym] = self.finalize_candles(df, symbol=sym, interval=interval)
                except Exception as e:
                    log.warning("batch parse error %s: %s (skipping)", sym, e)
                    result[sym] = None
        except Exception as e:
            log.error("batch download failed: %s (skipping bar)", e)
            for sym in symbols:
                result[sym] = None
        return result
