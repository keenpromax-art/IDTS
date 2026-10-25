from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class YFinanceProvider:
    name = "yfinance"

    # how long a bar series stays valid, per interval. A 5-minute bar only
    # changes every 5 minutes, so re-downloading it every scan cycle is
    # pure waste - and it was what made a full market scan take 40 seconds.
    TTL_BY_INTERVAL = {
        "1m": 45,
        "2m": 90,
        "5m": 240,
        "15m": 600,
        "30m": 1200,
        "1h": 1800,
        "1d": 3600,
    }

    def __init__(self, cache_ttl: int = 30):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}
        self._last_call = 0.0
        self._min_interval = 0.2

    def _ttl(self, interval: str) -> float:
        ttl = self.TTL_BY_INTERVAL.get(interval)
        if ttl is None:
            return float(self.cache_ttl)
        return max(float(self.cache_ttl), float(ttl))

    def _rate_limit(self):
        now = time.time()
        if now - self._last_call < self._min_interval:
            time.sleep(self._min_interval - (now - self._last_call))
        self._last_call = time.time()

    def _cache_key(self, symbol, interval, lookback_days):
        return (symbol, interval, lookback_days)

    def get_candles(self, symbol, interval="5m", lookback_days=5) -> Optional[pd.DataFrame]:
        key = self._cache_key(symbol, interval, lookback_days)
        cached = self._cache.get(key)
        if cached and (time.time() - cached[1]) < self._ttl(interval):
            return cached[0].copy() if cached[0] is not None else None
        # intraday limits
        period = f"{min(lookback_days, 59)}d" if interval in ("5m", "15m", "30m", "1h") else f"{min(lookback_days, 7)}d"
        if interval == "1m":
            period = "7d"
        self._rate_limit()
        for attempt in range(3):
            try:
                t = yf.Ticker(symbol)
                df = t.history(period=period, interval=interval, auto_adjust=False)
                if df is None or df.empty:
                    self._cache[key] = (None, time.time())
                    return None
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                df = df[["open", "high", "low", "close", "volume"]]
                # tz-aware index -> UTC
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                df.index.name = "ts"
                from trader.core.data_quality import is_usable
                if not is_usable(df):
                    self._cache[key] = (None, time.time())
                    return None
                self._cache[key] = (df, time.time())
                return df.copy()
            except Exception as e:
                log.warning("yfinance error %s attempt %d: %s", symbol, attempt, e)
                time.sleep(1.0 * (attempt + 1))
        self._cache[key] = (None, time.time())
        return None

    def batch_download(self, symbols, interval="5m", lookback_days=5) -> dict:
        result = {}
        # a batch is cached as a whole - the bars only change every interval
        bkey = ("__batch__", interval, lookback_days, tuple(symbols))
        cached = self._cache.get(bkey)
        if cached and (time.time() - cached[1]) < self._ttl(interval):
            return {k: (v.copy() if v is not None else None) for k, v in cached[0].items()}
        try:
            period = f"{min(lookback_days, 59)}d" if interval != "1m" else "7d"
            self._rate_limit()
            data = yf.download(symbols, period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = data.copy()
                    else:
                        df = data[sym].copy() if sym in data.columns.get_level_values(0) else None
                    if df is None or df.empty:
                        result[sym] = None
                        continue
                    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    df = df[["open", "high", "low", "close", "volume"]].dropna()
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    df.index.name = "ts"
                    from trader.core.data_quality import is_usable
                    if not is_usable(df):
                        result[sym] = None
                        continue
                    result[sym] = df
                except Exception as e:
                    log.warning("batch parse error %s: %s", sym, e)
                    result[sym] = None
        except Exception as e:
            log.error("batch download failed: %s", e)
            for sym in symbols:
                result[sym] = None
        self._cache[bkey] = (dict(result), time.time())
        return result
