from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import pandas as pd

from trader.data.base import DataProvider
from trader.data.yf_provider import YFinanceProvider
from trader.data.angel_provider import AngelProvider
from trader.scanner.resilience import ProviderResilience, RetryPolicy

log = logging.getLogger(__name__)


class ValidatedProvider(DataProvider):
    """Enforce the common candle contract on any provider.

    Composition root wrapper: even a third-party provider that never calls
    ``DataProvider.finalize_candles`` cannot leak malformed frames into the
    indicators when built through :func:`build_provider`.
    """

    def __init__(self, provider: DataProvider, *, interval: str = "5m"):
        self.provider = provider
        self.name = getattr(provider, "name", "validated")
        self.available = getattr(provider, "available", True)
        self.interval = interval

    def _clean(self, df, symbol, interval):
        return self.finalize_candles(df, symbol=symbol, interval=interval)

    def get_candles(self, symbol, interval="5m", lookback_days=5):
        return self._clean(self.provider.get_candles(symbol, interval, lookback_days),
                           symbol, interval)

    def batch_download(self, symbols, interval="5m", lookback_days=5):
        raw = (self.provider.batch_download(symbols, interval, lookback_days)
               if hasattr(self.provider, "batch_download") else
               {s: self.provider.get_candles(s, interval, lookback_days) for s in symbols})
        return {s: self._clean(raw.get(s), s, interval) for s in symbols}

    def health(self):
        h = getattr(self.provider, "health", None)
        return h() if callable(h) else {"name": self.name}


class FallbackProvider(DataProvider):
    """Primary provider with genuine per-symbol fallback.

    The previous implementation never used the primary in ``batch_download``
    (Angel has no native batch API), so every India symbol came from Yahoo.
    Here a primary batch is used when available, and *only* the symbols it
    failed to deliver are retried against the fallback - independently.
    """

    name = "fallback"

    def __init__(self, primary, fallback, *, max_workers: int = 8,
                 primary_resilience: Optional[ProviderResilience] = None,
                 fallback_resilience: Optional[ProviderResilience] = None):
        self.primary = primary
        self.fallback = fallback
        self.name = getattr(primary, "name", "fallback")
        self.available = getattr(primary, "available", True)
        self.max_workers = max(1, int(max_workers))
        rp = RetryPolicy(retries=1, base_delay=0.25, max_delay=2.0)
        self.primary_resilience = primary_resilience or ProviderResilience(
            self.name + "-primary", negative_ttl_seconds=20.0,
            failure_threshold=5, recovery_seconds=30.0, retry=rp)
        self.fallback_resilience = fallback_resilience or ProviderResilience(
            getattr(fallback, "name", "fallback"), negative_ttl_seconds=20.0,
            failure_threshold=5, recovery_seconds=30.0, retry=rp)
        self.fallback_used = 0
        self.primary_used = 0

    @staticmethod
    def _ok(df) -> bool:
        return df is not None and hasattr(df, "empty") and not df.empty

    def _key(self, symbol, interval, lookback_days):
        return f"{symbol}|{interval}|{lookback_days}"

    def get_candles(self, symbol, interval="5m", lookback_days=5):
        key = self._key(symbol, interval, lookback_days)

        def from_primary():
            df = self.primary.get_candles(symbol, interval, lookback_days)
            if not self._ok(df):
                raise ValueError("no primary data")
            return df

        out = self.primary_resilience.fetch(key, from_primary)
        if out.ok:
            self.primary_used += 1
            return self.finalize_candles(out.value, symbol=symbol, interval=interval)

        def from_fallback():
            df = self.fallback.get_candles(symbol, interval, lookback_days)
            if not self._ok(df):
                raise ValueError("no fallback data")
            return df

        fout = self.fallback_resilience.fetch(key, from_fallback)
        if fout.ok:
            self.fallback_used += 1
            return self.finalize_candles(fout.value, symbol=symbol, interval=interval)
        return None

    def batch_download(self, symbols, interval="5m", lookback_days=5):
        symbols = list(symbols)
        if not symbols:
            return {}
        result: Dict[str, Optional[pd.DataFrame]] = {}

        # 1) Ask the primary once when it supports a batch API. This is the
        #    path that was dead before: hasattr is now true for Angel.
        primary_batch: Dict[str, object] = {}
        try:
            if hasattr(self.primary, "batch_download"):
                primary_batch = self.primary.batch_download(symbols, interval, lookback_days) or {}
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futs = {ex.submit(self.primary.get_candles, s, interval, lookback_days): s
                            for s in symbols}
                    for fut in as_completed(futs):
                        try:
                            primary_batch[futs[fut]] = fut.result()
                        except Exception:
                            primary_batch[futs[fut]] = None
        except Exception as e:
            log.warning("primary batch failed: %s", e)
            primary_batch = {}

        # 2) Per-symbol finalization + per-symbol fallback. A single bad symbol
        #    can never invalidate the rest of the batch.
        missing = []
        for s in symbols:
            df = self.finalize_candles(primary_batch.get(s), symbol=s, interval=interval)
            if self._ok(df):
                self.primary_used += 1
                result[s] = df
            else:
                missing.append(s)

        if missing:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self._fallback_one, s, interval, lookback_days): s
                        for s in missing}
                for fut in as_completed(futs):
                    s = futs[fut]
                    try:
                        result[s] = fut.result()
                    except Exception:
                        result[s] = None
        for s in symbols:
            result.setdefault(s, None)
        return result

    def _fallback_one(self, symbol, interval, lookback_days):
        key = self._key(symbol, interval, lookback_days)

        def fn():
            df = self.fallback.get_candles(symbol, interval, lookback_days)
            if not self._ok(df):
                raise ValueError("no fallback data")
            return df

        out = self.fallback_resilience.fetch(key, fn)
        if out.ok:
            self.fallback_used += 1
            return self.finalize_candles(out.value, symbol=symbol, interval=interval)
        return None

    @property
    def health(self):
        return {
            "primary": self.primary_resilience.health,
            "fallback": self.fallback_resilience.health,
            "primary_used": self.primary_used,
            "fallback_used": self.fallback_used,
            "fallback_rate": (self.fallback_used /
                              max(1, self.primary_used + self.fallback_used)),
        }


def build_provider(market: str, cfg: Optional[dict] = None) -> DataProvider:
    """Build the provider stack for one market.

    ``cfg`` is optional for backward compatibility. Each market can be
    configured independently through ``cfg['data'][market]`` /
    ``cfg['data']['default']`` with keys ``primary``, ``use_angel``,
    ``max_workers``, ``negative_cache_seconds`` and ``rate_limit_seconds``.
    """
    dcfg = {}
    if cfg:
        all_cfg = (cfg.get("data") or {})
        dcfg = dict(all_cfg.get("default") or {})
        dcfg.update(all_cfg.get(market) or {})
    yf = YFinanceProvider()
    max_workers = int(dcfg.get("max_workers", 8))
    neg_ttl = float(dcfg.get("negative_cache_seconds", 20.0))
    rate = float(dcfg.get("rate_limit_seconds", 0.0))

    use_angel = dcfg.get("use_angel")
    if market == "India" and use_angel is None:
        use_angel = bool((cfg or {}).get("angel", {}).get("enabled_if_creds", True))

    if use_angel:
        angel = AngelProvider(rate_limit=float(
            dcfg.get("angel_rate_limit_seconds",
                     (cfg or {}).get("angel", {}).get("rate_limit_seconds", 0.34))))
        if getattr(angel, "available", False):
            fb = FallbackProvider(
                angel, yf, max_workers=max_workers,
                primary_resilience=ProviderResilience(
                    "angel", negative_ttl_seconds=neg_ttl,
                    min_interval_seconds=rate,
                    retry=RetryPolicy(retries=1, base_delay=0.4, max_delay=2.0)),
            )
            return ValidatedProvider(fb)
    return ValidatedProvider(yf)
