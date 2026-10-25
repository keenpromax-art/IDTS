from __future__ import annotations

import os
from typing import Optional
import pandas as pd

from trader.data.base import DataProvider
from trader.data.yf_provider import YFinanceProvider
from trader.data.angel_provider import AngelProvider


class FallbackProvider(DataProvider):
    name = "fallback"

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.name = getattr(primary, "name", "fallback")
        self.available = getattr(primary, "available", True)

    def get_candles(self, symbol, interval="5m", lookback_days=5):
        df = self.primary.get_candles(symbol, interval, lookback_days)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = self.fallback.get_candles(symbol, interval, lookback_days)
        return df

    def batch_download(self, symbols, interval="5m", lookback_days=5):
        # try primary first
        out = {}
        try:
            primary_res = self.primary.batch_download(symbols, interval, lookback_days) if hasattr(self.primary, "batch_download") else None
        except Exception:
            primary_res = None
        if isinstance(primary_res, dict):
            for s in symbols:
                df = primary_res.get(s)
                if df is None or (hasattr(df, "empty") and df.empty):
                    df = self.fallback.get_candles(s, interval, lookback_days)
                out[s] = df
            return out
        return self.fallback.batch_download(symbols, interval, lookback_days) if hasattr(self.fallback, "batch_download") else {s: self.get_candles(s, interval, lookback_days) for s in symbols}


def build_provider(market: str) -> DataProvider:
    yf = YFinanceProvider()
    if market == "India":
        angel = AngelProvider()
        if angel.available:
            return FallbackProvider(angel, yf)
    return yf
