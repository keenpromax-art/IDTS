from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def get_candles(self, symbol: str, interval: str = "5m", lookback_days: int = 5) -> Optional[pd.DataFrame]:
        ...

    def batch_download(self, symbols, interval="5m", lookback_days=5) -> dict:
        out = {}
        for s in symbols:
            out[s] = self.get_candles(s, interval, lookback_days)
        return out

    def get_latest_price(self, symbol: str) -> Optional[float]:
        df = self.get_candles(symbol, "5m", 1)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])
