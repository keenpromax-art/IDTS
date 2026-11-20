from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from trader.scanner.candles import normalize_candles, validate_candles


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

    # -- common boundary validation -------------------------------------
    @staticmethod
    def finalize_candles(df: Optional[pd.DataFrame], *, symbol: str = "",
                         interval: str = "5m", stale_after_seconds: Optional[float] = None,
                         require_completed_last: bool = False) -> Optional[pd.DataFrame]:
        """Normalize and validate every frame leaving a provider.

        Providers must call this before returning. It is the single place where
        the OHLCV/UTC contract is enforced, so Angel, Yahoo, the fallback
        wrapper and any future provider cannot silently diverge.
        """
        norm = normalize_candles(df)
        if norm is None:
            return None
        report = validate_candles(
            norm, symbol=symbol, interval=interval,
            stale_after_seconds=stale_after_seconds,
            require_completed_last=require_completed_last,
        )
        return norm if report.ok else None
