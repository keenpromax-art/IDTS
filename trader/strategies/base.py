from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from trader.core.models import Signal, Side


class Strategy(ABC):
    """Base class for every signal generator.

    `kind` tells the selector how a strategy behaves so it can be matched to
    the prevailing market regime:

        trend     - rides an existing move (EMA crosses, MACD, Supertrend)
        meanrev   - fades an extreme move back to the mean (RSI, VWAP, bands)
        breakout  - enters when price leaves a range (Donchian, ORB, squeeze)
        volume    - needs participation to confirm (volume breakout, MFI)
    """

    name: str = "base"
    kind: str = "trend"
    label: str = ""

    @abstractmethod
    def generate(self, df: pd.DataFrame, symbol: str = "", market: str = "") -> Optional[Signal]:
        ...

    def generate_from(self, ctx) -> Optional[Signal]:
        """Signal from a pre-computed context.

        The engine builds one Ctx per symbol and shares it with every
        strategy, so indicators are calculated once instead of once per
        strategy. The mini-backtest replays the same context bar by bar with
        its cursor, so `generate` must see the cursor position - which is why
        the shared context is passed through.
        """
        try:
            return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
        except TypeError:
            # a strategy written without the ctx parameter
            return self.generate(ctx.df, ctx.symbol, ctx.market)

    def params(self) -> dict:
        return {}

    def display_name(self) -> str:
        return self.label or self.name
