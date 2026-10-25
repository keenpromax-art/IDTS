from __future__ import annotations

import numpy as np
import pandas as pd

from trader.core.indicators import atr
from trader.core.models import Signal, Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal


class StochasticStrategy(Strategy):
    """Stochastic %K crossing out of an extreme with %D confirmation."""

    name = "stochastic"
    kind = "meanrev"
    label = "Stochastic Reversal"

    def __init__(self, oversold: float = 20, overbought: float = 80,
                 max_adx: float = 30):
        self.oversold = oversold
        self.overbought = overbought
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0 or c.adx > self.max_adx:
            return None
        k, kd = c.val(c.k), c.val(c.d)
        if pd.isna(k) or pd.isna(kd):
            return None
        if k < self.oversold and k > kd:
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason=f"stochastic oversold ({k:.0f}) turning up")
        if k > self.overbought and k < kd:
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.6,
                               reason=f"stochastic overbought ({k:.0f}) turning down")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class CCIStrategy(Strategy):
    """CCI leaving an extreme zone back inside it."""

    name = "cci"
    kind = "meanrev"
    label = "CCI Reversal"

    def __init__(self, oversold: float = -100, overbought: float = 100,
                 max_adx: float = 30):
        self.oversold = oversold
        self.overbought = overbought
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0 or c.adx > self.max_adx:
            return None
        v = c.val(c.cci_s)
        if pd.isna(v):
            return None
        if v < self.oversold and c.bar_up():
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason=f"CCI oversold ({v:.0f}) recovering")
        if v > self.overbought and c.bar_down():
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.6,
                               reason=f"CCI overbought ({v:.0f}) fading")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class ROCStrategy(Strategy):
    """Rate-of-change momentum with a zero-line break."""

    name = "roc"
    kind = "trend"
    label = "ROC Momentum"

    def __init__(self, period: int = 12, threshold: float = 0.5):
        self.period = period
        self.threshold = threshold

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 40:
            return None
        if c.atr <= 0:
            return None
        v = c.val(c.roc_s)
        pv = c.prev_val(c.roc_s)
        if pd.isna(v) or pd.isna(pv):
            return None
        if pv <= 0 and v > self.threshold:
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.55,
                               reason=f"ROC broke above zero ({v:.1f}%)")
        if pv >= 0 and v < -self.threshold:
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.55,
                               reason=f"ROC broke below zero ({v:.1f}%)")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class MFIStrategy(Strategy):
    """Money Flow Index - RSI weighted by volume."""

    name = "mfi"
    kind = "meanrev"
    label = "MFI Reversal"

    def __init__(self, oversold: float = 20, overbought: float = 80,
                 max_adx: float = 30):
        self.oversold = oversold
        self.overbought = overbought
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0 or c.adx > self.max_adx:
            return None
        v = c.val(c.mfi_s)
        if pd.isna(v):
            return None
        if v < self.oversold and c.bar_up():
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason=f"MFI oversold ({v:.0f})")
        if v > self.overbought and c.bar_down():
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.6,
                               reason=f"MFI overbought ({v:.0f})")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class WilliamsRStrategy(Strategy):
    """Williams %R snapping back from an extreme."""

    name = "williams_r"
    kind = "meanrev"
    label = "Williams %R"

    def __init__(self, oversold: float = -80, overbought: float = -20,
                 max_adx: float = 30):
        self.oversold = oversold
        self.overbought = overbought
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0 or c.adx > self.max_adx:
            return None
        v = c.val(c.wr)
        if pd.isna(v):
            return None
        if v < self.oversold and c.bar_up():
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.55,
                               reason=f"Williams %R oversold ({v:.0f})")
        if v > self.overbought and c.bar_down():
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.55,
                               reason=f"Williams %R overbought ({v:.0f})")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class RSIReversalStrategy(Strategy):
    """RSI(14) extremes plus a Bollinger band touch."""

    name = "rsi_reversal"
    kind = "meanrev"
    label = "RSI + Band Reversal"

    def __init__(self, rsi_period: int = 14, oversold: float = 30,
                 overbought: float = 70, max_adx: float = 30):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0 or c.adx > self.max_adx:
            return None
        r = c.val(c.rsi_s)
        lo = c.val(c.bb_low)
        up = c.val(c.bb_up)
        if pd.isna(r) or pd.isna(lo) or pd.isna(up):
            return None
        if r < self.oversold and c.last_low <= lo * 1.002:
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop),
                               confidence=min(1.0, (self.oversold - r) / 20.0 + 0.4),
                               reason="RSI oversold + lower band touch")
        if r > self.overbought and c.last_high >= up * 0.998:
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry),
                               confidence=min(1.0, (r - self.overbought) / 20.0 + 0.4),
                               reason="RSI overbought + upper band touch")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
