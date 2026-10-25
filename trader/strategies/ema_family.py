from __future__ import annotations

import numpy as np
import pandas as pd

from trader.core.indicators import adx, atr, ema
from trader.core.models import Signal, Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal


class EMACrossStrategy(Strategy):
    """Classic 9/21 EMA crossover with an ADX trend filter."""

    name = "ema_cross"
    kind = "trend"
    label = "EMA 9/21 Cross"

    def __init__(self, fast: int = 9, slow: int = 21, adx_period: int = 14, adx_min: float = 20):
        self.fast = fast
        self.slow = slow
        self.adx_period = adx_period
        self.adx_min = adx_min

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.slow + 5:
            return None
        if c.atr <= 0:
            return None
        if c.adx < self.adx_min:
            return None
        fast = c._ema(self.fast)
        slow = c._ema(self.slow)
        if c.crossed_up(fast, slow):
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop),
                               confidence=min(1.0, c.adx / 40.0),
                               reason=f"EMA {self.fast}/{self.slow} bullish cross (ADX {c.adx:.0f})")
        if c.crossed_down(fast, slow):
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry),
                               confidence=min(1.0, c.adx / 40.0),
                               reason=f"EMA {self.fast}/{self.slow} bearish cross (ADX {c.adx:.0f})")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class EMA2050Strategy(Strategy):
    """20/50 EMA moving-average cross (the medium-term MVA pair)."""

    name = "ema_20_50"
    kind = "trend"
    label = "EMA 20/50 Cross"

    def __init__(self, fast: int = 20, slow: int = 50, adx_min: float = 18):
        self.fast = fast
        self.slow = slow
        self.adx_min = adx_min

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.slow + 5:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.atr <= 0 or c.adx < self.adx_min:
            return None
        fast = c._ema(self.fast)
        slow = c._ema(self.slow)
        if c.crossed_up(fast, slow):
            entry = c.px
            stop = min(c.val(fast), entry - 1.5 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk,
                               confidence=min(1.0, 0.5 + c.adx / 60.0),
                               reason=f"EMA {self.fast}/{self.slow} golden cross")
        if c.crossed_down(fast, slow):
            entry = c.px
            stop = max(c.val(slow), entry + 1.5 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk,
                               confidence=min(1.0, 0.5 + c.adx / 60.0),
                               reason=f"EMA {self.fast}/{self.slow} death cross")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class EMA50200Strategy(Strategy):
    """50/200 EMA golden- or death-cross on the long-term trend."""

    name = "ema_50_200"
    kind = "trend"
    label = "EMA 50/200 Cross"

    def __init__(self, fast: int = 50, slow: int = 200, min_bars: int = 210):
        self.fast = fast
        self.slow = slow
        self.min_bars = min_bars

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.min_bars:
            return None
        if c.atr <= 0:
            return None
        fast = c._ema(self.fast)
        slow = c._ema(self.slow)
        if c.crossed_up(fast, slow):
            entry = c.px
            stop = min(c.val(fast), entry - 2.0 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 3.0 * risk, confidence=0.75,
                               reason=f"EMA {self.fast}/{self.slow} golden cross")
        if c.crossed_down(fast, slow):
            entry = c.px
            stop = max(c.val(fast), entry + 2.0 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 3.0 * risk, confidence=0.75,
                               reason=f"EMA {self.fast}/{self.slow} death cross")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class EMA821Strategy(Strategy):
    """8/21 EMA cross - a faster momentum read than 9/21."""

    name = "ema_8_21"
    kind = "trend"
    label = "EMA 8/21 Cross"

    def __init__(self, fast: int = 8, slow: int = 21):
        self.fast = fast
        self.slow = slow

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.slow + 5:
            return None
        if c.atr <= 0:
            return None
        fast = c._ema(self.fast)
        slow = c._ema(self.slow)
        if c.crossed_up(fast, slow):
            entry = c.px
            stop = entry - 1.25 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 1.75 * (entry - stop), confidence=0.55,
                               reason=f"EMA {self.fast}/{self.slow} bullish cross")
        if c.crossed_down(fast, slow):
            entry = c.px
            stop = entry + 1.25 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 1.75 * (stop - entry), confidence=0.55,
                               reason=f"EMA {self.fast}/{self.slow} bearish cross")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class EMATrendStackStrategy(Strategy):
    """Enters with the stack EMA20 > EMA50 > EMA200 when price pulls back to EMA20."""

    name = "ema_stack"
    kind = "trend"
    label = "EMA Stack Pullback"

    def __init__(self, pullback_atr: float = 0.35):
        self.pullback_atr = pullback_atr

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 205:
            return None
        if c.atr <= 0:
            return None
        e20, e50, e200 = c.val(c.ema20), c.val(c.ema50), c.val(c.ema200)
        if any(pd.isna(x) for x in (e20, e50, e200)):
            return None
        if e20 > e50 > e200:
            # price pulled back to within pullback_atr of EMA20 and turned up
            if abs(c.px - e20) <= self.pullback_atr * c.atr and c.bar_up():
                entry = c.px
                stop = min(e50, entry - 1.5 * c.atr)
                risk = entry - stop
                if risk <= 0:
                    return None
                return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                                   entry + 2.5 * risk, confidence=0.7,
                                   reason="EMA20>EMA50>EMA200 pullback long")
        if e20 < e50 < e200:
            if abs(c.px - e20) <= self.pullback_atr * c.atr and c.bar_down():
                entry = c.px
                stop = max(e50, entry + 1.5 * c.atr)
                risk = stop - entry
                if risk <= 0:
                    return None
                return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                                   entry - 2.5 * risk, confidence=0.7,
                                   reason="EMA20<EMA50<EMA200 pullback short")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
