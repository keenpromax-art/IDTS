from __future__ import annotations

import numpy as np
import pandas as pd

from trader.core.indicators import atr, ema, macd
from trader.core.models import Signal, Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal


class MACDMomentumStrategy(Strategy):
    """MACD signal-line cross confirmed by volume."""

    name = "macd_momentum"
    kind = "trend"
    label = "MACD Cross + Volume"

    def __init__(self, vol_mult: float = 0.8):
        self.vol_mult = vol_mult

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 40:
            return None
        if c.atr <= 0:
            return None
        vol_ok = c.vol_ratio >= self.vol_mult
        if c.crossed_up(c.macd_line, c.macd_signal) and c.val(c.macd_hist) > 0 and vol_ok:
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason="MACD bullish cross + volume")
        if c.crossed_down(c.macd_line, c.macd_signal) and c.val(c.macd_hist) < 0 and vol_ok:
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.6,
                               reason="MACD bearish cross + volume")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class MACDZeroCrossStrategy(Strategy):
    """MACD line crossing the zero line - a stronger trend confirmation."""

    name = "macd_zero"
    kind = "trend"
    label = "MACD Zero Cross"

    def __init__(self):
        pass

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 40:
            return None
        if c.atr <= 0:
            return None
        if c.crossed_up(c.macd_line, c.macd_line * 0):
            pass
        # zero line = constant zero array
        zero = np.zeros(len(df))
        if c.crossed_up(c.macd_line, zero):
            entry = c.px
            stop = entry - 1.75 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.65,
                               reason="MACD crossed above zero")
        if c.crossed_down(c.macd_line, zero):
            entry = c.px
            stop = entry + 1.75 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.65,
                               reason="MACD crossed below zero")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class MACDDivergenceStrategy(Strategy):
    """Price makes a new extreme but MACD does not - fade or follow the divergence."""

    name = "macd_divergence"
    kind = "meanrev"
    label = "MACD Divergence"

    def __init__(self, lookback: int = 20, max_adx: float = 30):
        self.lookback = lookback
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 40:
            return None
        if c.atr <= 0:
            return None
        # a divergence is a mean-reversion idea - do not take it in a trend
        if c.adx > self.max_adx:
            return None
        n = len(c.close)
        if n < self.lookback + 5:
            return None
        window_lo = float(np.nanmin(c.low[-(self.lookback + 1):-1]))
        window_hi = float(np.nanmax(c.high[-(self.lookback + 1):-1]))
        macd_window = c.macd_line[-(self.lookback + 1):-1]
        macd_window = macd_window[~np.isnan(macd_window)]
        if len(macd_window) < 3:
            return None
        # bullish divergence: price lower low, MACD higher low
        if c.px < window_lo and c.val(c.macd_line) > float(macd_window[0]):
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason="bullish MACD divergence")
        if c.px > window_hi and c.val(c.macd_line) < float(macd_window[0]):
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.6,
                               reason="bearish MACD divergence")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
