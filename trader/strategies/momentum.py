from __future__ import annotations

import numpy as np
import pandas as pd

from trader.core.indicators import atr, ema
from trader.core.models import Signal, Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal


class VWAPStrategy(Strategy):
    """VWAP trend-pullback and VWAP mean-reversion in one strategy."""

    name = "vwap"
    kind = "trend"
    label = "VWAP"

    def __init__(self, trend_mode: bool = True, reversion_mode: bool = True,
                 dev_atr: float = 1.5, target_rr: float = 2.0):
        self.trend_mode = trend_mode
        self.reversion_mode = reversion_mode
        self.dev_atr = dev_atr
        self.target_rr = target_rr

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0:
            return None
        dev = c.dev_atr()
        prev_low = c.prev_low
        prev_high = c.prev_high
        if self.trend_mode:
            if c.px > c.vwap and c.val(c.ema9) > c.val(c.ema21) and prev_low <= c.vwap * 1.002:
                entry = c.px
                stop = min(c.vwap - c.atr, entry - 1.0 * c.atr)
                risk = entry - stop
                if risk > 0:
                    return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                                       entry + self.target_rr * risk, confidence=0.6,
                                       reason="VWAP trend pullback long")
            if c.px < c.vwap and c.val(c.ema9) < c.val(c.ema21) and prev_high >= c.vwap * 0.998:
                entry = c.px
                stop = max(c.vwap + c.atr, entry + 1.0 * c.atr)
                risk = stop - entry
                if risk > 0:
                    return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                                       entry - self.target_rr * risk, confidence=0.6,
                                       reason="VWAP trend pullback short")
        if self.reversion_mode:
            if dev < -self.dev_atr:
                entry = c.px
                stop = entry - 1.5 * c.atr
                risk = entry - stop
                if risk > 0:
                    return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                                       entry + 1.5 * risk, confidence=0.55,
                                       reason="VWAP mean reversion long")
            if dev > self.dev_atr:
                entry = c.px
                stop = entry + 1.5 * c.atr
                risk = stop - entry
                if risk > 0:
                    return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                                       entry - 1.5 * risk, confidence=0.55,
                                       reason="VWAP mean reversion short")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class VWAPBandStrategy(Strategy):
    """Standard-deviation bands around VWAP - a reversion band, not a trend one."""

    name = "vwap_band"
    kind = "meanrev"
    label = "VWAP Deviation Band"

    def __init__(self, dev_atr: float = 2.0, band_atr: float = 0.75, max_adx: float = 30):
        self.dev_atr = dev_atr
        self.band_atr = band_atr
        self.max_adx = max_adx

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0:
            return None
        # fading a stretched VWAP only works in a calm market
        if c.adx > self.max_adx:
            return None
        dev = c.dev_atr()
        band = self.band_atr * c.atr
        if dev <= -self.dev_atr and c.px - c.vwap <= -band:
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               c.vwap, confidence=0.6,
                               reason=f"{dev:.1f} ATR below VWAP")
        if dev >= self.dev_atr and c.px - c.vwap >= band:
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               c.vwap, confidence=0.6,
                               reason=f"{dev:.1f} ATR above VWAP")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class ORBStrategy(Strategy):
    """Opening Range Breakout over the first bars of the session."""

    name = "orb"
    kind = "breakout"
    label = "Opening Range Breakout"

    def __init__(self, range_bars: int = 3, atr_period: int = 14):
        self.range_bars = range_bars
        self.atr_period = atr_period

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.range_bars + 5:
            return None
        if c.atr <= 0:
            return None
        # opening range = the first bars of the session the cursor is in
        day_idx = c.session_bars()
        if len(day_idx) < self.range_bars + 1:
            return None
        range_bars_idx = day_idx[: self.range_bars]
        range_high = float(np.max(c.high[range_bars_idx]))
        range_low = float(np.min(c.low[range_bars_idx]))
        if c.px > range_high:
            entry = c.px
            stop = range_low
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 1.5 * risk,
                               confidence=min(1.0, (entry - range_high) / c.atr + 0.5),
                               reason="ORB breakout above range")
        if c.px < range_low:
            entry = c.px
            stop = range_high
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 1.5 * risk,
                               confidence=min(1.0, (range_low - entry) / c.atr + 0.5),
                               reason="ORB breakdown below range")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class SupertrendStrategy(Strategy):
    """Supertrend(10,3) direction flip."""

    name = "supertrend"
    kind = "trend"
    label = "Supertrend"

    def __init__(self, period: int = 10, mult: float = 3.0):
        self.period = period
        self.mult = mult

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.period + 5:
            return None
        if c.atr <= 0:
            return None
        d_now = int(c.st_dir[-1])
        d_prev = int(c.st_dir[-2]) if len(c.st_dir) > 1 else d_now
        st_now = c.val(c.st_line)
        if d_prev == -1 and d_now == 1:
            entry = c.px
            stop = st_now
            risk = entry - stop
            if risk <= 0:
                stop = entry - 1.5 * c.atr
                risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.6,
                               reason="Supertrend flipped bullish")
        if d_prev == 1 and d_now == -1:
            entry = c.px
            stop = st_now
            risk = stop - entry
            if risk <= 0:
                stop = entry + 1.5 * c.atr
                risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.6,
                               reason="Supertrend flipped bearish")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class BBSqueezeStrategy(Strategy):
    """Bollinger squeeze release."""

    name = "bb_squeeze"
    kind = "breakout"
    label = "BB Squeeze"

    def __init__(self, period: int = 20, std: float = 2.0, squeeze_lookback: int = 20):
        self.period = period
        self.std = std
        self.squeeze_lookback = squeeze_lookback

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.period + self.squeeze_lookback + 5:
            return None
        if c.atr <= 0:
            return None
        up, lo = c.val(c.bb_up), c.val(c.bb_low)
        if pd.isna(up) or pd.isna(lo):
            return None
        prior = (c.bb_up - c.bb_low) / c.bb_mid * 100
        prior = prior[~np.isnan(prior)]
        if prior.size == 0:
            return None
        min_bw = float(prior[-self.squeeze_lookback:].min())
        if min_bw <= 0 or c.band_width() > min_bw * 1.15:
            return None
        if c.px > up:
            entry = c.px
            stop = lo
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.6,
                               reason="BB squeeze breakout up")
        if c.px < lo:
            entry = c.px
            stop = up
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.6,
                               reason="BB squeeze breakout down")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class PivotBreakoutStrategy(Strategy):
    """Break of the previous session's high/low or its VWAP pivot."""

    name = "pivot"
    kind = "breakout"
    label = "Pivot Break"

    def __init__(self, lookback: int = 40):
        self.lookback = lookback

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0:
            return None
        # the session before the one the cursor is in
        prev_idx = c.prev_session_bars()
        if len(prev_idx) == 0:
            return None
        prev_high = float(np.max(c.high[prev_idx]))
        prev_low = float(np.min(c.low[prev_idx]))
        prev_close = float(c.close[prev_idx][-1])
        pivot = (prev_high + prev_low + prev_close) / 3.0
        if c.px > max(prev_high, pivot):
            entry = c.px
            stop = max(pivot, entry - 1.5 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.6,
                               reason="broke previous session high")
        if c.px < min(prev_low, pivot):
            entry = c.px
            stop = min(pivot, entry + 1.5 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.6,
                               reason="broke previous session low")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
