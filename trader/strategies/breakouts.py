from __future__ import annotations

import numpy as np
import pandas as pd

from trader.core.indicators import atr, highest_high, lowest_low
from trader.core.np_indicators import rolling_max_np, rolling_min_np
from trader.core.models import Signal, Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal


class DonchianBreakoutStrategy(Strategy):
    """Price breaks the 20-bar Donchian channel."""

    name = "donchian"
    kind = "breakout"
    label = "Donchian 20 Breakout"

    def __init__(self, period: int = 20, min_bars: int = 25):
        self.period = period
        self.min_bars = min_bars

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.min_bars:
            return None
        if c.atr <= 0:
            return None
        hi = c.val(c.donch_hi_prev)
        lo = c.val(c.donch_lo_prev)
        if pd.isna(hi) or pd.isna(lo):
            return None
        if c.px > hi:
            entry = c.px
            stop = max(lo, entry - 1.5 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.6,
                               reason=f"broke {self.period}-bar Donchian high")
        if c.px < lo:
            entry = c.px
            stop = min(hi, entry + 1.5 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.6,
                               reason=f"broke {self.period}-bar Donchian low")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class KeltnerBreakoutStrategy(Strategy):
    """Price closes outside the Keltner channel built on EMA20 +/- 1.5 ATR."""

    name = "keltner"
    kind = "breakout"
    label = "Keltner Breakout"

    def __init__(self, period: int = 20, mult: float = 1.5):
        self.period = period
        self.mult = mult

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0:
            return None
        up, lo = c.val(c.kelt_hi_prev), c.val(c.kelt_lo_prev)
        if pd.isna(up) or pd.isna(lo):
            return None
        if c.px > up:
            entry = c.px
            stop = max(lo, entry - 1.5 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.6,
                               reason="closed above Keltner channel")
        if c.px < lo:
            entry = c.px
            stop = min(up, entry + 1.5 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.6,
                               reason="closed below Keltner channel")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class BollingerExpansionStrategy(Strategy):
    """Band expansion breakout - bands widen after a contraction."""

    name = "bb_expansion"
    kind = "breakout"
    label = "BB Expansion"

    def __init__(self, period: int = 20, std: float = 2.0, squeeze_lookback: int = 20):
        self.period = period
        self.std = std
        self.squeeze_lookback = squeeze_lookback

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.period + 5:
            return None
        if c.atr <= 0:
            return None
        up, lo = c.val(c.bb_up_prev), c.val(c.bb_low_prev)
        if pd.isna(up) or pd.isna(lo):
            return None
        bw = c.band_width()
        if pd.isna(bw):
            return None
        prior = (c.bb_up - c.bb_low) / c.bb_mid * 100
        prior = prior[~np.isnan(prior)]
        if prior.size == 0:
            return None
        avg_bw = float(prior[-self.squeeze_lookback:].mean())
        if avg_bw <= 0:
            return None
        expanding = bw > avg_bw * 1.15
        if not expanding:
            return None
        if c.px > up:
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason="Bollinger bands expanding up")
        if c.px < lo:
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.6,
                               reason="Bollinger bands expanding down")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class VolumeBreakoutStrategy(Strategy):
    """A range breakout backed by a spike in volume."""

    name = "volume_breakout"
    kind = "volume"
    label = "Volume Breakout"

    def __init__(self, period: int = 20, vol_mult: float = 1.5):
        self.period = period
        self.vol_mult = vol_mult

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0:
            return None
        if c.vol_ratio < self.vol_mult:
            return None
        hi = c.val(c.donch_hi_prev)
        lo = c.val(c.donch_lo_prev)
        if pd.isna(hi) or pd.isna(lo):
            return None
        if c.px > hi:
            entry = c.px
            stop = max(lo, entry - 1.5 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk,
                               confidence=min(1.0, 0.5 + c.vol_ratio / 8.0),
                               reason=f"range break on {c.vol_ratio:.1f}x volume")
        if c.px < lo:
            entry = c.px
            stop = min(hi, entry + 1.5 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk,
                               confidence=min(1.0, 0.5 + c.vol_ratio / 8.0),
                               reason=f"range break on {c.vol_ratio:.1f}x volume")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class HighLowBreakoutStrategy(Strategy):
    """Breaks the highest high / lowest low of a longer lookback window."""

    name = "range_extreme"
    kind = "breakout"
    label = "Range Extreme Break"

    def __init__(self, period: int = 60):
        self.period = period

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.period + 5:
            return None
        if c.atr <= 0:
            return None
        hi = c.val(c.high60_prev) if self.period == 60 else c.val(rolling_max_np(c.high, self.period))
        lo = c.val(c.low60_prev) if self.period == 60 else c.val(rolling_min_np(c.low, self.period))
        if pd.isna(hi) or pd.isna(lo):
            return None
        if c.px > hi:
            entry = c.px
            stop = entry - 1.75 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.65,
                               reason=f"broke {self.period}-bar high")
        if c.px < lo:
            entry = c.px
            stop = entry + 1.75 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.65,
                               reason=f"broke {self.period}-bar low")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class PSARStrategy(Strategy):
    """Parabolic SAR flip."""

    name = "psar"
    kind = "trend"
    label = "Parabolic SAR"

    def __init__(self, step: float = 0.02, max_step: float = 0.2):
        self.step = step
        self.max_step = max_step

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        if c.atr <= 0:
            return None
        sar_now = c.val(c.psar_s)
        sar_prev = c.prev_val(c.psar_s)
        if pd.isna(sar_now) or pd.isna(sar_prev):
            return None
        if c.px > sar_now and c.prev_px <= sar_prev:
            # price crossed up through the SAR
            entry = c.px
            stop = sar_now
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.6,
                               reason="Parabolic SAR flip bullish")
        if c.px < sar_now and c.prev_px >= sar_prev:
            entry = c.px
            stop = sar_now
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.6,
                               reason="Parabolic SAR flip bearish")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class ADXDirectionalStrategy(Strategy):
    """+DI crossing -DI while ADX says the move has strength."""

    name = "adx_di"
    kind = "trend"
    label = "ADX +DI/-DI"

    def __init__(self, adx_min: float = 22):
        self.adx_min = adx_min

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 35:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.atr <= 0 or c.adx < self.adx_min:
            return None
        plus, minus = c.plus_di, c.minus_di
        if c.crossed_up(plus, minus):
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop),
                               confidence=min(1.0, c.adx / 40.0),
                               reason=f"+DI crossed -DI (ADX {c.adx:.0f})")
        if c.crossed_down(plus, minus):
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry),
                               confidence=min(1.0, c.adx / 40.0),
                               reason=f"-DI crossed +DI (ADX {c.adx:.0f})")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class IchimokuStrategy(Strategy):
    """Price crossing the Ichimoku cloud with a Tenkan/Kijun cross."""

    name = "ichimoku"
    kind = "trend"
    label = "Ichimoku Cloud"

    def __init__(self, min_bars: int = 60):
        self.min_bars = min_bars

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.min_bars:
            return None
        if c.atr <= 0:
            return None
        sa, sb = c.val(c.senkou_a), c.val(c.senkou_b)
        if pd.isna(sa) or pd.isna(sb):
            return None
        top = max(sa, sb)
        bottom = min(sa, sb)
        if c.crossed_up(c.tenkan, c.kijun) and c.px > top:
            entry = c.px
            stop = max(bottom, entry - 1.5 * c.atr)
            risk = entry - stop
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * risk, confidence=0.7,
                               reason="Ichimoku TK cross above cloud")
        if c.crossed_down(c.tenkan, c.kijun) and c.px < bottom:
            entry = c.px
            stop = min(top, entry + 1.5 * c.atr)
            risk = stop - entry
            if risk <= 0:
                return None
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * risk, confidence=0.7,
                               reason="Ichimoku TK cross below cloud")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
class TripleScreenStrategy(Strategy):
    """Triple Screen: higher-timeframe trend, then ADX strength, then a trigger."""

    name = "triple_screen"
    kind = "trend"
    label = "Triple Screen"

    def __init__(self, trend_ema: int = 50, adx_min: float = 20):
        self.trend_ema = trend_ema
        self.adx_min = adx_min

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 60:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.atr <= 0 or c.adx < self.adx_min:
            return None
        from trader.core.indicators import ema
        trend = c._ema(self.trend_ema)
        if c.val(trend) > c.val(c.ema20) and c.crossed_up(c.ema20, c.ema9):
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.65,
                               reason="triple screen long")
        if c.val(trend) < c.val(c.ema20) and c.crossed_down(c.ema20, c.ema9):
            entry = c.px
            stop = entry + 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.SHORT, entry, stop,
                               entry - 2.0 * (stop - entry), confidence=0.65,
                               reason="triple screen short")
        return None

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
