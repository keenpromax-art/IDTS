from __future__ import annotations

"""Concrete rule definitions.

Each rule names the conditions that must line up. `required` conditions gate
the trade; `optional` ones only add confidence.
"""

import numpy as np
import pandas as pd

from trader.strategies.context import Ctx
from trader.strategies.rule import Condition, Rule


def _above(a, b) -> bool:
    return a > b


def _below(a, b) -> bool:
    return a < b


# --- condition factories -------------------------------------------------

def price_above_ema(period: int, text: str = None):
    def test(ctx: Ctx) -> bool:
        e = {20: ctx.ema20, 21: ctx.ema21, 50: ctx.ema50, 200: ctx.ema200}.get(period)
        if e is None:
            return False
        v = ctx.val(e)
        return (not pd.isna(v)) and ctx.px > v
    return Condition("px_ema%d" % period, text or f"price above EMA{period}", test)


def price_below_ema(period: int, text: str = None):
    def test(ctx: Ctx) -> bool:
        e = {20: ctx.ema20, 21: ctx.ema21, 50: ctx.ema50, 200: ctx.ema200}.get(period)
        if e is None:
            return False
        v = ctx.val(e)
        return (not pd.isna(v)) and ctx.px < v
    return Condition("px_ema%d" % period, text or f"price below EMA{period}", test)


def ema_stack_bullish(text="EMA20 > EMA50 > EMA200"):
    def test(ctx: Ctx) -> bool:
        a, b, d = ctx.val(ctx.ema20), ctx.val(ctx.ema50), ctx.val(ctx.ema200)
        if any(pd.isna(x) for x in (a, b, d)):
            return False
        return a > b > d
    return Condition("stack_bull", text, test)


def ema_stack_bearish(text="EMA20 < EMA50 < EMA200"):
    def test(ctx: Ctx) -> bool:
        a, b, d = ctx.val(ctx.ema20), ctx.val(ctx.ema50), ctx.val(ctx.ema200)
        if any(pd.isna(x) for x in (a, b, d)):
            return False
        return a < b < d
    return Condition("stack_bear", text, test)


def adx_above(threshold: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        return ctx.adx >= threshold
    return Condition("adx", text or f"ADX >= {threshold:.0f}", test)


def adx_below(threshold: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        return ctx.adx < threshold
    return Condition("adx", text or f"ADX < {threshold:.0f}", test)


def adx_rising(text="ADX rising"):
    def test(ctx: Ctx) -> bool:
        if len(ctx.adx_s) < 3:
            return False
        v = ctx.val(ctx.adx_s)
        p = ctx.prev_val(ctx.adx_s)
        return (not pd.isna(v)) and (not pd.isna(p)) and v > p
    return Condition("adx_rising", text, test)


def rsi_between(lo: float, hi: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.rsi_s)
        return (not pd.isna(v)) and lo <= v <= hi
    return Condition("rsi_band", text or f"RSI {lo:.0f}-{hi:.0f}", test)


def rsi_below(level: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.rsi_s)
        return (not pd.isna(v)) and v < level
    return Condition("rsi_lo", text or f"RSI < {level:.0f}", test)


def rsi_above(level: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.rsi_s)
        return (not pd.isna(v)) and v > level
    return Condition("rsi_hi", text or f"RSI > {level:.0f}", test)


def macd_hist_above(level: float = 0.0, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.macd_hist)
        return (not pd.isna(v)) and v > level
    return Condition("macd_hist", text or "MACD histogram positive", test)


def macd_hist_below(level: float = 0.0, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.macd_hist)
        return (not pd.isna(v)) and v < level
    return Condition("macd_hist", text or "MACD histogram negative", test)


def macd_cross_up(text="MACD cross up"):
    def test(ctx: Ctx) -> bool:
        return ctx.crossed_up(ctx.macd_line, ctx.macd_signal)
    return Condition("macd_cross", text, test)


def macd_cross_down(text="MACD cross down"):
    def test(ctx: Ctx) -> bool:
        return ctx.crossed_down(ctx.macd_line, ctx.macd_signal)
    return Condition("macd_cross", text, test)


def volume_above(mult: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        return ctx.vol_ratio >= mult
    return Condition("vol", text or f"volume >= {mult:.1f}x avg", test)


def volume_below(mult: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        return ctx.vol_ratio <= mult
    return Condition("vol", text or f"volume <= {mult:.1f}x avg", test)


def above_vwap(text="price above VWAP"):
    def test(ctx: Ctx) -> bool:
        return ctx.px > ctx.vwap
    return Condition("vwap", text, test)


def below_vwap(text="price below VWAP"):
    def test(ctx: Ctx) -> bool:
        return ctx.px < ctx.vwap
    return Condition("vwap", text, test)


def vwap_dev_below(atr_mult: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        return ctx.dev_atr() <= -atr_mult
    return Condition("vwap_dev", text or f"price {atr_mult:.1f} ATR below VWAP", test)


def vwap_dev_above(atr_mult: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        return ctx.dev_atr() >= atr_mult
    return Condition("vwap_dev", text or f"price {atr_mult:.1f} ATR above VWAP", test)


def donchian_break_up(period: int = 20, text: str = None):
    def test(ctx: Ctx) -> bool:
        # compare against the range that existed BEFORE this bar
        v = ctx.val(ctx.donch_hi_prev)
        return (not pd.isna(v)) and ctx.px > v
    return Condition("donch", text or f"broke {period}-bar high", test)


def donchian_break_down(period: int = 20, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.donch_lo_prev)
        return (not pd.isna(v)) and ctx.px < v
    return Condition("donch", text or f"broke {period}-bar low", test)


def supertrend_bull(text="Supertrend bullish"):
    def test(ctx: Ctx) -> bool:
        return int(ctx.st_dir.iloc[-1]) == 1
    return Condition("st_bull", text, test)


def supertrend_bear(text="Supertrend bearish"):
    def test(ctx: Ctx) -> bool:
        return int(ctx.st_dir.iloc[-1]) == -1
    return Condition("st_bear", text, test)


def bar_up(text="last bar closed up"):
    def test(ctx: Ctx) -> bool:
        return ctx.close.iloc[-1] > ctx.close.iloc[-2]
    return Condition("bar", text, test)


def bar_down(text="last bar closed down"):
    def test(ctx: Ctx) -> bool:
        return ctx.close.iloc[-1] < ctx.close.iloc[-2]
    return Condition("bar", text, test)


def near_ema(period: int, atr_mult: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        e = {20: ctx.ema20, 21: ctx.ema21, 50: ctx.ema50}.get(period)
        if e is None or ctx.atr <= 0:
            return False
        v = ctx.val(e)
        return (not pd.isna(v)) and abs(ctx.px - v) <= atr_mult * ctx.atr
    return Condition("near_ema%d" % period, text or f"price within {atr_mult:.1f} ATR of EMA{period}", test)


def bb_upper_break(text="close above upper band"):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.bb_up_prev)
        return (not pd.isna(v)) and ctx.px > v
    return Condition("bb", text, test)


def bb_lower_break(text="close below lower band"):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.bb_low_prev)
        return (not pd.isna(v)) and ctx.px < v
    return Condition("bb", text, test)


def band_squeeze(lookback: int = 20, mult: float = 0.8, text: str = None):
    """True when the bars BEFORE this one were compressed (a squeeze)."""
    def test(ctx: Ctx) -> bool:
        n = len(ctx.bb_mid)
        if n < lookback + 30:
            return False
        prior = (ctx.bb_up - ctx.bb_low) / ctx.bb_mid * 100
        prior = prior[~np.isnan(prior)]
        if prior.size < lookback + 10:
            return False
        recent = prior[-lookback - 1:-1]
        base = prior[-lookback - 30:-lookback - 1]
        if recent.size == 0 or base.size == 0:
            return False
        avg = float(base.mean())
        if avg <= 0:
            return False
        return float(recent.min()) <= avg * mult
    return Condition("squeeze", text or "Bollinger squeeze before this bar", test)


def obv_rising(lookback: int = 5, text="OBV rising"):
    def test(ctx: Ctx) -> bool:
        if len(ctx.obv_s) < lookback + 1:
            return False
        v = float(ctx.obv_s.iloc[-1])
        p = float(ctx.obv_s.iloc[-lookback - 1])
        return v > p
    return Condition("obv_up", text, test)


def obv_falling(lookback: int = 5, text="OBV falling"):
    def test(ctx: Ctx) -> bool:
        if len(ctx.obv_s) < lookback + 1:
            return False
        v = float(ctx.obv_s.iloc[-1])
        p = float(ctx.obv_s.iloc[-lookback - 1])
        return v < p
    return Condition("obv_dn", text, test)


def mfi_above(level: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.mfi_s)
        return (not pd.isna(v)) and v > level
    return Condition("mfi", text or f"MFI > {level:.0f}", test)


def mfi_below(level: float, text: str = None):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.mfi_s)
        return (not pd.isna(v)) and v < level
    return Condition("mfi", text or f"MFI < {level:.0f}", test)


def keltner_break_up(text="close above Keltner upper"):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.kelt_hi_prev)
        return (not pd.isna(v)) and ctx.px > v
    return Condition("kelt_up", text, test)


def keltner_break_down(text="close below Keltner lower"):
    def test(ctx: Ctx) -> bool:
        v = ctx.val(ctx.kelt_lo_prev)
        return (not pd.isna(v)) and ctx.px < v
    return Condition("kelt_dn", text, test)


# --- the rule set --------------------------------------------------------

def build_rules() -> list:
    """Return every rule the system knows about."""
    return [
        Rule(
            name="rule_trend_stack",
            label="Trend Stack Pullback",
            kind="trend",
            side="long",
            stop_atr=1.5, target_rr=2.5, stop_ref="ema",
            conditions=[
                Condition("stack", "EMA20 > EMA50 > EMA200", ema_stack_bullish().test),
                Condition("px50", "price above EMA50", price_above_ema(50).test),
                Condition("adx", "ADX >= 22", adx_above(22).test),
                Condition("near20", "pullback to EMA20", near_ema(20, 0.4).test),
                Condition("bar", "bar closed up", bar_up().test, required=False),
                Condition("macd", "MACD histogram positive", macd_hist_above().test, required=False),
            ],
        ),
        Rule(
            name="rule_trend_stack_short",
            label="Trend Stack Pullback Short",
            kind="trend",
            side="short",
            stop_atr=1.5, target_rr=2.5, stop_ref="ema",
            conditions=[
                Condition("stack", "EMA20 < EMA50 < EMA200", ema_stack_bearish().test),
                Condition("px50", "price below EMA50", price_below_ema(50).test),
                Condition("adx", "ADX >= 22", adx_above(22).test),
                Condition("near20", "pullback to EMA20", near_ema(20, 0.4).test),
                Condition("bar", "bar closed down", bar_down().test, required=False),
                Condition("macd", "MACD histogram negative", macd_hist_below().test, required=False),
            ],
        ),
        Rule(
            name="rule_golden_cross",
            label="Golden Cross Continuation",
            kind="trend",
            side="long",
            stop_atr=1.75, target_rr=3.0,
            conditions=[
                Condition("ema50_200", "EMA50 crossing above EMA200", _ema50_200_cross_up),
                Condition("px50", "price above EMA50", price_above_ema(50).test),
                Condition("adx", "ADX >= 20", adx_above(20).test),
                Condition("vol", "volume >= 1.2x avg", volume_above(1.2).test, required=False),
            ],
        ),
        Rule(
            name="rule_macd_ema",
            label="MACD + EMA Confirmation",
            kind="trend",
            side="long",
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("macd", "MACD bullish cross", macd_cross_up().test),
                Condition("ema", "price above EMA20", price_above_ema(20).test),
                Condition("adx", "ADX >= 20", adx_above(20).test),
                Condition("vol", "volume >= 1.0x avg", volume_above(1.0).test, required=False),
            ],
        ),
        Rule(
            name="rule_macd_ema_short",
            label="MACD + EMA Confirmation Short",
            kind="trend",
            side="short",
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("macd", "MACD bearish cross", macd_cross_down().test),
                Condition("ema", "price below EMA20", price_below_ema(20).test),
                Condition("adx", "ADX >= 20", adx_above(20).test),
                Condition("vol", "volume >= 1.0x avg", volume_above(1.0).test, required=False),
            ],
        ),
        Rule(
            name="rule_breakout_volume",
            label="Breakout + Volume + Trend",
            kind="breakout",
            side="long",
            stop_atr=1.5, target_rr=2.5,
            conditions=[
                Condition("donch", "broke 20-bar high", donchian_break_up().test),
                Condition("vol", "volume >= 1.5x avg", volume_above(1.5).test),
                Condition("ema", "price above EMA50", price_above_ema(50).test),
                Condition("adx", "ADX >= 25", adx_above(25).test),
                Condition("vwap", "price above VWAP", above_vwap().test, required=False),
            ],
        ),
        Rule(
            name="rule_breakout_volume_short",
            label="Breakdown + Volume + Trend",
            kind="breakout",
            side="short",
            stop_atr=1.5, target_rr=2.5,
            conditions=[
                Condition("donch", "broke 20-bar low", donchian_break_down().test),
                Condition("vol", "volume >= 1.5x avg", volume_above(1.5).test),
                Condition("ema", "price below EMA50", price_below_ema(50).test),
                Condition("adx", "ADX >= 25", adx_above(25).test),
                Condition("vwap", "price below VWAP", below_vwap().test, required=False),
            ],
        ),
        Rule(
            name="rule_squeeze_release",
            label="Squeeze Release",
            kind="breakout",
            side="auto",
            bull_keys=("kelt_up",),
            bear_keys=("kelt_dn",),
            stop_atr=1.5, target_rr=2.5,
            conditions=[
                Condition("squeeze", "Bollinger squeeze in progress", band_squeeze().test),
                Condition("kelt_up", "close above Keltner upper", keltner_break_up().test, group="direction"),
                Condition("kelt_dn", "close below Keltner lower", keltner_break_down().test, group="direction"),
                Condition("adx", "ADX rising", adx_rising().test),
                Condition("vol", "volume >= 1.3x avg", volume_above(1.3).test, required=False),
            ],
        ),
        Rule(
            name="rule_vwap_reversion",
            label="VWAP Reversion in Uptrend",
            kind="meanrev",
            side="long",
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("dev", "price 1.5 ATR below VWAP", vwap_dev_below(1.5).test),
                Condition("rsi", "RSI < 40", rsi_below(40).test),
                Condition("stack", "EMA20 > EMA50", _ema20_above_50),
                Condition("vwap", "price still below VWAP", below_vwap().test),
            ],
        ),
        Rule(
            name="rule_vwap_reversion_short",
            label="VWAP Reversion in Downtrend",
            kind="meanrev",
            side="short",
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("dev", "price 1.5 ATR above VWAP", vwap_dev_above(1.5).test),
                Condition("rsi", "RSI > 60", rsi_above(60).test),
                Condition("stack", "EMA20 < EMA50", _ema20_below_50),
                Condition("vwap", "price still above VWAP", above_vwap().test),
            ],
        ),
        Rule(
            name="rule_supertrend_volume",
            label="Supertrend Flip + Volume",
            kind="trend",
            side="auto",
            bull_keys=("st_bull",),
            bear_keys=("st_bear",),
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("st_bull", "Supertrend flipped bullish", _st_flipped_up, group="direction"),
                Condition("st_bear", "Supertrend flipped bearish", _st_flipped_down, group="direction"),
                Condition("vol", "volume >= 1.3x avg", volume_above(1.3).test),
                Condition("adx", "ADX >= 20", adx_above(20).test),
            ],
        ),
        Rule(
            name="rule_rsi_mfi",
            label="RSI + MFI Double Oversold",
            kind="meanrev",
            side="long",
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("rsi", "RSI < 35", rsi_below(35).test),
                Condition("mfi", "MFI < 30", mfi_below(30).test),
                Condition("adx", "ADX < 30 (ranging)", adx_below(30).test),
                Condition("bb", "close below lower band", bb_lower_break().test, required=False),
            ],
        ),
        Rule(
            name="rule_rsi_mfi_short",
            label="RSI + MFI Double Overbought",
            kind="meanrev",
            side="short",
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("rsi", "RSI > 65", rsi_above(65).test),
                Condition("mfi", "MFI > 70", mfi_above(70).test),
                Condition("adx", "ADX < 30 (ranging)", adx_below(30).test),
                Condition("bb", "close above upper band", bb_upper_break().test, required=False),
            ],
        ),
        Rule(
            name="rule_ema20_50_cross",
            label="EMA 20/50 Cross + ADX",
            kind="trend",
            side="auto",
            bull_keys=("cross_up",),
            bear_keys=("cross_dn",),
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("cross_up", "EMA20 crossing above EMA50", _ema20_50_cross_up, group="direction"),
                Condition("cross_dn", "EMA20 crossing below EMA50", _ema20_50_cross_down, group="direction"),
                Condition("adx", "ADX >= 20", adx_above(20).test),
                Condition("vol", "volume >= 1.0x avg", volume_above(1.0).test, required=False),
            ],
        ),
        Rule(
            name="rule_keltner_trend",
            label="Keltner Break + EMA Stack",
            kind="breakout",
            side="auto",
            bull_keys=("kelt_up",),
            bear_keys=("kelt_dn",),
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("kelt_up", "close above Keltner upper", keltner_break_up().test, group="direction"),
                Condition("kelt_dn", "close below Keltner lower", keltner_break_down().test, group="direction"),
                Condition("stack", "EMA20 > EMA50 > EMA200", ema_stack_bullish().test),
                Condition("vwap", "price above VWAP", above_vwap().test, required=False),
            ],
        ),
        Rule(
            name="rule_obv_momentum",
            label="OBV + Price Momentum",
            kind="volume",
            side="auto",
            bull_keys=("obv_up",),
            bear_keys=("obv_dn",),
            stop_atr=1.5, target_rr=2.0,
            conditions=[
                Condition("obv_up", "OBV rising", obv_rising().test, group="direction"),
                Condition("obv_dn", "OBV falling", obv_falling().test, group="direction"),
                Condition("rsi", "RSI 45-70", rsi_between(45, 70).test),
                Condition("adx", "ADX >= 20", adx_above(20).test),
                Condition("macd", "MACD histogram positive", macd_hist_above().test, required=False),
            ],
        ),
    ]


# --- helper tests used above ---------------------------------------------

def _ema50_200_cross_up(ctx: Ctx) -> bool:
    return ctx.crossed_up(ctx.ema50, ctx.ema200)


def _ema20_50_cross_up(ctx: Ctx) -> bool:
    return ctx.crossed_up(ctx.ema20, ctx.ema50)


def _ema20_50_cross_down(ctx: Ctx) -> bool:
    return ctx.crossed_down(ctx.ema20, ctx.ema50)


def _ema20_above_50(ctx: Ctx) -> bool:
    a, b = ctx.val(ctx.ema20), ctx.val(ctx.ema50)
    return (not pd.isna(a)) and (not pd.isna(b)) and a > b


def _ema20_below_50(ctx: Ctx) -> bool:
    a, b = ctx.val(ctx.ema20), ctx.val(ctx.ema50)
    return (not pd.isna(a)) and (not pd.isna(b)) and a < b


def _st_flipped_up(ctx: Ctx) -> bool:
    if len(ctx.st_dir) < 2:
        return False
    return int(ctx.st_dir.iloc[-2]) == -1 and int(ctx.st_dir.iloc[-1]) == 1


def _st_flipped_down(ctx: Ctx) -> bool:
    if len(ctx.st_dir) < 2:
        return False
    return int(ctx.st_dir.iloc[-2]) == 1 and int(ctx.st_dir.iloc[-1]) == -1
