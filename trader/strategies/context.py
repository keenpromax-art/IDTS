from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from trader.core.models import Side, Signal
from trader.core.np_indicators import (
    adx_np, atr_np, bollinger_np, cci_np, crossed_down_np, crossed_up_np,
    ichimoku_np, keltner_np, macd_np, mfi_np, obv_np, psar_np, rsi_np, roc_np,
    rolling_max_np, rolling_mean_np, rolling_min_np, stochastic_np,
    supertrend_np, vwap_np, williams_r_np, _ema_np_fast,
)


def _shift1(arr: np.ndarray) -> np.ndarray:
    """Shift an array right by one (the previous bar's value)."""
    if len(arr) == 0:
        return arr.copy()
    out = np.empty_like(arr)
    out[0] = np.nan
    out[1:] = arr[:-1]
    return out


class Ctx:
    """Pre-computed indicator state for one symbol's candle frame.

    Every indicator is calculated once per symbol with numpy and shared with
    all the strategies, so adding setups does not slow the scan down.

    The context also has a *cursor*: `set_bar(i)` moves the "current bar" so
    the same object can be replayed bar by bar. All indicators are causal
    (they only look backwards), so the value the cursor reads at bar i is
    identical to what a context built from `df[:i+1]` would hold. That is
    what makes the mini-backtest fast: one context, N bar evaluations,
    instead of N contexts.
    """

    def __init__(self, df: pd.DataFrame, symbol: str = "", market: str = ""):
        self.df = df
        self.symbol = symbol
        self.market = market
        self._bar = max(0, len(df) - 1)
        self._day_key_cache = None
        self._prev_session = None
        self._prev_session_pos = -2
        self._ema_cache: dict = {}

        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.volume = df["volume"].to_numpy(dtype=float)
        n = len(self.close)

        # --- core series ------------------------------------------------
        self.atr_s = atr_np(self.high, self.low, self.close, 14)
        self.rsi_s = rsi_np(self.close, 14)
        self.adx_s, self.plus_di, self.minus_di = adx_np(self.high, self.low, self.close, 14)
        self.ema9 = _ema_np_fast(self.close, 9)
        self.ema20 = _ema_np_fast(self.close, 20)
        self.ema21 = _ema_np_fast(self.close, 21)
        self.ema50 = _ema_np_fast(self.close, 50)
        self.ema200 = _ema_np_fast(self.close, 200)
        self.macd_line, self.macd_signal, self.macd_hist = macd_np(self.close)
        self.bb_mid, self.bb_up, self.bb_low = bollinger_np(self.close, 20, 2.0)
        self.vwap_s = vwap_np(self.high, self.low, self.close, self.volume, self._day_keys())
        self.vol_ma = rolling_mean_np(self.volume, 20)
        self.st_line, self.st_dir = supertrend_np(self.high, self.low, self.close, self.atr_s, 10, 3.0)
        self.psar_s = psar_np(self.high, self.low)
        self.k, self.d = stochastic_np(self.high, self.low, self.close, 14, 3)
        self.wr = williams_r_np(self.high, self.low, self.close, 14)
        self.cci_s = cci_np(self.high, self.low, self.close, 20)
        self.roc_s = roc_np(self.close, 12)
        self.mfi_s = mfi_np(self.high, self.low, self.close, self.volume, 14)
        self.obv_s = obv_np(self.close, self.volume)
        self.kelt_lo, self.kelt_mid, self.kelt_hi = keltner_np(
            self.high, self.low, self.close, self.atr_s, 20, 1.5)
        self.tenkan, self.kijun, self.senkou_a, self.senkou_b = ichimoku_np(
            self.high, self.low, self.close)
        self.donch_hi = rolling_max_np(self.high, 20)
        self.donch_lo = rolling_min_np(self.low, 20)
        self.high60 = rolling_max_np(self.high, 60)
        self.low60 = rolling_min_np(self.low, 60)
        # levels from *before* the current bar - a breakout compares the
        # current close against the range that existed before it
        self.donch_hi_prev = _shift1(self.donch_hi)
        self.donch_lo_prev = _shift1(self.donch_lo)
        self.high60_prev = _shift1(self.high60)
        self.low60_prev = _shift1(self.low60)
        self.kelt_hi_prev = _shift1(self.kelt_hi)
        self.kelt_lo_prev = _shift1(self.kelt_lo)
        self.bb_up_prev = _shift1(self.bb_up)
        self.bb_low_prev = _shift1(self.bb_low)
        self._ema_cache: dict = {}
        # previous bar's values, so strategies do not have to touch pandas
        self.prev_close_arr = _shift1(self.close)
        self.prev_high_arr = _shift1(self.high)
        self.prev_low_arr = _shift1(self.low)

    # -- cursor ----------------------------------------------------------
    def set_bar(self, i: int):
        """Move the cursor to bar *i* (used by the mini-backtest)."""
        self._bar = max(0, min(int(i), len(self.close) - 1))

    @property
    def bar(self) -> int:
        return self._bar

    @property
    def available(self) -> int:
        """How many bars are usable at the cursor (i.e. cursor index + 1)."""
        return self._bar + 1

    # -- cursor-aware values ---------------------------------------------
    @property
    def last(self):
        return self.df.iloc[self._bar]

    @property
    def prev(self):
        i = max(0, self._bar - 1)
        return self.df.iloc[i]

    @property
    def last_close(self) -> float:
        return float(self.close[self._bar])

    @property
    def last_high(self) -> float:
        return float(self.high[self._bar])

    @property
    def last_low(self) -> float:
        return float(self.low[self._bar])

    @property
    def prev_close(self) -> float:
        return float(self.prev_close_arr[max(0, self._bar - 1)])

    @property
    def prev_high(self) -> float:
        return float(self.prev_high_arr[max(0, self._bar - 1)])

    @property
    def prev_low(self) -> float:
        return float(self.prev_low_arr[max(0, self._bar - 1)])

    @property
    def px(self) -> float:
        return float(self.close[self._bar])

    @property
    def prev_px(self) -> float:
        i = max(0, self._bar - 1)
        return float(self.close[i])

    @property
    def atr(self) -> float:
        v = self.atr_s[self._bar]
        return 0.0 if (v is None or np.isnan(v)) else float(v)

    @property
    def adx(self) -> float:
        v = self.adx_s[self._bar]
        return 0.0 if (v is None or np.isnan(v)) else float(v)

    @property
    def vwap(self) -> float:
        v = self.vwap_s[self._bar]
        return self.px if (v is None or np.isnan(v)) else float(v)

    @property
    def vol_ratio(self) -> float:
        vm = self.vol_ma[self._bar]
        if vm is None or np.isnan(vm) or vm <= 0:
            return 1.0
        return float(self.volume[self._bar]) / float(vm)

    def val(self, series) -> float:
        if series is None or len(series) == 0:
            return float("nan")
        i = min(self._bar, len(series) - 1)
        v = series[i]
        return float("nan") if pd.isna(v) else float(v)

    def prev_val(self, series) -> float:
        if series is None or len(series) < 2:
            return float("nan")
        i = max(0, min(self._bar - 1, len(series) - 1))
        v = series[i]
        return float("nan") if pd.isna(v) else float(v)

    def crossed_up(self, fast, slow) -> bool:
        return crossed_up_np(fast, slow, self._bar)

    def crossed_down(self, fast, slow) -> bool:
        return crossed_down_np(fast, slow, self._bar)

    def dev_atr(self) -> float:
        """How far price sits from VWAP, in ATR units."""
        if self.atr <= 0:
            return 0.0
        return (self.px - self.vwap) / self.atr

    def band_width(self) -> float:
        up, lo = self.val(self.bb_up), self.val(self.bb_low)
        mid = self.val(self.bb_mid)
        if mid <= 0 or np.isnan(up) or np.isnan(lo):
            return float("nan")
        return (up - lo) / mid * 100

    def bar_up(self) -> bool:
        i = self._bar
        return i > 0 and self.close[i] > self.close[i - 1]

    def bar_down(self) -> bool:
        i = self._bar
        return i > 0 and self.close[i] < self.close[i - 1]

    # -- session helpers --------------------------------------------------
    def _day_keys(self):
        """Day identifier per bar so VWAP / session logic can group bars.

        Cached - it is asked for on every bar of every backtest.
        """
        if self._day_key_cache is not None:
            return self._day_key_cache
        idx = self.df.index
        keys = None
        try:
            if isinstance(idx, pd.MultiIndex):
                for name in ("day", "date"):
                    if name in idx.names:
                        keys = idx.get_level_values(name)
                        break
                if keys is None:
                    keys = idx.get_level_values(0)
            else:
                keys = idx.date
        except Exception:
            keys = None
        self._day_key_cache = keys
        return keys

    def session_bars(self):
        """Indices of the bars belonging to the session at the cursor."""
        keys = self._day_keys()
        if keys is None:
            return np.arange(len(self.close))
        today = keys[self._bar]
        return np.flatnonzero(np.array([k == today for k in keys]))

    def prev_session_bars(self):
        """Indices of the bars belonging to the previous session."""
        keys = self._day_keys()
        if keys is None:
            return np.array([], dtype=int)
        # the previous session only changes at a session boundary, so cache it
        pos = self._bar
        if self._prev_session is not None and pos == self._prev_session_pos:
            return self._prev_session
        today = keys[pos]
        idx = np.flatnonzero(np.array([k == today for k in keys]))
        if idx.size == 0:
            out = np.array([], dtype=int)
        else:
            first_today = idx[0]
            prev = np.flatnonzero(np.arange(len(keys)) < first_today)
            if prev.size == 0:
                out = np.array([], dtype=int)
            else:
                last_day = keys[prev[-1]]
                out = np.flatnonzero(np.array([k == last_day for k in keys]))
        self._prev_session = out
        self._prev_session_pos = pos
        return out

    def _ema(self, period: int):
        """EMA of any period, memoised (the common ones are pre-built)."""
        preset = {9: self.ema9, 20: self.ema20, 21: self.ema21,
                  50: self.ema50, 200: self.ema200}
        if period in preset:
            return preset[period]
        if period not in self._ema_cache:
            self._ema_cache[period] = _ema_np_fast(self.close, period)
        return self._ema_cache[period]


def _shift1(arr: np.ndarray) -> np.ndarray:
    """Shift an array right by one (the previous bar's value)."""
    if len(arr) == 0:
        return arr.copy()
    out = np.empty_like(arr)
    out[0] = np.nan
    out[1:] = arr[:-1]
    return out


def make_signal(symbol, market, strategy, side, entry, stop, target,
                confidence=0.5, reason="") -> Optional[Signal]:
    """Build a Signal, guarding against a zero, inverted or NaN stop."""
    if not all(np.isfinite(x) for x in (entry, stop, target)):
        return None
    if side == Side.LONG:
        risk = entry - stop
    else:
        risk = stop - entry
    if not (risk > 0):
        return None
    return Signal(symbol=symbol, market=market, strategy=strategy, side=side,
                  entry=entry, stop=stop, target=target,
                  confidence=float(max(0.0, min(1.0, confidence))), reason=reason)
