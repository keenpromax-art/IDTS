"""Numpy implementations of the indicators used by the scanner.

The pandas versions in ``trader.core.indicators`` stay as the public API,
but the hot path (one Ctx per symbol per scan) uses these. With ~500 symbols
and 45 setups per symbol, per-call overhead dominates, and numpy arrays cut
the indicator cost by roughly an order of magnitude.
"""
from __future__ import annotations

import numpy as np


def _ema_np(x: np.ndarray, span: int) -> np.ndarray:
    """Wilder-style EMA matching pandas ewm(span, adjust=False)."""
    n = len(x)
    if n == 0:
        return x.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty(n, dtype=float)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _ema_np_fast(x: np.ndarray, span: int) -> np.ndarray:
    """EMA matching pandas ewm(span, adjust=False).

        out[i] = alpha*x[i] + beta*out[i-1],  out[0] = x[0]

    This is a first-order IIR filter, so scipy's lfilter does it in C when
    scipy is available. We fall back to a plain numpy loop otherwise.
    """
    n = len(x)
    if n == 0:
        return x.copy()
    if n == 1:
        return x.copy()
    alpha = 2.0 / (span + 1.0)
    beta = 1.0 - alpha
    try:
        from scipy.signal import lfilter
        # y[i] = beta*y[i-1] + alpha*x[i] with y[-1] = x[0] as the seed
        zi = np.array([beta * x[0]], dtype=float)
        out, _ = lfilter([alpha], [1.0, -beta], x, zi=zi)
        out[0] = x[0]
        return out
    except Exception:
        out = np.empty(n, dtype=float)
        out[0] = x[0]
        for i in range(1, n):
            out[i] = alpha * x[i] + beta * out[i - 1]
        return out


def true_range_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    if n == 0:
        return np.zeros(0)
    prev = np.empty(n)
    prev[0] = close[0]
    prev[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return tr


def atr_np(high, low, close, period=14):
    tr = true_range_np(high, low, close)
    return _ema_np_fast(tr, period)


def rsi_np(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    delta = np.diff(close)                 # n-1
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    avg_gain = np.empty(n - 1)
    avg_loss = np.empty(n - 1)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n - 1):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
    out[1:] = 100 - 100 / (1 + rs)
    return out


def macd_np(close: np.ndarray, fast=12, slow=26, signal=9):
    m = _ema_np_fast(close, fast) - _ema_np_fast(close, slow)
    s = _ema_np_fast(m, signal)
    return m, s, m - s


def bollinger_np(close: np.ndarray, period=20, std=2.0):
    n = len(close)
    mid = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    if n < period:
        return mid, mid + std * sd, mid - std * sd
    # rolling mean and std via cumulative sums
    csum = np.concatenate(([0.0], np.cumsum(close)))
    csum2 = np.concatenate(([0.0], np.cumsum(close * close)))
    idx = np.arange(period, n + 1)
    s1 = csum[idx] - csum[idx - period]
    s2 = csum2[idx] - csum2[idx - period]
    mean = s1 / period
    var = np.maximum(s2 / period - mean * mean, 0.0)
    sd[period - 1:] = np.sqrt(var)
    mid[period - 1:] = mean
    return mid, mid + std * sd, mid - std * sd


def vwap_np(high, low, close, volume, day_key=None):
    if len(close) == 0:
        return np.zeros(0)
    tp = (high + low + close) / 3.0
    pv = tp * volume
    n = len(close)
    if n == 0:
        return np.zeros(0)
    if day_key is None:
        return np.cumsum(pv) / np.maximum(np.cumsum(volume), 1e-12)
    # reset per day
    out = np.empty(n, dtype=float)
    cum_pv = 0.0
    cum_v = 0.0
    prev_key = None
    for i in range(n):
        k = day_key[i]
        if k != prev_key:
            cum_pv = 0.0
            cum_v = 0.0
            prev_key = k
        cum_pv += pv[i]
        cum_v += volume[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else tp[i]
    return out


def adx_np(high, low, close, period=14):
    n = len(close)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    up = np.diff(high, prepend=high[0])       # n
    down = -np.diff(low, prepend=low[0])      # n
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range_np(high, low, close)
    a = _ema_np_fast(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * _ema_np_fast(plus_dm, period) / a
        mdi = 100 * _ema_np_fast(minus_dm, period) / a
        dx = np.where((pdi + mdi) > 0, np.abs(pdi - mdi) / (pdi + mdi) * 100, 0.0)
    return _ema_np_fast(dx, period), pdi, mdi


def supertrend_np(high, low, close, atr_vals, period=10, mult=3.0):
    n = len(close)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr_vals
    lower = hl2 - mult * atr_vals
    fu = np.empty(n)
    fl = np.empty(n)
    direction = np.empty(n, dtype=int)
    st = np.empty(n)
    fu[0] = upper[0]
    fl[0] = lower[0]
    direction[0] = 1
    st[0] = fl[0]
    for i in range(1, n):
        pu, pl = fu[i - 1], fl[i - 1]
        fu[i] = upper[i] if (upper[i] < pu or close[i - 1] > pu) else pu
        fl[i] = lower[i] if (lower[i] > pl or close[i - 1] < pl) else pl
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < pl else 1
        else:
            direction[i] = 1 if close[i] > pu else -1
        st[i] = fl[i] if direction[i] == 1 else fu[i]
    return st, direction


def psar_np(high, low, step=0.02, max_step=0.2):
    n = len(high)
    if n < 2:
        return np.zeros(n)
    out = np.zeros(n)
    bull = True
    af = step
    ep = high[0]
    sar = low[0]
    out[0] = sar
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar:
                bull = False
                sar = ep
                ep = low[i]
                af = step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar:
                bull = True
                sar = ep
                ep = high[i]
                af = step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
        out[i] = sar
    return out


def rolling_max_np(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    # sliding window max via stride tricks
    windows = np.lib.stride_tricks.sliding_window_view(x, period)
    out[period - 1:] = windows.max(axis=1)
    return out


def rolling_min_np(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, period)
    out[period - 1:] = windows.min(axis=1)
    return out


def rolling_sum_np(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    csum = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(period, n + 1)
    out[period - 1:] = csum[idx] - csum[idx - period]
    return out


def rolling_mean_np(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    csum = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(period, n + 1)
    out[period - 1:] = (csum[idx] - csum[idx - period]) / period
    return out


def stochastic_np(high, low, close, k_period=14, d_period=3):
    hh = rolling_max_np(high, k_period)
    ll = rolling_min_np(low, k_period)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(rng > 0, 100 * (close - ll) / rng, 50.0)
    k = np.where(np.isnan(rng), np.nan, k)
    d = rolling_mean_np(np.nan_to_num(k), d_period)
    return k, d


def williams_r_np(high, low, close, period=14):
    hh = rolling_max_np(high, period)
    ll = rolling_min_np(low, period)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.where(rng > 0, -100 * (hh - close) / rng, -50.0)
    return np.where(np.isnan(rng), np.nan, wr)


def cci_np(high, low, close, period=20):
    if len(close) == 0:
        return np.zeros(0)
    tp = (high + low + close) / 3.0
    ma = rolling_mean_np(tp, period)
    dev = np.abs(tp - ma)
    md = rolling_mean_np(np.nan_to_num(dev), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(md > 0, (tp - ma) / (0.015 * md), 0.0)
    return np.where(np.isnan(ma), np.nan, cci)


def roc_np(close: np.ndarray, period: int = 12) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    prev = close[:-period]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period:] = (close[period:] / prev - 1.0) * 100.0
    return out


def obv_np(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    if len(close) == 0:
        return np.zeros(0)
    direction = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(direction * volume)


def mfi_np(high, low, close, volume, period=14):
    if len(close) == 0:
        return np.zeros(0)
    tp = (high + low + close) / 3.0
    raw = tp * volume
    diff = np.diff(tp, prepend=tp[0])
    up = np.where(diff > 0, raw, 0.0)
    down = np.where(diff < 0, raw, 0.0)
    pos = rolling_sum_np(up, period)
    neg = rolling_sum_np(down, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neg > 0, pos / neg, np.inf)
    return 100 - 100 / (1 + ratio)


def keltner_np(high, low, close, atr_vals, period=20, mult=1.5):
    mid = _ema_np_fast(close, period)
    return mid - mult * atr_vals, mid, mid + mult * atr_vals


def ichimoku_np(high, low, close, conv=9, base=26, span=52):
    n = len(close)
    tenkan = (rolling_max_np(high, conv) + rolling_min_np(low, conv)) / 2.0
    kijun = (rolling_max_np(high, base) + rolling_min_np(low, base)) / 2.0
    senkou_a = np.concatenate([np.full(base, np.nan), (tenkan + kijun) / 2.0])[:n]
    senkou_b = np.concatenate([
        np.full(span, np.nan),
        (rolling_max_np(high, span) + rolling_min_np(low, span)) / 2.0,
    ])[:n]
    return tenkan, kijun, senkou_a, senkou_b


def crossed_up_np(fast: np.ndarray, slow: np.ndarray, bar: int = -1) -> bool:
    """True when `fast` crossed above `slow` at bar `bar` (default: last)."""
    n = len(fast)
    if n < 2:
        return False
    i = n - 1 if bar < 0 else min(int(bar), n - 1)
    if i < 1:
        return False
    f0, f1 = fast[i - 1], fast[i]
    s0, s1 = slow[i - 1], slow[i]
    if any(np.isnan(x) for x in (f0, f1, s0, s1)):
        return False
    return bool(f0 <= s0 and f1 > s1)


def crossed_down_np(fast: np.ndarray, slow: np.ndarray, bar: int = -1) -> bool:
    n = len(fast)
    if n < 2:
        return False
    i = n - 1 if bar < 0 else min(int(bar), n - 1)
    if i < 1:
        return False
    f0, f1 = fast[i - 1], fast[i]
    s0, s1 = slow[i - 1], slow[i]
    if any(np.isnan(x) for x in (f0, f1, s0, s1)):
        return False
    return bool(f0 >= s0 and f1 < s1)
