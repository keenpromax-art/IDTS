from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    if isinstance(df.index, pd.DatetimeIndex):
        groups = df.index.date
        cumvol = df["volume"].groupby(groups).cumsum()
        cum_pv = pv.groupby(groups).cumsum()
        return cum_pv / cumvol.replace(0, np.nan)
    cumvol = df["volume"].cumsum()
    return pv.cumsum() / cumvol.replace(0, np.nan)


def bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = series.rolling(period).mean()
    sd = series.rolling(period).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    return mid, upper, lower


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_ = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Supertrend line and direction.

    Implemented with numpy arrays rather than per-bar .iloc writes - the old
    version dominated the whole scan's runtime.
    """
    atr_ = atr(df, period)
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy(dtype=float)
    upper = hl2 + mult * atr_.to_numpy(dtype=float)
    lower = hl2 - mult * atr_.to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return (pd.Series(index=df.index, dtype=float),
                pd.Series(index=df.index, dtype=int))
    final_upper = np.empty(n, dtype=float)
    final_lower = np.empty(n, dtype=float)
    direction = np.empty(n, dtype=int)
    st = np.empty(n, dtype=float)

    final_upper[0] = upper[0]
    final_lower[0] = lower[0]
    direction[0] = 1
    st[0] = final_lower[0]

    for i in range(1, n):
        pu, pl = final_upper[i - 1], final_lower[i - 1]
        # basic upper band: lower of the raw band and the previous, unless
        # price closed above the previous band
        final_upper[i] = upper[i] if (upper[i] < pu or close[i - 1] > pu) else pu
        final_lower[i] = lower[i] if (lower[i] > pl or close[i - 1] < pl) else pl
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < pl else 1
        else:
            direction[i] = 1 if close[i] > pu else -1
        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return (pd.Series(st, index=df.index), pd.Series(direction, index=df.index))


def donchian_high(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()


def donchian_low(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()


def highest_high(df: pd.DataFrame, period: int) -> pd.Series:
    return df["high"].rolling(period).max()


def lowest_low(df: pd.DataFrame, period: int) -> pd.Series:
    return df["low"].rolling(period).min()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """%K and %D of the classic stochastic oscillator."""
    hh = highest_high(df, k_period)
    ll = lowest_low(df, k_period)
    rng = (hh - ll).replace(0, np.nan)
    k = 100 * (df["close"] - ll) / rng
    d = k.rolling(d_period).mean()
    return k, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hh = highest_high(df, period)
    ll = lowest_low(df, period)
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - df["close"]) / rng


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean().replace(0, np.nan)
    return (tp - ma) / (0.015 * md)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    return series.pct_change(period) * 100


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    up = raw.where(tp.diff() > 0, 0.0)
    down = raw.where(tp.diff() < 0, 0.0)
    pos = up.rolling(period).sum()
    neg = down.rolling(period).sum()
    ratio = pos / neg.replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    if n < 2:
        return pd.Series(index=df.index, dtype=float)
    out = np.zeros(n, dtype=float)
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
    return pd.Series(out, index=df.index)


def keltner(df: pd.DataFrame, period: int = 20, mult: float = 1.5):
    mid = ema(df["close"], period)
    a = atr(df, period)
    return mid - mult * a, mid, mid + mult * a


def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span: int = 52):
    """Return (tenkan, kijun, senkou_a, senkou_b)."""
    tenkan = (highest_high(df, conv) + lowest_low(df, conv)) / 2
    kijun = (highest_high(df, base) + lowest_low(df, base)) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(base)
    senkou_b = ((highest_high(df, span) + lowest_low(df, span)) / 2).shift(base)
    return tenkan, kijun, senkou_a, senkou_b


def atr_percentile(series: pd.Series, window: int = 100) -> float:
    """Percentile rank of the latest ATR value inside its recent window."""
    vals = series.dropna().tail(window)
    if len(vals) == 0:
        return 50.0
    last = float(vals.iloc[-1])
    return float((vals <= last).sum() / len(vals) * 100)


def ema_stack(series: pd.Series, fast: int, mid: int, slow: int) -> bool:
    """True when fast > mid > slow (bullish alignment)."""
    f = ema(series, fast)
    m = ema(series, mid)
    s = ema(series, slow)
    if pd.isna(f.iloc[-1]) or pd.isna(m.iloc[-1]) or pd.isna(s.iloc[-1]):
        return False
    return bool(f.iloc[-1] > m.iloc[-1] > s.iloc[-1])
