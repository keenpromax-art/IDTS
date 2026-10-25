import numpy as np
import pandas as pd
from trader.core.indicators import rsi, atr, ema, vwap, bollinger, macd, adx, supertrend
from trader.strategies import ALL_STRATEGIES


def make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    open_ = close + rng.normal(0, 0.1, n)
    vol = rng.integers(1000, 5000, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                      index=pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"))
    return df


def test_indicators_no_crash():
    df = make_df(120)
    assert rsi(df["close"]).notna().sum() > 0
    assert atr(df).notna().sum() > 0
    assert ema(df["close"], 9).notna().sum() > 0
    assert vwap(df).notna().sum() > 0
    mid, up, lo = bollinger(df["close"])
    assert up.notna().sum() > 0
    m, s, h = macd(df["close"])
    assert m.notna().sum() > 0
    assert adx(df).notna().sum() > 0
    st, d = supertrend(df)
    assert st.notna().sum() > 0


def test_strategies_no_crash():
    df = make_df(120)
    for S in ALL_STRATEGIES:
        s = S()
        try:
            sig = s.generate(df, "TEST", "India")
        except Exception as e:
            raise AssertionError(f"{S.__name__} crashed: {e}")
        if sig is not None:
            assert sig.stop != sig.entry
            assert sig.target != sig.entry
