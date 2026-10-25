import sys
import pandas as pd
import numpy as np
sys.path.insert(0, r'C:\IDTS')

# Create sample OHLCV data
np.random.seed(42)
n = 100
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
high = close + np.abs(np.random.randn(n) * 0.3)
low = close - np.abs(np.random.randn(n) * 0.3)
vol = np.random.randint(500, 2000, n)
df = pd.DataFrame({"close": close, "high": high, "low": low, "volume": vol})

from trader.core.np_indicators import (
    atr_np, rsi_np, vwap_np, _ema_np_fast,
    macd_np, bollinger_np, stochastic_np, williams_r_np,
    cci_np, obv_np, supertrend_np,
)

close_arr = df["close"].to_numpy(dtype=float)
high_arr = df["high"].to_numpy(dtype=float)
low_arr = df["low"].to_numpy(dtype=float)
vol_arr = df["volume"].to_numpy(dtype=float)

# Test all the new indicators
print("=== MACD ===")
macd_line, macd_signal, macd_hist = macd_np(close_arr, 12, 26, 9)
print(f"macd_line[-1]: {macd_line[-1]}")
print(f"macd_signal[-1]: {macd_signal[-1]}")
print(f"macd_hist[-1]: {macd_hist[-1]}")

print("\n=== Bollinger Bands ===")
bb_mid, bb_upper, bb_lower = bollinger_np(close_arr, 20, 2.0)
print(f"bb_mid[-1]: {bb_mid[-1]}")
print(f"bb_upper[-1]: {bb_upper[-1]}")
print(f"bb_lower[-1]: {bb_lower[-1]}")
bb_range = bb_upper[-1] - bb_lower[-1]
bb_pos = (close_arr[-1] - bb_lower[-1]) / bb_range if bb_range != 0 else None
print(f"bb_pos: {bb_pos}")

print("\n=== Stochastic ===")
stoch_k, stoch_d = stochastic_np(high_arr, low_arr, close_arr, 14, 3)
print(f"stoch_k[-1]: {stoch_k[-1]}")
print(f"stoch_d[-1]: {stoch_d[-1]}")

print("\n=== Williams %R ===")
willr = williams_r_np(high_arr, low_arr, close_arr, 14)
print(f"willr[-1]: {willr[-1]}")

print("\n=== CCI ===")
cci_val = cci_np(high_arr, low_arr, close_arr, 20)
print(f"cci[-1]: {cci_val[-1]}")

print("\n=== OBV ===")
obv_arr = obv_np(close_arr, vol_arr)
print(f"obv[-1]: {obv_arr[-1]}")

print("\n=== Supertrend ===")
st = supertrend_np(high_arr, low_arr, close_arr, atr_np(high_arr, low_arr, close_arr, 14), 10, 3.0)
print(f"supertrend[-1]: {st[-1]}")

print("\n=== All new metrics test PASSED ===")