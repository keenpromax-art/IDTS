import sys
sys.path.insert(0, r'C:\IDTS')

from trader.core.indicators import atr, adx, rsi, stochastic, mfi, obv, cci, roc, williams_r
import pandas as pd
import numpy as np

# Create sample data with all needed columns
dates = pd.date_range('2020-01-01', periods=100, freq='D')
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
volumes = np.random.randint(1000, 10000, 100)
highs = prices * (1 + np.abs(np.random.randn(100)) * 0.005)
lows = prices * (1 - np.abs(np.random.randn(100)) * 0.005)

df = pd.DataFrame({
    'close': prices,
    'high': highs,
    'low': lows,
    'open': prices * (1 + np.random.randn(100) * 0.001),
    'volume': volumes
}, index=dates)

# Test ATR
atr_val = atr(df, 14)
print(f"ATR (14): {atr_val.iloc[-1]:.4f}")

# Test ADX
adx_val = adx(df, 14)
print(f"ADX (14): {adx_val.iloc[-1]:.4f}")

# Test RSI
rsi_val = rsi(df, 14)
rsi_result = rsi_val.iloc[-1] if hasattr(rsi_val, 'iloc') else rsi_val
print(f"RSI (14): {rsi_result:.4f}")

# Test Stochastic
stoch_val = stoch(df, 14)
print(f"STOCH K: {stoch_val.iloc[-1]:.4f}")

# Test MFI
mfi_val = mfi(df, 14)
print(f"MFI (14): {mfi_val.iloc[-1]:.4f}")

# Test OBV
obv_val = obv(df)
print(f"OBV final: {obv_val.iloc[-1]:.0f}")

print("\nAll basic indicators work!")