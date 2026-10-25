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

from trader.core.engine import Engine
engine = Engine()

# Test _scan_metrics
metrics = engine._scan_metrics(df, float(close[-1]), type('obj', (object,), {'adx': 24.0})())
print("Metrics keys:", list(metrics.keys()))
print("atr_pct:", metrics.get("atr_pct"))
print("adx:", metrics.get("adx"))
print("rsi:", metrics.get("rsi"))
print("rvol:", metrics.get("rvol"))
print("vwap:", metrics.get("vwap"))
print("vwap_dist:", metrics.get("vwap_dist"))
print("spark len:", len(metrics.get("spark", [])))
print("chart len:", len(metrics.get("chart", [])))
print("ema20:", metrics.get("ema20"))
print("ema50:", metrics.get("ema50"))
print("macd_line:", metrics.get("macd_line"))
print("macd_signal:", metrics.get("macd_signal"))
print("macd_hist:", metrics.get("macd_hist"))
print("bb_mid:", metrics.get("bb_mid"))
print("bb_upper:", metrics.get("bb_upper"))
print("bb_lower:", metrics.get("bb_lower"))
print("bb_pos:", metrics.get("bb_pos"))
print("stoch_k:", metrics.get("stoch_k"))
print("stoch_d:", metrics.get("stoch_d"))
print("willr:", metrics.get("willr"))
print("cci:", metrics.get("cci"))
print("obv:", metrics.get("obv"))
print("supertrend:", metrics.get("supertrend"))