import sys
sys.path.insert(0, r'C:\IDTS')

import numpy as np
import pandas as pd
from trader.core.regime import classify_regime, should_trade

# Create data with a strong trend
print("=== Strong trend data ===")
np.random.seed(42)
n = 200
trend = 0.05
closes = 100 + np.cumsum(np.random.randn(n) * 0.01 + trend)
highs = closes * (1 + np.abs(np.random.randn(n)) * 0.008)
lows = closes * (1 - np.abs(np.random.randn(n)) * 0.008)
vols = np.random.randint(2000, 8000, n)
df = pd.DataFrame({'close': closes, 'high': highs, 'low': lows, 'volume': vols})
df.index = pd.date_range('2023-01-01', periods=n, freq='D')

r = classify_regime(df)
print(f"regime: {r['regime']}")
print(f"adx: {r['adx']:.1f}")
print(f"atr_pct: {r['atr_pct']:.2f}%")
print(f"volume_ratio: {r['volume_ratio']:.2f}")
print(f"confidence: {r['confidence']}")
print(f"should_trade: {should_trade(r['regime'])}")

# Create range-bound data
print("\n=== Range-bound data ===")
closes2 = 100 + np.cumsum(np.random.randn(n) * 0.005)
highs2 = closes2 * (1 + np.abs(np.random.randn(n)) * 0.003)
lows2 = closes2 * (1 - np.abs(np.random.randn(n)) * 0.003)
vols2 = np.random.randint(1500, 4000, n)
df2 = pd.DataFrame({'close': closes2, 'high': highs2, 'low': lows2, 'volume': vols2})
df2.index = pd.date_range('2023-01-01', periods=n, freq='D')

r2 = classify_regime(df2)
print(f"regime: {r2['regime']}")
print(f"adx: {r2['adx']:.1f}")
print(f"atr_pct: {r2['atr_pct']:.2f}%")
print(f"volume_ratio: {r2['volume_ratio']:.2f}")
print(f"confidence: {r2['confidence']}")
print(f"should_trade: {should_trade(r2['regime'])}")

# Create low liquidity data
print("\n=== Low liquidity data ===")
closes3 = 100 + np.cumsum(np.random.randn(n) * 0.01)
highs3 = closes3 * (1 + np.abs(np.random.randn(n)) * 0.01)
lows3 = closes3 * (1 - np.abs(np.random.randn(n)) * 0.003)
vols3 = np.random.randint(100, 500, n)
df3 = pd.DataFrame({'close': closes3, 'high': highs3, 'low': lows3, 'volume': vols3})
df3.index = pd.date_range('2023-01-01', periods=n, freq='D')

r3 = classify_regime(df3)
print(f"regime: {r3['regime']}")
print(f"adx: {r3['adx']:.1f}")
print(f"atr_pct: {r3['atr_pct']:.2f}%")
print(f"volume_ratio: {r3['volume_ratio']:.2f}")
print(f"confidence: {r3['confidence']}")
print(f"should_trade: {should_trade(r3['regime'])}")