import sys
sys.path.insert(0, r'C:\IDTS')

from trader.core.regime import detect_regime, classify_regime, should_trade, REGIMES_TO_ABSTAIN
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n = 150
closes = 100 + np.cumsum(np.random.randn(n) * 0.02)
highs = closes * (1 + np.abs(np.random.randn(n)) * 0.005)
lows = closes * (1 - np.abs(np.random.randn(n)) * 0.005)
vols = np.random.randint(1000, 5000, n)
df = pd.DataFrame({'close': closes, 'high': highs, 'low': lows, 'volume': vols})
df.index = pd.date_range('2023-01-01', periods=n, freq='D')

# Test detect_regime
r = detect_regime(df)
print('detect_regime result:')
for k, v in r.items():
    print(f'  {k}: {v}')

# Test should_trade
print(f'\nshould_trade (uncertain regime): {should_trade("uncertain")}')
print(f'should_trade (trend regime): {should_trade("trend")}')
print(f'should_trade (None regime): {should_trade(None)}')

# Test with actual data
r2 = classify_regime(df)
print(f'\nclassify_regime result: regime={r2["regime"]}, adx={r2["adx"]:.1f}, '
      f'atr_pct={r2["atr_pct"]:.2f}%, vol_ratio={r2["volume_ratio"]:.2f}, '
      f'conf={r2["confidence"]}')

# Test with very small data
r3 = classify_regime(pd.DataFrame())
print(f'\nEmpty DataFrame result regime: {r3["regime"]}')

# Test with None
r4 = classify_regime(None)
print(f'None data result regime: {r4["regime"]}')

# Test REGIMES_TO_ABSTAIN
print(f'\nREGIMES_TO_ABSTAIN: {REGIMES_TO_ABSTAIN}')
print(f'uncertain in ABSTAIN: {"uncertain" in REGIMES_TO_ABSTAIN}')
print(f'trend in ABSTAIN: {"trend" in REGIMES_TO_ABSTAIN}')

print('\nAll basic tests passed!')