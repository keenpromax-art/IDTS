import sys
sys.path.insert(0, r'C:\IDTS')

import numpy as np
import pandas as pd
from trader.core.regime import classify_regime, should_trade, detect_regime, REGIMES_TO_ABSTAIN
from trader.core.walkforward.walkforward import WalkForwardConfig, run_walk_forward_validation
from trader.strategies.base import Strategy

# Create a simple test strategy
class TestStrategy(Strategy):
    name = "test_strategy"
    kind = "meanrev"
    
    def generate(self, df, symbol="", market="", ctx=None):
        c = ctx if ctx is not None else type('Ctx', (), {'available': len(df), 'atr': 1.0, 'adx': 20})()
        if c.available < 30:
            return None
        k = stochastic(df, 14).iloc[-1] if len(stochastic(df, 14)) > 0 else 50
        kd = stochastic(df, 14).iloc[-2] if len(stochastic(df, 14), 2) > 1 else 50
        
        # Regime-aware version
        regime_info = classify_regime(df)
        
        # Mean reversion only in range-bound or uncertain regimes
        if regime_info['regime'] in ['range_bound', 'uncertain']:
            if k < 20:  # oversold
                entry = df['close'].iloc[-1]
                stop = entry - 1.5 * c.atr
                return type('Signal', (), {
                    'symbol': symbol, 'market': market, 'name': self.name,
                    'side': 'LONG', 'entry': entry, 'stop': stop,
                    'target': entry + 2.0 * (entry - stop),
                    'confidence': 0.6,
                    'reason': f"range_mean_rev {regime_info['regime']}"
                })
        return None

# Create test data
np.random.seed(42)
n = 500
dates = pd.date_range('2020-01-01', periods=n, freq='D')
closes = 100 + np.cumsum(np.random.randn(n) * 0.02)
highs = closes * (1 + np.abs(np.random.randn(n)) * 0.005)
lows = closes * (1 - np.abs(np.random.randn(n)) * 0.005)
vols = np.random.randint(1000, 5000, n)

df = pd.DataFrame({'close': closes, 'high': highs, 'low': lows, 'volume': vols})
df.index = dates

# Walk-forward validation
cfg = WalkForwardConfig(training_bars=200, validation_bars=50, test_bars=100, step_bars=50, min_data_bars=150)

strategy = TestStrategy()
result = run_walk_forward_validation(strategy, df, "TEST", {'risk': {}}, config=cfg)

print(f'Walk-forward results:')
print(f'  Folds: {len(result.folds)}')
print(f'  Best fold: {result.best_fold_id}')
print(f'  Worst fold: {result.worst_fold_id}')
print(f'  Stable: {result.stable}')
print(f'  Aggregate expectancy: {result.aggregate_metrics.get("expectancy_r", 0):.4f}')
print(f'  Aggregate total trades: {result.aggregate_metrics.get("total_trades", 0)}')

# Compare with and without regime filtering
# Count trades per regime
regime_counts = {}
for i, fold in enumerate(result.folds):
    for bar_idx in range(len(fold.test_df)):
        bar_data = fold.test_df.iloc[bar_idx] if bar_idx < len(fold.test_df) else None
        if bar_data is not None:
            r = classify_regime(bar_data.to_frame().T)
            reg = r['regime']
            regime_counts[reg] = regime_counts.get(reg, 0) + 1

print(f'\nRegime distribution across test periods:')
for reg, count in sorted(regime_counts.items()):
    print(f'  {reg}: {count} bars ({count/len(regime_counts)*100:.1f}%)')

print(f'\nRegimes to abstain from: {REGIMES_TO_ABSTAIN}')
print(f'Regimes available for trading: {set(regime_counts.keys()) - REGIMES_TO_ABSTAIN}')

# Calculate hypothetical results
# Always-trade (no regime filter) vs regime-aware
total_test_bars = sum(len(f.test_df) for f in result.folds)
always_trade_pct = 100.0
regime_aware_pct = len(set(regime_counts.keys()) - REGIMES_TO_ABSTAIN) / len(regime_counts) * 100.0

print(f'\nAlways trade: {always_trade_pct:.1f}% of bars')
print(f'Regime-aware trade: {regime_aware_pct:.1f}% of bars')
print(f'Opportunity cost (bars skipped): {100 - regime_aware_pct:.1f}%')

print('\nAll tests completed!')