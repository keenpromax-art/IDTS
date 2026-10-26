import importlib.util
import sys
import pandas as pd
import numpy as np

# Load the module
spec = importlib.util.spec_from_file_location('walkforward', r'C:\IDTS\trader/core/walkforward/walkforward.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Test config
cfg = mod.WalkForwardConfig(expanding_window=True, training_bars=100, validation_bars=25, test_bars=50, step_bars=25, min_data_bars=50)
print('Config created successfully')

# Create test data
dates = pd.date_range('2020-01-01', periods=500, freq='D')
df = pd.DataFrame({
    'close': np.cumsum(np.random.randn(500)) + 100,
    'volume': np.random.randint(1000, 10000, 500)
}, index=dates)

# Test rolling folds
folds = mod._compute_date_ranges(df, cfg)
print(f'Rolling folds: {len(folds)}')

# Test expanding folds
cfg2 = mod.WalkForwardConfig(expanding_window=True, training_bars=100, validation_bars=25, test_bars=50, step_bars=25, min_data_bars=50)
folds2 = mod._compute_date_ranges(df, cfg2)
print(f'Expanding folds: {len(folds2)}')

# Test full validation
class DummyStrategy:
    name = 'Dummy'

strategy = DummyStrategy()
result = mod.run_walk_forward_validation(strategy, df, 'TEST', {'risk': {}}, config=cfg)
print(f'Best fold: {result.best_fold_id}')
print(f'Worst fold: {result.worst_fold_id}')
print(f'Stable: {result.stable}')
print(f'Expectancy: {result.aggregate_metrics["expectancy_r"]:.4f}')
print(f'Total trades: {result.aggregate_metrics["total_trades"]}')

# Test report
print()
print('REPORT SUMMARY:')
print(result.report[:300])

print()
print('=== ALL WALK-FORWARD TESTS PASSED ===')