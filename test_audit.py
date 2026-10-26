import sys
sys.path.insert(0, r'C:\IDTS')

from trader.core.audit import AuditFramework

# Minimal test
try:
    fw = AuditFramework(
        cfg={'risk': {'max_daily_loss_pct': 2.0}, 'markets': {}},
        markets_cfg={},
        step_bars=50,
        expanding_window=False,
        min_data_bars=150
    )
    print('Success!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()