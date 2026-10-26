import sys
sys.path.insert(0, r'C:\IDTS')

# Manually test the _ensure_risk_cfg and AuditFramework initialization
from trader.core.audit import _ensure_risk_cfg, AuditFramework

# Test _ensure_risk_cfg first
cfg = {'risk': {'max_daily_loss_pct': 2.0}}
ensured = _ensure_risk_cfg(cfg)
print(f'_ensure_risk_cfg result: {ensured}')
print(f'has risk key: {"risk" in ensured}')

# Now create AuditFramework
try:
    fw = AuditFramework(
        cfg={'risk': {'max_daily_loss_pct': 2.0}, 'markets': {}},
        markets_cfg={},
        step_bars=50,
        expanding_window=False,
        min_data_bars=150
    )
    print('AuditFramework created successfully!')
except Exception as e:
    print(f'Error creating AuditFramework: {e}')
    import traceback
    traceback.print_exc()
"