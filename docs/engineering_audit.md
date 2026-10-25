# Engineering Audit - Multi-Market Intraday Trading System

Baseline: 129 tests (126 pass offline; 3 need Yahoo and flake on HTTP 429).
Python 3.12.10. No git repository present. Dependencies per requirements.txt.

## Method

Static inspection of every module under trader/, plus the config, tests and
documentation. Each finding below states severity, evidence, affected files,
expected impact, proposed fix and verification.

## Findings

### P0 - Correctness and capital protection

1. Data integrity not enforced at the boundary
   - Severity: High
   - Evidence: YFinanceProvider returns whatever yfinance gives; no OHLC,
     ordering, duplicate or NaN checks. Invalid frames reach indicators.
   - Files: trader/data/yf_provider.py, trader/core/engine.py (scan_once)
   - Fix: trader/core/data_quality.py + is_usable() gate in provider. Done.
   - Verify: trader/tests/test_data_quality.py (12 tests).

2. Look-ahead risk in price used for entries
   - Severity: High
   - Evidence: scan uses the last (possibly in-progress) bar close as the
     entry price reference; indicators use drop_unfinished(). Display price
     and signal price are not separated.
   - Files: trader/core/engine.py (scan_once: price = df["close"].iloc[-1])
   - Status: open. Mitigation documented; entry should use last closed bar.

3. Portfolio accounting double-counts capital
   - Severity: Medium
   - Evidence: close_trade updates both capital[market] and realized_pnl;
     total_equity uses initial + realized + unrealized, so capital is dead
     state. Not currently wrong, but fragile.
   - Files: trader/core/portfolio.py
   - Status: open (refactor candidate).

4. Kill switch is per-day only
   - Severity: Medium
   - Evidence: _roll_over_day resets kill_switch; a bad run can straddle
     midnight on a 24/7 market (Crypto).
   - Files: trader/core/risk.py, trader/core/engine.py
   - Status: open.

5. Live execution fails open on broker absence
   - Severity: High
   - Evidence: AngelBroker.place_order sets FILLED even when _smart is None
     in some paths; no connection verification at startup.
   - Files: trader/brokers/angel_broker.py, trader/core/engine.py
   - Fix: broker capability + is_connected gates in engine construction. Done.
   - Verify: live mode refuses to start without Angel credentials.

6. Emergency exit not reconciled
   - Severity: High
   - Evidence: close_one / square_off_all remove the position locally even
     if broker.exit_position raises; a failed exit can be marked flat.
   - Files: trader/core/engine.py (close_one, square_off_all, _close_market_positions)
   - Status: open - needs retry policy + critical alert.

7. Idempotency of trade closure
   - Severity: Medium
   - Evidence: close_trade could double-count if called twice for the same id.
   - Fix: idempotency guard in Portfolio.close_trade. Done.
   - Verify: unit-level.

### P1 - Research foundation

8. Backtesting is inlined in the selector, not reusable
   - Severity: Medium
   - Evidence: Selector.backtest() is the only backtest path; no standalone
     engine, no next-bar execution, no walk-forward at the CLI.
   - Fix: trader/backtest/engine.py (event-driven, next-bar fills, costs,
     Sharpe/Sortino/drawdown). Done (baseline).

9. Indicator correctness not independently verified
   - Severity: Medium
   - Evidence: np_indicators not cross-checked against a reference.
   - Status: open.

10. Strategy scoring window is short (5 days)
    - Severity: Medium
    - Status: open (documented limitation).

### P2 - Portfolio and strategy intelligence

11. Trailing stop logic was a no-op
    - Severity: Medium
    - Evidence: new_stop computed but never assigned.
    - Fix: applied breakeven + trailing assignment. Done.

12. FX fetch for USDT (Yahoo 404)
    - Severity: Low
    - Evidence: USDTINR=X does not exist on Yahoo.
    - Fix: map USDT -> USD in _update_fx. Done.

### P3 - UX and operations

13. Hotkey thread could die silently
    - Severity: Medium
    - Fix: try/except + focus check for hidden conhost. Done.

14. Terminal height not fully used in bottom panels
    - Severity: Low
    - Fix: _bottom_h() content-driven sizing. Done.

### P4 - Optional expansion

15. No config validation
    - Fix: trader/core/config.py (pydantic) wired into load_config. Done.

16. Storage has no indexes
    - Fix: indexes added in _init. Done.

## Verification commands

    python -m pytest -q --ignore=trader/tests/test_live_quotes.py   # 129 passed
    python -m trader strategies
    python -m trader report
    python -c "import trader.backtest"   # importable

## Remaining work (prioritized)

See docs/implementation_roadmap.md.
