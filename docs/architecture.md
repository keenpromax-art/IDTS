# Architecture - Multi-Market Intraday Trading System

## Layers

1. Configuration (trader/config/*.yaml) - validated by trader/core/config.py.
2. Market data (trader/data/*) - DataProvider interface; yfinance + Angel One.
3. Data quality (trader/core/data_quality.py) - OHLC/ordering/duplicate/stale checks.
4. Market clock (trader/core/market_clock.py) - sessions, holidays, DST, always_open.
5. Strategy engine (trader/strategies/*) - single setups + composite rules + custom.
6. Selector (trader/core/selector.py) - regime, backtest, scoring, market-level gate.
7. Risk (trader/core/risk.py) - sizing, limits, kill switch, session edges.
8. Portfolio (trader/core/portfolio.py) - positions, trades, equity, FX conversion.
9. Execution (trader/brokers/*) - paper (default) / Angel (gated).
10. Persistence (trader/core/storage.py) - SQLite WAL, indexed.
11. Orchestration (trader/core/engine.py) - scan loop, quotes, FX, close handling.
12. Backtesting (trader/backtest/*) - event-driven, next-bar fills.
13. Presentation (trader/ui/*) - Rich Live dashboard + hotkey thread.

## Threading

- Scan thread: single writer of engine state; publishes last_scan per symbol.
- Price thread: quotes + exit checks under portfolio lock.
- Render thread: reads snapshots only; never network.
- Hotkey thread: flags and engine API calls only.

## Live-trading gates (all required)

1. --mode live
2. live_trading_confirmed: true
3. typed confirmation phrase
4. Angel credentials + is_connected()
5. broker.capabilities.live_allowed
6. startup reconciliation

Missing any gate keeps the system in paper mode (fail closed).
