# Trading System - Multi-Market Intraday Platform

## Overview
A terminal-based intraday trading system for 9 markets (India, Japan, Korea, Hong Kong, Australia, Brazil, Europe, USA, and Crypto) covering the clock from 04:30 to 01:30 IST, nearly without a break.

## Key Features
- **883 symbols** across 9 markets scanned on 60-second cycles
- **45 trading setups** (29 single + 16 composite rules) with full explainability
- **Risk management** with ATR-based stops, position sizing, and daily loss limits
- **Paper trading by default** with seamless transition to live mode
- **Rich terminal dashboard** with real-time metrics and analytics

## Architecture
- **9 markets**: India (NSE), Japan (TSE), Korea (KRX), Hong Kong (HKEX), Australia (ASX), Brazil (B3), Europe (LSE), USA (NYSE/NASDAQ), and Crypto (Binance)
- **883 total symbols** with intelligent data fetching (yfinance + Angel One SmartAPI fallback)
- **45 single setups** + **16 composite rules** = 65 total trading strategies
- **numpy-optimized indicators** for performance (~6ms per symbol per cycle)
- **Cursor-based mini-backtest** for strategy ranking and scoring

## Trading Logic
1. **Scan**: Fetch 5-minute candles for all open markets (no look-ahead)
2. **Score**: Run 45 setups + 16 composite rules against each symbol
3. **Rank**: Score every setup, rank by weighted expectancy
4. **Risk check**: Position sizing, limits, session windows
5. **Execute**: Paper or live order with stop attached

## Performance
- **~6ms** per symbol for all indicators (numpy core)
- **~14ms** for 45 setups against shared context
- **~20ms** mini-backtest per symbol
- **Full 883-symbol scan**: minutes (network dominates)
- **Quote refresh**: ~30 symbols/second with 12 workers

## Screenshots & Snapshot
![Dashboard Screenshot](docs/architecture.md)
![Scanner View](docs/implementation_roadmap.md)
![Risk Panel](docs/known_limitations.md)

## Configuration
All settings in 	rader/config/settings.yaml:
- Mode: paper / live
- Capital per market
- Risk limits and position sizing
- Strategy enable/disable lists
- Rule parameters and thresholds
- Live trading confirmation required

## Live Trading Safety
Three independent gates required:
1. --mode live on command line
2. live_trading_confirmed: true in settings.yaml
3. Interactive confirmation phrase at startup

Any one missing ⇒ stays in paper mode.

## Research & Testing
- **129 tests** covering indicators, strategies, rules, selector, risk, currency, dashboard
- **152 tests** total (3 flaky due to Yahoo rate limits)
- Test snapshots at 100/120/160/200/240 columns, all themes
- Invariants: never raises, valid signals, proper sides/reasons

## Disclaimer
Paper results do not guarantee live results. This is not financial advice. Trading involves risk.

---
*Paper-first, terminal-based intraday trading system for nine markets.*
