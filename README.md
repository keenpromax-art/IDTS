# Trading System - Multi-Market Intraday Platform

## Overview
A terminal-based intraday trading system for 9 markets (India, Japan, Korea, Hong Kong, Australia, Brazil, Europe, USA, and Crypto) covering the clock from 04:30 to 01:30 IST, nearly without a break.

## Key Features
- **883 symbols** across 9 markets scanned on 60-second cycles
- **29 single setups** and **16 composite rules** with full explainability
- **Risk management** with ATR-based stops, position sizing, and daily loss limits
  - **0.40** weight on expectancy
  - **0.20** weight on profit factor
  - **0.15** weight on win rate
  - **0.15** weight on regime fit
  - **0.10** weight on confidence
  - **0.75%** risk per trade
  - **1.5%** max daily loss
  - **20%** max position size
  - **30 minutes** cooldown after loss
  - **first 5 min** no-entry window
  - **last 30 min** no-entry window
  - **12 minutes** before close square-off
  - **1.5R** minimum risk-reward
- **Paper trading by default** with seamless transition to live mode
- **Rich terminal dashboard** with real-time metrics and analytics
- **Live quotes** enabled with price refresh
- **kind** attribute for strategy regime classification

## Strategy Setup
- **29 single setups**: ema_cross, ema_20_50, ema_50_200, ema_8_21, ema_stack, macd_momentum, macd_zero, macd_divergence, vwap, vwap_band, supertrend, rsi_reversal, stochastic, cci, roc, mfi, williams_r, adx_di, psar, triple_screen, ichimoku, donchian, keltner, bb_expansion, bb_squeeze, volume_breakout, range_extreme, pivot, orb
- **16 composite rules**: `rule_trend_stack`, `rule_trend_stack_short`, `rule_golden_cross`, `rule_macd_ema`, `rule_macd_ema_short`, `rule_breakout_volume`, `rule_breakout_volume_short`, `rule_squeeze_release`, `rule_vwap_reversion`, `rule_vwap_reversion_short`, `rule_supertrend_volume`, `rule_rsi_mfi`, `rule_rsi_mfi_short`, `rule_ema20_50_cross`, `rule_keltner_trend`, `rule_obv_momentum`

## Architecture
- **9 markets**: India (NSE), Japan (TSE), Korea (KRX), Hong Kong (HKEX), Australia (ASX), Brazil (B3), Europe (LSE) UK, USA (NYSE/NASDAQ), and Crypto (Binance) CR
- **883 total symbols** with intelligent data fetching (yfinance + Angel One SmartAPI fallback)
- **numpy-optimized indicators** for performance (~6ms per symbol per cycle)
- **Cursor-based mini-backtest** for strategy ranking and scoring
- Every strategy declares a **kind** attribute: `trend`, `meanrev`, `breakout`, or `volume`

## Configuration
All settings in `trader/config/settings.yaml`:
- Mode: paper / live
- Capital per market:
  - **1,000,000** for India, Japan, Korea
  - **100,000** for Hong Kong, Australia, Brazil
  - **10,000** for Europe, USA, Crypto
- Risk limits and position sizing:
  - **0.40** weight on expectancy
  - **0.20** weight on profit factor
  - **0.15** weight on win rate
  - **0.15** weight on regime fit
  - **0.10** weight on confidence
  - **0.75%** risk per trade
  - **1.5%** max daily loss
  - **20%** max position size
  - **30 minutes** cooldown after loss
  - **first 5 min** no-entry window
  - **last 30 min** no-entry window
  - **12 minutes** before close square-off
  - **1.5R** minimum risk-reward
- Strategy enable/disable lists
- Rule parameters and thresholds
- Live trading confirmation required

## Live Trading Safety
Three independent gates required:
1. `--mode live` on command line
2. `live_trading_confirmed: true` in settings.yaml
3. Interactive confirmation phrase at startup

Any one missing ⇒ stays in paper mode.

## Research & Testing
- **129 tests** covering indicators, strategies, rules, selector, risk, currency, dashboard
- **152 tests** total (3 flaky due to Yahoo rate limits)
- Test snapshots at 100/120/160/200/240 columns, all themes
- Invariants: never raises, valid signals, proper sides/reasons

## Custom Strategy Workflow
- `python -m trader add-strategy` - interactive wizard to add new strategies
- `python -m trader strategies` - list all loaded strategies and rules
- Strategies can be defined in `custom_strategies.yaml` or as Python plugins in `trader/custom/`
- YAML DSL supports trigger, filters, stop, target, confidence, and indicator conditions
- `trigger`: crosses_above/crosses_below comparison of two indicators
- `filters`: additional conditions that must all hold
- `stop`: atr/mult, percent, points, swing, or indicator-based
- `target`: r_multiple/atc/percent/points-based
- `confidence`: 0.0 to 1.0 value
- Every strategy declares a **kind** attribute: `trend`, `meanrev`, `breakout`, or `volume`

## Live Quotes
- **live_quotes**: true (default)
- **price_refresh_seconds**: 1.0
- **quotes 0s** - instant quote metadata from yfinance fast_info
- Quote thread fetches ~30 symbols/second with 12 workers

## Disclaimer
Paper results do not guarantee live results. This is not financial advice. Trading involves risk. Use at your own risk.

## Hotkeys
`q` - quit | `x` - close position | `p` - previous | `Tab` - navigate | `r` - refresh

## Key Commands
- `python -m trader --mode paper` - start in paper mode
- `python -m trader demo` - demo mode
- `python -m trader report` - generate trade report

## Market Sessions
- India: open 09:15, close 15:30, INR, .NS
- Japan: open 09:00, close 15:00, JPY, .T (lunch 11:30-12:30)
- Hong Kong: open 09:30, close 16:00, HKD, .HK
- Australia: open 10:00, close 16:00, AUD, .AX
- Korea: open 09:00, close 15:30, KRW, .KS
- Brazil: open 10:00, close 17:00, BRL, .SA
- Europe: open 08:00, close 16:30, GBP, .L
- USA: open 09:30, close 16:00, USD
- Crypto: open 00:00, close 23:59, USDT

---
*Paper-first, terminal-based intraday trading system for nine markets.*