# Multi-Market Intraday Trading System

A terminal-based intraday trading system for **nine markets** — India (NSE),
Japan (TSE), Korea (KRX), Hong Kong (HKEX), Australia (ASX), Brazil (B3),
Europe (LSE), the USA (NYSE/NASDAQ) and Crypto (Binance) — covering the clock
from 04:30 to 01:30 IST, nearly without a break.

It scans **883 symbols** on every cycle, runs **29 single setups** and
**16 composite rules** (45 in total) against each symbol, picks the setup that
fits the current market regime, and opens and manages the trade — with a stop
loss attached from the moment it opens.

**Paper by default.** It runs with zero credentials against yfinance. Drop in
Angel One credentials and India switches to the SmartAPI feed automatically.
Live trading needs three independent confirmations before a single real order
can be placed.

> **This is a research and paper-trading tool, not an investment product.**
> See [Disclaimer](#disclaimer).

---

## Contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Markets](#markets)
- [How a trade happens](#how-a-trade-happens)
- [Strategies](#strategies)
- [Rules](#rules)
- [Add your own strategy](#add-your-own-strategy)
- [Risk management](#risk-management)
- [The dashboard](#the-dashboard)
  - [Session report](#session-report)
- [Configuration](#configuration)
- [Data sources](#data-sources)
- [Paper to live](#paper-to-live)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Performance](#performance)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Quick start

Requirements: **Python 3.11+**

```bash
pip install -r requirements.txt
```

Copy the environment template (optional — only needed for Angel One):

```bash
cp trader\.env.example trader\.env
```

Then run:

```bash
python -m trader --mode paper     # the live dashboard
python -m trader demo             # UI preview with fake data (no network)
python -m trader strategies       # list every setup that is loaded
python -m trader add-strategy     # build your own setup interactively
python -m trader report           # per-market / per-strategy results
```

`demo` is the fastest way to see the interface — it fills every panel with
synthetic data, works on a weekend, and needs no network.

---

## What it does

| | |
|---|---|
| **Scans** | Every symbol in every open market, on a 60-second cycle |
| **Decides** | Scores every setup that is signalling *right now* and picks the best fit |
| **Explains** | Every trade records the exact conditions that lined up |
| **Sizes** | Fixed-fractional risk: position size comes from the stop distance; crypto trades in coin fractions to the market's `qty_step` |
| **Protects** | ATR-based stop on every position, forced flat before each close |
| **Survives** | One bad symbol is logged and skipped; the scan and UI carry on |
| **Persists** | SQLite for trades, signals, orders and equity; CSV report on exit |

---

## Markets

| Market | Code | Exchange | Local hours | Lunch | Suffix | Currency | Capital | Symbols |
|---|---|---|---|---|---|---|---|---|
| India | IN | NSE | 09:15–15:30 IST | — | `.NS` | INR | 1,000,000 | 209 |
| Japan | JP | TSE | 09:00–15:00 JST | 11:30–12:30 | `.T` | JPY | 1,000,000 | 100 |
| Korea | KR | KRX | 09:00–15:30 KST | — | `.KS` | KRW | 1,000,000 | 99 |
| Hong Kong | HK | HKEX | 09:30–16:00 HKT | — | `.HK` | HKD | 100,000 | 100 |
| Australia | AU | ASX | 10:00–16:00 Sydney | — | `.AX` | AUD | 100,000 | 98 |
| Brazil | BR | B3 | 10:00–17:00 BRT | — | `.SA` | BRL | 100,000 | 41 |
| Europe | UK | LSE | 08:00–16:30 London | — | `.L` | GBP | 10,000 | 91 |
| USA | US | NYSE | 09:30–16:00 ET | — | — | USD | 10,000 | 100 |
| Crypto | CR | Binance | 00:00–23:59 UTC | — | — | USDT | 10,000 | 45 |

**Why these nine.** Together they cover the clock from 04:30 to 01:30 IST
without a break, and most of that window has two or three markets running at
once: Asia (Australia, Japan, Korea), then Asia + India, then India + Europe,
then Europe + Brazil + USA, and Crypto the whole time. When you are trading
one market, the system has a second market to look at.

**Korea and Brazil** were picked for depth over headlines: both are large,
heavily traded markets whose data is complete on yfinance, and each fills a
slot where the system previously had only one market open.

**Crypto runs 24 hours,** so it is modelled as an always-open market (`always_open`
in `markets.yaml`): it never shows a countdown, is never force-closed, and it
can be switched on and off by hand — `c` in the dashboard, `--no-crypto` on the
CLI or `crypto_enabled: false` in `settings.yaml`. Keep capital small; USDT
converts to INR at the USD rate.

Every symbol in a watchlist was checked to have 5-minute intraday history on
yfinance — a ticker that only has daily data is dead weight for an intraday
system, so those were removed. Brazil ends up with 41 rather than 100 because
Yahoo does not carry intraday history for most of the B3, and Europe with 91
because yfinance only carries it for the larger LSE names.

> **The 01:30–04:30 IST gap is real.** No major equity market trades then
> — the US has just closed and Australia has not opened. The only venues
> trading in that window (Mexico, Colombia) do not publish intraday data on
> yfinance, so they are not in the list. The system correctly shows every
> market as closed and sleeps until Australia opens. Crypto fills part of this
> gap if you switch it on.

Sessions are timezone-aware (`zoneinfo`), so daylight saving is handled
automatically — Hong Kong does not observe it, Sydney and London do. Weekends
and the holiday list in `trader/config/markets.yaml` are respected. Only
**open** markets are scanned; closed markets show a card and sleep.

P&L is tracked per market in its own currency internally and shown combined in
INR using FX rates: the engine fetches `{CURRENCY}INR=X` for every currency a
configured market trades in (`JPYINR=X`, `HKDINR=X`, `AUDINR=X`, `KRWINR=X`,
`BRLINR=X`, …) every five minutes, falling back to built-in static rates on
failure. Adding a market automatically adds its currency — no code change.
The dashboard converts **every displayed price** — scanner Last, active
entry/last/SL/TP, completed entry/exit, the detail view and all P&L — to INR
at those same rates, so numbers from different markets are directly
comparable. Execution still happens at the exchange's native price.

---

## How a trade happens

This is the core loop. Every cycle, for every symbol in every open market:

```
1.  fetch 5-minute candles            (Angel One for NSE, else yfinance)
2.  drop the in-progress bar            → no look-ahead, ever
3.  compute every indicator once        → shared context, ~6 ms
4.  run all 45 setups against it        → 29 setups + 16 rules
5.  score the ones signalling now       → expectancy, PF, win rate, regime, confidence
6.  pick the best above min_score       → e.g. "Keltner Break + EMA Stack, 0.87"
7.  risk check                          → size, limits, session window
8.  place the order with a stop         → paper or live broker
9.  publish the row to the dashboard    → the scanner fills in progressively
```

**Step 5 in detail.** The selector mini-backtests each signalling setup over
the last few sessions using the same stop/target/slippage/charges as live, and
scores it:

| Component | Default weight | Meaning |
|---|---|---|
| Expectancy | 0.40 | average R per trade |
| Profit factor | 0.20 | gross win ÷ gross loss |
| Win rate | 0.15 | share of winners |
| Regime fit | 0.15 | does the setup suit a trending or ranging market? |
| Confidence | 0.10 | how strong is the live signal? |

A setup must clear **every** gate before it may be traded:

| Gate | Default | |
|---|---|---|
| `min_trade_count` | 8 | enough history to judge it at all |
| `min_expectancy_r` | 0.0 | it must not have been losing |
| `min_profit_factor` | 1.1 | gross win must clearly beat gross loss |
| `min_win_rate` | 0.35 | a workable share of winners |
| `min_signal_confidence` | 0.55 | the live signal must be convincing too |
| `min_score` | 0.55 | the weighted score must clear the bar |

Zero expectancy is worth only 0.35 on the expectancy axis, not 0.5 — a setup
that has been going nowhere must not look half-good just because it carries
the heaviest weight.

**Regime fit** uses the setup's own declared `kind` (`trend`, `meanrev`,
`breakout`, `volume`), so a new setup gets sensible treatment with no config
change:

| Regime | trend | meanrev | breakout |
|---|---|---|---|
| Trending (ADX ≥ 25) | 1.0 | 0.3 | 1.0 |
| Ranging (ADX < 25) | 0.3 | 1.0 | 0.3 |

**Step 6 in detail — the market decides.** "A setup exists for this symbol" is
not a reason to trade it. Once the whole market has been scanned, the
candidates are ranked against each other and only the strongest few are taken:

| Gate | Default | |
|---|---|---|
| `min_market_score` | 0.60 | a market-wide bar, higher than the per-setup one |
| `max_trades_per_market_per_cycle` | 2 | at most this many new trades per market per cycle |

So a busy market produces two good trades, not forty mediocre ones.

**Two more rules that keep the noise down**

- **Mean-reversion setups refuse a strong trend.** Fading an extreme inside a
  trend is how a mean-reversion system loses money, so every mean-reversion
  setup (RSI, Stochastic, CCI, MFI, Williams %R, MACD divergence, VWAP band)
  declines to fire above ADX 30.
- **A setup is graded on history only.** The bar the live signal came from is
  excluded from its own backtest, so a setup cannot be graded on the very
  trade that is asking for an entry.

---

## Strategies

29 single setups — one idea, one trigger. Each declares its `kind`, which the
selector uses to match it to the market regime.

| Group | Setups |
|---|---|
| Moving averages | EMA 9/21 · **EMA 20/50** · **EMA 50/200** · EMA 8/21 · EMA-stack pullback |
| MACD | signal cross + volume · zero-line cross · divergence |
| Oscillators | Stochastic · CCI · ROC · MFI · Williams %R · RSI + band |
| Trend systems | Supertrend · Parabolic SAR · ADX +DI/−DI · Triple Screen · Ichimoku |
| Breakouts | Donchian · Keltner · BB expansion · BB squeeze · range extreme · ORB · pivot |
| Volume | volume-confirmed range break |
| VWAP | VWAP trend-pullback · VWAP deviation band |

The **EMA 20/50** pair is the medium-term moving average cross; **EMA 50/200**
is the long-term golden / death cross.

Config keys, for when you want to write an `enabled:` list:

| Group | Keys |
|---|---|
| Moving averages | `ema_cross` · `ema_20_50` · `ema_50_200` · `ema_8_21` · `ema_stack` |
| MACD | `macd_momentum` · `macd_zero` · `macd_divergence` |
| Oscillators | `stochastic` · `cci` · `roc` · `mfi` · `williams_r` · `rsi_reversal` |
| Trend systems | `supertrend` · `psar` · `adx_di` · `triple_screen` · `ichimoku` |
| Breakouts | `donchian` · `keltner` · `bb_expansion` · `bb_squeeze` · `range_extreme` · `orb` · `pivot` |
| Volume | `volume_breakout` |
| VWAP | `vwap` · `vwap_band` |

---

## Rules

Rules are **composite setups**. A rule only fires when *every* required
condition lines up on the **same closed bar**. Optional conditions add
confidence but are not required.

| Rule | Required conditions | Optional |
|---|---|---|
| `rule_trend_stack` | EMA20>EMA50>EMA200 · price>EMA50 · ADX≥22 · pullback to EMA20 | bar up · MACD hist + |
| `rule_trend_stack_short` | mirror image | bar down · MACD hist − |
| `rule_golden_cross` | EMA50 crossing above EMA200 · price>EMA50 · ADX≥20 | volume ≥1.2× |
| `rule_macd_ema` | MACD bullish cross · price>EMA20 · ADX≥20 | volume ≥1.0× |
| `rule_macd_ema_short` | mirror image | volume ≥1.0× |
| `rule_breakout_volume` | 20-bar high break · volume ≥1.5× · price>EMA50 · ADX≥25 | price>VWAP |
| `rule_breakout_volume_short` | mirror image | price<VWAP |
| `rule_squeeze_release` | BB squeeze before this bar · Keltner break · ADX rising | volume ≥1.3× |
| `rule_vwap_reversion` | 1.5 ATR below VWAP · RSI<40 · EMA20>EMA50 · still below VWAP | — |
| `rule_vwap_reversion_short` | mirror image | — |
| `rule_supertrend_volume` | Supertrend flip · volume ≥1.3× · ADX≥20 | — |
| `rule_rsi_mfi` | RSI<35 · MFI<30 · ADX<30 (ranging) | below lower band |
| `rule_rsi_mfi_short` | mirror image | above upper band |
| `rule_ema20_50_cross` | EMA20/50 cross · ADX≥20 | volume ≥1.0× |
| `rule_keltner_trend` | Keltner break · EMA20>EMA50>EMA200 | price>VWAP |
| `rule_obv_momentum` | OBV confirms the move · RSI 45–70 · ADX≥20 | MACD hist + |

Every trade records **which conditions aligned**, and the dashboard shows it:

```
Keltner Break + EMA Stack | score 0.99 | trending
EMA20 > EMA50 > EMA200 + close above Keltner upper + price above VWAP
```

Enable, disable and tune them in `trader/config/settings.yaml`:

```yaml
strategies:
  enabled: [ema_20_50, ema_50_200, macd_momentum, ...]   # omit the key for all
  disabled: []
rules:
  enabled: [rule_trend_stack, rule_golden_cross, ...]    # omit the key for all
  disabled: []
rule_params:
  rule_trend_stack:
    min_confidence: 0.55
    stop_atr: 1.5
    target_rr: 2.5
```

---

## Add your own strategy

Two ways, neither of which needs you to touch the engine.

### The wizard

```bash
python -m trader add-strategy
```

It asks a handful of questions — what fires it, what else must hold, where the
stop goes, where the target is — and writes the result into
`trader/config/custom_strategies.yaml`. Your setup is live on the next start
and is scored, ranked and traded under exactly the same rules as the built-in
ones.

### Writing it by hand

```yaml
strategies:
  my_ema_cross:
    label: "My EMA Cross"
    kind: trend                      # trend | meanrev | breakout | volume
    trigger:                         # the comparison that fires it
      left:  {indicator: ema, period: 5}
      op:    crosses_above
      right: {indicator: ema, period: 34}
    filters:                         # must ALL hold as well
      - {left: {indicator: adx}, op: ">=", right: 25}
      - {left: {indicator: rsi}, op: "<",  right: 70}
    stop:    {type: atr, mult: 1.5}   # atr | percent | points | swing | indicator
    target:  {type: r_multiple, value: 2.0}  # r_multiple | atr | percent | points
    confidence: 0.65
```

**Indicators:** `close` `open` `high` `low` `volume` `price` `ema` `rsi` `atr`
`adx` `macd` `bb` `vwap` `donchian` `supertrend` `psar` `stoch` `cci` `roc`
`mfi` `keltner`

**Operators:** `>` `>=` `<` `<=` `==` `crosses_above` `crosses_below`

For `macd`, `bb`, `donchian`, `supertrend`, `stoch` and `keltner` add a
`part:` — `macd: line | signal | hist`, `bb: upper | mid | lower`,
`donchian: high | low`, `supertrend: line | direction`, `stoch: k | d`,
`keltner: upper | mid | lower`.

### Writing it in Python

For anything the DSL cannot express, drop a file into `trader/custom/`. Any
`Strategy` subclass in it is discovered at start-up:

```python
# trader/custom/my_setup.py
from trader.core.models import Side
from trader.strategies.base import Strategy
from trader.strategies.context import make_signal


class MySetup(Strategy):
    name = "my_setup"
    kind = "trend"
    label = "My Setup"

    def generate(self, df, symbol="", market="", ctx=None):
        from trader.strategies.context import Ctx
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30 or c.atr <= 0:
            return None
        ema = c._ema(20)
        if not c.crossed_up(c.close, ema):
            return None
        entry = c.px
        stop = entry - 1.5 * c.atr
        return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                           entry + 2.0 * (entry - stop), confidence=0.6,
                           reason="close crossed above EMA20")
```

### See what is loaded

```bash
python -m trader strategies
```

lists every built-in setup, every rule and every one of yours, with its kind,
its label and whether it is on.

Your setups run automatically. To turn one off, list it under
`custom_strategies.disabled` in `settings.yaml`.

---

## Risk management

Every control below is on by default and applies to every order.

| Control | Default | |
|---|---|---|
| Risk per trade | 0.75% of equity | `position = (equity × risk%) ÷ (entry − stop)` |
| Max position notional | 20% of equity | a very tight stop cannot ask for a huge position |
| Stop loss | always set | ATR-based, or a structural level (EMA50, swing low) |
| Target | min 1.5R | rules often use 2.0–3.0R |
| Trailing stop | on | breakeven after +1R, then an ATR trail |
| Max concurrent positions | 6 total, 3 per market | |
| Max trades per symbol per day | 2 | |
| Max daily loss | 1.5% of that market's equity | triggers the kill switch: stop entries and flatten |
| Cooldown after a loss | 30 minutes | on that symbol |
| No new entries | first 5 min, last 30 min | of the session |
| Forced exit | 12 min before the close | Crypto is exempt — it is always open and lets scores breathe |
| New trades per market per cycle | 2 | only the best-ranked candidates are taken |
| Duplicate positions | not allowed | a symbol already open is not re-entered |

The daily loss limit is measured in the market's **own** currency against that
market's own equity, and it is a *daily* limit: when the day rolls over the
kill switch is released and the counters reset. The system is never left
halted for good.

The scan runs on a **steady cadence**: the cycle is exactly
`scan_interval_seconds`, not scan-time + interval, and the footer shows the
time to the next scan.

### Live quotes

A background thread refreshes the latest price every second for the symbols on
screen and for any open position, so:

- prices in the scanner and the open-position P&L **move every second**
- a stop or target is acted on within a second or two, not at the next scan
- the footer shows how fresh the quotes are (`quotes 0s`)

Open positions are always quoted first, then the symbols on screen. Press
`v` to switch live quotes off if you would rather not spend the bandwidth.

**Whole-market re-scan.** On top of that, the thread works through *every*
symbol of every open market in a rolling queue, so a whole market is re-quoted
end to end roughly every `fast_scan_seconds` (default 5). The footer shows the
progress (`scan 64/209`). The queue is worked through continuously rather than
in one long batch, so the scanner keeps moving the whole time.

```yaml
live_quotes: true
price_refresh_seconds: 1.0      # positions and on-screen rows
fast_scan_seconds: 5.0          # every symbol of every market
price_refresh_max_workers: 12
```

> The feed is the exchange's quote metadata, which is *fresher* than the last
> completed intraday bar but still delayed by the exchange (typically ~15
> minutes). Use it to watch prices move and to trigger exits promptly, not as
> a tick-accurate feed.

Press `f` (or `:scan`) to force a scan cycle to run immediately — the engine
wakes up, scans every open market and then returns to its regular cadence.

---

## The dashboard

```
  4  HEADER     mode badge · equity + session sparkline · day P&L · clock
                realised/unrealised · expectancy · risk · hi/lo · drawdown
                W/L · PF · avg W/L · kill switch · feed age · scan time
  ?  MARKETS    up to nine cards (status, countdown, breadth, day P&L,
                positions, loss budget) — or one dense line per market
                when vertical space is tight
  ?  SCANNER    open markets only, signals first, sorted by score; live
                metrics, ATR/ADX/RSI/VWAP, plan, R:R, setups, score,
                signal, age, sparkline and the gate reason
  ?  BOTTOM     ACTIVE | COMPLETED over RISK | LOG   (wide layout)
                ACTIVE, then COMPLETED | LOG          (narrow layout)
  1  FOOTER     hotkeys · command prompt · armed-close hint · scan stats
```

Panel heights are allocated by content and terminal height: the scanner and
bottom row are capped as a share of the budget, Active Trades shrinks first on
a short terminal, and the scanner gets any leftover height. The market section
uses full cards when they fit and falls back to one information-dense line per
market (then a packed chip strip) instead of clipping. The middle column is
split **Completed Trades** over **Risk**, sharing height 3:2 once something has
closed (1:1 while empty); Risk keeps its TOTAL row even when only four lines
are available. Active Trades reserves its TOTAL row and paginates the visible
positions, with the full count in the title.

The wide layout starts at 160 columns (140 with `ui.density: wide`, 200 with
`compact`). Width-dependent tables drop their lowest-priority columns rather
than overflow, so Symbol and P&L survive even at 100 columns.

Every price and P&L in the dashboard is converted to **INR** at the live cross
rate: scanner Last, active entry/last/SL/TP, completed entry/exit, the risk
panel's exposure/risk/reward, the detail view and the market cards. The account
is managed in INR, so an Australian stop and a US entry are read in the same
currency. Orders still execute at the exchange's native price.

### Panels

**Scanner** — one row per symbol, signalling rows first, then by score:

```
 Mkt  Symbol         Last       %Chg   RVOL   ATR%   ADX  RSI  VWAP%   Regime    Strategy   Plan                    R:R  Setups  Score     Signal  Age  Spark       Why
 IN   HDFCBANK.NS   ₹1,279.00  ▲0.20%  1.8x   1.4%    24   56  ▲0.50%  trending  MACD+EMA   ₹1,279/₹1,252/₹1,340   2.5     2    ▰▰▰▰▱ 0.80  ▲ LONG   12s  ▂▃▅▆▇█▅▃  MACD Cross 0.82
```

Live rows carry the full gate reason in *Why* (e.g. `needs 20 trades (8)`),
near-miss rows are dimmed, and while the first scan is still running the
prices already on the quote feed are shown as prefills.

**Active trades** — quantity, strategy, age, stop state and an `SL ●──── TP`
gauge with MFE/MAE excursion:

```
 Mkt  Symbol        Side    Qty  Strategy          Entry       Last        P&L      R     Age  Stop  Gauge         Exc
 IN   RELIANCE.NS  ▲ LONG    10  EMA 9/21 Cross    ₹2,500.00   ₹2,530.00   +₹300   +0.6R    5m   SL    ──────●─────  +0.6/+0.6
      TOTAL                                                                  +₹300
```

The TOTAL row is always reserved; if there are more positions than rows the
title shows the full count and the selected/first symbol's full setup label and
reason appear underneath when height allows. `SL` → `BE` → `TRAIL` tracks the
stop as it moves.

**Completed trades** — paged history (`n` / `b`) with quantity, strategy,
open time and age, plus a session strip (`N · win% · gross · fees · net · PF ·
expectancy · avg W/L`):

```
 Mkt  Symbol     Side      Qty  Strategy              Entry       Exit        Rsn     P&L       R     Open  Age
 IN   TCS.NS    ▼ SHORT      5  Opening Range Break…  ₹3,800.00   ₹3,720.00  TARGET   +₹388   +1.0R   09:14    30m
```

`TARGET` green · `SL` red · `TRAIL` blue · `SQ-OFF` amber · `KILL` red ·
`MANUAL` magenta.

**Risk** — what the open book can still lose or make, per market, in INR. A
summary line shows today's loss against the daily limit with a block gauge,
total open risk as % of equity, and the largest single-symbol concentration:

```
 Day loss 0.42% / 1.5% ▰▰░░░░░░ · at risk 0.31% · top RELIANCE.NS 38%
 Mkt   Exposure      Risk     Reward   %Eq   R:R
 US    ₹3,05,623     ₹739     ₹1,826   0.1   2.5
 BR    ₹27,740       ₹268     ₹806     0.1   3.0
 TOTAL ₹3,35,893     ₹1,065   ₹2,811   0.2   2.6
```

Exposure is marked at the live price; Risk and Reward are the distances from
that price to each stop and target, so they shrink as a trade walks towards
either. A stop the price has already passed counts as zero risk. The panel
shares the middle column with Completed Trades and drops its header before it
ever drops the TOTAL line on a short terminal.

**Log** — coloured level tag, timestamp, message. Exit lines keep the
exchange-native fill price: it is the execution record, while the panels above
it are INR-converted. `l` opens the full-screen log, `e` filters all/WARN/ERROR
and `w` toggles wrapping.

```
19:28:22  ORDER   OPEN Europe FCIT.L SHORT via vwap qty 15
19:27:31  SIGNAL  Europe FCIT.L SHORT via vwap qty 15
19:27:30  INFO    India closed, sleeping; next open 09:15
```

### Views

| Key | View |
|-----|------|
| `m` | Monitor — the default multi-panel layout |
| `i` | Positions — full table with entry/last/stop/target, R, age, stop state and gauge |
| `a` | Analytics — session equity curve, strategy leaderboard, exit reasons, P&L by hour |
| `y` | Diagnostics — signal funnel, timings, rejection counts, per-market data health |
| `l` | Log — full-screen, with level filter and wrapping |
| `r` | Report — session visual summary of every closed trade: P&L, win rate, curve, R distribution, market/strategy/exit breakdown and recent closes |
| `d` / `Enter` | Detail for the selected scanner row: plan, why, chart, indicators, setup stats, live position |
| `Esc` | Close a full-screen view or overlay |

Every view renders at any width from 100 to 240 columns and in all three
themes; empty states explain themselves (`No open positions`,
`No completed trades yet`, `Waiting for the first scan`).

### Session report

Press `r` from the dashboard — or run `:report` / `:view report` — to open a
read-only report of every trade closed in the current session. It uses the
same closed-trade records as the Completed panel, shows every amount in INR
after charges, and never changes positions, orders or strategy state. Press
`Esc` to return to the monitor.

The layout is kept under 80 columns so it reads cleanly on GitHub without
sideways scrolling:

```text
SESSION REPORT · closed trades
──────────────────────────────────────────────────────────────────────────────
Closed 27 · 10W/17L · win 37.0% · net -₹35,635 · PF 0.23
Gross +₹10,930 / -₹46,565 · charges ₹25,977 · avg trade -₹1,320
Avg win/loss · best/worst · long/short · hold · exit mix
──────────────────────────────────────────────────────────────────────────────
Cumulative P&L · after charges        │ R distribution
peak · low · now + braille curve      │ bins <= -1R … > 2R · AvgR
──────────────────────────────────────────────────────────────────────────────
By market                             │ By strategy
N · Win · Net · PF · AvgR             │ N · Win · Net · PF · AvgR
──────────────────────────────────────────────────────────────────────────────
Wide only (200+ cols): By exit · N · net · avg
Tall + wide: Recent closes · time · symbol · strategy · side · ...
──────────────────────────────────────────────────────────────────────────────
```

- **Summary** answers “how did the session go?”: closed count, wins/losses,
  win rate, net P&L after charges, profit factor, expectancy, gross win/loss,
  charges, average trade, average win/loss, best and worst trade, long/short
  split, average hold time and the exit mix.
- **Cumulative P&L** is the running total of closed-trade P&L after charges,
  with the peak, low and current value above a braille equity curve.
- **R distribution** bins every closed trade by R multiple (`<= -1R`, `-1..0`,
  `0..1`, `1..2`, `> 2R`) and shows the average R, so a positive P&L made from
  one large winner is visible at a glance.
- **By market / By strategy** show trade count, win rate, net INR, profit
  factor and average R for each group. Below 110 columns the strategy names
  use a shorter compact column.
- **By exit** (200+ columns) shows count, net INR and average INR by exit
  reason. **Recent closes** appears when the terminal is also tall enough and
  lists the newest trades first with time, market, symbol, strategy, side,
  exit, quantity, hold time, net and R.
- An empty session shows a short explanation instead of an empty grid; the
  report fills in as trades close.

### Hotkeys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `x` | Square-off all positions |
| `p` | Pause / resume scanning |
| `v` | Live quotes on / off |
| `c` | Crypto market on / off |
| `Tab` | Cycle the market filter (All open → AU → JP → KR → HK → IN → BR → UK → US → CR) |
| `Backspace` | Cycle the market filter backwards |
| `←` `→` | Page the scanner table |
| `↑` `↓` | Select a scanner row |
| `d` / `Enter` | Detail view of the selected symbol (plan, chart, indicators, setup stats, live position) |
| `m` / `i` / `a` / `y` / `l` / `r` | Monitor / Positions / Analytics / Diagnostics / Log / Report view |
| `Esc` | Close a full-screen view or overlay |
| `n` / `b` | Next / previous page of completed trades |
| `e` / `w` | Cycle log level (all → WARN → ERROR) / toggle log wrapping (in the Log view) |
| `f` | Force a scan right now |
| `s` | Cycle the scanner sort: score → %change → signal → symbol |
| `g` | Cycle the signal filter: all → longs → shorts |
| `1`…`9` | Close position N — press once to arm, press again within 6s to confirm |
| `:` or `/` | Command bar |
| `h` or `?` | Help overlay with every key and command |

Keys work in the command bar and overlay too (`q` always quits), and the
hotkey thread logs internal errors to the Log panel instead of dying silently.

### Command bar

Press `:` and run one of:

| Command | Action |
|---------|--------|
| `:filter <text>` | Filter the scanner by symbol / strategy (`:filter off` clears) |
| `:sort score\|chg\|signal\|symbol` | Change the scanner sort |
| `:longs` `:shorts` `:all` | Show only long / short / all signals |
| `:close N` or `:close SYMBOL` | Close one position (paper) |
| `:flat` | Square-off every position |
| `:pause` / `:resume` | Pause or resume scanning |
| `:crypto on` / `:crypto off` | Turn the whole crypto market on / off |
| `:kill` | Trip the daily kill switch now |
| `:scan` | Force a scan |
| `:report` | Open the session report |
| `:view monitor\|positions\|analytics\|diagnostics\|log\|report` | Switch the full-screen view |
| `:log all\|warn\|error\|wrap` | Set the log filter or toggle wrapping |
| `:density auto\|compact\|normal\|wide` | Change the wide-layout threshold |

A bare word (e.g. `sol`) filters the scanner on it. Empty inputs are treated
as a text filter, so it reads like a search box.

### Empty states

Tables explain themselves instead of showing a bare header:

```
India closed, opens in 14h29m
First scan running…
Scanning Europe 45/100
No open positions
No completed trades yet
```

The scan publishes rows per symbol, so the scanner fills in progressively
while a cycle is still running. The footer shows the live progress.

### Themes

Set `theme:` in `trader/config/settings.yaml`:

| Value | |
|---|---|
| `default` | green / red on dark borders |
| `colorblind` | blue / orange instead of green / red |
| `mono` | no colour, for limited palettes |

Colour is **never** the only signal: P&L always carries a `+`/`−` sign and
`▲`/`▼` arrows, and signals are always labelled.

Numbers are right-aligned, every market's money is converted to INR before it
is displayed (lakh grouping: `₹38,07,404`), and empty statistics render `—`
rather than `0.00%`.

---

## Configuration

`trader/config/settings.yaml` — everything is here.

| Key | Default | |
|---|---|---|
| `mode` | `paper` | `paper` or `live` |
| `live_trading_confirmed` | `false` | must be `true` for live mode |
| `crypto_enabled` | `true` | master switch for the Crypto market |
| `theme` | `default` | `default` / `colorblind` / `mono` |
| `ui.density` | `auto` | wide-layout threshold: `auto` (160), `compact` (200), `normal` (160), `wide` (140) |
| `scan_interval_seconds` | `60` | cycle length |
| `candle_interval` | `5m` | `1m` or `5m` |
| `lookback_days` | `5` | days of history to fetch |
| `risk.*` | see [risk management](#risk-management) | position sizing and limits |
| `capital.*` | per market | starting paper capital |
| `strategies.enabled` | all | which single setups run |
| `strategies.disabled` | — | setups to switch off |
| `rules.enabled` | all | which composite rules run |
| `rule_params.*` | — | per-rule overrides |
| `custom_strategies.enabled` / `.disabled` | all on | which of *your* setups run |
| `selector.min_trade_count` | `8` | backtest trades needed to judge a setup |
| `selector.min_expectancy_r` | `0.0` | a setup must not have been losing |
| `selector.min_profit_factor` | `1.1` | gross win must beat gross loss |
| `selector.min_win_rate` | `0.35` | a workable share of winners |
| `selector.min_signal_confidence` | `0.55` | the live signal must be convincing |
| `selector.min_score` | `0.55` | score a setup needs |
| `selector.min_market_score` | `0.60` | market-wide bar for taking a trade |
| `selector.max_trades_per_market_per_cycle` | `2` | new trades per market per cycle |
| `selector.score_refresh_cycles` | `5` | cycles between re-scoring |
| `selector.weights.*` | see [how a trade happens](#how-a-trade-happens) | scoring weights |
| `live_quotes` | `true` | the fast quote thread |
| `price_refresh_seconds` | `1.0` | positions and on-screen rows |
| `fast_scan_seconds` | `5.0` | whole-market re-quote period |
| `paper.slippage_bps` | `5` | simulated slippage |
| `paper.brokerage.*` | per market | brokerage, STT, exchange charges, GST |
| `angel.*` | — | SmartAPI rate limit and instrument cache |
| `db_path` | `data/trader.db` | SQLite file |
| `csv_dir` | `data/reports` | session CSV output |

`trader/config/markets.yaml` — sessions, timezones, holidays per market. An
optional `qty_step` (the Crypto market sets `0.0001`) makes a market size in
coin fractions instead of whole units.
`trader/config/watchlists.yaml` — the symbols to scan.
`trader/config/custom_strategies.yaml` — your own setups.

---

## Data sources

| Market | Source |
|---|---|
| India | **Angel One SmartAPI** when credentials work, otherwise yfinance |
| All others (incl. Crypto) | yfinance |

The instrument master (symbol → token) is downloaded once a day and cached.
If Angel One authentication fails, the system falls back to yfinance, stays in
paper mode and logs a warning.

### Market close handling

A market we are not scanning cannot protect its positions, so:

- when a session ends, every position in that market is closed at the last
  price seen
- the same happens during the Japan lunch break (11:30–12:30)
- positions are already flat 12 minutes before the close
- a position restored after a restart is closed if its market is not trading
- Crypto is exempt: it never closes, so nothing is force-flattened

Set `risk.close_on_market_closed: false` in `settings.yaml` to hold positions
through a break instead.

### yfinance limits

Intraday history is limited: **1m ≈ 7 days**, **5m ≈ 60 days**. Some
exchanges also publish with a delay — the dashboard shows the data age on
every row and in each market card, and marks stale data with `*`.

---

## Paper to live

Live mode needs **all three** of these. Any one missing keeps the system in
paper mode.

1. `python -m trader --mode live`
2. `live_trading_confirmed: true` in `trader/config/settings.yaml`
3. The interactive confirmation phrase typed at startup

Checklist:

- [ ] Run paper mode for a full session and read the log
- [ ] Check `python -m trader report` — which setups actually work for you
- [ ] Add Angel One credentials to `trader/.env`
- [ ] Set `live_trading_confirmed: true`
- [ ] Start with a small `capital.*` value
- [ ] Run `python -m trader --mode live` and confirm at the prompt
- [ ] Watch the header: it turns into a solid red **LIVE** bar
- [ ] Keep the kill switch (`x`) within reach

In live mode the Angel Broker places real `INTRADAY` (MIS) orders. The
strategy, risk, selector and dashboard code is identical — only the broker
changes.

---

## Project layout

```
trader/
  __main__.py            CLI entry point
  config/
    settings.yaml        mode, risk, timeframes, strategies, rules, theme
    markets.yaml         sessions, timezones, holidays
    watchlists.yaml      symbols per market
  core/
    models.py            Candle, Signal, Order, Position, Trade, Regime
    market_clock.py      session status and countdowns
    indicators.py        public (pandas) indicator API
    np_indicators.py     numpy indicator core — the hot path
    prices.py            PriceTicker -- the fast quote thread
    engine.py            the scan loop
    risk.py              sizing, limits, kill switch
    selector.py          strategy ranking
    portfolio.py         positions, trades, equity
    storage.py           SQLite persistence
  data/
    base.py              DataProvider interface
    yf_provider.py       yfinance, batched
    angel_provider.py    SmartAPI candles + instrument master
    factory.py           picks the provider per market
  brokers/
    base.py              Broker interface
    paper_broker.py      simulated fills, slippage, charges
    angel_broker.py      real MIS orders
  strategies/
    base.py              Strategy interface (+ kind)
    context.py           Ctx -- all indicators for one symbol, computed once
    ema_family.py        EMA 9/21, 20/50, 50/200, 8/21, stack
    macd_family.py       MACD cross, zero-cross, divergence
    oscillators.py       Stochastic, CCI, ROC, MFI, Williams %R, RSI
    breakouts.py         Donchian, Keltner, BB, volume, PSAR, ADX, Ichimoku
    momentum.py          VWAP, ORB, Supertrend, squeeze, pivot
    rule.py              the rules engine
    rules.py             16 concrete composite setups
  custom/                your own Python strategies (drop a file in)
  ui/
    theme.py             Rich theme, three variants
    format.py            money, P&L, bars, gauges, chips
    dashboard.py         the Rich Live layout + hotkey thread + command bar
  config/
    custom_strategies.yaml   your own setups, in YAML
  tests/
```

`design.md` has the full architecture, including the data flow, the rules
engine, the selection maths, the performance budget and an explicit list of
known weak points for a reviewer to challenge.

---

## Testing

```bash
python -m pytest -q
```

**152 tests** in ~75 s. Three of them (`test_live_quotes.py`) hit Yahoo
directly and can 429 under rate limiting — rerun those separately. The tests
cover the market clock, indicators, all 45 setups, all 16 rules, the
rules engine semantics, the selector's strictness gates, the market-close
behaviour, custom strategies, risk limits, currency handling, storage, and
dashboard snapshots at 100/120/160/200/240 columns, in every theme, with
empty, sparse and 15-position sessions.

The README is tested too — `test_readme.py` fails if a default in
`settings.yaml` changes and the README has not caught up.

---

## Performance

| Stage | Cost |
|---|---|
| All indicators for one symbol | ~6 ms |
| 45 setups against that shared context | ~14 ms |
| Mini-backtest (one context, replayed bar by bar) | ~20 ms |
| 883 symbols across markets | minutes, dominated by network |

The trick that keeps this fast: every indicator is computed **once** per
symbol with numpy, and the mini-backtest replays that single context with a
cursor instead of rebuilding it for every bar. Adding setups is nearly free.

The first cycle after startup is the slowest — it downloads history for every
symbol. Later cycles reuse the cache.

---

## Known limitations

The honest list — improvements are welcome in exactly these places:

- **yfinance is a single point of failure.** Every market except India rides
  on it; aggressive polling meets HTTP 429 rate limits (three tests currently
  fail on this). No retry-with-jitter policy on the quote thread yet.
- **The mini-backtest window is short.** `lookback_days: 5` means the
  selector's expectancy/PF/win-rate gates judge a setup on a handful of
  sessions — statistically thin, though at least every gate is required.
- **No walk-forward or paper/equity curve sanity check** on the selector's
  scores; scores refresh every 5 cycles but nothing detects regime shifts.
- **Windows-only hotkeys.** `msvcrt` input only works in a real Windows
  console; on Linux/macOS the dashboard read-only still works but keys do
  nothing. Rendering itself is platform-neutral and all views have tests that
  render them without a terminal.
- **Crypto never sleeps.** Position limits and the daily kill switch reset on
  midnight boundaries per market timezone; a 24/7 market's "day" edges are
  otherwise unguarded.
- **One-writer thread model.** The engine thread writes all state; hotkeys and
  the quote thread mutate only flags and a lock-guarded portfolio. Simple and
  safe, but the engine's scan cost caps the whole system's responsiveness.

---

## Troubleshooting

| Symptom | |
|---|---|
| Scanner empty, "First scan running…" | the first cycle is downloading history; watch the footer progress |
| `no data` for a symbol | the ticker does not exist on yfinance — check the spelling and suffix |
| Everything shows `*` | the data is stale; yfinance delays some exchanges |
| Angel One not being used | check `trader/.env` and `logs/trader.log` for the login result |
| Live mode refuses to start | one of the three live gates is missing |
| Terminal looks garbled | the terminal is narrower than 120 columns; the tables drop columns to fit |
| Hotkeys do nothing | run it in a Windows terminal; check the Log panel for `hotkey error` lines |

---

## Disclaimer

**Paper results do not guarantee live results.**
This software is not financial advice.
Trading leveraged and intraday instruments carries a substantial risk of loss
and is not suitable for everyone. You are responsible for every order the
system places, including the ones it places on your behalf in live mode.
Use at your own risk.
