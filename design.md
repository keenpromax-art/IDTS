# Design — Multi-Market Intraday Trading System

Paper-first, terminal-based intraday trading system for nine markets:
India (NSE), Japan (TSE), Korea (KRX), Hong Kong (HKEX), Australia (ASX),
Brazil (B3), Europe (LSE), USA (NYSE/NASDAQ) and Crypto (Binance).

- **Paper by default.** Live trading requires three independent gates.
- **Angel One first for India.** yfinance is the fallback for every market.
- **Rules, not vibes.** A trade only fires when a named setup's conditions
  actually line up on the same closed bar.

---

## 0. How to review this document

This document is written so a reviewer (human or AI) can challenge it without
reading the code first. Three things to know up front:

1. **Every number here is real.** 883 symbols, 45 setups (29 singles + 16
   rules), 60-second scan cadence, 5-minute candles, ~6 ms per indicator pass.
   The README mirrors most of them and a test (`test_readme.py`) fails the
   suite if the two drift apart.
2. **§22 lists the known weak points**, ranked by how much they limit the
   system, followed by concrete review questions. Optimisation work that
   starts elsewhere is mostly effort spent on things that already work.
3. **§23 lists what a reviewer should poke at**, and points to the file where
   each answer lives.

---

## 1. Facts at a glance

| | |
|---|---|
| Markets | 9 (AU, JP, KR, HK, IN, BR, UK, US — equities; CR — Binance perps/spot tickers) |
| Symbols total | 883 (India 209, Japan 100, Korea 99, HK 100, AU 98, BR 41, UK 91, US 100, CR 45) |
| Setups | 45 = 29 single + 16 composite rules |
| Cadence | 60 s scan cycle (offset from cycle *start*), 5 m candles, 1 s quote refresh |
| Sessions | ~04:30 → 01:30 IST covered; Crypto never closes |
| Threads | scan (single writer), price ticker, render, hotkey, CLI |
| Storage | SQLite (WAL): signals, orders, trades, equity, open positions |
| Tests | 129 (three need Yahoo directly and flake on HTTP 429) |
| Line count | ~12,000 LOC Python, stdlib + rich + yfinance + pandas/numpy + scipy (optional) |

---

## 2. Runtime architecture

```
                        ┌──────────────────────────────┐
  keyboard (msvcrt) ──▶ │  Hotkey thread                │
                        └───────────────┬──────────────┘
                                        │ pause / square-off / filter /
                                        │ sort / page / commands / crypto
                                        ▼
  ┌──────────────┐   candles    ┌──────────────────┐   orders   ┌──────────────┐
  │  yfinance    │────────────▶ │                  │──────────▶ │  PaperBroker │
  │  SmartAPI    │              │   Scan engine    │            └──────────────┘
  └──────────────┘              │   (one thread)   │            ┌──────────────┐
                                │                  │──────────▶ │  AngelBroker │
  ┌──────────────┐              │  per market:     │            └──────────────┘
  │  SQLite      │◀──────────── │  fetch → signal  │
  │  trades /    │   persist    │  → select → risk │            ┌──────────────┐
  │  signals /   │              │  → size → order  │──────────▶ │   Selector   │
  │  equity      │              └────────┬─────────┘            │  (ranking)  │
  └──────────────┘                       │                      └──────────────┘
                                         │ snapshot (thread-safe copy)
                                         ▼
                               ┌───────────────────┐         ┌──────────────┐
                               │  Dashboard         │  flags  │ PriceTicker  │
  ┌──────────────────┐         │  render thread      │◀─────── │  thread pool │
  │  Quote thread    │ prices  │  (Rich Live, 4 fps) │         │  ~30 sym/s   │
```

**Threading rules**

| Thread | Owns | Rule |
|---|---|---|
| Scan | engine state | Single writer. Publishes `last_scan[market]` per symbol. |
| Render | nothing | Reads a snapshot only. Never calls the network. |
| Hotkey | flags only | Sets `paused` / filter / sort / page; asks the engine to act via its API (which is what the CLI does too). Never mutates positions directly. |
| Price ticker | latest prices | Writes `latest_prices` and `price_history` under the portfolio lock; may fire stop/target exits. |

The dashboard **must never block on network I/O**. All fetching happens on the
scan and price threads; the render thread only reads dictionaries that are
replaced atomically (`self.last_scan[market] = list(rows)`).

**Interactive layer.** The hotkey thread (`dashboard._hotkey_loop`) debounces
key repeats, handles the command bar (a small input state machine), overlays
(help / detail, `dashboard._view`), breaks out of the render's scan sleep via
`engine.scan_now` (`threading.Event`), and hands the engine only flag changes
and engine-API calls: `_set_crypto` → engine market filtering; `_quick_close`
/ `:close` → `engine.close_one(pos)` (full broker + portfolio + storage +
log flow, identical to a target stop). Per-key exceptions are caught and
logged as `hotkey error: …` so a bug can never silently kill key input again.
The window-focus check accepts keys when the console window is not visible
(Windows Terminal hosts the console in a hidden conhost window that never
matches the foreground handle).

---

## 3. Module map

```
trader/
  __main__.py            CLI: --mode paper|live, report, demo
  config/
    settings.yaml        mode, risk, timeframes, strategies, rules, theme
    markets.yaml         sessions, timezones, holidays per market
    watchlists.yaml      symbols per market
  core/
    models.py            Candle, Signal, Order, Position, Trade, Regime
    market_clock.py      session status, countdowns, holidays, always_open
    indicators.py        public (pandas) indicator API
    np_indicators.py     numpy indicator core — the hot path
    prices.py            PriceTicker — 1 s refresh + rolling whole-market queue
    engine.py            scan loop orchestrator + FX refresh + close_one
    risk.py              sizing, limits, kill switch
    selector.py          strategy ranking (explainable)
    portfolio.py         positions, trades, equity, P&L, FX conversion
    storage.py           SQLite persistence
    custom.py            user-defined strategies (YAML DSL + plugins)
    strategy_cli.py      `strategies` and `add-strategy` commands
  data/
    base.py              DataProvider ABC
    yf_provider.py       yfinance + batch download + cache
    angel_provider.py    SmartAPI candles + instrument master
    factory.py           picks Angel for NSE if creds work, else yfinance
  brokers/
    base.py              Broker ABC
    paper_broker.py      simulated fills, slippage, brokerage + taxes
    angel_broker.py      real MIS orders (disabled unless LIVE)
  strategies/
    base.py              Strategy ABC (+ `kind`)
    context.py           Ctx — all indicators for one symbol, computed once
    ema_family.py        EMA 9/21, 20/50, 50/200, 8/21, stack pullback
    macd_family.py       MACD cross, zero-cross, divergence
    oscillators.py       Stochastic, CCI, ROC, MFI, Williams %R, RSI
    breakouts.py         Donchian, Keltner, BB expansion, volume, PSAR,
                         ADX+DI, Ichimoku, triple screen, range extreme
    momentum.py          VWAP, VWAP band, ORB, Supertrend, BB squeeze, pivot
    rule.py              Rule engine (conditions + groups)
    rules.py             16 concrete composite setups
  custom/                user Python strategies, auto-discovered
  ui/
    theme.py             Rich Theme: default / colorblind / mono
    format.py            fmt_money, fmt_pnl, bar, gauge, sparkline, chips
    dashboard.py         Rich Live layout, hotkey thread, command bar,
                         overlays, sort/filter, selection, quick-close
  tests/                 129 tests; README defaults are asserted too
```

---

## 4. Data flow — one scan cycle

```
for each market:
    status = clock.status(market)
    if closed  → log once, sleep (crypto: always_open → never closed)
    if open:
        candles = provider.get_candles(symbol)      # batch where possible
        closed   = drop_unfinished(candles)          # ← no look-ahead
        ctx      = Ctx(closed)                       # all indicators, once
        signals  = {s.generate_from(ctx) for s in setups}
        best     = selector.evaluate(symbol, signals)   # scored + explained
        if best and risk.can_open(...):
            qty   = risk.position_size(equity, entry, stop)
            order = broker.place_order(order)
            persist
```

### 4.1 No look-ahead

`drop_unfinished()` removes the last bar before any indicator runs. yfinance
and SmartAPI both return the *in-progress* bar as the final row, so using it
would leak future information into the signal.

Rules:

- Indicators, strategies, rules and the mini-backtest all read `closed` only.
- The **live price** shown in the UI may come from the in-progress bar — that
  is a display concern, not a signal concern.

### 4.2 Incremental publishing

`self.last_scan[market] = list(rows)` runs after every symbol, so the scanner
table fills in progressively during a long cycle. The footer shows
`scanning Europe 45/100`.

### 4.3 Forced scans and day rollover

`engine.scan_now` (a `threading.Event`) is set by the hotkeys `f` / `:scan`;
the scan loop breaks its sleep the moment it is set, so the extra wait after a
manual scan is at most a fraction of a second, and the regular cadence resumes
afterwards. Midnight rollovers are per-market (that market's own timezone
defines "today" for its daily-loss counter, trade counts and cooldowns).

---

## 5. Strategy system

### 5.1 Two tiers

**Single setups (29)** — one idea, one trigger.

| Group | Strategies |
|---|---|
| Moving averages | `ema_cross` (9/21), `ema_20_50`, `ema_50_200`, `ema_8_21`, `ema_stack` |
| MACD | `macd_momentum`, `macd_zero`, `macd_divergence` |
| Oscillators | `stochastic`, `cci`, `roc`, `mfi`, `williams_r`, `rsi_reversal` |
| Trend systems | `supertrend`, `psar`, `adx_di`, `triple_screen`, `ichimoku` |
| Breakouts | `donchian`, `keltner`, `bb_expansion`, `bb_squeeze`, `range_extreme`, `orb`, `pivot` |
| Volume | `volume_breakout` |
| VWAP | `vwap`, `vwap_band` |

**Rules (16)** — composite setups that require **several conditions to align
simultaneously** on the same closed bar. This is the "numbers aligned" layer.

### 5.2 The `kind` attribute

Every strategy declares how it behaves:

```python
class EMACrossStrategy(Strategy):
    name = "ema_cross"
    kind = "trend"      # trend | meanrev | breakout | volume
```

The selector uses `kind` to match a strategy to the market regime, so a new
strategy gets sensible treatment without touching `settings.yaml`. The legacy
`trend_strategies` / `meanrev_strategies` config lists remain only as a
fallback for strategies with no `kind`.

### 5.3 The Ctx — compute once, share everywhere

```python
ctx = Ctx(closed, symbol, market)   # ~6 ms for all indicators
signal = strategy.generate_from(ctx)
```

`Ctx` holds every indicator as a numpy array, memoised. A symbol costs one
indicator pass no matter how many setups inspect it. Without this, 45 setups ×
883 symbols would be ~40,000 redundant indicator computations per cycle.

**Performance rule:** any indicator used by more than one setup belongs in
`Ctx`. Strategies must not call `trader.core.indicators` (the pandas API) in
the hot path — they use the numpy arrays on `ctx`.

### 5.4 The cursor — replaying bars for the mini-backtest

All indicators are *causal*: they only look backwards. So the value an
indicator holds at bar `i` is identical whether it was computed over the whole
frame or over `df[:i+1]`.

`Ctx` therefore has a **cursor**:

```python
ctx.set_bar(i)     # the context now behaves as if bar i were the latest
```

The mini-backtest builds **one** context and walks the cursor forward, instead
of rebuilding a context for every bar:

```python
ctx = Ctx(df, symbol)
for i in range(n):
    ctx.set_bar(i)
    sig = strategy.generate_from(ctx)
```

That turned the backtest from 570 ms to 20 ms per symbol — the difference
between a scan that finishes and one that does not.

Every accessor is cursor-aware: `ctx.px`, `ctx.atr`, `ctx.val(series)`,
`ctx.crossed_up(a, b)`, `ctx.session_bars()`, `ctx.prev_session_bars()`.

### 5.5 Breakouts compare against prior bars

A breakout compares the current close to the range that existed **before** the
bar:

```python
ctx.donch_hi_prev   # rolling_max(high, 20) shifted by one bar
```

Using the unshifted level (which includes the current bar) makes a break
mathematically near-impossible. `Ctx` exposes both `donch_hi` and
`donch_hi_prev`, and the breakout setups and rules use the prior level.

---

## 6. The rules engine

### 6.1 Model

```python
Condition(key, text, test, required=True, group="")
Rule(name, label, conditions, side="auto", kind="trend",
     stop_atr=1.5, target_rr=2.0, min_confidence=0.5, min_optional=0,
     stop_ref="atr", bull_keys=(), bear_keys=())
```

- A **condition** is a named boolean test over a `Ctx`.
- A **group** is a set of mutually exclusive conditions (e.g. "close above
  Keltner" vs "close below Keltner"). A group passes when **at least one**
  member passes.
- Conditions without a group are **AND**-required.
- Optional conditions (`required=False`) only add confidence.

### 6.2 Evaluation

```
evaluate(ctx):
    every ungrouped required condition must pass          → else None
    every required group needs >= 1 member to pass        → else None
    optional conditions add confidence
    confidence = min(1, 0.45 + 0.55 * matched / total)
    confidence < min_confidence                            → None
    side = long / short / auto (from the "direction" group)
    → (side, [matched condition texts], confidence)
```

The returned condition texts become the signal's `reason`, so the dashboard can
show exactly *why* the system wanted to trade:

```
EMA20 > EMA50 > EMA200 + price above EMA50 + ADX >= 22 + pullback to EMA20
```

### 6.3 Rule catalogue

| Rule | Required conditions | Optional |
|---|---|---|
| `rule_trend_stack` | EMA20>EMA50>EMA200, price>EMA50, ADX≥22, pullback to EMA20 | bar up, MACD hist + |
| `rule_trend_stack_short` | mirror image | bar down, MACD hist − |
| `rule_golden_cross` | EMA50 crossing above EMA200, price>EMA50, ADX≥20 | volume ≥1.2× |
| `rule_macd_ema` | MACD bullish cross, price>EMA20, ADX≥20 | volume ≥1.0× |
| `rule_macd_ema_short` | mirror image | volume ≥1.0× |
| `rule_breakout_volume` | 20-bar high break, volume ≥1.5×, price>EMA50, ADX≥25 | price>VWAP |
| `rule_breakout_volume_short` | mirror image | price<VWAP |
| `rule_squeeze_release` | BB squeeze before this bar, Keltner break, ADX rising | volume ≥1.3× |
| `rule_vwap_reversion` | 1.5 ATR below VWAP, RSI<40, EMA20>EMA50, still below VWAP | — |
| `rule_vwap_reversion_short` | mirror image | — |
| `rule_supertrend_volume` | Supertrend flip, volume ≥1.3×, ADX≥20 | — |
| `rule_rsi_mfi` | RSI<35, MFI<30, ADX<30 (ranging) | below lower band |
| `rule_rsi_mfi_short` | mirror image | above upper band |
| `rule_ema20_50_cross` | EMA20/50 cross, ADX≥20 | volume ≥1.0× |
| `rule_keltner_trend` | Keltner break, EMA20>EMA50>EMA200 | price>VWAP |
| `rule_obv_momentum` | OBV direction, RSI 45–70, ADX≥20 | MACD hist + |

### 6.4 Direction groups

Rules that can fire either way use a `"direction"` group so the engine knows
the bias:

```python
Condition("kelt_up", "close above Keltner upper", ..., group="direction"),
Condition("kelt_dn", "close below Keltner lower", ..., group="direction"),
```

`bull_keys` / `bear_keys` map the winning member to a side. Without this a rule
would need *both* "above" and "below" to be true and could never fire.

### 6.5 Breaks compare against prior bars

Identical to §5.5: breakout conditions use `donch_hi_prev` / `kelt_hi_prev`,
never the level that includes the current bar.

### 6.6 Configuration

```yaml
rules:
  enabled: [rule_trend_stack, rule_golden_cross, ...]
  disabled: []
rule_params:
  rule_trend_stack:
    min_confidence: 0.55
    stop_atr: 1.5
    target_rr: 2.5
```

Omitting `enabled` enables every rule. `rule_params` overrides thresholds per
rule without editing code.

---

## 6A. User-defined strategies

Two extension points, neither of which requires touching the engine.

### 6A.1 Declarative (YAML)

`trader/config/custom_strategies.yaml` describes a setup as a **trigger**
(one comparison), a list of **filters** (more comparisons that must all
hold), and a **stop** and **target** rule:

```yaml
strategies:
  my_ema_cross:
    label: "My EMA Cross"
    kind: trend
    trigger:
      left:  {indicator: ema, period: 5}
      op:    crosses_above
      right: {indicator: ema, period: 34}
    filters:
      - {left: {indicator: adx}, op: ">=", right: 25}
    stop:    {type: atr, mult: 1.5}
    target:  {type: r_multiple, value: 2.0}
    confidence: 0.65
```

Indicators available: `close` `open` `high` `low` `volume` `price` `ema` `rsi`
`atr` `adx` `macd` `bb` `vwap` `donchian` `supertrend` `psar` `stoch` `cci`
`roc` `mfi` `keltner`.

Operators: `>` `>=` `<` `<=` `==` `crosses_above` `crosses_below`.

Stop types: `atr` · `percent` · `points` · `swing` · `indicator`.
Target types: `r_multiple` · `atr` · `percent` · `points`.

`trader/core/custom.py` turns a spec into a `DeclarativeStrategy`, which is an
ordinary `Strategy` — it is scored, ranked and traded exactly like a built-in
one.

### 6A.2 Python plugins

Any `.py` file in `trader/custom/` that defines a `Strategy` subclass is
imported at start-up and added to the pool. Files beginning with `_` are
ignored. A broken plugin is reported in the log and does not stop the
engine.

### 6A.3 Building one

```bash
python -m trader add-strategy     # interactive wizard
python -m trader strategies       # list everything that is loaded
```

The wizard asks what fires the setup, what else must hold, and where the stop
and target go, then writes the YAML (preserving the file's comment header).

Custom setups run by default. `custom_strategies.enabled` / `.disabled` in
`settings.yaml` control them independently of the built-in list.

---

## 7. Strategy selection (explainable)

### 7.1 The gates

A setup must clear **every** gate before it may be traded. The point is that
"a setup exists for this symbol" is not a reason to trade it.

| Gate | Default | |
|---|---|---|
| `min_trade_count` | 8 | enough history to judge it at all |
| `min_expectancy_r` | 0.0 | it must not have been losing |
| `min_profit_factor` | 1.1 | gross win must clearly beat gross loss |
| `min_win_rate` | 0.35 | a workable share of winners |
| `min_signal_confidence` | 0.55 | the live signal must be convincing too |
| `min_score` | 0.55 | the weighted score must clear the bar |

### 7.2 Scoring

| Component | Weight | |
|---|---|---|
| Expectancy (R per trade) | 0.40 | zero expectancy is worth only 0.35, not 0.5 |
| Profit factor | 0.20 | `min(pf / 2, 1)` |
| Win rate | 0.15 | |
| Regime fit | 0.15 | see below |
| Live confidence | 0.10 | |

Expectancy is mapped so negative expectancy scores below zero expectancy,
which scores below positive expectancy. A setup that has been going nowhere
must not look half-good just because it carries the heaviest weight.

**Regime fit** uses the strategy's `kind`:

| Regime | trend | meanrev | breakout |
|---|---|---|---|
| Trending | 1.0 | 0.3 | 1.0 |
| Ranging | 0.3 | 1.0 | 0.3 |

### 7.3 The market decides

Once a market has been scanned, its candidates are ranked against each other
and only the strongest few are taken:

| Gate | Default |
|---|---|
| `min_market_score` | 0.60 |
| `max_trades_per_market_per_cycle` | 2 |

A busy market produces two good trades, not forty mediocre ones. The
per-symbol decision is *which setup*; the per-market decision is *whether to
trade at all*.

### 7.4 Grading on history only

The bar the live signal was generated on is **excluded** from that setup's
backtest. Without this, a setup is graded on the very trade that is asking
for an entry, which inflates its expectancy exactly when it matters most.

### 7.5 Mean-reversion setups refuse trends

Fading an extreme inside a strong trend is how a mean-reversion system loses
money. Every mean-reversion setup (RSI, Stochastic, CCI, MFI, Williams %R,
MACD divergence, VWAP band) declines to fire above ADX 30.

### 7.6 The reason string

```
Keltner Break + EMA Stack | score 0.99 | trending
EMA20 > EMA50 > EMA200 + close above Keltner upper + price above VWAP
```

Scores are cached per symbol and refreshed every `score_refresh_cycles`
(default 5) cycles — a full mini-backtest of 45 setups is far too expensive to
run every cycle.

---

## 8. Risk management

Mandatory, and applied to every order.

| Control | Default |
|---|---|
| Position size | `(equity × risk_per_trade%) / (entry − stop)`, floored to whole units |
| Stop loss | Always set. ATR-based, or a structural level (EMA50, swing low) |
| Target | Minimum 1.5R; rules often use 2.0–3.0R |
| Trailing stop | Optional; breakeven move after +1R, then ATR trail |
| Max concurrent positions | 6 total, 3 per market |
| Max trades per symbol per day | 2 |
| Max daily loss | 1.5% ⇒ kill switch: stop entries, square off everything |
| Cooldown after a loss | 30 minutes on that symbol |
| No new entries | First 5 minutes, last 30 minutes of the session |
| New trades per market per cycle | 2 (only the best-ranked candidates) |
| Forced exit | All positions flat 12 minutes before the close |

Additional hard rules:

- **No duplicate position per symbol.** A symbol already open is not re-entered.
- **No overnight holds.** Intraday only, every market with a session clock.
- **Every order carries a stop.** A position with no stop is never opened.

### 8.1 A market we do not scan cannot protect its positions

A stop loss is only enforced while we are watching the market. The moment a
market stops being scannable, any position in it is unmanaged, so it is closed
immediately:

| Situation | Action |
|---|---|
| Session over (`CLOSED`) | Close every position in that market at the last price seen |
| Lunch break (`LUNCH`, Japan 11:30–12:30) | Same — we are not scanning, so we cannot protect it |
| 12 min before the close (`PRE-CLOSE`) | Forced flat, so the exit is not at the closing bell |
| Restart with a restored position in a closed market | Closed at the last known price on startup |

The exit reason is `SQUAREOFF`, the trade is recorded, and the position is
removed from the broker as well as the portfolio.

```yaml
risk:
  square_off_before_close_minutes: 12
  close_on_market_closed: true   # set false to keep positions through a break
```

Because the market is not scanned after it closes, the "last price" is the
final close of the session — which is the correct fill for a market order
against a closed market.

**Crypto is exempt everywhere:** `always_open` returns `status OPEN`
persistently, so `should_square_off` is never triggered, entries have no
session-edge exclusion window and the 12-minute pre-close never applies. The
kill switch and per-symbol cooldowns still work; only session edges do not.

---

## 9. Market sessions

| Market | Code | Exchange | Local hours | Lunch | Suffix | Currency | Symbols |
|---|---|---|---|---|---|---|---|
| India | IN | NSE | 09:15–15:30 IST | — | `.NS` | INR | 209 |
| Japan | JP | TSE | 09:00–15:00 JST | 11:30–12:30 | `.T` | JPY | 100 |
| Korea | KR | KRX | 09:00–15:30 KST | — | `.KS` | KRW | 99 |
| Hong Kong | HK | HKEX | 09:30–16:00 HKT | — | `.HK` | HKD | 100 |
| Australia | AU | ASX | 10:00–16:00 Sydney | — | `.AX` | AUD | 98 |
| Brazil | BR | B3 | 10:00–17:00 BRT | — | `.SA` | BRL | 41 |
| Europe | UK | LSE | 08:00–16:30 London | — | `.L` | GBP | 91 |
| USA | US | NYSE | 09:30–16:00 ET | — | — | USD | 100 |
| Crypto | CR | Binance | 00:00–23:59 UTC | — | — | USDT | 45 |

**Why the equities were chosen this way** — depth over headlines. Japan,
Korea, Hong Kong and Australia keep the Asian session alive; India anchors the
middle of the day; Brazil plugs the Atlantic gap between London's close and
New York's open. Every watchlist symbol was verified to have 5-minute history
on yfinance (Brazil ends at 41 and Europe at 91 because Yahoo does not carry
intraday data for the rest).

**Why Crypto.** It is the only instrument that covers the 01:30–04:30 IST dead
zone, and it doubles as a smoke test for the always-open market machinery
(`MarketClock.status` → persistent `OPEN`, no pre-close sweep). It is behind
its own switch (`crypto_enabled`, `--no-crypto`, dashboard `c`) and the
smallest capital of all markets, because data is as Yahoo-delayed as
everywhere else.

`MarketClock` returns `OPEN` / `PRE-CLOSE` / `LUNCH` / `CLOSED` plus the
countdown to the next open or close. All times are timezone-aware via
`zoneinfo`, so daylight saving is handled automatically (Hong Kong does not
observe it, Sydney and London do). Weekends and a configurable holiday list
are respected.

Only `OPEN` markets are scanned and enabled markets are filtered in
`Engine.__init__` (`crypto_enabled` × per-market `enabled`), so disabling a
market removes it from scanning *and* from every UI surface in one place.

P&L is tracked per market in its own currency and converted to INR with FX
rates: fetched every 5 minutes per market currency (`{CUR}INR=X`), with
static fallbacks in `portfolio.fx_rates` and the report path. USDT is wired to
the USD rate.

---

## 10. Live-trading safety

Three independent gates, all required:

1. `--mode live` on the command line
2. `live_trading_confirmed: true` in `settings.yaml`
3. An interactive typed confirmation phrase at startup

Any one missing ⇒ the system stays in paper mode. In LIVE mode the header
becomes a solid red bar and the order path switches to `AngelBroker`.

Additional safeguards:

- Angel auth failure ⇒ fall back to yfinance for data **and** stay in paper.
- The Angel order path is unreachable unless `mode == "live"` and the
  confirmation flag is set, so a code path change cannot leak a live order.
- A live capital cap can be configured.

---

## 11. Persistence

SQLite (`data/trader.db`, WAL mode) stores signals, orders, trades, equity
snapshots and open positions. On restart in the same session, open positions
are restored with their stop, target and trailing state — and closed at once
if their market is no longer trading.

On shutdown, a CSV of the session's trades plus a summary is written to
`data/reports/`.

`python -m trader report` prints per-market, per-strategy performance so the
user can see which setups actually work. It uses a static FX table
(`data/reports` also gets the CSV).

---

## 12. Dashboard

State the UI keeps: `market_filter`, `signal_filter` (None/long/short),
`text_filter`, `sort_mode` (score/chg/signal/symbol), `_page`, `_selected`
(row cursor), `_view` (None/help/detail), `_input_mode`/`_input_buf`
(command bar), `_pending_close`/`_pending_close_at` (close arm window),
`_stop`. The engine exposes `scan_now` and `close_one`, so the UI mutates
*nothing* of the trade state itself.

```
  3  HEADER     mode badge · equity + day sparkline · day P&L · win rate · IST clock
  ?  MARKETS    up to 9 cards: status, local time, countdown, session bar,
                source + age, day P&L, open positions (adaptive grid)
   ? SCANNER    open markets only, signals first, sorted by score
                (filters in the title, ◆ selection marker, WHY below)
                ACTIVE (with why-lines) | COMPLETED | LOG — bottom row
  1  FOOTER     hotkeys · command prompt · armed-close hint · scan stats
```

Layout budget: header 3 rows fixed; market cards fit their content; the
bottom row (active + completed + log) gets its **measured content height**
(`_bottom_h`, capped 8–16 rows), and the scanner takes **everything that is
left** — no fixed rations, no black voids at taller terminals. Below 160
columns Active stacks over Completed | Log instead. The footer is always
rendered (heights are clamped when the terminal is too short).

Design rules:

- **Panels size to their content.** Extra terminal height goes to the scanner,
  never to an empty void.
- **Tables shrink, then drop, by priority.** A column's width is reduced
  towards its minimum before any column is dropped, and `P&L` is never dropped.
  The full "why this trade" reason moves to lines under the table rather than
  being truncated.
- **Empty tables explain themselves** — "India closed, opens in 14h29m",
  "First scan running…" — instead of showing a bare header.
- **Colour is never the only signal.** P&L carries `+`/`−` and `▲`/`▼`;
  themes are `default`, `colorblind` (blue/orange) and `mono`.
- **Numbers are right-aligned**, INR uses lakh grouping (`₹38,07,404`), and
  empty statistics render `—` rather than `0.00%`.
- **One truth per action.** Quick-close (`1..9`) arms, a second press inside
  6 s confirms; the footer shows the armed state. Nothing else closes a
  position silently.

Hotkeys `q x p v c Tab Backspace ← → ↑ ↓ d f s g r 1..9 : / h ?` and the
command bar (`:filter/:sort/:close/:flat/:crypto/:kill/:scan/...`) are as
documented in the README; `q` always works, one press, no confirm.

---

## 12A. Live quotes

yfinance has no websocket feed, but the quote metadata behind
`Ticker.fast_info` is refreshed continuously by the exchange and is *fresher*
than the last completed intraday bar. That is enough to watch prices move and
to react to a stop or target being hit promptly.

`trader/core/prices.py` holds a `PriceTicker`: a thread pool fetches the last
price for a set of symbols (~30 symbols/second measured). The engine runs it
on its own thread:

```
every price_refresh_seconds:
    symbols = open positions first, then the symbols on screen
    quotes  = ticker.refresh(symbols)
    update the scanner rows with the new price and % change
    check every open position for a stop or target hit   <- exits happen here
every fast_scan_seconds:
    roll through every remaining symbol of every market  <- whole-market queue
```

A single lock guards the portfolio so the scan thread and the quote thread
never interleave while touching it. The scan thread keeps its own cadence
(`scan_interval_seconds` measured from the start of the cycle, not from the end
of it) so the cycle time does not drift with the scan duration.

Every quote also lands in `engine.price_history[symbol]` (a rolling deque,
180 long) under the same lock; the detail overlay uses it to show the
last-minute price change.

The feed is the exchange's quote metadata — fresher than the last intraday
bar, but still exchange-delayed. It is for watching movement and firing exits
promptly, not for tick-accurate pricing.

---

## 13. Robustness

| Failure | Handling |
|---|---|
| yfinance error | Retry with backoff, skip the symbol, mark stale |
| One symbol raises | Log and continue; never abort the cycle |
| Empty data | Row shown as "no data" / "error", no crash |
| Stale data | Marked `*`, age shown in the scanner and market cards |
| Angel login fails | yfinance fallback, warning surfaced, paper mode |
| Panel render raises | Caught; the scan thread continues |
| Keyboard interrupt | Graceful shutdown, report still written |
| A blocked entry | Counted and reported once per cycle, not once per symbol |
| The daily loss limit is hit | Kill switch flattens and halts *for the day*, then releases |
| Two threads writing | One lock around the portfolio; SQLite in WAL mode |
| A hotkey handler raises | Caught in the loop, logged (`hotkey error: …`), keys keep working |
| FX fetch fails | Static fallback rates are used for the cycle; retry in 5 min |
| Yahoo 429 on quotes | Manifests as stale `quotes age` in the footer; no crash (and one known pain point — §22) |

**yfinance intraday limits** are respected: 1m ≈ 7 days, 5m ≈ 60 days.

---

## 14. Performance budget

| Stage | Budget |
|---|---|
| `Ctx` build (all indicators, numpy) | ~6 ms per symbol |
| 45 setups against a shared `Ctx` | ~14 ms per symbol |
| mini-backtest (one `Ctx`, cursor replay) | ~20 ms per symbol |
| one full scan cycle, all 883 symbols | minutes — network fetches dominate |
| quote refresh (12 workers) | ~30 symbols/s |

Optimisations already in place:

- numpy indicator core instead of pandas (`np_indicators.py`)
- `supertrend` and `psar` use arrays, not per-bar `.iloc` writes
- `scipy.signal.lfilter` for the EMA recursion when scipy is present
- one `Ctx` per symbol, shared by every setup
- **cursor replay** in the mini-backtest: one context, N bar evaluations
- batch downloads (`yf.download`) to cut request count
- selector scores cached for several cycles
- per-market sessions are scanned in sequence; a closed market costs one
  clock lookup, not a scan

The first cycle after startup is the slowest — it downloads history for every
symbol. Later cycles reuse the cache.

---

## 15. Testing

| Suite | Covers |
|---|---|
| `test_market_clock.py` | session status, weekends, pre-close, countdowns, always-open |
| `test_indicators_strategies.py` | indicators and every setup on synthetic data |
| `test_strategies_rules.py` | 29 setups + 16 rules: no crashes, valid signals, rule grouping, confidence thresholds, shared-ctx equivalence, prior-bar breakouts |
| `test_selector*.py` | gates strictness, weights, regime fit, grading-window exclusion |
| `test_risk*.py`, `test_daily_limits.py` | sizing, caps, kill switch, cooldowns, session edges |
| `test_currency.py`, `test_liabilities.py` | per-market currency handling, FX math |
| `test_dashboard.py` | snapshot rendering at multiple sizes, column fit, empty states, overlays |
| `test_market_close.py` | close-on-closed-market sweep, restored-position handling |
| `test_readme.py` | README ↔ settings.yaml ↔ strategy counts agree |
| `test_live_quotes.py` | PriceTicker — requires network; 429-flaky |

Invariants asserted for every setup and rule:

- never raises on any input length (including empty and 5-bar frames)
- a signal always has `stop != entry` and `target != entry`
- a signal always has a valid side and a non-empty reason
- a rule with a failing required condition never fires
- a direction group fires when exactly one member passes
- a shared `Ctx` gives the same signal as a solo context

---

## 16. Deliberate non-choices

| Not doing | Why |
|---|---|
| Position sizing by volatility or Kelly | Fixed-fractional risk is predictable and auditable |
| Machine-learned strategy selection | The brief asks for *explainable* selection |
| Tick data / websocket feed | 5m bars and 60s cycles are the target |
| Partial fills, order books | Paper realism only needs fills, slippage and charges |
| Multi-timeframe backtesting | The mini-backtest ranks setups, it is not a research tool |
| Cross-platform hotkeys | msvcrt + window-focus detection are Windows-native; port later if needed |

---

## 22. Known weak points — ranked

A review that changes one of these changes the system's usefulness the most:

1. **Single-source data risk (yfinance).** Eight markets, the quote thread and
   the FX fetch all ride on one rate-limited provider. A jittered, tokened
   retry policy — or a second provider for quotes only — would remove the
   most likely production failure. Files: `trader/core/prices.py`,
   `trader/data/yf_provider.py`.
2. **Small grading window.** The selector gates judge a setup on ~5 days of
   5-minute bars (≈ 75 sessions × ~75 bars). Statistically thin; a setup that
   worked last week can dominate today's trades. Options: longer lookback for
   scoring only, exponential decay on old trades, or an explicit regime
   freshness check. Files: `trader/core/selector.py`, `engine.lookback_days`.
3. **No score-drift detection.** Scores refresh every 5 cycles, but nothing
   notices a setup's expectancy decay across days. A simple rolling
   expectancy chart per setup in `report` would surface it. Files:
   `selector.py`, `storage.py`.
4. **24/7 market edges.** Crypto has no session edges, so the daily-loss
   counter's midnight rollover is the only boundary; a bad run can straddle
   it and dodge the kill switch briefly. Options: optional rolling 24 h loss
   limit. File: `trader/core/risk.py`.
5. **Scan cost scales with symbol count.** 883 symbols × (fetch + 6 ms
   evaluate) makes a full cycle minutes on a slow link; the quote thread
   papers over it for the visible rows only. Options: per-market parallel
   fetch pools (careful: yfinance limits), or a symbol-priority tier on the
   scan. File: `trader/core/engine.py`, `data/yf_provider.py`.
6. **Windows-only interactivity.** msvcrt + focus detection. Cross-platform
   input (via a tiny crossterm-equivalent or `termios` niche) is possible but
   untested. File: `trader/ui/dashboard.py` (isolated — `_hotkey_loop`,
   `_console_has_focus`).
7. **Paper fills assume last-price liquidity.** Slippage is a flat bps model;
   no impact model for illiquid mid-caps. Acceptable for paper, worth stating
   in any performance claim.

## 23. Review pointers

If reviewing for improvements, these questions and files yield the highest
leverage, in the same order as §22:

- "How would you make quote fetching survive Yahoo rate limits?" →
  `prices.py`, `yf_provider.py`
- "Is a 5-day grading window defensible? What would you replace it with?" →
  `selector.py::_evaluate_gates`
- "Show me where a market's 'today' is defined and think hard about Crypto" →
  `engine._roll_over_day`, `market_clock.py`
- "Where is the one place all state changes flow through, and what breaks if
  another thread writes engine state?" → `engine.py` threading rules
- "Which UI state lives on the dashboard and which on the engine, and why?" →
  `dashboard.py` state block, `engine.scan_now` / `close_one`

---

## Disclaimer

Paper results do not guarantee live results. This software is not financial
advice. Trading involves risk. Use at your own risk.
