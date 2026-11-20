# IDTS Ticker-Scanner Architecture Audit

Baseline measured on this checkout (2026-11-20): `python -m pytest trader/tests -q`
→ **213 passed, 3 failed, 8m18s**. The 3 failures are network-bound
`test_live_quotes.py` cases (Yahoo 429 / no prices), matching the known flake in
`README.md`. All offline tests pass. This audit is the basis for the phased
upgrade; every defect below was verified against source, not inferred.

## Lifecycle traced

`streamlit_app.py:62-77` / `dashboard.py:3245-3266` (`scan_loop`)
→ `Engine.scan_once` (`core/engine.py:698`) → `_scan_market` (`:802`)
→ `prov.batch_download` (`:818`) → `drop_unfinished` (`:27`) → `Ctx` (`:873`)
→ `Strategy.generate_from` ×45 (`:876`) → `Selector.evaluate` (`:900`)
→ `_trade_candidates` (`:297`) → `_open_candidate` (`:321`)
→ `RiskManager.can_open` → `PaperBroker.place_order` (`:372`)
→ `_check_exits` (`:770`). A separate thread refreshes quotes via
`PriceTicker` (`core/prices.py:57`, engine `:1030-1152`).

## Phase 1 — confirmed defects (exact files / lines)

| # | Defect | Evidence | Fix target |
|---|--------|----------|------------|
| 1.1 | Angel One never used for India candles | `data/factory.py:46-49` wires Angel as primary, but `data/angel_provider.py` has **no `batch_download`**, so `hasattr` at `factory.py:31` is False and line 41 calls Yahoo for every symbol. The per-symbol loop at `:34-40` is dead code. | `FallbackProvider.batch_download` + `AngelProvider.batch_download`, per-symbol fallback. |
| 1.2 | Candle validation only on Yahoo | `data/yf_provider.py:129,178` calls `is_usable`; `angel_provider.py:167-174`, `factory.py:21-41`, `data/base.py:15-25` return raw frames. `data_integrity.py` is never called in production. Zero-volume semantics disagree (`data_quality.py:128` `<0` vs `data_integrity.py:161` `<=0`). | Common validation in `DataProvider` boundary. |
| 1.3 | Stale candles cannot block a trade | `engine.py:854-862` computes `stale` but only stores it in the row (`:977`). `RiskManager.can_open(stale_data=True)` exists (`core/risk.py:74,79`) but engine calls it with 6 positional args (`:332-335`), so it always defaults False. | Pass `stale_data` + skip incomplete bars. |
| 1.4 | Three unreconciled price notions | Signal entry/stop/target from `dfc` (penultimate bar, `engine.py:848,873`); fill reference from **dropped** bar `df["close"].iloc[-1]` (`:849,951,372`); live `fast_info` quotes only touch display/exits (`:1107,1165`). No deviation cap. | `ExecutableQuote` resolution + deviation guard. |
| 1.5 | Unconditional last-bar removal | `engine.py:27-33` (`iloc[:-1]`), `selector.py:160` — purely positional, no timestamp/close-time check. | Timestamp-aware completion. |
| 1.6 | No signal deduplication | No seen-set anywhere; `models.py:57` `Signal.ts` is wall-clock, not candle time. Only symbol-keyed risk gates (`risk.py:104-113`) limit repeats. | `(symbol, strategy, direction, candle_ts)` dedup. |
| 1.7 | Selector cache ignores new signals | `selector.py:39,308-314` caches scores for `score_refresh_cycles`. On a refresh cycle, strategies without a live signal are stored as `eligible=False, sig=None` (`:339-343`). A signal appearing on a later cached cycle is rejected at `:358` and never re-graded. `max_backtest_setups:4` also writes off other firing strategies. | Grade newly appearing signals before cache use; keep historical grading independent of current firing. |
| 1.8 | Weak resilience layer | Backoff+retry exists only for quotes (`yf_provider.py:17-45`, used by `prices.py:55`); candle downloads have no backoff. `TTL_BY_INTERVAL`/`_ttl` (`yf_provider.py:51,83`) dead. No negative cache, no source-level circuit breaker. `PriceTicker.price()` (`:83-85`) returns stale values unchecked. Angel `_rate_limit` (`:65-69`) is not thread-safe. | Shared resilience primitives. |
| 1.9 | Config keys silently dropped by validation | `settings.yaml:25-26,168,179-180` not declared in `config.py:17-57` → removed by `load_config()` (`__main__.py:16-31`), disabling rolling-loss and scoring-lookback settings. | Declare fields (no gate weakened). |

## Phase 2-7 gaps

- **Pre-screening:** none. All 45 strategies run on every symbol (`engine.py:876`); cheapest insertion point is after `engine.py:862`, before regime/strategies/backtest.
- **Multi-timeframe:** none. Only one interval (`candle_interval`); `INTERVAL_MAP` exists in `angel_provider.py:13` but 1m/15m/30m/1h are unused.
- **Provider abstraction:** `data/base.py` exposes candles only; no batch quotes, streaming, health, quote timestamps, or shared rate limits. `quote_provider.py` is an unused stub. No websocket code anywhere (`prices.py:3` explicitly says none).
- **Ranking:** `selector.py:139-258` has expectancy/PF/win-rate/max-DD, but no Sharpe/Sortino, regime persistence, walk-forward, liquidity/freshness inputs; entries use same-bar close (`close[i]`) with slippage only (`:193-221`) — not next-bar execution. No explicit spread/slippage model despite `core/execution.py` existing.
- **Dashboard:** generic market tables only (`ui/dashboard.py`); no dedicated scanner views/FOMT-style result schema.
- **Telemetry:** only `last_cycle_stats` (`engine.py:796`) and `PriceTicker.errors`; no latency, provider fallback, coverage or per-strategy counts. No benchmark harness.

## Non-negotiable preservation

Paper-first broker selection (`engine.py:73-86`), the three live gates, all
`risk.can_open` gates, and strategy compatibility remain untouched. New
behaviour is additive/config-gated and defaults preserve current semantics
where a threshold is not supplied.
