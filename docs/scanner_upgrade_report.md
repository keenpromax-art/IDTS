# IDTS Scanner Upgrade — Implementation & Verification Report

Date: 2026-11-20. Working tree: `C:\IDTS`. Audit: `docs/scanner_audit.md`.

## Test results (actual, full offline suite)

| Stage | Result | Duration |
|-------|--------|----------|
| Baseline (before changes) | 213 passed, 3 failed | 498.80s |
| After Phase 1 | 233 passed, 3 failed | 415.54s |
| After Phase 2 | 243 passed, 3 failed | 427.63s |
| After Phase 3 | 253 passed, 3 failed | 428.79s |
| After Phase 4 | 265 passed, 3 failed | 431.85s |
| After Phase 5 | 277 passed, 3 failed | 433.44s |
| After Phase 6 | 287 passed, 3 failed | 423.75s |
| After Phase 7 (final) | **292 passed, 3 failed** | 448.49s |

The same 3 failures exist in the baseline and in every phase:
`test_live_quotes.py::{test_ticker_returns_prices_for_known_symbols,
test_ticker_handles_a_bad_symbol, test_ticker_tracks_quote_age}`. They require
live Yahoo metadata and fail with no prices / HTTP 429 in this environment.
No offline test regressed. 79 new deterministic tests were added across
`test_scanner_phase1.py` … `test_scanner_phase7.py`.

**No improvement claim is made from the test-duration column** — it varies with
network conditions and test count; it is not a benchmark.

## Benchmarks (mocked providers, measured not extrapolated)

Command: `python -m trader.scanner.benchmark --symbols 883 2000 5000`
Config: 1% deterministic pre-screen pass rate, `max_backtest_setups: 1`,
60 bars/symbol, no delay, no injected failures, no network, no live quotes.

| Symbols | Pre-screen s (sym/s) | End-to-end s (sym/s) | Download s | Processing s | Full scans | Prescreened |
|--------:|---------------------:|---------------------:|-----------:|-------------:|-----------:|------------:|
| 883  | 28.45 (31.0)  | 34.65 (25.5)  | 3.94  | 30.71  | 5  | 878  |
| 2000 | 48.01 (41.7)  | 78.01 (25.6)  | 8.13  | 69.88  | 19 | 1981 |
| 5000 | 107.59 (46.5) | 193.80 (25.8) | 21.13 | 172.66 | 42 | 4958 |

Raw JSON: `bench_883.json`, `bench_2000.json`, `bench_5000.json`.
No comparable benchmark existed before this work, so **no speedup percentage is
claimed**. The dominant cost is the graded backtest (up to 20 sessions × 45
strategies per surviving symbol); pre-screening is what makes 5,000 symbols
tractable. Each number is a single run on this machine, not a distribution.

## What was implemented

- **Phase 1** — `trader/scanner/candles.py`, `resilience.py`;
  `data/{base,factory,angel_provider,yf_provider}.py`; `core/{engine,selector,config,prices}.py`.
  Angel is now actually used for India batch candles with per-symbol fallback;
  one validation/normalization boundary for every provider; timestamp-aware bar
  completion; stale candles and gapped prices block entries; signal prices,
  executable quotes and display prices are separated; dedup key is
  `(symbol, strategy, direction, candle_ts)`; selector re-grades newly firing
  strategies even on cached cycles; negative caching, retry/backoff, circuit
  breakers and thread-safe rate limiting.
- **Phase 2** — `trader/scanner/prescreen.py` with all 11 requested filters,
  config-gated and off by default; inserted before the 45-strategy loop.
- **Phase 3** — `trader/scanner/timeframes.py`: 1m/5m/15m/30m/1h/1d, OHLC
  aggregation for higher TFs, correlation-aware family voting (one vote per
  trend/momentum/volatility/volume family), explicit conflicts.
- **Phase 4** — `trader/scanner/providers.py`: timestamped `Quote`, source
  health, candle/batch-candle/quote/batch-quote/stream APIs, Angel websocket
  detection with polling degradation, `IncrementalCandleStore`, bounded
  gateway with shared rate limit.
- **Phase 5** — `trader/scanner/ranking.py`: net expectancy, Sharpe, Sortino,
  drawdown, Wilson win-rate lower bound, sample size, walk-forward folds,
  configurable enhanced score; next-bar execution with spread/slippage/
  commission and gap-through-stop invalidation in `Selector.backtest`.
- **Phase 6** — `trader/scanner/views.py` plus a Scanner tab in `streamlit_app.py`
  with the nine requested views and the full result schema.
- **Phase 7** — `trader/scanner/telemetry.py` (coverage, latency, failure/fallback
  rates, signal counts) wired into `Engine.scan_once`; `trader/scanner/benchmark.py`.

## Unresolved limitations

1. Three network-dependent quote tests still fail offline; unchanged from baseline.
2. Angel websocket support is detected dynamically from the installed SmartAPI
   client and degrades to polling Angel's 1m candle API. It has **not** been
   validated against a live Angel account (no credentials in this environment).
3. Yahoo remains a research/fallback source; no claim of guaranteed real-time data.
4. Relative-strength screening requires the caller to populate
   `engine.benchmark_candles[market]`; the engine does not fetch a benchmark index.
5. Enhanced ranking, streaming and gateways are opt-in (`enabled: false`) to
   preserve paper-first defaults and existing scoring behaviour.
6. Next-bar backtest execution is now the default; historical backtest numbers
   will differ. `selector.execution_model.mode: same_bar` restores legacy behaviour.
7. The pre-screen "spread" constraint is a high-low range proxy. True bid/ask
   spread data is not available in this codebase and is not fabricated.
8. Benchmark frames carry a fixed 2026-01-05 timestamp, so all surviving symbols
   are correctly flagged stale; no orders are opened in the benchmark. Execution
   cost handling is covered by unit tests, not by the benchmark.
9. No lint/typecheck command is defined in this repository (no pyproject/setup.cfg
   with such scripts); verification is the pytest suite only.
