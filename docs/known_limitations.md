# Known Limitations

- yfinance is the only data source for 8 of 9 markets; HTTP 429 rate limits
  can make quotes stale (test_live_quotes.py flakes).
- Backtest window is ~5 days of 5m bars; statistically thin.
- No walk-forward CLI yet (library helper exists).
- Windows-only hotkeys (msvcrt).
- Crypto has no session edges; daily loss resets at midnight IST.
- Paper fills use a flat bps slippage model; no market impact.
- Angel One integration is incomplete (token/master handling, stop orders).
- Single-writer thread model caps responsiveness on slow links.
