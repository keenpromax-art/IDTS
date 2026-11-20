"""Reproducible scanner benchmarks with mocked providers (no network).

Run:
    python -m trader.scanner.benchmark
    python -m trader.scanner.benchmark --symbols 883 2000 5000

The benchmark measures two separate things and never extrapolates:
  1. pre-screen-only throughput over the entire universe;
  2. a real end-to-end ``Engine.scan_once`` with pre-screening enabled, so
     the full strategy/backtest path runs only on surviving symbols.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from zoneinfo import ZoneInfo

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.data.base import DataProvider
from trader.scanner.prescreen import PreScreener
from trader.scanner.candles import completed_candles

PKG = Path(__file__).resolve().parents[1]
MARKETS_CFG = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


class MockCandleProvider(DataProvider):
    """Deterministic OHLCV generator with configurable latency and failures."""

    name = "mock"

    def __init__(self, bars: int = 60, delay: float = 0.0, failure_rate: float = 0.0,
                 pass_every: int = 20):
        self.bars = int(bars)
        self.delay = float(delay)
        self.failure_rate = float(failure_rate)
        self.pass_every = max(1, int(pass_every))
        self.calls = 0
        self.failures = 0

    def frame(self, symbol: str) -> pd.DataFrame:
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        n = self.bars
        idx = pd.date_range(pd.Timestamp("2026-01-05", tz="UTC"), periods=n, freq="5min")
        # Every pass_every-th symbol gets a strong move so pre-screen results
        # are deterministic and roughly 1/pass_every of the universe survives.
        strong = (seed % self.pass_every) == 0
        drift = 1.5 if strong else 0.01
        close = 100.0 + drift * np.arange(n) + rng.normal(0, 0.15, n)
        spread = 0.4 + np.abs(rng.normal(0, 0.1, n))
        vol = 1000.0 * (5.0 if strong else 1.0) + rng.normal(0, 50, n)
        return pd.DataFrame({
            "open": close - 0.05, "high": close + spread,
            "low": close - spread, "close": close,
            "volume": np.maximum(1.0, vol),
        }, index=idx)

    def get_candles(self, symbol, interval="5m", lookback_days=5):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.failure_rate and (self.calls % max(1, int(1 / self.failure_rate))) == 0:
            self.failures += 1
            return None
        return self.frame(symbol)

    def batch_download(self, symbols, interval="5m", lookback_days=5):
        return {s: self.get_candles(s, interval, lookback_days) for s in symbols}


def _build_engine(n_symbols: int, tmp: str, pass_every: int, provider: MockCandleProvider):
    settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
    markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]
    cfg = dict(settings)
    cfg["db_path"] = str(Path(tmp) / "bench.db")
    cfg["live_quotes"] = False
    cfg["data"] = dict(cfg.get("data") or {})
    cfg["data"]["prescreen"] = {
        "enabled": True, "min_abs_pct_move": 10.0, "lookback_bars": 20,
    }
    # Keep the measured full-scan cost bounded: one graded setup per symbol
    # instead of four. This is a benchmark configuration, not a product change.
    cfg["selector"] = dict(cfg.get("selector") or {})
    cfg["selector"]["max_backtest_setups"] = 1
    symbols = [f"BENCH{i:06d}" for i in range(n_symbols)]
    eng = Engine(cfg, {"India": markets["India"]}, {"India": symbols}, "paper", False)
    eng.fx_last_update = time.time()
    eng.providers["India"] = provider
    eng._trade_candidates = lambda market, candidates: None
    return eng, symbols


def run_benchmark(n_symbols: int, *, pass_every: int = 20,
                  delay: float = 0.0, failure_rate: float = 0.0) -> Dict[str, Any]:
    provider = MockCandleProvider(pass_every=pass_every, delay=delay,
                                  failure_rate=failure_rate)
    screener = PreScreener({"enabled": True, "min_abs_pct_move": 10.0,
                            "lookback_bars": 20})

    with tempfile.TemporaryDirectory() as tmp:
        eng, symbols = _build_engine(n_symbols, tmp, pass_every, provider)
        frames = {s: completed_candles(provider.frame(s), "5m") for s in symbols}

        t0 = time.perf_counter()
        passed = sum(1 for s in symbols if screener.screen(frames[s]).passed)
        prescreen_s = time.perf_counter() - t0

        def status(*args, **kwargs):
            # patched onto the class, so the first positional arg may be self
            market = args[1] if len(args) > 1 else args[0]
            tz = ZoneInfo(MARKETS_CFG[market]["timezone"])
            return {"status": "OPEN", "next_open": None,
                    "local_time": datetime.now(tz),
                    "open_dt": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2),
                    "close_dt": pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2),
                    "next_close": None, "countdown_seconds": 7200, "in_lunch": False}

        original = MarketClock.status
        MarketClock.status = status
        try:
            t1 = time.perf_counter()
            eng.scan_once()
            total_s = time.perf_counter() - t1
        finally:
            MarketClock.status = original

        tel = dict(eng.last_telemetry or {})
        del eng
        import gc
        gc.collect()
        time.sleep(0.1)
   
    return {
        "symbols": n_symbols,
        "prescreen_survivors": passed,
        "prescreen_only_s": round(prescreen_s, 4),
        "prescreen_symbols_per_second": round(n_symbols / max(1e-9, prescreen_s), 1),
        "end_to_end_s": round(total_s, 4),
        "end_to_end_symbols_per_second": round(n_symbols / max(1e-9, total_s), 1),
        "provider_calls": provider.calls,
        "provider_failures": provider.failures,
        "telemetry": tel,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="IDTS scanner benchmark (mocked providers)")
    ap.add_argument("--symbols", type=int, nargs="+", default=[883, 2000, 5000])
    ap.add_argument("--pass-every", type=int, default=100,
                    help="one in N symbols survives the deterministic pre-screen")
    ap.add_argument("--out", type=str, default=None, help="write JSON results here")
    args = ap.parse_args(argv)
    out = []
    for n in args.symbols:
        result = run_benchmark(n, pass_every=args.pass_every)
        out.append(result)
        print(json.dumps(result), flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
