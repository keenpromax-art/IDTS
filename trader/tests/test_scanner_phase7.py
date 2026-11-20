"""Deterministic tests for Phase 7 telemetry and benchmark harness."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from zoneinfo import ZoneInfo

from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
from trader.scanner.benchmark import MockCandleProvider, run_benchmark
from trader.scanner.telemetry import CycleTelemetry, TelemetryCollector

PKG = Path(__file__).resolve().parents[1]
settings = yaml.safe_load(open(PKG / "config" / "settings.yaml"))
markets = yaml.safe_load(open(PKG / "config" / "markets.yaml"))["markets"]


def test_cycle_telemetry_computes_all_rates():
    t = CycleTelemetry(requested=100, scanned=80, missing_data=10, stale=5,
                       download_latency_s=2.0, processing_latency_s=3.0, total_s=5.0,
                       provider_success=8, provider_failure=2, fallback_used=2,
                       quotes_checked=100, quotes_ok=90, quotes_stale=5, quotes_delayed=4,
                       signals_by_strategy={"ema_cross": 3},
                       signals_by_market={"India": 3})
    s = t.snapshot()
    assert s["symbols_requested"] == 100 and s["symbols_scanned"] == 80
    # throughput counts every symbol processed (scanned + missing + prescreened)
    assert s["symbols_per_second"] == 18.0
    assert s["provider_failure_rate"] == pytest.approx(0.2)
    assert s["fallback_rate"] == pytest.approx(0.2)
    assert s["quote_coverage_pct"] == pytest.approx(90.0)
    assert s["data_download_latency_s"] == 2.0
    assert s["processing_latency_s"] == 3.0
    assert s["signals_by_strategy"]["ema_cross"] == 3


def test_telemetry_collector_keeps_bounded_history():
    tc = TelemetryCollector(history=2)
    for _ in range(4):
        tc.finish(tc.new_cycle())
    assert len(tc.history) == 2
    assert tc.latest()


def test_engine_records_telemetry_after_scan(tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["live_quotes"] = False
    eng = Engine(cfg, {"India": markets["India"]}, {"India": ["A", "B"]}, "paper", False)
    eng.fx_last_update = 1e12
    eng.providers["India"] = MockCandleProvider()
    open_status = {"status": "OPEN", "next_open": None,
                   "local_time": datetime.now(ZoneInfo(markets["India"]["timezone"])),
                   "open_dt": datetime.now(timezone.utc) - timedelta(hours=2),
                   "close_dt": datetime.now(timezone.utc) + timedelta(hours=2),
                   "next_close": None, "countdown_seconds": 7200, "in_lunch": False}
    with patch.object(MarketClock, "status", return_value=open_status), \
         patch.object(eng, "_trade_candidates"):
        eng.scan_once()
    tel = eng.last_telemetry
    for key in ("symbols_requested", "symbols_scanned", "symbols_missing_data",
                "stale_quotes", "data_download_latency_s", "processing_latency_s",
                "total_scan_duration_s", "symbols_per_second", "quote_coverage_pct",
                "provider_failure_rate", "signals_by_strategy", "signals_by_market"):
        assert key in tel, key
    assert tel["symbols_requested"] == 2
    assert tel["total_scan_duration_s"] > 0


def test_mock_provider_is_deterministic_and_reports_failures():
    p = MockCandleProvider(failure_rate=0.1)
    a = p.frame("SAME")
    b = p.frame("SAME")
    pd.testing.assert_frame_equal(a, b)
    assert p.get_candles("X") is not None
    failed = [p.get_candles(f"F{i}") is None for i in range(15)]
    assert any(failed) and p.failures > 0


def test_benchmark_runs_small_universe_end_to_end():
    result = run_benchmark(24, pass_every=6)
    assert result["symbols"] == 24
    assert result["prescreen_survivors"] > 0
    assert result["prescreen_symbols_per_second"] > 0
    assert result["end_to_end_s"] > 0
    assert result["end_to_end_symbols_per_second"] > 0
    tel = result["telemetry"]
    assert tel["symbols_scanned"] + tel["symbols_prescreened"] == 24
    assert tel["symbols_requested"] == 24
