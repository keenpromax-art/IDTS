"""Scanner telemetry: coverage, latency, provider health and signal counts.

All values are measured, never estimated. Rates are computed from counters at
snapshot time so a partial cycle still produces a coherent report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CycleTelemetry:
    requested: int = 0
    scanned: int = 0
    missing_data: int = 0
    prescreened: int = 0
    stale: int = 0
    errors: int = 0
    download_latency_s: float = 0.0
    processing_latency_s: float = 0.0
    total_s: float = 0.0
    provider_success: int = 0
    provider_failure: int = 0
    fallback_used: int = 0
    quotes_ok: int = 0
    quotes_checked: int = 0
    quotes_stale: int = 0
    quotes_delayed: int = 0
    signals_by_strategy: Dict[str, int] = field(default_factory=dict)
    signals_by_market: Dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def snapshot(self) -> Dict[str, Any]:
        effective = max(1e-9, self.total_s)
        return {
            "symbols_requested": self.requested,
            "symbols_scanned": self.scanned,
            "symbols_missing_data": self.missing_data,
            "symbols_prescreened": self.prescreened,
            "symbols_stale": self.stale,
            "symbol_errors": self.errors,
            "data_download_latency_s": round(self.download_latency_s, 6),
            "processing_latency_s": round(self.processing_latency_s, 6),
            "total_scan_duration_s": round(self.total_s, 6),
            "symbols_per_second": round(
                (self.scanned + self.prescreened + self.missing_data + self.errors) / effective, 3),
            "provider_success": self.provider_success,
            "provider_failure": self.provider_failure,
            "provider_failure_rate": round(
                self.provider_failure / max(1, self.provider_success + self.provider_failure), 6),
            "fallback_used": self.fallback_used,
            "fallback_rate": round(
                self.fallback_used / max(1, self.provider_success + self.fallback_used), 6),
            "quote_coverage_pct": round(
                100.0 * self.quotes_ok / max(1, self.quotes_checked), 3),
            "stale_quotes": self.quotes_stale,
            "delayed_quotes": self.quotes_delayed,
            "signals_by_strategy": dict(sorted(self.signals_by_strategy.items())),
            "signals_by_market": dict(sorted(self.signals_by_market.items())),
            "timestamp": self.started_at,
        }


class TelemetryCollector:
    """Keep the latest cycle and a bounded rolling history."""

    def __init__(self, history: int = 50):
        self.history_size = max(1, int(history))
        self.current: Optional[CycleTelemetry] = None
        self.history: List[Dict[str, Any]] = []

    def new_cycle(self) -> CycleTelemetry:
        self.current = CycleTelemetry()
        return self.current

    def finish(self, tel: CycleTelemetry) -> Dict[str, Any]:
        snap = tel.snapshot()
        self.history.append(snap)
        self.history = self.history[-self.history_size:]
        self.current = None
        return snap

    def latest(self) -> Dict[str, Any]:
        if self.current is not None:
            return self.current.snapshot()
        return self.history[-1] if self.history else {}
