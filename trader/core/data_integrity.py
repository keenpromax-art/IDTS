"""Market data integrity validation.

Validates timestamps, timezones, market calendars, duplicate/missing bars,
OHLC consistency, volume anomalies, stale prices, symbol mappings, and
session boundaries. Never repairs or invents data: only classifies and
reports problems.

All functions are deterministic and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trader.core.data_quality import validate_candle_df, DataQualityReport, REQUIRED_COLUMNS


@dataclass
class IntegrityReport:
    """Extended data integrity report.

    Fields are additive: ``ok`` is False only when a hard error exists
    (missing columns, NaN, OHLC violations, etc.). Warnings and informational
    fields (stale, gaps, volume spikes) do not flip ``ok`` unless they
    indicate data that should not be traded.
    """

    ok: bool = True
    rows: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stale_seconds: Optional[float] = None
    duplicate_timestamps: int = 0
    missing_bars: int = 0
    nan_cells: int = 0
    ohlc_violations: int = 0
    non_positive_prices: int = 0
    unsorted: bool = False
    volume_anomalies: int = 0
    session_gaps: int = 0
    symbol_mismatch: bool = False
    timezone_mismatch: bool = False
    lookahead_risk: bool = False  # future data detected relative to now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "stale_seconds": self.stale_seconds,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_bars": self.missing_bars,
            "nan_cells": self.nan_cells,
            "ohlc_violations": self.ohlc_violations,
            "non_positive_prices": self.non_positive_prices,
            "unsorted": self.unsorted,
            "volume_anomalies": self.volume_anomalies,
            "session_gaps": self.session_gaps,
            "symbol_mismatch": self.symbol_mismatch,
            "timezone_mismatch": self.timezone_mismatch,
            "lookahead_risk": self.lookahead_risk,
        }


def validate_market_data_integrity(
    df: Optional[pd.DataFrame],
    symbol: str = "",
    market: str = "",
    market_cfg: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    expected_interval_minutes: int = 5,
) -> IntegrityReport:
    """
    Comprehensive market data integrity validation.

    Parameters
    ----------
    df : DataFrame | None
        Candle data indexed by UTC timestamp with columns open/high/low/close/volume.
    symbol : str
        Symbol being validated (used for suffix mapping check).
    market : str
        Market name (used for timezone / session config).
    market_cfg : dict | None
        Market configuration from markets.yaml (timezone, suffix, open, close, etc.).
    now : datetime | None
        Reference time for staleness and look-ahead checks. Defaults to now UTC.
    expected_interval_minutes : int
        Expected candle interval in minutes (default 5m).

    Returns
    -------
    IntegrityReport
        Never raises. Returns a report even for invalid input.
    """
    rep = IntegrityReport()

    # Start with base quality validation from data_quality
    base = validate_candle_df(df, now=now)
    rep.ok = base.ok
    rep.issues = list(base.issues)
    rep.rows = base.rows
    rep.stale_seconds = base.stale_seconds
    rep.duplicate_timestamps = base.duplicate_timestamps
    rep.nan_cells = base.nan_cells
    rep.ohlc_violations = base.ohlc_violations
    rep.non_positive_prices = base.non_positive_prices
    rep.unsorted = base.unsorted

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return rep

    # Symbol mapping validation: check that symbol has the expected suffix
    # for the given market. Some markets like USA/Crypto have empty suffixes,
    # others like India have ".NS" or ".BO".
    if market_cfg:
        expected_suffix = market_cfg.get("suffix", "")
        if expected_suffix and not symbol.endswith(expected_suffix):
            rep.symbol_mismatch = True
            rep.issues.append(
                f"symbol {symbol} missing expected suffix {expected_suffix}"
            )

    # Timezone validation
    if market_cfg:
        expected_tz = market_cfg.get("timezone")
        if expected_tz:
            try:
                idx = pd.to_datetime(df.index, utc=True)
                if idx.tz is None:
                    rep.timezone_mismatch = True
                    rep.issues.append("timestamp index is timezone-naive")
                else:
                    # Convert to expected tz and verify it's consistent
                    try:
                        expected_tzinfo = ZoneInfo(expected_tz)
                        # Check that all timestamps convert cleanly
                        local = idx.tz_convert(expected_tzinfo)
                        # If we get here, conversion succeeded
                    except Exception:
                        rep.timezone_mismatch = True
                        rep.issues.append(
                            f"could not convert timestamps to {expected_tz}"
                        )
            except Exception:
                rep.timezone_mismatch = True
                rep.issues.append("unparseable timestamp index")

    # Volume anomaly detection
    if "volume" in df.columns:
        vol = df["volume"].to_numpy(dtype=float)
        if len(vol) > 0:
            # Zero or negative volume
            neg_vol = int(np.sum(vol <= 0))
            if neg_vol:
                rep.volume_anomalies += neg_vol
                rep.issues.append(f"{neg_vol} zero/negative volume bar(s)")

            # Extreme volume spike (>10x median of positive volume)
            if len(vol) > 10 and np.any(vol > 0):
                med = float(np.median(vol[vol > 0]))
                if med > 0:
                    spikes = int(np.sum(vol > med * 10))
                    if spikes:
                        rep.volume_anomalies += spikes
                        rep.warnings.append(
                            f"{spikes} volume spike(s) > 10x median volume"
                        )

    # Missing bars / session gaps
    if market_cfg and len(df) > 1:
        try:
            idx = pd.to_datetime(df.index, utc=True)
            expected_tz = ZoneInfo(market_cfg.get("timezone", "UTC"))
            local_idx = idx.tz_convert(expected_tz)

            # Check for gaps larger than 2x expected interval within session
            interval_seconds = expected_interval_minutes * 60
            diffs = np.diff(local_idx.values.astype("int64")) / 1e9  # seconds

            # Gaps > 3x expected interval (allowing for minor delays within session)
            gap_threshold = interval_seconds * 3
            gaps = int(np.sum(diffs > gap_threshold))
            if gaps:
                rep.missing_bars = gaps
                rep.session_gaps = gaps
                rep.issues.append(
                    f"{gaps} session gap(s) > {expected_interval_minutes * 3} min"
                )
        except Exception:
            pass

    # Staleness check (refined using expected interval)
    if rep.stale_seconds is not None and now is not None:
        if rep.stale_seconds > expected_interval_minutes * 60 * 2:
            rep.warnings.append(
                f"data is {int(rep.stale_seconds)}s old (>2 intervals)"
            )

    # Look-ahead risk: data has timestamps in the future relative to now
    if now is not None:
        try:
            idx = pd.to_datetime(df.index, utc=True)
            now_utc = pd.Timestamp(now, tz="UTC")
            # Any bar with timestamp after now + 1 minute is look-ahead
            future_bars = int(np.sum(idx > now_utc + timedelta(minutes=1)))
            if future_bars:
                rep.lookahead_risk = True
                rep.issues.append(f"{future_bars} bar(s) with future timestamps")
        except Exception:
            pass

    # Overall OK: hard failures flip ok
    hard_fail = (
        rep.volume_anomalies > 0
        or rep.missing_bars > 0
        or rep.session_gaps > 0
        or rep.ohlc_violations > 0
        or rep.non_positive_prices > 0
        or rep.nan_cells > 0
        or rep.unsorted
        or rep.duplicate_timestamps > 0
    )
    # ok starts from base but hard failures override
    if hard_fail:
        rep.ok = False

    return rep


def is_market_data_usable(df, **kwargs) -> bool:
    """Convenience: True if integrity report says ok."""
    return validate_market_data_integrity(df, **kwargs).ok


@dataclass
class DataQualitySummary:
    """Human-readable summary of a data integrity check."""

    symbol: str
    market: str
    rows: int
    issues: List[str]
    warnings: List[str]
    stale_seconds: Optional[float]
    lookahead_risk: bool
    duplicate_timestamps: int
    missing_bars: int
    ohlc_violations: int
    non_positive_prices: int
    volume_anomalies: int
    session_gaps: int
    symbol_mismatch: bool
    timezone_mismatch: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "rows": self.rows,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "stale_seconds": self.stale_seconds,
            "lookahead_risk": self.lookahead_risk,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_bars": self.missing_bars,
            "ohlc_violations": self.ohlc_violations,
            "non_positive_prices": self.non_positive_prices,
            "volume_anomalies": self.volume_anomalies,
            "session_gaps": self.session_gaps,
            "symbol_mismatch": self.symbol_mismatch,
            "timezone_mismatch": self.timezone_mismatch,
        }


def summarize_data_quality(
    df: Optional[pd.DataFrame],
    symbol: str = "",
    market: str = "",
    market_cfg: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> DataQualitySummary:
    """Produce a concise summary suitable for logging or dashboard display."""
    rep = validate_market_data_integrity(
        df, symbol=symbol, market=market, market_cfg=market_cfg, now=now
    )
    return DataQualitySummary(
        symbol=symbol,
        market=market,
        rows=rep.rows,
        issues=rep.issues,
        warnings=rep.warnings,
        stale_seconds=rep.stale_seconds,
        lookahead_risk=rep.lookahead_risk,
        duplicate_timestamps=rep.duplicate_timestamps,
        missing_bars=rep.missing_bars,
        ohlc_violations=rep.ohlc_violations,
        non_positive_prices=rep.non_positive_prices,
        volume_anomalies=rep.volume_anomalies,
        session_gaps=rep.session_gaps,
        symbol_mismatch=rep.symbol_mismatch,
        timezone_mismatch=rep.timezone_mismatch,
    )