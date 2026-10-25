"""Data-quality checks for market data frames.

Every provider returns a DataFrame indexed by UTC timestamp with columns
open/high/low/close/volume. This module validates that contract and reports
problems instead of silently letting bad data reach the indicators.

Nothing here fabricates or repairs data: it only classifies what arrived.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class DataQualityReport:
    ok: bool = True
    rows: int = 0
    issues: List[str] = field(default_factory=list)
    stale_seconds: Optional[float] = None
    duplicate_timestamps: int = 0
    nan_cells: int = 0
    ohlc_violations: int = 0
    non_positive_prices: int = 0
    unsorted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "issues": list(self.issues),
            "stale_seconds": self.stale_seconds,
            "duplicate_timestamps": self.duplicate_timestamps,
            "nan_cells": self.nan_cells,
            "ohlc_violations": self.ohlc_violations,
            "non_positive_prices": self.non_positive_prices,
            "unsorted": self.unsorted,
        }


def validate_candle_df(df: Optional[pd.DataFrame],
                      now: Optional[pd.Timestamp] = None,
                      stale_after_seconds: Optional[float] = None) -> DataQualityReport:
    """Validate a candle frame. Never raises; returns a report."""
    rep = DataQualityReport()
    if df is None:
        rep.ok = False
        rep.issues.append("no data")
        return rep
    if not isinstance(df, pd.DataFrame):
        rep.ok = False
        rep.issues.append("not a dataframe")
        return rep
    if df.empty:
        rep.ok = False
        rep.issues.append("empty frame")
        return rep
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.ok = False
        rep.issues.append(f"missing columns: {missing}")
        return rep
    rep.rows = int(len(df))

    # nan cells
    nan = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    rep.nan_cells = nan
    if nan:
        rep.issues.append(f"{nan} NaN cell(s)")
        rep.ok = False

    # timestamp index
    try:
        idx = pd.to_datetime(df.index, utc=True)
    except Exception:
        try:
            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
        except Exception:
            rep.ok = False
            rep.issues.append("unparseable timestamp index")
            return rep

    # ordering
    if not idx.is_monotonic_increasing:
        rep.unsorted = True
        rep.issues.append("timestamps not sorted ascending")
        rep.ok = False
    # duplicates
    dup = int(idx.duplicated().sum())
    rep.duplicate_timestamps = dup
    if dup:
        rep.issues.append(f"{dup} duplicate timestamp(s)")
        rep.ok = False

    # OHLC invariants
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    viol = 0
    viol += int(np.sum((h < o) | (h < c) | (h < l)))
    viol += int(np.sum((l > o) | (l > c) | (l > h)))
    viol += int(np.sum(h < l))
    rep.ohlc_violations = viol
    if viol:
        rep.issues.append(f"{viol} OHLC invariant violation(s)")
        rep.ok = False

    # non-positive prices
    np_ = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
    rep.non_positive_prices = np_
    if np_:
        rep.issues.append(f"{np_} non-positive price(s)")
        rep.ok = False

    # negative volume
    if np.any(v < 0):
        rep.issues.append("negative volume")
        rep.ok = False

    # staleness
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        try:
            now = pd.Timestamp(now)
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            else:
                now = now.tz_convert("UTC")
        except Exception:
            now = pd.Timestamp.now(tz="UTC")
    try:
        last = idx[-1]
        age = (now - last).total_seconds()
        rep.stale_seconds = float(age)
        if stale_after_seconds is not None and age > stale_after_seconds:
            rep.issues.append(f"stale: last bar {int(age)}s old")
            rep.ok = False
    except Exception:
        pass

    return rep


def is_usable(df: Optional[pd.DataFrame], **kw) -> bool:
    return validate_candle_df(df, **kw).ok
