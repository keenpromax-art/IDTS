"""Timestamp-aware candle completion and price-source separation.

A candle indexed at ``t`` with interval ``X`` covers ``[t, t+X)``. It is only
*completed* once the wall clock has passed ``t + X`` (plus a small grace for
provider latency). Positional ``iloc[:-1]`` drops a finished bar one cycle too
late and keeps an unfinished bar one cycle too early; this module replaces it
with an explicit timestamp test.

Nothing here repairs or invents market data. Rows with unparseable timestamps
or invalid OHLCV values are rejected, never interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from trader.core.data_quality import REQUIRED_COLUMNS, validate_candle_df, DataQualityReport

INTERVAL_SECONDS = {
    "1m": 60,
    "2m": 120,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

DEFAULT_COMPLETION_GRACE_SECONDS = 5.0


def interval_seconds(interval: str) -> int:
    """Return the wall-clock duration of one bar for *interval*.

    Unknown intervals fall back to 5 minutes (the system default) rather than
    guessing a shorter, less conservative value.
    """
    key = str(interval or "").strip().lower()
    return int(INTERVAL_SECONDS.get(key, 300))


def normalize_candles(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a canonical UTC-indexed OHLCV frame, or ``None`` if unusable.

    Canonical form: UTC ``DatetimeIndex`` named ``ts``, sorted ascending, no
    duplicate timestamps (the last observation of a duplicate wins), float
    OHLCV columns in a stable order, no NaN/inf rows.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None
    try:
        out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    except Exception:
        return None
    idx = out.index
    try:
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            out.index = idx.tz_convert("UTC") if str(idx.tz) != "UTC" else idx
        else:
            out.index = pd.to_datetime(idx, utc=True)
    except Exception:
        return None
    out = out[~out.index.isna()]
    if out.empty:
        return None
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    columns = []
    for col in REQUIRED_COLUMNS:
        try:
            arr = np.asarray(out[col].to_numpy(), dtype=float)
        except (TypeError, ValueError):
            try:
                arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            except Exception:
                return None
        columns.append(arr)
    mat = np.column_stack(columns) if columns else None
    if mat is None or mat.size == 0:
        return None
    finite = np.isfinite(mat).all(axis=1)
    if not finite.all():
        if not finite.any():
            return None
        out = out.iloc[finite]
        mat = mat[finite]
    for col, arr in zip(REQUIRED_COLUMNS, mat.T):
        out[col] = arr
    out.index.name = "ts"
    return out


def validate_candles(df: Optional[pd.DataFrame], *, symbol: str = "",
                     interval: str = "5m", stale_after_seconds: Optional[float] = None,
                     now: Optional[pd.Timestamp] = None,
                     require_completed_last: bool = False) -> DataQualityReport:
    """Validate a frame against the common provider contract.

    Wraps the established :func:`validate_candle_df` checks and adds bar-spacing
    validation so a provider returning irregular timestamps is rejected.
    """
    # Validate the *raw* frame so duplicates, bad ordering and NaNs are
    # reported instead of being silently cleaned away by normalization.
    rep = validate_candle_df(df, now=now, stale_after_seconds=stale_after_seconds)
    norm = normalize_candles(df)
    if norm is None:
        if rep.ok:
            rep.ok = False
            rep.issues.append("failed common candle normalization")
        return rep
    if len(norm) >= 2:
        try:
            deltas = pd.Series(norm.index).diff().dt.total_seconds().to_numpy()[1:]
            expected = interval_seconds(interval)
            # Allow gaps (halt / no trades) but never sub-interval spacing,
            # which indicates overlapping or mislabelled bars.
            short = int(np.sum(deltas < expected * 0.5))
            if short:
                rep.ok = False
                rep.issues.append(f"{short} sub-interval bar spacing error(s)")
        except Exception:
            pass
    if require_completed_last and not rep.ok:
        return rep
    if require_completed_last and norm.index[-1] > pd.Timestamp.now(tz="UTC") + timedelta(seconds=1):
        rep.ok = False
        rep.issues.append("last candle timestamp is in the future")
    return rep


def is_usable_candles(df: Optional[pd.DataFrame], **kwargs) -> bool:
    return validate_candles(df, **kwargs).ok


def bar_close_time(ts, interval: str):
    """Wall-clock UTC close time of the bar beginning at *ts*."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t + timedelta(seconds=interval_seconds(interval))


def is_completed(ts, interval: str, *, now: Optional[datetime] = None,
                 grace_seconds: float = DEFAULT_COMPLETION_GRACE_SECONDS) -> bool:
    """True when the bar at *ts* has certainly finished forming."""
    close = bar_close_time(ts, interval)
    if now is None:
        ref = pd.Timestamp.now(tz="UTC")
    else:
        ref = pd.Timestamp(now)
        if ref.tzinfo is None:
            ref = ref.tz_localize("UTC")
        else:
            ref = ref.tz_convert("UTC")
    return ref >= close + timedelta(seconds=max(0.0, float(grace_seconds)))


def completed_candles(df: Optional[pd.DataFrame], interval: str = "5m", *,
                      now: Optional[datetime] = None,
                      grace_seconds: float = DEFAULT_COMPLETION_GRACE_SECONDS) -> pd.DataFrame:
    """Drop only the last bar when its timestamp proves it is still forming.

    Unlike ``iloc[:-1]``, a fully completed last bar is retained, so signals
    are generated on the newest closed candle rather than one bar late.
    """
    norm = normalize_candles(df)
    if norm is None or norm.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    if len(norm) == 1:
        if is_completed(norm.index[-1], interval, now=now, grace_seconds=grace_seconds):
            return norm
        return norm.iloc[0:0]
    if is_completed(norm.index[-1], interval, now=now, grace_seconds=grace_seconds):
        return norm
    return norm.iloc[:-1].copy()


@dataclass(frozen=True)
class ExecutableQuote:
    """The price actually usable for an order, separated from signal prices."""

    price: float
    source: str
    ts: datetime
    age_seconds: float
    fresh: bool
    signal_price: float
    deviation_pct: float

    @property
    def executable(self) -> bool:
        return self.fresh and self.price > 0


def resolve_execution_price(signal_price: float, *,
                            live_price: Optional[float] = None,
                            live_ts: Optional[datetime] = None,
                            max_quote_age_seconds: float = 30.0,
                            max_deviation_pct: float = 1.0,
                            now: Optional[datetime] = None) -> Optional[ExecutableQuote]:
    """Choose the executable price and reject a market that moved too far.

    A live quote is preferred when it is fresh. A stale quote is never used.
    If the chosen price deviates from the completed-candle signal price by more
    than ``max_deviation_pct`` the setup is *not* executable — entering into a
    gapped market would invalidate the stop/target plan.
    """
    try:
        sig = float(signal_price)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(sig) or sig <= 0:
        return None
    ref = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    else:
        ref = ref.tz_convert("UTC")

    price, source, fresh, ts = sig, "candle", False, ref
    try:
        if live_price is not None:
            lp = float(live_price)
            if np.isfinite(lp) and lp > 0:
                qts = pd.Timestamp(live_ts) if live_ts is not None else ref
                if qts.tzinfo is None:
                    qts = qts.tz_localize("UTC")
                else:
                    qts = qts.tz_convert("UTC")
                age = max(0.0, (ref - qts).total_seconds())
                if age <= float(max_quote_age_seconds):
                    price, source, fresh, ts = lp, "live", True, qts
    except (TypeError, ValueError):
        pass

    age = max(0.0, (ref - ts).total_seconds())
    deviation = abs(price / sig - 1.0) * 100.0
    if deviation > float(max_deviation_pct) + 1e-9:
        return None
    return ExecutableQuote(
        price=float(price), source=source, ts=ts.to_pydatetime(),
        age_seconds=age, fresh=fresh, signal_price=sig, deviation_pct=deviation,
    )
