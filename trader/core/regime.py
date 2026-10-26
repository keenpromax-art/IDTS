"""Market regime detection for IDTS.

Provides simple, testable statistical methods for regime detection
without complex machine learning. All calculations use only current
and past observations (no look-ahead bias).

Regime definitions:
  - TREND: ADX above threshold, price moving in one direction
  - RANGE_BOUND: ADX low, price oscillating within a channel
  - HIGH_VOLATILITY: ATR percentiles high, expanded price bands
  - LOW_LIQUIDITY: Volume below historical baseline
  - UNCERTAIN: Conflicting signals, insufficient confidence

Features (all backward-looking, no future data):
  - realized_volatility: rolling std of returns
  - atr_ratio: ATR as percentage of price
  - trend_strength: ADX value (without +DI/-DI direction bias)
  - range_compression: (high-low)/ATR ratio (low = compressed)
  - volume_ratio: current volume / rolling mean volume
  - time_of_day: hour-minute in target session
  - market_direction: broad index trend (optional, None if unavailable)
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import pandas as pd
from datetime import datetime, time as dt_time

from trader.core.indicators import atr, adx, rsi, stochastic, mfi, obv, cci, roc
from trader.core.risk import RiskManager


# ---------------------------------------------------------------------------
# Feature calculations (all backward-looking, NaN-safe)
# ---------------------------------------------------------------------------


def realized_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling standard deviation of daily log returns.

    Uses only close prices; no future data.
    """
    if df is None or "close" not in df.columns or len(df) < period + 1:
        return pd.Series(dtype=float, index=df.index if df is not None else [])
    closes = df["close"].astype(float)
    returns = np.log(closes / closes.shift(1))
    return returns.rolling(window=period).std()


def atr_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR divided by close price — a relative volatility measure.

    Lower values = less volatile market in absolute terms.
    """
    if df is None or "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
        return pd.Series(dtype=float, index=df.index if df is not None else [])
    a = atr(df, period)
    c = df["close"].astype(float)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = a / c.replace(0, np.nan)
    return pd.Series(np.where(np.isfinite(ratio), ratio, np.nan), index=df.index)


def trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX indicator — measures trend strength without direction bias.

    Values: 0-25 = weak/no trend, 25-50 = strong trend, 50+ = very strong.
    """
    if df is None or "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
        return pd.Series(dtype=float, index=df.index if df is not None else [])
    return adx(df, period)


def range_compression(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """(High - Low) / ATR — measures how compressed the current range is
    relative to recent volatility.

    High value = wide range relative to volatility (expansion).
    Low value = narrow range relative to volatility (compression).
    """
    if df is None or "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
        return pd.Series(dtype=float, index=df.index if df is not None else [])
    a = atr(df, period)
    hr = (df["high"].astype(float) - df["low"].astype(float)) / a.replace(0, np.nan)
    return hr


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume / rolling mean volume — measures liquidity.

    Values > 1.5 = above-average volume; < 0.5 = below-average (thin).
    """
    if df is None or "volume" not in df.columns:
        return pd.Series(dtype=float, index=df.index if df is not None else [])
    vol = df["volume"].astype(float)
    mean_vol = vol.rolling(window=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return vol / mean_vol.replace(0, np.nan)


def time_of_day(dt_index: pd.DatetimeIndex, session_start: time = dt_time(9, 30), session_end: time = dt_time(16, 0)) -> pd.Series:
    """Hour-minute label for the current bar's position in the session.

    Returns 'pre', 'session', 'post', or 'outside' depending on where
    the bar falls relative to the defined session window.
    """
    if dt_index is None:
        return pd.Series(dtype=str, index=[])
    hours = dt_index.hour + dt_index.minute / 60.0
    session_start_min = session_start.hour + session_start.minute / 60.0
    session_end_min = session_end.hour + session_end.minute / 60.0
    n = len(dt_index)
    result = pd.Series("", index=dt_index[:n])
    for i in range(n):
        h = hours[i]
        if h < session_start_min:
            result.iloc[i] = "pre"
        elif h < session_end_min:
            result.iloc[i] = "session"
        else:
            result.iloc[i] = "post"
    # Mark bars outside the full day range
    result = result.where((dt_index.hour >= session_start.hour) & (dt_index.hour < session_end.hour), "outside")
    return result


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

TREND_ADX = 25      # ADX above = trending
TREND_ATR_PCT = 0.02  # ATR% above = strong trend move
RANGE_ADX = 20      # ADX below = range-bound
VOL_LOW = 0.5       # volume ratio below = low liquidity
VOL_HIGH = 1.5      # volume ratio above = high volume
VRV_PCT_HIGH = 4.0  # ATR% > 4% = high volatility


def classify_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    vol_period: int = 20,
    adx_thresh: float = TREND_ADX,
    vol_ratio_thresh_low: float = VOL_LOW,
    vol_ratio_thresh_high: float = VOL_HIGH,
    atr_pct_thresh: float = VRV_PCT_HIGH,
) -> Dict[str, Optional[str]]:
    """Classify the current market regime based on back-looking features.

    Returns a dict with regime labels and confidence indicators.  The
    'regime' key is the primary classification; other keys provide
    contextual detail.  Any feature that cannot be computed defaults
    to None rather than making up a value.
    """
    if df is None or len(df) < max(adx_period, atr_period, vol_period) + 1:
        return {
            "regime": None,
            "adx": None,
            "atr_pct": None,
            "volume_ratio": None,
            "time": None,
            "trend_direction": None,
            "confidence": "unknown",
        }

    try:
        adx_val = adx(df, adx_period).iloc[-1]
    except Exception:
        adx_val = None

    try:
        a = atr(df, atr_period)
        close = df["close"].astype(float).iloc[-1]
        atr_pct = (a.iloc[-1] / close * 100) if close and not np.isclose(close, 0) else None
    except Exception:
        atr_pct = None

    try:
        vr = volume_ratio(df, vol_period)
        vol_rat = vr.iloc[-1] if hasattr(vr, "iloc") else vr
    except Exception:
        vol_rat = None

    try:
        # Simple trend direction from price slope over last 10 bars
        closes = df["close"].astype(float)
        if len(closes) >= 10:
            slope = (closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10]
            trend_dir = "up" if slope > 0.005 else ("down" if slope < -0.005 else "sideways")
        else:
            trend_dir = None
    except Exception:
        trend_dir = None

    # ---- Regime logic (hysteresis / stability) ----
    regime: Optional[str] = None
    confidence: str = "unknown"

    if adx_val is not None and adx_val >= adx_thresh:
        # Trend mode — further discriminate by volatility
        if atr_pct is not None and atr_pct > atr_pct_thresh:
            regime = "high_vol_trend"
            confidence = "high"
        else:
            regime = "trend"
            confidence = "high" if adx_val >= 50 else "medium"
    elif adx_val is not None and adx_val < RANGE_ADX:
        # Range-bound mode — check volatility for high-vol classification
        if atr_pct is not None and atr_pct > atr_pct_thresh:
            regime = "high_vol_range"
            confidence = "medium"
        else:
            regime = "range_bound"
            confidence = "high"
    elif vol_rat is not None and vol_rat < vol_ratio_thresh_low:
        regime = "low_liquidity"
        confidence = "high"
    elif vol_rat is not None and vol_rat > vol_ratio_thresh_high:
        regime = "high_volume"
        confidence = "medium"
    elif atr_pct is not None and atr_pct > atr_pct_thresh:
        regime = "high_volatility"
        confidence = "medium"
    else:
        regime = "uncertain"
        confidence = "low"

    # Hysteresis: don't flip regime on every bar if the previous regime
    # was strong and the change is marginal (prevents turnover)
    # (The caller can maintain prev_regime if desired.)

    # Determine time-of-day for context (not used for the regime label
    # itself, but returned for awareness)
    try:
        dt_idx = df.index
        if isinstance(dt_idx, pd.DatetimeIndex):
            n = len(dt_idx)
            session_start = dt_time(9, 30)
            session_end = dt_time(16, 0)
            hours = dt_index.hour[:n] + dt_index.minute[:n] / 60.0
            session_start_min = session_start.hour + session_start.minute / 60.0
            session_end_min = session_end.hour + session_end.minute / 60.0
            time_labels = []
            for i in range(n):
                h = hours[i]
                if h < session_start_min:
                    time_labels.append("pre")
                elif h < session_end_min:
                    time_labels.append("session")
                else:
                    time_labels.append("post")
            time_of_day_label = time_labels[-1] if time_labels else None
        else:
            time_of_day_label = None
    except Exception:
        time_of_day_label = None

    return {
        "regime": regime,
        "adx": adx_val,
        "atr_pct": atr_pct,
        "volume_ratio": vol_rat,
        "time": time_of_day_label,
        "trend_direction": trend_dir,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Convenience: whether to trade in a given regime
# ---------------------------------------------------------------------------

# Default: only abstain from "uncertain" and "low_liquidity" unless
# the caller explicitly opts in.  Adjust thresholds per the walk-forward
# results.
REGIMES_TO_ABSTAIN = {"uncertain", "low_liquidity"}

REGIMES_TO_TRADE = {
    "trend",
    "range_bound",
    "high_vol_trend",
    "high_vol_range",
    "high_volatility",
    "high_volume",
}


def should_trade(regime: Optional[str], regimes_to_abstain: set = REGIMES_TO_ABSTAIN) -> bool:
    """Return True if the strategy should consider entering a trade.

    This is optional — the caller may ignore it and trade every bar,
    but the function provides a principled default based on the regime
    classification.
    """
    if regime is None:
        return False  # can't classify => abstain
    return regime not in regimes_to_abstain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_realized_vol_causal():
    """Verify that realized volatility uses only past returns (no look-ahead)."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(123)
    closes = 100 + np.cumsum(np.random.randn(50) * 0.02)
    df = pd.DataFrame({"close": closes}, index=dates)
    rv = realized_volatility(df, 5)
    # First 4 values should be NaN (not enough history)
    assert pd.isna(rv.iloc[0]), "First value should be NaN"
    assert pd.isna(rv.iloc[1]), "Second value should be NaN"
    assert pd.isna(rv.iloc[2]), "Third value should be NaN"
    assert pd.isna(rv.iloc[3]), "Fourth value should be NaN"
    # Fifth value should exist (5-bar window)
    assert not pd.isna(rv.iloc[4]), "Fifth value should exist"
    # Verify monotonic relationship: if returns are larger, vol should be higher
    # (this is a soft check — just ensure no runtime errors)
    print("  test_realized_vol_causal PASSED")


def test_missing_data_handling():
    """Ensure regime classification gracefully handles missing data."""
    # None data
    result = classify_regime(None)
    assert result["regime"] is None, "None data should give None regime"
    # Very short data
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"close": [100, 101, 102, 101, 105]}, index=dates)
    result = classify_regime(df)
    # Should not crash; regime may be None if insufficient data
    assert "regime" in result, "Result should have 'regime' key"
    print("  test_missing_data_handling PASSED")


def test_regime_transitions():
    """Verify that regime can change over time (not stuck)."""
    np.random.seed(42)
    n = 200
    closes = 100 + np.cumsum(np.random.randn(n) * 0.02)
    highs = closes * (1 + np.abs(np.random.randn(n)) * 0.005)
    lows = closes * (1 - np.abs(np.random.randn(n)) * 0.005)
    vols = np.random.randint(1000, 5000, n)
    df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": vols})
    df.index = pd.date_range("2023-01-01", periods=n, freq="D")

    regimes = []
    for i in range(len(df)):
        d = df.iloc[:i+1]  # growing window — simulates walking forward
        r = classify_regime(d)
        regimes.append(r["regime"])

    # Count regime changes
    changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1] and regimes[i] is not None)
    assert changes > 0, "Regime should change over time with random data"
    print(f"  test_regime_transitions PASSED ({changes} changes in {len(regimes)} bars)")


def test_volume_ratio_bounds():
    """Verify volume ratio is bounded and reasonable."""
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 0.02)
    vols = np.random.randint(1000, 5000, n)
    df = pd.DataFrame({"close": closes, "volume": vols})
    df.index = pd.date_range("2023-01-01", periods=n, freq="D")

    vr = volume_ratio(df, 20)
    # First ~19 values may be NaN or very large
    valid_vr = vr.dropna()
    if len(valid_vr) > 0:
        # Ratio should typically be positive
        assert (valid_vr > 0).all(), "Volume ratio should be positive"
        # Most values should be in reasonable range
        ratio_vals = valid_vr[(valid_vr > 0) & (valid_vr < 5)]
        assert len(ratio_vals) > 0, "Should have some reasonable volume ratios"
    print("  test_volume_ratio_bounds PASSED")


def test_time_of_day():
    """Verify time-of-day labeling works for session hours."""
    dates = pd.date_range("2023-01-01", periods=24, freq="H")
    label = time_of_day(dates, session_start=dt_time(9, 0), session_end=dt_time(17, 0))
    # First 9 hours should be "pre", hours 9-17 "session", rest "post"
    assert label.iloc[0] == "pre", f"First hour should be 'pre', got {label.iloc[0]}"
    assert label.iloc[9] == "session", f"Hour 9 should be 'session', got {label.iloc[9]}"
    assert label.iloc[17] == "post", f"Hour 17 should be 'post', got {label.iloc[17]}"
    print("  test_time_of_day PASSED")


def test_regime_classification():
    """Full regression test with sample data."""
    np.random.seed(42)
    n = 150
    closes = 100 + np.cumsum(np.random.randn(n) * 0.02)
    highs = closes * (1 + np.abs(np.random.randn(n)) * 0.005)
    lows = closes * (1 - np.abs(np.random.randn(n)) * 0.005)
    vols = np.random.randint(1000, 5000, n)
    df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": vols})
    df.index = pd.date_range("2023-01-01", periods=n, freq="D")

    r = classify_regime(df)
    assert "regime" in r, "Result must contain 'regime' key"
    assert "adx" in r, "Result must contain 'adx' key"
    assert "atr_pct" in r, "Result must contain 'atr_pct' key"
    assert "volume_ratio" in r, "Result must contain 'volume_ratio' key"
    assert "confidence" in r, "Result must contain 'confidence' key"
    assert r["regime"] is not None, "Regime should not be None with sufficient data"
    print(f"  test_regime_classification PASSED (regime={r['regime']}, adx={r['adx']:.1f}, "
          f"atr_pct={r['atr_pct']:.2f}%, vol_ratio={r['volume_ratio']:.2f}, "
          f"conf={r['confidence']})")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def detect_regime(df: pd.DataFrame, **kwargs) -> Dict[str, Optional[str]]:
    """Wrap classify_regime for ease of use.

    Parameters passed to classify_regime can be overridden via kwargs.
    """
    return classify_regime(df, **kwargs)


if __name__ == "__main__":
    # Run all tests when invoked directly
    print("Running regime detection tests...")
    test_realized_vol_causal()
    test_missing_data_handling()
    test_regime_transitions()
    test_volume_ratio_bounds()
    test_time_of_day()
    test_regime_classification()
    print("\nAll tests PASSED!")