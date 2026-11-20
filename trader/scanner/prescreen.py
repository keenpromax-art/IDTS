"""Configurable, low-cost pre-screening of the ticker universe.

This layer runs *before* the 45 strategies and the graded backtest. Every
feature is computed from a single completed-candle frame with numpy/pandas,
so a rejected symbol costs one pass over ~60 bars instead of 45 indicator
contexts plus a backtest.

Design rules:
  * Disabled by default: with no config every symbol passes, preserving
    legacy behaviour exactly.
  * No fabrication: missing benchmark data leaves relative-strength
    unavailable, so that filter cannot silently pass or fail.
  * "Spread" is a high-low range proxy, never advertised as a real bid/ask
    spread. No microstructure data is invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from trader.core.np_indicators import atr_np
from trader.scanner.candles import normalize_candles


@dataclass
class ScreenConfig:
    enabled: bool = False
    lookback_bars: int = 20
    orb_bars: int = 3
    combine: str = "all"                     # "all" | "any"
    min_abs_pct_move: Optional[float] = None
    max_abs_pct_move: Optional[float] = None
    min_relative_volume: Optional[float] = None
    min_volume_acceleration: Optional[float] = None
    min_traded_value: Optional[float] = None
    min_atr_expansion: Optional[float] = None
    max_vwap_deviation_atr: Optional[float] = None
    orb_breakout: bool = False
    min_abs_gap_pct: Optional[float] = None
    min_relative_strength_pct: Optional[float] = None
    max_range_pct: Optional[float] = None
    min_trend_persistence: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ScreenConfig":
        raw = dict(raw or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ScreenResult:
    passed: bool
    features: Dict[str, float] = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["passed"] = bool(self.passed)
        return d


def _safe_ratio(a: float, b: float) -> float:
    try:
        b = float(b)
        if not np.isfinite(b) or abs(b) < 1e-12:
            return float("nan")
        out = float(a) / b
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


class PreScreener:
    """Evaluate the configured cheap filters for one symbol."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = ScreenConfig.from_dict(cfg)

    # -- features -------------------------------------------------------
    def features(self, df: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        f: Dict[str, float] = {}
        df = normalize_candles(df)
        if df is None or len(df) < 10:
            return f
        n = min(self.cfg.lookback_bars, len(df) - 1)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        opn = df["open"].to_numpy(dtype=float)

        f["pct_move"] = (_safe_ratio(close[-1], close[-1 - n]) - 1.0) * 100.0
        f["abs_pct_move"] = abs(f["pct_move"])
        prior_vol = volume[-1 - n:-1]
        f["relative_volume"] = _safe_ratio(volume[-1], np.mean(prior_vol)) if prior_vol.size else float("nan")
        f["volume_acceleration"] = (_safe_ratio(volume[-1] - volume[-2], np.mean(prior_vol))
                                    if prior_vol.size else float("nan"))
        # Price * volume is a traded-value proxy, not exchange turnover.
        f["traded_value"] = float(close[-1] * volume[-1])
        f["range_pct"] = _safe_ratio(high[-1] - low[-1], close[-1]) * 100.0
        f["abs_gap_pct"] = abs(_safe_ratio(opn[-1] - close[-2], close[-2])) * 100.0

        atr = atr_np(high, low, close, 14)
        valid_atr = atr[np.isfinite(atr) & (atr > 0)]
        f["atr"] = float(atr[-1]) if np.isfinite(atr[-1]) else float("nan")
        f["atr_expansion"] = (_safe_ratio(atr[-1], np.median(valid_atr[-n:]))
                              if valid_atr.size else float("nan"))
        pv = volume[-n:]
        vwap = _safe_ratio(float(np.sum(close[-n:] * pv)), float(np.sum(pv)))
        f["vwap_deviation_atr"] = (_safe_ratio(close[-1] - vwap, atr[-1])
                                   if np.isfinite(vwap) else float("nan"))
        f["trend_persistence"] = float(np.mean(np.diff(close[-n - 1:]) > 0))
        f["orb_breakout"] = 0.0
        try:
            dates = pd.Series(df.index.date)
            last_date = dates.iloc[-1]
            session_pos = np.flatnonzero((dates == last_date).to_numpy())
            opening = session_pos[:max(1, int(self.cfg.orb_bars))]
            orb_high = float(np.max(high[opening]))
            orb_low = float(np.min(low[opening]))
            f["orb_breakout"] = 1.0 if (close[-1] > orb_high or close[-1] < orb_low) else 0.0
        except Exception:
            pass

        if benchmark is not None:
            bdf = normalize_candles(benchmark)
            if bdf is not None and len(bdf) >= n + 1:
                bc = bdf["close"].to_numpy(dtype=float)
                bmove = (_safe_ratio(bc[-1], bc[-1 - n]) - 1.0) * 100.0
                if np.isfinite(bmove):
                    f["benchmark_pct_move"] = bmove
                    f["relative_strength_pct"] = f["pct_move"] - bmove
        return f

    # -- filters ---------------------------------------------------------
    def screen(self, df: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None,
               symbol: str = "") -> ScreenResult:
        f = self.features(df, benchmark)
        cfg = self.cfg
        if not cfg.enabled or not f:
            return ScreenResult(True, f, ["pre-screen disabled" if not cfg.enabled else "no features"])
        passed, reasons, unavailable = [], [], []

        def check(name, key, op, threshold):
            if threshold is None:
                return
            value = f.get(key, float("nan"))
            if not np.isfinite(value):
                unavailable.append(key)
                return
            ok = value >= threshold if op == ">=" else value <= threshold
            (passed if ok else reasons).append(f"{key}={value:.4g} {op} {threshold:g}")

        check("min_abs_pct_move", "abs_pct_move", ">=", cfg.min_abs_pct_move)
        check("max_abs_pct_move", "abs_pct_move", "<=", cfg.max_abs_pct_move)
        check("min_relative_volume", "relative_volume", ">=", cfg.min_relative_volume)
        check("min_volume_acceleration", "volume_acceleration", ">=", cfg.min_volume_acceleration)
        check("min_traded_value", "traded_value", ">=", cfg.min_traded_value)
        check("min_atr_expansion", "atr_expansion", ">=", cfg.min_atr_expansion)
        check("max_vwap_deviation_atr", "vwap_deviation_atr", "<=", cfg.max_vwap_deviation_atr)
        check("min_abs_gap_pct", "abs_gap_pct", ">=", cfg.min_abs_gap_pct)
        check("min_relative_strength_pct", "relative_strength_pct", ">=", cfg.min_relative_strength_pct)
        check("max_range_pct", "range_pct", "<=", cfg.max_range_pct)
        check("min_trend_persistence", "trend_persistence", ">=", cfg.min_trend_persistence)
        if cfg.orb_breakout:
            if f.get("orb_breakout", 0.0) >= 1:
                passed.append("orb_breakout")
            else:
                reasons.append("no orb_breakout")

        ok = bool(reasons) is False if cfg.combine == "all" else bool(passed)
        return ScreenResult(ok, f, reasons, unavailable)

    @property
    def active(self) -> bool:
        return self.cfg.enabled
