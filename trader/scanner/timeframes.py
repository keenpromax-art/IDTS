"""Multi-timeframe analysis with correlation-aware confirmation counting.

Supports 1m, 5m, 15m, 30m, 1h and 1d. Higher timeframes can be derived from a
finer completed frame by OHLC aggregation (no data is invented); finer frames
must be supplied by a provider.

Correlated indicators are grouped into families. Each family contributes at
most one vote, so EMA + ADX/DI + Supertrend cannot masquerade as three
independent confirmations, nor can RSI + MACD + ROC.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trader.core.np_indicators import adx_np, atr_np, macd_np, roc_np, rsi_np, _ema_np_fast
from trader.scanner.candles import interval_seconds, normalize_candles

TF_ORDER = ("1m", "5m", "15m", "30m", "1h", "1d")
_RESAMPLE_RULE = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "1d": "1D",
}


def resample_candles(df: pd.DataFrame, target: str) -> Optional[pd.DataFrame]:
    """Aggregate a finer OHLCV frame to a coarser timeframe.

    Returns ``None`` when the target is not coarser than the source (a 1m bar
    cannot be reconstructed from 5m data without fabricating prices).
    """
    norm = normalize_candles(df)
    if norm is None or len(norm) < 2:
        return None
    if target not in TF_ORDER:
        return None
    deltas = pd.Series(norm.index).diff().dt.total_seconds().dropna()
    if deltas.empty:
        return None
    source_sec = float(deltas.median())
    if interval_seconds(target) <= source_sec:
        return None
    try:
        agg = norm.resample(_RESAMPLE_RULE[target], label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])
        return agg if not agg.empty else None
    except Exception:
        return None


def available_frames(frames: Dict[str, pd.DataFrame], primary: str,
                     timeframes: List[str], derive: bool = True) -> Dict[str, pd.DataFrame]:
    """Build the timeframe map, deriving coarser frames when allowed."""
    out: Dict[str, pd.DataFrame] = {}
    base = normalize_candles(frames.get(primary))
    if base is None:
        return out
    for tf in timeframes:
        if tf in frames and frames[tf] is not None and not getattr(frames[tf], "empty", True):
            norm = normalize_candles(frames[tf])
            if norm is not None:
                out[tf] = norm
        elif derive and tf != primary:
            agg = resample_candles(base, tf)
            if agg is not None:
                out[tf] = agg
    return out


@dataclass
class TimeframeSnapshot:
    timeframe: str
    bars: int
    trend_dir: int = 0
    trend_strength: float = 0.0
    momentum_dir: int = 0
    vol_expansion: bool = False
    volume_participation: bool = False
    family_votes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class MTFResult:
    direction: int = 0
    alignment: float = 0.0
    momentum_confirmed: bool = False
    volatility_expansion: bool = False
    families_used: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    per_timeframe: Dict[str, TimeframeSnapshot] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["per_timeframe"] = {k: v.to_dict() for k, v in self.per_timeframe.items()}
        return d


def _sign(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > 1e-12:
        return 1
    if x < -1e-12:
        return -1
    return 0


class MultiTimeframeAnalyzer:
    """Summarise trend/momentum/volatility alignment across timeframes."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.primary = str(cfg.get("primary", "5m"))
        self.timeframes = list(cfg.get("timeframes") or ["5m", "15m", "1h", "1d"])
        self.derive = bool(cfg.get("derive", True))
        self.block_on_conflict = bool(cfg.get("block_on_conflict", False))
        self.min_alignment = float(cfg.get("min_alignment", 0.6))
        self.adx_threshold = float(cfg.get("adx_threshold", 20.0))
        self.atr_expansion_threshold = float(cfg.get("atr_expansion_threshold", 1.1))
        self.weights = dict(cfg.get("weights") or {
            "trend": 0.4, "momentum": 0.25, "volatility": 0.2, "volume": 0.15,
        })

    def snapshot(self, tf: str, df: pd.DataFrame) -> TimeframeSnapshot:
        norm = normalize_candles(df)
        snap = TimeframeSnapshot(timeframe=tf, bars=0 if norm is None else len(norm))
        if norm is None or len(norm) < 30:
            return snap
        close = norm["close"].to_numpy(dtype=float)
        high = norm["high"].to_numpy(dtype=float)
        low = norm["low"].to_numpy(dtype=float)
        vol = norm["volume"].to_numpy(dtype=float)

        ema_fast = _ema_np_fast(close, 9)
        ema_slow = _ema_np_fast(close, 21)
        slope = _sign(ema_fast[-1] - ema_slow[-1])
        adx, di_plus, di_minus = adx_np(high, low, close, 14)
        di_dir = _sign(float(di_plus[-1]) - float(di_minus[-1]))
        strength = float(adx[-1]) if np.isfinite(adx[-1]) else 0.0
        # One trend vote, not one per correlated indicator.
        votes = [v for v in (slope, di_dir) if v]
        snap.trend_dir = int(np.sign(np.sum(votes))) if votes else 0
        snap.trend_strength = strength if strength >= self.adx_threshold else 0.0
        if not snap.trend_strength:
            snap.trend_dir = 0

        rsi = rsi_np(close, 14)
        roc = roc_np(close, 12)
        macd, _, hist = macd_np(close)
        m_votes = [_sign(float(rsi[-1]) - 50.0), _sign(float(roc[-1])), _sign(float(hist[-1]))]
        m_votes = [v for v in m_votes if v]
        snap.momentum_dir = int(np.sign(np.sum(m_votes))) if m_votes else 0

        atr = atr_np(high, low, close, 14)
        valid = atr[np.isfinite(atr) & (atr > 0)]
        if valid.size >= 10 and np.isfinite(atr[-1]):
            snap.vol_expansion = bool(atr[-1] / np.median(valid[-50:]) >= self.atr_expansion_threshold)
        if len(vol) >= 20:
            snap.volume_participation = bool(vol[-1] > np.mean(vol[-20:-1]))

        snap.family_votes = {
            "trend": snap.trend_dir,
            "momentum": snap.momentum_dir,
            "volatility": int(snap.vol_expansion),
            "volume": int(snap.volume_participation),
        }
        return snap

    def analyze(self, symbol: str, frames: Dict[str, pd.DataFrame],
                signal_side: Optional[Any] = None) -> MTFResult:
        result = MTFResult()
        if not self.enabled:
            return result
        usable = available_frames(frames, self.primary, self.timeframes, self.derive)
        for tf in TF_ORDER:
            if tf in usable:
                result.per_timeframe[tf] = self.snapshot(tf, usable[tf])
        if not result.per_timeframe:
            return result

        ranked = [result.per_timeframe[tf] for tf in TF_ORDER if tf in result.per_timeframe]
        trend_dirs = [s.trend_dir for s in ranked if s.trend_dir]
        result.direction = int(np.sign(np.sum(trend_dirs))) if trend_dirs else 0
        result.alignment = (sum(1 for d in trend_dirs if d == result.direction) / len(trend_dirs)
                            if trend_dirs else 0.0)

        mom_dirs = [s.momentum_dir for s in ranked if s.momentum_dir]
        result.momentum_confirmed = bool(
            mom_dirs and result.direction and
            sum(1 for d in mom_dirs if d == result.direction) / len(mom_dirs) >= self.min_alignment)
        result.volatility_expansion = any(s.vol_expansion for s in ranked)

        # Conflicts are reported explicitly; higher timeframes outweigh lower.
        higher = [s for s in ranked if interval_seconds(s.timeframe) >= 3600]
        lower = [s for s in ranked if interval_seconds(s.timeframe) <= 900]
        for h in higher:
            for l in lower:
                if h.trend_dir and l.trend_dir and h.trend_dir != l.trend_dir:
                    result.conflicts.append(f"{h.timeframe} {h.trend_dir:+d} vs {l.timeframe} {l.trend_dir:+d}")
        if signal_side is not None:
            side = 1 if str(getattr(signal_side, "value", signal_side)).upper() == "LONG" else -1
            for h in higher:
                if h.trend_dir and h.trend_dir != side:
                    result.conflicts.append(f"signal {side:+d} against {h.timeframe} trend")
        result.conflicts = list(dict.fromkeys(result.conflicts))

        # One vote per family, aggregated across timeframes (not per indicator).
        used, total, weight_sum = [], 0.0, 0.0
        for family in ("trend", "momentum", "volatility", "volume"):
            votes = [s.family_votes.get(family, 0) for s in ranked]
            nonzero = [v for v in votes if v]
            if nonzero:
                used.append(family)
            w = float(self.weights.get(family, 0.0))
            vote = np.sign(np.sum(votes)) if family in ("trend", "momentum") else (1 if any(votes) else 0)
            total += w * float(vote)
            weight_sum += w
        result.families_used = used
        result.score = float(total / weight_sum) if weight_sum else 0.0
        return result

    @property
    def active(self) -> bool:
        return self.enabled
