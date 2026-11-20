"""Quantitative ranking metrics and realistic execution modelling.

All metrics are derived from realised trade R-multiples; no value is guessed.
``compute_performance_metrics`` is pure and deterministic so it can be tested
against hand-computed fixtures.

Execution modelling enforces next-bar fills: a signal generated on bar i is
entered at bar i+1's open, with spread, slippage and commission applied. If the
next open gaps through the stop, the trade is exited at that open (gap loss)
rather than being filled at the stop price.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExecutionModel:
    mode: str = "next_bar"            # "next_bar" | "same_bar" (legacy)
    slippage_bps: float = 5.0
    spread_bps: float = 2.0
    commission_bps: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict[str, Any]], slippage_bps: float = 5.0) -> "ExecutionModel":
        cfg = dict(cfg or {})
        return cls(
            mode=str(cfg.get("mode", "next_bar")),
            slippage_bps=float(cfg.get("slippage_bps", slippage_bps)),
            spread_bps=float(cfg.get("spread_bps", 2.0)),
            commission_bps=float(cfg.get("commission_bps", 0.0)),
        )

    @property
    def cost_fraction(self) -> float:
        return (self.slippage_bps + self.spread_bps + self.commission_bps) / 10000.0

    def entry_price(self, reference: float, is_long: bool) -> float:
        sign = 1.0 if is_long else -1.0
        return float(reference) * (1.0 + sign * self.cost_fraction)

    def exit_price(self, reference: float, is_long: bool) -> float:
        sign = 1.0 if is_long else -1.0
        return float(reference) * (1.0 - sign * self.cost_fraction)


@dataclass
class PerformanceMetrics:
    trades: int = 0
    expectancy_r: float = 0.0
    net_expectancy_r: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    win_rate_lower_bound: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    sample_size: int = 0

    def to_dict(self):
        return asdict(self)


def compute_performance_metrics(r_multiples: List[float],
                                costs_r: Optional[List[float]] = None,
                                annualise_bars: int = 252) -> PerformanceMetrics:
    """Sharpe, Sortino, drawdown and sample-aware stats from R-multiple trades."""
    r = np.asarray([x for x in (r_multiples or []) if np.isfinite(x)], dtype=float)
    m = PerformanceMetrics(trades=int(r.size), sample_size=int(r.size))
    if r.size == 0:
        return m
    m.expectancy_r = float(np.mean(r))
    if costs_r is not None:
        c = np.asarray([x for x in costs_r if np.isfinite(x)], dtype=float)
        if c.size == r.size:
            m.net_expectancy_r = float(np.mean(r - c))
    m.net_expectancy_r = m.net_expectancy_r if costs_r is not None else m.expectancy_r
    wins = r[r > 0]
    losses = r[r <= 0]
    m.win_rate = float(wins.size / r.size)
    gross_win = float(np.sum(wins))
    gross_loss = abs(float(np.sum(losses)))
    m.profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (
        2.0 if gross_win > 0 else 0.0)

    sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    if sd > 1e-12:
        m.sharpe = float(np.mean(r) / sd * np.sqrt(max(1, annualise_bars)))
    elif np.mean(r) > 0:
        m.sharpe = 10.0
    downside = r[r < 0]
    dsd = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    if dsd > 1e-12:
        m.sortino = float(np.mean(r) / dsd * np.sqrt(max(1, annualise_bars)))
    elif np.mean(r) > 0:
        m.sortino = 10.0
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    m.max_drawdown = float(np.max(peak - equity)) if equity.size else 0.0

    # Wilson lower bound: a 60% win rate over 8 trades is not evidence.
    z = 1.96
    n = float(r.size)
    p = m.win_rate
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    m.win_rate_lower_bound = float(max(0.0, centre - half))
    return m


def walk_forward_folds(n: int, train_bars: int, test_bars: int,
                       folds: int = 3) -> List[Tuple[int, int, int, int]]:
    """Non-overlapping chronological train/test index ranges."""
    train_bars, test_bars, folds = int(train_bars), int(test_bars), int(folds)
    out = []
    if train_bars < 10 or test_bars < 5 or folds < 1:
        return out
    block = train_bars + test_bars
    if n < block * folds:
        folds = max(1, n // block)
    for k in range(folds):
        start = k * block
        tr_end = start + train_bars
        te_end = tr_end + test_bars
        if te_end > n:
            break
        out.append((start, tr_end, tr_end, te_end))
    return out


@dataclass
class RankingContext:
    """Non-backtest inputs to the enhanced score, all optional."""

    relative_strength: float = 0.0
    liquidity_quality: float = 0.5
    data_freshness: float = 1.0
    walk_forward_score: float = 0.0
    regime_compatibility: Optional[float] = None

    def to_dict(self):
        return asdict(self)


DEFAULT_ENHANCED_WEIGHTS = {
    "net_expectancy": 0.22,
    "sharpe": 0.12,
    "sortino": 0.10,
    "sample_size": 0.08,
    "max_drawdown": 0.08,
    "regime": 0.12,
    "relative_strength": 0.08,
    "liquidity": 0.06,
    "freshness": 0.06,
    "walk_forward": 0.08,
}


class QualityRanker:
    """Blend legacy and enhanced scores without changing ``Selector.score``."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.blend = float(cfg.get("blend", 1.0))
        self.weights = dict(cfg.get("weights") or DEFAULT_ENHANCED_WEIGHTS)

    @staticmethod
    def _unit(x: float, lo: float, hi: float) -> float:
        if not np.isfinite(x) or hi <= lo:
            return 0.5
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))

    def enhanced_score(self, stats, regime_fit: float, confidence: float,
                       ctx: RankingContext) -> float:
        w = self.weights
        r = getattr(stats, "expectancy_r", 0.0)
        parts = {
            "net_expectancy": self._unit(getattr(stats, "net_expectancy_r", r), -0.3, 0.6),
            "sharpe": self._unit(getattr(stats, "sharpe", 0.0), -1.0, 2.0),
            "sortino": self._unit(getattr(stats, "sortino", 0.0), -1.0, 2.5),
            "sample_size": self._unit(getattr(stats, "sample_size", stats.trades), 5, 60),
            "max_drawdown": 1.0 - self._unit(getattr(stats, "max_drawdown", 0.0), 0.0, 10.0),
            "regime": max(0.0, min(1.0, ctx.regime_compatibility
                                   if ctx.regime_compatibility is not None else regime_fit)),
            "relative_strength": self._unit(ctx.relative_strength, -2.0, 2.0),
            "liquidity": max(0.0, min(1.0, ctx.liquidity_quality)),
            "freshness": max(0.0, min(1.0, ctx.data_freshness)),
            "walk_forward": self._unit(ctx.walk_forward_score, -0.5, 0.5),
        }
        total = sum(float(w.get(k, 0.0)) * v for k, v in parts.items())
        weight = sum(float(w.get(k, 0.0)) for k in parts)
        return float(total / weight) if weight else 0.5
