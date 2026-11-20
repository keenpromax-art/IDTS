"""Dedicated scanner views over the engine's latest scan rows.

Pure data layer: no Streamlit/Rich dependency, so view selection and field
mapping are deterministic and unit-testable. Each row exposes the execution
facts a trader needs to judge a candidate, including quote age and the exact
source of the price shown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

VIEW_NAMES = (
    "top_momentum",
    "strongest_bullish",
    "strongest_bearish",
    "volume_breakouts",
    "volatility_expansion",
    "mean_reversion",
    "relative_strength_leaders",
    "relative_strength_laggards",
    "mtf_alignment",
)


@dataclass
class ResultRow:
    ticker: str
    market: str
    direction: str
    quote: Optional[float]
    quote_age_seconds: Optional[float]
    quote_source: str
    setup: str
    confidence: Optional[float]
    relative_volume: Optional[float]
    atr: Optional[float]
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    risk_reward: Optional[float]
    explanation: str
    score: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


class ScannerViews:
    """Build ranked view models from ``engine.last_scan``."""

    def __init__(self, engine):
        self.engine = engine

    def _rows(self):
        for market, rows in (getattr(self.engine, "last_scan", {}) or {}).items():
            for r in rows or []:
                yield market, r

    def _quote(self, symbol: str, row: Dict[str, Any]):
        ticker = getattr(self.engine, "ticker", None)
        if ticker is not None:
            try:
                q = ticker.quote(symbol)
                if q is not None:
                    return q[0], max(0.0, time.time() - q[1]), "live"
            except Exception:
                pass
        up = row.get("updated")
        age = None
        if isinstance(up, datetime):
            ref = up if up.tzinfo else up.replace(tzinfo=timezone.utc)
            age = max(0.0, (datetime.now(timezone.utc) - ref).total_seconds())
        return row.get("price"), age, "candle"

    def _map(self, market: str, r: Dict[str, Any]) -> ResultRow:
        px, age, source = self._quote(r.get("symbol", ""), r)
        plan = r.get("plan") or {}
        metrics = r.get("metrics") or {}
        features = ((r.get("screen") or {}).get("features") or {})
        mtf = r.get("mtf") or {}
        parts = []
        if r.get("best_reason"):
            parts.append(str(r["best_reason"]))
        if r.get("regime"):
            parts.append(f"regime={r['regime']}")
        if r.get("gate"):
            parts.append(f"gate={r['gate']}")
        if mtf:
            parts.append(f"mtf align={mtf.get('alignment', 0):.2f}"
                         + (f" conflicts={mtf.get('conflicts')}" if mtf.get("conflicts") else ""))
        return ResultRow(
            ticker=r.get("symbol", ""), market=market,
            direction=str(r.get("signal") or "NONE"),
            quote=_num(px), quote_age_seconds=age, quote_source=source,
            setup=str(r.get("best") or r.get("best_label") or "-"),
            confidence=_num(r.get("confidence")),
            relative_volume=_num(metrics.get("relative_volume", features.get("relative_volume"))),
            atr=_num(metrics.get("atr", features.get("atr"))),
            entry=_num(plan.get("entry")), stop=_num(plan.get("stop")),
            target=_num(plan.get("target")), risk_reward=_num(plan.get("rr")),
            explanation=" | ".join(p for p in parts if p) or "no explanation available",
            score=_num(r.get("score")),
        )

    def _kind(self, name: str) -> str:
        sel = getattr(self.engine, "selector", None)
        obj = (getattr(sel, "_kinds", {}) or {}).get(name)
        return str(getattr(obj, "kind", "") or "")

    def build(self, view: str, limit: int = 25) -> List[ResultRow]:
        if view not in VIEW_NAMES:
            raise ValueError(f"unknown scanner view: {view}")
        items = [self._map(m, r) for m, r in self._rows()]
        actionable = [x for x in items if x.direction in ("LONG", "SHORT")]

        if view == "top_momentum":
            items = sorted(items, key=lambda x: (x.direction == "NONE",
                                                 -(abs(_num(self._raw_change(x)) or 0))))
        elif view == "strongest_bullish":
            items = sorted([x for x in actionable if x.direction == "LONG"],
                           key=lambda x: -(x.score or 0) - (x.confidence or 0))
        elif view == "strongest_bearish":
            items = sorted([x for x in actionable if x.direction == "SHORT"],
                           key=lambda x: -(x.score or 0) - (x.confidence or 0))
        elif view == "volume_breakouts":
            items = sorted([x for x in items if (x.relative_volume or 0) >= 1.5],
                           key=lambda x: -(x.relative_volume or 0))
        elif view == "volatility_expansion":
            def exp(x):
                r = self._raw_row(x)
                m = r.get("metrics") or {}
                f = (r.get("screen") or {}).get("features") or {}
                return _num(m.get("atr_expansion", f.get("atr_expansion"))) or 0
            items = sorted([x for x in items if exp(x) > 1.0], key=lambda x: -exp(x))
        elif view == "mean_reversion":
            items = [x for x in actionable if self._kind(x.setup) == "meanrev"]
        elif view in ("relative_strength_leaders", "relative_strength_laggards"):
            def rs(x):
                r = self._raw_row(x)
                m = r.get("metrics") or {}
                f = (r.get("screen") or {}).get("features") or {}
                return _num(m.get("relative_strength", f.get("relative_strength_pct")))
            items = [x for x in items if rs(x) is not None]
            items.sort(key=lambda x: rs(x), reverse=view.endswith("leaders"))
        elif view == "mtf_alignment":
            def align(x):
                r = self._raw_row(x)
                return _num((r.get("mtf") or {}).get("alignment")) or 0
            items = sorted([x for x in items if align(x) >= 0.5], key=lambda x: -align(x))
        return items[:max(0, int(limit))]

    def _index(self):
        return {(r.get("market"), r.get("symbol")): r
                for m, rows in (getattr(self.engine, "last_scan", {}) or {}).items()
                for r in (rows or [])}

    def _raw_row(self, item: ResultRow) -> Dict[str, Any]:
        return self._index().get((item.market, item.ticker), {})

    def _raw_change(self, item: ResultRow):
        r = self._raw_row(item)
        return r.get("change")

    def all_views(self, limit: int = 25) -> Dict[str, List[ResultRow]]:
        return {name: self.build(name, limit) for name in VIEW_NAMES}
