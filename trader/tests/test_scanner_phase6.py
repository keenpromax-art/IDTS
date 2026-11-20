"""Deterministic tests for Phase 6 scanner views."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trader.scanner.views import VIEW_NAMES, ResultRow, ScannerViews


class FakeTicker:
    def __init__(self, quotes=None):
        self.quotes = quotes or {}

    def quote(self, symbol):
        if symbol in self.quotes:
            px, ts = self.quotes[symbol]
            return px, ts
        return None


def row(symbol, signal="NONE", market="India", score=0.5, confidence=0.7,
        change=1.0, rv=None, atr_exp=None, rs=None, mtf_align=None,
        setup="ema_cross", plan=None, kind="trend"):
    metrics = {}
    if rv is not None:
        metrics["relative_volume"] = rv
    if atr_exp is not None:
        metrics["atr_expansion"] = atr_exp
    if rs is not None:
        metrics["relative_strength"] = rs
    features = {}
    if rv is not None:
        features["relative_volume"] = rv
    if atr_exp is not None:
        features["atr_expansion"] = atr_exp
    return {
        "symbol": symbol, "market": market, "price": 100.0, "change": change,
        "regime": "trending", "best": setup, "signal": signal, "score": score,
        "confidence": confidence, "best_reason": f"reason {symbol}", "gate": "",
        "plan": plan if plan is not None else
                {"entry": 100.0, "stop": 98.0, "target": 104.0, "rr": 2.0},
        "metrics": metrics, "screen": {"features": features},
        "mtf": {"alignment": mtf_align, "conflicts": [], "families_used": ["trend"]}
                if mtf_align is not None else None,
        "updated": datetime.now(timezone.utc), "stale": False,
        "best_label": setup,
    }


class Kind:
    def __init__(self, kind):
        self.kind = kind


def engine(rows, quotes=None, kinds=None):
    sel = SimpleNamespace(_kinds={k: Kind(v) for k, v in (kinds or {}).items()})
    return SimpleNamespace(last_scan={"India": rows},
                           ticker=FakeTicker(quotes), selector=sel)


def test_all_nine_views_exist():
    assert len(VIEW_NAMES) == 9
    assert set(VIEW_NAMES) == {
        "top_momentum", "strongest_bullish", "strongest_bearish",
        "volume_breakouts", "volatility_expansion", "mean_reversion",
        "relative_strength_leaders", "relative_strength_laggards", "mtf_alignment",
    }


def test_result_row_has_all_required_fields():
    r = ResultRow(ticker="A", market="India", direction="LONG", quote=100.0,
                  quote_age_seconds=1.0, quote_source="live", setup="ema_cross",
                  confidence=0.8, relative_volume=2.0, atr=1.5, entry=100.0,
                  stop=98.0, target=104.0, risk_reward=2.0, explanation="x")
    d = r.to_dict()
    for key in ("ticker", "market", "direction", "quote", "quote_age_seconds",
                "setup", "confidence", "relative_volume", "atr", "entry", "stop",
                "target", "risk_reward", "explanation"):
        assert key in d


def test_bullish_and_bearish_views_rank_by_score_and_confidence():
    rows = [row("A", "LONG", score=0.6, confidence=0.7),
            row("B", "LONG", score=0.9, confidence=0.8),
            row("C", "SHORT", score=0.8, confidence=0.8)]
    sv = ScannerViews(engine(rows))
    assert [x.ticker for x in sv.build("strongest_bullish")] == ["B", "A"]
    assert [x.ticker for x in sv.build("strongest_bearish")] == ["C"]


def test_momentum_view_prefers_actionable_and_largest_move():
    rows = [row("A", signal="NONE", change=20),
            row("B", "LONG", change=2), row("C", "SHORT", change=-5)]
    sv = ScannerViews(engine(rows))
    out = sv.build("top_momentum")
    assert [x.ticker for x in out[:2]] == ["C", "B"]


def test_volume_and_volatility_views():
    rows = [row("A", rv=3.0, atr_exp=1.5, signal="LONG"),
            row("B", rv=0.8, atr_exp=1.1), row("C", rv=2.0, atr_exp=0.9)]
    sv = ScannerViews(engine(rows))
    assert [x.ticker for x in sv.build("volume_breakouts")] == ["A", "C"]
    assert [x.ticker for x in sv.build("volatility_expansion")] == ["A", "B"]


def test_mean_reversion_uses_strategy_kind_not_name_heuristics():
    rows = [row("A", "LONG", setup="rsi_reversal"),
            row("B", "LONG", setup="ema_cross")]
    sv = ScannerViews(engine(rows, kinds={"rsi_reversal": "meanrev",
                                          "ema_cross": "trend"}))
    assert [x.ticker for x in sv.build("mean_reversion")] == ["A"]


def test_relative_strength_leaders_and_laggards():
    rows = [row("A", rs=2.0), row("B", rs=-1.5), row("C", rs=0.2)]
    sv = ScannerViews(engine(rows))
    assert [x.ticker for x in sv.build("relative_strength_leaders")] == ["A", "C", "B"]
    assert [x.ticker for x in sv.build("relative_strength_laggards")] == ["B", "C", "A"]


def test_mtf_alignment_view_and_explanation():
    rows = [row("A", "LONG", mtf_align=0.9), row("B", mtf_align=0.2)]
    sv = ScannerViews(engine(rows))
    out = sv.build("mtf_alignment")
    assert [x.ticker for x in out] == ["A"]
    assert "mtf align=0.90" in out[0].explanation
    assert "reason A" in out[0].explanation


def test_quote_source_and_age_are_reported():
    now = time.time()
    eng = engine([row("A", "LONG"), row("B", "LONG")],
                 quotes={"A": (101.0, now - 3.0)})
    out = {x.ticker: x for x in ScannerViews(eng).build("strongest_bullish")}
    assert out["A"].quote == 101.0 and out["A"].quote_source == "live"
    assert out["A"].quote_age_seconds >= 3.0
    assert out["B"].quote_source == "candle"


def test_all_views_and_invalid_view():
    sv = engine([row("A", "LONG", rv=2.0, atr_exp=1.2, rs=1.0, mtf_align=0.8)])
    views = ScannerViews(sv).all_views(limit=5)
    assert set(views) == set(VIEW_NAMES)
    with pytest.raises(ValueError):
        ScannerViews(sv).build("not_a_view")
