from __future__ import annotations

"""Composite trading setups ("rules").

A rule is a set of named conditions. It only produces a signal when *every*
required condition lines up on the same closed bar - that is the alignment
test. Optional conditions add confidence when they agree.

Each condition is a callable ``fn(ctx) -> bool`` where ``ctx`` is a
``trader.strategies.context.Ctx`` holding every pre-computed indicator.

The reason string on the signal lists the conditions that were met, so the
dashboard can show exactly why the system wanted to trade.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from trader.core.models import Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal

Condition = Callable[[Ctx], bool]


@dataclass
class Condition:
    """One named test inside a rule.

    `group` marks a set of mutually-exclusive conditions - for example
    "close above Keltner" vs "close below Keltner". A group passes when at
    least one of its members passes, and the group as a whole is required
    unless `required` is False.
    """

    key: str
    text: str
    test: Condition
    required: bool = True
    weight: float = 1.0
    group: str = ""

    def evaluate(self, ctx: Ctx) -> bool:
        try:
            return bool(self.test(ctx))
        except Exception:
            return False


class Rule(Strategy):
    """A named setup that fires only when its conditions align."""

    kind = "trend"

    def __init__(self, name: str, label: str, conditions: List[Condition],
                 side: str = "auto", kind: str = "trend",
                 stop_atr: float = 1.5, target_rr: float = 2.0,
                 min_optional: int = 0, min_confidence: float = 0.5,
                 stop_ref: str = "atr", target_ref: str = "atr",
                 bull_keys: tuple = (), bear_keys: tuple = ()):
        self.name = name
        self.label = label
        self.kind = kind
        self.conditions = conditions
        self.side = side          # "long", "short" or "auto"
        self.stop_atr = stop_atr
        self.target_rr = target_rr
        self.min_optional = min_optional
        self.min_confidence = min_confidence
        self.stop_ref = stop_ref  # "atr" | "ema" | "swing"
        self.target_ref = target_ref
        # condition keys that imply a long / short bias when side == "auto"
        self.bull_keys = bull_keys
        self.bear_keys = bear_keys

    # -- introspection ---------------------------------------------------
    def required_conditions(self) -> List[Condition]:
        """Conditions that must all pass (groups are handled separately)."""
        return [c for c in self.conditions if c.required and not c.group]

    def optional_conditions(self) -> List[Condition]:
        return [c for c in self.conditions if not c.required and not c.group]

    def groups(self) -> dict:
        """group name -> [conditions], for the mutually-exclusive sets."""
        out: dict = {}
        for c in self.conditions:
            if c.group:
                out.setdefault(c.group, []).append(c)
        return out

    def required_groups(self) -> dict:
        return {k: v for k, v in self.groups().items()
                if any(c.required for c in v)}

    def params(self) -> dict:
        return {
            "conditions": [c.text for c in self.conditions],
            "required": len(self.required_conditions()),
            "groups": {k: len(v) for k, v in self.groups().items()},
            "stop_atr": self.stop_atr,
            "target_rr": self.target_rr,
        }

    # -- evaluation ------------------------------------------------------
    def evaluate(self, ctx: Ctx) -> Optional[Tuple[Side, List[str], float]]:
        """Return (side, matched condition texts, confidence) or None."""
        if ctx.available < 30:
            return None
        required = self.required_conditions()
        if not required and not self.required_groups():
            return None
        matched: List[str] = []
        for cond in required:
            if not cond.evaluate(ctx):
                return None
            matched.append(cond.text)

        # every required group needs at least one member to pass
        group_hits = {}
        for name, members in self.required_groups().items():
            winners = [c for c in members if c.evaluate(ctx)]
            if not winners:
                return None
            group_hits[name] = winners
            matched.extend(c.text for c in winners)

        optional = self.optional_conditions()
        opt_hits = 0
        for cond in optional:
            if cond.evaluate(ctx):
                opt_hits += 1
                matched.append(cond.text)
        if opt_hits < self.min_optional:
            return None

        total = len(required) + len(self.required_groups()) + len(optional)
        confidence = (len(required) + len(self.required_groups()) + opt_hits) / total if total else 0.0
        confidence = min(1.0, 0.45 + 0.55 * confidence)
        if confidence < self.min_confidence:
            return None

        side = self._resolve_side(ctx, group_hits)
        if side is None:
            return None
        return side, matched, confidence

    def _resolve_side(self, ctx: Ctx, group_hits=None) -> Optional[Side]:
        if self.side == "long":
            return Side.LONG
        if self.side == "short":
            return Side.SHORT
        group_hits = group_hits or {}
        # a "direction" group tells us the bias
        winners = group_hits.get("direction", [])
        if winners:
            keys = {c.key for c in winners}
            if keys & set(self.bull_keys):
                return Side.LONG
            if keys & set(self.bear_keys):
                return Side.SHORT
            return None
        # otherwise infer from the bias conditions that passed
        matched_keys = {c.key for c in self.conditions if c.evaluate(ctx)}
        bull = bool(matched_keys & set(self.bull_keys))
        bear = bool(matched_keys & set(self.bear_keys))
        if bull and not bear:
            return Side.LONG
        if bear and not bull:
            return Side.SHORT
        return None

    def _stop_price(self, ctx: Ctx, side: Side) -> Optional[float]:
        entry = ctx.px
        if self.stop_ref == "ema":
            base = ctx.val(ctx.ema50)
            if not pd.isna(base):
                return base
        if self.stop_ref == "swing":
            end = ctx.bar + 1
            start = max(0, end - 10)
            if side == Side.LONG:
                return float(np.nanmin(ctx.low[start:end]))
            return float(np.nanmax(ctx.high[start:end]))
        if ctx.atr <= 0:
            return None
        return entry - self.stop_atr * ctx.atr if side == Side.LONG else entry + self.stop_atr * ctx.atr

    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < 30:
            return None
        ctx = c
        result = self.evaluate(ctx)
        if result is None:
            return None
        side, matched, confidence = result
        entry = ctx.px
        stop = self._stop_price(ctx, side)
        if stop is None:
            return None
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        target = entry + self.target_rr * risk if side == Side.LONG else entry - self.target_rr * risk
        reason = " + ".join(matched)
        return make_signal(symbol, market, self.name, side, entry, stop, target,
                           confidence=confidence, reason=reason)

    def generate_from(self, ctx):
        """Rules evaluate against the shared per-symbol context."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)
