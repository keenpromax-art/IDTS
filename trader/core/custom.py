"""User-defined strategies.

Two ways to add your own setup without touching the engine:

1. **Declarative (YAML)** - describe the setup in
   ``trader/config/custom_strategies.yaml``. A setup is a *trigger* (one
   comparison), a set of *filters* (more comparisons that must also hold),
   and a *stop* and *target* rule.

2. **Python plugin** - drop a file into ``trader/custom/`` that defines a
   subclass of ``trader.strategies.base.Strategy``. It is discovered and
   loaded at start-up.

The declarative form covers most ideas; the Python form covers the rest.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trader.core.models import Side, Signal
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx, make_signal

# ---------------------------------------------------------------- indicators
# name -> (ctx attribute, default params)

INDICATORS: Dict[str, str] = {
    "close": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "price": "close",
    "ema": "ema",
    "rsi": "rsi",
    "atr": "atr",
    "adx": "adx",
    "macd": "macd",
    "bb": "bb",
    "vwap": "vwap",
    "donchian": "donchian",
    "supertrend": "supertrend",
    "psar": "psar",
    "stoch": "stoch",
    "cci": "cci",
    "roc": "roc",
    "mfi": "mfi",
    "keltner": "keltner",
}

OPERATORS = (">=", "<=", ">", "<", "==", "crosses_above", "crosses_below")


class CustomStrategyError(Exception):
    """Raised when a custom strategy definition is malformed."""


def _ctx_ema(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx._ema(int(spec.get("period", 20)))


def _ctx_rsi(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.rsi_s


def _ctx_atr(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.atr_s


def _ctx_adx(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.adx_s


def _ctx_macd(ctx: Ctx, spec: dict) -> np.ndarray:
    part = str(spec.get("part", "line")).lower()
    return {"line": ctx.macd_line, "signal": ctx.macd_signal,
            "hist": ctx.macd_hist}.get(part, ctx.macd_line)


def _ctx_bb(ctx: Ctx, spec: dict) -> np.ndarray:
    part = str(spec.get("part", "upper")).lower()
    return {"upper": ctx.bb_up, "mid": ctx.bb_mid,
            "lower": ctx.bb_low}.get(part, ctx.bb_up)


def _ctx_vwap(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.vwap_s


def _ctx_donchian(ctx: Ctx, spec: dict) -> np.ndarray:
    part = str(spec.get("part", "high")).lower()
    return ctx.donch_hi_prev if part == "high" else ctx.donch_lo_prev


def _ctx_supertrend(ctx: Ctx, spec: dict) -> np.ndarray:
    part = str(spec.get("part", "line")).lower()
    return ctx.st_dir if part == "direction" else ctx.st_line


def _ctx_psar(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.psar_s


def _ctx_stoch(ctx: Ctx, spec: dict) -> np.ndarray:
    part = str(spec.get("part", "k")).lower()
    return ctx.d if part == "d" else ctx.k


def _ctx_cci(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.cci_s


def _ctx_roc(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.roc_s


def _ctx_mfi(ctx: Ctx, spec: dict) -> np.ndarray:
    return ctx.mfi_s


def _ctx_keltner(ctx: Ctx, spec: dict) -> np.ndarray:
    part = str(spec.get("part", "upper")).lower()
    return {"upper": ctx.kelt_hi_prev, "mid": ctx.kelt_mid,
            "lower": ctx.kelt_lo_prev}.get(part, ctx.kelt_hi_prev)


_SERIES_GETTERS = {
    "ema": _ctx_ema,
    "rsi": _ctx_rsi,
    "atr": _ctx_atr,
    "adx": _ctx_adx,
    "macd": _ctx_macd,
    "bb": _ctx_bb,
    "vwap": _ctx_vwap,
    "donchian": _ctx_donchian,
    "supertrend": _ctx_supertrend,
    "psar": _ctx_psar,
    "stoch": _ctx_stoch,
    "cci": _ctx_cci,
    "roc": _ctx_roc,
    "mfi": _ctx_mfi,
    "keltner": _ctx_keltner,
}


def resolve_series(ctx: Ctx, spec: Any) -> np.ndarray:
    """Turn an indicator spec into a numpy series.

    A spec is either a bare string ("ema") or a mapping
    ``{indicator: ema, period: 20, part: upper}``.
    """
    if isinstance(spec, str):
        spec = {"indicator": spec}
    if not isinstance(spec, dict):
        raise CustomStrategyError(f"bad indicator spec: {spec!r}")
    name = str(spec.get("indicator", "")).lower()
    if name in ("close", "open", "high", "low", "volume", "price"):
        return getattr(ctx, "close" if name == "price" else name)
    getter = _SERIES_GETTERS.get(name)
    if getter is None:
        raise CustomStrategyError(
            f"unknown indicator '{name}'. Available: {', '.join(sorted(INDICATORS))}")
    return getter(ctx, spec)


def resolve_value(ctx: Ctx, spec: Any):
    """A spec is either a number or an indicator spec."""
    if isinstance(spec, (int, float)):
        return float(spec)
    series = resolve_series(ctx, spec)
    return series


def _compare(ctx: Ctx, left: Any, op: str, right: Any) -> bool:
    """Evaluate one comparison at the current bar."""
    op = op.lower().strip()
    if op in ("crosses_above", "crosses_below"):
        lser = resolve_series(ctx, left) if not isinstance(left, (int, float)) else None
        rser = resolve_series(ctx, right) if not isinstance(right, (int, float)) else None
        if lser is None or rser is None:
            raise CustomStrategyError(
                "crosses_above/below need two indicators, not a fixed number")
        i = ctx.bar
        if i < 1:
            return False
        l0, l1 = lser[i - 1], lser[i]
        r0, r1 = rser[i - 1], rser[i]
        if any(np.isnan(x) for x in (l0, l1, r0, r1)):
            return False
        if op == "crosses_above":
            return bool(l0 <= r0 and l1 > r1)
        return bool(l0 >= r0 and l1 < r1)

    lval = resolve_value(ctx, left)
    rval = resolve_value(ctx, right)
    if isinstance(lval, np.ndarray):
        lval = float(lval[ctx.bar])
    if isinstance(rval, np.ndarray):
        rval = float(rval[ctx.bar])
    if np.isnan(lval) or np.isnan(rval):
        return False
    if op == ">=":
        return bool(lval >= rval)
    if op == "<=":
        return bool(lval <= rval)
    if op == ">":
        return bool(lval > rval)
    if op == "<":
        return bool(lval < rval)
    if op == "==":
        return bool(abs(lval - rval) < 1e-9)
    raise CustomStrategyError(f"unknown operator '{op}'")


class DeclarativeStrategy(Strategy):
    """A strategy defined in YAML."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = dict(spec or {})
        self.kind = str(self.spec.get("kind", "trend")).lower()
        self.label = str(self.spec.get("label") or name.replace("_", " ").title())
        self.trigger = self.spec.get("trigger")
        if not self.trigger:
            raise CustomStrategyError(f"custom strategy '{name}' has no trigger")
        self.filters = list(self.spec.get("filters", []) or [])
        self.stop_rule = self.spec.get("stop", {"type": "atr", "mult": 1.5})
        self.target_rule = self.spec.get("target", {"type": "r_multiple", "value": 2.0})
        self.side = str(self.spec.get("side", "auto")).lower()
        self.confidence = float(self.spec.get("confidence", 0.6))
        self.min_bars = int(self.spec.get("min_bars", 35))

    # -- helpers ---------------------------------------------------------
    def _stop_price(self, ctx: Ctx, side: Side) -> Optional[float]:
        entry = ctx.px
        rule = self.stop_rule or {}
        rtype = str(rule.get("type", "atr")).lower()
        if rtype == "atr":
            if ctx.atr <= 0:
                return None
            dist = float(rule.get("mult", 1.5)) * ctx.atr
        elif rtype == "percent":
            dist = entry * float(rule.get("value", 1.0)) / 100.0
        elif rtype == "points":
            dist = float(rule.get("value", 1.0))
        elif rtype == "swing":
            n = int(rule.get("lookback", 10))
            start = max(0, ctx.bar + 1 - n)
            end = ctx.bar + 1
            dist = (entry - float(np.nanmin(ctx.low[start:end]))) if side == Side.LONG \
                else (float(np.nanmax(ctx.high[start:end])) - entry)
        elif rtype == "indicator":
            base = resolve_value(ctx, rule.get("value"))
            if isinstance(base, np.ndarray):
                base = float(base[ctx.bar])
            if np.isnan(base):
                return None
            return base
        else:
            raise CustomStrategyError(f"unknown stop type '{rtype}'")
        if dist <= 0:
            return None
        return entry - dist if side == Side.LONG else entry + dist

    def _target_price(self, ctx: Ctx, side: Side, risk: float) -> Optional[float]:
        entry = ctx.px
        rule = self.target_rule or {}
        rtype = str(rule.get("type", "r_multiple")).lower()
        if rtype == "r_multiple":
            dist = float(rule.get("value", 2.0)) * risk
        elif rtype == "atr":
            if ctx.atr <= 0:
                return None
            dist = float(rule.get("mult", 3.0)) * ctx.atr
        elif rtype == "percent":
            dist = entry * float(rule.get("value", 3.0)) / 100.0
        elif rtype == "points":
            dist = float(rule.get("value", 5.0))
        else:
            raise CustomStrategyError(f"unknown target type '{rtype}'")
        if dist <= 0:
            return None
        return entry + dist if side == Side.LONG else entry - dist

    def _resolve_side(self, trigger: dict) -> Optional[Side]:
        if self.side == "long":
            return Side.LONG
        if self.side == "short":
            return Side.SHORT
        op = str(trigger.get("op", "")).lower()
        if op == "crosses_above":
            return Side.LONG
        if op == "crosses_below":
            return Side.SHORT
        if op in (">", ">="):
            return Side.LONG
        if op in ("<", "<="):
            return Side.SHORT
        return None

    # -- evaluation ------------------------------------------------------
    def generate(self, df, symbol="", market="", ctx=None):
        if df is None:
            return None
        c = ctx if ctx is not None else Ctx(df, symbol, market)
        if c.available < self.min_bars or c.atr <= 0:
            return None
        try:
            if not _compare(c, self.trigger.get("left"), self.trigger.get("op", ">="),
                            self.trigger.get("right")):
                return None
            for flt in self.filters:
                if not _compare(c, flt.get("left"), flt.get("op", ">="),
                                flt.get("right")):
                    return None
        except CustomStrategyError:
            raise
        except Exception:
            return None

        side = self._resolve_side(self.trigger)
        if side is None:
            return None
        stop = self._stop_price(c, side)
        if stop is None:
            return None
        risk = abs(c.px - stop)
        if risk <= 0:
            return None
        target = self._target_price(c, side, risk)
        if target is None:
            return None
        desc = self._describe()
        return make_signal(symbol, market, self.name, side, c.px, stop, target,
                           confidence=self.confidence, reason=desc)

    def generate_from(self, ctx):
        """Use the shared per-symbol context instead of recomputing indicators."""
        return self.generate(ctx.df, ctx.symbol, ctx.market, ctx=ctx)

    def _describe(self) -> str:
        parts = [self._fmt_cmp(self.trigger)]
        for f in self.filters:
            parts.append(self._fmt_cmp(f))
        return " + ".join(parts)

    @staticmethod
    def _fmt_cmp(cmp: dict) -> str:
        def side(s):
            if isinstance(s, (int, float)):
                return f"{s:g}"
            if isinstance(s, str):
                return s
            name = s.get("indicator", "?")
            extra = ""
            if "period" in s:
                extra = str(s["period"])
            elif "part" in s:
                extra = str(s["part"])
            return f"{name}{extra}"
        op = cmp.get("op", ">=")
        op_txt = {"crosses_above": "crosses >", "crosses_below": "crosses <"}.get(op, op)
        return f"{side(cmp.get('left'))} {op_txt} {side(cmp.get('right'))}"

    def params(self) -> dict:
        return {
            "kind": self.kind,
            "trigger": self.trigger,
            "filters": self.filters,
            "stop": self.stop_rule,
            "target": self.target_rule,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------- loading

def load_declarative(path: str) -> List[DeclarativeStrategy]:
    """Load every setup defined in a custom_strategies.yaml file."""
    if not path or not os.path.exists(path):
        return []
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = []
    entries = data.get("strategies") or {}
    for name, spec in entries.items():
        if not spec:
            continue
        try:
            out.append(DeclarativeStrategy(name, spec))
        except CustomStrategyError as e:
            raise CustomStrategyError(f"{path}: {e}") from e
    return out


def load_plugins(folder: str) -> List[Strategy]:
    """Import every .py file in a folder and collect Strategy subclasses."""
    if not folder or not os.path.isdir(folder):
        return []
    found: List[Strategy] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(folder, fname)
        mod_name = f"trader_custom_{os.path.splitext(fname)[0]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        except Exception as e:
            raise CustomStrategyError(f"plugin {path} failed to load: {e}") from e
        for obj in vars(mod).values():
            if (isinstance(obj, type) and issubclass(obj, Strategy)
                    and obj is not Strategy and not obj.__abstractmethods__
                    and obj.__module__ == mod_name):
                found.append(obj())
    return found


def custom_template() -> str:
    """A commented example file to copy and edit."""
    return """# Your own strategies. Copy a block, change the name, edit the numbers.
#
# A setup has three parts:
#   trigger  - the comparison that fires it
#   filters  - more comparisons that must also hold (all of them)
#   stop     - where the protective stop goes
#   target   - where the profit is taken
#
# indicators: close open high low volume price ema rsi atr adx macd bb vwap
#              donchian supertrend psar stoch cci roc mfi keltner
# operators:  > >= < <= == crosses_above crosses_below
#
# For macd / bb / donchian / supertrend / stoch / keltner add "part":
#   macd: line | signal | hist      bb: upper | mid | lower
#   donchian: high | low            supertrend: line | direction
#   stoch: k | d                   keltner: upper | mid | lower

strategies:

  # my_ema_cross:
  #   label: "My EMA Cross"
  #   kind: trend
  #   trigger:
  #     left: {indicator: ema, period: 5}
  #     op: crosses_above
  #     right: {indicator: ema, period: 34}
  #   filters:
  #     - {left: {indicator: adx}, op: ">=", right: 25}
  #     - {left: {indicator: rsi}, op: "<", right: 70}
  #   stop: {type: atr, mult: 1.5}
  #   target: {type: r_multiple, value: 2.0}
  #   confidence: 0.65

  # my_rsi_bounce:
  #   label: "My RSI Bounce"
  #   kind: meanrev
  #   trigger:
  #     left: {indicator: rsi, period: 14}
  #     op: "<"
  #     right: 25
  #   filters:
  #     - {left: {indicator: adx}, op: "<", right: 30}
  #   stop: {type: swing, lookback: 10}
  #   target: {type: percent, value: 2.0}
  #   confidence: 0.6
"""
