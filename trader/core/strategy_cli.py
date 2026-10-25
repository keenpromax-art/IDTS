"""``python -m trader strategies`` and ``python -m trader add-strategy``.

* ``strategies``  - list every setup that is loaded, built-in and custom.
* ``add-strategy`` - an interactive builder that writes a new setup into
  ``trader/config/custom_strategies.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml

from trader.core.custom import (
    INDICATORS, OPERATORS, CustomStrategyError, DeclarativeStrategy,
    custom_template, load_declarative, load_plugins,
)

# ------------------------------------------------------------------ listing

KIND_LABEL = {
    "trend": "trend following",
    "meanrev": "mean reversion",
    "breakout": "breakout",
    "volume": "volume driven",
}


def collect_strategies(settings: dict, base: Path):
    """Every setup the engine would run: built-in, rules, custom."""
    from trader.strategies import ALL_RULES, ALL_STRATEGIES
    from trader.core.engine import Engine

    built_in = list(ALL_STRATEGIES)
    rules = list(ALL_RULES)
    custom = []
    yaml_path = settings.get("custom_strategies_file") or str(base / "config" / "custom_strategies.yaml")
    plugin_dir = settings.get("custom_strategies_dir") or str(base / "custom")
    try:
        custom.extend(load_declarative(yaml_path))
    except Exception as e:
        print(f"! custom YAML failed to load: {e}")
    try:
        custom.extend(load_plugins(plugin_dir))
    except Exception as e:
        print(f"! custom plugin failed to load: {e}")
    return built_in, rules, custom


def print_strategies(settings: dict, base: Path) -> int:
    built_in, rules, custom = collect_strategies(settings, base)
    enabled = set((settings.get("strategies") or {}).get("enabled") or [])
    disabled = set((settings.get("strategies") or {}).get("disabled") or [])

    def mark(name, on):
        if name in disabled:
            return "off"
        if enabled and name not in enabled:
            return "off"
        return "on"

    print(f"\nSingle setups ({len(built_in)})")
    print(f"  {'name':22} {'kind':10} {'label':30} state")
    for c in built_in:
        print(f"  {c.name:22} {c.kind:10} {c.label:30} {mark(c.name, True)}")

    print(f"\nRules ({len(rules)})")
    print(f"  {'name':30} {'kind':10} {'required':9} label")
    for r in rules:
        req = len(r.required_conditions()) + len(r.required_groups())
        print(f"  {r.name:30} {r.kind:10} {req:<9} {r.label}")

    print(f"\nYour strategies ({len(custom)})")
    if not custom:
        print("  none yet - add one with:  python -m trader add-strategy")
    for c in custom:
        kind = getattr(c, "kind", "trend")
        label = getattr(c, "label", c.name)
        print(f"  {c.name:30} {kind:10} {'':9} {label}")
        if isinstance(c, DeclarativeStrategy):
            print(f"      trigger: {c._describe()}")

    total = len(built_in) + len(rules) + len(custom)
    print(f"\n{total} setups available. Custom setups run unless they are listed in")
    print("`strategies.disabled` or `custom_strategies.disabled` in settings.yaml.\n")
    _print_gates(settings)
    return 0


def _print_gates(settings: dict):
    """Show the rules a setup must clear before it can be traded."""
    s = settings.get("selector", {}) or {}
    r = settings.get("risk", {}) or {}
    print("A setup must clear every one of these before it is traded:")
    gates = [
        ("min_trade_count", s.get("min_trade_count", 8), "backtest trades needed to judge it"),
        ("min_expectancy_r", s.get("min_expectancy_r", 0.0), "it must not have been losing"),
        ("min_profit_factor", s.get("min_profit_factor", 1.1), "gross win must beat gross loss"),
        ("min_win_rate", s.get("min_win_rate", 0.35), "a workable share of winners"),
        ("min_signal_confidence", s.get("min_signal_confidence", 0.55), "the live signal must be convincing"),
        ("min_score", s.get("min_score", 0.55), "the weighted score must clear the bar"),
        ("min_market_score", s.get("min_market_score", 0.60), "market-wide bar for taking a trade"),
        ("max_trades_per_market_per_cycle", s.get("max_trades_per_market_per_cycle", 2),
         "new trades per market per cycle"),
        ("max_concurrent_positions", r.get("max_concurrent_positions", 6), "open at once, in total"),
        ("max_positions_per_market", r.get("max_positions_per_market", 3), "open at once, per market"),
        ("max_trades_per_symbol_per_day", r.get("max_trades_per_symbol_per_day", 2), "per symbol"),
        ("max_daily_loss_pct", r.get("max_daily_loss_pct", 1.5), "kill switch"),
    ]
    print(f"  {'gate':34} {'value':>8}  why")
    for name, value, why in gates:
        print(f"  {name:34} {str(value):>8}  {why}")
    print()


# ------------------------------------------------------------------ wizard

INDICATOR_CHOICES = [
    ("ema", "EMA - exponential moving average", {"period": 20}),
    ("rsi", "RSI - relative strength index", {"period": 14}),
    ("adx", "ADX - trend strength", {}),
    ("atr", "ATR - average true range", {"period": 14}),
    ("macd", "MACD - use part: line | signal | hist", {}),
    ("bb", "Bollinger bands - use part: upper | mid | lower", {}),
    ("vwap", "VWAP - session volume weighted average", {}),
    ("donchian", "Donchian channel - use part: high | low", {"period": 20}),
    ("supertrend", "Supertrend - use part: line | direction", {}),
    ("close", "Close price", {}),
    ("price", "Price (same as close)", {}),
]

STOP_CHOICES = [
    ("atr", "ATR multiple", {"type": "atr", "mult": 1.5}),
    ("percent", "Percentage of price", {"type": "percent", "value": 1.0}),
    ("swing", "Recent swing low / high", {"type": "swing", "lookback": 10}),
    ("indicator", "An indicator level (e.g. EMA50)", {"type": "indicator", "value": {"indicator": "ema", "period": 50}}),
]

TARGET_CHOICES = [
    ("r_multiple", "R multiple (times the risk)", {"type": "r_multiple", "value": 2.0}),
    ("atr", "ATR multiple", {"type": "atr", "mult": 3.0}),
    ("percent", "Percentage of price", {"type": "percent", "value": 3.0}),
]


def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    if not ans and default is not None:
        return default
    return ans


def _pick(prompt: str, choices, allow_none=False):
    print(f"\n{prompt}")
    for i, (key, desc, _) in enumerate(choices, 1):
        print(f"  {i:2}. {key:12} {desc}")
    if allow_none:
        print(f"  0.  (none)")
    while True:
        ans = _ask("choice", "1" if not allow_none else "0")
        try:
            n = int(ans)
        except ValueError:
            print("  please enter a number")
            continue
        if allow_none and n == 0:
            return None
        if 1 <= n <= len(choices):
            return choices[n - 1]
        print("  out of range")


def _ask_num(prompt: str, default: float) -> float:
    raw = _ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _indicator_spec(ctx: dict) -> dict:
    """Turn the wizard's answers into an indicator spec."""
    name, _, defaults = ctx
    spec = {"indicator": name}
    for k, v in defaults.items():
        spec[k] = v
    if name in ("ema", "rsi", "atr", "donchian"):
        raw = _ask("  period", str(spec.get("period")))
        try:
            spec["period"] = int(raw)
        except ValueError:
            pass
    if name in ("macd", "bb", "donchian", "supertrend", "stoch", "keltner"):
        part_default = {"macd": "line", "bb": "upper", "donchian": "high",
                        "supertrend": "line", "stoch": "k", "keltner": "upper"}[name]
        spec["part"] = _ask("  part", part_default)
    return spec


def _write_strategies(path: Path, strategies: dict) -> None:
    """Write the strategies block, keeping the commented header intact.

    PyYAML cannot round-trip comments, so the text above the top-level
    `strategies:` key is preserved verbatim and only the block below it is
    replaced.
    """
    header = ""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        cut = None
        for i, line in enumerate(lines):
            if line.startswith("strategies:"):
                cut = i
                break
        if cut is not None:
            header = "".join(lines[:cut])
        else:
            header = text if text.endswith("\n") else text + "\n"
    block = yaml.safe_dump({"strategies": strategies}, sort_keys=False,
                            default_flow_style=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + block, encoding="utf-8")


def add_strategy_wizard(settings: dict, base: Path) -> int:
    """Ask a few questions and write a new declarative setup."""
    print("\n=== Add a strategy ===")
    print("This writes a new setup into trader/config/custom_strategies.yaml.")
    print("It runs like any built-in setup: it is scored, ranked and traded")
    print("under exactly the same rules.\n")

    name = _ask("name for the setup (lowercase, underscores)", "my_setup")
    name = name.strip().lower().replace(" ", "_")
    if not name or not name.replace("_", "").isalnum():
        print("! that name is not usable")
        return 1
    label = _ask("display name", name.replace("_", " ").title())

    print("\n1. The trigger - what fires the setup")
    left_choice = _pick("Left side:", INDICATOR_CHOICES)
    left = _indicator_spec(left_choice)
    op = _ask(f"operator {', '.join(OPERATORS)}", "crosses_above")
    if op not in OPERATORS:
        print("! unknown operator")
        return 1
    right_choice = _pick("Right side:", INDICATOR_CHOICES, allow_none=True)
    if right_choice is None:
        right = 0.0
    else:
        right = _indicator_spec(right_choice)
        if right["indicator"] in ("close", "price"):
            right = right["indicator"]

    print("\n2. Filters - conditions that must also hold (all of them)")
    filters = []
    while True:
        add = _ask("add a filter? [y/N]", "n")
        if add.lower() not in ("y", "yes"):
            break
        f_left_choice = _pick("Filter left side:", INDICATOR_CHOICES)
        f_left = _indicator_spec(f_left_choice)
        f_op = _ask(f"operator {', '.join(OPERATORS)}", ">=")
        if f_op not in OPERATORS:
            print("  ! unknown operator, filter skipped")
            continue
        f_right_choice = _pick("Filter right side:", INDICATOR_CHOICES, allow_none=True)
        if f_right_choice is None:
            f_right = 0.0
        else:
            f_right = _indicator_spec(f_right_choice)
            if f_right["indicator"] in ("close", "price"):
                f_right = f_right["indicator"]
        filters.append({"left": f_left, "op": f_op, "right": f_right})

    print("\n3. Stop loss")
    stop_choice = _pick("Stop type:", STOP_CHOICES)
    stop = dict(stop_choice[2])
    if stop_choice[0] == "atr":
        stop["mult"] = _ask_num("  ATR multiple", stop.get("mult", 1.5))
    elif stop_choice[0] == "percent":
        stop["value"] = _ask_num("  percent", stop.get("value", 1.0))
    elif stop_choice[0] == "swing":
        raw = _ask("  lookback bars", str(stop.get("lookback", 10)))
        try:
            stop["lookback"] = int(raw)
        except ValueError:
            pass

    print("\n4. Target")
    target_choice = _pick("Target type:", TARGET_CHOICES)
    target = dict(target_choice[2])
    if target_choice[0] == "r_multiple":
        target["value"] = _ask_num("  R multiple", target.get("value", 2.0))
    elif target_choice[0] == "atr":
        target["mult"] = _ask_num("  ATR multiple", target.get("mult", 3.0))
    elif target_choice[0] == "percent":
        target["value"] = _ask_num("  percent", target.get("value", 3.0))

    kind = _ask("kind: trend | meanrev | breakout | volume", "trend")
    if kind not in KIND_LABEL:
        kind = "trend"
    conf = _ask("confidence 0.0-1.0", "0.6")
    try:
        conf = float(conf)
    except ValueError:
        conf = 0.6

    spec = {
        "label": label,
        "kind": kind,
        "trigger": {"left": left, "op": op, "right": right},
        "filters": filters,
        "stop": stop,
        "target": target,
        "confidence": conf,
    }

    # validate before writing
    try:
        DeclarativeStrategy(name, spec)
    except CustomStrategyError as e:
        print(f"! the setup is not valid: {e}")
        return 1

    path = Path(settings.get("custom_strategies_file")
                or (base / "config" / "custom_strategies.yaml"))
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if not data.get("strategies"):
        data["strategies"] = {}
    if name in data["strategies"]:
        if _ask(f"'{name}' already exists - overwrite? [y/N]", "n").lower() not in ("y", "yes"):
            print("cancelled")
            return 1
    data["strategies"][name] = spec
    _write_strategies(path, data["strategies"])

    print(f"\nSaved '{name}' to {path}")
    print("It is live on the next start. Check it with:")
    print(f"  python -m trader strategies")
    return 0
