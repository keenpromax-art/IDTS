"""Debug the specific failing tests."""
import sys
sys.path.insert(0, r"C:\IDTS")

import yaml
from pathlib import Path

from trader.core.selector import Selector
from trader.core.models import Signal, Side
from trader.core.execution import ExecutionModel, FillAssumptions, FillModel
from trader.core.data_quality import validate_candle_df
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx
from trader.tests.test_custom_strategies import make_df, DeclarativeStrategy
import pandas as pd

PK = Path(r"C:\IDTS\trader")

# Test 1: test_selector_picks_a_user_defined_setup
print("=" * 60)
print("TEST 1: test_selector_picks_a_user_defined_setup")
print("=" * 60)

settings = yaml.safe_load(open(PK / "config" / "settings.yaml"))
settings = dict(settings)
settings["selector"] = dict(settings["selector"])
settings["selector"].update({"min_trade_count": 2, "min_profit_factor": 0.5,
                             "min_win_rate": 0.2, "min_expectancy_r": -1.0,
                             "min_score": 0.1})

sel = Selector(settings)

spec = {
    "label": "Mine", "kind": "trend",
    "trigger": {"left": {"indicator": "ema", "period": 5}, "op": "crosses_above",
                "right": {"indicator": "ema", "period": 20}},
    "filters": [{"left": {"indicator": "adx"}, "op": ">=", "right": 10}],
    "stop": {"type": "atr", "mult": 1.5},
    "target": {"type": "r_multiple", "value": 2.0},
    "confidence": 0.65,
}
s = DeclarativeStrategy("my_ema_cross", spec)
sel.set_strategies([s])
df = make_df(n=320, drift=0.0, vol=1.0, seed=7)
signals = picked = 0
for i in range(60, len(df)):
    ctx = Ctx(df.iloc[:i], "T", "India")
    sig = s.generate_from(ctx)
    if sig is None:
        continue
    signals += 1
    res = sel.evaluate("T", df.iloc[:i], [s], {"my_ema_cross": sig}, i)
    if res is not None:
        picked += 1
        print(f"  i={i}: picked={res[0]}, score={res[1]:.2f}, bs.trades={res[2].trades}, win_rate={res[2].win_rate:.2f}, pf={res[2].profit_factor:.2f}")

print(f"\nTotal signals: {signals}, picked: {picked}")
print(f"Expected: picked > 0")

# Test 2: test_backtest_sees_the_cursor
print("\n" + "=" * 60)
print("TEST 2: test_backtest_sees_the_cursor")
print("=" * 60)

# Reset selector
sel2 = Selector(settings)
s2 = DeclarativeStrategy("my_ema_cross", spec)
sel2.set_strategies([s2])

stats_fail = 0
for i in range(60, min(80, len(df))):
    ctx = Ctx(df.iloc[:i], "T", "India")
    sig = s2.generate_from(ctx)
    if sig is None:
        continue
    try:
        stats = sel2.backtest(s2, df.iloc[:i], "T")
        print(f"  i={i}: trades={stats.trades}, win_rate={stats.win_rate:.2f}, pf={stats.profit_factor:.2f}")
        if stats.trades > 0:
            print(f"    SUCCESS at i={i}")
        else:
            stats_fail += 1
            if stats_fail <= 3:
                print(f"    FAIL: no trades at i={i}")
    except Exception as e:
        print(f"  i={i}: ERROR: {type(e).__name__}: {e}")

print(f"\nTest 2: {stats_fail} backtests with 0 trades out of {min(80, len(df)) - 60} attempts")