"""Detailed debug of test_selector_picks_a_user_defined_setup."""
import sys
sys.path.insert(0, r"C:\IDTS")

import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from trader.core.selector import Selector
from trader.core.models import Signal, Side
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx
from trader.tests.test_custom_strategies import make_df, DeclarativeStrategy

PK = Path(r"C:\IDTS\trader")

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

# Call evaluate and trace
signals = 0
picked = 0
for i in range(60, min(80, len(df))):
    ctx = Ctx(df.iloc[:i], "T", "India")
    sig = s.generate_from(ctx)
    if sig is None:
        continue
    signals += 1
    try:
        res = sel.evaluate("T", df.iloc[:i], [s], {"my_ema_cross": sig}, i)
        if res is not None:
            picked += 1
            bs = res[2]
            print(f"  i={i}: PICKED! strategy={res[0]}, score={res[1]:.2f}, trades={bs.trades}, pf={bs.profit_factor:.2f}, wr={bs.win_rate:.2f}, eligible={bs.eligible}")
            # Debug eligible
            if not bs.eligible:
                print(f"    Not eligible because: trades<{sel.min_trades}={bs.trades<sel.min_trades}, expectancy_r<{sel.min_expectancy_r}={bs.expectancy_r<sel.min_expectancy_r}, pf<{sel.min_profit_factor}={bs.profit_factor<sel.min_profit_factor}, wr<{sel.min_win_rate}={bs.win_rate<sel.min_win_rate}")
        else:
            print(f"  i={i}: res is None or rejected")
    except Exception as e:
        import traceback
        print(f"  i={i}: ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

print(f"\nTotal signals (60-80): {signals}, picked: {picked}")

# Now test full range
print("\n--- Full range test ---")
signals2 = 0
picked2 = 0
for i in range(60, len(df)):
    ctx = Ctx(df.iloc[:i], "T", "India")
    sig = s.generate_from(ctx)
    if sig is None:
        continue
    signals2 += 1
    try:
        res = sel.evaluate("T", df.iloc[:i], [s], {"my_ema_cross": sig}, i)
        if res is not None:
            picked2 += 1
    except Exception as e:
        pass

print(f"Total signals (60-320): {signals2}, picked: {picked2}")