"""Debug signal generation."""
import sys
sys.path.insert(0, r"C:\IDTS")

import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from trader.core.selector import Selector
from trader.core.models import Signal, Side
from trader.core.execution import ExecutionModel, FillAssumptions, FillModel
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

# Check when signal fires
print("Checking signal generation across bars:")
for i in range(60, 120):
    ctx = Ctx(df.iloc[:i], "T", "India")
    sig = s.generate_from(ctx)
    if sig is not None:
        print(f"  i={i}: sig={sig.side} entry={sig.entry:.2f} stop={sig.stop:.2f} target={sig.target:.2f}")

# Also check the data
print(f"\nData sample close: {df['close'].values[:5]}")
print(f"Data length: {len(df)}")