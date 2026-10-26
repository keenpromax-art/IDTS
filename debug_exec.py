"""Debug the custom strategy test."""
import sys
sys.path.insert(0, r"C:\IDTS")

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from trader.core.selector import Selector
from trader.core.models import Signal, Side
from trader.core.execution import ExecutionModel, FillAssumptions, FillModel
from trader.core.data_quality import validate_candle_df
from trader.strategies.base import Strategy
from trader.strategies.context import Ctx
from trader.tests.test_custom_strategies import make_df

PK = Path(r"C:\IDTS\trader")

# Load settings
settings = yaml.safe_load(open(PK / "config" / "settings.yaml"))
settings = dict(settings)
settings["selector"] = dict(settings["selector"])
settings["selector"].update({"min_trade_count": 2, "min_profit_factor": 0.5,
                             "min_win_rate": 0.2, "min_expectancy_r": -1.0,
                             "min_score": 0.1})

sel = Selector(settings)

# Make data
df = make_df(n=320, drift=0.0, vol=1.0, seed=7)

# Create a simple custom strategy class
class MyEMACross(Strategy):
    name = "my_ema_cross"
    def generate_from(self, ctx):
        close = ctx.close
        if len(close) < 20:
            return None
        if len(close) < 2:
            return None
        # Simple EMA cross using close prices
        ema_fast = np.mean(close[-5:])
        ema_slow = np.mean(close[-20:]) if len(close) >= 20 else np.mean(close)
        last_close = float(close[-1])
        if ema_fast > ema_slow:
            return Signal(symbol=ctx.symbol, market=ctx.market, side=Side.LONG,
                         strategy=self.name, entry=last_close,
                         stop=last_close * 0.98, target=last_close * 1.04,
                         confidence=0.65)
        elif ema_fast < ema_slow:
            return Signal(symbol=ctx.symbol, market=ctx.market, side=Side.SHORT,
                         strategy=self.name, entry=last_close,
                         stop=last_close * 1.02, target=last_close * 0.96,
                         confidence=0.65)
        return None
    
    def generate(self, df, symbol, market, ctx=None):
        return self.generate_from(ctx)

s = MyEMACross()
sel.set_strategies([s])

# Test evaluate a few times
print(f"Data length: {len(df)}")
print(f"Data sample close: {df['close'].values[:5]}")

signals = picked = 0
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
            print(f"  i={i}: picked={res[0]}, score={res[1]:.2f}, trades={bs.trades}, win_rate={bs.win_rate:.2f}, pf={bs.profit_factor:.2f}, eligible={bs.eligible}")
        else:
            print(f"  i={i}: res is None")
    except Exception as e:
        print(f"  i={i}: ERROR: {type(e).__name__}: {e}")

print(f"\nTotal signals: {signals}, picked: {picked}")