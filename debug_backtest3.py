"""Debug backtest at signal bar."""
import sys
sys.path.insert(0, r"C:\IDTS")

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from trader.core.selector import Selector
from trader.core.models import Signal, Side, OrderType, Order, OrderStatus, ExitReason
from trader.core.execution import ExecutionModel, FillAssumptions, FillModel
from trader.core.data_quality import validate_candle_df
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

# Test at i=68 where LONG signal fires
i = 68
ctx = Ctx(df.iloc[:i], "T", "India")
sig = s.generate_from(ctx)
print(f"i={i}: sig={sig}")
if sig is None:
    print("  No signal")
else:
    print(f"  sig.side={sig.side}, entry={sig.entry:.2f}, stop={sig.stop:.2f}, target={sig.target:.2f}")
    
    # Run the backtest
    try:
        stats = sel.backtest(s, df.iloc[:i], "T")
        print(f"\nBacktest results:")
        print(f"  trades={stats.trades}")
        print(f"  wins={stats.wins}, losses={stats.losses}")
        print(f"  win_rate={stats.win_rate:.2f}")
        print(f"  profit_factor={stats.profit_factor:.2f}")
        print(f"  expectancy_r={stats.expectancy_r:.2f}")
        print(f"  max_drawdown={stats.max_drawdown:.2f}")
    except Exception as e:
        import traceback
        print(f"\n  ERROR running backtest: {type(e).__name__}: {e}")
        traceback.print_exc()