"""More detailed debug of backtest execution."""
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
import pandas as pd

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

# Test backtest at i=100
i = 100
ctx = Ctx(df.iloc[:i], "T", "India")
sig = s.generate_from(ctx)
print(f"i={i}, sig={sig}")
if sig is None:
    print("  No signal generated")
else:
    print(f"  sig.side={sig.side}, entry={sig.entry}, stop={sig.stop}, target={sig.target}")
    
    # Now trace through the backtest manually
    df_slice = df.iloc[:i]
    df_backtest = sel._backtest_window(df_slice)
    print(f"  backtest window: {len(df_backtest)} bars")
    df_backtest = df_backtest.iloc[:-1]  # drop live bar
    print(f"  after dropping live bar: {len(df_backtest)} bars")
    if len(df_backtest) < 40:
        print("  Not enough bars for backtest")
    else:
        # Manual execution trace
        from trader.core.np_indicators import adx_np, atr_np
        high = df_backtest["high"].to_numpy(dtype=float)
        low = df_backtest["low"].to_numpy(dtype=float)
        close = df_backtest["close"].to_numpy(dtype=float)
        
        slippage_bps = 5.0 / 10000.0
        entry_raw = float(close[0])
        
        # Simulate the execution model
        assumptions = FillAssumptions(
            spread_bps=2.0, fill_model=FillModel.CONSERVATIVE,
            slippage_bps=5.0, latency_seconds=0.0,
            fill_delay_seconds=0.0, partial_fill_probability=0.0,
            min_fill_ratio=1.0, gap_fill_at_worst=True,
            brokerage_config={})
        execution = ExecutionModel(assumptions=assumptions, market_cfg={})
        
        # Execute market order
        res = execution.execute_market_order(
            Order(id='test', symbol='T', market='', side=sig.side,
                  qty=1.0, order_type=OrderType.MARKET,
                  price=entry_raw, stop=float(sig.stop), target=float(sig.target),
                  strategy=s.name),
            reference_price=entry_raw,
            signal_time=df_backtest.index[0],
            high=float(high[0]),
            low=float(low[0])
        )
        print(f"  Execution result: status={res.status}, avg_price={res.avg_price}")
        print(f"  limitation_notes: {res.limitation_notes}")
        
        # Check stop/target collision
        exit_price, exit_reason = execution.check_stop_target_collision(
            sig.side, res.avg_price, float(sig.stop), float(sig.target),
            float(high[0]), float(low[0])
        )
        print(f"  Collision check: exit_price={exit_price}, exit_reason={exit_reason}")