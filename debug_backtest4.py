"""Debug backtrace execution details."""
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
df_slice = df.iloc[:i]
ctx = Ctx(df_slice, "T", "India")
sig = s.generate_from(ctx)
print(f"i={i}: sig={sig.side} entry={sig.entry:.2f} stop={sig.stop:.2f} target={sig.target:.2f}")

# Now trace the backtest execution step by step
df_backtest = sel._backtest_window(df_slice)
print(f"Backtest window: {len(df_backtest)} bars")
df_backtest = df_backtest.iloc[:-1]  # drop live bar
print(f"After dropping live bar: {len(df_backtest)} bars")

high = df_backtest["high"].to_numpy(dtype=float)
low = df_backtest["low"].to_numpy(dtype=float)
close = df_backtest["close"].to_numpy(dtype=float)

# The backtest starts from bar 0 of the sliced frame
# But we need to map to the original data
# Let's just use the first few bars of df_backtest

# Get the execution model from the selector
assumptions = FillAssumptions(
    spread_bps=2.0, fill_model=FillModel.CONSERVATIVE,
    slippage_bps=5.0, latency_seconds=0.0,
    fill_delay_seconds=0.0, partial_fill_probability=0.0,
    min_fill_ratio=1.0, gap_fill_at_worst=True,
    brokerage_config={})
execution = ExecutionModel(assumptions=assumptions, market_cfg={})

# Execute market order on bar 0 of the backtest window
# reference_price = close[0], high[0], low[0]
ref_price = float(close[0])
h = float(high[0])
l = float(low[0])

print(f"\nBar 0 of backtest: close={ref_price:.2f}, high={h:.2f}, low={l:.2f}")

res = execution.execute_market_order(
    Order(id='test', symbol='T', market='', side=sig.side,
          qty=1.0, order_type=OrderType.MARKET,
          price=ref_price, stop=float(sig.stop), target=float(sig.target),
          strategy=s.name),
    reference_price=ref_price,
    signal_time=df_backtest.index[0],
    high=h,
    low=l,
)
print(f"Execution result: status={res.status}, avg_price={res.avg_price:.4f}")
print(f"  limitation_notes: {res.limitation_notes}")

# Check stop/target collision
exit_price, exit_reason = execution.check_stop_target_collision(
    sig.side, res.avg_price, float(sig.stop), float(sig.target), h, l
)
print(f"Collision check: exit_price={exit_price}, exit_reason={exit_reason}")

# Now check what happens in the while loop
# The backtest walks bars, and at each bar it:
# 1. Generates signal
# 2. Executes market order
# 3. Checks stop/target collision
# 4. Calculates PnL

# Let me trace the first iteration of the backtest while loop
print(f"\n--- Tracing backtest iteration ---")
ctx.set_bar(0)  # bar 0 of the backtest window
sig2 = None
try:
    sig2 = s.generate_from(ctx)
except Exception as e:
    print(f"  Strategy generate error: {e}")

print(f"  Signal from ctx: {sig2}")

if sig2 is not None:
    entry_raw = float(close[0])
    print(f"  entry_raw={entry_raw:.2f}")
    
    # The execution model call in the backtest
    res2 = execution.execute_market_order(
        Order(
            id='test2', symbol='T', market='', side=sig2.side,
            qty=1.0, order_type=OrderType.MARKET,
            price=entry_raw, stop=float(sig2.stop), target=float(sig2.target),
            strategy=s.name,
        ),
        reference_price=entry_raw,
        signal_time=df_backtest.index[0],
        high=h,
        low=l,
    )
    print(f"  Execution result: status={res2.status}, avg_price={res2.avg_price:.4f}")
    print(f"  limitation_notes: {res2.limitation_notes}")
    
    stop = float(sig2.stop) if sig2.stop else 0.0
    target = float(sig2.target) if sig2.target else 0.0
    
    exit_price2, exit_reason2 = execution.check_stop_target_collision(
        sig2.side, res2.avg_price, stop, target, h, l
    )
    print(f"  Collision check: exit_price={exit_price2}, exit_reason={exit_reason2}")
    
    if exit_price2 is not None:
        print(f"  Would exit at {exit_price2:.2f} with reason {exit_reason2}")
        # Calculate PnL
        if sig2.side == Side.LONG:
            if exit_reason2 == ExitReason.STOPLOSS.value:
                pnl = exit_price2 - res2.avg_price
            else:
                pnl = exit_price2 - res2.avg_price
        else:
            if exit_reason2 == ExitReason.STOPLOSS.value:
                pnl = res2.avg_price - exit_price2
            else:
                pnl = res2.avg_price - exit_price2
        print(f"  PnL would be: {pnl:.2f}")