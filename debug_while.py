"""Debug the backtest while loop."""
import sys
sys.path.insert(0, r"C:\IDTS")

import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from trader.core.selector import Selector
from trader.core.models import Signal, Side, OrderType, Order, OrderStatus, ExitReason
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

# Test backtest at i=68 directly
i_orig = 68
df_slice = df.iloc[:i_orig]
ctx = Ctx(df_slice, "T", "India")
sig = s.generate_from(ctx)
print(f"Signal: side={sig.side}, entry={sig.entry:.2f}, stop={sig.stop:.2f}, target={sig.target:.2f}")

# Now manually replicate the backtest loop
df_backtest = sel._backtest_window(df_slice)
df_backtest = df_backtest.iloc[:-1]  # drop live bar
print(f"Backtest window: {len(df_backtest)} bars")

high = df_backtest["high"].to_numpy(dtype=float)
low = df_backtest["low"].to_numpy(dtype=float)
close = df_backtest["close"].to_numpy(dtype=float)

n = len(df_backtest)
print(f"n = {n}")

# Build execution model
paper_cfg = sel.cfg.get("paper", {}) or {}
slip_bps = float(paper_cfg.get("slippage_bps", 5))
spread_bps = float(paper_cfg.get("spread_bps", 2))
partial_prob = float(paper_cfg.get("partial_fill_probability", 0))
gap_fill_worst = bool(paper_cfg.get("gap_fill_at_worst", True))

assumptions = FillAssumptions(
    spread_bps=spread_bps,
    fill_model=FillModel.CONSERVATIVE,
    slippage_bps=slip_bps,
    latency_seconds=0.0,
    fill_delay_seconds=0.0,
    partial_fill_probability=partial_prob,
    min_fill_ratio=1.0,
    gap_fill_at_worst=gap_fill_worst,
    brokerage_config=paper_cfg.get("brokerage", {}),
)
execution = ExecutionModel(assumptions=assumptions, market_cfg={})

# Track trade execution
trades_executed = 0
i = 0
while i < n - 1:
    ctx.set_bar(i)
    sig = None
    try:
        sig = s.generate_from(ctx)
    except Exception as e:
        sig = None
        print(f"  i={i}: strategy exception: {e}")
    
    if sig is None:
        print(f"  i={i}: NO SIGNAL")
        i += 1
        continue
    
    entry_raw = float(close[i])
    print(f"  i={i}: SIGNAL! side={sig.side}, entry_raw={entry_raw:.2f}, stop={sig.stop:.2f}, target={sig.target:.2f}")
    print(f"    high[{i}]={high[i]:.2f}, low[{i}]={low[i]:.2f}")
    
    # Execute market order
    try:
        res = execution.execute_market_order(
            Order(
                id='test', symbol='T', market='', side=sig.side,
                qty=1.0, order_type=OrderType.MARKET,
                price=entry_raw, stop=float(sig.stop), target=float(sig.target),
                strategy=s.name,
            ),
            reference_price=entry_raw,
            signal_time=df_backtest.index[i],
            high=float(high[i]),
            low=float(low[i]),
        )
        print(f"    Execution: status={res.status}, avg_price={res.avg_price:.4f}")
        print(f"    Limitation notes: {res.limitation_notes}")
    except Exception as e:
        print(f"    Execution ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        i += 1
        continue
    
    # Check if filled
    if res.status != OrderStatus.FILLED or res.qty_filled == 0:
        print(f"    NOT FILLED, skipping")
        i += 1
        continue
    
    trades_executed += 1
    print(f"    -> TRADE EXECUTED! (trade #{trades_executed})")
    
    stop = float(sig.stop) if sig.stop else 0.0
    target = float(sig.target) if sig.target else 0.0
    
    # Check stop/target collision
    exit_price, exit_reason = execution.check_stop_target_collision(
        sig.side, res.avg_price, stop, target, float(high[i]), float(low[i])
    )
    print(f"    Collision: exit_price={exit_price}, exit_reason={exit_reason}")
    
    if exit_price is None:
        print(f"    No exit (neither stop nor target hit)")
        i += 1
        continue
    
    # Determine PnL
    if sig.side == Side.LONG:
        if exit_reason == ExitReason.STOPLOSS.value:
            pnl = exit_price - res.avg_price
        else:
            pnl = exit_price - res.avg_price
    else:
        if exit_reason == ExitReason.STOPLOSS.value:
            pnl = res.avg_price - exit_price
        else:
            pnl = res.avg_price - exit_price
    
    risk = abs(res.avg_price - stop) if stop > 0 else abs(res.avg_price - entry_raw)
    print(f"    PnL={pnl:.2f}, risk={risk:.2f}")
    
    i += 1

print(f"\nTotal trades executed: {trades_executed}")