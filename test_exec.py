"""Quick test of execution model."""
import pandas as pd
import numpy as np
from trader.core.models import Signal, Side, OrderType, Order, OrderStatus
from trader.core.execution import ExecutionModel, FillAssumptions, FillModel

# Create execution model with conservative assumptions
assumptions = FillAssumptions(spread_bps=2.0, fill_model=FillModel.CONSERVATIVE,
                               slippage_bps=5.0, latency_seconds=0.0,
                               fill_delay_seconds=0.0, partial_fill_probability=0.0,
                               min_fill_ratio=1.0, gap_fill_at_worst=True,
                               brokerage_config={})
execution = ExecutionModel(assumptions=assumptions, market_cfg={})

# Create a LONG signal
sig = Signal(symbol='T', market='', side=Side.LONG, strategy='test',
             entry=100.0, stop=99.0, target=103.0, confidence=0.8)

# Create order
order = Order(
    id='test', symbol='T', market='', side=Side.LONG,
    qty=1.0, order_type=OrderType.MARKET,
    price=100.0, stop=99.0, target=103.0,
    strategy='test',
)

# Execute market order with high/low for gap check
res = execution.execute_market_order(order, reference_price=100.0, signal_time=pd.Timestamp('2024-01-02 09:15', tz='UTC'),
                                      high=101.0, low=99.0)
print('Fill result:')
print(f'  status: {res.status}')
print(f'  avg_price: {res.avg_price}')
print(f'  fills: {res.fills}')
print(f'  limitation_notes: {res.limitation_notes}')

# Now check stop/target collision
exit_price, exit_reason = execution.check_stop_target_collision(
    Side.LONG, res.avg_price, 99.0, 103.0, 101.0, 99.0
)
print(f'\nCollision check:')
print(f'  exit_price: {exit_price}')
print(f'  exit_reason: {exit_reason}')

# Now test SHORT
print('\n--- SHORT test ---')
sig_s = Signal(symbol='T', market='', side=Side.SHORT, strategy='test',
               entry=100.0, stop=101.0, target=97.0, confidence=0.8)
order_s = Order(
    id='test2', symbol='T', market='', side=Side.SHORT,
    qty=1.0, order_type=OrderType.MARKET,
    price=100.0, stop=101.0, target=97.0,
    strategy='test',
)
res_s = execution.execute_market_order(order_s, reference_price=100.0, signal_time=pd.Timestamp('2024-01-02 09:15', tz='UTC'),
                                        high=101.0, low=99.0)
print(f'Fill result: status={res_s.status}, avg_price={res_s.avg_price}')

exit_price_s, exit_reason_s = execution.check_stop_target_collision(
    Side.SHORT, res_s.avg_price, 101.0, 97.0, 101.0, 99.0
)
print(f'Collision check: exit_price={exit_price_s}, exit_reason={exit_reason_s}')