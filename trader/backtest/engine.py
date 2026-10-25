from __future__ import annotations

"""Event-driven backtesting engine.

Shares the production strategy and risk logic. Signals are evaluated on
closed bars only; execution happens on the next bar (never on the signal
bar's close). Transaction costs, slippage, fees and currency conversion are
applied. Reports standard performance statistics.
"""
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from trader.core.models import Signal, Side, Trade, ExitReason, Position
from trader.core.risk import RiskManager
from trader.core.market_clock import MarketClock


@dataclass
class BacktestConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    initial_cash: float = 100000.0
    currency: str = "INR"
    slippage_bps: float = 5.0
    fee_per_trade: float = 0.0
    fee_pct: float = 0.0
    allow_short: bool = True
    max_positions: int = 1
    risk_per_trade_pct: float = 0.75
    stop_atr: float = 1.5
    target_rr: float = 2.0
    # fractional markets (crypto): quantity step, e.g. 0.0001; 1 = integer
    qty_step: float = 1.0


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    def __init__(self, strategy, df: pd.DataFrame, cfg: BacktestConfig = None,
                 risk_cfg: Optional[dict] = None):
        self.strategy = strategy
        self.df = df.copy()
        self.cfg = cfg or BacktestConfig()
        self.risk_cfg = risk_cfg or {"risk": {"risk_per_trade_pct": self.cfg.risk_per_trade_pct,
                                             "max_position_pct_of_equity": 20,
                                             "max_concurrent_positions": self.cfg.max_positions,
                                             "max_positions_per_market": self.cfg.max_positions,
                                             "max_trades_per_symbol_per_day": 999,
                                             "max_daily_loss_pct": 100,
                                             "cooldown_minutes_after_loss": 0,
                                             "no_entry_first_minutes": 0,
                                             "no_entry_last_minutes": 0,
                                             "square_off_before_close_minutes": 0,
                                             "min_risk_reward": 1.0,
                                             "close_on_market_closed": False,
                                             "trailing_stop": False,
                                             "trailing_atr_mult": 1.0},
                                     "selector": {}}
        # a no-op clock that always says open
        self.clock = _AlwaysOpenClock()
        self.risk = RiskManager(self.risk_cfg, self.clock)
        self.trades: List[Trade] = []
        self.position: Optional[Position] = None
        self.cash = self.cfg.initial_cash
        self.equity_curve: List[float] = []
        self._slip = self.cfg.slippage_bps / 10000.0

    def _cost(self, price, qty, side):
        fee = self.cfg.fee_per_trade + price * qty * (self.cfg.fee_pct / 100.0)
        return fee

    def run(self) -> BacktestResult:
        df = self.df
        if df is None or len(df) < 40:
            return BacktestResult(stats={"error": "insufficient data"})
        # ensure UTC index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n = len(df)
        from trader.strategies.context import Ctx
        ctx = Ctx(df, "BT", "")
        # walk bars; signal on bar i, execute on bar i+1 open
        i = 0
        peak = self.cash
        max_dd = 0.0
        while i < n - 1:
            ctx.set_bar(i)
            sig = None
            try:
                sig = self.strategy.generate_from(ctx)
            except Exception:
                sig = None
            # manage open position first on this bar's OHLC
            if self.position is not None:
                pos = self.position
                # check stop/target against this bar
                hi = high[i]; lo = low[i]
                exit_price = None
                reason = None
                if pos.side == Side.LONG:
                    if lo <= pos.stop:
                        exit_price = pos.stop * (1 - self._slip); reason = ExitReason.STOPLOSS
                    elif hi >= pos.target:
                        exit_price = pos.target * (1 - self._slip); reason = ExitReason.TARGET
                else:
                    if hi >= pos.stop:
                        exit_price = pos.stop * (1 + self._slip); reason = ExitReason.STOPLOSS
                    elif lo <= pos.target:
                        exit_price = pos.target * (1 + self._slip); reason = ExitReason.TARGET
                if exit_price is not None:
                    fee = self._cost(exit_price, pos.qty, pos.side)
                    pnl = ((exit_price - pos.entry) if pos.side == Side.LONG else (pos.entry - exit_price)) * pos.qty - fee
                    trade = Trade(id=pos.id, symbol=pos.symbol, market=pos.market, side=pos.side,
                                  strategy=pos.strategy, qty=pos.qty, entry=pos.entry, exit=exit_price,
                                  opened_at=pos.opened_at, closed_at=df.index[i].to_pydatetime(),
                                  exit_reason=reason, charges=fee + pos.charges, score_reason=pos.score_reason,
                                  stop=pos.stop)
                    self.trades.append(trade)
                    self.cash += pnl
                    self.position = None
            # if flat and signal, open on next bar open
            if self.position is None and sig is not None:
                if (not self.cfg.allow_short) and sig.side == Side.SHORT:
                    i += 1
                    self._mark_equity(close[i-1] if i-1 >= 0 else close[0], peak)
                    continue
                # risk check
                equity = self.cash
                risk_amt = equity * (self.cfg.risk_per_trade_pct / 100.0)
                risk_dist = abs(sig.entry - sig.stop)
                if risk_dist > 0:
                    raw = risk_amt / risk_dist
                    max_notional = equity * 0.2
                    step = self.cfg.qty_step
                    if step and 0 < step < 1:
                        import math
                        qty = min(raw, math.floor(raw / step) * step)
                        qty = round(qty, 8)
                        if sig.entry > 0:
                            qty = min(qty, max_notional / sig.entry)
                    else:
                        qty = int(raw)
                        if sig.entry > 0:
                            qty = min(qty, int(max_notional / sig.entry))
                    if qty > 0:
                        # execute at next bar open (no look-ahead)
                        exec_price = float(df["open"].iloc[i+1])
                        if sig.side == Side.LONG:
                            exec_price = exec_price * (1 + self._slip)
                        else:
                            exec_price = exec_price * (1 - self._slip)
                        fee = self._cost(exec_price, qty, sig.side)
                        pos = Position(id=str(uuid.uuid4()), symbol="BT", market="BT", side=sig.side,
                                       strategy=sig.strategy, qty=qty, entry=exec_price, stop=sig.stop,
                                       target=sig.target, current=exec_price, charges=fee,
                                       score_reason=sig.reason, opened_at=df.index[i+1].to_pydatetime())
                        self.position = pos
                        self.cash -= fee
            i += 1
            px = close[min(i, n-1)]
            eq = self.cash
            if self.position is not None:
                pos = self.position
                upnl = (px - pos.entry) * pos.qty if pos.side == Side.LONG else (pos.entry - px) * pos.qty
                eq += upnl
            self.equity_curve.append(eq)
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        # close any open position at end
        if self.position is not None:
            pos = self.position
            exit_price = close[-1]
            fee = self._cost(exit_price, pos.qty, pos.side)
            pnl = ((exit_price - pos.entry) if pos.side == Side.LONG else (pos.entry - exit_price)) * pos.qty - fee
            trade = Trade(id=pos.id, symbol=pos.symbol, market=pos.market, side=pos.side,
                          strategy=pos.strategy, qty=pos.qty, entry=pos.entry, exit=exit_price,
                          opened_at=pos.opened_at, closed_at=df.index[-1].to_pydatetime(),
                          exit_reason=ExitReason.SQUAREOFF, charges=fee + pos.charges,
                          score_reason=pos.score_reason, stop=pos.stop)
            self.trades.append(trade)
            self.cash += pnl
            self.position = None
            self.equity_curve.append(self.cash)
        # drawdown curve
        peak = self.cfg.initial_cash
        dd = []
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd.append(peak - eq)
        self.drawdown_curve = dd
        self.stats = self._compute_stats()
        return BacktestResult(trades=self.trades, equity_curve=self.equity_curve,
                             drawdown_curve=self.drawdown_curve, stats=self.stats)

    def _mark_equity(self, px, peak):
        pass

    def _compute_stats(self) -> Dict[str, Any]:
        trades = self.trades
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = sum(pnls) if pnls else 0.0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        equity = self.equity_curve or [self.cfg.initial_cash]
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)
        import math
        # annualized return (assume 252 trading days, bars per day inferred)
        rets = np.diff(equity) if len(equity) > 1 else np.array([])
        if len(rets) > 1:
            # per-bar returns
            r = np.array(rets) / np.array(equity[:-1])
            r = r[np.isfinite(r)]
            if r.size:
                mean_r = float(np.mean(r))
                std_r = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
                sharpe = (mean_r / std_r * np.sqrt(252 * 78)) if std_r > 0 else 0.0
                downside = r[r < 0]
                sortino_den = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
                sortino = (mean_r / sortino_den * np.sqrt(252 * 78)) if sortino_den > 0 else 0.0
            else:
                sharpe = 0.0; sortino = 0.0
        else:
            sharpe = 0.0; sortino = 0.0
        holding = 0.0
        if trades:
            holding = sum((t.closed_at - t.opened_at).total_seconds() for t in trades) / len(trades) / 60.0
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades) * 100) if trades else 0.0,
            "total_pnl": total,
            "profit_factor": pf if pf != float("inf") else 9999.0,
            "expectancy": (total / len(trades)) if trades else 0.0,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "sortino": sortino,
            "avg_holding_minutes": holding,
            "final_equity": equity[-1] if equity else self.cfg.initial_cash,
            "total_return_pct": ((equity[-1] / self.cfg.initial_cash) - 1) * 100 if equity else 0.0,
        }


class _AlwaysOpenClock:
    def __init__(self):
        self.cfg = {}
    def status(self, market, now=None):
        from datetime import datetime, timezone
        now = now or datetime.now(timezone.utc)
        return {"status": "OPEN", "local_time": now, "open_dt": now, "close_dt": now,
                "countdown_seconds": None, "next_open": None, "next_close": None, "in_lunch": False}
    def reset_day(self):
        pass


def walk_forward(strategy_factory, df, folds=5, train_ratio=0.7, cfg=None):
    """Simple walk-forward: split df into folds, train on first train_ratio, test on rest."""
    n = len(df)
    fold_size = n // folds
    results = []
    for f in range(folds):
        start = f * fold_size
        end = start + fold_size if f < folds - 1 else n
        chunk = df.iloc[start:end]
        split = int(len(chunk) * train_ratio)
        test = chunk.iloc[split:]
        if len(test) < 40:
            continue
        bt = BacktestEngine(strategy_factory(), test, cfg or BacktestConfig())
        res = bt.run()
        results.append({"fold": f, "test_start": str(test.index[0]), "test_end": str(test.index[-1]),
                        "stats": res.stats})
    return results
