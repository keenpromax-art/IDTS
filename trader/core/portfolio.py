from __future__ import annotations

from typing import List, Dict, Optional
from datetime import datetime, timezone

from trader.core.models import Position, Trade, Side
from trader.core.risk import RiskManager


class Portfolio:
    def __init__(self, cfg: dict, risk: RiskManager, markets_cfg: dict = None):
        self.cfg = cfg
        self.risk = risk
        self.markets_cfg = markets_cfg or {}
        self.capital: Dict[str, float] = dict(cfg.get("capital", {}))
        self.initial_capital: Dict[str, float] = dict(self.capital)
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.realized_pnl: Dict[str, float] = {m: 0.0 for m in self.capital}
        self.unrealized_pnl: Dict[str, float] = {m: 0.0 for m in self.capital}
        self.fx_rates: Dict[str, float] = {"INR": 1.0, "USD": 83.0, "USDT": 83.0, "JPY": 0.55,
                                          "GBP": 105.0, "HKD": 10.6, "AUD": 54.0,
                                          "KRW": 0.075, "BRL": 15.0}
        # fx rates are X->INR
        # baselines for the *day* P&L, so the daily loss limit counts today only
        self._day_realized: Dict[str, float] = {m: 0.0 for m in self.capital}
        self._day_unrealized: Dict[str, float] = {m: 0.0 for m in self.capital}

    def start_new_day(self):
        """Reset the day baselines - called when the trading day rolls over."""
        self._day_realized = dict(self.realized_pnl)
        self._day_unrealized = dict(self.unrealized_pnl)

    def set_fx(self, rates: Dict[str, float]):
        self.fx_rates.update(rates)

    def total_equity_inr(self) -> float:
        total = 0.0
        for m, cap in self.capital.items():
            # equity = initial capital + realized + unrealized
            eq = self.initial_capital.get(m, cap) + self.realized_pnl.get(m, 0) + self.unrealized_pnl.get(m, 0)
            cur = self.fx_rates.get(self._currency(m), 1.0)
            total += eq * cur
        return total

    def _currency(self, market: str) -> str:
        """The currency a market trades in, from the market config."""
        cfg = self.markets_cfg.get(market) or {}
        cur = cfg.get("currency")
        if cur:
            return cur
        return {"India": "INR", "Japan": "JPY", "Europe": "GBP", "USA": "USD"}.get(market, "INR")

    def day_pnl_inr(self) -> float:
        """P&L since the start of the trading day, converted to INR."""
        total = 0.0
        for m in self.capital:
            total += self.day_pnl_local(m) * self.fx_rates.get(self._currency(m), 1.0)
        return total

    def day_pnl_local(self, market: str) -> float:
        """P&L for one market today, in that market's own currency.

        The daily loss limit is a percentage of the market's own equity, so
        the two sides of that comparison must be in the same currency.
        """
        realized = self.realized_pnl.get(market, 0.0) - self._day_realized.get(market, 0.0)
        unrealized = self.unrealized_pnl.get(market, 0.0) - self._day_unrealized.get(market, 0.0)
        return realized + unrealized

    def day_realized_inr(self) -> float:
        """Realised P&L since the start of the trading day, in INR."""
        total = 0.0
        for m in self.capital:
            realized = self.realized_pnl.get(m, 0.0) - self._day_realized.get(m, 0.0)
            total += realized * self.fx_rates.get(self._currency(m), 1.0)
        return total

    def day_unrealized_inr(self) -> float:
        """Unrealised P&L since the start of the trading day, in INR."""
        total = 0.0
        for m in self.capital:
            unrealized = self.unrealized_pnl.get(m, 0.0) - self._day_unrealized.get(m, 0.0)
            total += unrealized * self.fx_rates.get(self._currency(m), 1.0)
        return total

    def equity_local(self, market: str) -> float:
        """Equity for one market, in that market's own currency."""
        return (self.initial_capital.get(market, 0.0)
                + self.realized_pnl.get(market, 0.0)
                + self.unrealized_pnl.get(market, 0.0))

    def add_position(self, pos: Position):
        self.positions.append(pos)

    def remove_position(self, pos_id: str) -> Optional[Position]:
        for i, p in enumerate(self.positions):
            if p.id == pos_id:
                return self.positions.pop(i)
        return None

    def find_position(self, symbol: str, market: str) -> Optional[Position]:
        for p in self.positions:
            if p.symbol == symbol and p.market == market:
                return p
        return None

    def update_prices(self, price_map: Dict[str, float]):
        for p in self.positions:
            if p.symbol in price_map:
                p.current = price_map[p.symbol]
        # unrealized per market
        for m in self.capital:
            self.unrealized_pnl[m] = 0.0
        for p in self.positions:
            if p.side == Side.LONG:
                upnl = (p.current - p.entry) * p.qty
            else:
                upnl = (p.entry - p.current) * p.qty
            self.unrealized_pnl[p.market] = self.unrealized_pnl.get(p.market, 0) + upnl

    def close_trade(self, pos: Position, exit_price: float, exit_reason, charges_delta: float = 0.0):
        from trader.core.models import Trade, ExitReason
        # idempotency: if this position was already closed, do not double-count
        if any(t.id == pos.id for t in self.trades):
            # return a synthetic trade without mutating state again
            existing = next(t for t in self.trades if t.id == pos.id)
            return existing
        pos.charges += charges_delta
        trade = Trade(
            id=pos.id,
            symbol=pos.symbol,
            market=pos.market,
            side=pos.side,
            strategy=pos.strategy,
            qty=pos.qty,
            entry=pos.entry,
            exit=exit_price,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
            exit_reason=exit_reason,
            charges=pos.charges,
            score_reason=pos.score_reason,
            stop=pos.stop,
        )
        self.trades.insert(0, trade)
        pnl = trade.pnl
        self.realized_pnl[pos.market] = self.realized_pnl.get(pos.market, 0) + pnl
        # update capital
        self.capital[pos.market] = self.capital.get(pos.market, 0) + pnl
        self.risk.record_trade_result(pos.symbol, pnl)
        return trade

    def fx_rate(self, market: str) -> float:
        """Rate to convert a market's currency into INR."""
        return float(self.fx_rates.get(self._currency(market), 1.0))

    def to_inr(self, amount: float, market: str) -> float:
        """Convert an amount in a market's currency into INR."""
        return amount * self.fx_rate(market)

    def stats(self) -> Dict[str, Any]:
        """Session statistics.

        Every total is converted to INR first - adding an AUD profit to an
        HKD loss is meaningless, so the money is normalised before it is
        added up.
        """
        trades = self.trades
        inr = [self.to_inr(t.pnl, t.market) for t in trades]
        wins = [p for p in inr if p > 0]
        losses = [p for p in inr if p <= 0]
        total_pnl = sum(inr)
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        best = max(inr, default=0.0)
        worst = min(inr, default=0.0)
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        r_multiples = [t.r_multiple for t in trades]
        return {
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "gross_win": gross_win,
            "gross_loss": gross_loss,
            "profit_factor": pf,
            "best": best,
            "worst": worst,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy_r": (sum(r_multiples) / len(r_multiples)) if r_multiples else 0.0,
            "currency": "INR",
        }

    def unrealized_inr(self) -> float:
        """Total unrealised P&L across all markets, in INR."""
        total = 0.0
        for p in self.positions:
            if p.side == Side.LONG:
                upnl = (p.current - p.entry) * p.qty
            else:
                upnl = (p.entry - p.current) * p.qty
            total += self.to_inr(upnl, p.market)
        return total

    def strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for t in self.trades:
            s = out.setdefault(t.strategy, {"trades": 0, "wins": 0, "pnl": 0.0})
            s["trades"] += 1
            pnl = self.to_inr(t.pnl, t.market)
            if pnl > 0:
                s["wins"] += 1
            s["pnl"] += pnl
        for s in out.values():
            s["win_rate"] = (s["wins"] / s["trades"] * 100) if s["trades"] else 0.0
        return out

    def per_market_pnl(self) -> Dict[str, float]:
        out = {}
        for m in self.capital:
            out[m] = self.realized_pnl.get(m, 0) + self.unrealized_pnl.get(m, 0)
        return out
