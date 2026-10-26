"""Realistic order-fill execution model.

Models bid/ask spread, slippage, brokerage, exchange fees, taxes,
latency, partial fills, liquidity constraints, and gap risk.
Distinguishes market, limit, stop, and stop-limit orders.

All monetary assumptions are configurable and labeled as assumptions
when direct market data (bid/ask) is unavailable.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
import random

from trader.core.models import Order, Position, Side, OrderStatus, ExitReason


class FillModel(str, Enum):
    """Fill assumption models when bid/ask data is unavailable."""
    CONSERVATIVE = "conservative"  # Always assume worst-case fill within spread
    MODERATE = "moderate"          # Assume mid-spread fill
    OPTIMISTIC = "optimistic"      # Assume best-case fill (not recommended for production)


@dataclass
class FillAssumptions:
    """Configurable execution assumptions sourced from settings.yaml."""

    # Spread assumptions (basis points of price) when bid/ask unavailable.
    # Labeled assumption: we do not have L2 order book data; this is a
    # conservative default that can be overridden per-market.
    spread_bps: float = 2.0

    # Fill model selection when bid/ask is unavailable
    fill_model: FillModel = FillModel.CONSERVATIVE

    # Slippage (basis points) for market orders; on top of spread
    slippage_bps: float = 5.0

    # Latency (seconds) between signal generation and order submission
    latency_seconds: float = 1.0

    # Fill delay (seconds) between order submission and fill event
    fill_delay_seconds: float = 0.5

    # Partial fill model
    partial_fill_probability: float = 0.0  # 0% by default (conservative)
    min_fill_ratio: float = 1.0  # Minimum fraction if partial fill occurs

    # Gap risk: if price gaps past stop, fill at worst of stop or gap price
    gap_fill_at_worst: bool = True

    # Brokerage configuration dict (from settings.yaml paper.brokerage)
    brokerage_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FillEvent:
    """A single fill event recorded during execution."""
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    timestamp: datetime
    reason: str  # "market", "limit", "stop", "target", "partial", "gap"
    slippage_bps: float
    charges: float
    is_partial: bool = False
    fill_ratio: float = 1.0


@dataclass
class ExecutionResult:
    """Result of executing an order."""
    order_id: str
    symbol: str
    side: Side
    qty_requested: float
    qty_filled: float
    avg_price: float
    total_charges: float
    timestamp: datetime
    status: OrderStatus
    fills: List[FillEvent]
    reject_reason: Optional[str] = None
    limitation_notes: List[str] = field(default_factory=list)


class ExecutionModel:
    """
    Realistic order-fill execution model.

    Separates signal time, submission time, fill time, and mark-to-market time.
    Models spread, slippage, brokerage, taxes, latency, partial fills, and gap risk.
    All assumptions are configurable and labeled when data is unavailable.
    """

    def __init__(self, assumptions: Optional[FillAssumptions] = None,
                 market_cfg: Optional[Dict[str, Any]] = None):
        self.assumptions = assumptions or FillAssumptions()
        self.market_cfg = market_cfg or {}
        self._fill_history: List[FillEvent] = []

    # ------------------------------------------------------------------
    # Spread estimation (labeled assumption when bid/ask unavailable)
    # ------------------------------------------------------------------

    def _spread_estimate(self, price: float) -> Tuple[float, float, float]:
        """Estimate bid/ask/spread when direct quote data is unavailable.

        Returns (bid, ask, mid) based on configurable spread assumption.
        The spread assumption is labeled in limitation_notes so the caller
        knows it is not from real L2 data.
        """
        spread_decimal = self.assumptions.spread_bps / 10000.0
        half_spread = price * spread_decimal / 2.0
        mid = price
        bid = round(mid - half_spread, 4)
        ask = round(mid + half_spread, 4)
        return bid, ask, mid

    # ------------------------------------------------------------------
    # Brokerage charges (from settings.yaml paper.brokerage config)
    # ------------------------------------------------------------------

    def _brokerage_charges(self, market: str, side: Side, qty: float, price: float) -> float:
        """Calculate brokerage + exchange fees + taxes using settings config."""
        cfg = self.assumptions.brokerage_config
        market_cfg = cfg.get(market, {})

        brokerage_flat = float(market_cfg.get("brokerage_flat", 0))
        stt_pct = float(market_cfg.get("stt_pct", 0)) / 100.0
        exch_pct = float(market_cfg.get("exchange_pct", 0)) / 100.0
        gst_pct = float(market_cfg.get("gst_pct", 0)) / 100.0

        turnover = qty * price
        stt = turnover * stt_pct
        exch = turnover * exch_pct
        gst = brokerage_flat * gst_pct

        return brokerage_flat + stt + exch + gst

    # ------------------------------------------------------------------
    # Slippage application
    # ------------------------------------------------------------------

    def _slippage_price(self, price: float, side: Side, is_buy: bool) -> float:
        """Apply configurable slippage (bps) to a price."""
        bps = self.assumptions.slippage_bps / 10000.0
        if is_buy:
            return round(price * (1 + bps), 4)
        return round(price * (1 - bps), 4)

    # ------------------------------------------------------------------
    # Order execution: market order
    # ------------------------------------------------------------------

    def execute_market_order(self, order: Order,
                             reference_price: float,
                             signal_time: datetime,
                             submission_time: Optional[datetime] = None,
                             fill_time: Optional[datetime] = None,
                             high: Optional[float] = None,
                             low: Optional[float] = None) -> ExecutionResult:
        """
        Execute a market order with realistic fill modeling.

        The order fills at the ask (buyer) or bid (seller) based on the
        configurable spread assumption, then slippage is applied on top.
        All assumptions are recorded in limitation_notes.
        """
        subm = submission_time or signal_time + timedelta(
            seconds=self.assumptions.latency_seconds)
        fill_t = fill_time or subm + timedelta(
            seconds=self.assumptions.fill_delay_seconds)

        bid, ask, mid = self._spread_estimate(reference_price)
        is_buy = order.side == Side.LONG

        # Conservative fill: buyer pays ask, seller hits bid
        if self.assumptions.fill_model == FillModel.CONSERVATIVE:
            fill_price = ask if is_buy else bid
        elif self.assumptions.fill_model == FillModel.MODERATE:
            fill_price = mid
        else:  # OPTIMISTIC
            fill_price = bid if is_buy else ask

        # Apply slippage on top of the spread fill
        fill_price = self._slippage_price(fill_price, order.side, is_buy)

        # Partial fill model
        qty = order.qty
        is_partial = False
        fill_ratio = 1.0
        if self.assumptions.partial_fill_probability > 0.0:
            if random.random() < self.assumptions.partial_fill_probability:
                fill_ratio = self.assumptions.min_fill_ratio
                qty = order.qty * fill_ratio
                is_partial = True

        charges = self._brokerage_charges(order.market, order.side, qty, fill_price)

        fill = FillEvent(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=fill_price,
            timestamp=fill_t,
            reason="market",
            slippage_bps=self.assumptions.slippage_bps,
            charges=charges,
            is_partial=is_partial,
            fill_ratio=fill_ratio,
        )
        self._fill_history.append(fill)

        notes: List[str] = []
        if self.assumptions.fill_model == FillModel.CONSERVATIVE:
            notes.append(
                "conservative fill assumption: buyer pays ask, seller hits bid"
            )
        if self.assumptions.spread_bps > 0:
            notes.append(
                f"spread assumed {self.assumptions.spread_bps} bps "
                "(not from L2 order book data)"
            )
        if self.assumptions.slippage_bps > 0:
            notes.append(
                f"slippage assumed {self.assumptions.slippage_bps} bps "
                "(configurable, not from direct market data)"
            )
        if self.assumptions.partial_fill_probability > 0:
            notes.append(
                f"partial fill possible: prob={self.assumptions.partial_fill_probability}, "
                f"min_ratio={self.assumptions.min_fill_ratio}"
            )

        return ExecutionResult(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty_requested=order.qty,
            qty_filled=qty,
            avg_price=fill_price,
            total_charges=charges,
            timestamp=fill_t,
            status=OrderStatus.FILLED,
            fills=[fill],
            limitation_notes=notes,
        )

    # ------------------------------------------------------------------
    # Order execution: limit order
    # ------------------------------------------------------------------

    def execute_limit_order(self, order: Order,
                            reference_price: float,
                            signal_time: datetime,
                            high: float, low: float,
                            submission_time: Optional[datetime] = None,
                            fill_time: Optional[datetime] = None) -> ExecutionResult:
        """
        Execute a limit order.

        For BUY limit: fills if low <= limit_price (price touched the limit)
        For SELL limit: fills if high >= limit_price

        Conservative assumption: fills at the limit price, not better.
        If the limit would be touched within the same candle as a stop,
        the worse outcome is assumed (handled at caller level).
        """
        subm = submission_time or signal_time + timedelta(
            seconds=self.assumptions.latency_seconds)
        fill_t = fill_time or subm + timedelta(
            seconds=self.assumptions.fill_delay_seconds)

        limit_price = order.price if order.price is not None else reference_price
        is_buy = order.side == Side.LONG

        # Check if limit would be touched
        fill_price: Optional[float] = None
        if is_buy:
            if low <= limit_price:
                fill_price = limit_price  # fill at limit price
        else:
            if high >= limit_price:
                fill_price = limit_price

        if fill_price is None:
            return ExecutionResult(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                qty_requested=order.qty,
                qty_filled=0.0,
                avg_price=0.0,
                total_charges=0.0,
                timestamp=fill_t,
                status=OrderStatus.PENDING,
                fills=[],
                limitation_notes=["limit order not filled - price did not touch limit"],
            )

        # Apply minimal slippage for limit orders
        fill_price = self._slippage_price(fill_price, order.side, is_buy)

        qty = order.qty
        charges = self._brokerage_charges(order.market, order.side, qty, fill_price)

        fill = FillEvent(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=fill_price,
            timestamp=fill_t,
            reason="limit",
            slippage_bps=self.assumptions.slippage_bps / 2.0,
            charges=charges,
        )
        self._fill_history.append(fill)

        return ExecutionResult(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty_requested=order.qty,
            qty_filled=qty,
            avg_price=fill_price,
            total_charges=charges,
            timestamp=fill_t,
            status=OrderStatus.FILLED,
            fills=[fill],
        )

    # ------------------------------------------------------------------
    # Order execution: stop order
    # ------------------------------------------------------------------

    def execute_stop_order(self, order: Order,
                           reference_price: float,
                           signal_time: datetime,
                           high: float, low: float,
                           submission_time: Optional[datetime] = None,
                           fill_time: Optional[datetime] = None) -> ExecutionResult:
        """
        Execute a stop order.

        Converts to market when the stop price is touched.

        For BUY stop: triggers when high >= stop_price
        For SELL stop: triggers when low <= stop_price

        Gap risk model: if price gaps past the stop, the fill occurs at
        the worst of (stop_price, gap_price) to model adverse fill.
        """
        subm = submission_time or signal_time + timedelta(
            seconds=self.assumptions.latency_seconds)
        fill_t = fill_time or subm + timedelta(
            seconds=self.assumptions.fill_delay_seconds)

        stop_price = order.stop if order.stop is not None else reference_price
        is_buy = order.side == Side.LONG

        # Check if stop is triggered
        triggered = False
        if is_buy:
            if high >= stop_price:
                triggered = True
        else:
            if low <= stop_price:
                triggered = True

        if not triggered:
            return ExecutionResult(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                qty_requested=order.qty,
                qty_filled=0.0,
                avg_price=0.0,
                total_charges=0.0,
                timestamp=fill_t,
                status=OrderStatus.PENDING,
                fills=[],
                limitation_notes=["stop order not triggered - price did not touch stop"],
            )

        # Once triggered, it becomes a market order
        # Apply gap risk conservatively
        bid, ask, mid = self._spread_estimate(reference_price)

        if is_buy:
            # BUY stop triggered: price gapped up.
            # Conservative: fill at max(ask, stop_price, gap_price=high)
            gap_price = high  # worst case: opened above stop
            fill_price = max(ask, stop_price)
            if self.assumptions.gap_fill_at_worst:
                fill_price = max(fill_price, gap_price)
        else:
            # SELL stop triggered: price gapped down.
            # Conservative: fill at min(bid, stop_price, gap_price=low)
            gap_price = low
            fill_price = min(bid, stop_price)
            if self.assumptions.gap_fill_at_worst:
                fill_price = min(fill_price, gap_price)

        # Apply slippage on top
        fill_price = self._slippage_price(fill_price, order.side, is_buy)

        qty = order.qty
        charges = self._brokerage_charges(order.market, order.side, qty, fill_price)

        fill = FillEvent(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=fill_price,
            timestamp=fill_t,
            reason="stop",
            slippage_bps=self.assumptions.slippage_bps,
            charges=charges,
        )
        self._fill_history.append(fill)

        notes: List[str] = []
        if self.assumptions.gap_fill_at_worst:
            notes.append(
                "gap risk: filled at worst of stop or gap price"
            )
        if self.assumptions.spread_bps > 0:
            notes.append(
                f"spread assumed {self.assumptions.spread_bps} bps "
                "(not from L2 order book data)"
            )
        if self.assumptions.slippage_bps > 0:
            notes.append(
                f"slippage assumed {self.assumptions.slippage_bps} bps "
                "(configurable, not from direct market data)"
            )

        return ExecutionResult(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty_requested=order.qty,
            qty_filled=qty,
            avg_price=fill_price,
            total_charges=charges,
            timestamp=fill_t,
            status=OrderStatus.FILLED,
            fills=[fill],
            limitation_notes=notes,
        )

    # ------------------------------------------------------------------
    # Order execution: stop-limit order
    # ------------------------------------------------------------------

    def execute_stop_limit_order(self, order: Order,
                                 reference_price: float,
                                 signal_time: datetime,
                                 high: float, low: float,
                                 submission_time: Optional[datetime] = None,
                                 fill_time: Optional[datetime] = None) -> ExecutionResult:
        """
        Execute a stop-limit order.

        First, the stop price is checked. If touched, it becomes a limit
        order that will only fill at (or better than) the limit price.

        For BUY stop-limit: stop triggers when high >= stop_price,
        then limit order fills at limit_price or lower.

        For SELL stop-limit: stop triggers when low <= stop_price,
        then limit order fills at limit_price or higher.

        Conservative: if stop triggers but limit cannot be filled, order
        remains unfilled (PENDING) rather than executing at a worse price.
        """
        subm = submission_time or signal_time + timedelta(
            seconds=self.assumptions.latency_seconds)
        fill_t = fill_time or subm + timedelta(
            seconds=self.assumptions.fill_delay_seconds)

        stop_price = order.stop if order.stop is not None else reference_price
        limit_price = order.price if order.price is not None else reference_price
        is_buy = order.side == Side.LONG

        # Step 1: check if stop is triggered
        triggered = False
        if is_buy:
            if high >= stop_price:
                triggered = True
        else:
            if low <= stop_price:
                triggered = True

        if not triggered:
            return ExecutionResult(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                qty_requested=order.qty,
                qty_filled=0.0,
                avg_price=0.0,
                total_charges=0.0,
                timestamp=fill_t,
                status=OrderStatus.PENDING,
                fills=[],
                limitation_notes=["stop-limit stop price not touched"],
            )

        # Step 2: stop triggered - now check if limit order fills
        fill_price: Optional[float] = None
        if is_buy:
            # BUY stop-limit: after stop triggers, limit buys at limit_price or lower
            # Conservative fill: only if low <= limit_price
            if low <= limit_price:
                fill_price = limit_price
        else:
            # SELL stop-limit: after stop triggers, limit sells at limit_price or higher
            if high >= limit_price:
                fill_price = limit_price

        if fill_price is None:
            # Stop triggered but limit cannot be filled at acceptable price
            return ExecutionResult(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                qty_requested=order.qty,
                qty_filled=0.0,
                avg_price=0.0,
                total_charges=0.0,
                timestamp=fill_t,
                status=OrderStatus.PENDING,
                fills=[],
                limitation_notes=["stop triggered but limit price not touchable"],
            )

        # Apply minimal slippage for limit-side fill
        fill_price = self._slippage_price(fill_price, order.side, is_buy)

        qty = order.qty
        charges = self._brokerage_charges(order.market, order.side, qty, fill_price)

        fill = FillEvent(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=fill_price,
            timestamp=fill_t,
            reason="stop-limit",
            slippage_bps=self.assumptions.slippage_bps / 2.0,
            charges=charges,
        )
        self._fill_history.append(fill)

        notes: List[str] = []
        if self.assumptions.spread_bps > 0:
            notes.append(
                f"spread assumed {self.assumptions.spread_bps} bps "
                "(not from L2 order book data)"
            )
        if self.assumptions.slippage_bps > 0:
            notes.append(
                f"slippage assumed {self.assumptions.slippage_bps} bps "
                "(configurable, not from direct market data)"
            )

        return ExecutionResult(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty_requested=order.qty,
            qty_filled=qty,
            avg_price=fill_price,
            total_charges=charges,
            timestamp=fill_t,
            status=OrderStatus.FILLED,
            fills=[fill],
        )

    # ------------------------------------------------------------------
    # Conservative stop/target collision modeling
    # ------------------------------------------------------------------

    def check_stop_target_collision(self, side: Side, entry: float, stop: float, target: float,
                                     high: float, low: float) -> Tuple[Optional[float], Optional[str]]:
        """
        Conservatively model stop/target collision within a single candle.

        If both stop and target would be hit within the same bar, assume the
        worse outcome for the trader (conservative).

        Returns (exit_price, exit_reason) or (None, None) if neither hit.
        """
        hit_stop = False
        hit_target = False

        if side == Side.LONG:
            if low <= stop:
                hit_stop = True
            if high >= target:
                hit_target = True
        else:  # SHORT
            if high >= stop:
                hit_stop = True
            if low <= target:
                hit_target = True

        if hit_stop and hit_target:
            # Conservative: assume stop is hit first (worse outcome)
            if side == Side.LONG:
                return stop, ExitReason.STOPLOSS.value
            else:
                return stop, ExitReason.STOPLOSS.value

        if hit_stop:
            return stop, ExitReason.STOPLOSS.value

        if hit_target:
            return target, ExitReason.TARGET.value

        return None, None

    # ------------------------------------------------------------------
    # Mark-to-market valuation (separate from execution/fill)
    # ------------------------------------------------------------------

    def mark_to_market(self, position: Position, current_price: float,
                       timestamp: datetime) -> Dict[str, Any]:
        """
        Mark-to-market a position at a given timestamp.

        This is purely valuation - does not execute or change position state.
        Separated from fill time to keep signal time, submission time,
        fill time, and mark-to-market time distinct.
        """
        if position.side == Side.LONG:
            unrealized = (current_price - position.entry) * position.qty
        else:
            unrealized = (position.entry - current_price) * position.qty

        return {
            "symbol": position.symbol,
            "market": position.market,
            "side": position.side,
            "qty": position.qty,
            "entry": position.entry,
            "current": current_price,
            "unrealized_pnl": unrealized,
            "timestamp": timestamp,
        }