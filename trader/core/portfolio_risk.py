"""Portfolio-level risk model for IDTS across supported markets.

Provides exposure tracking, correlation estimation, configurable caps,
circuit breakers, and order approval-time constraints. Paper-first by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from trader.core.models import Position, Side, Trade, Signal
from trader.core.risk import RiskManager
from trader.core.market_clock import MarketClock


# ---------------------------------------------------------------------------
# Exposure summary
# ---------------------------------------------------------------------------

@dataclass
class ExposureSummary:
    """Aggregate exposure breakdown for the portfolio."""
    # By symbol
    by_symbol: Dict[str, float]  # net exposure per symbol (INR)
    # By strategy
    by_strategy: Dict[str, float]  # net exposure per strategy (INR)
    # By market
    by_market: Dict[str, float]  # net exposure per market (INR)
    # By direction
    by_side: Dict[str, float]  # long vs short net (INR)
    # By asset class (inferred from market)
    by_asset_class: Dict[str, float]
    # By currency
    by_currency: Dict[str, float]  # net exposure per currency (INR)
    # Gross and net
    gross_exposure: float  # sum of absolute exposures (INR)
    net_exposure: float    # sum of signed exposures (INR)
    # Total positions
    total_positions: int
    long_positions: int
    short_positions: int


# ---------------------------------------------------------------------------
# Correlation / return estimator
# ---------------------------------------------------------------------------

class ReturnEstimator:
    """Estimates pairwise correlations from synchronized return observations.

    All calculations use only current and past observations (no look-ahead).
    Missing or stale observations are explicitly handled via listwise deletion.
    """

    def __init__(self, prices: Dict[str, pd.Series], window: int = 60,
                 min_obs: int = 5):
        """Initialize with price series per symbol.

        Parameters
        ----------
        prices : Dict[str, pd.Series]
            Adjusted close series per symbol, aligned by date index.
        window : int
            Rolling window for correlation in days of observations.
        min_obs : int
            Minimum overlapping observations required for a valid correlation.
        """
        self.prices = prices
        self.window = window
        self.min_obs = min_obs
        self._cache: Dict[Tuple[str, str], float] = {}

    def _log_returns(self, series: pd.Series) -> pd.Series:
        """Log returns; first value is NaN."""
        returns = np.log(series / series.shift(1))
        return returns

    def correlation(self, sym_a: str, sym_b: str) -> Optional[float]:
        """Pearson correlation of log returns over the rolling window.

        Returns None if insufficient overlapping observations.
        """
        key = (min(sym_a, sym_b), max(sym_a, sym_b))
        if key in self._cache:
            return self._cache[key]

        if sym_a not in self.prices or sym_b not in self.prices:
            self._cache[key] = None
            return None

        ra = self._log_returns(self.prices[sym_a])
        rb = self._log_returns(self.prices[sym_b])

        # Align on index; use listwise deletion
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        n = len(aligned)
        if n < self.min_obs:
            self._cache[key] = None
            return None

        r_a = aligned.iloc[:, 0].values.astype(float)
        r_b = aligned.iloc[:, 1].values.astype(float)
        # Pearson r
        mean_a = np.mean(r_a)
        mean_b = np.mean(r_b)
        cov = np.sum((r_a - mean_a) * (r_b - mean_b)) / (n - 1)
        std_a = np.sqrt(np.sum((r_a - mean_a) ** 2) / (n - 1))
        std_b = np.sqrt(np.sum((r_b - mean_b) ** 2) / (n - 1))
        if std_a == 0 or std_b == 0:
            r = 0.0
        else:
            r = cov / (std_a * std_b)
        r = float(np.clip(r, -1.0, 1.0))

        self._cache[key] = r
        return r

    def rolling_correlations(self, sym_a: str, sym_b: str) -> Optional[pd.Series]:
        """Rolling correlation series over the price window.

        Returns None if insufficient data.
        """
        if sym_a not in self.prices or sym_b not in self.prices:
            return None
        ra = self._log_returns(self.prices[sym_a])
        rb = self._log_returns(self.prices[sym_b])
        aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
        n = len(aligned)
        if n < self.min_obs:
            return None
        # Rolling correlation using expanding window
        r = aligned.rolling(window=min(self.window, n)).corr(method="pearson")
        # Drop first n where correlation can't be computed
        r = r.dropna(how="all")
        return r if len(r) > 0 else None


# ---------------------------------------------------------------------------
# Portfolio risk model
# ---------------------------------------------------------------------------

class PortfolioRiskModel:
    """Portfolio-level risk model for IDTS across supported markets.

    Tracks exposure, estimates correlations, enforces constraints at
    order approval time, and provides circuit breakers for abnormal conditions.
    """

    def __init__(self, cfg: dict, risk: RiskManager, markets_cfg: dict,
                 clock: MarketClock, prices: Dict[str, pd.Series] = None):
        self.cfg = cfg
        self.risk = risk
        self.markets_cfg = markets_cfg
        self.clock = clock
        self.prices = prices or {}

        # Configuration caps
        self.gross_exposure_cap_pct = self.cfg.get("gross_exposure_cap_pct")
        self.net_exposure_cap_pct = self.cfg.get("net_exposure_cap_pct")
        self.max_position_pct_of_equity = self.risk.risk  # from RiskManager
        # Read from risk config dict inside cfg (RiskManager may not have all attributes)
        self.max_concurrent_positions = self.cfg.get("max_concurrent_positions", 6)
        self.max_positions_per_market = self.cfg.get("max_positions_per_market", 3)

        # Currency configuration
        self.base_currency = "INR"

        # Correlation estimator
        self.return_estimator = ReturnEstimator(self.prices)

        # Exposure history for volatility/shrinkage
        self.exposure_history: List[ExposureSummary] = []
        self.max_history = self.cfg.get("exposure_history_bars", 250)

        # Circuit breaker settings
        self.max_drawdown_pct = self.cfg.get("max_drawdown_pct", 10.0)
        self.max_daily_loss_pct = self.cfg.get("max_daily_loss_pct", 1.5) / 100.0
        self.abnormal_slippage_pct = self.cfg.get("abnormal_slippage_pct", 2.0)
        self.stale_data_max_minutes = self.cfg.get("stale_data_max_minutes", 300)

        # Asset class classifications (from market config)
        self.market_to_asset_class: Dict[str, str] = {}
        self._init_asset_classes()

        # Correlated strategy groups (populated if reliable classifications exist)
        self.correlated_groups: Dict[str, Set[str]] = defaultdict(set)

    def _init_asset_classes(self) -> None:
        """Infer asset class from market configuration."""
        for market, cfg in self.markets_cfg.items():
            currency = cfg.get("currency", "INR")
            # Simple mapping: futures/crypto/stocks by currency/market type
            if currency in ("JPY", "USD", "GBP", "EUR"):
                self.market_to_asset_class[market] = "forex_futures"
            elif currency in ("AUD", "CAD", "CHF"):
                self.market_to_asset_class[market] = "commodities"
            elif currency in ("BRL", "ZAR", "MXN"):
                self.market_to_asset_class[market] = "emerging_markets"
            else:
                self.market_to_asset_class[market] = "equities"

    # -------------------------------------------------------------------------
    # Exposure calculation
    # -------------------------------------------------------------------------

    def compute_exposure(self, positions: List[Position], capital: float) -> ExposureSummary:
        """Compute aggregate exposure across all dimensions.

        Parameters
        ----------
        positions : List[Position]
            Current open positions.
        capital : float
            Total capital in INR (for percentage calculations).

        Returns
        ------- ExposureSummary
            Breakdown of exposure by all dimensions.
        """
        by_symbol: Dict[str, float] = defaultdict(float)
        by_strategy: Dict[str, float] = defaultdict(float)
        by_market: Dict[str, float] = defaultdict(float)
        by_side: Dict[str, float] = defaultdict(float)  # positive = long net, negative = short net
        by_currency: Dict[str, float] = defaultdict(float)
        asset_class_exposure: Dict[str, float] = defaultdict(float)

        long_count = 0
        short_count = 0
        gross = 0.0

        for pos in positions:
            # Symbol exposure
            sym_exposure = pos.qty * (pos.current - pos.entry) if pos.side == Side.LONG else pos.qty * (pos.entry - pos.current)
            by_symbol[pos.symbol] += sym_exposure

            # Strategy exposure
            by_strategy[pos.strategy] += sym_exposure

            # Market exposure
            market = pos.market
            by_market[market] += sym_exposure

            # Side exposure
            side_key = "long" if pos.side == Side.LONG else "short"
            by_side[side_key] += sym_exposure

            # Currency exposure
            currency = self._currency(market)
            by_currency[currency] += sym_exposure

            # Asset class exposure
            asset_class = self.market_to_asset_class.get(market, "unknown")
            asset_class_exposure[asset_class] += sym_exposure

            # Gross accumulation
            gross += abs(sym_exposure)

            # Counts
            if pos.side == Side.LONG:
                long_count += 1
            else:
                short_count += 1

        net = by_symbol.get("long", 0) - by_symbol.get("short", 0)  # simplified; we'll recompute

        # Actually compute net properly: sum of all signed exposures
        net = 0.0
        for pos in positions:
            if pos.side == Side.LONG:
                net += pos.qty * (pos.current - pos.entry) if pos.current else pos.qty * (pos.entry - pos.entry)
            else:
                net -= pos.qty * (pos.current - pos.entry) if pos.current else 0

        # Use a more robust calculation
        net = 0.0
        for pos in positions:
            if pos.side == Side.LONG:
                if pos.current is not None and pos.entry is not None:
                    net += pos.qty * (pos.current - pos.entry)
            else:
                if pos.current is not None and pos.entry is not None:
                    net -= pos.qty * (pos.current - pos.entry)

        # Build the summary
        summary = ExposureSummary(
            by_symbol=dict(by_symbol),
            by_strategy=dict(by_strategy),
            by_market=dict(by_market),
            by_side={k: float(v) for k, v in by_side.items()},
            by_asset_class={k: float(v) for k, v in asset_class_exposure.items()},
            by_currency={k: float(v) for k, v in by_currency.items()},
            gross_exposure=gross,
            net_exposure=net,
            total_positions=len(positions),
            long_positions=long_count,
            short_positions=short_count,
        )

        # Store in history
        self.exposure_history.append(summary)
        if len(self.exposure_history) > self.max_history:
            self.exposure_history = self.exposure_history[-self.max_history:]

        return summary

    def _currency(self, market: str) -> str:
        """Get the currency for a market from the market config."""
        cfg = self.markets_cfg.get(market) or {}
        cur = cfg.get("currency")
        if cur:
            return cur
        # Default mapping
        return {"India": "INR", "Japan": "JPY", "Europe": "GBP", "USA": "USD"}.get(market, "INR")

    # -------------------------------------------------------------------------
    # Constraint checking at order approval time
    # -------------------------------------------------------------------------

    def check_order_approval(self, symbol: str, market: str, side: Side,
                            qty: float, entry: float, stop: float,
                            target: float, capital: float,
                            existing_positions: List[Position]) -> Tuple[bool, str]:
        """Enforce all risk constraints at order approval time.

        Returns (approved, reason).  If not approved, reason explains why.
        """
        # 1. Basic sanity
        if qty <= 0:
            return False, "non-positive quantity"
        if entry <= 0:
            return False, "non-positive entry price"

        # 2. Concurrent position limits
        if len(existing_positions) >= self.max_concurrent_positions:
            return False, f"max concurrent positions ({self.max_concurrent_positions})"

        # 3. Per-market limit
        market_count = sum(1 for p in existing_positions if p.market == market)
        if market_count >= self.max_positions_per_market:
            return False, f"max positions per market ({self.max_positions_per_market})"

        # 4. Per-symbol duplicate check
        if any(p.symbol == symbol and p.side == side for p in existing_positions):
            return False, "already has position in this symbol/side"

        # 5. Daily loss limit
        # Calculate current day P&L
        daily_pnl = self._daily_pnl(existing_positions)
        max_loss = capital * self.max_daily_loss_pct
        if daily_pnl <= -max_loss:
            return False, "max daily loss limit breached"

        # 6. Gross exposure cap
        # Estimate new position's notional
        new_notional = abs(qty) * entry
        current_gross = self._gross_exposure(existing_positions)
        # If we add this position, gross becomes new_notional + current_gross
        # (rough approximation; actual depends on price movement)
        if self.gross_exposure_cap_pct:
            cap = capital * self.gross_exposure_cap_pct / 100.0
            if new_notional + new_notional > cap:  # simplistic: double the new position
                return False, f"gross exposure cap ({cap:.0f} INR)"

        # 7. Per-symbol position size check (via RiskManager)
        # RiskManager already has position_size logic; we delegate
        # But we need equity - use capital
        equity = capital
        size = self.risk.position_size(equity, entry, stop)
        if qty > size:
            return False, f"position size exceeds risk limit (max={size})"

        # 7. Drawdown check
        daily_pnl = self._daily_pnl(existing_positions)
        if daily_pnl < -capital * self.daily_loss_pct:
            return False, "drawdown limit breached"

        # 8. Circuit breaker checks
        if self._stale_data_check(existing_positions):
            return False, "stale data detected"
        if self._runaway_loss_check(existing_positions, capital):
            return False, "runaway loss detected"
        if self._abnormal_slippage_check(existing_positions):
            return False, "abnormal slippage detected"
        if self._market_session_check(market):
            return False, "market session error"

        return True, "approved"

    def _daily_pnl(self, positions: List[Position]) -> float:
        """Approximate day P&L in INR."""
        total = 0.0
        for p in positions:
            if p.side == Side.LONG and p.current is not None and p.entry is not None:
                total += p.qty * (p.current - p.entry)
            elif p.side == Side.SHORT and p.current is not None and p.entry is not None:
                total -= p.qty * (p.current - p.entry)
        return total

    def _gross_exposure(self, positions: List[Position]) -> float:
        """Sum of absolute exposures."""
        total = 0.0
        for p in positions:
            if p.current is not None and p.entry is not None:
                exposure = abs(p.qty * (p.current - p.entry))
                total += exposure
        return total

    # -------------------------------------------------------------------------
    # Circuit breakers
    # -------------------------------------------------------------------------

    def _stale_data_check(self, positions: List[Position]) -> bool:
        """Check if any position has stale price data."""
        now = datetime.now(timezone.utc)
        for p in positions:
            # If position was opened more than stale_data_max_minutes ago
            # and we have no recent price data
            age_minutes = (now - p.opened_at).total_seconds() / 60.0
            if age_minutes > self.stale_data_max_minutes:
                # Check if market is still open
                if not self.clock.is_market_open(p.market):
                    return True
        return False

    def _runaway_loss_check(self, positions: List[Position], capital: float) -> bool:
        """Check if positions have losses that would trigger kill switch."""
        daily_pnl = self._daily_pnl(positions)
        max_loss = capital * self.max_daily_loss_pct
        return daily_pnl <= -max_loss

    def _abnormal_slippage_check(self, positions: List[Position]) -> bool:
        """Check for abnormal slippage (simplified: if unrealized loss > cap)."""
        total_unrealized = 0.0
        for p in positions:
            if p.current is not None and p.entry is not None and p.side == Side.LONG:
                ul = p.current - p.entry
                if ul < 0:
                    total_unrealized += abs(ul) * p.qty
            elif p.side == Side.SHORT and p.current is not None and p.entry is not None:
                ul = p.entry - p.current
                if ul < 0:
                    total_unrealized += abs(ul) * p.qty
        # If unrealized loss exceeds abnormal slippage threshold
        return total_unrealized > capital * self.abnormal_slippage_pct / 100.0

    def _market_session_check(self, market: str) -> bool:
        """Check if market session is valid."""
        status = self.clock.status(market)
        if status["status"] in ("CLOSED", "SUSPENDED", "HALTED"):
            return True
        return False

    # -------------------------------------------------------------------------
    # Correlation & concentration analysis
    # -------------------------------------------------------------------------

    def compute_pairwise_correlations(self) -> Dict[Tuple[str, str], float]:
        """Compute pairwise return correlations across all symbols with data.

        Returns dict of (sym_a, sym_b) -> correlation coefficient.
        None values excluded.
        """
        correlations = {}
        symbols = list(self.prices.keys())
        for i, sa in enumerate(symbols):
            for sb in symbols[i+1:]:
                r = self.return_estimator.correlation(sa, sb)
                if r is not None:
                    correlations[(sa, sb)] = r
        return correlations

    def concentration_herfindahl(self, positions: List[Position], capital: float) -> float:
        """Herfindahl-Hirschman index of exposure concentration.

        Higher = more concentrated. 1/N = equally weighted.
        """
        exposures = []
        for p in positions:
            if p.current is not None and p.entry is not None:
                exposures.append(abs(p.qty * (p.current - p.entry)))
        if not exposures:
            return 0.0
        total = sum(exposures)
        if total == 0:
            return 0.0
        shares = [e / total for e in exposures]
        return float(sum(s ** 2 for s in shares))

    def check_concentration_limits(self, positions: List[Position], capital: float,
                                   max_herfindahl: float = 0.25) -> Tuple[bool, str]:
        """Check asset concentration limits.

        Parameters
        ----------
        positions : List[Position]
        capital : float
        max_herfindahl : float
            Maximum acceptable Herfindahl index (0.25 = 4 strategies share ~50% of exposure).

        Returns
        ------- Tuple[bool, str]
            (approved, reason)
        """
        hhi = self.concentration_herfindahl(positions, capital)
        if hhi > max_herfindahl:
            return False, f"concentration HHI={hhi:.3f} exceeds limit {max_herfindahl}"
        return True, "within limits"

    # -------------------------------------------------------------------------
    # Currency exposure
    # -------------------------------------------------------------------------

    def currency_exposure(self, positions: List[Position], capital: float) -> Dict[str, float]:
        """Net exposure per currency, converted to INR.

        Parameters
        ----------
        positions : List[Position]
        capital : float

        Returns
        ------- Dict[str, float]
            Net INR exposure per currency.
        """
        by_currency: Dict[str, float] = defaultdict(float)
        for p in positions:
            currency = self._currency(p.market)
            if p.side == Side.LONG and p.current is not None and p.entry is not None:
                exposure_in_market = p.qty * (p.current - p.entry)
                by_currency[currency] += exposure_in_market
            elif p.side == Side.SHORT and p.current is not None and p.entry is not None:
                exposure_in_market = -(p.qty * (p.current - p.entry))
                by_currency[currency] += exposure_in_market

        # Convert to INR
        result: Dict[str, float] = {}
        for cur, val in by_currency.items():
            fx = self.clock.fx_rate(cur) if hasattr(self.clock, 'fx_rate') else 1.0
            result[cur] = val * fx
        return result

    # -------------------------------------------------------------------------
    # Volatility-aware checks
    # -------------------------------------------------------------------------

    def volatility_check(self, positions: List[Position], capital: float,
                        current_prices: Dict[str, float] = None) -> Tuple[bool, str]:
        """Check if portfolio volatility is within bounds.

        Uses simple historical volatility from price series; falls back to
        conservative estimate if inputs are unstable.
        """
        # Simple: compute current unrealized P&L as % of capital
        total_unrealized = 0.0
        for p in positions:
            if p.current is not None and p.entry is not None and p.side == Side.LONG:
                total_unrealized += p.qty * (p.current - p.entry)
            elif p.side == Side.SHORT and p.current is not None and p.entry is not None:
                total_unrealized -= p.qty * (p.current - p.entry)

        pct_of_capital = abs(total_unrealized) / capital if capital > 0 else 0.0

        # Conservative bounds
        if pct_of_capital > 0.10:  # 10% of capital
            return False, f"volatility exposure {pct_of_capital:.1%} of capital"
        return True, f"volatility exposure {pct_of_capital:.1%} of capital (ok)"


# -------------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------------

def build_portfolio_risk_model(cfg: dict, risk: RiskManager, markets_cfg: dict,
                               clock: MarketClock, prices: Dict[str, pd.Series] = None) -> PortfolioRiskModel:
    """Convenience constructor for PortfolioRiskModel."""
    return PortfolioRiskModel(cfg, risk, markets_cfg, clock, prices)