from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import pandas as pd

from trader.core.market_clock import MarketClock
from trader.core.risk import RiskManager
from trader.core.portfolio import Portfolio
from trader.core.selector import Selector
from trader.core.models import Signal, Side, Order, OrderStatus, ExitReason, OrderType, Position
from trader.data.factory import build_provider
from trader.brokers.paper_broker import PaperBroker
from trader.brokers.angel_broker import AngelBroker
from trader.strategies import ALL_STRATEGIES, ALL_RULES
from trader.core.storage import Storage
from trader.core.prices import PriceTicker

log = logging.getLogger(__name__)


def drop_unfinished(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # assume last bar may be in-progress; drop it
    if len(df) > 1:
        return df.iloc[:-1].copy()
    return df


class Engine:
    def __init__(self, cfg: dict, markets_cfg: dict, watchlists: dict, mode: str,
                 live_confirmed: bool = False):
        self.cfg = cfg
        # manual on/off switches: a market (or crypto as a whole) can be
        # disabled from settings.yaml or from a market's own `enabled` flag
        crypto_on = bool(cfg.get("crypto_enabled",
                      cfg.get("crypto",
                      cfg.get("crypto_on",
                      cfg.get("enable_crypto", True)))))
        filtered_markets = {}
        for m, mc in (markets_cfg or {}).items():
            mc = dict(mc or {})
            if mc.get("crypto") and not crypto_on:
                continue
            if mc.get("enabled", True) is False:
                continue
            filtered_markets[m] = mc
        self.markets_cfg = filtered_markets
        self.watchlists = {m: watchlists.get(m, []) for m in self.markets_cfg}
        self.crypto_enabled = crypto_on
        self.disabled_markets = set()
        self.mode = mode
        self.live_confirmed = live_confirmed
        self.clock = MarketClock(markets_cfg)
        self.risk = RiskManager(cfg, self.clock)
        self.portfolio = Portfolio(cfg, self.risk, markets_cfg)
        self.selector = Selector(cfg)
        self.strategies = self._build_strategies()
        self.selector.set_strategies(self.strategies)
        # close positions in markets we are not scanning (session over / lunch)
        self._close_on_market_closed = bool(
            (cfg.get("risk", {}) or {}).get("close_on_market_closed", True))
        self.providers: Dict[str, Any] = {}
        for m in markets_cfg:
            self.providers[m] = build_provider(m)
        # broker
        if mode == "live" and live_confirmed:
            from trader.data.angel_provider import AngelProvider
            ap = AngelProvider()
            if not getattr(ap, "available", False):
                raise RuntimeError("Angel One unavailable: " + str(getattr(ap, "error", "unknown")))
            self.broker = AngelBroker(ap)
            if not self.broker.is_connected():
                raise RuntimeError("Angel broker did not connect")
            if not getattr(self.broker, "capabilities", None) or not self.broker.capabilities.live_allowed:
                raise RuntimeError("broker does not permit live execution")
            self.broker_name = "angel"
        else:
            self.broker = PaperBroker(cfg, cfg.get("paper", {}).get("slippage_bps", 5))
            self.broker_name = "paper"
        self.storage = Storage(cfg.get("db_path", "data/trader.db"))
        self.cycle = 0
        self.paused = False
        self.running = True
        # set by the dashboard (hotkey `f` / command `scan`) to force an
        # immediate scan instead of waiting for the interval to elapse
        self.scan_now = threading.Event()
        self.logs: List[str] = []
        self.last_scan: Dict[str, Any] = {}  # market -> list of scan rows
        self.equity_history: List[float] = []
        self.last_scan_elapsed: float = 0.0
        self.scan_progress: str = ""
        self.scan_market: str = ""
        # read-only scan telemetry for the dashboard's analytics/diagnostics
        self.last_cycle_stats: Dict[str, int] = {}
        self.last_skipped: Dict[str, int] = {}
        self.skip_counts: Dict[str, int] = {}
        self._opened_cycle: int = 0
        # live quotes
        self.ticker = PriceTicker(max_workers=cfg.get("price_refresh_max_workers", 12))
        self.live_quotes = bool(cfg.get("live_quotes", True))
        self.price_refresh_seconds = float(cfg.get("price_refresh_seconds", 1.0))
        # how often a whole live market should be re-quoted, end to end
        self.fast_scan_seconds = float(cfg.get("fast_scan_seconds", 5.0))
        self._fast_queue: List[str] = []
        self._fast_cursor: int = 0
        self.last_quote_at: float = 0.0
        self._price_thread: Optional[threading.Thread] = None
        # one lock so the scan thread and the quote thread never interleave
        # while touching the portfolio
        self.lock = threading.RLock()
        self._closed_logged: set = set()
        self._kill_flattened: bool = False
        self._session_date: str = self._today()
        self._skipped: Dict[str, int] = {}
        self.next_scan_at: float = 0.0
        self.latest_prices: Dict[str, float] = {}
        # rolling per-symbol price history for the dashboard's detail view
        self.price_history: Dict[str, Any] = {}
        self._price_hist_len = 180
        self.current_scanning: Optional[str] = None
        self.fx_last_update = 0.0
        # restore positions from db if same session
        try:
            self._restore()
        except Exception as e:
            log.warning("restore failed: %s", e)
        # a position whose market is not trading must not be carried over
        try:
            self._close_stale_positions()
        except Exception as e:
            log.warning("stale position cleanup failed: %s", e)
        # reconcile local state against the broker on startup
        try:
            broker_positions = self.broker.reconcile(self._price_fn) if hasattr(self.broker, "reconcile") else None
            if isinstance(broker_positions, list) and broker_positions:
                # adopt any broker-side positions we do not have locally
                local_ids = {p.id for p in self.portfolio.positions}
                for bp in broker_positions:
                    if bp.id not in local_ids:
                        self.portfolio.add_position(bp)
                        self.storage.save_position(bp)
        except Exception as e:
            log.warning("broker reconcile failed: %s", e)
        self.log(f"Engine started — {mode} mode, {sum(len(v) for v in watchlists.values())} symbols, broker {self.broker_name}")

    def set_market_enabled(self, market: str, enabled: bool):
        """Manually turn a market (e.g. Crypto) on or off at runtime."""
        if market not in self.markets_cfg and not enabled:
            # nothing to do; market already absent
            pass
        # update config flag
        if market == "Crypto" or (self.markets_cfg.get(market, {}).get("crypto")):
            self.cfg["crypto_enabled"] = bool(enabled)
            self.crypto_enabled = bool(enabled)
        if not enabled:
            # close positions in that market
            try:
                self._close_market_positions(market, __import__("trader.core.models", fromlist=["ExitReason"]).ExitReason.SQUAREOFF, note="disabled")
            except Exception:
                pass
            self.markets_cfg.pop(market, None)
            self.watchlists.pop(market, None)
        self.log(f"{market} {'enabled' if enabled else 'disabled'}")

    def _build_strategies(self):
        """Assemble the strategy list: single setups + the enabled rules.

        settings.yaml controls which of each group is active:

            strategies:
              enabled: [ema_cross, ema_20_50, ...]   # omit for "all"
              disabled: [some_name]
            rules:
              enabled: [rule_trend_stack, ...]        # omit for "all"
              disabled: []
        """
        from trader.strategies import ALL_STRATEGIES, ALL_RULES

        s_cfg = self.cfg.get("strategies", {}) or {}
        enabled = s_cfg.get("enabled")
        disabled = set(s_cfg.get("disabled", []) or [])
        if enabled:
            wanted = set(enabled)
            singles = [c for c in ALL_STRATEGIES if c.name in wanted and c.name not in disabled]
        else:
            singles = [c for c in ALL_STRATEGIES if c.name not in disabled]

        r_cfg = self.cfg.get("rules", {}) or {}
        r_enabled = r_cfg.get("enabled")
        r_disabled = set(r_cfg.get("disabled", []) or [])
        if r_enabled:
            wanted = set(r_enabled)
            rules = [r for r in ALL_RULES if r.name in wanted and r.name not in r_disabled]
        else:
            rules = [r for r in ALL_RULES if r.name not in r_disabled]

        # per-rule overrides from settings.yaml
        overrides = self.cfg.get("rule_params", {}) or {}
        for r in rules:
            o = overrides.get(r.name)
            if not o:
                continue
            if "min_confidence" in o:
                r.min_confidence = float(o["min_confidence"])
            if "stop_atr" in o:
                r.stop_atr = float(o["stop_atr"])
            if "target_rr" in o:
                r.target_rr = float(o["target_rr"])
            if "min_optional" in o:
                r.min_optional = int(o["min_optional"])

        # the user's own strategies: declarative YAML + Python plugins
        custom = list(self._load_custom_strategies())

        # `strategies.enabled` filters the built-in setups. Custom setups are
        # controlled separately so adding one never needs an edit elsewhere.
        c_cfg = self.cfg.get("custom_strategies", {}) or {}
        c_disabled = set(c_cfg.get("disabled", []) or [])
        c_enabled = c_cfg.get("enabled")
        if c_enabled is not None:
            custom = [c for c in custom if c.name in set(c_enabled)]
        custom = [c for c in custom if c.name not in disabled]
        custom = [c for c in custom if c.name not in c_disabled]

        return [c() for c in singles] + rules + custom

    def _load_custom_strategies(self):
        """Load the user's own setups: YAML definitions and Python plugins."""
        from trader.core.custom import load_declarative, load_plugins
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = []
        yaml_path = self.cfg.get("custom_strategies_file") or os.path.join(
            base, "config", "custom_strategies.yaml")
        try:
            out.extend(load_declarative(yaml_path))
        except Exception as e:
            log.warning("custom strategies (yaml) failed to load: %s", e)
            self.log(f"custom strategy file error: {e}", "error")
        plugin_dir = self.cfg.get("custom_strategies_dir") or os.path.join(base, "custom")
        try:
            out.extend(load_plugins(plugin_dir))
        except Exception as e:
            log.warning("custom strategies (plugins) failed to load: %s", e)
            self.log(f"custom strategy plugin error: {e}", "error")
        return out

    @staticmethod
    def _today() -> str:
        """Today's date in IST - the reference day for daily limits."""
        from zoneinfo import ZoneInfo
        return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()

    def _roll_over_day(self):
        """Reset the daily limits when the trading day changes.

        The daily loss limit, the kill switch and the per-symbol trade
        counters are all *daily*. Without this the system would stay
        halted for good after one bad session.
        """
        today = self._today()
        if today == self._session_date:
            return
        self._session_date = today
        self._kill_flattened = False
        self._closed_logged.clear()
        was_killed = self.risk.kill_switch
        self.risk.reset_day()
        self.portfolio.start_new_day()
        self.log(f"new trading day {today} — daily limits reset"
                 + (" (kill switch released)" if was_killed else ""), "info")

    def log(self, msg: str, level: str = ""):
        from zoneinfo import ZoneInfo
        ts = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S")
        if level:
            msg = f"{level.upper()} {msg}"
        entry = f"[{ts}] {msg}"
        self.logs.insert(0, entry)
        self.logs = self.logs[:50]
        log.info(msg)

    def _restore(self):
        positions = self.storage.load_open_positions()
        for p in positions:
            self.portfolio.positions.append(p)
            if hasattr(self.broker, "positions") and isinstance(self.broker.positions, dict):
                self.broker.positions[p.id] = p

    def _trade_candidates(self, market: str, candidates: list):
        """Open the best-ranked candidates in a market.

        The per-symbol selector picks the best setup for each symbol, but
        "a setup exists for this symbol" is not a reason to trade it. The
        candidates are ranked across the whole market and only the strongest
        few are taken, so a busy market does not turn into a spray of
        low-conviction trades.
        """
        if not candidates or self.paused:
            return
        sel_cfg = self.cfg.get("selector", {}) or {}
        min_score = float(sel_cfg.get("min_market_score", 0.0))
        max_trades = int(sel_cfg.get("max_trades_per_market_per_cycle", 2))

        ranked = sorted(candidates, key=lambda c: -(c["score"] or 0.0))
        if min_score > 0:
            ranked = [c for c in ranked if (c["score"] or 0.0) >= min_score]
        if max_trades > 0:
            ranked = ranked[:max_trades]

        for cand in ranked:
            self._open_candidate(market, cand)

    def _open_candidate(self, market: str, cand: dict):
        """Size, risk-check and open one candidate position."""
        best_sig = cand.get("sig")
        if best_sig is None:
            return
        symbol = cand["symbol"]
        price = cand["price"]
        best_name = cand["name"]
        best_label = cand["label"]
        best_score = cand["score"]
        regime = cand["regime"]
        ok, reason = self.risk.can_open(
            market, symbol, best_sig.side, self.portfolio.positions,
            self.portfolio.equity_local(market),
            self.portfolio.day_pnl_local(market))
        if not ok:
            # a blocked entry is normal, not an event - count it and report
            # once per cycle instead of flooding the log
            self._skipped[reason] = self._skipped.get(reason, 0) + 1
            return
        equity = self.portfolio.equity_local(market)
        # crypto trades coin fractions: the market config carries qty_step
        step = (self.markets_cfg.get(market) or {}).get("qty_step")
        qty = self.risk.position_size(equity, best_sig.entry, best_sig.stop, step)
        if qty <= 0:
            return
        risk = abs(best_sig.entry - best_sig.stop)
        reward = abs(best_sig.target - best_sig.entry)
        if risk <= 0 or reward / risk < self.cfg["risk"]["min_risk_reward"]:
            return
        # the reason explains *why* this setup fired: score + regime + the
        # conditions that aligned
        parts = [best_label or best_name]
        if best_score is not None:
            parts.append(f"score {best_score:.2f}")
        parts.append(regime.label)
        if best_sig.reason:
            parts.append(best_sig.reason)
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            market=market,
            side=best_sig.side,
            qty=qty,
            order_type=OrderType.MARKET,
            price=best_sig.entry,
            stop=best_sig.stop,
            target=best_sig.target,
            strategy=best_sig.strategy,
            score_reason=" | ".join(parts),
        )
        filled = self.broker.place_order(order, lambda s: price)
        if filled.status != OrderStatus.FILLED:
            self.log(f"order rejected {symbol}", "error")
            return
        pos = None
        if isinstance(self.broker.positions, dict):
            pos = self.broker.positions.get(order.id)
        if pos is None:
            pos = Position(
                id=order.id, symbol=symbol, market=market, side=best_sig.side,
                strategy=best_sig.strategy, qty=qty, entry=filled.filled_price or best_sig.entry,
                stop=best_sig.stop, target=best_sig.target, current=price,
                charges=filled.charges, score_reason=order.score_reason,
            )
            if isinstance(self.broker.positions, dict):
                self.broker.positions[order.id] = pos
        self.portfolio.add_position(pos)
        self._opened_cycle += 1
        self.storage.save_position(pos)
        self.storage.save_signal(symbol, market, best_sig)
        self.storage.save_order(filled)
        self.log(f"OPEN {market} {symbol} {best_sig.side.value} via {best_sig.strategy} "
                 f"qty {qty} (score {best_score:.2f})", "order")

    def _close_market_positions(self, market: str, reason: ExitReason = ExitReason.SQUAREOFF,
                                note: str = "") -> int:
        """Close every open position belonging to a market.

        Called when a market stops being scannable (session over, or a
        lunch break) and for positions restored from a previous session.
        The exit price is the last price we saw for the symbol - there is no
        live market to quote against once the session is over.
        """
        positions_here = [p for p in self.portfolio.positions if p.market == market]
        if not positions_here:
            return 0
        closed = 0
        for p in positions_here:
            price = p.current
            pp = self._price_fn(p.symbol)
            if pp is not None:
                price = pp
            try:
                fill, ch = self.broker.exit_position(p, price, reason)
                trade = self.portfolio.close_trade(p, fill, reason, ch)
                self.portfolio.remove_position(p.id)
                if hasattr(self.broker, "positions") and isinstance(self.broker.positions, dict):
                    self.broker.positions.pop(p.id, None)
                self.storage.save_trade(trade)
                self.storage.delete_position(p.id)
                closed += 1
                self.log(f"closed {market} {p.symbol} @ {fill:.2f} ({note or reason.value})", "order")
            except Exception as e:
                self.log(f"close error {p.symbol}: {e}", "error")
        return closed

    def _close_stale_positions(self):
        """Close restored positions whose market is no longer trading.

        A position carried over from a previous session must never be traded
        again - it is closed at the last known price as soon as we notice.
        A market that is merely PRE-CLOSE or on a lunch break counts as not
        tradable: we are not watching it, so we cannot protect it.
        """
        for market in self.markets_cfg:
            if market in self.disabled_markets:
                continue
            st = self.clock.status(market)
            status = st["status"]
            if status == "OPEN":
                continue
            held = [p for p in self.portfolio.positions if p.market == market]
            if not held:
                continue
            note = {"PRE-CLOSE": "pre-close", "LUNCH": "lunch break"}.get(
                status, "market closed")
            n = self._close_market_positions(market, ExitReason.SQUAREOFF, note=note)
            if n:
                self.log(f"{market} {note} - closed {n} restored position(s)", "order")

    def _price_fn(self, symbol: str) -> Optional[float]:
        if symbol in self.latest_prices:
            return self.latest_prices[symbol]
        for m, rows in self.last_scan.items():
            for r in rows:
                if r.get("symbol") == symbol:
                    return r.get("price")
        return None

    def _note_price(self, symbol: str, price: Optional[float]):
        """Record a rolling price history tick for the dashboard detail view."""
        if symbol is None or price is None:
            return
        try:
            from collections import deque
            h = self.price_history.get(symbol)
            if h is None:
                h = deque(maxlen=self._price_hist_len)
                self.price_history[symbol] = h
            h.append(float(price))
        except Exception:
            pass

    def _scan_metrics(self, df: pd.DataFrame, price: float, regime) -> Dict[str, Any]:
        """Indicator snapshot for the dashboard only.

        Nothing here feeds a trading decision - the strategies have already
        run and the selector has already scored. Computing it after the fact
        keeps the display rich without touching the signal path.
        """
        out: Dict[str, Any] = {
            "atr_pct": None, "adx": float(getattr(regime, "adx", 0.0) or 0.0),
            "rsi": None, "rvol": None, "vwap": None, "vwap_dist": None,
            "spark": [], "chart": [], "ema20": None, "ema50": None,
            "macd_line": None, "macd_signal": None, "macd_hist": None,
            "bb_mid": None, "bb_upper": None, "bb_lower": None,
            "bb_pos": None, "stoch_k": None, "stoch_d": None,
            "willr": None, "cci": None, "obv": None,
            "supertrend": None,
            "high_pct": None, "low_pct": None,
            "close_pct_from_open": None,
            "volume_ratio": None,
            "price_vs_vwap": None,
            "days_since_open": None,
        }
        try:
            import numpy as np
            from trader.core.np_indicators import (
                atr_np, rsi_np, vwap_np, _ema_np_fast,
                macd_np, bollinger_np, stochastic_np, williams_r_np,
                cci_np, obv_np, supertrend_np,
            )
            close = df["close"].to_numpy(dtype=float)
            high = df["high"].to_numpy(dtype=float)
            low = df["low"].to_numpy(dtype=float)
            vol = df["volume"].to_numpy(dtype=float) if "volume" in df else None
            out["spark"] = [float(x) for x in close[-20:]]
            out["chart"] = [float(x) for x in close[-60:]]
            if len(close) >= 15 and price:
                atr = float(atr_np(high, low, close, 14)[-1])
                if atr == atr:
                    out["atr_pct"] = atr / price * 100.0
                rsi = float(rsi_np(close, 14)[-1])
                if rsi == rsi:
                    out["rsi"] = rsi
            if len(close) >= 50:
                e20 = float(_ema_np_fast(close, 20)[-1])
                e50 = float(_ema_np_fast(close, 50)[-1])
                out["ema20"] = e20 if e20 == e20 else None
                out["ema50"] = e50 if e50 == e50 else None
            if vol is not None and len(vol) >= 21:
                base = float(vol[-21:-1].mean())
                if base > 0:
                    out["rvol"] = float(vol[-1] / base)
            if vol is not None:
                n = min(len(close), 110)
                vw = float(vwap_np(high[-n:], low[-n:], close[-n:], vol[-n:])[-1])
                if vw == vw and vw:
                    out["vwap"] = vw
                    if price:
                        out["vwap_dist"] = (price - vw) / vw * 100.0
            # MACD (12,26,9)
            if len(close) >= 30:
                macd_line, macd_signal, macd_hist = macd_np(close, 12, 26, 9)
                out["macd_line"] = float(macd_line[-1]) if macd_line[-1] == macd_line[-1] else None
                out["macd_signal"] = float(macd_signal[-1]) if macd_signal[-1] == macd_signal[-1] else None
                out["macd_hist"] = float(macd_hist[-1]) if macd_hist[-1] == macd_hist[-1] else None
            # Bollinger Bands (20,2.0)
            if len(close) >= 25:
                bb_mid, bb_upper, bb_lower = bollinger_np(close, 20, 2.0)
                out["bb_mid"] = float(bb_mid[-1]) if bb_mid[-1] == bb_mid[-1] else None
                out["bb_upper"] = float(bb_upper[-1]) if bb_upper[-1] == bb_upper[-1] else None
                out["bb_lower"] = float(bb_lower[-1]) if bb_lower[-1] == bb_lower[-1] else None
                # %B position: (price - lower) / (upper - lower)
                if out["bb_upper"] is not None and out["bb_lower"] is not None and out["bb_mid"] is not None:
                    bb_range = out["bb_upper"] - out["bb_lower"]
                    if bb_range != 0:
                        out["bb_pos"] = (price - out["bb_lower"]) / bb_range
                    else:
                        out["bb_pos"] = None
            # Stochastic (14,3)
            if len(close) >= 20:
                stoch_k, stoch_d = stochastic_np(high, low, close, 14, 3)
                out["stoch_k"] = float(stoch_k[-1]) if stoch_k[-1] == stoch_k[-1] else None
                out["stoch_d"] = float(stoch_d[-1]) if stoch_d[-1] == stoch_d[-1] else None
            # Williams %R (14)
            if len(close) >= 20:
                willr = williams_r_np(high, low, close, 14)
                out["willr"] = float(willr[-1]) if willr[-1] == willr[-1] else None
            # CCI (20)
            if len(close) >= 25:
                cci_val = cci_np(high, low, close, 20)
                out["cci"] = float(cci_val[-1]) if cci_val[-1] == cci_val[-1] else None
            # OBV
            if vol is not None and len(close) >= 2:
                obv = obv_np(close, vol)
                out["obv"] = float(obv[-1]) if obv[-1] == obv[-1] else None
            # Supertrend (10,3.0) - needs ATR values
            if len(close) >= 20:
                atr_vals = atr_np(high, low, close, 10)
                st, _ = supertrend_np(high, low, close, atr_vals, 10, 3.0)
                out["supertrend"] = float(st[-1]) if st[-1] == st[-1] else None
            # High/Low percent from first close
            if len(close) >= 2:
                first_close = close[0]
                if first_close != 0:
                    out["high_pct"] = (float(high[-1]) - first_close) / first_close * 100.0
                    out["low_pct"] = (float(low[-1]) - first_close) / first_close * 100.0
            # Close % from open (first bar of session)
            if len(close) >= 2:
                out["close_pct_from_open"] = (float(close[-1]) - float(close[0])) / abs(float(close[0])) * 100.0 if float(close[0]) != 0 else None
            # Volume ratio (current vs 20-bar avg)
            if vol is not None and len(vol) >= 21:
                base = float(vol[-21:-1].mean())
                if base > 0:
                    out["volume_ratio"] = float(vol[-1]) / base
            # Price vs VWAP distance
            if out["vwap"] is not None and price:
                out["price_vs_vwap"] = (price - out["vwap"]) / out["vwap"] * 100.0
        except Exception:
            pass
        return out

    def _gate_reason(self, symbol: str, live_signals: dict) -> str:
        """The first quality gate a live setup failed, for the scanner's Why.

        The selector already made the decision; this only explains it. It
        reads the cached backtest and never mutates anything.
        """
        try:
            cache = self.selector._cache.get(symbol)
            if not cache:
                return ""
            scores = cache[1]
            cand = None
            for name, sig in live_signals.items():
                if sig is None or name not in scores:
                    continue
                if cand is None or sig.confidence > cand[1].confidence:
                    cand = (name, sig)
            if cand is None:
                return ""
            name, sig = cand
            sc, bs, _reg, _sig = scores[name]
            sel = self.selector
            if not bs.eligible:
                if bs.trades < sel.min_trades:
                    return f"needs {sel.min_trades} trades ({bs.trades})"
                if bs.expectancy_r < sel.min_expectancy_r:
                    return f"exp {bs.expectancy_r:+.2f}R"
                if bs.profit_factor < sel.min_profit_factor:
                    return f"PF {bs.profit_factor:.2f} gate"
                if bs.win_rate < sel.min_win_rate:
                    return f"WR {bs.win_rate * 100:.0f}% gate"
                return "backtest not eligible"
            if sc < sel.min_score:
                return f"score {sc:.2f} < {sel.min_score:.2f}"
            if sig.confidence < sel.min_signal_confidence:
                return f"conf {sig.confidence:.2f} < {sel.min_signal_confidence:.2f}"
        except Exception:
            pass
        return ""

    def close_one(self, pos, retries: int = 3) -> bool:
        """Manually close a single position (dashboard 'close N' / number keys).

        Retries on transient failure; if all retries fail, the position is
        NOT marked flat and a critical alert is logged.
        """
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                with self.lock:
                    price = self._price_fn(pos.symbol)
                    if price is None:
                        price = pos.current
                    fill, ch = self.broker.exit_position(pos, price, ExitReason.SQUAREOFF)
                    trade = self.portfolio.close_trade(pos, fill, ExitReason.SQUAREOFF, ch)
                    self.portfolio.remove_position(pos.id)
                    if isinstance(self.broker.positions, dict):
                        self.broker.positions.pop(pos.id, None)
                    self.storage.delete_position(pos.id)
                    self.storage.save_trade(trade)
                self.log(f"EXIT {pos.market} {pos.symbol} manual @ {fill:.2f}", "order")
                return True
            except Exception as e:
                last_err = e
                self.log(f"manual close attempt {attempt}/{retries} failed {pos.symbol}: {e}", "warn")
                time.sleep(0.5 * attempt)
        # all retries failed: do NOT mark flat
        self.log(f"CRITICAL: failed to close {pos.market} {pos.symbol} after {retries} attempts: {last_err}", "error")
        return False

    def _update_fx(self):
        if time.time() - self.fx_last_update < 300:
            return
        try:
            import yfinance as yf
            # every currency a configured market trades in, converted to INR
            wanted = {"INR"}
            for m in self.markets_cfg.values():
                cur = m.get("currency")
                if cur and cur != "INR":
                    wanted.add(cur)
            # USDT is pegged to USD; Yahoo has no USDTINR pair
            fetch_map = {"USDT": "USD"}
            for cur in sorted(wanted):
                if cur == "INR":
                    continue
                qcur = fetch_map.get(cur, cur)
                try:
                    t = yf.Ticker(f"{qcur}INR=X")
                    h = t.history(period="1d")
                    if h is not None and not h.empty:
                        rate = float(h["Close"].iloc[-1])
                        self.portfolio.fx_rates[cur] = rate
                        if cur != qcur:
                            self.portfolio.fx_rates[qcur] = rate
                except Exception as e:
                    log.debug("fx %s failed: %s", cur, e)
            self.portfolio.fx_rates["INR"] = 1.0
        except Exception as e:
            log.warning("fx update failed: %s", e)
        self.fx_last_update = time.time()

    def scan_once(self):
        self.cycle += 1
        _scan_start = time.time()
        self._opened_cycle = 0
        cycle_scanned = 0
        cycle_signals = 0
        cycle_blocked = 0
        cycle_candidates = 0
        cycle_stale = 0
        self._update_fx()
        self._roll_over_day()
        self.portfolio.update_prices({})
        price_map = {}
        # The kill switch is a daily circuit breaker: once it fires we flatten
        # and stop taking new entries for the rest of the day. It is reset when
        # the day rolls over, so the system is never dead for good.
        if self.risk.kill_switch and not self._kill_flattened:
            self._kill_flattened = True
            try:
                n = self.square_off_all()
                self.log(f"kill switch: flattened {n} position(s), no new entries today", "warn")
            except Exception as e:
                self.log(f"kill square-off error: {e}", "error")
        for market in self.markets_cfg:
            st = self.clock.status(market)
            status = st["status"]
            is_open = status in ("OPEN", "PRE-CLOSE")
            # a market we are not scanning cannot protect its positions:
            # close them the moment the session ends
            if not is_open and status in ("CLOSED", "LUNCH"):
                if self._close_on_market_closed:
                    self._close_market_positions(
                        market, ExitReason.SQUAREOFF,
                        note="session over" if status == "CLOSED" else "lunch break")
            # forced flat shortly before the close
            if self.risk.should_square_off(market):
                self._close_market_positions(market, ExitReason.SQUAREOFF, note="pre-close")
            if not is_open:
                if market not in self._closed_logged and self.cycle <= 2:
                    self._closed_logged.add(market)
                    no = st.get("next_open")
                    txt = f"{market} closed, sleeping"
                    if no is not None:
                        try:
                            txt += f"; next open {no.strftime('%H:%M')}"
                        except Exception:
                            txt += f"; next open {no}"
                    self.log(txt)
                self.last_scan[market] = []
                continue
            symbols = self.watchlists.get(market, [])
            prov = self.providers[market]
            rows = []
            candidates = []
            # publish results incrementally so the dashboard fills in as we go
            self.last_scan[market] = rows
            if hasattr(prov, "batch_download"):
                data_map = prov.batch_download(symbols, self.cfg.get("candle_interval", "5m"),
                                             self.cfg.get("lookback_days", 5))
            else:
                data_map = {s: prov.get_candles(s, self.cfg.get("candle_interval", "5m"),
                                                self.cfg.get("lookback_days", 5)) for s in symbols}
            self.scan_market = market
            for sym_i, symbol in enumerate(symbols):
                self.current_scanning = f"{market} {symbol}"
                self.scan_progress = f"{market} {sym_i + 1}/{len(symbols)}"
                try:
                    df = data_map.get(symbol)
                    if df is None or df.empty:
                        rows.append({"symbol": symbol, "market": market, "price": None, "change": None,
                                     "regime": "no data", "best": None, "score": None,
                                     "signal": "NONE", "setups": 0, "firings": [],
                                     "metrics": {}, "plan": None, "gate": "",
                                     "updated": datetime.now(timezone.utc), "stale": True})
                        self.last_scan[market] = list(rows)
                        continue
                    # staleness
                    last_ts = df.index[-1]
                    age_bars = 0
                    try:
                        # count bars since last? just store ts
                        pass
                    except Exception:
                        pass
                    dfc = drop_unfinished(df)
                    price = float(df["close"].iloc[-1])
                    # % change from prev close (first bar of day? use prev bar)
                    change = 0.0
                    if len(df) >= 2:
                        change = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
                    stale = False
                    try:
                        now_utc = datetime.now(timezone.utc)
                        age = (now_utc - pd.Timestamp(last_ts).tz_convert("UTC").to_pydatetime()).total_seconds()
                        interval_sec = 300 if self.cfg.get("candle_interval","5m")=="5m" else 60
                        if age > 2 * interval_sec:
                            stale = True
                    except Exception:
                        pass
                    # simple: store last_ts
                    # regime
                    regime = self.selector.detect_regime(dfc)
                    metrics = self._scan_metrics(dfc, price, regime)
                    # run strategies - one shared Ctx for the whole symbol
                    live_signals = {}
                    firings = []
                    ctx = None
                    try:
                        from trader.strategies.context import Ctx
                        ctx = Ctx(dfc, symbol, market)
                    except Exception as e:
                        log.debug("ctx failed %s: %s", symbol, e)
                    for st in self.strategies:
                        try:
                            if ctx is not None:
                                sig = st.generate_from(ctx)
                            else:
                                sig = st.generate(dfc, symbol, market)
                        except Exception as e:
                            sig = None
                            self.log(f"strategy err {st.name} {symbol}: {e}", "error")
                        live_signals[st.name] = sig
                        if sig is not None:
                            try:
                                label = st.display_name()
                            except Exception:
                                label = st.name
                            firings.append({
                                "name": st.name, "label": label,
                                "side": sig.side.value,
                                "confidence": float(getattr(sig, "confidence", 0.0) or 0.0),
                                "entry": sig.entry, "stop": sig.stop, "target": sig.target,
                                "reason": sig.reason or "",
                            })
                    firings.sort(key=lambda f: -f["confidence"])
                    # selector
                    result = self.selector.evaluate(symbol, dfc, self.strategies, live_signals, self.cycle)
                    best_name = None
                    best_score = None
                    best_sig = None
                    best_bs = None
                    best_label = None
                    best_reason = None
                    best_reason_short = None
                    if result is not None:
                        best_name, best_score, best_bs, regime, best_sig = result[0], result[1], result[2], result[3], result[4]
                        st_obj = next((s for s in self.strategies if s.name == best_name), None)
                        best_label = st_obj.display_name() if st_obj is not None else best_name
                        best_reason = best_sig.reason if best_sig is not None else ""
                        # a compact form for the scanner column
                        best_reason_short = " + ".join(best_reason.split(" + ")[:2]) if best_reason else ""
                    # why a fired setup was not taken (first quality gate)
                    gate = ""
                    if best_sig is None and firings:
                        gate = self._gate_reason(symbol, live_signals)
                    # signal state: only the setup that would actually be
                    # traded counts - showing an arbitrary other setup's
                    # signal makes the table look busier than it is
                    if best_sig is not None:
                        signal_state = best_sig.side.value
                    else:
                        signal_state = "NONE"
                    plan = None
                    if best_sig is not None:
                        risk = abs(best_sig.entry - best_sig.stop)
                        reward = abs(best_sig.target - best_sig.entry)
                        plan = {"entry": best_sig.entry, "stop": best_sig.stop,
                                "target": best_sig.target,
                                "rr": (reward / risk) if risk > 0 else None}
                    stats_brief = None
                    if best_bs is not None:
                        stats_brief = {
                            "trades": best_bs.trades, "win_rate": best_bs.win_rate,
                            "profit_factor": best_bs.profit_factor,
                            "expectancy_r": best_bs.expectancy_r,
                        }
                    regime_fit = None
                    if best_name:
                        try:
                            regime_fit = self.selector.regime_fit(best_name, regime)
                        except Exception:
                            regime_fit = None
                    # collect the candidate; the trade decision is made once
                    # the whole market has been scanned, so only the best
                    # ranked setups in a market are ever taken
                    if best_sig is not None:
                        candidates.append({
                            "symbol": symbol, "price": price, "score": best_score,
                            "sig": best_sig, "name": best_name, "label": best_label,
                            "reason": best_reason, "regime": regime,
                        })
                    rows.append({
                        "symbol": symbol,
                        "market": market,
                        "price": price,
                        "prev_close": float(df["close"].iloc[-2]) if len(df) >= 2 else None,
                        "change": change,
                        "regime": regime.label,
                        "best": best_name,
                        "best_label": best_label or best_name,
                        "best_reason": best_reason or "",
                        "best_reason_short": best_reason_short or "",
                        "gate": gate,
                        "score": best_score,
                        "signal": signal_state,
                        "confidence": float(getattr(best_sig, "confidence", 0.0) or 0.0) if best_sig is not None else None,
                        "regime_fit": regime_fit,
                        "plan": plan,
                        "stats": stats_brief,
                        "setups": len(firings),
                        "firings": firings,
                        "metrics": metrics,
                        "updated": datetime.now(timezone.utc),
                        "stale": stale,
                    })
                    price_map[symbol] = price
                    cycle_scanned += 1
                    if stale:
                        cycle_stale += 1
                    if signal_state != "NONE":
                        cycle_signals += 1
                    if gate:
                        cycle_blocked += 1
                except Exception as e:
                    self.log(f"scan error {market} {symbol}: {e}", "error")
                    rows.append({"symbol": symbol, "market": market, "price": None, "change": None,
                                 "regime": "error", "best": None, "score": None,
                                 "signal": "NONE", "setups": 0, "firings": [],
                                 "metrics": {}, "plan": None, "gate": "",
                                 "updated": datetime.now(timezone.utc), "stale": True})
                # publish so the dashboard updates while the scan is still running
                self.last_scan[market] = list(rows)
            self.last_scan[market] = rows
            self.scan_market = ""
            cycle_candidates += len(candidates)
            # the market has been scanned - now decide what to actually trade
            with self.lock:
                self._trade_candidates(market, candidates)
        self._check_exits(price_map)
        self.latest_prices.update(price_map)
        for sym, px in price_map.items():
            self._note_price(sym, px)
        with self.lock:
            self.portfolio.update_prices(price_map)
        # update positions current via price map already
        # persist equity snapshot
        try:
            self.storage.save_equity_snapshot(self.portfolio.total_equity_inr(), self.portfolio.day_pnl_inr())
        except Exception as e:
            log.warning("persist snapshot failed: %s", e)
        self.equity_history.append(self.portfolio.total_equity_inr())
        self.equity_history = self.equity_history[-60:]
        self.last_scan_elapsed = time.time() - _scan_start
        self.current_scanning = None
        self.scan_progress = ""
        # report what was blocked, once per cycle instead of once per symbol
        if self._skipped:
            summary = ", ".join(f"{n} {reason}" for reason, n in
                                sorted(self._skipped.items(), key=lambda kv: -kv[1]))
            self.log(f"skipped {sum(self._skipped.values())} candidate(s): {summary}", "info")
            self.last_skipped = dict(self._skipped)
            for reason, n in self._skipped.items():
                self.skip_counts[reason] = self.skip_counts.get(reason, 0) + n
            self._skipped = {}
        self.last_cycle_stats = {
            "scanned": cycle_scanned, "signals": cycle_signals,
            "candidates": cycle_candidates, "opened": self._opened_cycle,
            "blocked_gate": cycle_blocked, "stale": cycle_stale,
        }

    # ------------------------------------------------------------ live quotes
    def start_price_thread(self):
        """Start the background thread that keeps prices and exits current."""
        if self._price_thread is not None:
            return
        self._price_thread = threading.Thread(target=self._price_loop, daemon=True,
                                              name="quotes")
        self._price_thread.start()

    def stop_price_thread(self):
        t = self._price_thread
        self._price_thread = None
        if t is not None and t.is_alive():
            self.running = False
            t.join(timeout=2.0)

    def _price_loop(self):
        """Refresh quotes continuously and act on them.

        This is what makes the dashboard move: prices are refreshed every
        second or so, and a position that hits its stop or target is closed
        within a second or two rather than at the next scan.

        Two tiers:
          1. the positions and the rows on screen - every second
          2. *every* symbol in every open market - worked through
             continuously, so a whole market is covered every few seconds
        """
        while self.running:
            started = time.time()
            try:
                if self.live_quotes and not self.paused:
                    self._refresh_quotes()
                    self._refresh_fast_batch(started)
            except Exception as e:
                log.debug("price loop error: %s", e)
                self.log(f"quote refresh error: {e}", "error")
            # keep the cadence steady whatever the refresh took
            wait = self.price_refresh_seconds - (time.time() - started)
            while wait > 0 and self.running:
                time.sleep(min(0.25, wait))
                wait = self.price_refresh_seconds - (time.time() - started)

    def _open_market_symbols(self) -> List[str]:
        """Every symbol in every market that is currently open."""
        out = []
        for market in self.markets_cfg:
            if market in self.disabled_markets:
                continue
            st = self.clock.status(market)
            if st["status"] not in ("OPEN", "PRE-CLOSE"):
                continue
            out.extend(self.watchlists.get(market, []))
        return out

    def _refresh_fast_batch(self, started: float):
        """Quote a slice of the open markets' symbols, within the time budget.

        The full list is worked through in order, so every symbol in a live
        market is refreshed roughly once per `fast_scan_seconds`, and the
        scanner keeps moving the whole time rather than pausing for a
        long batch.
        """
        budget = self.fast_scan_seconds * 0.55
        # rebuild the queue when the set of open markets changes
        symbols = self._open_market_symbols()
        if symbols != self._fast_queue:
            self._fast_queue = list(symbols)
            self._fast_cursor = 0
        if not self._fast_queue:
            return
        elapsed = time.time() - started
        room = budget - elapsed
        if room <= 0.05:
            return
        # how many can we quote in the time we have? (~30 symbols/second)
        capacity = max(1, int(room * 30))
        batch = []
        n = len(self._fast_queue)
        for i in range(capacity):
            batch.append(self._fast_queue[(self._fast_cursor + i) % n])
        self._fast_cursor = (self._fast_cursor + len(batch)) % n
        quotes = self.ticker.refresh(batch)
        if not quotes:
            return
        self.last_quote_at = time.time()
        with self.lock:
            for sym, px in quotes.items():
                self._note_price(sym, px)
            for rows in self.last_scan.values():
                for r in rows:
                    sym = r.get("symbol")
                    if sym in quotes:
                        r["live_price"] = quotes[sym]
                        r["live_change"] = self._live_change(r, quotes[sym])
        self._check_exits(quotes)

    def _watched_symbols(self) -> List[str]:
        """The symbols worth quoting right now.

        Open positions come first - they are the ones whose exits we act on -
        then the symbols on screen, then the rest.
        """
        with self.lock:
            positions = [p.symbol for p in self.portfolio.positions]
        visible = []
        for rows in self.last_scan.values():
            for r in rows[:25]:
                if r.get("symbol"):
                    visible.append(r["symbol"])
        seen = set()
        out = []
        for s in positions + visible:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _refresh_quotes(self):
        """Fetch live prices, update the display and act on exits."""
        symbols = self._watched_symbols()
        if not symbols:
            return
        quotes = self.ticker.refresh(symbols)
        self.last_quote_at = time.time()
        if not quotes:
            return
        # update what the scanner shows
        with self.lock:
            for sym, px in quotes.items():
                self._note_price(sym, px)
            for rows in self.last_scan.values():
                for r in rows:
                    sym = r.get("symbol")
                    if sym in quotes:
                        r["live_price"] = quotes[sym]
                        r["live_change"] = self._live_change(r, quotes[sym])
        # act on the new prices straight away
        self._check_exits(quotes)

    def _live_change(self, row: dict, price: float) -> Optional[float]:
        """% change of the live price against the reference close."""
        base = row.get("prev_close")
        if base:
            return (price / base - 1) * 100
        return row.get("change")

    def _check_exits(self, price_map):
        with self.lock:
            self._check_exits_locked(price_map)

    def _check_exits_locked(self, price_map):
        for pos in list(self.portfolio.positions):
            price = price_map.get(pos.symbol, pos.current)
            if price is None:
                continue
            pos.current = float(price)
            hit = None
            if pos.side == Side.LONG:
                if price <= pos.stop:
                    hit = ExitReason.STOPLOSS
                elif price >= pos.target:
                    hit = ExitReason.TARGET
            else:
                if price >= pos.stop:
                    hit = ExitReason.STOPLOSS
                elif price <= pos.target:
                    hit = ExitReason.TARGET
            # trailing stop logic: move to breakeven after +1R, then trail
            if hit is None and self.cfg["risk"].get("trailing_stop"):
                risk = abs(pos.entry - pos.stop) if abs(pos.entry - pos.stop) > 0 else 0.0
                mult = float(self.cfg["risk"].get("trailing_atr_mult", 1.0))
                trail_dist = risk * mult if risk > 0 else 0.0
                if pos.side == Side.LONG:
                    if risk > 0 and price >= pos.entry + risk and not pos.moved_to_be:
                        pos.stop = pos.entry
                        pos.moved_to_be = True
                    if trail_dist > 0:
                        new_stop = price - trail_dist
                        if new_stop > pos.stop:
                            pos.stop = new_stop
                else:
                    if risk > 0 and price <= pos.entry - risk and not pos.moved_to_be:
                        pos.stop = pos.entry
                        pos.moved_to_be = True
                    if trail_dist > 0:
                        new_stop = price + trail_dist
                        if new_stop < pos.stop:
                            pos.stop = new_stop
            if hit is not None:
                try:
                    fill, ch = self.broker.exit_position(pos, price, hit)
                    trade = self.portfolio.close_trade(pos, fill, hit, ch)
                    self.portfolio.remove_position(pos.id)
                    if isinstance(self.broker.positions, dict):
                        self.broker.positions.pop(pos.id, None)
                    self.storage.delete_position(pos.id)
                    self.storage.save_trade(trade)
                    self.log(f"EXIT {pos.market} {pos.symbol} {hit.value} @ {fill:.2f}", "order")
                except Exception as e:
                    self.log(f"exit error {pos.symbol}: {e}", "error")

    def square_off_all(self) -> int:
        """Flatten every open position. Returns how many were closed."""
        try:
            res = self.broker.square_off_all(self._price_fn)
            n = 0
            for item in res:
                pos, fill, ch = item[0], item[1], item[2]
                trade = self.portfolio.close_trade(pos, fill, ExitReason.SQUAREOFF, ch)
                self.portfolio.remove_position(pos.id)
                if isinstance(self.broker.positions, dict):
                    self.broker.positions.pop(pos.id, None)
                self.storage.save_trade(trade)
                self.storage.delete_position(pos.id)
                n += 1
            if n:
                self.log(f"square-off all: closed {n} position(s)", "order")
            return n
        except Exception as e:
            self.log(f"square-off all error: {e}", "error")
            return 0

    def shutdown_report(self):
        try:
            import csv, os
            os.makedirs(self.cfg.get("csv_dir", "data/reports"), exist_ok=True)
            path = os.path.join(self.cfg.get("csv_dir", "data/reports"), f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                # P&L is per market in its own currency, with the INR value
                # alongside so the rows can be added up
                w.writerow(["market", "symbol", "side", "strategy", "entry", "exit",
                            "qty", "ccy", "pnl", "pnl_inr", "exit_reason", "duration"])
                for t in self.portfolio.trades:
                    ccy = self.portfolio._currency(t.market)
                    w.writerow([t.market, t.symbol, t.side.value, t.strategy, t.entry, t.exit,
                                t.qty, ccy, t.pnl,
                                self.portfolio.to_inr(t.pnl, t.market),
                                t.exit_reason.value,
                                (t.closed_at - t.opened_at).total_seconds()])
            self.log(f"report written {path}", "info")
        except Exception as e:
            self.log(f"report error: {e}", "error")
