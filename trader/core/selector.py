from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from trader.core.models import Signal, Side, BacktestStats, Regime
from trader.core.indicators import adx, atr, ema
from trader.strategies.base import Strategy

log = logging.getLogger(__name__)


class Selector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sel = cfg.get("selector", {})
        self.min_trades = self.sel.get("min_trade_count", 8)
        self.min_expectancy_r = float(self.sel.get("min_expectancy_r", 0.0))
        self.min_profit_factor = float(self.sel.get("min_profit_factor", 1.1))
        self.min_win_rate = float(self.sel.get("min_win_rate", 0.35))
        self.min_signal_confidence = float(self.sel.get("min_signal_confidence", 0.55))
        self.sessions = self.sel.get("backtest_sessions", 5)
        # exponential decay half-life in cycles: expectancy half-life = HALF_LIFE cycles
        self.half_life_cycles = float(self.sel.get("expectancy_half_life", 0.0))
        # rolling weighted stats per symbol (for drift detection)
        self._rolling: Dict[str, Dict[str, float]] = {}  # symbol -> {r:0,w:0,pn:0,c:0}
        
        self.min_score = self.sel.get("min_score", 0.35)
        self.weights = self.sel.get("weights", {})
        self.trend_strats = set(self.sel.get("trend_strategies", []))
        self.meanrev_strats = set(self.sel.get("meanrev_strategies", []))
        self.adx_trend = self.sel.get("regime", {}).get("adx_trending_threshold", 25)
        self.atr_pct_high = self.sel.get("regime", {}).get("atr_percentile_high", 65)
        self._cache: Dict[str, Tuple[int, Dict[str, float]]] = {}  # symbol -> (cycle, scores)
        self._cycle = 0
        self._kinds: Dict[str, Any] = {}

    def set_strategies(self, strategies):
        """Tell the selector what each strategy is, so regime fit works."""
        self._kinds = {s.name: s for s in strategies}
    def detect_regime(self, df: pd.DataFrame) -> Regime:
        """Classify the regime: trending or ranging.

        Uses the numpy indicator core - the pandas version cost ~29ms a call
        and it is called twice per symbol on every scan.
        """
        r = Regime(symbol="")
        if df is None or len(df) < 30:
            return r
        from trader.core.np_indicators import adx_np, atr_np
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        adx_series = adx_np(high, low, close, 14)[0]
        atr_series = atr_np(high, low, close, 14)
        a = adx_series[-1]
        r.adx = 0.0 if (a is None or np.isnan(a)) else float(a)
        recent = atr_series[max(0, len(atr_series) - 100):]
        recent = recent[~np.isnan(recent)]
        if recent.size:
            r.atr_percentile = float((recent <= recent[-1]).sum() / recent.size * 100)
        r.trending = r.adx >= self.adx_trend
        r.ranging = not r.trending
        r.label = "trending" if r.trending else "ranging"
        return r

    def regime_fit(self, strategy_name: str, regime: Regime) -> float:
        """How well a strategy suits the current regime.

        Strategies declare their own kind (trend / meanrev / breakout), so
        new strategies get the right treatment without a config change.
        """
        kind = getattr(self._kinds.get(strategy_name, None), "kind", None)
        if kind == "trend":
            trend_like = True
            mean_like = False
        elif kind in ("meanrev",):
            trend_like = False
            mean_like = True
        elif kind in ("breakout", "volume"):
            # breakouts work in both, but prefer expansion
            trend_like = True
            mean_like = False
        else:
            # fall back to the configured name lists
            if strategy_name in self.trend_strats:
                trend_like, mean_like = True, False
            elif strategy_name in self.meanrev_strats:
                trend_like, mean_like = False, True
            else:
                trend_like = mean_like = False
        if regime.trending:
            if trend_like:
                return 1.0
            if mean_like:
                return 0.3
            return 0.6
        if regime.ranging:
            if mean_like:
                return 1.0
            if trend_like:
                return 0.3
            return 0.6
        return 0.5

    def _backtest_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slice the frame to the configured number of sessions.

        `backtest_sessions` is a count of *sessions*, not bars, so the bars
        per session are estimated from the data itself (the most common
        number of bars sharing a date).
        """
        sessions = max(1, int(self.sessions))
        max_bars = int(self.sel.get("backtest_max_bars", 250))
        if len(df) <= max_bars:
            return df
        try:
            idx = df.index
            if isinstance(idx, pd.MultiIndex):
                keys = idx.get_level_values(idx.names[0])
            else:
                keys = idx.date
            counts = pd.Series(list(keys)).value_counts()
            per_session = int(counts.iloc[0]) if len(counts) else 0
        except Exception:
            per_session = 0
        if per_session <= 0:
            return df.tail(max_bars)
        want = min(len(df), sessions * per_session, max_bars)
        return df.tail(want)

    def backtest(self, strategy: Strategy, df: pd.DataFrame, symbol: str,
                 ctx=None) -> BacktestStats:
        """Replay a strategy over *past* bars to score it for this symbol.

        The last bar is excluded on purpose: that is the bar the live signal
        was just generated on, so including it would let a setup be graded on
        the very trade that is asking for an entry. Scoring on history only
        keeps the ranking honest.

        The indicators are causal, so one Ctx built over the frame can be
        replayed bar by bar with the cursor - the value the cursor reads at
        bar i equals what a context built from df[:i+1] would hold. That
        turns N context builds into one.
        """
        stats = BacktestStats(strategy=strategy.name, symbol=symbol)
        if df is None or len(df) < 40:
            return stats
        df = self._backtest_window(df)
        if len(df) < 40:
            return stats
        # drop the bar the live signal came from - grade the setup on history
        df = df.iloc[:-1]
        if len(df) < 40:
            return stats
        slippage_bps = self.cfg.get("paper", {}).get("slippage_bps", 5) / 10000.0

        from trader.strategies.context import Ctx
        try:
            if ctx is None:
                ctx = Ctx(df, symbol, "")
        except Exception:
            return stats

        trades_pnl = []
        risks = []
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        i = 0
        n = len(df)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        while i < n - 1:
            ctx.set_bar(i)
            sig = None
            try:
                sig = strategy.generate_from(ctx)
            except Exception:
                sig = None
            if sig is None:
                i += 1
                continue
            entry_raw = float(close[i])
            if sig.side == Side.LONG:
                entry = entry_raw * (1 + slippage_bps)
            else:
                entry = entry_raw * (1 - slippage_bps)
            stop = float(sig.stop)
            target = float(sig.target)
            exit_price = None
            exit_idx = None
            for j in range(i + 1, n):
                hi = float(high[j])
                lo = float(low[j])
                if sig.side == Side.LONG:
                    if lo <= stop:
                        exit_price = stop * (1 - slippage_bps)
                        exit_idx = j
                        break
                    if hi >= target:
                        exit_price = target * (1 - slippage_bps)
                        exit_idx = j
                        break
                else:
                    if hi >= stop:
                        exit_price = stop * (1 + slippage_bps)
                        exit_idx = j
                        break
                    if lo <= target:
                        exit_price = target * (1 + slippage_bps)
                        exit_idx = j
                        break
            if exit_price is None:
                exit_price = float(close[-1])
                exit_idx = n - 1
            if sig.side == Side.LONG:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            risk = abs(entry - stop)
            if risk <= 0:
                i += 1
                continue
            trades_pnl.append(pnl)
            risks.append(risk)
            cum += pnl
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            i = exit_idx + 1 if exit_idx is not None else i + 1

        stats.trades = len(trades_pnl)
        wins = [p for p in trades_pnl if p > 0]
        losses = [p for p in trades_pnl if p <= 0]
        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.win_rate = (len(wins) / len(trades_pnl)) if trades_pnl else 0.0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        stats.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (2.0 if gross_win > 0 else 0.0)
        # expectancy in R multiples: profit normalised by the risk taken
        if trades_pnl and risks:
            rs = [p / r for p, r in zip(trades_pnl, risks)]
            stats.expectancy_r = sum(rs) / len(rs)
        else:
            stats.expectancy_r = 0.0
        stats.max_drawdown = float(max_dd)
        stats.eligible = self._is_eligible(stats)
        return stats

    def _is_eligible(self, stats: BacktestStats) -> bool:
        """Quality gates a setup must clear before it may be traded.

        Enough history, and then actual evidence of working: a positive
        expectancy, a profit factor above 1, and a win rate worth having. A
        setup that has been losing does not get traded because the market
        "feels" like its kind.
        """
        if stats.trades < self.min_trades:
            return False
        if stats.expectancy_r < self.min_expectancy_r:
            return False
        if stats.profit_factor < self.min_profit_factor:
            return False
        if stats.win_rate < self.min_win_rate:
            return False
        return True

    def score(self, stats: BacktestStats, regime: Regime, confidence: float) -> float:
        """Weighted score in 0..1.

        Expectancy is mapped so that *zero* expectancy is worth only 0.35, not
        0.5 - a setup that has been going nowhere must not look half-good
        just because it carries the heaviest weight.
        """
        w = self.weights
        exp = stats.expectancy_r
        # -0.5R -> 0.0, 0R -> 0.35, +0.5R -> 0.7, +1R+ -> 1.0
        if exp <= 0:
            exp_norm = 0.35 * (1.0 + max(-1.0, exp / 0.5) * 0.7)
        else:
            exp_norm = min(1.0, 0.35 + 0.65 * min(1.0, exp / 0.5))
        exp_norm = max(0.0, min(1.0, exp_norm))
        pf_norm = min(1.0, stats.profit_factor / 2.0)
        wr_norm = stats.win_rate
        rf = self.regime_fit(stats.strategy, regime)
        conf = min(1.0, max(0.0, confidence))
        total = (
            w.get("expectancy", 0.35) * exp_norm +
            w.get("profit_factor", 0.20) * pf_norm +
            w.get("win_rate", 0.15) * wr_norm +
            w.get("regime_fit", 0.15) * rf +
            w.get("confidence", 0.15) * conf
        )
        return float(total)

    def evaluate(self, symbol: str, df: pd.DataFrame, strategies: List[Strategy],
                 live_signals: Dict[str, Optional[Signal]], cycle: int) -> Optional[Tuple[str, float, BacktestStats, Regime, Optional[Signal]]]:
        # cache check
        cached = self._cache.get(symbol)
        refresh_every = self.sel.get("score_refresh_cycles", 5)
        if cached and (cycle - cached[0]) < refresh_every:
            scores = cached[1]
            regime = cached[2]
        else:
            regime = self.detect_regime(df)
            scores = {}
            # Only the strongest live signals are worth a backtest. Grading
            # every setup that happens to fire is what made a scan take
            # 40 seconds a market.
            max_bt = int(self.sel.get("max_backtest_setups", 6))
            live = [(st, live_signals.get(st.name)) for st in strategies
                    if live_signals.get(st.name) is not None]
            live.sort(key=lambda pair: -pair[1].confidence)
            if max_bt > 0:
                live = live[:max_bt]
            graded = {st.name for st, _ in live}
            # one context for the whole backtest window, shared by every
            # setup that is graded
            bt_ctx = None
            if graded:
                try:
                    from trader.strategies.context import Ctx
                    window = self._backtest_window(df)
                    bt_ctx = Ctx(window, symbol, "")
                except Exception:
                    bt_ctx = None
            for st in strategies:
                sig = live_signals.get(st.name)
                if sig is None or st.name not in graded:
                    bs = BacktestStats(strategy=st.name, symbol=symbol)
                    bs.eligible = False
                    scores[st.name] = (0.0, bs, regime, None)
                    continue
                bs = self.backtest(st, df, symbol, ctx=bt_ctx)
                conf = sig.confidence if sig is not None else 0.0
                sc = self.score(bs, regime, conf)
                scores[st.name] = (sc, bs, regime, sig)
            self._cache[symbol] = (cycle, scores, regime)
        # retrieve
        cached = self._cache.get(symbol)
        scores = cached[1]
        regime = cached[2]
        # candidates: strategies signalling now
        candidates = []
        for name, val in scores.items():
            sc, bs, reg, sig = val
            if not bs.eligible:
                continue
            if live_signals.get(name) is None:
                continue
            if sc < self.min_score:
                continue
            # a weak live signal is not worth a good historical score
            if sig is not None and sig.confidence < self.min_signal_confidence:
                continue
            candidates.append((name, sc, bs, regime, live_signals.get(name)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
