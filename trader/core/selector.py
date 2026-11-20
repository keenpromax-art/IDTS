from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from trader.core.models import Signal, Side, BacktestStats, Regime
from trader.core.indicators import adx, atr, ema
from trader.strategies.base import Strategy
from trader.scanner.ranking import (
    ExecutionModel, QualityRanker, RankingContext,
    compute_performance_metrics, walk_forward_folds,
)

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
        # scoring gates grade setups over a longer lookback than live signal
        # generation; defaults to `backtest_sessions` when not configured
        self.scoring_lookback_days = int(self.sel.get("scoring_lookback_days", self.sessions))
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
        # symbol -> (cycle, {name: (score, BacktestStats, Regime, Signal)}, Regime)
        self._cache: Dict[str, Tuple[int, Dict[str, Any], Regime]] = {}
        self._cycle = 0
        self._kinds: Dict[str, Any] = {}
        self.exec_model = ExecutionModel.from_cfg(
            self.sel.get("execution_model"),
            slippage_bps=float((cfg.get("paper") or {}).get("slippage_bps", 5.0)))
        self.ranker = QualityRanker(self.sel.get("ranking"))
        wf_cfg = self.sel.get("ranking") or {}
        self.wf_train = int(wf_cfg.get("walk_forward_train_bars", 120))
        self.wf_test = int(wf_cfg.get("walk_forward_test_bars", 60))
        self.wf_folds = int(wf_cfg.get("walk_forward_folds", 3))
        self.wf_enabled = bool(wf_cfg.get("walk_forward_enabled", False))
        self._wf_cache: Dict[tuple, float] = {}

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

    def _backtest_window(self, df: pd.DataFrame, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """Slice the frame to the configured number of sessions.

        `backtest_sessions` is a count of *sessions*, not bars, so the bars
        per session are estimated from the data itself (the most common
        number of bars sharing a date). `lookback_days` overrides the
        session count - scoring gates use a longer window than live signal
        generation.
        """
        sessions = max(1, int(lookback_days if lookback_days is not None else self.sessions))
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
                 ctx=None, lookback_days: Optional[int] = None) -> BacktestStats:
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
        df = self._backtest_window(df, lookback_days=lookback_days)
        if len(df) < 40:
            return stats
        # drop the bar the live signal came from - grade the setup on history
        df = df.iloc[:-1]
        if len(df) < 40:
            return stats
        model = self.exec_model

        from trader.strategies.context import Ctx
        try:
            if ctx is None:
                ctx = Ctx(df, symbol, "")
        except Exception:
            return stats

        i = 0
        n = len(df)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        opn = df["open"].to_numpy(dtype=float)
        rs: list = []
        costs_r: list = []

        while i < n - 1:
            ctx.set_bar(i)
            try:
                sig = strategy.generate_from(ctx)
            except Exception:
                sig = None
            if sig is None:
                i += 1
                continue
            is_long = sig.side == Side.LONG
            stop = float(sig.stop)
            target = float(sig.target)
            # Next-bar execution: the signal is known at bar i's close, so it
            # can only be filled at bar i+1's open.
            entry_idx = i + 1 if model.mode == "next_bar" else i
            if entry_idx >= n:
                break
            entry_raw = float(opn[entry_idx] if model.mode == "next_bar" else close[i])
            entry = model.entry_price(entry_raw, is_long)
            exit_price = None
            exit_raw = None
            exit_idx = None

            if model.mode == "next_bar" and (
                    (is_long and entry_raw <= stop) or (not is_long and entry_raw >= stop)):
                # The next open already breached the plan's stop: the setup is
                # invalidated. Do not invent a fill at a price that no longer
                # satisfies the risk defined by the signal.
                i = entry_idx + 1
                continue
            else:
                for j in range(entry_idx, n):
                    hi, lo = float(high[j]), float(low[j])
                    if is_long:
                        if lo <= stop:
                            exit_raw, exit_idx = stop, j
                        elif hi >= target:
                            exit_raw, exit_idx = target, j
                    else:
                        if hi >= stop:
                            exit_raw, exit_idx = stop, j
                        elif lo <= target:
                            exit_raw, exit_idx = target, j
                    if exit_idx is not None:
                        break
                if exit_idx is None:
                    exit_raw, exit_idx = float(close[-1]), n - 1
                exit_price = model.exit_price(exit_raw, is_long)

            risk = abs(entry - stop)
            if risk <= 1e-12:
                i = max(i + 1, entry_idx)
                continue
            pnl = (exit_price - entry) if is_long else (entry - exit_price)
            rs.append(pnl / risk)
            cost = abs(entry - entry_raw) + abs(exit_price - exit_raw)
            costs_r.append(cost / risk)
            i = max(entry_idx + 1, exit_idx + 1)

        metrics = compute_performance_metrics(rs, costs_r)
        stats.trades = metrics.trades
        stats.wins = sum(1 for r in rs if r > 0)
        stats.losses = stats.trades - stats.wins
        stats.win_rate = metrics.win_rate
        stats.profit_factor = metrics.profit_factor
        stats.expectancy_r = metrics.expectancy_r
        stats.net_expectancy_r = metrics.net_expectancy_r
        stats.sharpe = metrics.sharpe
        stats.sortino = metrics.sortino
        stats.max_drawdown = metrics.max_drawdown
        stats.sample_size = metrics.sample_size
        stats.win_rate_lower_bound = metrics.win_rate_lower_bound
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

    def score(self, stats: BacktestStats, regime: Regime, confidence: float,
              context: Optional[RankingContext] = None) -> float:
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
        if context is not None and self.ranker.enabled:
            enhanced = self.ranker.enhanced_score(stats, rf, conf, context)
            blend = max(0.0, min(1.0, self.ranker.blend))
            total = (1.0 - blend) * total + blend * enhanced
        return float(total)

    def walk_forward(self, strategy: Strategy, df: pd.DataFrame, symbol: str) -> float:
        """Chronological out-of-sample expectancy across non-overlapping folds.

        No parameters are fitted here, so this is a stability check, not an
        optimisation: the same strategy rules are replayed on later bars.
        """
        folds = walk_forward_folds(len(df), self.wf_train, self.wf_test, self.wf_folds)
        scores = []
        for _, _, ts, te in folds:
            window = df.iloc[ts:te]
            st = self.backtest(strategy, window, symbol)
            if st.trades >= max(3, self.min_trades // 3):
                scores.append(st.expectancy_r)
        return float(np.mean(scores)) if scores else 0.0

    def _grade(self, symbol: str, df: pd.DataFrame, entries, scores: Dict[str, Any],
               regime: Regime, context: Optional[Dict[str, Any]] = None):
        """Backtest and score a set of ``(strategy, signal)`` pairs.

        Historical grading depends only on past bars and the strategy's own
        declaration - never on whether another strategy fired in this candle.
        """
        if not entries:
            return
        try:
            from trader.strategies.context import Ctx
            window = self._backtest_window(df, lookback_days=self.scoring_lookback_days)
            bt_ctx = Ctx(window, symbol, "")
        except Exception:
            bt_ctx = None
        context = context or {}
        for st, sig in entries:
            bs = self.backtest(st, df, symbol, ctx=bt_ctx,
                               lookback_days=self.scoring_lookback_days)
            wf = 0.0
            if self.ranker.enabled and self.wf_enabled:
                key = (symbol, st.name)
                if key not in self._wf_cache:
                    self._wf_cache[key] = self.walk_forward(st, df, symbol)
                wf = self._wf_cache[key]
            rctx = RankingContext(
                relative_strength=float(context.get("relative_strength", 0.0)),
                liquidity_quality=float(context.get("liquidity_quality", 0.5)),
                data_freshness=float(context.get("data_freshness", 1.0)),
                walk_forward_score=wf,
            )
            bs.walk_forward_score = wf
            bs.liquidity_quality = rctx.liquidity_quality
            bs.data_freshness = rctx.data_freshness
            bs.relative_strength = rctx.relative_strength
            sc = self.score(bs, regime, float(sig.confidence), context=rctx)
            scores[st.name] = (sc, bs, regime, sig)

    def evaluate(self, symbol: str, df: pd.DataFrame, strategies: List[Strategy],
                 live_signals: Dict[str, Optional[Signal]], cycle: int,
                 context: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, float, BacktestStats, Regime, Optional[Signal]]]:
        cached = self._cache.get(symbol)
        refresh_every = max(1, int(self.sel.get("score_refresh_cycles", 5)))
        max_bt = int(self.sel.get("max_backtest_setups", 6))
        fresh = cached is None or (cycle - cached[0]) >= refresh_every

        if fresh:
            regime = self.detect_regime(df)
            scores: Dict[str, Any] = {}
            live = [(st, live_signals[st.name]) for st in strategies
                    if live_signals.get(st.name) is not None]
            live.sort(key=lambda pair: -pair[1].confidence)
            graded = live[:max_bt] if max_bt > 0 else live
            for st in strategies:
                scores[st.name] = (0.0, BacktestStats(strategy=st.name, symbol=symbol), regime, None)
            self._grade(symbol, df, graded, scores, regime, context)
            self._cache[symbol] = (cycle, scores, regime)
        else:
            regime = cached[2]
            scores = dict(cached[1])
            # A strategy can start firing on a cached cycle. Grading it only
            # at the next refresh made a valid, newly appearing signal vanish
            # for up to score_refresh_cycles - 1 cycles. Grade it now.
            pending = []
            for st in strategies:
                sig = live_signals.get(st.name)
                if sig is None:
                    continue
                entry = scores.get(st.name)
                if entry is None or not entry[1].eligible or entry[3] is None:
                    pending.append((st, sig))
                else:
                    # keep the cached historical score, but attach the current
                    # signal so the confidence gate uses this candle's data
                    scores[st.name] = (entry[0], entry[1], entry[2], sig)
            pending.sort(key=lambda pair: -pair[1].confidence)
            if max_bt > 0:
                pending = pending[:max_bt]
            if pending:
                self._grade(symbol, df, pending, scores, regime, context)
                self._cache[symbol] = (cycle, scores, regime)

        candidates = []
        for name, val in scores.items():
            sc, bs, _reg, _sig = val
            if not bs.eligible:
                continue
            sig = live_signals.get(name)
            if sig is None:
                continue
            if sc < self.min_score:
                continue
            if sig.confidence < self.min_signal_confidence:
                continue
            candidates.append((name, sc, bs, regime, sig))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
