from __future__ import annotations

"""Typed configuration validation.

Wraps the raw YAML dicts in pydantic models so invalid configuration fails
fast with a readable message instead of crashing deep inside the engine.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

try:
    from pydantic import ConfigDict
except Exception:  # pydantic v1 fallback
    ConfigDict = None


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = Field(default=0.75, gt=0, le=100)
    max_position_pct_of_equity: float = Field(default=20, gt=0, le=100)
    max_concurrent_positions: int = Field(default=6, ge=1)
    max_positions_per_market: int = Field(default=3, ge=1)
    max_trades_per_symbol_per_day: int = Field(default=2, ge=1)
    max_daily_loss_pct: float = Field(default=1.5, gt=0, le=100)
    cooldown_minutes_after_loss: int = Field(default=30, ge=0)
    no_entry_first_minutes: int = Field(default=5, ge=0)
    no_entry_last_minutes: int = Field(default=30, ge=0)
    square_off_before_close_minutes: int = Field(default=12, ge=0)
    min_risk_reward: float = Field(default=1.5, ge=0)
    close_on_market_closed: bool = True
    trailing_stop: bool = True
    trailing_atr_mult: float = Field(default=1.0, gt=0)


class SelectorConfig(BaseModel):
    min_trade_count: int = Field(default=8, ge=1)
    min_expectancy_r: float = 0.0
    min_profit_factor: float = Field(default=1.1, ge=0)
    min_win_rate: float = Field(default=0.35, ge=0, le=1)
    min_signal_confidence: float = Field(default=0.55, ge=0, le=1)
    min_score: float = Field(default=0.55, ge=0, le=1)
    min_market_score: float = Field(default=0.60, ge=0, le=1)
    max_trades_per_market_per_cycle: int = Field(default=2, ge=0)
    score_refresh_cycles: int = Field(default=5, ge=1)
    backtest_sessions: int = Field(default=5, ge=1)
    backtest_max_bars: int = Field(default=250, ge=10)
    max_backtest_setups: int = Field(default=6, ge=1)
    regime: Dict[str, Any] = Field(default_factory=lambda: {"adx_trending_threshold": 25})
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "expectancy": 0.40, "profit_factor": 0.20, "win_rate": 0.15,
        "regime_fit": 0.15, "confidence": 0.10,
    })

    @validator("weights")
    def _weights_sum(cls, v):
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("selector weights must sum to 1.0")
        return v


class Settings(BaseModel):
    mode: str = "paper"
    live_trading_confirmed: bool = False
    crypto_enabled: bool = True
    theme: str = "default"
    scan_interval_seconds: int = Field(default=60, ge=1)
    candle_interval: str = "5m"
    lookback_days: int = Field(default=5, ge=1, le=60)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    capital: Dict[str, float] = Field(default_factory=dict)
    paper: Dict[str, Any] = Field(default_factory=dict)
    strategies: Dict[str, Any] = Field(default_factory=dict)
    rules: Dict[str, Any] = Field(default_factory=dict)
    rule_params: Dict[str, Any] = Field(default_factory=dict)
    custom_strategies: Dict[str, Any] = Field(default_factory=dict)
    live_quotes: bool = True
    price_refresh_seconds: float = Field(default=1.0, gt=0)
    fast_scan_seconds: float = Field(default=5.0, gt=0)
    price_refresh_max_workers: int = Field(default=12, ge=1)
    db_path: str = "data/trader.db"
    csv_dir: str = "data/reports"
    ui: Dict[str, Any] = Field(default_factory=dict)

    @validator("mode")
    def _mode(cls, v):
        if v not in ("paper", "live"):
            raise ValueError("mode must be paper or live")
        return v

    @validator("theme")
    def _theme(cls, v):
        if v not in ("default", "colorblind", "mono"):
            raise ValueError("theme must be default/colorblind/mono")
        return v


def validate_settings(raw: Dict[str, Any]) -> Settings:
    return Settings(**(raw or {}))


def validate_markets(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict) or "markets" not in raw:
        raise ValueError("markets.yaml must contain a 'markets' mapping")
    markets = raw["markets"]
    if not isinstance(markets, dict) or not markets:
        raise ValueError("'markets' must be a non-empty mapping")
    for name, cfg in markets.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"market {name}: config must be a mapping")
        if "timezone" not in cfg:
            raise ValueError(f"market {name}: missing timezone")
        if not cfg.get("always_open"):
            for k in ("open", "close"):
                if k not in cfg:
                    raise ValueError(f"market {name}: missing {k}")
        if "code" not in cfg or "currency" not in cfg:
            raise ValueError(f"market {name}: missing code/currency")
        step = cfg.get("qty_step")
        if step is not None:
            try:
                step = float(step)
            except (TypeError, ValueError):
                raise ValueError(f"market {name}: qty_step must be a number")
            if not (step > 0):
                raise ValueError(f"market {name}: qty_step must be positive")
    return markets


def validate_watchlists(markets: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(raw, dict) or "watchlists" not in raw:
        raise ValueError("watchlists.yaml must contain a 'watchlists' mapping")
    wl = raw["watchlists"]
    if not isinstance(wl, dict):
        raise ValueError("'watchlists' must be a mapping")
    for m in markets:
        if m not in wl:
            raise ValueError(f"market {m} has no watchlist")
        if not isinstance(wl[m], list) or len(wl[m]) < 1:
            raise ValueError(f"market {m} watchlist must be a non-empty list")
        if len(wl[m]) != len(set(wl[m])):
            raise ValueError(f"market {m} watchlist has duplicates")
    return wl
