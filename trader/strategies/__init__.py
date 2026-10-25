from trader.strategies.base import Strategy
from trader.strategies.context import Ctx
from trader.strategies.rule import Rule, Condition

# single-indicator / classic setups
from trader.strategies.ema_family import (
    EMACrossStrategy, EMA2050Strategy, EMA50200Strategy, EMA821Strategy,
    EMATrendStackStrategy,
)
from trader.strategies.macd_family import (
    MACDMomentumStrategy, MACDZeroCrossStrategy, MACDDivergenceStrategy,
)
from trader.strategies.oscillators import (
    StochasticStrategy, CCIStrategy, ROCStrategy, MFIStrategy,
    WilliamsRStrategy, RSIReversalStrategy,
)
from trader.strategies.breakouts import (
    DonchianBreakoutStrategy, KeltnerBreakoutStrategy, BollingerExpansionStrategy,
    VolumeBreakoutStrategy, HighLowBreakoutStrategy, PSARStrategy,
    ADXDirectionalStrategy, IchimokuStrategy, TripleScreenStrategy,
)
from trader.strategies.momentum import (
    VWAPStrategy, VWAPBandStrategy, ORBStrategy, SupertrendStrategy,
    BBSqueezeStrategy, PivotBreakoutStrategy,
)
from trader.strategies.rules import build_rules

# Every strategy the engine can run. Rules are added separately because they
# are configured (enabled/disabled) in settings.yaml.
ALL_STRATEGIES = [
    EMACrossStrategy,
    EMA2050Strategy,
    EMA50200Strategy,
    EMA821Strategy,
    EMATrendStackStrategy,
    MACDMomentumStrategy,
    MACDZeroCrossStrategy,
    MACDDivergenceStrategy,
    StochasticStrategy,
    CCIStrategy,
    ROCStrategy,
    MFIStrategy,
    WilliamsRStrategy,
    RSIReversalStrategy,
    DonchianBreakoutStrategy,
    KeltnerBreakoutStrategy,
    BollingerExpansionStrategy,
    VolumeBreakoutStrategy,
    HighLowBreakoutStrategy,
    PSARStrategy,
    ADXDirectionalStrategy,
    IchimokuStrategy,
    TripleScreenStrategy,
    VWAPStrategy,
    VWAPBandStrategy,
    ORBStrategy,
    SupertrendStrategy,
    BBSqueezeStrategy,
    PivotBreakoutStrategy,
]

ALL_RULES = build_rules()


def strategy_registry():
    """name -> class for every single-indicator strategy."""
    return {cls.name: cls for cls in ALL_STRATEGIES}


def rule_registry():
    """name -> Rule instance for every composite setup."""
    return {r.name: r for r in ALL_RULES}
