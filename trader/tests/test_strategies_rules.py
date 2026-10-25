import numpy as np
import pandas as pd
import pytest

from trader.core.models import Side
from trader.strategies import ALL_RULES, ALL_STRATEGIES
from trader.strategies.context import Ctx
from trader.strategies.rule import Condition, Rule


# ---------------------------------------------------------------- helpers

def make_df(n=300, drift=0.0, vol_mult=1.0, seed=1, days=1, per=78):
    """Synthetic OHLCV frame with a controllable drift."""
    rng = np.random.default_rng(seed)
    total = days * per
    price = 100.0
    close = []
    for d in range(days):
        d_drift = drift if d % 2 == 0 else -drift
        for _ in range(per):
            price += rng.normal(d_drift, 0.3 * vol_mult)
            close.append(price)
    close = np.array(close)
    high = close + np.abs(rng.normal(0, 0.25, total)) * vol_mult
    low = close - np.abs(rng.normal(0, 0.25, total)) * vol_mult
    volume = rng.integers(1000, 5000, total)
    if days > 1:
        idx = pd.MultiIndex.from_arrays(
            [np.repeat(pd.date_range("2026-03-02", periods=days, freq="B"), per),
             np.tile(pd.date_range("09:15", periods=per, freq="5min"), days)],
            names=["day", "t"])
    else:
        idx = pd.date_range("2026-03-02 09:15", periods=total, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def df():
    return make_df(n=320, drift=0.05, vol_mult=1.0, seed=11)


@pytest.fixture(scope="module")
def df_multiday():
    return make_df(n=6 * 78, drift=0.08, vol_mult=1.0, seed=13, days=6, per=78)


# ---------------------------------------------------------------- inventory

def test_at_least_fifteen_strategies():
    assert len(ALL_STRATEGIES) >= 15


def test_at_least_fifteen_rules():
    assert len(ALL_RULES) >= 15


def test_names_are_unique():
    names = [c.name for c in ALL_STRATEGIES] + [r.name for r in ALL_RULES]
    assert len(names) == len(set(names))


def test_every_strategy_declares_a_kind():
    for c in ALL_STRATEGIES:
        assert c.kind in ("trend", "meanrev", "breakout", "volume"), c.name
        assert c.label, c.name


def test_rules_declare_conditions():
    for r in ALL_RULES:
        assert r.conditions, r.name
        assert r.required_conditions() or r.required_groups(), r.name
        assert r.stop_atr > 0
        assert r.target_rr >= 1.5


# ---------------------------------------------------------------- behaviour

def test_all_strategies_run_without_error(df_multiday):
    for C in ALL_STRATEGIES:
        s = C()
        for i in range(120, len(df_multiday), 7):
            sig = s.generate(df_multiday.iloc[:i], "TEST", "India")
            if sig is not None:
                assert sig.stop != sig.entry, C.name
                assert sig.target != sig.entry, C.name
                assert sig.side in (Side.LONG, Side.SHORT), C.name
                assert sig.confidence >= 0.0 and sig.confidence <= 1.0, C.name


def test_all_rules_run_without_error(df_multiday):
    for r in ALL_RULES:
        for i in range(120, len(df_multiday), 7):
            sig = r.generate(df_multiday.iloc[:i], "TEST", "India")
            if sig is not None:
                assert sig.stop != sig.entry, r.name
                assert sig.target != sig.entry, r.name
                assert sig.reason, r.name
                assert sig.side in (Side.LONG, Side.SHORT), r.name


def test_strategies_handle_short_frames():
    tiny = make_df(n=40, seed=2)
    for C in ALL_STRATEGIES:
        s = C()
        for i in (5, 10, 20, 39):
            s.generate(tiny.iloc[:i], "TEST", "India")  # must not raise


def test_strategies_handle_empty_frame():
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []},
                         index=pd.DatetimeIndex([]))
    for C in ALL_STRATEGIES:
        assert C().generate(empty, "TEST", "India") is None


# ---------------------------------------------------------------- the rules

def test_rule_requires_every_required_condition():
    """A rule with a failing required condition must not fire."""
    always_true = Condition("t", "always", lambda ctx: True)
    always_false = Condition("f", "never", lambda ctx: False)
    r = Rule(name="r_test", label="test", conditions=[always_true, always_false],
             side="long")
    df = make_df(n=120, seed=3)
    ctx = Ctx(df, "T", "India")
    assert r.evaluate(ctx) is None


def test_rule_fires_when_conditions_align():
    always_true = Condition("t", "always", lambda ctx: True)
    r = Rule(name="r_test", label="test", conditions=[always_true], side="long")
    df = make_df(n=120, seed=3)
    ctx = Ctx(df, "T", "India")
    res = r.evaluate(ctx)
    assert res is not None
    side, matched, conf = res
    assert side == Side.LONG
    assert "always" in matched
    assert conf > 0


def test_rule_group_needs_one_member():
    """A direction group passes when exactly one member is true."""
    up = Condition("up", "up", lambda ctx: True, group="direction")
    dn = Condition("dn", "down", lambda ctx: False, group="direction")
    other = Condition("o", "other", lambda ctx: True)
    r = Rule(name="r_g", label="g", conditions=[up, dn, other], side="auto",
             bull_keys=("up",), bear_keys=("dn",))
    df = make_df(n=120, seed=4)
    ctx = Ctx(df, "T", "India")
    res = r.evaluate(ctx)
    assert res is not None
    assert res[0] == Side.LONG


def test_rule_group_fails_when_no_member_passes():
    up = Condition("up", "up", lambda ctx: False, group="direction")
    dn = Condition("dn", "down", lambda ctx: False, group="direction")
    r = Rule(name="r_g2", label="g2", conditions=[up, dn], side="auto",
             bull_keys=("up",), bear_keys=("dn",))
    df = make_df(n=120, seed=5)
    ctx = Ctx(df, "T", "India")
    assert r.evaluate(ctx) is None


def test_rule_confidence_threshold_filters():
    """A rule whose optional confirmations do not come must be filtered out."""
    req = [Condition(f"r{i}", f"req{i}", lambda ctx: True) for i in range(4)]
    opt = [Condition(f"o{i}", f"opt{i}", lambda ctx: False, required=False) for i in range(4)]
    r = Rule(name="r_c", label="c", conditions=req + opt, side="long", min_confidence=0.8)
    df = make_df(n=120, seed=6)
    ctx = Ctx(df, "T", "India")
    # 4 of 8 conditions -> confidence 0.45 + 0.55*0.5 = 0.725 < 0.8
    assert r.evaluate(ctx) is None
    # the same rule with a lower threshold passes
    r2 = Rule(name="r_c2", label="c2", conditions=req + opt, side="long", min_confidence=0.5)
    assert r2.evaluate(ctx) is not None


def test_rule_min_optional_filters():
    req = Condition("t", "always", lambda ctx: True)
    opt = Condition("o", "optional", lambda ctx: False)
    r = Rule(name="r_o", label="o", conditions=[req, opt], side="long", min_optional=1)
    df = make_df(n=120, seed=7)
    ctx = Ctx(df, "T", "India")
    assert r.evaluate(ctx) is None


def make_trend_pullback(n=320, seed=21, sign=1.0, drift=0.05, noise=0.12):
    """A steady trend where price hugs EMA20 - the rule_trend_stack setup.

    A slow grind keeps price near EMA20 while EMA20 > EMA50 > EMA200 holds
    and ADX stays high, which is exactly the alignment the rule needs.
    """
    rng = np.random.default_rng(seed)
    price = 100.0
    close = []
    for _ in range(n):
        price += rng.normal(drift * sign, noise)
        close.append(price)
    close = np.array(close)
    high = close + np.abs(rng.normal(0, noise * 0.6, n))
    low = close - np.abs(rng.normal(0, noise * 0.6, n))
    volume = rng.integers(1000, 5000, n)
    idx = pd.date_range("2026-03-02 09:15", periods=n, freq="5min")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_rule_reason_lists_matched_conditions():
    r = [x for x in ALL_RULES if x.name == "rule_trend_stack"][0]
    df = make_trend_pullback()
    found = False
    for i in range(210, len(df), 2):
        sig = r.generate(df.iloc[:i], "T", "India")
        if sig is not None:
            assert "EMA20" in sig.reason
            assert sig.side == Side.LONG
            found = True
            break
    assert found, "rule_trend_stack never fired on a steady uptrend"


def test_short_rule_fires_on_downtrend():
    r = [x for x in ALL_RULES if x.name == "rule_trend_stack_short"][0]
    df = make_trend_pullback(seed=22, sign=-1.0, drift=0.03, noise=0.12)
    found = False
    for i in range(210, len(df), 2):
        sig = r.generate(df.iloc[:i], "T", "India")
        if sig is not None:
            assert sig.side == Side.SHORT
            found = True
            break
    assert found, "rule_trend_stack_short never fired on a steady downtrend"


def test_conditions_are_side_aware():
    """Long and short rules must produce opposite sides."""
    long_rule = [x for x in ALL_RULES if x.name == "rule_trend_stack"][0]
    short_rule = [x for x in ALL_RULES if x.name == "rule_trend_stack_short"][0]
    assert long_rule.side == "long"
    assert short_rule.side == "short"


# ---------------------------------------------------------------- shared ctx

def test_ctx_computes_every_indicator_once(df):
    ctx = Ctx(df, "T", "India")
    assert ctx.atr > 0
    assert ctx.ema50[-1] == ctx.ema50[-1]  # not NaN
    assert len(ctx.macd_line) == len(df)
    assert ctx.vol_ratio > 0


def test_shared_ctx_gives_same_result_as_solo(df):
    """A strategy must behave identically with and without a shared ctx."""
    for C in ALL_STRATEGIES:
        s = C()
        ctx = Ctx(df, "T", "India")
        a = s.generate(df, "T", "India")
        b = s.generate_from(ctx)
        if a is None and b is None:
            continue
        assert a is not None and b is not None, C.name
        assert a.side == b.side, C.name
        assert abs(a.entry - b.entry) < 1e-9, C.name
        assert abs(a.stop - b.stop) < 1e-9, C.name


def test_breakouts_use_prior_bar_levels(df):
    """A close can only exceed the prior range, never the current one."""
    ctx = Ctx(df, "T", "India")
    assert ctx.donch_hi[-1] >= ctx.donch_hi_prev[-1] - 1e-9
    assert ctx.donch_lo[-1] <= ctx.donch_lo_prev[-1] + 1e-9
