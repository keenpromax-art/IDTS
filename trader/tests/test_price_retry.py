"""Tests for fetch_with_retry and PriceTicker failure handling."""

from unittest.mock import patch

import pytest

from trader.core.prices import PriceTicker, STALE_QUOTE_SECONDS
from trader.data.yf_provider import fetch_with_retry


class TransientError(Exception):
    """Simulates a transient provider failure."""


def test_retry_returns_success_on_first_try():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return 42.0

    assert fetch_with_retry(fn) == 42.0
    assert calls["n"] == 1


def test_retry_with_jitter_retries_on_transient_failure():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("blip")
        return 7.0

    assert fetch_with_retry(fn) == 7.0
    assert calls["n"] == 3


def test_retry_raises_last_exception_after_exhaustion():
    last = TransientError("permanent")

    def fn():
        raise last

    with pytest.raises(TransientError, match="permanent"):
        fetch_with_retry(fn, retries=2)


def test_price_ticker_marks_stale_on_total_failure():
    t = PriceTicker(max_workers=2)
    try:
        t._quote = lambda symbol: None  # every provider call fails
        out = t.refresh(["AAAA", "BBBB"])
        assert out == {}
        assert t.price("AAAA") is None
        assert t.age("AAAA") is not None
        assert t.age("AAAA") > STALE_QUOTE_SECONDS
    finally:
        t.close()
