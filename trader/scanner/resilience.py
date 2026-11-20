"""Resilience primitives shared by every data provider.

Short negative caching, jittered exponential backoff, per-source circuit
breakers and thread-safe rate limiting live here so no provider reimplements
them. All clocks and sleep functions are injectable, which makes the failure
modes deterministically testable without waiting in real time.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-source breaker: fail fast while open, probe once half-open.

    A source is an entire provider (Yahoo, Angel), not a symbol: repeated
    failures for one bad ticker must not take a healthy provider offline.
    """

    def __init__(self, name: str = "source", failure_threshold: int = 5,
                 recovery_seconds: float = 30.0, half_open_max_calls: int = 1,
                 *, clock: Optional[Callable[[], float]] = None):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_seconds = float(recovery_seconds)
        self.half_open_max_calls = max(1, int(half_open_max_calls))
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_calls = 0
        self.trips = 0

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self):
        if self._state is BreakerState.OPEN:
            if self._clock() - self._opened_at >= self.recovery_seconds:
                self._state = BreakerState.HALF_OPEN
                self._half_calls = 0

    def allow(self) -> bool:
        with self._lock:
            self._maybe_half_open()
            if self._state is BreakerState.CLOSED:
                return True
            if self._state is BreakerState.HALF_OPEN and self._half_calls < self.half_open_max_calls:
                self._half_calls += 1
                return True
            return False

    def record_success(self):
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._half_calls = 0

    def record_failure(self):
        with self._lock:
            if self._state is BreakerState.HALF_OPEN:
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()
                self._half_calls = 0
                self.trips += 1
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()
                self.trips += 1

    def reset(self):
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._half_calls = 0


@dataclass
class NegativeEntry:
    failed_at: float
    failures: int = 1


class NegativeCache:
    """Short-lived suppression of known-missing keys.

    Prevents hammering a provider for a symbol that has no data (delisted,
    wrong suffix, halted), while expiring quickly enough that a symbol coming
    back is picked up without a restart.
    """

    def __init__(self, ttl_seconds: float = 15.0, max_entries: int = 10000,
                 *, clock: Optional[Callable[[], float]] = None):
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = max(1, int(max_entries))
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._entries: Dict[str, NegativeEntry] = {}
        self.hits = 0
        self.misses = 0

    def is_blocked(self, key: str) -> bool:
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                self.misses += 1
                return False
            if self._clock() - e.failed_at >= self.ttl_seconds:
                self._entries.pop(key, None)
                self.misses += 1
                return False
            self.hits += 1
            return True

    def record(self, key: str):
        with self._lock:
            now = self._clock()
            e = self._entries.get(key)
            if e and now - e.failed_at < self.ttl_seconds:
                e.failures += 1
                e.failed_at = now
            else:
                self._entries[key] = NegativeEntry(failed_at=now)
            if len(self._entries) > self.max_entries:
                oldest = min(self._entries, key=lambda k: self._entries[k].failed_at)
                self._entries.pop(oldest, None)

    def clear(self, key: Optional[str] = None):
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RateLimiter:
    """Thread-safe minimum-spacing limiter, optionally keyed per source."""

    def __init__(self, min_interval_seconds: float = 0.0,
                 *, clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None):
        self.min_interval = max(0.0, float(min_interval_seconds))
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}

    def acquire(self, key: str = "default"):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = self._clock()
            last = self._last.get(key, 0.0)
            wait = self.min_interval - (now - last)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
            self._last[key] = now


@dataclass
class RetryPolicy:
    retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.multiplier ** attempt), self.max_delay)
        if self.jitter:
            delay += random.random() * self.base_delay * 0.5
        return delay


@dataclass
class FetchOutcome:
    ok: bool
    value: Any = None
    error: Optional[str] = None
    attempts: int = 0
    cache_hit: bool = False
    breaker_blocked: bool = False


class ProviderResilience:
    """Combine negative cache, breaker, limiter and retry into one call path."""

    def __init__(self, name: str = "provider", *, negative_ttl_seconds: float = 15.0,
                 failure_threshold: int = 5, recovery_seconds: float = 30.0,
                 min_interval_seconds: float = 0.0, retry: Optional[RetryPolicy] = None,
                 clock: Optional[Callable[[], float]] = None,
                 sleep: Optional[Callable[[float], None]] = None):
        self.name = name
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.retry = retry or RetryPolicy()
        self.negative = NegativeCache(ttl_seconds=negative_ttl_seconds, clock=self.clock)
        self.breaker = CircuitBreaker(name, failure_threshold=failure_threshold,
                                      recovery_seconds=recovery_seconds, clock=self.clock)
        self.limiter = RateLimiter(min_interval_seconds, clock=self.clock, sleep=self.sleep)
        self.successes = 0
        self.failures = 0

    def fetch(self, key: str, fn: Callable[[], Any], *,
              is_valid: Optional[Callable[[Any], bool]] = None) -> FetchOutcome:
        """Fetch *key* through every resilience layer.

        ``fn`` must raise on failure. A result rejected by ``is_valid`` counts
        as a failure (e.g. an empty candle frame), which feeds the negative
        cache and breaker without fabricating data.
        """
        if self.negative.is_blocked(key):
            return FetchOutcome(False, error="negative cache", cache_hit=True)
        if not self.breaker.allow():
            return FetchOutcome(False, error="circuit breaker open", breaker_blocked=True)
        last_error = "unknown error"
        for attempt in range(self.retry.retries + 1):
            self.limiter.acquire(key)
            try:
                value = fn()
                if is_valid is not None and not is_valid(value):
                    raise ValueError("empty or invalid result")
                self.breaker.record_success()
                self.successes += 1
                return FetchOutcome(True, value=value, attempts=attempt + 1)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retry.retries:
                    self.sleep(self.retry.delay_for(attempt))
        self.breaker.record_failure()
        self.negative.record(key)
        self.failures += 1
        return FetchOutcome(False, error=last_error, attempts=self.retry.retries + 1)

    @property
    def health(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.breaker.state.value,
            "successes": self.successes,
            "failures": self.failures,
            "negative_entries": len(self.negative),
            "negative_hits": self.negative.hits,
            "breaker_trips": self.breaker.trips,
        }
