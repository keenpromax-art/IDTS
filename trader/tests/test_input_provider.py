from __future__ import annotations

import types

import pytest

from trader.ui.input_base import InputProvider
from trader.ui.input_win import WindowsInputProvider
from trader.ui.dashboard import Dashboard


class _FakeStorage:
    def load_equity_history(self, n):  # pragma: no cover - exercised via exception path
        raise RuntimeError("no storage in tests")


class _FakeEngine:
    """Minimal engine stand-in for Dashboard wiring tests."""

    def __init__(self):
        self.cfg = {}
        self.markets_cfg = {"India": {}}
        self.storage = _FakeStorage()
        self.running = True
        self.logged = []

    def log(self, msg, level="info"):
        self.logged.append((msg, level))


class _OnceProvider(InputProvider):
    """Returns b'q' on the first poll, then None forever."""

    def __init__(self):
        self.calls = 0

    def poll(self):
        self.calls += 1
        return b"q" if self.calls == 1 else None


def test_input_provider_is_abstract():
    with pytest.raises(TypeError):
        InputProvider()


def test_windows_input_provider_poll_returns_none_when_no_keys(monkeypatch):
    import trader.ui.input_win as iw

    monkeypatch.setattr(
        iw,
        "msvcrt",
        types.SimpleNamespace(kbhit=lambda: False, getch=lambda: b"x"),
    )
    assert WindowsInputProvider().poll() is None


def test_windows_input_provider_poll_returns_key(monkeypatch):
    import trader.ui.input_win as iw

    monkeypatch.setattr(
        iw,
        "msvcrt",
        types.SimpleNamespace(kbhit=lambda: True, getch=lambda: b"q"),
    )
    assert WindowsInputProvider().poll() == b"q"


def test_dashboard_uses_input_provider():
    engine = _FakeEngine()
    provider = _OnceProvider()
    dash = Dashboard(engine, input_provider=provider)

    # the constructor falls back to WindowsInputProvider when none is given
    assert dash._input_provider is provider

    dash._hotkey_loop()

    assert engine.running is False
    assert dash._stop is True
    assert provider.calls >= 1
