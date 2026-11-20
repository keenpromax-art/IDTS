"""Cross-frame body render cache for ``Dashboard.render()``.

Covers the body cache already implemented in ``trader/ui/dashboard.py``:

* same cache key + within the 1s TTL  -> the cached body object is reused
* changed engine/UI state              -> cache key changes, body is re-rendered
* TTL expiry                           -> body is re-rendered even with an unchanged key
"""

import time

from rich.console import Console

from trader.tests.test_dashboard import _demo_engine
from trader.ui.dashboard import Dashboard
from trader.ui.theme import build_theme

TTL = 1.0  # seconds; the TTL hard-coded in Dashboard.render()


def _make_dash(tmp_path, width=120, height=40):
    engine = _demo_engine(tmp_path)
    dash = Dashboard(engine)
    dash.console = Console(width=width, height=height,
                           theme=build_theme("default"), record=True)
    return dash


def test_body_cache_reuses_same_object_within_ttl(tmp_path):
    dash = _make_dash(tmp_path)
    dash.render()                            # populate the cache
    cached = dash._body_cache
    assert cached is not None                # a body was actually cached

    # second frame: same key, well within the TTL -> identity must be preserved
    dash.render()
    assert dash._body_cache is cached


def test_body_cache_stable_across_multiple_frames(tmp_path):
    dash = _make_dash(tmp_path)
    dash.render()
    cached = dash._body_cache
    for _ in range(5):                       # rapid frames, nothing changed
        dash.render()
        assert dash._body_cache is cached    # still the same object every time


def test_changed_engine_state_invalidates_body_cache(tmp_path):
    dash = _make_dash(tmp_path)
    dash.render()
    cached = dash._body_cache
    assert cached is not None

    # ``paused`` is part of the engine-state signature in the cache key
    dash.engine.paused = not dash.engine.paused
    dash.render()
    assert dash._body_cache is not cached    # key changed -> body re-rendered


def test_changed_ui_state_invalidates_body_cache(tmp_path):
    dash = _make_dash(tmp_path)
    dash.render()
    cached = dash._body_cache
    assert cached is not None

    # sort mode is part of the body cache key
    dash._sort_mode = "chg"
    dash.render()
    assert dash._body_cache is not cached    # key changed -> body re-rendered


def test_body_cache_expires_after_ttl(tmp_path):
    dash = _make_dash(tmp_path)
    dash.render()
    cached = dash._body_cache
    assert cached is not None

    # pretend the TTL has elapsed while the cache key is unchanged
    dash._body_cache_at = time.time() - 2.0
    dash.render()
    assert dash._body_cache is not cached    # TTL expired -> body re-rendered
