import numpy as np
import pandas as pd

from trader.core.data_quality import validate_candle_df, is_usable


def _good(n=20):
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="5min", tz="UTC")
    rng = np.arange(n)
    df = pd.DataFrame({
        "open": 100 + rng,
        "high": 101 + rng,
        "low": 99 + rng,
        "close": 100.5 + rng,
        "volume": 1000 + rng,
    }, index=idx)
    return df


def test_good_frame_ok():
    r = validate_candle_df(_good())
    assert r.ok is True
    assert r.issues == []
    assert r.rows == 20
    assert is_usable(_good()) is True


def test_empty_frame():
    r = validate_candle_df(pd.DataFrame())
    assert r.ok is False
    assert "empty" in " ".join(r.issues).lower()


def test_none():
    r = validate_candle_df(None)
    assert r.ok is False


def test_missing_columns():
    df = _good().drop(columns=["volume"])
    r = validate_candle_df(df)
    assert r.ok is False
    assert any("missing columns" in i for i in r.issues)


def test_ohlc_violation():
    df = _good().copy()
    df.loc[df.index[3], "high"] = df.loc[df.index[3], "low"] - 1
    r = validate_candle_df(df)
    assert r.ok is False
    assert r.ohlc_violations > 0


def test_non_positive_price():
    df = _good().copy()
    df.loc[df.index[1], "close"] = 0
    r = validate_candle_df(df)
    assert r.ok is False
    assert r.non_positive_prices > 0


def test_duplicate_timestamps():
    df = _good().copy()
    df = pd.concat([df, df.iloc[[0]]])
    r = validate_candle_df(df)
    assert r.ok is False
    assert r.duplicate_timestamps >= 1


def test_unsorted():
    df = _good().iloc[::-1].copy()
    r = validate_candle_df(df)
    assert r.ok is False
    assert r.unsorted is True


def test_nan_cells():
    df = _good().copy()
    df.loc[df.index[2], "close"] = float("nan")
    r = validate_candle_df(df)
    assert r.ok is False
    assert r.nan_cells >= 1


def test_negative_volume():
    df = _good().copy()
    df.loc[df.index[0], "volume"] = -5
    r = validate_candle_df(df)
    assert r.ok is False


def test_stale_detection():
    df = _good()
    now = df.index[-1] + pd.Timedelta(hours=2)
    r = validate_candle_df(df, now=now, stale_after_seconds=300)
    assert r.ok is False
    assert r.stale_seconds is not None
    assert r.stale_seconds > 300


def test_fresh_not_stale():
    df = _good()
    now = df.index[-1] + pd.Timedelta(seconds=10)
    r = validate_candle_df(df, now=now, stale_after_seconds=300)
    assert r.ok is True
