from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import pandas as pd

from trader.core.data_quality import validate_candle_df

log = logging.getLogger(__name__)


def validate_provider_frame(df, symbol="", interval="5m", stale_after_seconds=None):
    """Run data-quality validation and log a concise summary on failure."""
    rep = validate_candle_df(df, stale_after_seconds=stale_after_seconds)
    if not rep.ok:
        log.warning("data quality %s: %s", symbol, "; ".join(rep.issues))
    return rep
