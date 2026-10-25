from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import pandas as pd

log = logging.getLogger(__name__)

INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}


class AngelProvider:
    name = "angel"

    def __init__(self, api_key=None, client_id=None, pin=None, totp_secret=None,
                 cache_dir="data", rate_limit=0.34):
        self.api_key = api_key or os.getenv("ANGEL_API_KEY")
        self.client_id = client_id or os.getenv("ANGEL_CLIENT_ID")
        self.pin = pin or os.getenv("ANGEL_PIN")
        self.totp_secret = totp_secret or os.getenv("ANGEL_TOTP_SECRET")
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_call = 0.0
        self._smart = None
        self._token_map: Dict[str, str] = {}
        self._master_loaded = False
        self.available = False
        self.error = None
        self._try_login()

    def _try_login(self):
        if not all([self.api_key, self.client_id, self.pin, self.totp_secret]):
            self.error = "missing credentials"
            return
        try:
            import pyotp
            from SmartApi import SmartConnect
            totp = pyotp.TOTP(self.totp_secret).now()
            self._smart = SmartConnect(api_key=self.api_key)
            data = self._smart.generateSession(self.client_id, self.pin, totp)
            if not data or not data.get("status"):
                self.error = str(data)
                return
            self.available = True
            log.info("Angel login ok")
        except Exception as e:
            self.error = str(e)
            log.warning("Angel login failed: %s", e)
            self._smart = None
            self.available = False

    def _rate_limit(self):
        now = time.time()
        if now - self._last_call < self.rate_limit:
            time.sleep(self.rate_limit - (now - self._last_call))
        self._last_call = time.time()

    def _load_master(self):
        if self._master_loaded:
            return
        self._master_loaded = True
        if not self.available:
            return
        path = os.path.join(self.cache_dir, "angel_master.json")
        # refresh if older than 24h
        refresh = True
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < 24 * 3600:
                refresh = False
        if refresh:
            try:
                import requests
                url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(path, "w") as f:
                    f.write(r.text)
            except Exception as e:
                log.warning("master download failed: %s", e)
                if not os.path.exists(path):
                    return
        try:
            with open(path) as f:
                data = json.load(f)
            for row in data:
                if row.get("exch_seg") == "NSE":
                    sym = row.get("symbol", "")
                    tok = row.get("token")
                    if sym and tok:
                        # store both with and without .NS
                        self._token_map[sym] = tok
                        if sym.endswith("-EQ"):
                            base = sym.replace("-EQ", "")
                            self._token_map[base + ".NS"] = tok
                            self._token_map[base] = tok
        except Exception as e:
            log.warning("master parse failed: %s", e)

    def _token_for(self, symbol: str) -> Optional[str]:
        self._load_master()
        if symbol in self._token_map:
            return self._token_map[symbol]
        base = symbol.replace(".NS", "")
        if base in self._token_map:
            return self._token_map[base]
        # try with -EQ
        key = base + "-EQ"
        if key in self._token_map:
            return self._token_map[key]
        return None

    def get_candles(self, symbol, interval="5m", lookback_days=5) -> Optional[pd.DataFrame]:
        if not self.available:
            return None
        token = self._token_for(symbol)
        if not token:
            return None
        angel_interval = INTERVAL_MAP.get(interval, "FIVE_MINUTE")
        todate = datetime.now(timezone.utc)
        fromdate = todate - timedelta(days=lookback_days)
        # Angel expects IST? API uses local exchange time strings. Use IST.
        ist = timezone(timedelta(hours=5, minutes=30))
        from_str = fromdate.astimezone(ist).strftime("%Y-%m-%d %H:%M")
        to_str = todate.astimezone(ist).strftime("%Y-%m-%d %H:%M")
        self._rate_limit()
        try:
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": angel_interval,
                "fromdate": from_str,
                "todate": to_str,
            }
            resp = self._smart.getCandleData(params)
            if not resp or not resp.get("status"):
                log.warning("angel candle failed %s: %s", symbol, resp)
                return None
            data = resp.get("data") or []
            if not data:
                return None
            rows = []
            for r in data:
                # [timestamp, open, high, low, close, volume]
                ts = pd.to_datetime(r[0])
                rows.append({
                    "ts": ts,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                })
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"])
            # Angel returns IST timestamps naive; localize to IST then UTC
            if df["ts"].dt.tz is None:
                df["ts"] = df["ts"].dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
            df = df.set_index("ts")
            df.index.name = "ts"
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            log.warning("angel exception %s: %s", symbol, e)
            return None
