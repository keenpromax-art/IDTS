from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Dict, Any, List
import pandas as pd
import yfinance as yf
from curl_cffi import requests

log = logging.getLogger(__name__)


def load_proxies(filepath: str) -> List[str]:
    """Load proxies from proxies.txt, filtering for HTTPS / valid entries."""
    proxies = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                import re
                m = re.search(r'\d{1,3}(?:\.\d{1,3}){3}:\d+', line)
                if m:
                    proxies.append(f"http://{m.group(0)}")
                elif line.lower().startswith('http://') or line.lower().startswith('https://'):
                    proxies.append(line)
                elif ':' in line:
                    p = line.split()[0]
                    if not p.lower().startswith('http'):
                        proxies.append(f"http://{p}")
                    else:
                        proxies.append(p)
    except Exception as e:
        log.warning("Failed to load proxies from %s: %s", filepath, e)
    return list(dict.fromkeys(proxies))


class ProxyPool:
    def __init__(self, proxies: List[str]):
        self.proxies = list(dict.fromkeys(proxies))
        self.stats: Dict[str, Dict[str, Any]] = {
            p: {"success": 0, "fail": 0, "cooldown_until": 0.0} for p in self.proxies
        }
        self._lock = threading.Lock()

    def validate(self):
        """Drop dead proxies at startup or re-validation."""
        valid = []
        for p in self.proxies:
            if self._test_proxy(p):
                valid.append(p)
            else:
                log.debug("Proxy failed validation: %s", p)
        with self._lock:
            self.proxies = valid
        log.info("ProxyPool validated: %d working proxies out of %d total", len(valid), len(self.stats))

    def _test_proxy(self, proxy: str) -> bool:
        try:
            s = requests.Session()
            s.proxies = {"http": proxy, "https": proxy}
            res = s.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d&interval=1m", timeout=3)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        return False

    def get_proxy(self) -> Optional[str]:
        now = time.time()
        with self._lock:
            available = []
            for p in self.proxies:
                info = self.stats[p]
                if info["cooldown_until"] <= now:
                    score = info["success"] - info["fail"]
                    available.append((score, info["success"], p))
            if not available:
                return None
            available.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return available[0][2]

    def report_success(self, proxy: str):
        if not proxy:
            return
        with self._lock:
            if proxy in self.stats:
                self.stats[proxy]["success"] += 1

    def report_failure(self, proxy: str):
        if not proxy:
            return
        now = time.time()
        with self._lock:
            if proxy in self.stats:
                self.stats[proxy]["fail"] += 1
                self.stats[proxy]["cooldown_until"] = now + 900  # 15-minute cooldown


class YahooFetcher:
    def __init__(self, pool: Optional[ProxyPool] = None, cache_ttl: int = 55):
        self.pool = pool
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def download(self, symbols, period="1d", interval="1m") -> Any:
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = tuple(sorted(set(symbols)))
        cache_key = (symbols, period, interval)

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[1]) < self.cache_ttl:
                return cached[0].copy() if hasattr(cached[0], "copy") else cached[0]

        # 1. Direct first
        data = self._fetch(symbols, period, interval, session=None)
        if data is not None:
            with self._lock:
                self._cache[cache_key] = (data, time.time())
            return data.copy() if hasattr(data, "copy") else data

        # 2. Fallback to proxies if pool available
        if self.pool:
            for _ in range(3):
                proxy = self.pool.get_proxy()
                if not proxy:
                    break
                try:
                    s = requests.Session()
                    s.proxies = {"http": proxy, "https": proxy}
                    data = self._fetch(symbols, period, interval, session=s)
                    if data is not None:
                        self.pool.report_success(proxy)
                        with self._lock:
                            self._cache[cache_key] = (data, time.time())
                        return data.copy() if hasattr(data, "copy") else data
                    else:
                        self.pool.report_failure(proxy)
                except Exception:
                    self.pool.report_failure(proxy)

        # 3. Final direct attempt
        data = self._fetch(symbols, period, interval, session=None)
        if data is not None:
            with self._lock:
                self._cache[cache_key] = (data, time.time())
            return data.copy() if hasattr(data, "copy") else data

        raise RuntimeError(f"YahooFetcher failed to fetch symbols {symbols}")

    def _fetch(self, symbols, period, interval, session) -> Optional[Any]:
        try:
            df = yf.download(
                list(symbols),
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
                session=session,
                timeout=10
            )
            if df is None or (hasattr(df, "empty") and df.empty):
                return None
            return df
        except Exception as e:
            log.debug("YahooFetcher fetch error (proxy=%s): %s", session is not None, e)
            return None


def start_background_validation(pool: ProxyPool, interval_seconds: int = 1800):
    def _run():
        while True:
            time.sleep(interval_seconds)
            try:
                pool.validate()
            except Exception as e:
                log.warning("Background proxy validation error: %s", e)
    t = threading.Thread(target=_run, daemon=True, name="ProxyValidator")
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing yf_proxy...")
    pool = ProxyPool(load_proxies("proxies.txt"))
    pool.validate()
    fetcher = YahooFetcher(pool, cache_ttl=55)
    try:
        df = fetcher.download(["AAPL", "MSFT"], period="1d", interval="1m")
        print("Fetched sample successfully:", type(df))
    except Exception as e:
        print("Fetch sample error:", e)
