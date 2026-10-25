import sys
import numpy as np, pandas as pd
import yaml

sys.path.insert(0, '.')
from trader.core.engine import Engine
from trader.core.market_clock import MarketClock
import zoneinfo

settings = yaml.safe_load(open("trader/config/settings.yaml"))
markets = yaml.safe_load(open("trader/config/markets.yaml"))["markets"]
wl = yaml.safe_load(open("trader/config/watchlists.yaml"))["watchlists"]
settings["db_path"] = "data/_mk.db"


def fake_status(self, market, now=None):
    z = zoneinfo.ZoneInfo(markets[market]["timezone"])
    n = __import__("datetime").datetime.now(z)
    return {"market": market, "status": "OPEN", "reason": None,
            "local_time": n.replace(hour=11, minute=0),
            "open_dt": n.replace(hour=9, minute=30), "close_dt": n.replace(hour=16, minute=0),
            "next_open": None, "next_close": n.replace(hour=16, minute=0),
            "countdown_seconds": 3600, "in_lunch": False}


MarketClock.status = fake_status
e = Engine(settings, markets, wl, "paper", False)

# symbol -> expected market
owner = {}
for m, syms in wl.items():
    for s in syms:
        owner[s] = m

rng = np.random.default_rng(4)
frames = {}
for m, syms in wl.items():
    for s in syms:
        n = 300
        p = 100 + np.cumsum(rng.normal(0, 0.3, n))
        idx = pd.date_range("2026-03-02 09:15", periods=n, freq="5min", tz="UTC")
        frames[s] = pd.DataFrame({"open": p, "high": p + .25, "low": p - .25, "close": p,
                                 "volume": rng.integers(1000, 9000, n)}, index=idx)

for m, prov in e.providers.items():
    syms = wl[m]
    prov.get_candles = lambda s, *a, **k: frames.get(s)
    prov.batch_download = lambda syms, *a, **k: {s: frames.get(s) for s in syms}

e.scan_once()

print("rows with a wrong market:",
      sum(1 for rows in e.last_scan.values() for r in rows
          if r.get("market") and owner.get(r["symbol"]) and r["market"] != owner[r["symbol"]]))
print("positions with a wrong market:",
      sum(1 for p in e.portfolio.positions if owner.get(p.symbol) and p.market != owner[p.symbol]))
print("positions:", len(e.portfolio.positions))
for p in e.portfolio.positions[:8]:
    print(f"  {p.market:12} {p.symbol:12} expected={owner.get(p.symbol)}")
