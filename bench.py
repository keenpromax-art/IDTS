import time, yaml
from pathlib import Path
PKG=Path('trader')
settings=yaml.safe_load(open(PKG/'config'/'settings.yaml'))
markets=yaml.safe_load(open(PKG/'config'/'markets.yaml'))['markets']
wl=yaml.safe_load(open(PKG/'config'/'watchlists.yaml'))['watchlists']
from trader.core.engine import Engine
e=Engine(settings,markets,wl,'paper',False)
orig=e.clock.status
def fake(market, now=None):
    st=orig(market, now)
    st['status']='OPEN'
    from datetime import datetime
    tz=__import__('zoneinfo').ZoneInfo(markets[market]['timezone'])
    st['local_time']=datetime.now(tz).replace(hour=10,minute=0)
    st['open_dt']=st['local_time'].replace(hour=9,minute=0)
    st['close_dt']=st['local_time'].replace(hour=15,minute=0)
    return st
e.clock.status=fake
t=time.time()
e.scan_once()
print('elapsed', round(time.time()-t,2), 'positions', len(e.portfolio.positions), 'prices', len(e.latest_prices))
