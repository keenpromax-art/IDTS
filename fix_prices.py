with open(r'C:\IDTS\trader\core\prices.py', 'r') as f:
    content = f.read()

old = '''    def _quote(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            price = yf.Ticker(symbol).fast_info.last_price
            if price is None:
                return None
            return float(price)
        except Exception:
            return None'''

new = '''    def _quote(self, symbol: str) -> Optional[float]:
        import yfinance as yf
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                price = ticker.fast_info.last_price
                if price is None:
                    return None
                return float(price)
            except Exception as e:
                if attempt < 2:
                    import time
                    import random
                    backoff = min(2 ** attempt, 30) + random.random()
                    time.sleep(backoff)
                else:
                    return None'''

if old in content:
    content = content.replace(old, new, 1)
    with open(r'C:\IDTS\trader\core\prices.py', 'w') as f:
        f.write(content)
    print('Done - rate limit retry added')
else:
    print('Old string not found')