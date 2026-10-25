import sys
sys.path.insert(0, r'C:\IDTS')

with open(r'C:\IDTS\trader\core\selector.py', 'r') as f:
    content = f.read()

old = '        self.sessions = self.sel.get("backtest_sessions", 5)'
new = """        self.sessions = self.sel.get("backtest_sessions", 5)
        # exponential decay half-life in cycles: expectancy half-life = HALF_LIFE cycles
        self.half_life_cycles = float(self.sel.get("expectancy_half_life", 0.0))
        # rolling weighted stats per symbol (for drift detection)
        self._rolling: Dict[str, Dict[str, float]] = {}  # symbol -> {r:0,w:0,pn:0,c:0}
        
self.min_trades = self.sel.get("min_trade_count", 8)"""

if old in content:
    content = content.replace(old, new, 1)
    with open(r'C:\IDTS\trader\core\selector.py', 'w') as f:
        f.write(content)
    print("Done")
else:
    print("Old string not found!")
    # Debug: show lines around 24
    lines = content.split('\n')
    for i in range(22, 27):
        print(f"Line {i+1}: {repr(lines[i])}")