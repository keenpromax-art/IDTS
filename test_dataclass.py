import sys
sys.path.insert(0, r'C:\IDTS')

from dataclasses import dataclass

@dataclass
class TestClass:
    cfg: dict
    
    def __post_init__(self):
        print(f'self.cfg = {self.cfg}')
        risk_key = "risk" in self.cfg
        print(f'"risk" in self.cfg: {risk_key}')
        if risk_key:
            risk_val = self.cfg["risk"]
            print(f'self.cfg["risk"]: {risk_val}')

t = TestClass({'risk': {'max_daily_loss_pct': 2.0}})
print('Done')