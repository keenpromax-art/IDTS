"""Drop-in Python strategies.

Any .py file in this folder that defines a subclass of
``trader.strategies.base.Strategy`` is discovered and loaded at start-up.

Example - ``trader/custom/my_setup.py``::

    from trader.core.models import Side
    from trader.strategies.base import Strategy
    from trader.strategies.context import make_signal


    class MySetup(Strategy):
        name = "my_setup"
        kind = "trend"
        label = "My Setup"

        def __init__(self, period: int = 20):
            self.period = period

        def generate(self, df, symbol="", market="", ctx=None):
            from trader.strategies.context import Ctx
            c = ctx if ctx is not None else Ctx(df, symbol, market)
            if c.available < 30 or c.atr <= 0:
                return None
            ema = c._ema(self.period)
            if not c.crossed_up(c.close, ema):
                return None
            entry = c.px
            stop = entry - 1.5 * c.atr
            return make_signal(symbol, market, self.name, Side.LONG, entry, stop,
                               entry + 2.0 * (entry - stop), confidence=0.6,
                               reason=f"close crossed above EMA{self.period}")

Files starting with an underscore are ignored.
"""
