from rich.theme import Theme

DEFAULT_STYLES = {
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "dim": "#565F6B",
    "border": "#30363D",
    "accent": "#58A6FF",
    "pnl.pos": "#3FB950",
    "pnl.neg": "#F85149",
    "pnl.zero": "#8B949E",
    "status.open": "#3FB950",
    "status.preclose": "#D29922",
    "status.closed": "#6E7681",
    "status.warn": "#D29922",
    "badge.paper": "black on #58A6FF",
    "badge.live": "bold white on #DA3633",
    "regime.trending": "#BC8CFF",
    "regime.ranging": "#39C5CF",
    "score.high": "#3FB950",
    "score.mid": "#D29922",
    "score.low": "#6E7681",
    "log.info": "#6E7681",
    "log.signal": "#58A6FF",
    "log.order": "#E6EDF3",
    "log.warn": "#D29922",
    "log.error": "bold #F85149",
    "chip.target": "#3FB950",
    "chip.sl": "#F85149",
    "chip.trail": "#58A6FF",
    "chip.sqoff": "#D29922",
    "chip.kill": "bold #F85149",
    # selection: full-row reverse video reads as an unambiguous highlight
    # without competing with the colour language (green=long/red=short)
    "sel": "reverse",
    "stale": "#D29922",
}

COLORBLIND_OVERRIDES = {
    "pnl.pos": "#1F77B4",
    "pnl.neg": "#FF7F0E",
    "status.open": "#1F77B4",
    "chip.target": "#1F77B4",
    "chip.sl": "#FF7F0E",
    "sel": "reverse",
    "stale": "#D29922",
}

MONO_OVERRIDES = {k: "white" for k in DEFAULT_STYLES if k.startswith(("pnl", "status", "chip", "score", "regime"))}


def build_theme(name="default"):
    styles = dict(DEFAULT_STYLES)
    if name == "colorblind":
        styles.update(COLORBLIND_OVERRIDES)
    elif name == "mono":
        styles.update(MONO_OVERRIDES)
    return Theme(styles)
