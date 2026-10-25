# Implementation Roadmap

P0 (correctness/safety) - done in this pass:
- Data-quality validation module + provider gate.
- Broker capability + live gate verification.
- Idempotent trade closure.
- Trailing-stop fix.
- USDT FX mapping.
- Hotkey thread hardening.
- Config validation (pydantic).
- Storage indexes.
- Panel sizing fix.

P0 - remaining:
- Separate display price from signal price (last closed bar).
- Emergency-exit retry policy + critical alert.
- Daily-loss rollover for 24/7 markets (rolling 24h option).

P1 (research foundation) - in progress:
- Standalone backtest engine (done, baseline).
- Walk-forward CLI command.
- Independent indicator verification against reference.
- Longer grading window / score drift detection.

P2 (portfolio/strategy intelligence):
- Risk aggregation across markets (gross/net exposure, correlation).
- Score drift detection and reporting.
- Liquidity/spread constraints in sizing.

P3 (UX/operations):
- Health metrics endpoint/log.
- Config migration path.
- Diagnostic CLI (health, validate-config).

P4 (optional):
- Additional providers.
- Separate web dashboard (only if justified).
