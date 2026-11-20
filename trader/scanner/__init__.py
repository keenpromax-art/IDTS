"""Cross-cutting primitives for the ticker scanner.

The modules here are deliberately dependency-light so they can be unit-tested
without a network, a broker, or a running engine. They are additive: existing
modules keep their public signatures and new behaviour is opt-in or defaults
to the legacy behaviour when no configuration is supplied.
"""
