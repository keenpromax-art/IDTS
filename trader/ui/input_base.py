from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod


class InputProvider(ABC):
    """Cross-platform source of key presses for the dashboard hotkey loop."""

    @abstractmethod
    def poll(self) -> Optional[bytes]:
        """Return one key press as bytes, or None if none available."""
