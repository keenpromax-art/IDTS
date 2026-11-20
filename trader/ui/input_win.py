from __future__ import annotations
import msvcrt
from typing import Optional

from trader.ui.input_base import InputProvider


class WindowsInputProvider(InputProvider):
    """Reads key presses from the Windows console via msvcrt."""

    def poll(self) -> Optional[bytes]:
        if msvcrt.kbhit():
            return msvcrt.getch()
        return None
