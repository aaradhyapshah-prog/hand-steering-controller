from __future__ import annotations

from pynput.keyboard import Controller, Key, KeyCode

from src.config import Config
from src.gestures import GestureState


class KeyboardController:
    """Send key down/up only when the logical input state changes."""

    def __init__(self, config: Config) -> None:
        self._kb = Controller()
        self._held: set[str] = set()
        self._keys = {
            "w": config.key_accel,
            "a": config.key_left,
            "d": config.key_right,
            "s": config.key_brake,
        }

    def apply(self, gesture: GestureState) -> None:
        desired: set[str] = set()
        if gesture.brake:
            desired.add(self._keys["s"])
        elif gesture.hands_detected == 2 and gesture.left and gesture.right:
            if gesture.accelerate:
                desired.add(self._keys["w"])
            if gesture.steer_left:
                desired.add(self._keys["a"])
            if gesture.steer_right:
                desired.add(self._keys["d"])
        self._sync(desired)

    def release_all(self) -> None:
        self._sync(set())

    def _sync(self, desired: set[str]) -> None:
        for key in self._held - desired:
            self._release(key)
        for key in desired - self._held:
            self._press(key)
        self._held = set(desired)

    def _press(self, key: str) -> None:
        self._kb.press(self._token(key))

    def _release(self, key: str) -> None:
        self._kb.release(self._token(key))

    @staticmethod
    def _token(key: str):
        lowered = key.lower()
        if lowered in {"space", "spacebar"}:
            return Key.space
        if len(key) == 1:
            return KeyCode.from_char(lowered)
        return KeyCode.from_char(lowered[0])
