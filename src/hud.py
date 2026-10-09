from __future__ import annotations

import cv2
import numpy as np

from src.config import Config
from src.gestures import GestureState
from src.tracker import HAND_CONNECTIONS, TrackedHand

GREEN = (80, 220, 80)
RED = (40, 40, 240)
ORANGE = (40, 160, 255)
CYAN = (255, 220, 80)
WHITE = (240, 240, 240)
YELLOW = (0, 220, 255)
DARK = (18, 18, 18)


def draw(
    frame: np.ndarray,
    hands: list[TrackedHand],
    gesture: GestureState,
    fps: float,
    config: Config,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for hand in hands:
        color = CYAN if hand.handedness.lower() == "left" else ORANGE
        _draw_hand(overlay, hand, w, h, color)

    if gesture.left and gesture.right:
        p1 = gesture.left.center_px(w, h)
        p2 = gesture.right.center_px(w, h)
        axis_color = RED if gesture.brake else GREEN
        cv2.line(overlay, p1, p2, axis_color, 4)
        cv2.circle(overlay, p1, 8, CYAN, -1)
        cv2.circle(overlay, p2, 8, ORANGE, -1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.circle(overlay, mid, 6, WHITE, -1)

    _draw_wheel(overlay, gesture, config)
    _draw_status(overlay, gesture, fps, config)
    if gesture.brake:
        _draw_stop_banner(overlay)

    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def _draw_hand(
    frame: np.ndarray, hand: TrackedHand, width: int, height: int, color: tuple[int, int, int]
) -> None:
    pts = [
        (int(lm[0] * width), int(lm[1] * height)) for lm in hand.landmarks
    ]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, WHITE, -1)
    label = f"{hand.handedness} {hand.score:.2f}"
    cv2.putText(
        frame, label, (pts[0][0] - 20, pts[0][1] + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


def _draw_wheel(frame: np.ndarray, gesture: GestureState, config: Config) -> None:
    h, w = frame.shape[:2]
    cx, cy, radius = w - 130, 150, 78
    cv2.circle(frame, (cx, cy), radius, WHITE, 3)
    cv2.circle(frame, (cx, cy), 8, WHITE, -1)

    angle = 0.0 if gesture.angle_deg is None else gesture.angle_deg
    if gesture.steer_left:
        color = ORANGE
    elif gesture.steer_right:
        color = CYAN
    elif gesture.brake:
        color = RED
    else:
        color = GREEN

    rad = np.radians(angle)
    dx = int(np.cos(rad) * (radius - 12))
    dy = int(np.sin(rad) * (radius - 12))
    cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 6)
    cv2.putText(
        frame, f"{angle:+.1f} deg", (cx - 70, cy + radius + 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"deadzone +/-{config.steer_deadzone_deg:.0f}",
        (cx - 78, cy + radius + 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA,
    )


def _draw_status(
    frame: np.ndarray, gesture: GestureState, fps: float, config: Config
) -> None:
    keys = [
        (f"ACCELERATING - {config.key_accel.upper()}", gesture.accelerate and not gesture.brake),
        (f"STEERING LEFT - {config.key_left.upper()}", gesture.steer_left),
        (f"STEERING RIGHT - {config.key_right.upper()}", gesture.steer_right),
        (f"STOP - {config.key_brake.upper()}", gesture.brake),
    ]
    x, y = 16, 36
    cv2.rectangle(frame, (8, 8), (430, 168), DARK, -1)
    cv2.putText(
        frame, f"FPS {fps:.0f}", (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA,
    )
    for i, (label, active) in enumerate(keys):
        color = GREEN if active else (90, 90, 90)
        prefix = "[ON]  " if active else "[off] "
        cv2.putText(
            frame, prefix + label, (x, y + 32 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA,
        )


def _draw_stop_banner(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h // 2 - 50), (w, h // 2 + 50), RED, -1)
    cv2.putText(
        frame, "STOP / BRAKING", (w // 2 - 220, h // 2 + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 4, cv2.LINE_AA,
    )
