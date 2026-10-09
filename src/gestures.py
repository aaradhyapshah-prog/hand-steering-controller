from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.tracker import TrackedHand

TIP_IDS = (4, 8, 12, 16, 20)
PIP_IDS = (3, 6, 10, 14, 18)
MCP_IDS = (2, 5, 9, 13, 17)


@dataclass
class GestureState:
    brake: bool = False
    accelerate: bool = False
    steer_left: bool = False
    steer_right: bool = False
    angle_deg: float | None = None
    left: TrackedHand | None = None
    right: TrackedHand | None = None
    left_open: bool = False
    right_open: bool = False
    hands_detected: int = 0


def count_extended_fingers(hand: TrackedHand, margin: float) -> int:
    lm = hand.landmarks
    count = 0
    if _thumb_extended(lm, hand.handedness, margin):
        count += 1
    for tip, pip, mcp in zip(TIP_IDS[1:], PIP_IDS[1:], MCP_IDS[1:]):
        if lm[tip, 1] < lm[pip, 1] - margin and lm[tip, 1] < lm[mcp, 1]:
            count += 1
    return count


def _thumb_extended(lm: np.ndarray, handedness: str, margin: float) -> bool:
    tip_x, ip_x, mcp_x = lm[4, 0], lm[3, 0], lm[2, 0]
    if handedness.lower().startswith("right"):
        return tip_x < ip_x - margin and tip_x < mcp_x
    return tip_x > ip_x + margin and tip_x > mcp_x


def steering_angle_deg(left: TrackedHand, right: TrackedHand) -> float:
    lx, ly = left.landmarks[0, 0], left.landmarks[0, 1]
    rx, ry = right.landmarks[0, 0], right.landmarks[0, 1]
    lmx, lmy = left.landmarks[9, 0], left.landmarks[9, 1]
    rmx, rmy = right.landmarks[9, 0], right.landmarks[9, 1]
    left_x, left_y = (lx + lmx) * 0.5, (ly + lmy) * 0.5
    right_x, right_y = (rx + rmx) * 0.5, (ry + rmy) * 0.5
    dx = right_x - left_x
    dy = right_y - left_y
    return float(np.degrees(np.arctan2(dy, dx)))


def interpret(hands: list[TrackedHand], config: Config) -> GestureState:
    state = GestureState(hands_detected=len(hands))
    by_name = {h.handedness.lower(): h for h in hands}
    state.left = by_name.get("left")
    state.right = by_name.get("right")
    if (state.left is None or state.right is None) and len(hands) == 2:
        ordered = sorted(hands, key=lambda h: h.landmarks[0, 0])
        state.left, state.right = ordered[0], ordered[1]

    if state.left is None or state.right is None:
        return state

    state.left_open = (
        count_extended_fingers(state.left, config.finger_extend_margin)
        >= config.open_finger_count
    )
    state.right_open = (
        count_extended_fingers(state.right, config.finger_extend_margin)
        >= config.open_finger_count
    )
    state.angle_deg = steering_angle_deg(state.left, state.right)

    if state.left_open and state.right_open:
        state.brake = True
        return state

    state.accelerate = True
    angle = state.angle_deg
    if angle <= -config.steer_deadzone_deg:
        state.steer_left = True
    elif angle >= config.steer_deadzone_deg:
        state.steer_right = True
    return state
