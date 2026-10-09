from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
HAND_LANDMARKER_MODEL = MODEL_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 60
    mirror: bool = True

    max_hands: int = 2
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1

    open_finger_count: int = 5
    finger_extend_margin: float = 0.02
    steer_deadzone_deg: float = 15.0

    key_accel: str = "w"
    key_left: str = "a"
    key_right: str = "d"
    key_brake: str = "s"
    quit_key: str = "q"

    window_name: str = "Hand Steering Controller"
