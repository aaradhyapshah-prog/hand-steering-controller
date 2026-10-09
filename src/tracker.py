from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np

from src.config import HAND_LANDMARKER_MODEL, HAND_LANDMARKER_URL, MODEL_DIR, Config

WRIST = 0
MIDDLE_MCP = 9

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class TrackedHand:
    handedness: str
    landmarks: np.ndarray
    score: float

    def center_px(self, width: int, height: int) -> tuple[int, int]:
        wrist = self.landmarks[WRIST]
        mcp = self.landmarks[MIDDLE_MCP]
        x = (wrist[0] + mcp[0]) * 0.5 * width
        y = (wrist[1] + mcp[1]) * 0.5 * height
        return int(x), int(y)


def ensure_hand_model(path: Path = HAND_LANDMARKER_MODEL) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand landmarker model to {path} ...")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
    return path


class HandTracker:
    """Hands tracker: MediaPipe Tasks API, with Solutions fallback on older packages."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.backend = "tasks"
        self._landmarker = None
        self._solutions_hands = None
        self._timestamp_ms = 0
        try:
            self._init_tasks()
        except Exception as exc:
            print(f"Tasks API unavailable ({exc}). Falling back to Solutions API.")
            self._init_solutions()

    def _init_tasks(self) -> None:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ensure_hand_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=self.config.max_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self.backend = "tasks"

    def _init_solutions(self) -> None:
        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "hands"):
            raise RuntimeError(
                "MediaPipe Solutions Hands is not installed. "
                "Tasks API is required on MediaPipe 1.x."
            )
        self._solutions_hands = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.backend = "solutions"

    def detect(self, bgr: np.ndarray, dt_ms: int) -> list[TrackedHand]:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        if self.backend == "tasks":
            hands = self._detect_tasks(rgb, dt_ms)
        else:
            hands = self._detect_solutions(rgb)
        if self.config.mirror:
            for hand in hands:
                if hand.handedness.lower() == "left":
                    hand.handedness = "Right"
                elif hand.handedness.lower() == "right":
                    hand.handedness = "Left"
        return hands

    def _detect_tasks(self, rgb: np.ndarray, dt_ms: int) -> list[TrackedHand]:
        self._timestamp_ms += max(int(dt_ms), 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        hands: list[TrackedHand] = []
        if not result.hand_landmarks:
            return hands
        for i, landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[i][0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            hands.append(
                TrackedHand(
                    handedness=category.category_name,
                    landmarks=pts,
                    score=float(category.score),
                )
            )
        return hands

    def _detect_solutions(self, rgb: np.ndarray) -> list[TrackedHand]:
        result = self._solutions_hands.process(rgb)
        hands: list[TrackedHand] = []
        if not result.multi_hand_landmarks:
            return hands
        for landmarks, handedness in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
            )
            category = handedness.classification[0]
            hands.append(
                TrackedHand(
                    handedness=category.label,
                    landmarks=pts,
                    score=float(category.score),
                )
            )
        return hands

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._solutions_hands is not None:
            self._solutions_hands.close()
            self._solutions_hands = None
