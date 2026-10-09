from __future__ import annotations

import sys

import cv2
import numpy as np

from src.config import Config


class Camera:
    def __init__(self, config: Config) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(config.camera_index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {config.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.mirror = config.mirror

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
