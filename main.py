#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.camera import Camera
from src.config import Config
from src.controller import KeyboardController
from src.gestures import interpret
from src.hud import draw
from src.tracker import HandTracker

BANNER = """
============================================================
  Hand Gesture Steering Controller
============================================================
  Webcam: default camera (index 0)
  Quit:   press Q in the video window (keys are released)

  Gestures
  --------
  STOP / BRAKE : both palms fully open (all 5 fingers)
                 -> hold S, release W/A/D
  ACCELERATE   : both hands visible, not in STOP
                 (fists or a normal driving grip) -> hold W
  STEER LEFT   : right hand higher than left (angle <= -deadzone) -> A
  STEER RIGHT  : left hand higher than right (angle >= +deadzone) -> D
  NEUTRAL      : angle inside the deadzone -> release A and D

  Fail-safe: missing a hand automatically releases every key.

  Focus the racing game after this window starts.
============================================================
"""


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Virtual hand-gesture steering for racing games")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--deadzone", type=float, default=15.0, help="Steer deadzone in degrees")
    parser.add_argument("--brake-key", default="s", help="Key held for STOP / brake")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    return Config(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        mirror=not args.no_mirror,
        steer_deadzone_deg=args.deadzone,
        key_brake=args.brake_key,
    )


def main() -> int:
    config = parse_args()
    print(BANNER)
    camera = None
    tracker = None
    keyboard = None
    try:
        camera = Camera(config)
        tracker = HandTracker(config)
        keyboard = KeyboardController(config)
        print(f"Tracker backend: {tracker.backend}")
        print("Running. Press Q in the preview window to quit.\n")

        prev = time.perf_counter()
        fps = 0.0
        while True:
            frame = camera.read()
            if frame is None:
                print("Camera frame dropped; releasing keys.")
                keyboard.release_all()
                continue

            now = time.perf_counter()
            dt_ms = int((now - prev) * 1000)
            instant_fps = 1.0 / max(now - prev, 1e-6)
            fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1
            prev = now

            hands = tracker.detect(frame, dt_ms)
            gesture = interpret(hands, config)
            keyboard.apply(gesture)
            vis = draw(frame, hands, gesture, fps, config)
            cv2.imshow(config.window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        if keyboard is not None:
            keyboard.release_all()
            print("All keys released.")
        if tracker is not None:
            tracker.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
