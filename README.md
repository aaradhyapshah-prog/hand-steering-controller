# Hand Steering Controller

Virtual steering for racing games (F1, etc.) using a webcam, OpenCV, MediaPipe Hands, and `pynput`.

Hold your hands like a steering wheel. Tilt to steer, close into a driving grip to accelerate, and open both palms to brake.

## Setup

A virtual environment is already in `.venv` (Python 3.14 + MediaPipe 1.1 Tasks API).

```powershell
cd $HOME\hand-steering-controller
.\.venv\Scripts\Activate.ps1
python main.py
```

The first run downloads `models/hand_landmarker.task`. Older MediaPipe packages that still ship Solutions Hands are used automatically if Tasks cannot start.

## Gestures

| Gesture | Meaning | Keys |
| --- | --- | --- |
| Both palms open (5 fingers) | STOP / brake | Hold `S`, release `W` `A` `D` |
| Both hands visible, not STOP | Accelerate | Hold `W` |
| Right hand higher (angle ≤ −15°) | Steer left | Hold `A` |
| Left hand higher (angle ≥ +15°) | Steer right | Hold `D` |
| Angle inside ±15° | Neutral steer | Release `A` and `D` |
| One or zero hands | Fail-safe | Release all keys |

Press **Q** in the preview window to quit. All keys are released on exit.

Focus the game after the preview window appears so WASD reaches the title.

## Options

```powershell
python main.py --deadzone 12 --brake-key s --width 1280 --height 720 --fps 60
python main.py --brake-key space
python main.py --camera 1 --no-mirror
```

## Layout

- `main.py` — capture loop, HUD, shutdown
- `src/camera.py` — webcam (low-latency settings)
- `src/tracker.py` — MediaPipe Hands (Tasks, then Solutions)
- `src/gestures.py` — palms, tilt angle, control state
- `src/controller.py` — edge-triggered key down/up
- `src/hud.py` — landmarks, tilt axis, wheel, key states, FPS
- `src/config.py` — thresholds and key map

## Notes

Some games ignore simulated keyboard events. If WASD does nothing, try windowed mode or a title that accepts standard keyboard input. Keep hands well lit and both palms in frame.
