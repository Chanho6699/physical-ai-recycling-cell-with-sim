"""Real webcam capture smoke test.

  WebcamSource -> get_frame() -> RGB image -> results/camera/webcam_capture.png

No YOLO, no ONNX/TensorRT, no Real2Sim mapping, no PyBullet object spawn,
no OpenVLA here -- just confirms a real webcam source returns a usable RGB
frame within this project's FrameSource structure.
"""

import argparse
from pathlib import Path

import cv2

from robot_sim.camera_utils import save_rgb_image
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "camera" / "webcam_capture.png"

WEBCAM_FAILURE_MESSAGE = """Webcam capture failed.

If you are running inside WSL:
1. Check whether /dev/video0 exists:
   ls /dev/video*

2. If no device exists, run this webcam test from Windows Python,
   or configure USB/IP camera passthrough.

This failure does not block the PyBullet virtual camera pipeline.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = None
    try:
        source = WebcamSource(camera_index=args.camera_index, width=args.width, height=args.height)

        frame = source.get_frame()
        print(f"Captured frame shape: {frame.shape}")
        print(f"Captured frame dtype: {frame.dtype}")

        saved_path = save_rgb_image(frame, args.output)
        print(f"Saved webcam image to: {saved_path}")

        if args.preview:
            try:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Webcam Capture Preview", frame_bgr)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Preview window could not be shown ({exc}); skipping preview.")
    except RuntimeError:
        print(WEBCAM_FAILURE_MESSAGE)
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    main()
