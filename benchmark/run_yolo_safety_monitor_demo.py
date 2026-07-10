"""YOLOSafetyMonitor smoke test against SimCameraSource or WebcamSource.

  FrameSource.get_frame() -> YOLOSafetyMonitor.check(frame) -> SafetyDecision

No YOLO training, no ONNX export, no TensorRT conversion, no Real2Sim
mapping, no ROS 2, no OpenVLA here -- just confirms a pretrained YOLO
model plugs into the existing SafetyMonitor interface end to end.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.run_webcam_capture import WEBCAM_FAILURE_MESSAGE
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_backend import PyBulletBackend
from safety.yolo_safety_monitor import YOLOSafetyMonitor
from vision.sim_camera_source import SimCameraSource
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "yolo_safety_debug.png"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sim", "webcam"], default="sim")
    parser.add_argument("--model-path", type=str, default="yolo26n.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--hazard-labels", type=str, default="person")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--save-debug-image", action="store_true")
    return parser.parse_args()


def draw_debug_image(frame: np.ndarray, detections: list) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(y1 - 12, 0)), f"{det['label']} {det['confidence']:.2f}", fill=(255, 0, 0))
    return np.array(image)


def main() -> None:
    args = parse_args()
    hazard_labels = {label.strip() for label in args.hazard_labels.split(",") if label.strip()}

    backend = None
    frame_source = None
    try:
        if args.source == "sim":
            backend = PyBulletBackend(gui=True)
            state = backend.reset()
            print("=== Reset State ===")
            print(state)
            frame_source = SimCameraSource(physics_client_id=backend.client_id)
        else:
            try:
                frame_source = WebcamSource(camera_index=args.camera_index)
            except RuntimeError:
                print(WEBCAM_FAILURE_MESSAGE)
                return

        frame = frame_source.get_frame()
        print(f"frame shape: {frame.shape}")
        print(f"frame dtype: {frame.dtype}")

        monitor = YOLOSafetyMonitor(
            model_path=args.model_path,
            hazard_labels=hazard_labels,
            confidence_threshold=args.confidence_threshold,
        )
        decision = monitor.check(frame)

        print(f"detections: {decision.detections}")
        print(f"emergency_stop: {decision.emergency_stop}")
        print(f"reason: {decision.reason}")

        if args.save_debug_image:
            debug_image = draw_debug_image(frame, decision.detections)
            saved_path = save_rgb_image(debug_image, str(DEBUG_IMAGE_PATH))
            print(f"Saved debug image to: {saved_path}")

        if args.source == "sim" and KEEP_GUI_OPEN:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        if frame_source is not None:
            frame_source.close()
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
