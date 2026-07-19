"""ONNXRuntimeYOLOSafetyMonitor smoke test against SimCameraSource, WebcamSource, or a static image.

  FrameSource.get_frame() -> ONNXRuntimeYOLOSafetyMonitor.check(frame) -> SafetyDecision

No YOLO training, no TensorRT, no Real2Sim mapping, no ROS 2, no OpenVLA
here -- just confirms the ONNX Runtime-based monitor plugs into the same
SafetyMonitor interface as MockSafetyMonitor/YOLOSafetyMonitor.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.run_webcam_capture import WEBCAM_FAILURE_MESSAGE
from benchmark.run_yolo_safety_monitor_demo import draw_debug_image
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_backend import PyBulletBackend
from safety.onnx_yolo_safety_monitor import ONNXRuntimeYOLOSafetyMonitor
from vision.sim_camera_source import SimCameraSource
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "onnx_yolo_safety_debug.png"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sim", "webcam", "image"], default="sim")
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--hazard-labels", type=str, default="person")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--save-debug-image", action="store_true")
    return parser.parse_args()


def get_frame(args: argparse.Namespace, resources: dict) -> np.ndarray:
    if args.source == "sim":
        backend = PyBulletBackend(gui=True)
        resources["backend"] = backend
        state = backend.reset()
        print("=== Reset State ===")
        print(state)
        source = SimCameraSource(physics_client_id=backend.client_id)
        resources["frame_source"] = source
        return source.get_frame()

    if args.source == "webcam":
        source = WebcamSource(camera_index=args.camera_index)
        resources["frame_source"] = source
        return source.get_frame()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()

    if args.source == "image" and not args.image_path:
        print("--image-path is required when --source image")
        return

    hazard_labels = {label.strip() for label in args.hazard_labels.split(",") if label.strip()}

    resources: dict = {}
    try:
        try:
            frame = get_frame(args, resources)
        except RuntimeError:
            print(WEBCAM_FAILURE_MESSAGE)
            return
        except FileNotFoundError as exc:
            print(exc)
            print("Check --image-path and try again.")
            return

        print(f"frame shape: {frame.shape}")
        print(f"frame dtype: {frame.dtype}")

        monitor = ONNXRuntimeYOLOSafetyMonitor(
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
        frame_source = resources.get("frame_source")
        if frame_source is not None:
            frame_source.close()
        backend = resources.get("backend")
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
