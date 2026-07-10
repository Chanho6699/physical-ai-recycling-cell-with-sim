"""ONNX YOLO smoke test.

  weights/yolo26n.onnx -> YOLO(...).predict(frame) -> detections

Confirms an ONNX-exported model (see tools/export_yolo_to_onnx.py) loads
and runs inference correctly, before it's wired into ONNX Model Evaluator
or a future ONNXRuntime-based SafetyMonitor (not built here yet).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from benchmark.run_webcam_capture import WEBCAM_FAILURE_MESSAGE
from benchmark.run_yolo_safety_monitor_demo import draw_debug_image
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_backend import PyBulletBackend
from vision.sim_camera_source import SimCameraSource
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "yolo_onnx_smoke_debug.png"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--source", choices=["sim", "webcam", "image"], default="sim")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--save-debug-image", action="store_true")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


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

    image = Image.open(args.image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()

    if args.source == "image" and not args.image_path:
        print("--image-path is required when --source image")
        return

    model_path = resolve(args.model_path)
    if not model_path.exists():
        print(f"ONNX model not found: {model_path}")
        print("Run tools/export_yolo_to_onnx.py first to create it.")
        return

    resources: dict = {}
    try:
        try:
            frame = get_frame(args, resources)
        except RuntimeError:
            print(WEBCAM_FAILURE_MESSAGE)
            return

        print(f"frame shape: {frame.shape}")
        print(f"frame dtype: {frame.dtype}")

        model = YOLO(str(model_path))

        start = time.perf_counter()
        results = model.predict(frame, verbose=False)[0]
        inference_ms = (time.perf_counter() - start) * 1000

        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < args.confidence_threshold:
                continue
            label = results.names[int(box.cls[0])]
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                {"label": label, "confidence": confidence, "bbox_xyxy": [x1, y1, x2, y2]}
            )

        print(f"detections: {detections}")
        print(f"inference_ms: {inference_ms:.2f}")

        if args.save_debug_image:
            debug_image = draw_debug_image(frame, detections)
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
