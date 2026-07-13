"""External-camera hand/arm intrusion safety monitor probe (v1).

  frame (image-path or webcam/camera-url)
  -> ArUcoTableMapper.detect_markers() (workspace polygon source, optional)
  -> ExternalCameraHandSafetyMonitor.check_frame()
  -> print hand_detected/hand_in_workspace/safety_decision
  -> (optional) save debug image

No YOLO, no PyBullet, no policy execution, no episode recording here --
confirms the real hand/arm intrusion detector (MediaPipe HandLandmarker)
and the ArUco-derived workspace polygon work correctly in isolation,
before wiring them into run_full_recycling_cell_demo.py --safety-mode
pause-resume --hand-safety-source external-camera.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.run_full_recycling_cell_demo import load_webcam_frame
from real2sim.aruco_table_mapper import ArUcoTableMapper
from safety.external_camera_hand_monitor import ExternalCameraHandSafetyMonitor, save_hand_safety_debug_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_SAFETY_CONFIG = "configs/hand_safety_config.json"
DEFAULT_ARUCO_CALIBRATION = "configs/real2sim_aruco_table_calibration.json"
DEBUG_OUTPUT_DIR = "results/safety_hand_debug"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-source", choices=["image", "webcam"], default="image")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--webcam-warmup-frames", type=int, default=10)
    parser.add_argument("--hand-safety-config", type=str, default=DEFAULT_HAND_SAFETY_CONFIG)
    parser.add_argument("--aruco-calibration", type=str, default=DEFAULT_ARUCO_CALIBRATION)
    parser.add_argument("--save-debug-image", action="store_true")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_workspace_polygon(args, frame: np.ndarray):
    """4 ArUco marker centers (perimeter order) as the workspace polygon,
    or None (letting ExternalCameraHandSafetyMonitor fall back to its
    own config ROI) if the calibration/markers aren't available."""
    try:
        aruco_mapper = ArUcoTableMapper(resolve(args.aruco_calibration))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ArUco mapper setup failed, falling back to hand-safety-config ROI if enabled: {exc}")
        return None

    marker_detections = aruco_mapper.detect_markers(frame)
    detected_marker_ids = sorted(marker_detections.keys())
    required_marker_ids = list(aruco_mapper.required_marker_ids)
    missing_marker_ids = [marker_id for marker_id in required_marker_ids if marker_id not in marker_detections]

    print("=== ArUco Workspace Markers ===")
    print(f"detected_marker_ids: {detected_marker_ids}")
    print(f"required_marker_ids: {required_marker_ids}")
    if missing_marker_ids:
        print(f"missing_marker_ids: {missing_marker_ids}")
        return None

    return [marker_detections[marker_id]["center"] for marker_id in required_marker_ids]


def main() -> None:
    args = parse_args()

    print("=== External Camera Hand Safety Probe ===")

    if args.image_source == "image":
        if not args.image_path:
            print("--image-path is required when --image-source image")
            return
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return
        frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    else:
        frame = load_webcam_frame(args)
        if frame is None:
            print("FAIL")
            return

    print(f"frame shape: {frame.shape}")

    workspace_polygon_px = build_workspace_polygon(args, frame)

    try:
        monitor = ExternalCameraHandSafetyMonitor(config_path=resolve(args.hand_safety_config))
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc))
        print("FAIL")
        return

    monitor.set_workspace_polygon(workspace_polygon_px)
    decision, debug = monitor.check_frame(frame, step_index=0)

    print(f"workspace_valid: {debug['workspace_valid']}")
    print(f"hand_detected: {debug['hand_detected']}")
    print(f"hand_in_workspace: {debug['hand_in_workspace']}")
    print(f"safety_decision: emergency_stop={decision.emergency_stop}")
    print(f"reason: {decision.reason}")

    if args.save_debug_image:
        saved_path = save_hand_safety_debug_image(frame, debug, resolve(DEBUG_OUTPUT_DIR))
        print(f"Saved debug image to: {saved_path}")

    print("PASS")


if __name__ == "__main__":
    main()
