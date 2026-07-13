"""ArUco table-plane Real2Sim mapping probe (v0) -- checks calibration without touching Panda.

  frame (image-path or webcam/camera-url)
  -> ONNXYOLODetector.detect()
  -> RecyclableObjectMapper.select_recyclable_by_target()
  -> ArUcoTableMapper.map_detection()  (marker detection + homography)
  -> print mapped position + full mapping debug breakdown
  -> (optional) save debug image (markers + table polygon + bbox)

No PyBullet, no policy execution, no SafetyGate, no recording here --
tape ArUco markers 0-3 (see benchmark/generate_aruco_markers.py) to the
four corners of the table's work area, then use this probe to confirm
all four are detected and the homography maps a test object sensibly
before wiring it into run_full_recycling_cell_demo.py --real2sim-mode aruco.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.run_full_recycling_cell_demo import load_webcam_frame
from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from perception.onnx_yolo_detector import ONNXYOLODetector
from real2sim.aruco_table_mapper import ArUcoTableMapper, draw_aruco_debug_image, print_aruco_mapping_debug
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import save_rgb_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"
DEFAULT_ARUCO_CALIBRATION = "configs/real2sim_aruco_table_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--aruco-calibration", type=str, default=DEFAULT_ARUCO_CALIBRATION)

    parser.add_argument("--image-source", choices=["image", "webcam"], default="image")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--webcam-warmup-frames", type=int, default=10)
    parser.add_argument("--save-debug-image", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/webcam")
    parser.add_argument("--marker-only", action="store_true")

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_marker_only(args: argparse.Namespace, sim_mapper: ArUcoTableMapper, frame: np.ndarray) -> None:
    """--marker-only: marker detection only, no YOLO, no object mapping."""
    marker_detections = sim_mapper.detect_markers(frame)
    detected_marker_ids = sorted(marker_detections.keys())
    required_marker_ids = list(sim_mapper.required_marker_ids)
    missing_marker_ids = sorted(m for m in required_marker_ids if m not in marker_detections)
    marker_centers_px = {str(marker_id): info["center"] for marker_id, info in marker_detections.items()}

    print("=== ArUco Markers ===")
    print(f"detected_marker_ids: {detected_marker_ids}")
    print(f"required_marker_ids: {required_marker_ids}")
    print(f"missing_marker_ids: {missing_marker_ids}")
    print(f"marker_centers_px: {marker_centers_px}")

    if args.save_debug_image:
        debug_image = draw_aruco_debug_image(
            frame,
            marker_detections,
            required_marker_ids,
            detection=None,
            mapped_position=None,
            task_goal=None,
            summary="mode=marker_only",
            draw_table_polygon=sim_mapper.debug_config.get("draw_table_polygon", True),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = resolve(args.output_dir) / f"aruco_marker_only_probe_{timestamp}.jpg"
        saved_path = save_rgb_image(debug_image, str(output_path))
        print(f"\nSaved debug image to: {saved_path}")

    if missing_marker_ids:
        print("\nFAIL")
    else:
        print("\nPASS")


def main() -> None:
    args = parse_args()

    try:
        sim_mapper = ArUcoTableMapper(resolve(args.aruco_calibration))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ArUco mapper setup failed: {exc}")
        print("FAIL")
        return

    task_goal = None
    if not args.marker_only:
        task_goal_parser = RuleBasedTaskGoalParser()
        task_goal = task_goal_parser.parse(args.instruction)
        if task_goal is None:
            print(f"Could not parse instruction: {args.instruction!r}")
            print("Supported objects: plastic bottle (플라스틱 병/페트병/병), plastic cup (컵/플라스틱 컵).")
            print("Supported bins: plastic bin (플라스틱 수거함/플라스틱 통).")
            return

        print("=== TaskGoal ===")
        print(task_goal)

        model_path = resolve(args.model_path)
        if not model_path.exists():
            print(f"ONNX model not found: {model_path}")
            print("Run tools/export_yolo_to_onnx.py first to create it.")
            return

    if args.image_source == "image":
        if not args.image_path:
            print("--image-path is required when --image-source image")
            return

        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            print("Check --image-path and try again.")
            return

        frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    else:
        frame = load_webcam_frame(args)
        if frame is None:
            print("FAIL")
            return

    print(f"frame shape: {frame.shape}")
    print(f"frame dtype: {frame.dtype}")

    if args.marker_only:
        run_marker_only(args, sim_mapper, frame)
        return

    detector = ONNXYOLODetector(model_path=str(model_path), confidence_threshold=args.confidence_threshold)
    detections = detector.detect(frame)
    print("=== Detections ===")
    print(detections)

    if not detections:
        print("No detections found in the frame. Try a lower --confidence-threshold or a different input.")
        print("FAIL")
        return

    recyclable_mapper = RecyclableObjectMapper()
    best = recyclable_mapper.select_recyclable_by_target(detections, task_goal.target_object)
    if best is None:
        print(f"No detection matching TaskGoal.target_object={task_goal.target_object!r} was found.")
        print("Try a different image, a lower --confidence-threshold, or a different --instruction.")
        print("FAIL")
        return

    detection, sim_object_type = best
    print("=== Selected Target ===")
    print(f"{detection.label} (confidence={detection.confidence:.2f}) -> {sim_object_type}")

    marker_detections = sim_mapper.detect_markers(frame)
    print("=== ArUco Markers ===")
    print(f"detected_marker_ids: {sorted(marker_detections.keys())}")

    mapped_position, mapping_debug = sim_mapper.map_detection(detection, frame)

    print()
    print_aruco_mapping_debug(mapping_debug)

    if args.save_debug_image:
        summary = f"mode={mapping_debug['mapping_mode']}"
        if mapping_debug.get("out_of_bounds"):
            summary += " (OUT OF BOUNDS)"
        # Show the raw computed position even when rejected, so the debug
        # image explains *why* -- the caller-facing mapped_position is
        # None on reject, but mapped_position_raw is still informative.
        display_position = mapping_debug.get("mapped_position_raw", mapped_position)
        debug_image = draw_aruco_debug_image(
            frame,
            marker_detections,
            sim_mapper.required_marker_ids,
            detection=detection,
            mapped_position=display_position,
            task_goal=task_goal,
            summary=summary,
            draw_table_polygon=sim_mapper.debug_config.get("draw_table_polygon", True),
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = resolve(args.output_dir) / f"aruco_mapping_probe_{timestamp}.jpg"
        saved_path = save_rgb_image(debug_image, str(output_path))
        print(f"\nSaved debug image to: {saved_path}")

    if mapped_position is None:
        print("\nFAIL")
        return

    print("\nPASS")


if __name__ == "__main__":
    main()
