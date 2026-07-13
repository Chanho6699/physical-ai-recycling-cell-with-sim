"""Real2Sim mapping probe (v1) -- checks calibration without touching Panda.

  frame (image-path or webcam/camera-url)
  -> ONNXYOLODetector.detect()
  -> RecyclableObjectMapper.select_recyclable_by_target()
  -> CalibratedImageToSimMapper.map_bbox_to_sim()
  -> print mapped position + full mapping debug breakdown
  -> (optional) save debug image

No PyBullet, no policy execution, no SafetyGate, no recording here --
this only exists to let the image_roi / axis_mapping / sim_workspace in
configs/real2sim_webcam_calibration.json be tuned quickly (e.g. by moving
the same object closer/farther/left/right and comparing mapped_position
across runs) without waiting on a full pick-and-place run each time.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.run_full_recycling_cell_demo import load_webcam_frame
from benchmark.run_task_goal_real2sim_panda_interrupt_demo import draw_debug_image
from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from perception.onnx_yolo_detector import ONNXYOLODetector
from real2sim.calibrated_image_to_sim_mapper import (
    CalibratedImageToSimMapper,
    draw_roi_rectangle,
    print_mapping_debug,
)
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import save_rgb_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"
DEFAULT_CALIBRATION_CONFIG = "configs/real2sim_webcam_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--real2sim-calibration", type=str, default=DEFAULT_CALIBRATION_CONFIG)

    parser.add_argument("--image-source", choices=["image", "webcam"], default="image")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--webcam-warmup-frames", type=int, default=10)
    parser.add_argument("--save-debug-image", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/webcam")

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

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
    print(f"bbox_xyxy: {detection.bbox_xyxy}")

    image_height, image_width = frame.shape[:2]
    sim_mapper = CalibratedImageToSimMapper.from_config_file(resolve(args.real2sim_calibration))
    mapped_position, mapping_debug = sim_mapper.map_bbox_to_sim(detection.bbox_xyxy, image_width, image_height)

    print()
    print_mapping_debug(mapping_debug)

    if args.save_debug_image:
        summary = f"mode={mapping_debug['mapping_mode']}"
        frame_with_roi = draw_roi_rectangle(frame, mapping_debug["image_roi"])
        debug_image = draw_debug_image(frame_with_roi, detection, mapped_position, task_goal, summary)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = resolve(args.output_dir) / f"real2sim_mapping_probe_{timestamp}.jpg"
        saved_path = save_rgb_image(debug_image, str(output_path))
        print(f"\nSaved debug image to: {saved_path}")

    print("\nPASS")


if __name__ == "__main__":
    main()
