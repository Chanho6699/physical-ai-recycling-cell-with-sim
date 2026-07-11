"""Natural-language TaskGoal + Image-based Real2Sim target selection demo (v0).

  instruction (Korean)
  -> RuleBasedTaskGoalParser -> TaskGoal
  -> image file -> ONNXYOLODetector -> detections
  -> RecyclableObjectMapper.select_recyclable_by_target(detections, TaskGoal.target_object)
  -> bbox center -> ImageToSimMapper -> PyBullet position
  -> PyBulletBackend.set_object_type()/set_object_position()
  -> run_dynamic_pick_place() (robot_sim.pick_place_policy)

No real LLM API, no VLA action generation, no Panda URDF backend, no
ROS 2, no TensorRT, no Isaac Sim, no latency benchmark here yet.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from perception.detection_types import Detection
from perception.onnx_yolo_detector import ONNXYOLODetector
from real2sim.image_to_sim_mapper import ImageToSimMapper
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pick_place_policy import run_dynamic_pick_place
from robot_sim.pybullet_backend import PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "task_goal_real2sim_debug.png"

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--save-debug-image", action="store_true")
    parser.add_argument("--gui", action="store_true", default=True)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def draw_debug_image(frame: np.ndarray, detection: Detection, sim_position: list, task_goal) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = detection.bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    center_x, center_y = detection.center_xy
    radius = 5
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=(255, 255, 0),
    )

    label_text = f"{detection.label} {detection.confidence:.2f}"
    goal_text = f"goal: {task_goal.target_object} -> {task_goal.target_bin}"
    sim_text = f"sim_pos=({sim_position[0]:.2f}, {sim_position[1]:.2f}, {sim_position[2]:.2f})"

    draw.text((x1, max(y1 - 40, 0)), goal_text, fill=(255, 0, 0))
    draw.text((x1, max(y1 - 22, 0)), label_text, fill=(255, 0, 0))
    draw.text((x1, min(y2 + 4, image.height - 12)), sim_text, fill=(255, 0, 0))

    return np.array(image)


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

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        print("Check --image-path and try again.")
        return

    model_path = resolve(args.model_path)
    if not model_path.exists():
        print(f"ONNX model not found: {model_path}")
        print("Run tools/export_yolo_to_onnx.py first to create it.")
        return

    frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    print(f"frame shape: {frame.shape}")
    print(f"frame dtype: {frame.dtype}")

    detector = ONNXYOLODetector(model_path=str(model_path), confidence_threshold=args.confidence_threshold)
    detections = detector.detect(frame)
    print(f"detections: {detections}")

    if not detections:
        print("No detections found in the image. Try a lower --confidence-threshold or a different image.")
        return

    recyclable_mapper = RecyclableObjectMapper()
    best = recyclable_mapper.select_recyclable_by_target(detections, task_goal.target_object)
    if best is None:
        print(f"No detection matching TaskGoal.target_object={task_goal.target_object!r} was found.")
        print("Try a different image, a lower --confidence-threshold, or a different --instruction.")
        return

    detection, sim_object_type = best
    print(
        f"Selected candidate matching TaskGoal: {detection.label} "
        f"(confidence={detection.confidence:.2f}) -> {sim_object_type}"
    )

    image_height, image_width = frame.shape[:2]
    sim_mapper = ImageToSimMapper(image_width=image_width, image_height=image_height)
    center_x, center_y = detection.center_xy
    sim_position = sim_mapper.image_point_to_sim_position(center_x, center_y)
    print(f"Mapped sim position: {sim_position}")

    backend = PyBulletBackend(gui=args.gui)
    try:
        state = backend.reset()
        print("=== Reset State ===")
        print(state)

        backend.set_object_type(sim_object_type)
        state = backend.set_object_position(sim_position)
        print("=== State After Real2Sim Mapping ===")
        print(state)

        if args.save_debug_image:
            debug_image = draw_debug_image(frame, detection, sim_position, task_goal)
            saved_path = save_rgb_image(debug_image, str(DEBUG_IMAGE_PATH))
            print(f"Saved debug image to: {saved_path}")

        final_state = run_dynamic_pick_place(backend)
        final_status = final_state["task_status"]
        print(f"\n=== Demo finished: task_status={final_status} ===")

        if KEEP_GUI_OPEN and args.gui:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
