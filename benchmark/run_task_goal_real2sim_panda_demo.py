"""Natural-language TaskGoal + Image-based Real2Sim -> Franka Panda pick-and-place (v1).

  instruction (Korean)
  -> RuleBasedTaskGoalParser -> TaskGoal
  -> image file -> ONNXYOLODetector -> detections
  -> RecyclableObjectMapper.select_recyclable_by_target(detections, TaskGoal.target_object)
  -> bbox center -> ImageToSimMapper (Panda workspace ranges) -> Panda sim position
  -> PyBulletPandaBackend.set_object_type()/set_object_position()
  -> move_end_effector_to() / close_gripper() / move_end_effector_to() / open_gripper()
     (NOT robot_sim.pick_place_policy.run_dynamic_pick_place -- that one
     builds delta RobotCommands for the simple sphere backend; the Panda
     backend is driven directly through its own IK-based methods)

No VLA action pipeline changes, no LeRobot, no ROS 2, no TensorRT, no
Isaac Sim, no OpenVLA fine-tuning, no Safety+Panda integration here yet.
"""

import argparse
import math
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
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "task_goal_real2sim_panda_debug.png"

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

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    parser.add_argument("--object-z", type=float, default=0.05)
    parser.add_argument("--bin-clearance", type=float, default=0.05)

    parser.add_argument("--sim-x-min", type=float, default=0.25)
    parser.add_argument("--sim-x-max", type=float, default=0.55)
    parser.add_argument("--sim-y-min", type=float, default=-0.25)
    parser.add_argument("--sim-y-max", type=float, default=0.25)

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
    sim_text = f"panda_sim_pos=({sim_position[0]:.2f}, {sim_position[1]:.2f}, {sim_position[2]:.2f})"

    draw.text((x1, max(y1 - 40, 0)), goal_text, fill=(255, 0, 0))
    draw.text((x1, max(y1 - 22, 0)), label_text, fill=(255, 0, 0))
    draw.text((x1, min(y2 + 4, image.height - 12)), sim_text, fill=(255, 0, 0))

    return np.array(image)


def print_grasp_diagnostics(state: dict) -> None:
    ee_position = state["end_effector_position"]
    object_position = state["object_position"]
    distance = math.sqrt(sum((ee_position[i] - object_position[i]) ** 2 for i in range(3)))

    print("Grasp diagnostics:")
    print(f"  end_effector_position: {ee_position}")
    print(f"  object_position: {object_position}")
    print(f"  distance: {distance:.4f}")
    print(f"  gripper_width: {state['gripper_width']:.4f}")
    print(f"  last_event: {state['last_event']}")


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
    print("=== Detections ===")
    print(detections)

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
    print("=== Selected Target ===")
    print(
        f"{detection.label} (confidence={detection.confidence:.2f}) -> {sim_object_type}"
    )

    image_height, image_width = frame.shape[:2]
    sim_mapper = ImageToSimMapper(
        image_width=image_width,
        image_height=image_height,
        sim_x_range=(args.sim_x_min, args.sim_x_max),
        sim_y_range=(args.sim_y_min, args.sim_y_max),
        object_z=args.object_z,
    )
    center_x, center_y = detection.center_xy
    sim_position = sim_mapper.image_point_to_sim_position(center_x, center_y)
    print("=== Mapped Panda Sim Position ===")
    print(sim_position)

    backend = PyBulletPandaBackend(gui=args.gui)
    try:
        try:
            state = backend.reset()
        except Exception as exc:
            print(f"Panda backend reset failed: {exc}")
            return

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

        print("\n=== Move Panda to Object ===")
        state = backend.move_end_effector_to(sim_position)
        print(state)

        print("\n=== Close Gripper ===")
        state = backend.close_gripper()
        print(state)

        if not state["held_object"]:
            print_grasp_diagnostics(state)

        bin_position = state["bin_position"]
        bin_target = [bin_position[0], bin_position[1], bin_position[2] + args.bin_clearance]

        print("\n=== Move Panda to Bin ===")
        state = backend.move_end_effector_to(bin_target)
        print(state)

        print("\n=== Open Gripper ===")
        state = backend.open_gripper()
        print(state)

        print("\n=== Final State ===")
        print(state)

        final_status = state["task_status"]
        print(f"\n=== Demo finished: task_status={final_status} ===")
        if final_status == "success":
            print("PASS")
        else:
            print("FAIL")
            print_grasp_diagnostics(state)

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
