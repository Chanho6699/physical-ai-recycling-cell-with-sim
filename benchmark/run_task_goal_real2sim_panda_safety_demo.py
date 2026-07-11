"""Safety Gate + TaskGoal + Image-based Real2Sim -> Franka Panda pick-and-place (v1).

Reuses the exact same TaskGoal parsing / detection / target selection /
Real2Sim mapping flow as run_task_goal_real2sim_panda_demo.py. The only
addition: before each of the 4 high-level Panda actions
(move_panda_to_object, close_gripper, move_panda_to_bin, open_gripper),
a SafetyGate.check() call decides whether the action is allowed to run.

  action_name -> safety_frame -> SafetyMonitor.check(frame) -> SafetyGateResult
  allowed=False -> command blocked, demo stops, task_status="blocked_by_safety"
  allowed=True  -> Panda action executes as usual

No interruptible mid-motion safety callback (move_end_effector_to itself
is not interrupted), no VLA pipeline changes, no LeRobot, no ROS 2, no
TensorRT, no Isaac Sim, no OpenVLA fine-tuning here yet.
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
from safety.mock_safety_monitor import MockSafetyMonitor
from safety.safety_gate import SafetyGate
from vision.sim_camera_source import SimCameraSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "task_goal_real2sim_panda_safety_debug.png"

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

    parser.add_argument("--safety-monitor", choices=["mock", "onnx"], default="onnx")
    parser.add_argument("--simulate-hazard", action="store_true")
    parser.add_argument("--safety-image-path", type=str, default=None)
    parser.add_argument("--safety-source", choices=["task_image", "sim_camera", "image"], default="task_image")

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_safety_monitor(args, model_path: Path):
    if args.safety_monitor == "mock":
        return MockSafetyMonitor(simulate_hazard=args.simulate_hazard)

    if args.simulate_hazard:
        print("--simulate-hazard is ignored with --safety-monitor onnx (mock-only option).")

    from safety.onnx_yolo_safety_monitor import ONNXRuntimeYOLOSafetyMonitor

    return ONNXRuntimeYOLOSafetyMonitor(model_path=str(model_path))


def get_safety_frame(args, backend, task_frame: np.ndarray, safety_image_frame) -> np.ndarray:
    if args.safety_source == "task_image":
        return task_frame
    if args.safety_source == "image":
        return safety_image_frame

    # sim_camera
    camera = SimCameraSource(physics_client_id=backend.client_id)
    return camera.get_frame()


def draw_debug_image(
    frame: np.ndarray, detection: Detection, sim_position: list, task_goal, safety_summary: str
) -> np.ndarray:
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

    draw.text((x1, max(y1 - 58, 0)), goal_text, fill=(255, 0, 0))
    draw.text((x1, max(y1 - 40, 0)), label_text, fill=(255, 0, 0))
    draw.text((x1, max(y1 - 22, 0)), sim_text, fill=(255, 0, 0))
    draw.text((x1, min(y2 + 4, image.height - 12)), safety_summary, fill=(255, 128, 0))

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

    if args.safety_source == "image" and not args.safety_image_path:
        print("--safety-image-path is required when --safety-source image")
        return

    safety_image_frame = None
    if args.safety_source == "image":
        safety_image_path = Path(args.safety_image_path)
        if not safety_image_path.exists():
            print(f"Safety image file not found: {safety_image_path}")
            print("Check --safety-image-path and try again.")
            return
        safety_image_frame = np.array(Image.open(safety_image_path).convert("RGB"), dtype=np.uint8)

    task_frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    print(f"frame shape: {task_frame.shape}")
    print(f"frame dtype: {task_frame.dtype}")

    detector = ONNXYOLODetector(model_path=str(model_path), confidence_threshold=args.confidence_threshold)
    detections = detector.detect(task_frame)
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
    print(f"{detection.label} (confidence={detection.confidence:.2f}) -> {sim_object_type}")

    image_height, image_width = task_frame.shape[:2]
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

    safety_monitor = build_safety_monitor(args, model_path)
    safety_gate = SafetyGate(safety_monitor)

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

        bin_position = state["bin_position"]
        bin_target = [bin_position[0], bin_position[1], bin_position[2] + args.bin_clearance]

        actions = [
            ("move_panda_to_object", "Move Panda to Object", lambda: backend.move_end_effector_to(sim_position)),
            ("close_gripper", "Close Gripper", lambda: backend.close_gripper()),
            ("move_panda_to_bin", "Move Panda to Bin", lambda: backend.move_end_effector_to(bin_target)),
            ("open_gripper", "Open Gripper", lambda: backend.open_gripper()),
        ]

        blocked_action = None
        safety_reason = None
        last_gate_result = None

        for action_name, action_label, action_fn in actions:
            safety_frame = get_safety_frame(args, backend, task_frame, safety_image_frame)
            gate_result = safety_gate.check(safety_frame, action_name)
            last_gate_result = gate_result

            print(f"\n=== Safety Check: {action_name} ===")
            print(f"emergency_stop={gate_result.decision.emergency_stop}, reason={gate_result.reason}")

            if not gate_result.allowed:
                print("COMMAND BLOCKED")
                print(gate_result)
                blocked_action = action_name
                safety_reason = gate_result.reason
                break

            print("COMMAND ALLOWED")
            print(f"\n=== {action_label} ===")
            state = action_fn()
            print(state)

            if action_name == "close_gripper" and not state["held_object"]:
                print_grasp_diagnostics(state)

        if blocked_action is not None:
            final_state = dict(state)
            final_state["task_status"] = "blocked_by_safety"
            final_state["last_event"] = f"safety_blocked:{blocked_action}"
            final_state["blocked_action"] = blocked_action
            final_state["safety_reason"] = safety_reason
        else:
            final_state = state

        print("\n=== Final State ===")
        print(final_state)

        if args.save_debug_image:
            safety_summary = (
                f"BLOCKED at {blocked_action} ({safety_reason})"
                if blocked_action is not None
                else "all safety checks allowed"
            )
            debug_image = draw_debug_image(task_frame, detection, sim_position, task_goal, safety_summary)
            saved_path = save_rgb_image(debug_image, str(DEBUG_IMAGE_PATH))
            print(f"Saved debug image to: {saved_path}")

        final_status = final_state["task_status"]
        print(f"\n=== Demo finished: task_status={final_status} ===")
        if final_status in ("success", "blocked_by_safety"):
            print("PASS")
        else:
            print("FAIL")
            print_grasp_diagnostics(final_state)

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
