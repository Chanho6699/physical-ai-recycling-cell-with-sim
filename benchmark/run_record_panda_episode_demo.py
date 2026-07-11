"""Records a Panda pick-and-place episode (success or interrupted) to
datasets/raw_episodes/ via TrajectoryRecorder.

Same TaskGoal parsing / detection / target selection / Real2Sim mapping
flow as run_task_goal_real2sim_panda_interrupt_demo.py, plus:

  - before/after each high-level action: TrajectoryRecorder.record_step()
  - during move_end_effector_to(): trajectory_callback records robot_state
    (and optionally a camera frame) every --record-every-n-steps steps
  - a deterministic mock interrupt (--interrupt-action / --interrupt-after-checks)
    identical in spirit to the interrupt demo, kept minimal here since the
    focus of this script is recording, not safety-monitor configurability
    (see run_task_goal_real2sim_panda_safety_demo.py /
    run_task_goal_real2sim_panda_interrupt_demo.py for --safety-monitor /
    --safety-source options)

No LeRobotDataset export, no OpenVLA fine-tuning, no ROS 2, no
TensorRT, no Isaac Sim, no VLA policy here yet.
"""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_collection.trajectory_recorder import TrajectoryRecorder
from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from perception.detection_types import Detection
from perception.onnx_yolo_detector import ONNXYOLODetector
from real2sim.image_to_sim_mapper import ImageToSimMapper
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import capture_pybullet_camera, save_rgb_image
from robot_sim.pybullet_panda_backend import DEFAULT_GRIPPER_STEPS, DEFAULT_MOVE_STEPS, PyBulletPandaBackend
from safety.mock_safety_monitor import MockSafetyMonitor
from safety.safety_gate import SafetyGate, SafetyGateResult
from safety.safety_types import SafetyDecision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "record_panda_episode_debug.png"

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"
DUMMY_SAFETY_FRAME = np.zeros((2, 2, 3), dtype=np.uint8)

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

    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--record-every-n-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="datasets/raw_episodes")

    parser.add_argument(
        "--interrupt-action",
        choices=["none", "move_panda_to_object", "move_panda_to_bin"],
        default="none",
    )
    parser.add_argument("--interrupt-after-checks", type=int, default=2)

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_safety_callback(args, safety_gate: SafetyGate):
    """Deterministic-interrupt-only safety callback (see module docstring
    for why this is simpler than the sibling safety/interrupt demos)."""
    call_counts: dict = {}

    def callback(action_name: str):
        if action_name == args.interrupt_action:
            call_counts[action_name] = call_counts.get(action_name, 0) + 1
            if call_counts[action_name] >= args.interrupt_after_checks:
                decision = SafetyDecision(
                    emergency_stop=True,
                    reason=f"deterministic_interrupt:{action_name}",
                    detections=[],
                )
                return SafetyGateResult(
                    allowed=False, decision=decision, action_name=action_name, reason=decision.reason
                )

        return safety_gate.check(DUMMY_SAFETY_FRAME, action_name)

    return callback


def make_trajectory_callback(recorder: TrajectoryRecorder, backend: PyBulletPandaBackend, args):
    def callback(action_name, step_index, robot_state):
        image = None
        if args.record_images:
            image = capture_pybullet_camera(physics_client_id=backend.client_id)

        recorder.record_step(
            phase=f"during_{action_name}",
            action_name=action_name,
            robot_state=robot_state,
            extra={"step_index": step_index},
            image=image,
        )

    return callback


def draw_debug_image(frame: np.ndarray, detection: Detection, sim_position: list, task_goal, summary: str) -> np.ndarray:
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
    draw.text((x1, min(y2 + 4, image.height - 12)), summary, fill=(255, 128, 0))

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
    # Panda-specific workspace ranges (NOT ImageToSimMapper's defaults,
    # which are tuned for the simple sphere backend's larger table and
    # object_z=0.53 -- Panda's table/object sit around z=0.05).
    sim_mapper = ImageToSimMapper(
        image_width=image_width,
        image_height=image_height,
        sim_x_range=(0.25, 0.55),
        sim_y_range=(-0.25, 0.25),
        object_z=0.05,
    )
    center_x, center_y = detection.center_xy
    sim_position = sim_mapper.image_point_to_sim_position(center_x, center_y)
    print("=== Mapped Panda Sim Position ===")
    print(sim_position)

    safety_gate = SafetyGate(MockSafetyMonitor(simulate_hazard=False))
    safety_callback = make_safety_callback(args, safety_gate)

    backend = PyBulletPandaBackend(gui=args.gui)
    recorder = TrajectoryRecorder(output_dir=args.output_dir)
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
        bin_target = [bin_position[0], bin_position[1], bin_position[2] + 0.05]

        episode_id = recorder.start_episode(
            instruction=args.instruction,
            task_goal=asdict(task_goal),
            metadata={
                "simulator": "pybullet_panda",
                "object_type": sim_object_type,
                "selected_detection": {
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bbox_xyxy": detection.bbox_xyxy,
                },
                "mapped_sim_position": sim_position,
                "bin_position": bin_position,
            },
        )
        print(f"=== Episode started: {episode_id} ===")

        recorder.record_step(phase="reset", action_name="reset", robot_state=state)

        trajectory_callback = make_trajectory_callback(recorder, backend, args)

        actions = [
            (
                "move_panda_to_object",
                "Move Panda to Object",
                {"type": "high_level", "name": "move_panda_to_object", "target_position": sim_position, "steps": DEFAULT_MOVE_STEPS},
                lambda: backend.move_end_effector_to(
                    sim_position,
                    safety_callback=safety_callback,
                    action_name="move_panda_to_object",
                    trajectory_callback=trajectory_callback,
                    trajectory_record_interval=args.record_every_n_steps,
                ),
            ),
            (
                "close_gripper",
                "Close Gripper",
                {"type": "high_level", "name": "close_gripper", "steps": DEFAULT_GRIPPER_STEPS},
                lambda: backend.close_gripper(),
            ),
            (
                "move_panda_to_bin",
                "Move Panda to Bin",
                {"type": "high_level", "name": "move_panda_to_bin", "target_position": bin_target, "steps": DEFAULT_MOVE_STEPS},
                lambda: backend.move_end_effector_to(
                    bin_target,
                    safety_callback=safety_callback,
                    action_name="move_panda_to_bin",
                    trajectory_callback=trajectory_callback,
                    trajectory_record_interval=args.record_every_n_steps,
                ),
            ),
            (
                "open_gripper",
                "Open Gripper",
                {"type": "high_level", "name": "open_gripper", "steps": DEFAULT_GRIPPER_STEPS},
                lambda: backend.open_gripper(),
            ),
        ]

        for action_name, action_label, high_level_action, action_fn in actions:
            print(f"\n=== Safety Check: {action_name} (pre-action) ===")
            gate_result = safety_callback(action_name)
            print(f"emergency_stop={gate_result.decision.emergency_stop}, reason={gate_result.reason}")

            safety_record = {"emergency_stop": gate_result.decision.emergency_stop, "reason": gate_result.reason}

            recorder.record_step(
                phase=f"before_{action_name}",
                action_name=action_name,
                robot_state=state,
                action=high_level_action,
                safety=safety_record,
                image=capture_pybullet_camera(physics_client_id=backend.client_id) if args.record_images else None,
            )

            if not gate_result.allowed:
                print("COMMAND BLOCKED")
                state = backend.get_state()
                state["task_status"] = "blocked_by_safety"
                state["last_event"] = f"safety_blocked:{action_name}"
                state["blocked_action"] = action_name
                state["safety_reason"] = gate_result.reason
                break

            print("COMMAND ALLOWED")
            print(f"\n=== {action_label} ===")
            state = action_fn()
            print(state)

            if action_name == "close_gripper" and not state["held_object"]:
                print_grasp_diagnostics(state)

            recorder.record_step(
                phase=f"after_{action_name}",
                action_name=action_name,
                robot_state=state,
                action=high_level_action,
                safety=safety_record,
                image=capture_pybullet_camera(physics_client_id=backend.client_id) if args.record_images else None,
            )

            if state["task_status"] == "interrupted_by_safety":
                print(f"!!! Motion interrupted mid-way during '{action_name}' !!!")
                break

        print("\n=== Final State ===")
        print(state)

        if args.save_debug_image:
            summary = f"task_status={state['task_status']}"
            debug_image = draw_debug_image(task_frame, detection, sim_position, task_goal, summary)
            saved_path = save_rgb_image(debug_image, str(DEBUG_IMAGE_PATH))
            print(f"Saved debug image to: {saved_path}")

        final_status = state["task_status"]
        success = final_status == "success"
        episode_record = recorder.finish_episode(final_state=state, success=success, status=final_status)
        print(f"\n=== Episode saved to: {episode_record['episode_dir']} ===")
        print(f"num_steps: {episode_record['num_steps']}")

        print(f"\n=== Demo finished: task_status={final_status} ===")
        if final_status in ("success", "blocked_by_safety", "interrupted_by_safety"):
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
