"""Full Recycling Cell Demo Runner -- single portfolio entry point (v0).

Ties together every stage already built and verified in its own
benchmark script into one end-to-end run:

  instruction -> TaskGoal -> image frame loading -> detection
  -> target selection -> Panda Real2Sim mapping -> PyBulletPandaBackend
  reset -> object/bin setup -> policy execution (scripted or
  DummyOpenVLAPolicy) -> optional SafetyGate -> optional
  TrajectoryRecorder -> final summary.

Two --policy backends share the same preprocessing/backend/safety/
recorder plumbing:

  scripted        A single deterministic move -> grasp -> move -> release
                  IK sequence via PyBulletPandaBackend.move_end_effector_to
                  (same shape as run_task_goal_real2sim_panda_demo.py).
  dummy-openvla   The per-step PolicyInput -> DummyOpenVLAPolicy ->
                  7-DoF action -> ActionAdapter -> apply_command() online
                  control loop (same shape as
                  run_dummy_openvla_policy_control_demo.py).

No real OpenVLA model, no OpenVLA fine-tuning, no LeRobot official
parquet/video conversion, no ROS 2, no TensorRT, no Isaac Sim, no
FastAPI OpenVLA server here yet.
"""

import argparse
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_task_goal_real2sim_panda_interrupt_demo import draw_debug_image
from data_collection.trajectory_recorder import TrajectoryRecorder
from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from perception.onnx_yolo_detector import ONNXYOLODetector
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from real2sim.image_to_sim_mapper import ImageToSimMapper
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from safety.mock_safety_monitor import MockSafetyMonitor
from safety.safety_gate import SafetyGate
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"

# Panda-specific Real2Sim workspace ranges (see
# run_task_goal_real2sim_panda_demo.py).
SIM_X_RANGE = (0.25, 0.55)
SIM_Y_RANGE = (-0.25, 0.25)
OBJECT_Z = 0.05

# The scripted policy issues one big move_end_effector_to() call per
# action rather than small per-step deltas, so it doesn't hit the
# diagonal-carry stall DummyOpenVLAPolicy's phase redesign works around
# -- it only needs a fixed clearance above the (solid-box) bin so it
# doesn't try to descend onto the box's lid.
SCRIPTED_BIN_APPROACH_CLEARANCE = 0.05

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)

    parser.add_argument("--image-source", choices=["image", "webcam"], default="image")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--webcam-warmup-frames", type=int, default=10)
    parser.add_argument("--save-webcam-frame", action="store_true")
    parser.add_argument("--save-debug-image", action="store_true")
    parser.add_argument("--webcam-output-dir", type=str, default="results/webcam")

    parser.add_argument("--policy", choices=["scripted", "dummy-openvla"], default="dummy-openvla")

    parser.add_argument("--safety-monitor", choices=["none", "mock", "onnx"], default="none")
    parser.add_argument("--simulate-hazard", action="store_true")

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--output-dir", type=str, default="datasets/raw_episodes")

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    parser.add_argument("--max-policy-steps", type=int, default=80)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--max-step-size", type=float, default=0.03)
    parser.add_argument("--position-tolerance", type=float, default=0.03)
    parser.add_argument("--carry-height", type=float, default=0.18)
    parser.add_argument("--grasp-z-offset", type=float, default=0.015)

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_safety_gate(args, model_path: Path):
    if args.safety_monitor == "none":
        return None
    if args.safety_monitor == "mock":
        return SafetyGate(MockSafetyMonitor(simulate_hazard=args.simulate_hazard))

    if args.simulate_hazard:
        print("--simulate-hazard is ignored with --safety-monitor onnx (mock-only option).")

    from safety.onnx_yolo_safety_monitor import ONNXRuntimeYOLOSafetyMonitor

    return SafetyGate(ONNXRuntimeYOLOSafetyMonitor(model_path=str(model_path)))


def load_webcam_frame(args):
    """Open the webcam (or camera_url relay stream, which takes priority
    when set), warm it up, and grab one frame. Returns None (with a
    printed explanation) instead of raising if the camera can't be
    opened or read -- WSL in particular often doesn't expose a webcam
    device at all, and that shouldn't crash the whole demo with a
    traceback.
    """
    try:
        source = WebcamSource(camera_index=args.camera_index, camera_url=args.camera_url)
    except RuntimeError:
        if not args.camera_url:
            print(f"Try a different --camera-index (currently {args.camera_index}, e.g. 0 or 1).")
        return None

    try:
        source.warmup(args.webcam_warmup_frames)
        return source.get_frame()
    except RuntimeError as exc:
        print(f"Failed to read frame from webcam: {exc}")
        if not args.camera_url:
            print(f"Try a different --camera-index (currently {args.camera_index}, e.g. 0 or 1).")
        return None
    finally:
        source.close()


def blocked_state(state: dict, action_name: str, reason: str) -> dict:
    final_state = dict(state)
    final_state["task_status"] = "blocked_by_safety"
    final_state["last_event"] = f"safety_blocked:{action_name}"
    final_state["blocked_action"] = action_name
    final_state["safety_reason"] = reason
    return final_state


def run_scripted_policy(args, backend, safety_gate, recorder, task_frame, sim_position, bin_position):
    bin_target = [bin_position[0], bin_position[1], bin_position[2] + SCRIPTED_BIN_APPROACH_CLEARANCE]

    actions = [
        ("move_to_object", lambda: backend.move_end_effector_to(sim_position)),
        ("close_gripper", lambda: backend.close_gripper()),
        ("move_above_bin", lambda: backend.move_end_effector_to(bin_target)),
        ("open_gripper", lambda: backend.open_gripper()),
    ]

    state = backend.get_state()
    policy_steps = 0

    for action_name, action_fn in actions:
        safety_record = None
        if safety_gate is not None:
            gate_result = safety_gate.check(task_frame, action_name)
            safety_record = {
                "emergency_stop": gate_result.decision.emergency_stop,
                "reason": gate_result.reason,
            }
            if not gate_result.allowed:
                print(f"[{action_name}] BLOCKED reason={gate_result.reason}")
                final_state = blocked_state(state, action_name, gate_result.reason)
                if recorder is not None:
                    recorder.record_step(
                        phase=action_name,
                        action_name=action_name,
                        robot_state=final_state,
                        safety=safety_record,
                        image=task_frame if args.record_images else None,
                    )
                return final_state, policy_steps

        state = action_fn()
        policy_steps += 1
        print(f"[{action_name}] status={state['task_status']} ee={state['end_effector_position']}")

        if recorder is not None:
            recorder.record_step(
                phase=action_name,
                action_name=action_name,
                robot_state=state,
                safety=safety_record,
                image=task_frame if args.record_images else None,
            )

    return state, policy_steps


def run_dummy_openvla_policy(args, backend, safety_gate, recorder, task_frame, task_goal, sim_position, bin_position):
    action_adapter = ActionAdapter()
    policy = DummyOpenVLAPolicy(
        max_step_size=args.max_step_size,
        position_tolerance=args.position_tolerance,
        carry_height=args.carry_height,
        grasp_z_offset=args.grasp_z_offset,
    )
    policy.reset()

    state = backend.get_state()

    for step_index in range(args.max_policy_steps):
        robot_state = backend.get_state()
        policy_input = PolicyInput(
            image=task_frame,
            instruction=args.instruction,
            robot_state=robot_state,
            task_goal=asdict(task_goal),
            target_object_position=sim_position,
            bin_position=bin_position,
            step_index=step_index,
            phase=policy.phase,
        )
        policy_output = policy.predict_action(policy_input)
        robot_command = action_adapter.convert(policy_output.action)

        safety_record = None
        if safety_gate is not None:
            gate_result = safety_gate.check(task_frame, policy_output.phase)
            safety_record = {
                "emergency_stop": gate_result.decision.emergency_stop,
                "reason": gate_result.reason,
            }
            if not gate_result.allowed:
                print(f"[step {step_index:02d}] phase={policy_output.phase} BLOCKED reason={gate_result.reason}")
                final_state = blocked_state(robot_state, policy_output.phase, gate_result.reason)
                if recorder is not None:
                    recorder.record_step(
                        phase=policy_output.phase,
                        action_name=policy_output.phase,
                        robot_state=final_state,
                        safety=safety_record,
                        image=task_frame if args.record_images else None,
                    )
                return final_state, step_index + 1

        state = backend.apply_command(robot_command, steps=args.steps_per_action)

        distance_to_target = (policy_output.info or {}).get("distance_to_target")
        dist_str = f"{distance_to_target:.3f}" if distance_to_target is not None else "n/a"
        print(
            f"[step {step_index:02d}] phase={policy_output.phase} dist={dist_str} "
            f"ee=[{state['end_effector_position'][0]:.3f}, "
            f"{state['end_effector_position'][1]:.3f}, "
            f"{state['end_effector_position'][2]:.3f}] "
            f"status={state['task_status']}"
        )

        if recorder is not None:
            recorder.record_step(
                phase=policy_output.phase,
                action_name=policy_output.phase,
                robot_state=state,
                action={"type": "openvla_style", "vector": policy_output.action, "info": policy_output.info},
                safety=safety_record,
                image=task_frame if args.record_images else None,
            )

        if state["task_status"] == "success" or policy_output.done:
            return state, step_index + 1

    return state, args.max_policy_steps


def main() -> None:
    args = parse_args()

    print("=== Full Recycling Cell Demo ===")
    print(f"policy: {args.policy}")
    print(f"safety_monitor: {args.safety_monitor}")
    print(f"record: {args.record}")

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

        task_frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    else:
        task_frame = load_webcam_frame(args)
        if task_frame is None:
            print("FAIL")
            return

        if args.save_webcam_frame:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = resolve(args.webcam_output_dir) / f"webcam_frame_{timestamp}.jpg"
            saved_path = save_rgb_image(task_frame, str(output_path))
            print(f"Saved webcam frame to: {saved_path}")

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
        sim_x_range=SIM_X_RANGE,
        sim_y_range=SIM_Y_RANGE,
        object_z=OBJECT_Z,
    )
    center_x, center_y = detection.center_xy
    sim_position = sim_mapper.image_point_to_sim_position(center_x, center_y)
    print("=== Mapped Panda Sim Position ===")
    print(sim_position)

    if args.save_debug_image:
        debug_image = draw_debug_image(task_frame, detection, sim_position, task_goal, f"policy={args.policy}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_output_path = resolve(args.webcam_output_dir) / f"webcam_detection_debug_{timestamp}.jpg"
        saved_debug_path = save_rgb_image(debug_image, str(debug_output_path))
        print(f"Saved debug detection image to: {saved_debug_path}")

    safety_gate = build_safety_gate(args, model_path)
    backend = PyBulletPandaBackend(gui=args.gui)
    recorder = TrajectoryRecorder(output_dir=args.output_dir) if args.record else None

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

        if recorder is not None:
            recorder.start_episode(
                instruction=args.instruction,
                task_goal=asdict(task_goal),
                metadata={
                    "runner": "full_recycling_cell_demo",
                    "policy": args.policy,
                    "instruction": args.instruction,
                    "task_goal": asdict(task_goal),
                    "selected_detection": {
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "bbox_xyxy": detection.bbox_xyxy,
                    },
                    "mapped_sim_position": sim_position,
                    "bin_position": bin_position,
                    "safety_monitor": args.safety_monitor,
                },
            )
            recorder.record_step(phase="reset", action_name="reset", robot_state=state)

        print("\n=== Policy Execution ===")
        if args.policy == "scripted":
            final_state, policy_steps = run_scripted_policy(
                args, backend, safety_gate, recorder, task_frame, sim_position, bin_position
            )
        else:
            final_state, policy_steps = run_dummy_openvla_policy(
                args, backend, safety_gate, recorder, task_frame, task_goal, sim_position, bin_position
            )

        print("\n=== Final State ===")
        print(final_state)

        final_status = final_state["task_status"]
        success = final_status == "success"

        recorded_episode = None
        if recorder is not None:
            episode_record = recorder.finish_episode(final_state=final_state, success=success, status=final_status)
            recorded_episode = episode_record["episode_dir"]
            print(f"\nEpisode saved to: {recorded_episode}")
            print(f"num_steps: {episode_record['num_steps']}")

        print("\n=== Full Demo Finished ===")
        print(f"policy: {args.policy}")
        print(f"policy_steps: {policy_steps}")
        print(f"final_status: {final_status}")
        print(f"last_event: {final_state['last_event']}")
        print(f"recorded_episode: {recorded_episode}")

        if final_status in ("success", "blocked_by_safety"):
            print("PASS")
        else:
            ee = final_state["end_effector_position"]
            final_distance_to_bin = math.sqrt(sum((ee[axis] - bin_position[axis]) ** 2 for axis in range(3)))
            print(f"final_distance_to_bin: {final_distance_to_bin:.4f}")
            print(f"held_object: {final_state.get('held_object')}")
            print("FAIL")

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
