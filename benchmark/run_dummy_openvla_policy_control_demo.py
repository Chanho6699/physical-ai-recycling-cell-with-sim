"""OpenVLA-style online control loop, driven by DummyOpenVLAPolicy (v0).

Reuses the same TaskGoal / detection / target selection / Real2Sim
mapping preprocessing as run_task_goal_real2sim_panda_demo.py, but
instead of a fixed 4-step script, this drives the arm through an actual
per-step control loop:

  while not done:
      observation = get_observation()          # PolicyInput
      action = policy.predict_action(obs)      # PolicyOutput (7-DoF)
      command = ActionAdapter.convert(action)  # RobotCommand (unmodified)
      safety_check (optional)
      backend.apply_command(command, steps=...)

DummyOpenVLAPolicy is a scripted oracle, not a model -- but it implements
the same BasePolicy interface a real OpenVLA policy (or a FastAPI
dummy-server client) would, so this loop is the one that would run
unchanged once a real policy is swapped in.

No real OpenVLA model, no OpenVLA fine-tuning, no FastAPI OpenVLA server,
no LeRobot official parquet/video conversion, no ROS 2, no TensorRT, no
Isaac Sim here yet.
"""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from data_collection.trajectory_recorder import TrajectoryRecorder
from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from action_adapter.adapter_v0 import ActionAdapter
from perception.onnx_yolo_detector import ONNXYOLODetector
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from real2sim.image_to_sim_mapper import ImageToSimMapper
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from safety.mock_safety_monitor import MockSafetyMonitor
from safety.safety_gate import SafetyGate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"

# Panda-specific Real2Sim workspace ranges (see
# run_task_goal_real2sim_panda_demo.py) -- kept as fixed constants here
# rather than CLI flags to match this script's narrower argument surface.
SIM_X_RANGE = (0.25, 0.55)
SIM_Y_RANGE = (-0.25, 0.25)
OBJECT_Z = 0.05

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)

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

    parser.add_argument("--safety-monitor", choices=["none", "mock", "onnx"], default="none")
    parser.add_argument("--simulate-hazard", action="store_true")

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--output-dir", type=str, default="datasets/raw_episodes")

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


def format_action(action: list) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in action) + "]"


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

    safety_gate = build_safety_gate(args, model_path)
    action_adapter = ActionAdapter()
    policy = DummyOpenVLAPolicy(
        max_step_size=args.max_step_size,
        position_tolerance=args.position_tolerance,
        carry_height=args.carry_height,
        grasp_z_offset=args.grasp_z_offset,
    )

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

        policy.reset()

        if recorder is not None:
            recorder.start_episode(
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
                    "policy": "DummyOpenVLAPolicy",
                },
            )
            recorder.record_step(phase="reset", action_name="reset", robot_state=state)

        blocked_action = None
        safety_reason = None
        steps_executed = 0

        print("\n=== Control Loop ===")
        for step_index in range(args.max_policy_steps):
            steps_executed = step_index + 1

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
                    blocked_action = policy_output.phase
                    safety_reason = gate_result.reason
                    break

            state = backend.apply_command(robot_command, steps=args.steps_per_action)

            distance_to_target = (policy_output.info or {}).get("distance_to_target")
            dist_str = f"{distance_to_target:.3f}" if distance_to_target is not None else "n/a"
            print(
                f"[step {step_index:02d}] phase={policy_output.phase} "
                f"dist={dist_str} "
                f"action={format_action(policy_output.action)} "
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

            if state["task_status"] == "success":
                break
            if policy_output.done:
                break

        if blocked_action is not None:
            final_state = dict(state)
            final_state["task_status"] = "blocked_by_safety"
            final_state["last_event"] = f"safety_blocked:{blocked_action}"
            final_state["blocked_action"] = blocked_action
            final_state["safety_reason"] = safety_reason
        else:
            final_state = backend.get_state()

        print("\n=== Final State ===")
        print(final_state)

        final_status = final_state["task_status"]
        success = final_status == "success"

        if recorder is not None:
            episode_record = recorder.finish_episode(final_state=final_state, success=success, status=final_status)
            print(f"\nEpisode saved to: {episode_record['episode_dir']}")
            print(f"num_steps: {episode_record['num_steps']}")

        print("\n=== Dummy OpenVLA Policy Demo Finished ===")
        print(f"policy_steps: {steps_executed}")
        print(f"final_status: {final_status}")
        print(f"last_event: {final_state['last_event']}")
        if final_status in ("success", "blocked_by_safety"):
            print("PASS")
        else:
            ee = final_state["end_effector_position"]
            final_distance_to_bin = math.sqrt(sum((ee[axis] - bin_position[axis]) ** 2 for axis in range(3)))
            print(f"final_distance_to_bin: {final_distance_to_bin:.4f}")
            print(f"held_object: {final_state.get('held_object')}")
            print(f"current_phase: {policy.phase}")
            print(f"last_policy_info: {policy.last_info}")
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
