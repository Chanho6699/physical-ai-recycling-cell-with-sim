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
import json
import math
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.run_task_goal_real2sim_panda_interrupt_demo import draw_debug_image
from data_collection.perception_episode_schema import (
    build_detections_section,
    build_episode_metadata,
    build_input_source_section,
    build_policy_observation_section,
    build_real2sim_section,
    build_result_section,
    build_robot_section,
    build_safety_section,
    build_selected_target_section,
    build_wrist_camera_section,
    write_episode_metadata_file,
)
from data_collection.trajectory_recorder import TrajectoryRecorder, to_jsonable
from llm_agent.rule_based_parser import RuleBasedTaskGoalParser
from perception.onnx_yolo_detector import ONNXYOLODetector
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.fastapi_vla_policy_client import FastAPIVLAPolicyClient
from policy.policy_types import PolicyInput
from real2sim.aruco_table_mapper import ArUcoTableMapper, draw_aruco_debug_image, print_aruco_mapping_debug
from real2sim.calibrated_image_to_sim_mapper import (
    CalibratedImageToSimMapper,
    draw_roi_rectangle,
    print_mapping_debug,
)
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from robot_sim.pybullet_wrist_camera import (
    PyBulletWristCamera,
    build_wrist_observation_metadata,
    print_wrist_refinement_debug,
    refine_target_with_wrist_camera,
    save_wrist_camera_outputs,
)
from safety.mock_hand_intrusion_monitor import MockHandIntrusionMonitor
from safety.mock_safety_monitor import MockSafetyMonitor
from safety.safety_gate import SafetyGate
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"

DEFAULT_CALIBRATION_CONFIG = "configs/real2sim_webcam_calibration.json"
DEFAULT_ARUCO_CALIBRATION = "configs/real2sim_aruco_table_calibration.json"
DEFAULT_WRIST_CAMERA_CONFIG = "configs/wrist_camera_config.json"
WRIST_CAMERA_OUTPUT_DIR = "results/wrist_camera"

# How far above the object (in the object's own xy) the end effector
# moves to before rendering the wrist camera in --wrist-camera-mode
# observe -- matches the "look straight down" pose the default
# configs/wrist_camera_config.json's camera_forward_local expects.
WRIST_CAMERA_OBSERVE_HEIGHT = 0.25

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

    parser.add_argument("--real2sim-mode", choices=["roi", "aruco"], default="roi")
    parser.add_argument("--real2sim-calibration", type=str, default=DEFAULT_CALIBRATION_CONFIG)
    parser.add_argument("--aruco-calibration", type=str, default=DEFAULT_ARUCO_CALIBRATION)

    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument("--print-mapping-debug", dest="print_mapping_debug", action="store_true")
    debug_group.add_argument("--no-print-mapping-debug", dest="print_mapping_debug", action="store_false")
    parser.set_defaults(print_mapping_debug=True)

    parser.add_argument("--wrist-camera-mode", choices=["off", "observe", "refine"], default="off")
    parser.add_argument("--wrist-camera-config", type=str, default=DEFAULT_WRIST_CAMERA_CONFIG)
    parser.add_argument("--save-wrist-camera-images", action="store_true")

    parser.add_argument("--wrist-refinement-policy", choices=["none", "blend", "override"], default="blend")
    parser.add_argument("--wrist-refinement-alpha", type=float, default=0.7)
    parser.add_argument("--refine-distance-threshold", type=float, default=0.08)
    parser.add_argument("--wrist-min-object-pixels", type=int, default=50)
    parser.add_argument("--wrist-max-refinement-delta", type=float, default=0.08)

    parser.add_argument("--policy-observation-source", choices=["none", "wrist"], default="none")
    parser.add_argument("--record-policy-observations", action="store_true")
    parser.add_argument("--policy-observation-save-interval", type=int, default=5)

    parser.add_argument("--policy-backend", choices=["local-dummy", "fastapi-dummy"], default="local-dummy")
    parser.add_argument("--policy-server-url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--policy-request-timeout", type=float, default=5.0)

    parser.add_argument("--policy", choices=["scripted", "dummy-openvla"], default="dummy-openvla")

    parser.add_argument("--safety-monitor", choices=["none", "mock", "onnx"], default="none")
    parser.add_argument("--simulate-hazard", action="store_true")

    parser.add_argument("--safety-mode", choices=["off", "block", "pause-resume"], default="off")
    parser.add_argument("--mock-hand-intrusion", action="store_true")
    parser.add_argument("--mock-hand-start-step", type=int, default=10)
    parser.add_argument("--mock-hand-end-step", type=int, default=20)
    parser.add_argument("--safety-resume-stable-steps", type=int, default=3)

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--output-dir", type=str, default="datasets/raw_episodes")

    perception_metadata_group = parser.add_mutually_exclusive_group()
    perception_metadata_group.add_argument(
        "--record-perception-metadata", dest="record_perception_metadata", action="store_true"
    )
    perception_metadata_group.add_argument(
        "--no-record-perception-metadata", dest="record_perception_metadata", action="store_false"
    )
    parser.set_defaults(record_perception_metadata=True)
    parser.add_argument("--episode-tag", type=str, default=None)

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

    parser.add_argument("--policy-step-delay", type=float, default=0.0)
    parser.add_argument("--simulation-step-delay", type=float, default=0.0)
    parser.add_argument("--slow-gui", action="store_true")

    return parser.parse_args()


# --slow-gui preset (only meaningful with --gui): slow enough for a
# person to actually follow the arm's motion without changing headless
# speed at all, since both delays stay 0.0 unless --gui slow mode (this
# preset or an explicit --policy-step-delay/--simulation-step-delay) asks
# for them.
SLOW_GUI_POLICY_STEP_DELAY = 0.05
SLOW_GUI_SIMULATION_STEP_DELAY = 0.003


def resolve_step_delays(args) -> tuple:
    """A value >0 on the explicit CLI flags always wins; --slow-gui only
    fills in its preset where the explicit flag was left at its 0.0
    default."""
    policy_step_delay = args.policy_step_delay if args.policy_step_delay > 0 else (
        SLOW_GUI_POLICY_STEP_DELAY if args.slow_gui else 0.0
    )
    simulation_step_delay = args.simulation_step_delay if args.simulation_step_delay > 0 else (
        SLOW_GUI_SIMULATION_STEP_DELAY if args.slow_gui else 0.0
    )
    return policy_step_delay, simulation_step_delay


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


def observe_wrist_camera(args, backend, sim_position: list) -> None:
    """--wrist-camera-mode observe: move above the object, render the
    wrist camera, and report how well it can re-locate the object from
    up close -- purely diagnostic. Nothing here changes sim_position or
    any policy target; pick-and-place runs exactly as it would with
    --wrist-camera-mode off right after this returns.
    """
    observe_position = [sim_position[0], sim_position[1], sim_position[2] + WRIST_CAMERA_OBSERVE_HEIGHT]
    backend.move_end_effector_to(observe_position)

    wrist_camera = PyBulletWristCamera(
        client_id=backend.client_id,
        robot_id=backend.robot_id,
        config_path=resolve(args.wrist_camera_config),
    )
    frame, render_debug = wrist_camera.render()
    estimated_position, estimate_debug = wrist_camera.estimate_object_position_from_segmentation(
        frame, backend._object_id
    )

    print("\n=== Wrist Camera Observation ===")
    print(f"object_visible: {estimate_debug['object_visible']}")
    if estimate_debug["object_visible"]:
        print(f"object_pixel_count: {estimate_debug['object_pixel_count']}")
        print(f"estimated_world_position: {estimated_position}")
        print(f"gt_object_position: {sim_position}")
        position_error_xy = math.sqrt(
            (estimated_position[0] - sim_position[0]) ** 2 + (estimated_position[1] - sim_position[1]) ** 2
        )
        print(f"position_error_xy: {position_error_xy:.4f}")
    else:
        print("Wrist camera could not see the object from the observe position.")

    if args.save_wrist_camera_images:
        saved_paths = save_wrist_camera_outputs(
            frame,
            {**render_debug, **estimate_debug},
            resolve(WRIST_CAMERA_OUTPUT_DIR),
            save_depth_colormap=wrist_camera.save_depth_colormap,
            save_segmentation_mask=wrist_camera.save_segmentation_mask,
            extra_debug={"object_position_gt": sim_position},
        )
        print(f"Saved wrist camera outputs: {saved_paths}")


def copy_perception_artifacts_into_episode(
    episode_dir: Path,
    saved_webcam_frame_path,
    saved_debug_path,
    mapping_debug: dict,
    wrist_refinement_debug,
) -> None:
    """Best-effort copy/write of already-saved perception artifacts into
    the episode folder (see docs/dataset_pipeline.md), so an episode
    carries its own external-camera frame/debug image and Real2Sim/wrist
    refinement debug JSON without needing results/webcam or
    results/wrist_camera to still exist later. Never raises -- a missing
    source path (e.g. --save-webcam-frame wasn't used) just means that
    particular artifact is skipped.
    """
    frames_dir = episode_dir / "frames"
    debug_dir = episode_dir / "debug"

    if saved_webcam_frame_path:
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(saved_webcam_frame_path, frames_dir / "external_webcam_frame.jpg")
        except OSError as exc:
            print(f"Could not copy webcam frame into episode folder: {exc}")

    if saved_debug_path:
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(saved_debug_path, frames_dir / "external_detection_debug.jpg")
        except OSError as exc:
            print(f"Could not copy debug detection image into episode folder: {exc}")

    debug_dir.mkdir(parents=True, exist_ok=True)
    mapping_debug_name = (
        "aruco_mapping_debug.json"
        if str(mapping_debug.get("mapping_mode", "")).startswith("aruco")
        else "real2sim_mapping_debug.json"
    )
    with open(debug_dir / mapping_debug_name, "w", encoding="utf-8") as debug_file:
        json.dump(to_jsonable(mapping_debug), debug_file, ensure_ascii=False, indent=2)

    if wrist_refinement_debug is not None:
        with open(debug_dir / "wrist_refinement_debug.json", "w", encoding="utf-8") as debug_file:
            json.dump(to_jsonable(wrist_refinement_debug), debug_file, ensure_ascii=False, indent=2)


def blocked_state(state: dict, action_name: str, reason: str) -> dict:
    final_state = dict(state)
    final_state["task_status"] = "blocked_by_safety"
    final_state["last_event"] = f"safety_blocked:{action_name}"
    final_state["blocked_action"] = action_name
    final_state["safety_reason"] = reason
    return final_state


def run_scripted_policy(
    args,
    backend,
    safety_gate,
    recorder,
    task_frame,
    sim_position,
    bin_position,
    policy_step_delay: float = 0.0,
    simulation_step_delay: float = 0.0,
):
    bin_target = [bin_position[0], bin_position[1], bin_position[2] + SCRIPTED_BIN_APPROACH_CLEARANCE]

    actions = [
        ("move_to_object", lambda: backend.move_end_effector_to(sim_position, step_delay=simulation_step_delay)),
        ("close_gripper", lambda: backend.close_gripper()),
        ("move_above_bin", lambda: backend.move_end_effector_to(bin_target, step_delay=simulation_step_delay)),
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

        if policy_step_delay > 0:
            time.sleep(policy_step_delay)

    return state, policy_steps


def run_dummy_openvla_policy(
    args,
    backend,
    safety_gate,
    recorder,
    task_frame,
    task_goal,
    sim_position,
    bin_position,
    policy_step_delay: float = 0.0,
    simulation_step_delay: float = 0.0,
    wrist_camera=None,
    hand_intrusion_gate=None,
):
    action_adapter = ActionAdapter()
    if args.policy_backend == "fastapi-dummy":
        policy = FastAPIVLAPolicyClient(server_url=args.policy_server_url, timeout=args.policy_request_timeout)
    else:
        policy = DummyOpenVLAPolicy(
            max_step_size=args.max_step_size,
            position_tolerance=args.position_tolerance,
            carry_height=args.carry_height,
            grasp_z_offset=args.grasp_z_offset,
        )
    policy.reset()

    state = backend.get_state()
    target_object_position = list(sim_position)
    wrist_refinement_state = {
        "wrist_refinement_attempted": False,
        "wrist_refinement_applied": False,
        "wrist_refinement_delta_xy": None,
        "wrist_refinement_debug": None,
    }
    policy_observation_state = {
        "policy_observation_source": args.policy_observation_source,
        "used_wrist_observation_steps": 0,
        "recorded_wrist_observation_steps": 0,
    }
    policy_backend_state = {
        "policy_backend": args.policy_backend,
        "policy_server_url": args.policy_server_url if args.policy_backend == "fastapi-dummy" else None,
        "avg_inference_latency_ms": None,
    }
    inference_latencies_ms = []

    # Safety Pause/Resume v0 state machine. Deliberately decoupled from
    # safety_gate (the existing --safety-monitor mock/onnx hard-block
    # path, unchanged by this feature): hand_intrusion_gate is only
    # non-None under --safety-mode pause-resume, and it never fails the
    # episode -- it only pauses/resumes action application.
    safety_state = "running"
    stable_clear_steps = 0
    safety_info = {
        "safety_mode": args.safety_mode,
        "safety_pause_count": 0,
        "safety_resume_count": 0,
        "paused_steps": 0,
        "final_safety_state": "running",
    }

    def with_wrist_refinement_info(state_dict: dict) -> dict:
        if inference_latencies_ms:
            policy_backend_state["avg_inference_latency_ms"] = round(
                sum(inference_latencies_ms) / len(inference_latencies_ms), 3
            )
        safety_info["final_safety_state"] = safety_state
        return {
            **state_dict,
            **wrist_refinement_state,
            **policy_observation_state,
            **policy_backend_state,
            **safety_info,
        }

    for step_index in range(args.max_policy_steps):
        robot_state = backend.get_state()
        refinement_event = None
        wrist_observation_metadata = None

        # VLA-ready per-step observation: render the wrist camera and feed
        # it into PolicyInput.image every step (not gated by phase/distance
        # the way refinement below is) -- this is the loop shape a real
        # VLA/visual policy would see, even though DummyOpenVLAPolicy
        # itself doesn't do any visual reasoning on it yet.
        policy_image = task_frame
        observation_source = None
        visual_observation = None
        wrist_frame = None

        if wrist_camera is not None and args.policy_observation_source == "wrist":
            wrist_frame, wrist_render_debug = wrist_camera.render()
            _, wrist_estimate_debug = wrist_camera.estimate_object_position_from_segmentation(
                wrist_frame, backend._object_id
            )
            policy_image = wrist_frame["rgb"]
            observation_source = "wrist"
            visual_observation = {
                "object_visible": wrist_estimate_debug["object_visible"],
                "object_pixel_count": wrist_estimate_debug["object_pixel_count"],
                "estimated_world_position": wrist_estimate_debug["estimated_world_position"],
            }
            wrist_observation_metadata = build_wrist_observation_metadata(
                step_index, wrist_frame, wrist_render_debug, wrist_estimate_debug
            )
            policy_observation_state["used_wrist_observation_steps"] += 1

        # Refine the grasp target exactly once, right as the arm is
        # closing in on the object (not before -- an early wrist-camera
        # look from far away, still mostly seeing the gripper/background,
        # would be far less reliable than one taken this close).
        if (
            wrist_camera is not None
            and args.wrist_camera_mode == "refine"
            and not wrist_refinement_state["wrist_refinement_attempted"]
            and policy.phase == "move_to_object"
            and not robot_state.get("held_object", False)
            and (policy.last_info or {}).get("distance_to_target") is not None
            and policy.last_info["distance_to_target"] <= args.refine_distance_threshold
        ):
            wrist_refinement_state["wrist_refinement_attempted"] = True
            refined_position, refinement_debug = refine_target_with_wrist_camera(
                backend,
                wrist_camera,
                target_object_position,
                backend._object_id,
                mode=args.wrist_refinement_policy,
                blend_alpha=args.wrist_refinement_alpha,
                min_object_pixels=args.wrist_min_object_pixels,
                max_refinement_delta=args.wrist_max_refinement_delta,
                frame=wrist_frame,
            )
            print()
            print_wrist_refinement_debug(refinement_debug)
            target_object_position = refined_position
            wrist_refinement_state["wrist_refinement_applied"] = refinement_debug["refinement_applied"]
            wrist_refinement_state["wrist_refinement_delta_xy"] = refinement_debug["xy_delta_from_coarse"]
            wrist_refinement_state["wrist_refinement_debug"] = refinement_debug
            refinement_event = {
                "event_type": "wrist_refinement",
                "coarse_target_position": refinement_debug["coarse_target_position"],
                "wrist_estimated_position": refinement_debug["wrist_estimated_position"],
                "refined_target_position": refinement_debug["refined_target_position"],
                "refinement_applied": refinement_debug["refinement_applied"],
            }

            if args.save_wrist_camera_images:
                refine_frame, refine_render_debug = wrist_camera.render()
                saved_wrist_paths = save_wrist_camera_outputs(
                    refine_frame,
                    {**refine_render_debug, **refinement_debug},
                    resolve(WRIST_CAMERA_OUTPUT_DIR),
                    save_depth_colormap=wrist_camera.save_depth_colormap,
                    save_segmentation_mask=wrist_camera.save_segmentation_mask,
                    extra_debug={"event": "wrist_refinement", "step_index": step_index},
                )
                print(f"Saved wrist camera outputs: {saved_wrist_paths}")
                refinement_event["saved_wrist_camera_paths"] = saved_wrist_paths

        # Safety Pause/Resume v0: mock-timed hand-intrusion check runs
        # every step, before the policy is ever called, so that a paused
        # step neither invokes the policy nor applies a robot command.
        if args.safety_mode == "pause-resume" and hand_intrusion_gate is not None:
            hand_intrusion_gate.safety_monitor.set_step(step_index)
            hand_gate_result = hand_intrusion_gate.check(task_frame, "safety_check")
            hand_detected = hand_gate_result.decision.emergency_stop

            safety_event = None
            if hand_detected:
                stable_clear_steps = 0
                if safety_state == "running":
                    safety_state = "paused_by_safety"
                    safety_info["safety_pause_count"] += 1
                    event_type = "safety_pause"
                else:
                    safety_state = "paused_by_safety"
                    event_type = "safety_still_paused"
                safety_info["paused_steps"] += 1
                safety_event = {
                    "event_type": event_type,
                    "reason": "mock_hand_intrusion",
                    "safety_mode": args.safety_mode,
                    "robot_action_applied": False,
                    "hand_detected": True,
                }
            elif safety_state in ("paused_by_safety", "resuming"):
                stable_clear_steps += 1
                safety_info["paused_steps"] += 1
                if stable_clear_steps >= args.safety_resume_stable_steps:
                    safety_state = "running"
                    safety_info["safety_resume_count"] += 1
                    safety_event = {
                        "event_type": "safety_resume",
                        "reason": "hand_cleared",
                        "stable_clear_steps": stable_clear_steps,
                        "robot_action_applied": False,
                        "hand_detected": False,
                    }
                    stable_clear_steps = 0
                else:
                    safety_state = "resuming"
                    safety_event = {
                        "event_type": "safety_still_paused",
                        "reason": "mock_hand_intrusion",
                        "stable_clear_steps": stable_clear_steps,
                        "robot_action_applied": False,
                        "hand_detected": False,
                    }

            if safety_event is not None:
                print(
                    f"[step {step_index:02d}] {safety_event['event_type']} "
                    f"reason={safety_event['reason']} safety_state={safety_state}"
                )
                if recorder is not None:
                    recorder.record_step(
                        phase=policy.phase,
                        action_name=safety_event["event_type"],
                        robot_state=robot_state,
                        extra=safety_event,
                    )
                if policy_step_delay > 0:
                    time.sleep(policy_step_delay)
                continue

        policy_input = PolicyInput(
            image=policy_image,
            instruction=args.instruction,
            robot_state=robot_state,
            task_goal=asdict(task_goal),
            target_object_position=target_object_position,
            bin_position=bin_position,
            step_index=step_index,
            phase=policy.phase,
            observation_source=observation_source,
            visual_observation=visual_observation,
        )
        policy_output = policy.predict_action(policy_input)
        robot_command = action_adapter.convert(policy_output.action)

        policy_output_info = policy_output.info or {}
        inference_latency_ms = policy_output_info.get("inference_latency_ms")
        if inference_latency_ms is not None:
            inference_latencies_ms.append(inference_latency_ms)

        policy_observation_event = None
        if args.record_policy_observations:
            policy_observation_event = {
                "policy_output": {
                    "action": policy_output.action,
                    "policy_backend": policy_output_info.get("policy_backend", args.policy_backend),
                    "inference_latency_ms": inference_latency_ms,
                    "used_image_input": policy_output_info.get("used_image_input"),
                    "observation_source": policy_output_info.get("observation_source"),
                }
            }
            if observation_source == "wrist":
                policy_observation_event["policy_input"] = {
                    "image_source": observation_source,
                    "image_shape": list(policy_image.shape),
                    "has_image": True,
                }
                policy_observation_event["wrist_observation"] = {
                    "object_visible": visual_observation["object_visible"],
                    "object_pixel_count": visual_observation["object_pixel_count"],
                    "object_bbox_px": wrist_observation_metadata["object_bbox_px"],
                    "estimated_world_position": visual_observation["estimated_world_position"],
                }

        if (
            args.record_images
            and wrist_frame is not None
            and recorder is not None
            and recorder.episode_dir is not None
            and step_index % max(args.policy_observation_save_interval, 1) == 0
        ):
            wrist_step_image_path = recorder.episode_dir / "frames" / f"wrist_policy_step_{step_index:06d}.png"
            save_rgb_image(policy_image, str(wrist_step_image_path))
            policy_observation_state["recorded_wrist_observation_steps"] += 1

        step_extra = None
        if refinement_event is not None or policy_observation_event is not None:
            step_extra = {**(refinement_event or {}), **(policy_observation_event or {})}

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
                        extra=step_extra,
                    )
                return with_wrist_refinement_info(final_state), step_index + 1

        state = backend.apply_command(robot_command, steps=args.steps_per_action, step_delay=simulation_step_delay)

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
                extra=step_extra,
            )

        if policy_step_delay > 0:
            time.sleep(policy_step_delay)

        if state["task_status"] == "success" or policy_output.done:
            return with_wrist_refinement_info(state), step_index + 1

    return with_wrist_refinement_info(state), args.max_policy_steps


def main() -> None:
    args = parse_args()

    policy_step_delay, simulation_step_delay = resolve_step_delays(args)

    print("=== Full Recycling Cell Demo ===")
    print(f"policy: {args.policy}")
    print(f"safety_monitor: {args.safety_monitor}")
    print(f"record: {args.record}")
    print(f"gui: {args.gui}")
    print(f"slow_gui: {args.slow_gui}")
    print(f"policy_step_delay: {policy_step_delay}")
    print(f"simulation_step_delay: {simulation_step_delay}")

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

    saved_webcam_frame_path = None
    saved_debug_path = None

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
            saved_webcam_frame_path = save_rgb_image(task_frame, str(output_path))
            print(f"Saved webcam frame to: {saved_webcam_frame_path}")

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

    if args.real2sim_mode == "aruco":
        try:
            aruco_mapper = ArUcoTableMapper(resolve(args.aruco_calibration))
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"ArUco mapper setup failed: {exc}")
            return

        marker_detections = aruco_mapper.detect_markers(task_frame)
        sim_position, mapping_debug = aruco_mapper.map_detection(detection, task_frame)
        print("=== Mapped Panda Sim Position ===")
        print(sim_position)
        print()
        if args.print_mapping_debug:
            print_aruco_mapping_debug(mapping_debug)

        if args.save_debug_image:
            summary = f"policy={args.policy}"
            if mapping_debug.get("out_of_bounds"):
                summary += " (OUT OF BOUNDS)"
            display_position = mapping_debug.get("mapped_position_raw", sim_position)
            debug_image = draw_aruco_debug_image(
                task_frame,
                marker_detections,
                aruco_mapper.required_marker_ids,
                detection=detection,
                mapped_position=display_position,
                task_goal=task_goal,
                summary=summary,
                draw_table_polygon=aruco_mapper.debug_config.get("draw_table_polygon", True),
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_output_path = resolve(args.webcam_output_dir) / f"webcam_detection_debug_{timestamp}.jpg"
            saved_debug_path = save_rgb_image(debug_image, str(debug_output_path))
            print(f"Saved debug detection image to: {saved_debug_path}")

        if sim_position is None:
            print("FAIL")
            return
    else:
        sim_mapper = CalibratedImageToSimMapper.from_config_file(resolve(args.real2sim_calibration))
        sim_position, mapping_debug = sim_mapper.map_bbox_to_sim(detection.bbox_xyxy, image_width, image_height)
        print("=== Mapped Panda Sim Position ===")
        print(sim_position)
        print()
        if args.print_mapping_debug:
            print_mapping_debug(mapping_debug)

        if args.save_debug_image:
            frame_with_roi = draw_roi_rectangle(task_frame, mapping_debug["image_roi"])
            debug_image = draw_debug_image(frame_with_roi, detection, sim_position, task_goal, f"policy={args.policy}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_output_path = resolve(args.webcam_output_dir) / f"webcam_detection_debug_{timestamp}.jpg"
            saved_debug_path = save_rgb_image(debug_image, str(debug_output_path))
            print(f"Saved debug detection image to: {saved_debug_path}")

    if args.policy_backend == "fastapi-dummy" and args.policy == "dummy-openvla":
        health_check_client = FastAPIVLAPolicyClient(
            server_url=args.policy_server_url, timeout=args.policy_request_timeout
        )
        try:
            health = health_check_client.check_health()
            print(f"policy_backend: fastapi-dummy ({args.policy_server_url})")
            print(f"FastAPI VLA policy server health: {health}")
        except RuntimeError as exc:
            print(str(exc))
            print("FAIL")
            return

    safety_gate = build_safety_gate(args, model_path)

    hand_intrusion_gate = None
    if args.safety_mode == "pause-resume":
        hand_intrusion_gate = SafetyGate(
            MockHandIntrusionMonitor(
                start_step=args.mock_hand_start_step,
                end_step=args.mock_hand_end_step,
                active=args.mock_hand_intrusion,
            )
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
                    "safety_mode": args.safety_mode,
                },
            )
            recorder.record_step(phase="reset", action_name="reset", robot_state=state)

        if args.wrist_camera_mode == "observe":
            observe_wrist_camera(args, backend, sim_position)

        wrist_camera = None
        needs_wrist_camera = args.wrist_camera_mode == "refine" or args.policy_observation_source == "wrist"
        if needs_wrist_camera:
            if args.policy != "dummy-openvla":
                print(
                    "--wrist-camera-mode refine / --policy-observation-source wrist are only wired into "
                    "--policy dummy-openvla; ignoring for scripted."
                )
            else:
                wrist_camera = PyBulletWristCamera(
                    client_id=backend.client_id,
                    robot_id=backend.robot_id,
                    config_path=resolve(args.wrist_camera_config),
                )

        if args.safety_mode == "pause-resume" and args.policy != "dummy-openvla":
            print("--safety-mode pause-resume is only wired into --policy dummy-openvla; ignoring for scripted.")

        print("\n=== Policy Execution ===")
        if args.policy == "scripted":
            final_state, policy_steps = run_scripted_policy(
                args,
                backend,
                safety_gate,
                recorder,
                task_frame,
                sim_position,
                bin_position,
                policy_step_delay=policy_step_delay,
                simulation_step_delay=simulation_step_delay,
            )
        else:
            final_state, policy_steps = run_dummy_openvla_policy(
                args,
                backend,
                safety_gate,
                recorder,
                task_frame,
                task_goal,
                sim_position,
                bin_position,
                policy_step_delay=policy_step_delay,
                simulation_step_delay=simulation_step_delay,
                wrist_camera=wrist_camera,
                hand_intrusion_gate=hand_intrusion_gate,
            )

        print("\n=== Final State ===")
        print(final_state)

        final_status = final_state["task_status"]
        success = final_status == "success"

        recorded_episode = None
        if recorder is not None:
            if args.record_perception_metadata:
                episode_metadata = build_episode_metadata(
                    episode_id=recorder.episode_dir.name if recorder.episode_dir else "",
                    task_goal=task_goal,
                    input_source=build_input_source_section(args, saved_webcam_frame_path),
                    detections=build_detections_section(detections),
                    selected_target=build_selected_target_section(detection, sim_object_type),
                    real2sim=build_real2sim_section(args.real2sim_mode, mapping_debug),
                    wrist_camera=build_wrist_camera_section(
                        args.wrist_camera_mode, final_state.get("wrist_refinement_debug")
                    ),
                    robot=build_robot_section(args.policy, policy_steps, final_state),
                    result=build_result_section(final_state, bin_position, success),
                    policy_observation=build_policy_observation_section(final_state),
                    safety=build_safety_section(args.safety_mode, args.mock_hand_intrusion, final_state),
                    episode_tag=args.episode_tag,
                )
                recorder.update_metadata(episode_metadata)

            episode_record = recorder.finish_episode(final_state=final_state, success=success, status=final_status)
            recorded_episode = episode_record["episode_dir"]
            print(f"\nEpisode saved to: {recorded_episode}")
            print(f"num_steps: {episode_record['num_steps']}")

            if args.record_perception_metadata:
                metadata_path = write_episode_metadata_file(recorded_episode, episode_metadata)
                print(f"Episode metadata saved to: {metadata_path}")

            if args.record_images:
                copy_perception_artifacts_into_episode(
                    Path(recorded_episode),
                    saved_webcam_frame_path=saved_webcam_frame_path,
                    saved_debug_path=saved_debug_path,
                    mapping_debug=mapping_debug,
                    wrist_refinement_debug=final_state.get("wrist_refinement_debug"),
                )

        print("\n=== Full Demo Finished ===")
        print(f"policy: {args.policy}")
        print(f"policy_backend: {final_state.get('policy_backend', args.policy_backend)}")
        print(f"policy_steps: {policy_steps}")
        print(f"final_status: {final_status}")
        print(f"last_event: {final_state['last_event']}")
        print(f"recorded_episode: {recorded_episode}")
        if args.wrist_camera_mode == "refine":
            print(f"wrist_refinement_applied: {final_state.get('wrist_refinement_applied')}")
            print(f"wrist_refinement_delta_xy: {final_state.get('wrist_refinement_delta_xy')}")
        if args.policy_observation_source != "none":
            print(f"policy_observation_source: {final_state.get('policy_observation_source')}")
            print(f"used_wrist_observation_steps: {final_state.get('used_wrist_observation_steps')}")
            print(f"recorded_wrist_observation_steps: {final_state.get('recorded_wrist_observation_steps')}")
        if args.policy_backend == "fastapi-dummy":
            print(f"avg_inference_latency_ms: {final_state.get('avg_inference_latency_ms')}")
        if args.safety_mode != "off":
            print(f"safety_mode: {args.safety_mode}")
            print(f"safety_pause_count: {final_state.get('safety_pause_count', 0)}")
            print(f"safety_resume_count: {final_state.get('safety_resume_count', 0)}")
            print(f"paused_steps: {final_state.get('paused_steps', 0)}")

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
