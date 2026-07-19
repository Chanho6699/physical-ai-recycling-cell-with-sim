"""Root Cause Analysis Phase 1 (see this task's chat report): instrumented,
paired 3-way rollout comparison -- Expert (DummyOpenVLAPolicy) / v2-2000
checkpoint / v2-4000 checkpoint -- on the SAME 40 fixed-center episodes
(benchmark/checkpoint_eval_positions.py's existing 20 train_distribution +
20 validation_distribution set, already used for every other checkpoint
comparison this session).

Read-only instrumentation ONLY: no training, no dataset writes, no
decoder/threshold/action-chunk/physics changes. The extra per-step
diagnostics (finger/object contact, object velocity) are read directly
off PyBulletPandaBackend's existing public attributes (client_id,
robot_id, _object_id, finger_joint_indices) via raw pybullet calls in
THIS file -- robot_sim/pybullet_panda_backend.py itself is not modified.

Each of the 3 policies is run as a SEPARATE invocation of this script
(--policy expert / v2_2000 / v2_4000), since v2_2000 and v2_4000 need
the vla_server /load_model'd with a different checkpoint each -- reusing
one already-running server for both would require a reload per episode.
Episode order/positions/seeds are identical across all 3 invocations
(byte-identical to benchmark/checkpoint_eval_positions.py's set), so the
3 output JSONs are directly joinable by (anchor_name, seed) for the
paired per-episode comparison this task's analysis needs.

Run:
  .venv-vla/bin/python -m benchmark.diagnose_expert_vs_v2_checkpoints \\
    --policy expert --output results/gripper_diagnosis/rootcause_expert.json

  # (server must already be /load_model'd with the matching checkpoint)
  .venv-vla/bin/python -m benchmark.diagnose_expert_vs_v2_checkpoints \\
    --policy v2_2000 --output results/gripper_diagnosis/rootcause_v2_2000.json
  .venv-vla/bin/python -m benchmark.diagnose_expert_vs_v2_checkpoints \\
    --policy v2_4000 --output results/gripper_diagnosis/rootcause_v2_4000.json
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pybullet as p

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.checkpoint_eval_positions import build_train_eval_positions, build_validation_eval_positions
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS
from benchmark.run_checkpoint_comparison_benchmark import (
    DEFAULT_BIN_POSITION,
    DEFAULT_MAX_POLICY_STEPS,
    DEFAULT_STEPS_PER_ACTION,
)
from benchmark.run_vla_action_direction_diagnostic import build_robot_state, image_hash, resolve
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

# Documented, fixed thresholds for event extraction (not tuned/searched --
# picked once, up front, and reused identically for all 3 policies so
# episode classification never differs by policy-specific tuning):
APPROACH_THRESHOLD_M = 0.10  # "got near the object" -- looser than the backend's own GRASP_THRESHOLD (0.05)
LIFT_HEIGHT_DELTA_M = 0.05  # object risen >= this much above ITS OWN initial z, while held, counts as "lifted"


def _distance_3d(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _diagnostic_extras(backend: PyBulletPandaBackend) -> dict:
    """Read-only extras not exposed by PyBulletPandaBackend.get_state():
    finger/object contact (via p.getContactPoints, robot linkIndexA in
    {9,10} = finger_joint_indices means a FINGER specifically touched the
    object -- pybullet indexes a link the same as the joint connecting it
    to its parent, the same convention this codebase already relies on
    for end_effector_link_index/arm_joint_indices elsewhere) and object
    linear velocity (via p.getBaseVelocity). Reads backend's own
    client_id/robot_id/_object_id/finger_joint_indices attributes
    directly -- does not call or alter any backend method."""
    contact_points = p.getContactPoints(
        bodyA=backend.robot_id, bodyB=backend._object_id, physicsClientId=backend.client_id
    )
    finger_link_indices = set(backend.finger_joint_indices)
    finger_contact = any(cp[3] in finger_link_indices for cp in contact_points)
    velocity, _angular_velocity = p.getBaseVelocity(backend._object_id, physicsClientId=backend.client_id)
    finger_states = p.getJointStates(backend.robot_id, backend.finger_joint_indices, physicsClientId=backend.client_id)
    return {
        "contact_detected": len(contact_points) > 0,
        "finger_contact_detected": finger_contact,
        "num_contact_points": len(contact_points),
        "object_linear_velocity": [float(v) for v in velocity],
        "object_linear_speed": float(np.linalg.norm(velocity)),
        "finger_left_qpos": float(finger_states[0][0]),
        "finger_right_qpos": float(finger_states[1][0]),
    }


def run_episode(
    policy_kind: str, policy, anchor_name: str, position: list, seed: int, split_tag: str,
    instruction: str, instruction_name: str, max_steps: int, steps_per_action: int, object_type: str, label: str,
) -> dict:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type(object_type)
    backend.set_object_position(list(position))
    backend.set_bin_position(list(DEFAULT_BIN_POSITION))
    policy.reset()

    initial_object_z = position[2]
    time_step = backend.time_step
    cumulative_physics_steps = 0

    step_rows = []
    ever_held = False
    first_grasp_step = None
    release_step = None
    drop_step = None
    first_approach_step = None
    first_close_cmd_step = None
    first_contact_step = None
    first_lift_step = None
    min_distance = None
    min_distance_step = None
    final_status = "running"
    success = False

    for step_index in range(max_steps):
        obs_time = time.perf_counter()
        robot_state, _state_8d, object_position = build_robot_state(backend)
        ee_position = list(robot_state["ee_position"])
        distance_to_object = _distance_3d(ee_position, object_position)
        extras_before = _diagnostic_extras(backend)

        if min_distance is None or distance_to_object < min_distance:
            min_distance = distance_to_object
            min_distance_step = step_index
        if first_approach_step is None and distance_to_object <= APPROACH_THRESHOLD_M:
            first_approach_step = step_index
        if first_contact_step is None and extras_before["finger_contact_detected"]:
            first_contact_step = step_index

        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()

        policy_input_kwargs = dict(
            image=main_image, instruction=instruction, robot_state=robot_state, task_goal={},
            target_object_position=object_position, bin_position=list(DEFAULT_BIN_POSITION),
            step_index=step_index, phase=policy.phase,
        )
        if policy_kind != "expert":
            policy_input_kwargs["images_by_role"] = {"main": main_image, "wrist": wrist_image}
            policy_input_kwargs["seed"] = seed
        policy_input = PolicyInput(**policy_input_kwargs)

        inference_start = time.perf_counter()
        policy_output = policy.predict_action(policy_input)
        inference_end = time.perf_counter()
        info = policy_output.info or {}

        action = [float(v) for v in policy_output.action]
        translation = action[0:3]
        rotation = action[3:6]
        gripper_raw = action[6]
        gripper_interpreted = "close" if gripper_raw >= 0.5 else "open"

        robot_command = ActionAdapter().convert(action)

        gripper_state_before = backend.get_state()["gripper_state"]
        apply_start = time.perf_counter()
        robot_state_after = backend.apply_command(robot_command, steps=steps_per_action)
        apply_end = time.perf_counter()
        cumulative_physics_steps += steps_per_action
        sim_time_s = cumulative_physics_steps * time_step

        gripper_state_after = robot_state_after["gripper_state"]
        gripper_actuated_this_step = gripper_state_before != gripper_state_after

        held_now = bool(robot_state_after["held_object"])
        final_status = robot_state_after["task_status"]

        if robot_command.gripper_command == "close" and first_close_cmd_step is None:
            first_close_cmd_step = step_index
        if held_now and not ever_held:
            ever_held = True
            first_grasp_step = step_index
        if (
            ever_held and held_now and first_lift_step is None
            and robot_state_after["object_position"][2] >= initial_object_z + LIFT_HEIGHT_DELTA_M
        ):
            first_lift_step = step_index
        if ever_held and not held_now and release_step is None and step_index != first_grasp_step:
            release_step = step_index
            if final_status != "success":
                drop_step = step_index

        extras_after = _diagnostic_extras(backend)

        step_rows.append({
            "step": step_index,
            "sim_time_s": sim_time_s,
            "object_position": list(object_position),
            "ee_position": ee_position,
            "distance_to_object": distance_to_object,
            "object_height": float(object_position[2]),
            "object_linear_speed": extras_before["object_linear_speed"],
            "gripper_width": robot_state.get("gripper_width"),
            "contact_detected": extras_before["contact_detected"],
            "finger_contact_detected": extras_before["finger_contact_detected"],
            "held_object": bool(robot_state.get("held_object", False)),
            "policy_translation_m": translation,
            "policy_rotation_axis_angle_rad": rotation,
            "policy_gripper_raw": gripper_raw,
            "policy_gripper_interpreted": gripper_interpreted,
            "applied_command": {
                "dx": robot_command.target_dx, "dy": robot_command.target_dy, "dz": robot_command.target_dz,
                "droll": robot_command.target_droll, "dpitch": robot_command.target_dpitch,
                "dyaw": robot_command.target_dyaw, "gripper_command": robot_command.gripper_command,
            },
            "gripper_actuated_this_step": gripper_actuated_this_step,
            "held_object_after": bool(held_now),
            "task_status_after": final_status,
            "object_height_after": float(robot_state_after["object_position"][2]),
            "object_linear_speed_after": extras_after["object_linear_speed"],
            "contact_detected_after": extras_after["contact_detected"],
            "finger_contact_detected_after": extras_after["finger_contact_detected"],
            "obs_time": obs_time,
            "inference_start": inference_start,
            "inference_end": inference_end,
            "apply_start": apply_start,
            "apply_end": apply_end,
            "inference_latency_s": inference_end - inference_start,
            "obs_to_apply_latency_s": apply_end - obs_time,
            "server_inference_latency_ms": info.get("inference_latency_ms"),
            "main_image_hash": image_hash(main_image),
        })

        if final_status == "success":
            success = True
            break

    backend.shutdown()

    close_step = None
    close_distance = None
    for row in step_rows:
        if row["applied_command"]["gripper_command"] == "close":
            close_step = row["step"]
            close_distance = row["distance_to_object"]
            break

    held_duration = None
    if first_grasp_step is not None:
        end_step = release_step if release_step is not None else (len(step_rows) - 1)
        held_duration = end_step - first_grasp_step

    return {
        "label": label,
        "policy_kind": policy_kind,
        "split_tag": split_tag,
        "anchor_name": anchor_name,
        "position": list(position),
        "seed": seed,
        "instruction_name": instruction_name,
        "num_steps": len(step_rows),
        "success": success,
        "final_task_status": final_status,
        "pick_success": ever_held,
        "min_distance": min_distance,
        "min_distance_step": min_distance_step,
        "first_approach_step": first_approach_step,
        "first_close_cmd_step": first_close_cmd_step,
        "first_close_distance": close_distance,
        "first_contact_step": first_contact_step,
        "first_grasp_step": first_grasp_step,
        "first_lift_step": first_lift_step,
        "held_duration_steps": held_duration,
        "release_step": release_step,
        "drop_step": drop_step,
        "rows": step_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=["expert", "v2_2000", "v2_4000"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--real-vla-config", type=str, default="configs/real_vla_backend_config.json")
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--max-policy-steps", type=int, default=DEFAULT_MAX_POLICY_STEPS)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]

    train_eval_positions = build_train_eval_positions()
    validation_eval_positions = build_validation_eval_positions()
    all_positions = [("train_distribution", p_) for p_ in train_eval_positions] + [
        ("validation_distribution", p_) for p_ in validation_eval_positions
    ]

    label = args.policy
    if args.policy == "expert":
        policy = DummyOpenVLAPolicy()
        model_id_or_path = "DummyOpenVLAPolicy (scripted oracle)"
    else:
        policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)
        health = policy.check_health()
        print(f"server health: {health}")
        if health.get("model_status") != "loaded":
            raise RuntimeError(f"Server model_status={health.get('model_status')!r}, expected 'loaded'.")
        if not (health.get("compatibility") or {}).get("passed"):
            raise RuntimeError(f"Server compatibility.passed is not True: {health.get('compatibility')}")
        model_id_or_path = health.get("model_id_or_path")
        expected_step = "002000" if args.policy == "v2_2000" else "004000"
        if expected_step not in str(model_id_or_path):
            raise RuntimeError(
                f"--policy {args.policy} expects a server loaded with checkpoint step {expected_step}, "
                f"but server reports model_id_or_path={model_id_or_path!r}. Reload the correct checkpoint first."
            )

    print(f"=== Root-cause diagnosis rollout -- policy={args.policy!r} model={model_id_or_path!r} ===")
    print(f"episodes: {len(all_positions)} (train_distribution=20, validation_distribution=20)")

    episodes = []
    start_time = time.time()
    for n, (split_tag, pos) in enumerate(all_positions, start=1):
        episode = run_episode(
            args.policy, policy, pos["anchor_name"], pos["position"], pos["seed"], split_tag,
            instruction, args.instruction_name, args.max_policy_steps, args.steps_per_action,
            args.object_type, label,
        )
        episodes.append(episode)
        print(
            f"[{n:02d}/{len(all_positions)}] split={split_tag:20s} anchor={pos['anchor_name']:16s} seed={pos['seed']:7d} "
            f"success={episode['success']} status={episode['final_task_status']:<10} steps={episode['num_steps']:3d} "
            f"pick={episode['pick_success']} first_close={episode['first_close_cmd_step']} "
            f"close_dist={episode['first_close_distance']} first_grasp={episode['first_grasp_step']} "
            f"first_contact={episode['first_contact_step']} held_dur={episode['held_duration_steps']}"
        )

    elapsed_s = time.time() - start_time
    result = {
        "policy_kind": args.policy,
        "model_id_or_path": model_id_or_path,
        "instruction_name": args.instruction_name,
        "instruction": instruction,
        "max_policy_steps": args.max_policy_steps,
        "steps_per_action": args.steps_per_action,
        "object_type": args.object_type,
        "approach_threshold_m": APPROACH_THRESHOLD_M,
        "lift_height_delta_m": LIFT_HEIGHT_DELTA_M,
        "num_episodes": len(episodes),
        "wall_clock_s": elapsed_s,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    success_rate = sum(1 for e in episodes if e["success"]) / len(episodes)
    pick_rate = sum(1 for e in episodes if e["pick_success"]) / len(episodes)
    print(f"\n=== Done: {len(episodes)} episodes, success_rate={success_rate:.2%}, pick_rate={pick_rate:.2%}, wall_clock={elapsed_s:.1f}s ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
