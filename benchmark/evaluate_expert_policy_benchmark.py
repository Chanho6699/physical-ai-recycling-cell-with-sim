"""Standalone DummyOpenVLAPolicy (scripted expert) benchmark (v0).

Purpose: BEFORE expanding the LeRobot dataset to 60-100 episodes, verify
that DummyOpenVLAPolicy (policy/dummy_openvla_policy.py) -- the scripted
oracle that generates every ground-truth action currently saved by
benchmark/collect_recycling_dataset.py -- is itself an accurate and
stable expert across a balanced grid of object positions, not just the
4 anchor positions collection has used so far.

This script does NOT run SmolVLA, does NOT call collect_recycling_dataset
(no dataset is written), and does NOT modify any production file. It
drives PyBulletPandaBackend + ActionAdapter directly, exactly like
collect_recycling_dataset.run_one_episode() does, but:
  - uses a 3x3 (near/mid/far x left/center/right) position grid instead
    of collect_recycling_dataset's 4 fixed anchors, to actually exercise
    corner/edge conditions;
  - skips camera rendering (DummyOpenVLAPolicy is image-blind -- see its
    module docstring -- so this has zero effect on the actions taken or
    the physics; it only saves wall-clock time here since no dataset
    frame is being written);
  - additionally captures the PRE-clamp raw position delta by transiently
    wrapping DummyOpenVLAPolicy._delta_to_target with a plain Python
    function-level monkeypatch scoped to one instance (mirrors
    collect_recycling_dataset._FrameInstrumentation's existing pattern
    for wrapping bound methods without touching the class/module itself),
    so "clamp 전/후 action" can be compared at the same step.

Workspace-violation bounds are NOT invented for this task: they reuse
the exact box already defined and enforced by
benchmark/run_full_recycling_cell_demo.py's --workspace-bounds default
("-0.1,0.9,-0.7,0.7,0.0,1.0", robot_base frame, see its parse_args()
help text) -- the one place in this codebase that already treats
"end effector left the reachable box" as a named condition
(task_status="aborted_workspace_exceeded"). This script only measures
against that box; it does not abort episodes (DummyOpenVLAPolicy has no
such abort path today).

Run:
  .venv-vla/bin/python -m benchmark.evaluate_expert_policy_benchmark \\
    --output results/expert_policy_benchmark/v1.json
"""

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, JITTER_RADIUS_M, jitter_position
from benchmark.run_full_recycling_cell_demo import _distance_3d, parse_workspace_bounds
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BIN_POSITION = [0.3, 0.35, 0.05]
DEFAULT_OBJECT_TYPE = "plastic_bottle"
DEFAULT_INSTRUCTION_NAME = "ko_full"

# Matches collect_recycling_dataset.py's own production defaults exactly
# (DEFAULT_MAX_STEPS_PER_EPISODE / DEFAULT_STEPS_PER_ACTION) -- this
# benchmark evaluates the SAME expert pipeline configuration that
# actually produces the training dataset, not a hypothetical one.
DEFAULT_MAX_STEPS_PER_EPISODE = 150
DEFAULT_STEPS_PER_ACTION = 40

# 3x3 grid spanning the same x/y range collect_recycling_dataset's 4
# anchors already use (x: 0.27-0.42, y: -0.18..+0.18) plus their exact
# midpoints, so "near/mid/far" and "left/center/right" are each a real,
# reachable point already known to be within this project's demonstrated
# working range -- not an arbitrary extrapolation beyond it.
GRID_X = {"near": 0.27, "mid": 0.345, "far": 0.42}
GRID_Y = {"left": 0.18, "center": 0.00, "right": -0.18}
OBJECT_Z = 0.05  # matches DEFAULT_POSITIONS' z in collect_recycling_dataset.py

GRID_POSITIONS = {
    f"{x_name}_{y_name}": [x_val, y_val, OBJECT_Z]
    for x_name, x_val in GRID_X.items()
    for y_name, y_val in GRID_Y.items()
}

DEFAULT_SEEDS = [0, 1, 2, 3]  # 9 anchors x 4 seeds = 36 episodes (>= the requested 30)

# Reused verbatim from run_full_recycling_cell_demo.py's own
# --workspace-bounds default -- see this module's docstring.
DEFAULT_WORKSPACE_BOUNDS_STR = "-0.1,0.9,-0.7,0.7,0.0,1.0"

# Same object-vs-arm-reach IK-residual heuristic used nowhere else in
# this codebase (no existing constant to reuse) -- see module docstring
# section 3. Flags a step where the achieved EE position ends up more
# than this far from the position apply_command() actually commanded,
# i.e. the arm could not converge onto the requested target within
# steps_per_action, which is a reasonable proxy for "target outside
# reachable workspace / obstructed" given this backend has no built-in
# IK-failure signal (calculateInverseKinematics never raises).
IK_RESIDUAL_VIOLATION_THRESHOLD_M = 0.05


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _classify_failure_reason(final_status: str, ever_held: bool) -> str:
    if final_status == "success":
        return "none (success)"
    if final_status == "released":
        return "released_away_from_bin"
    if final_status == "grasped":
        return "grasped_then_timeout_before_release"
    if ever_held:
        return "held_then_lost_track"
    return "never_grasped_timeout"


class _ClampInstrumentation:
    """Transiently wraps ONE DummyOpenVLAPolicy instance's own
    _delta_to_target bound method to additionally record the pre-clamp
    raw_delta alongside the clamped_delta it already returns -- without
    touching policy/dummy_openvla_policy.py. Mirrors the bound-method
    wrapping pattern collect_recycling_dataset._FrameInstrumentation
    already uses for open_gripper/close_gripper."""

    def __init__(self, policy: DummyOpenVLAPolicy):
        self.policy = policy
        self._original = policy._delta_to_target
        self.last_raw_delta = None
        self.last_clamped_delta = None
        self.last_distance = None

    def __enter__(self):
        def wrapped(current_position, target_position):
            if target_position is None:
                self.last_raw_delta = [0.0, 0.0, 0.0]
                self.last_clamped_delta = [0.0, 0.0, 0.0]
                self.last_distance = 0.0
                return [0.0, 0.0, 0.0], 0.0
            raw_delta = [target_position[axis] - current_position[axis] for axis in range(3)]
            clamped_delta, distance = self._original(current_position, target_position)
            self.last_raw_delta = raw_delta
            self.last_clamped_delta = clamped_delta
            self.last_distance = distance
            return clamped_delta, distance

        self.policy._delta_to_target = wrapped
        return self

    def __exit__(self, *exc_info):
        self.policy._delta_to_target = self._original


def run_episode(
    position_name, position, instruction, instruction_name, bin_position, seed,
    max_steps, steps_per_action, object_type, workspace_bounds,
):
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type(object_type)
    # Actually relocates the PHYSICAL bin, not just what the policy is
    # told -- previously this function's own bin_position parameter only
    # ever reached PolicyInput.bin_position (the policy's target), while
    # PyBulletPandaBackend kept its reset()-default bin position, so
    # open_gripper()'s own place-success check was silently comparing
    # against the WRONG bin whenever a non-default bin_position was ever
    # passed in (never exercised before the v2 dataset task, since every
    # prior caller here always passed the same default bin_position --
    # see this task's chat report on why train80's own single, never-
    # varied physical bin position needed the same fix in
    # collect_recycling_dataset.run_one_episode()).
    backend.set_bin_position(list(bin_position))
    backend.set_object_position(list(position))
    policy = DummyOpenVLAPolicy()
    policy.reset()
    action_adapter = ActionAdapter()

    x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds

    step_rows = []
    first_distance_to_object = None
    ever_held = False
    first_grasp_step = None
    first_close_command_step = None
    distance_at_first_close_command = None
    final_status = "running"
    success = False
    workspace_violations = 0
    ik_residual_violations = 0
    max_ik_residual = 0.0
    consistency_examples = []

    with _ClampInstrumentation(policy) as clamp_instr:
        for step_index in range(max_steps):
            robot_state = backend.get_state()
            object_position = list(robot_state["object_position"])
            ee_position = list(robot_state["end_effector_position"])
            distance_to_object = _distance_3d(ee_position, object_position)
            if first_distance_to_object is None:
                first_distance_to_object = distance_to_object

            policy_input = PolicyInput(
                image=None,
                instruction=instruction,
                robot_state=robot_state,
                task_goal={},
                target_object_position=object_position,
                bin_position=bin_position,
                step_index=step_index,
                phase=policy.phase,
            )
            policy_output = policy.predict_action(policy_input)

            saved_action = list(policy_output.action)  # exactly what collect_recycling_dataset.py would add_frame()
            robot_command = action_adapter.convert(policy_output.action)
            commanded_translation = [robot_command.target_dx, robot_command.target_dy, robot_command.target_dz]

            if robot_command.gripper_command == "close" and first_close_command_step is None:
                first_close_command_step = step_index
                distance_at_first_close_command = distance_to_object

            pre_command_ee = ee_position
            target_ee_commanded = [pre_command_ee[i] + commanded_translation[i] for i in range(3)]

            if len(consistency_examples) < 8:
                consistency_examples.append({
                    "episode_position_name": position_name,
                    "seed": seed,
                    "step": step_index,
                    "phase": policy_output.phase,
                    "raw_delta_pre_clamp": clamp_instr.last_raw_delta,
                    "clamped_delta_post_clamp": clamp_instr.last_clamped_delta[:] if clamp_instr.last_clamped_delta else None,
                    "saved_action_translation": saved_action[0:3],
                    "simulator_command_translation": commanded_translation,
                    "action_dataset_vs_simulator_identical": saved_action[0:3] == commanded_translation,
                })

            robot_state_after = backend.apply_command(robot_command, steps=steps_per_action)
            ee_position_after = list(robot_state_after["end_effector_position"])
            final_status = robot_state_after["task_status"]
            held_now = bool(robot_state_after["held_object"])

            if held_now and not ever_held:
                ever_held = True
                first_grasp_step = step_index

            in_bounds = (
                x_min <= ee_position_after[0] <= x_max
                and y_min <= ee_position_after[1] <= y_max
                and z_min <= ee_position_after[2] <= z_max
            )
            if not in_bounds:
                workspace_violations += 1

            ik_residual = _distance_3d(ee_position_after, target_ee_commanded)
            max_ik_residual = max(max_ik_residual, ik_residual)
            if ik_residual > IK_RESIDUAL_VIOLATION_THRESHOLD_M:
                ik_residual_violations += 1

            step_rows.append({
                "step": step_index,
                "phase": policy_output.phase,
                "distance_to_object": distance_to_object,
                "gripper_command": robot_command.gripper_command,
                "translation": commanded_translation,
                "rotation": [robot_command.target_droll, robot_command.target_dpitch, robot_command.target_dyaw],
                "held_object": held_now,
                "task_status": final_status,
                "in_workspace_bounds": in_bounds,
                "ik_residual_m": ik_residual,
            })

            if final_status == "success" or policy_output.done:
                success = final_status == "success"
                break

    final_robot_state = backend.get_state()
    final_object_position = list(final_robot_state["object_position"])
    final_ee_position = list(final_robot_state["end_effector_position"])
    final_distance_to_object = _distance_3d(final_ee_position, final_object_position)
    final_distance_object_to_bin = _distance_3d(final_object_position, bin_position)
    backend.shutdown()

    translations = [r["translation"] for r in step_rows]
    rotations = [r["rotation"] for r in step_rows]

    return {
        "position_name": position_name,
        "position": list(position),
        "instruction_name": instruction_name,
        "seed": seed,
        "num_steps": len(step_rows),
        "success": success,
        "final_task_status": final_status,
        "pick_success": ever_held,
        "first_grasp_step": first_grasp_step,
        "first_close_command_step": first_close_command_step,
        "distance_at_first_close_command": distance_at_first_close_command,
        "first_distance_to_object": first_distance_to_object,
        "final_distance_to_object": final_distance_to_object,
        "final_distance_object_to_bin": final_distance_object_to_bin,
        "distance_improvement": (
            (first_distance_to_object - final_distance_to_object) if first_distance_to_object is not None else None
        ),
        "workspace_violations": workspace_violations,
        "ik_residual_violations": ik_residual_violations,
        "max_ik_residual_m": max_ik_residual,
        "failure_reason": _classify_failure_reason(final_status, ever_held),
        "translations": translations,
        "rotations": rotations,
        "consistency_examples": consistency_examples,
        "rows": step_rows,
    }


def _axis_stats(vectors, axis):
    values = [v[axis] for v in vectors]
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return {"min": min(values), "max": max(values), "mean": mean, "std": math.sqrt(variance)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default=DEFAULT_OBJECT_TYPE)
    parser.add_argument("--instruction-name", type=str, default=DEFAULT_INSTRUCTION_NAME, choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--workspace-bounds", type=str, default=DEFAULT_WORKSPACE_BOUNDS_STR)
    parser.add_argument("--position-seed-base", type=int, default=7000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]
    workspace_bounds = parse_workspace_bounds(args.workspace_bounds)

    print(f"=== DummyOpenVLAPolicy standalone expert benchmark ===")
    print(f"grid positions ({len(GRID_POSITIONS)}): {GRID_POSITIONS}")
    print(f"seeds: {args.seeds} (jitter radius {JITTER_RADIUS_M} m)")
    print(f"max_steps_per_episode={args.max_steps_per_episode}, steps_per_action={args.steps_per_action}")
    print(f"workspace_bounds={workspace_bounds}")

    episodes = []
    total = len(GRID_POSITIONS) * len(args.seeds)
    n = 0
    start_time = time.time()
    for position_index, (position_name, anchor_position) in enumerate(GRID_POSITIONS.items()):
        for seed in args.seeds:
            n += 1
            jitter_seed = args.position_seed_base + position_index * 1000 + seed
            rng = random.Random(jitter_seed)
            position = jitter_position(anchor_position, rng)
            episode = run_episode(
                position_name, position, instruction, args.instruction_name, DEFAULT_BIN_POSITION, jitter_seed,
                args.max_steps_per_episode, args.steps_per_action, args.object_type, workspace_bounds,
            )
            episodes.append(episode)
            print(
                f"[{n:02d}/{total}] pos={position_name:<12} seed={jitter_seed:5d} "
                f"success={episode['success']} status={episode['final_task_status']:<10} "
                f"steps={episode['num_steps']:3d} pick={episode['pick_success']} "
                f"dist_improve={episode['distance_improvement']:.4f} "
                f"ws_viol={episode['workspace_violations']} ik_viol={episode['ik_residual_violations']}"
            )

    elapsed_s = time.time() - start_time

    all_translations = [t for e in episodes for t in e["translations"]]
    all_rotations = [r for e in episodes for r in e["rotations"]]
    translation_stats = {axis_name: _axis_stats(all_translations, axis) for axis, axis_name in enumerate(["x", "y", "z"])}
    rotation_stats = {axis_name: _axis_stats(all_rotations, axis) for axis, axis_name in enumerate(["roll", "pitch", "yaw"])}

    result = {
        "grid_positions": GRID_POSITIONS,
        "seeds_requested": args.seeds,
        "position_seed_base": args.position_seed_base,
        "instruction_name": args.instruction_name,
        "instruction": instruction,
        "max_steps_per_episode": args.max_steps_per_episode,
        "steps_per_action": args.steps_per_action,
        "object_type": args.object_type,
        "workspace_bounds": list(workspace_bounds),
        "ik_residual_violation_threshold_m": IK_RESIDUAL_VIOLATION_THRESHOLD_M,
        "num_episodes": len(episodes),
        "wall_clock_s": elapsed_s,
        "timestamp": datetime.now().isoformat(),
        "translation_axis_stats": translation_stats,
        "rotation_axis_stats": rotation_stats,
        "episodes": episodes,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    success_rate = sum(1 for e in episodes if e["success"]) / len(episodes)
    pick_rate = sum(1 for e in episodes if e["pick_success"]) / len(episodes)
    print()
    print(f"=== Done: {len(episodes)} episodes, success_rate={success_rate:.2%}, pick_rate={pick_rate:.2%}, wall_clock={elapsed_s:.1f}s ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
