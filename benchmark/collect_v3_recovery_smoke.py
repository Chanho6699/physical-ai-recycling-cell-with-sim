"""V3 Recovery Data Collector -- Fast Smoke Validation (see this task's
chat report). Collects a 50-episode SMOKE dataset that targets the three
concrete v2 gaps found in results/dataset_analysis/v2_approach_coverage.md:
(1) EE initial position identical in 160/160 v2 episodes, (2) 0/160
recovery trajectories, (3) only 6.71% of approach frames within 0.05m and
~1 frame of near-target dwelling before every close.

Reuses benchmark/collect_recycling_dataset.py's FEATURES/DEFAULT_INSTRUCTIONS,
benchmark/v2_dataset_positions.py's OBJECT_ANCHORS/BIN_POSITIONS/OBJECT_Z,
action_adapter.adapter_v0.ActionAdapter, and PyBulletPandaBackend/
DummyOpenVLAPolicy UNCHANGED except DummyOpenVLAPolicy's new (default-off)
stabilization_steps parameter (see its own module for why that one, small,
backward-compatible addition was needed).

Data-integrity rule (see chat report item 2): every recorded frame's
"action" is the EXPERT'S OWN, freshly recomputed corrective action for
that frame's REAL (possibly perturbed) observation -- never noise added to
a stored action, never an action recorded against a stale/mismatched
observation. This holds structurally because:
  - EE-initial-position randomization is a PRE-episode move (backend.
    move_end_effector_to(), an existing, unmodified production method) that
    completes and settles BEFORE the recorded frame loop starts -- it is
    never itself added as a dataset frame.
  - Controlled perturbation (mode B) is likewise an INTERSTITIAL,
    unrecorded backend.move_end_effector_to() call between two recorded
    frames -- the very next recorded frame reads the ACTUAL post-
    perturbation robot state via the same backend.get_state()/
    get_libero_observation_state() calls run_one_episode() already uses,
    and DummyOpenVLAPolicy computes its action from THAT real state
    (policy_input.robot_state["end_effector_position"]) -- nothing here
    is faked or offset only in the action.
  - Near-target stabilization (mode C) is DummyOpenVLAPolicy's own new
    "stabilize" phase, computing a genuine (smaller-magnitude) corrective
    delta toward the grasp target from the real current EE position each
    frame -- not a copied zero-action.

Run:
  .venv-vla/bin/python -m benchmark.collect_v3_recovery_smoke --episodes 5 --dry-run-tag preflight
  .venv-vla/bin/python -m benchmark.collect_v3_recovery_smoke --episodes 50
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, FEATURES
from benchmark.v2_dataset_positions import BIN_POSITIONS, OBJECT_ANCHORS, OBJECT_ANCHOR_NAMES, OBJECT_Z
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO_ID = "local/recycling_cell_v3_recovery_smoke50"
DEFAULT_ROOT = "datasets/recycling_v3_recovery_smoke50"
DEFAULT_FPS = 10
DEFAULT_OBJECT_TYPE = "plastic_bottle"
DEFAULT_MAX_STEPS_PER_EPISODE = 150
DEFAULT_STEPS_PER_ACTION = 40

# --- Mode A: EE initial-position randomization ---
EE_INIT_OFFSET_RANGE_M = (0.04, 0.04, 0.03)  # (x, y, z) half-widths
EE_INIT_SETTLE_STEPS = 100
EE_INIT_TOLERANCE_M = 0.01  # post-settle EE must land within this of the requested target

# --- Mode B: controlled perturbation ---
PERTURBATION_TRIGGER_RANGE_M = (0.05, 0.15)
PERTURBATION_OFFSET_RANGE_M = (0.015, 0.035)
OVERSHOOT_RANGE_M = (0.015, 0.030)
PERTURBATION_SETTLE_STEPS = 40

# --- Mode C: near-target stabilization ---
STABILIZATION_STEPS = 4
STABILIZATION_STEP_SIZE_M = 0.01

OBJECT_JITTER_RADIUS_M = 0.015
BIN_NAMES = list(BIN_POSITIONS.keys())


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _jitter_object_xy(x: float, y: float, z: float, rng: random.Random) -> list:
    angle = rng.uniform(0, 2 * math.pi)
    radius = rng.uniform(0, OBJECT_JITTER_RADIUS_M)
    return [x + radius * math.cos(angle), y + radius * math.sin(angle), z]


def build_plan(num_episodes: int, seed_base: int) -> list:
    """50 planned episode specs -- 10 ee_init_only / 20 perturb_xy
    (split ~7/7/6 across x/y/diagonal) / 10 perturb_overshoot /
    10 near_target -- exactly the 'recommended composition' from this
    task's chat report. Deterministic given seed_base, so a preflight
    run with --episodes 5 exercises a representative slice of all 4
    groups (not just the first group)."""
    groups = (
        [("ee_init_only", None) for _ in range(10)]
        + [("perturb_xy", "x") for _ in range(7)]
        + [("perturb_xy", "y") for _ in range(7)]
        + [("perturb_xy", "diagonal") for _ in range(6)]
        + [("perturb_overshoot", "overshoot") for _ in range(10)]
        + [("near_target", None) for _ in range(10)]
    )
    rng = random.Random(seed_base)
    rng.shuffle(groups)

    plan = []
    for i, (scenario_group, perturb_type) in enumerate(groups):
        anchor_name = OBJECT_ANCHOR_NAMES[i % len(OBJECT_ANCHOR_NAMES)]
        bin_name = BIN_NAMES[i % len(BIN_NAMES)]
        plan.append({
            "plan_index": i,
            "scenario_group": scenario_group,
            "perturbation_type": perturb_type,
            "anchor_name": anchor_name,
            "bin_name": bin_name,
            "seed": seed_base + i * 100,
        })
    if num_episodes < len(plan):
        plan = plan[:num_episodes]
    return plan


RECOVERY_TYPES_EVEN5 = ["x", "y", "diagonal", "overshoot", "near_target"]


def build_recovery_plan_even5(num_episodes: int, seed_base: int) -> list:
    """General-purpose recovery plan for benchmark/build_v3_dataset.py
    (see this task's chat report): splits num_episodes as evenly as
    possible across the 5 named recovery types (x/y/diagonal/overshoot/
    near_target -- the remainder, if num_episodes % 5 != 0, is spread
    over the first few types, one extra each). Every entry gets EE-init
    randomization (unconditional in run_recovery_episode() regardless of
    scenario_group -- see that function), matching this task's '모든
    episode에 초기 위치 랜덤화를 함께 적용' allowance; there is no
    separate 'ee_init_only' bucket here (unlike build_plan()'s smoke-50
    composition, which used one as a dedicated control group) because
    this task's spec lists exactly 5 recovery types to distribute
    evenly, not 6."""
    base_count, remainder = divmod(num_episodes, len(RECOVERY_TYPES_EVEN5))
    counts = [base_count + (1 if i < remainder else 0) for i in range(len(RECOVERY_TYPES_EVEN5))]

    groups = []
    for recovery_type, count in zip(RECOVERY_TYPES_EVEN5, counts):
        scenario_group = "near_target" if recovery_type == "near_target" else f"perturb_{recovery_type}"
        perturbation_type = None if recovery_type == "near_target" else recovery_type
        groups.extend([(scenario_group, perturbation_type) for _ in range(count)])

    rng = random.Random(seed_base)
    rng.shuffle(groups)

    plan = []
    for i, (scenario_group, perturbation_type) in enumerate(groups):
        anchor_name = OBJECT_ANCHOR_NAMES[i % len(OBJECT_ANCHOR_NAMES)]
        bin_name = BIN_NAMES[i % len(BIN_NAMES)]
        plan.append({
            "plan_index": i,
            "scenario_group": scenario_group,
            "perturbation_type": perturbation_type,
            "anchor_name": anchor_name,
            "bin_name": bin_name,
            "seed": seed_base + i * 100,
        })
    return plan


def randomize_ee_initial_pose(backend: PyBulletPandaBackend, rng: random.Random) -> dict:
    """PRE-episode move only -- NOT recorded as a dataset frame (see this
    module's own data-integrity docstring). Returns metadata; raises
    ValueError if the settled EE position isn't within EE_INIT_TOLERANCE_M
    of the requested target (caller treats that as a failed attempt, per
    this task's 'physically unstable teleport frame 금지' rule)."""
    default_ee_position = list(backend.get_state()["end_effector_position"])
    dx = rng.uniform(-EE_INIT_OFFSET_RANGE_M[0], EE_INIT_OFFSET_RANGE_M[0])
    dy = rng.uniform(-EE_INIT_OFFSET_RANGE_M[1], EE_INIT_OFFSET_RANGE_M[1])
    dz = rng.uniform(-EE_INIT_OFFSET_RANGE_M[2], EE_INIT_OFFSET_RANGE_M[2])
    target = [default_ee_position[0] + dx, default_ee_position[1] + dy, default_ee_position[2] + dz]

    backend.move_end_effector_to(target, target_orientation=backend.default_orientation, steps=EE_INIT_SETTLE_STEPS)
    actual_position = list(backend.get_state()["end_effector_position"])

    if not all(math.isfinite(v) for v in actual_position):
        raise ValueError(f"EE init randomization produced non-finite position: {actual_position}")
    settle_error = _dist(actual_position, target)
    if settle_error > EE_INIT_TOLERANCE_M:
        raise ValueError(f"EE init randomization did not settle (error={settle_error:.4f}m > {EE_INIT_TOLERANCE_M}m): target={target} actual={actual_position}")

    return {
        "default_ee_position": default_ee_position,
        "requested_offset": [dx, dy, dz],
        "requested_target": target,
        "actual_initial_ee_position": actual_position,
        "settle_error_m": settle_error,
    }


def apply_controlled_perturbation(
    backend: PyBulletPandaBackend, object_position: list, perturbation_type: str, step_index: int, rng: random.Random,
) -> dict:
    """INTERSTITIAL, unrecorded backend move (see module docstring) --
    the very next iteration of the caller's recorded loop reads the real
    post-perturbation state and lets the (unmodified) expert recompute a
    genuine corrective action from it."""
    ee_before = list(backend.get_state()["end_effector_position"])
    distance_before = _dist(ee_before, object_position)

    if perturbation_type == "x":
        mag = rng.uniform(*PERTURBATION_OFFSET_RANGE_M) * rng.choice([-1, 1])
        requested_offset = [mag, 0.0, 0.0]
    elif perturbation_type == "y":
        mag = rng.uniform(*PERTURBATION_OFFSET_RANGE_M) * rng.choice([-1, 1])
        requested_offset = [0.0, mag, 0.0]
    elif perturbation_type == "diagonal":
        mag = rng.uniform(*PERTURBATION_OFFSET_RANGE_M)
        sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])
        requested_offset = [sx * mag / math.sqrt(2), sy * mag / math.sqrt(2), 0.0]
    elif perturbation_type == "overshoot":
        overshoot_mag = rng.uniform(*OVERSHOOT_RANGE_M)
        direction = [object_position[i] - ee_before[i] for i in range(2)]
        norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2) or 1.0
        unit = [direction[0] / norm, direction[1] / norm]
        travel = distance_before + overshoot_mag
        requested_offset = [unit[0] * travel - 0.0, unit[1] * travel - 0.0, 0.0]
        # (overshoot's "offset" is naturally large -- it re-travels the
        # full remaining approach distance plus the overshoot margin, not
        # a small perturbation delta like x/y/diagonal -- reported as-is.)
    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    target = [ee_before[i] + requested_offset[i] for i in range(3)]
    backend.move_end_effector_to(target, target_orientation=backend.default_orientation, steps=PERTURBATION_SETTLE_STEPS)
    ee_after = list(backend.get_state()["end_effector_position"])
    if not all(math.isfinite(v) for v in ee_after):
        raise ValueError(f"Perturbation produced non-finite EE position: {ee_after}")

    return {
        "perturbation_type": perturbation_type,
        "perturbation_step": step_index,
        "requested_offset": requested_offset,
        "actual_offset": [ee_after[i] - ee_before[i] for i in range(3)],
        "pre_perturbation_distance": distance_before,
        "post_perturbation_distance": _dist(ee_after, object_position),
    }


def run_recovery_episode(dataset, plan_entry: dict, instruction: str, instruction_name: str, max_steps: int, steps_per_action: int, object_type: str) -> dict:
    rng = random.Random(plan_entry["seed"])
    anchor_x, anchor_y = OBJECT_ANCHORS[plan_entry["anchor_name"]]
    object_position = _jitter_object_xy(anchor_x, anchor_y, OBJECT_Z, rng)
    bin_position = list(BIN_POSITIONS[plan_entry["bin_name"]])

    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type(object_type)
    backend.set_bin_position(bin_position)
    backend.set_object_position(object_position)

    ee_init_meta = randomize_ee_initial_pose(backend, rng)

    stabilization_steps = STABILIZATION_STEPS if plan_entry["scenario_group"] == "near_target" else 0
    policy = DummyOpenVLAPolicy(stabilization_steps=stabilization_steps, stabilization_step_size=STABILIZATION_STEP_SIZE_M)
    policy.reset()
    action_adapter = ActionAdapter()

    perturbation_type = plan_entry["perturbation_type"]
    perturbation_applied = False
    perturbation_meta = None
    max_distance_after_perturbation = None
    recovery_completion_step = None
    correction_step_count = None

    near_target_entry_step = None
    close_step = None
    close_distance = None

    success = False
    num_frames = 0
    final_status = "running"
    final_phase = policy.phase

    for step_index in range(max_steps):
        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()
        state_8d = backend.get_libero_observation_state()
        robot_state = backend.get_state()
        ee_position = robot_state["end_effector_position"]
        object_position_now = robot_state["object_position"]
        distance = _dist(ee_position, object_position_now)

        if (
            perturbation_type is not None and not perturbation_applied
            and policy.phase == "move_to_object"
            and PERTURBATION_TRIGGER_RANGE_M[0] <= distance <= PERTURBATION_TRIGGER_RANGE_M[1]
        ):
            perturbation_meta = apply_controlled_perturbation(backend, object_position_now, perturbation_type, step_index, rng)
            perturbation_applied = True
            max_distance_after_perturbation = perturbation_meta["post_perturbation_distance"]
            # Re-read state AFTER the (unrecorded) perturbation move before
            # continuing this same step -- the frame we're about to record
            # below must reflect the REAL post-perturbation observation.
            robot_state = backend.get_state()
            state_8d = backend.get_libero_observation_state()
            main_image = backend.render_main_camera()
            wrist_image = backend.render_wrist_camera()
            ee_position = robot_state["end_effector_position"]
            distance = _dist(ee_position, object_position_now)

        if perturbation_applied and recovery_completion_step is None and perturbation_meta is not None:
            max_distance_after_perturbation = max(max_distance_after_perturbation, distance)
            if distance <= perturbation_meta["pre_perturbation_distance"]:
                recovery_completion_step = step_index
                correction_step_count = step_index - perturbation_meta["perturbation_step"]

        if policy.phase == "stabilize" and near_target_entry_step is None:
            near_target_entry_step = step_index

        policy_input = PolicyInput(
            image=main_image, instruction=instruction, robot_state=robot_state, task_goal={},
            target_object_position=object_position_now, bin_position=bin_position,
            step_index=step_index, phase=policy.phase,
        )
        policy_output = policy.predict_action(policy_input)
        robot_command = action_adapter.convert(policy_output.action)

        if robot_command.gripper_command == "close" and close_step is None:
            close_step = step_index
            close_distance = distance

        dataset.add_frame({
            "observation.images.image": main_image,
            "observation.images.image2": wrist_image,
            "observation.state": np.array(state_8d, dtype=np.float32),
            "action": np.array(policy_output.action, dtype=np.float32),
            "task": instruction,
        })
        num_frames += 1

        robot_state_after = backend.apply_command(robot_command, steps=steps_per_action)
        final_status = robot_state_after["task_status"]
        final_phase = policy.phase
        if final_status == "success" or policy_output.done:
            success = final_status == "success"
            break

    backend.shutdown()

    return {
        "success": success, "num_frames": num_frames, "final_status": final_status, "final_phase": final_phase,
        "object_position": object_position, "bin_position": bin_position,
        "ee_init": ee_init_meta,
        "perturbation": perturbation_meta,
        "max_distance_after_perturbation": max_distance_after_perturbation,
        "recovery_completion_step": recovery_completion_step,
        "correction_step_count": correction_step_count,
        "stabilization_steps": stabilization_steps,
        "near_target_entry_step": near_target_entry_step,
        "close_step": close_step,
        "close_distance": close_distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    args = parser.parse_args()

    max_attempts = args.max_attempts or max(args.episodes + 20, int(args.episodes * 1.4))
    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]

    root = resolve(args.root)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=DEFAULT_FPS, features=FEATURES, root=str(root),
        robot_type="franka_panda_pybullet", use_videos=False,
    )

    plan = build_plan(args.episodes, args.seed_base)
    retry_pool = list(plan)
    manifest_path = root / "collection_manifest.jsonl"
    failed_path = root / "failed_attempts.jsonl"

    saved = 0
    attempt = 0
    crashes = 0
    print(f"=== Collecting {args.episodes} v3 recovery-smoke episodes -> {root} ===")
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file, \
             open(failed_path, "w", encoding="utf-8") as failed_file:
            plan_iter = iter(retry_pool)
            while saved < args.episodes and attempt < max_attempts:
                try:
                    plan_entry = next(plan_iter)
                except StopIteration:
                    # Ran out of planned slots but still short -- retry with
                    # fresh seeds cycling back through the same plan.
                    retry_pool = [
                        {**p, "seed": p["seed"] + 555 * (attempt + 1)} for p in plan
                    ]
                    plan_iter = iter(retry_pool)
                    plan_entry = next(plan_iter)

                attempt += 1
                try:
                    result = run_recovery_episode(
                        dataset, plan_entry, instruction, args.instruction_name,
                        DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION, DEFAULT_OBJECT_TYPE,
                    )
                except (ValueError, RuntimeError) as exc:
                    dataset.clear_episode_buffer()
                    failed_file.write(json.dumps({
                        "attempt": attempt, "plan_index": plan_entry["plan_index"],
                        "scenario_group": plan_entry["scenario_group"], "seed": plan_entry["seed"],
                        "reason": str(exc),
                    }) + "\n")
                    failed_file.flush()
                    crashes += 1
                    print(f"[attempt {attempt:04d}] plan={plan_entry['plan_index']:2d} scenario={plan_entry['scenario_group']:16s} CRASH: {exc}")
                    continue

                if result["success"]:
                    dataset.save_episode()
                    manifest_file.write(json.dumps({
                        "attempt": attempt, "episode_index": saved,
                        "scenario_group": plan_entry["scenario_group"],
                        "perturbation_type": plan_entry["perturbation_type"],
                        "object_anchor_name": plan_entry["anchor_name"], "bin_name": plan_entry["bin_name"],
                        "position": result["object_position"], "bin_position": result["bin_position"],
                        "seed": plan_entry["seed"], "instruction_name": args.instruction_name, "instruction": instruction,
                        "success": True, "final_status": result["final_status"], "final_phase": result["final_phase"],
                        "num_frames": result["num_frames"], "saved": True,
                        "ee_init_requested_offset": result["ee_init"]["requested_offset"],
                        "ee_init_actual_position": result["ee_init"]["actual_initial_ee_position"],
                        "ee_init_settle_error_m": result["ee_init"]["settle_error_m"],
                        "perturbation": result["perturbation"],
                        "max_distance_after_perturbation": result["max_distance_after_perturbation"],
                        "recovery_completion_step": result["recovery_completion_step"],
                        "correction_step_count": result["correction_step_count"],
                        "stabilization_steps": result["stabilization_steps"],
                        "near_target_entry_step": result["near_target_entry_step"],
                        "close_step": result["close_step"], "close_distance": result["close_distance"],
                    }) + "\n")
                    manifest_file.flush()
                    saved += 1
                else:
                    dataset.clear_episode_buffer()
                    failed_file.write(json.dumps({
                        "attempt": attempt, "plan_index": plan_entry["plan_index"],
                        "scenario_group": plan_entry["scenario_group"], "seed": plan_entry["seed"],
                        "reason": f"episode_did_not_succeed: final_status={result['final_status']}",
                    }) + "\n")
                    failed_file.flush()

                print(
                    f"[attempt {attempt:04d}] plan={plan_entry['plan_index']:2d} scenario={plan_entry['scenario_group']:16s} "
                    f"success={result['success']} status={result['final_status']:<10} frames={result['num_frames']:3d} "
                    f"saved={saved}/{args.episodes}"
                )
    finally:
        dataset.finalize()

    run_summary = {
        "attempted_episodes": attempt, "saved_episodes": saved, "collection_crashes": crashes,
        "expert_pick_success_rate": saved / attempt if attempt else None,
        "target_episodes": args.episodes,
    }
    with open(root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\n=== Done: {saved}/{args.episodes} saved, {attempt} attempts, {crashes} crashes ===")
    print(f"Dataset root: {root}")
    print(f"Manifest: {manifest_path}")
    print(f"Failed attempts: {failed_path}")
    print(f"Run summary: {root / 'run_summary.json'}")


if __name__ == "__main__":
    main()
