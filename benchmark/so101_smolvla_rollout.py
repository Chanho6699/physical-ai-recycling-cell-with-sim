"""SO-101 SmolVLA sanity checkpoint MODEL-ONLY closed-loop rollout (see
this task's chat report, section 10). Loads the fine-tuned checkpoint
directly (same load path as benchmark/so101_smolvla_checkpoint_inference_eval.py
-- SmolVLAPolicy.from_pretrained() + make_pre_post_processors(), NOT
vla_server/vla_adapters/policy_semantics, which are 100% hardcoded to
the Panda/LIBERO embodiment) and drives robot_sim.so101_pybullet_backend
directly with the model's OWN raw absolute joint-target actions --
NO scripted expert waypoints are ever consulted or blended in during
the rollout (this task's own "expert fallback 금지, scripted expert와
섞지 않음").

Reuses (does NOT reimplement):
  - benchmark.so101_smolvla_checkpoint_inference_eval's own
    load_policy_and_processors()/predict_action() -- the SAME checkpoint
    load and inference call already validated end-to-end.
  - benchmark.so101_dataset_schema's own pack_state()/SO101_JOINT_NAMES.
  - benchmark.benchmark_so101_bin_diagnostic's own FIXED_BIN_MODE_*
    constants and benchmark.evaluate_so101_expert_small_randomization's
    own sample_object_position() -- reconstructs the EXACT scene a
    validation seed was collected with.
  - benchmark.so101_scripted_expert's own LINEAR_SPEED_PASS_MPS/
    ANGULAR_SPEED_PASS_RADPS/BIN_INNER_XY_TOLERANCE_M/
    BIN_CENTER_BELOW_RIM_TOLERANCE_M threshold VALUES for the final
    success judgment -- but NOT evaluate_bin_place_success()/
    compute_bin_success_debug() themselves, since those two functions
    require expert-waypoint-specific fields (rise_reached/
    pre_place_reached/descend_reached/retreat_reached,
    object_separated_during_wait) that simply do not exist for a
    model-only rollout with no expert phase structure. This script's
    own model_rollout_success_debug() below is a DELIBERATELY SEPARATE,
    smaller check -- same geometric thresholds, no waypoint-completion
    concept -- never conflated with the production bin_success_debug.

Does NOT modify expert waypoints/clearances/bin geometry/camera/
success criterion/settle threshold/action schema/joint order/Panda
backend. Does NOT collect a dataset.

Run:
  .venv-vla/bin/python -m benchmark.so101_smolvla_rollout
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_MODE_ANCHOR_OFFSET_XY, FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_dataset_schema import SO101_JOINT_NAMES, pack_state
from benchmark.so101_scripted_expert import (
    ANGULAR_SPEED_PASS_RADPS,
    BIN_CENTER_BELOW_RIM_TOLERANCE_M,
    BIN_INNER_XY_TOLERANCE_M,
    LINEAR_SPEED_PASS_MPS,
)
from benchmark.so101_smolvla_checkpoint_inference_eval import CHECKPOINT_DIR, TASK_TEXT, load_policy_and_processors
from robot_sim.so101_pybullet_backend import ARM_JOINT_NAMES, DEFAULT_SCENE_CONFIG, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "split.json"
ROLLOUT_RESULTS_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "rollout_results.json"

MAX_ROLLOUT_STEPS = 90  # training episodes were 68-69 steps; margin added since a sanity-trained policy may run long
SETTLE_CHECK_STEPS = 60  # simplified (NOT the expert's full 1080-step continuous-stability loop) settle window after the rollout's own last step
NUM_ROLLOUT_SEEDS = 3


def build_rollout_backend(seed: int) -> So101PyBulletBackend:
    from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE

    sampled_object_position = sample_object_position(seed, FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE)
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    return So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config={"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY},
    )


def joint_limits(backend: So101PyBulletBackend) -> dict:
    limits = {}
    for name in ARM_JOINT_NAMES:
        info = backend.joint_info_by_name[name]
        limits[name] = (info["lower"], info["upper"])
    return limits


def model_rollout_success_debug(backend: So101PyBulletBackend, grasp_was_ever_established: bool) -> dict:
    """Deliberately separate from benchmark.so101_scripted_expert's own
    compute_bin_success_debug()/evaluate_bin_place_success() -- reuses
    the SAME geometric threshold VALUES (bin inner bounds, rim_z,
    tolerances, settle speed thresholds) but has NO expert-waypoint-
    completion concept (no rise/pre_place/descend/retreat -- a raw
    policy never announces those phases). See this file's own
    docstring."""
    bin_info = backend.get_bin_debug_info()
    rim_z = bin_info["rim_z"]
    inner_bounds = {
        "x_min": bin_info["inner_x_min"], "x_max": bin_info["inner_x_max"],
        "y_min": bin_info["inner_y_min"], "y_max": bin_info["inner_y_max"],
    }
    final_object_position, _ = backend.get_object_pose()
    final_aabb_min, final_aabb_max = p.getAABB(backend.object_id, physicsClientId=backend.client_id)

    center_inside = (
        inner_bounds["x_min"] <= final_object_position[0] <= inner_bounds["x_max"]
        and inner_bounds["y_min"] <= final_object_position[1] <= inner_bounds["y_max"]
    )
    protrusion_x = max(inner_bounds["x_min"] - final_aabb_min[0], final_aabb_max[0] - inner_bounds["x_max"], 0.0)
    protrusion_y = max(inner_bounds["y_min"] - final_aabb_min[1], final_aabb_max[1] - inner_bounds["y_max"], 0.0)
    inside_inner_xy = center_inside and protrusion_x <= BIN_INNER_XY_TOLERANCE_M and protrusion_y <= BIN_INNER_XY_TOLERANCE_M

    center_rim_delta = final_object_position[2] - rim_z
    object_center_below_rim = center_rim_delta < -BIN_CENTER_BELOW_RIM_TOLERANCE_M
    object_top_below_rim = (final_aabb_max[2] - rim_z) < 0.0

    grasp_state = backend.get_grasp_state()
    object_separated = grasp_state["grasp_constraint_id"] is None

    # Simplified settle check (NOT the expert's continuous-stability
    # loop) -- steps SETTLE_CHECK_STEPS physics steps and checks the
    # object's speed at the END of that window only.
    for _ in range(SETTLE_CHECK_STEPS):
        backend.step(1)
    velocity, angular_velocity = p.getBaseVelocity(backend.object_id, physicsClientId=backend.client_id)
    linear_speed = float(np.linalg.norm(velocity))
    angular_speed = float(np.linalg.norm(angular_velocity))
    settle_success = linear_speed <= LINEAR_SPEED_PASS_MPS and angular_speed <= ANGULAR_SPEED_PASS_RADPS

    return {
        "grasp_was_ever_established": grasp_was_ever_established,
        "object_separated": object_separated,
        "inside_inner_xy": inside_inner_xy,
        "object_center_below_rim": object_center_below_rim,
        "object_top_below_rim": object_top_below_rim,
        "settle_success_simplified": settle_success,
        "final_linear_speed_mps": linear_speed,
        "final_angular_speed_radps": angular_speed,
        "object_final_xyz": list(final_object_position),
        "rim_z": rim_z,
        "center_rim_delta": center_rim_delta,
        "model_rollout_place_success": bool(
            grasp_was_ever_established and object_separated and inside_inner_xy and object_center_below_rim and settle_success
        ),
    }


def predict_action_in_rollout(policy, preprocessor, postprocessor, observation: dict) -> np.ndarray:
    """Rollout-specific action query -- deliberately does NOT call
    policy.reset() per step (unlike
    benchmark.so101_smolvla_checkpoint_inference_eval.predict_action(),
    which is correct for INDEPENDENT per-frame offline evaluation but
    wrong here: policy.reset() clears SmolVLAPolicy's own internal
    n_action_steps action queue, whose own docstring says it "should be
    called whenever the ENVIRONMENT is reset" -- i.e. once per episode.
    Calling it every step (the bug this task's chat report found and
    verified with debug logging: 10/10 steps each triggered a fresh
    50-action chunk inference, using only index 0 of each and
    discarding the other 49) defeats select_action()'s own queue
    entirely. The caller (run_one_rollout()) calls policy.reset() ONCE
    before the step loop; this function must never call it again."""
    batch = preprocessor(observation)
    with torch.no_grad():
        raw_action = policy.select_action(batch)
    final_action = postprocessor(raw_action)
    return final_action.squeeze(0).cpu().numpy()


def run_one_rollout(policy, preprocessor, postprocessor, seed: int, max_steps: int = MAX_ROLLOUT_STEPS, debug_queue_log: bool = False) -> dict:
    from lerobot.utils.constants import ACTION as _ACTION_QUEUE_KEY

    backend = build_rollout_backend(seed)
    limits = None
    step_log = []
    failure_reason = None
    grasp_was_ever_established = False
    aborted_early = False
    queue_debug_log = [] if debug_queue_log else None

    try:
        backend.reset()
        policy.reset()  # ONCE per episode -- see predict_action_in_rollout()'s own docstring
        limits = joint_limits(backend)

        for step_index in range(max_steps):
            obs = backend.get_observation()
            image = backend.render_front_camera()
            state = pack_state(obs["joint_positions"], obs["gripper_position_normalized"])

            observation_dict = {
                "observation.images.front": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
                "observation.state": torch.from_numpy(state),
                "task": TASK_TEXT,
            }
            if debug_queue_log:
                # ACTION QUEUE REGRESSION CHECK (see this task's chat
                # report, "action queue 회귀 확인") -- read BEFORE the
                # call, matching SmolVLAPolicy._check_get_actions_condition()'s
                # own "queue empty -> new chunk inference" logic exactly,
                # so this log reflects what the library itself will do,
                # not a guess.
                queue_len_before = len(policy._queues[_ACTION_QUEUE_KEY]) if hasattr(policy, "_queues") else None
                will_call_inference = queue_len_before == 0
            action = predict_action_in_rollout(policy, preprocessor, postprocessor, observation_dict)
            if debug_queue_log:
                queue_len_after = len(policy._queues[_ACTION_QUEUE_KEY])
                queue_debug_log.append({
                    "step": step_index, "queue_len_before_call": queue_len_before, "queue_len_after_call": queue_len_after,
                    "inference_call_triggered": will_call_inference,
                })

            if not np.all(np.isfinite(action)):
                failure_reason = f"nan_or_inf_action_at_step_{step_index}"
                aborted_early = True
                break

            arm_targets_raw = action[:5].tolist()
            gripper_raw = float(action[5])

            clamped_arm_targets = []
            joint_limit_clamped = False
            for i, name in enumerate(ARM_JOINT_NAMES):
                lower, upper = limits[name]
                v = arm_targets_raw[i]
                clamped = min(max(v, lower), upper)
                if clamped != v:
                    joint_limit_clamped = True
                clamped_arm_targets.append(clamped)
            gripper_normalized = min(max(gripper_raw / 100.0, 0.0), 1.0)

            backend.apply_joint_target(clamped_arm_targets)
            backend.set_gripper(gripper_normalized)

            if backend.is_grasped():
                grasp_was_ever_established = True

            ee_position, _ = backend.get_end_effector_pose()
            object_position, _ = backend.get_object_pose()
            object_gripper_distance = float(np.linalg.norm(np.array(ee_position) - np.array(object_position)))

            step_log.append({
                "step": step_index, "raw_action": action.tolist(), "joint_limit_clamped": joint_limit_clamped,
                "is_grasped": backend.is_grasped(), "object_gripper_distance_m": object_gripper_distance,
            })

        if not aborted_early:
            success_debug = model_rollout_success_debug(backend, grasp_was_ever_established)
        else:
            success_debug = None

        min_object_gripper_distance_m = min((s["object_gripper_distance_m"] for s in step_log), default=None)

        return {
            "seed": seed, "steps_executed": len(step_log), "aborted_early": aborted_early,
            "failure_reason": failure_reason, "grasp_was_ever_established": grasp_was_ever_established,
            "any_joint_limit_clamp_triggered": any(s["joint_limit_clamped"] for s in step_log),
            "joint_limit_clamp_count": sum(1 for s in step_log if s["joint_limit_clamped"]),
            "min_object_gripper_distance_m": min_object_gripper_distance_m,
            "model_rollout_success_debug": success_debug,
            "model_rollout_place_success": success_debug["model_rollout_place_success"] if success_debug else False,
            "queue_debug_log": queue_debug_log,
        }
    finally:
        backend.close()


def main() -> None:
    # CLI overrides purely additive -- see the identical note in
    # benchmark/so101_smolvla_checkpoint_inference_eval.py's own main().
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--split-path", type=str, default=str(SPLIT_PATH))
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-rollout-steps", type=int, default=MAX_ROLLOUT_STEPS)
    parser.add_argument("--output-path", type=str, default=str(ROLLOUT_RESULTS_PATH))
    parser.add_argument("--debug-queue-log", action="store_true", help="record per-step action-queue length + inference-call trigger (see this task's chat report, 'action queue 회귀 확인') -- diagnostic only, off by default")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    if args.rollout_seeds is not None:
        rollout_seeds = args.rollout_seeds
    else:
        split = json.loads(Path(args.split_path).read_text())
        rollout_seeds = split["validation_episodes"][:NUM_ROLLOUT_SEEDS]

    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint_dir)

    results = []
    for seed in rollout_seeds:
        print(f"=== Rollout seed {seed} ===")
        result = run_one_rollout(policy, preprocessor, postprocessor, seed, max_steps=args.max_rollout_steps, debug_queue_log=args.debug_queue_log)
        results.append(result)
        print(f"  steps_executed={result['steps_executed']} aborted_early={result['aborted_early']} "
              f"failure_reason={result['failure_reason']} grasp_ever={result['grasp_was_ever_established']} "
              f"place_success={result['model_rollout_place_success']}")

    summary = {
        "rollout_seeds": rollout_seeds,
        "max_rollout_steps": args.max_rollout_steps,
        "checkpoint": str(checkpoint_dir),
        "results": results,
        "success_count": sum(1 for r in results if r["model_rollout_place_success"]),
        "aborted_count": sum(1 for r in results if r["aborted_early"]),
        "grasp_established_count": sum(1 for r in results if r["grasp_was_ever_established"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"success_count: {summary['success_count']}/{len(results)}")
    print(f"aborted_count: {summary['aborted_count']}/{len(results)}")
    print(f"grasp_established_count: {summary['grasp_established_count']}/{len(results)}")
    print(f"\nRollout results JSON: {output_path}")


if __name__ == "__main__":
    main()
