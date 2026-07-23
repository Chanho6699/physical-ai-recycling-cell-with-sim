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
from robot_sim.so101_pybullet_backend import (
    ARM_JOINT_NAMES,
    DEFAULT_SCENE_CONFIG,
    GRASP_DISTANCE_THRESHOLD_M,
    GRASP_GRIPPER_CLOSED_THRESHOLD,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "split.json"
ROLLOUT_RESULTS_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "rollout_results.json"

MAX_ROLLOUT_STEPS = 90  # training episodes were 68-69 steps; margin added since a sanity-trained policy may run long
SETTLE_CHECK_STEPS = 60  # simplified (NOT the expert's full 1080-step continuous-stability loop) settle window after the rollout's own last step
NUM_ROLLOUT_SEEDS = 3

DEFAULT_POLICY_NOISE_BASE_SEED = 100000
POLICY_NOISE_SEED_BLOCK_SIZE = 1000  # must exceed the largest possible repeat_id -- see derive_policy_noise_seed()


def derive_policy_noise_seed(base_policy_seed: int, environment_seed: int, repeat_id: int) -> int:
    """Deterministic per-(environment_seed, repeat_id) policy noise seed (see
    this task's chat report, '기존 방식은 noise seed 하나를 모든 environment
    seed가 공유' -- a single shared noise seed across all 40 environment
    seeds meant the 40 rollouts at a given noise seed were NOT 40
    independent draws, just one shared noise draw replayed against 40
    different object positions, which can make results look artificially
    bimodal/polarized).

    Deliberately a pure arithmetic function of its three integer inputs --
    NEVER Python's built-in hash() (process-randomized by default unless
    PYTHONHASHSEED is fixed, so a hash()-derived seed would silently break
    reproducibility across separate process launches, defeating the whole
    point of this feature). The same (base_policy_seed, environment_seed,
    repeat_id) always yields the same derived seed; different
    (environment_seed, repeat_id) pairs never collide as long as
    0 <= repeat_id < POLICY_NOISE_SEED_BLOCK_SIZE, since environment_seed *
    POLICY_NOISE_SEED_BLOCK_SIZE reserves a non-overlapping block of
    POLICY_NOISE_SEED_BLOCK_SIZE seed values per distinct environment_seed."""
    if not (0 <= repeat_id < POLICY_NOISE_SEED_BLOCK_SIZE):
        raise ValueError(f"repeat_id={repeat_id} out of supported range [0, {POLICY_NOISE_SEED_BLOCK_SIZE})")
    return base_policy_seed + environment_seed * POLICY_NOISE_SEED_BLOCK_SIZE + repeat_id


def build_rollout_backend(
    seed: int, object_position_override: list = None, object_footprint_xy_override: list = None,
    object_shape_override: str = None, object_radius_override: float = None,
) -> So101PyBulletBackend:
    """`object_position_override` (see this task's chat report, "XY 외삽
    평가") -- when given, used VERBATIM as the object's spawn position
    instead of sample_object_position(seed, ...). `seed` is then only a
    bookkeeping label (still passed through to callers/logs), never
    consulted for the object position. None (default) preserves the
    exact pre-existing seed-based sampling behavior used by every
    official baseline evaluation to date -- this parameter changes
    nothing for any existing call site that doesn't pass it.

    `object_footprint_xy_override` (see this task's chat report, "Stage
    1B: rectangular-box shape generalization") -- when given, overrides
    the object's own half-extent XY footprint (object HEIGHT is
    unaffected -- only this project's cube-vs-box shape experiments
    ever pass this). None (default) preserves the exact pre-existing
    cube footprint (DEFAULT_SCENE_CONFIG's own object_footprint_xy,
    unchanged) for every cube-only call site."""
    from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE

    if object_position_override is not None:
        sampled_object_position = object_position_override
    else:
        sampled_object_position = sample_object_position(seed, FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE)
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config_override = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY}
    if object_footprint_xy_override is not None:
        scene_config_override["object_footprint_xy"] = object_footprint_xy_override
    # object_shape_override/object_radius_override (see this task's chat
    # report, "Stage 1B cylinder zero-shot 평가") -- additive, same
    # pattern as object_footprint_xy_override above. None (default, every
    # existing cube/box call site) leaves scene_config's own "box"
    # default untouched.
    if object_shape_override is not None:
        scene_config_override["object_shape"] = object_shape_override
    if object_radius_override is not None:
        scene_config_override["object_radius"] = object_radius_override
    return So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config=scene_config_override,
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


def predict_action_in_rollout(policy, preprocessor, postprocessor, observation: dict, noise_generator: "torch.Generator | None" = None) -> np.ndarray:
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
    before the step loop; this function must never call it again.

    `noise_generator` (see this task's chat report, "validation/test
    평가 재현성") -- when given, draws the SAME shape/dtype/device flow-
    matching noise tensor SmolVLAPolicy._get_action_chunk() would
    otherwise sample internally via its own sample_noise() (unseeded,
    global torch RNG -- see lerobot's own VLAFlowMatching.sample_noise()),
    but from a caller-owned, explicitly-seeded torch.Generator instead,
    and passes it in via select_action()'s own existing `noise=` kwarg
    -- a parameter the library already exposes for exactly this
    purpose. Only drawn on steps where the action queue is actually
    empty (a new chunk will actually be generated this call) -- reusing
    the same "queue empty" check this file's own debug_queue_log/
    diagnostic_log paths already use, so a generator draw is never
    wasted on a step that only pops an already-queued action. This
    changes NOTHING about SmolVLAPolicy/VLAFlowMatching's own code, and
    NEVER touches the global torch RNG (torch.manual_seed is never
    called) -- so it cannot affect training (lerobot-train never calls
    select_action()/passes `noise=`, only this rollout script's own
    calls do, and only when a caller explicitly opts in)."""
    from lerobot.utils.constants import ACTION as _ACTION_QUEUE_KEY

    noise = None
    if noise_generator is not None and hasattr(policy, "_queues") and len(policy._queues[_ACTION_QUEUE_KEY]) == 0:
        device = next(policy.parameters()).device
        shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
        noise = torch.normal(mean=0.0, std=1.0, size=shape, generator=noise_generator, device=device, dtype=torch.float32)

    batch = preprocessor(observation)
    with torch.no_grad():
        raw_action = policy.select_action(batch, noise=noise)
    final_action = postprocessor(raw_action)
    return final_action.squeeze(0).cpu().numpy()


def run_one_rollout(
    policy, preprocessor, postprocessor, seed: int, max_steps: int = MAX_ROLLOUT_STEPS,
    debug_queue_log: bool = False, diagnostic_log: bool = False, policy_noise_seed: int = None,
    object_position_override: list = None, object_footprint_xy_override: list = None,
    object_shape_override: str = None, object_radius_override: float = None,
) -> dict:
    """`diagnostic_log=True` (see this task's chat report, "pre-grasp
    실패 원인을 정확히 구분하기 위한 진단 rollout") ONLY appends
    additional READ-ONLY observation fields to a separate
    `diagnostic_step_log` list -- every value it reads (gripper real
    position, object velocity, grasp-condition constants) was already
    exposed by robot_sim/so101_pybullet_backend.py's own public API
    before this change. It never alters arm_targets_raw/
    clamped_arm_targets/gripper_normalized or the order
    apply_joint_target()/set_gripper()/is_grasped() are called in --
    those lines are byte-for-byte unchanged from before this task, so
    control timing and rollout results (grasp/place/min-distance) are
    identical to a `diagnostic_log=False` run on the same seed.

    `policy_noise_seed` (see this task's chat report, "validation/test
    평가 재현성 확보") -- deliberately SEPARATE from `seed` (the
    environment/object-position seed, consumed by
    build_rollout_backend()'s own sample_object_position() call, a
    completely different RNG namespace). None (default) preserves the
    exact pre-existing behavior (policy draws its own unseeded noise
    every call, byte-identical to before this task). An int here
    creates ONE torch.Generator for this whole episode and threads it
    into predict_action_in_rollout() -- see that function's own
    docstring for why this cannot affect training."""
    from lerobot.utils.constants import ACTION as _ACTION_QUEUE_KEY

    backend = build_rollout_backend(
        seed, object_position_override=object_position_override, object_footprint_xy_override=object_footprint_xy_override,
        object_shape_override=object_shape_override, object_radius_override=object_radius_override,
    )
    limits = None
    step_log = []
    failure_reason = None
    grasp_was_ever_established = False
    aborted_early = False
    queue_debug_log = [] if debug_queue_log else None
    diagnostic_step_log = [] if diagnostic_log else None
    inference_call_counter = {"count": 0}
    noise_generator = None
    if policy_noise_seed is not None:
        noise_generator = torch.Generator(device=next(policy.parameters()).device).manual_seed(policy_noise_seed)

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
            if debug_queue_log or diagnostic_log:
                # ACTION QUEUE REGRESSION CHECK (see this task's chat
                # report, "action queue 회귀 확인") -- read BEFORE the
                # call, matching SmolVLAPolicy._check_get_actions_condition()'s
                # own "queue empty -> new chunk inference" logic exactly,
                # so this log reflects what the library itself will do,
                # not a guess.
                queue_len_before = len(policy._queues[_ACTION_QUEUE_KEY]) if hasattr(policy, "_queues") else None
                will_call_inference = queue_len_before == 0
                if will_call_inference:
                    inference_call_counter["count"] += 1
            action = predict_action_in_rollout(policy, preprocessor, postprocessor, observation_dict, noise_generator=noise_generator)
            if debug_queue_log or diagnostic_log:
                queue_len_after = len(policy._queues[_ACTION_QUEUE_KEY])
            if debug_queue_log:
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

            if diagnostic_log:
                # Every value read below was ALREADY computed above for
                # control (obs, action, gripper_normalized,
                # object_gripper_distance) or is a pure read of already-
                # public backend state (get_grasp_state, get_object_velocity)
                # -- nothing here feeds back into a control decision, and
                # this block runs strictly AFTER apply_joint_target()/
                # set_gripper()/is_grasped() so it cannot reorder them.
                grasp_state = backend.get_grasp_state()
                object_linear_velocity, object_angular_velocity = backend.get_object_velocity()
                # Same two conditions _maybe_trigger_grasp() itself checks
                # (robot_sim/so101_pybullet_backend.py) -- reusing the
                # imported threshold CONSTANTS, not re-derived/guessed.
                grasp_condition_met = (
                    object_gripper_distance <= GRASP_DISTANCE_THRESHOLD_M
                    and obs["gripper_position_normalized"] <= GRASP_GRIPPER_CLOSED_THRESHOLD
                )
                diagnostic_step_log.append({
                    "seed": seed,
                    "step": step_index,
                    "control_step": step_index,  # 1 control step == 1 macro sim step in this rollout; no separate sim-timestamp counter exists
                    "inference_call_index": inference_call_counter["count"],
                    "inference_call_triggered_this_step": will_call_inference,
                    "predicted_arm_action_rad": arm_targets_raw,
                    "predicted_gripper_action_raw": gripper_raw,
                    "applied_gripper_command_normalized": gripper_normalized,
                    "joint_state_rad": list(obs["joint_positions"]),
                    "end_effector_position": list(ee_position),
                    "object_position": list(object_position),
                    "ee_object_distance_m": object_gripper_distance,
                    "object_height_m": object_position[2],
                    "grasped": backend.is_grasped(),
                    "grasp_condition_met_this_step": grasp_condition_met,
                    "grasp_distance_threshold_m": GRASP_DISTANCE_THRESHOLD_M,
                    "grasp_gripper_closed_threshold": GRASP_GRIPPER_CLOSED_THRESHOLD,
                    "phase_label": "grasped" if backend.is_grasped() else "not_grasped",
                    "gripper_joint_position_normalized": obs["gripper_position_normalized"],
                    "object_linear_velocity_mps": object_linear_velocity,
                    "object_angular_velocity_radps": object_angular_velocity,
                    "action_queue_length_after_call": queue_len_after,
                    "grasp_distance_at_trigger": grasp_state["grasp_distance_at_trigger"],
                    "grasp_gripper_normalized_at_trigger": grasp_state["grasp_gripper_normalized_at_trigger"],
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
            "diagnostic_log": diagnostic_step_log,
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
    parser.add_argument("--diagnostic-log", action="store_true", help="record full per-step pre-grasp diagnostic fields (EE/object pose, gripper real position, grasp-condition check, action-chunk index, object velocity) -- observation-only, does not change control/results. Off by default.")
    parser.add_argument("--diagnostic-log-dir", type=str, default=None, help="directory to write one <seed>.json per seed's full diagnostic_log into (only used with --diagnostic-log)")
    parser.add_argument("--policy-noise-seed", type=int, default=None, help="explicit seed for the flow-matching noise torch.Generator (see this task's chat report, 'validation/test 평가 재현성 확보') -- deliberately SEPARATE from --rollout-seeds (the environment/object-position seed). None (default) preserves the exact pre-existing unseeded behavior. Mutually exclusive with --policy-noise-repeats (a single shared noise seed across all environment seeds is exactly the flawed protocol --policy-noise-repeats replaces).")
    parser.add_argument("--policy-noise-repeats", type=int, default=None, help="see this task's chat report, 'policy-noise 평가 프로토콜을 바로잡고' -- when set, each (environment_seed, repeat_id) combination for repeat_id in range(this value) gets its OWN derive_policy_noise_seed(base, environment_seed, repeat_id) instead of one noise seed shared across all environment seeds. Mutually exclusive with --policy-noise-seed.")
    parser.add_argument("--policy-noise-base-seed", type=int, default=DEFAULT_POLICY_NOISE_BASE_SEED, help="base_policy_seed passed to derive_policy_noise_seed() -- only used with --policy-noise-repeats")
    args = parser.parse_args()

    if args.policy_noise_seed is not None and args.policy_noise_repeats is not None:
        parser.error("--policy-noise-seed and --policy-noise-repeats are mutually exclusive -- pick one evaluation protocol (single shared noise seed, or per-(environment_seed, repeat_id) independent derived seeds).")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output_path)

    if args.rollout_seeds is not None:
        rollout_seeds = args.rollout_seeds
    else:
        split = json.loads(Path(args.split_path).read_text())
        rollout_seeds = split["validation_episodes"][:NUM_ROLLOUT_SEEDS]

    repeat_ids = list(range(args.policy_noise_repeats)) if args.policy_noise_repeats is not None else [None]

    # Fail fast on any derived-seed collision across (environment_seed, repeat_id)
    # combinations -- see derive_policy_noise_seed()'s own docstring for why this
    # cannot happen algebraically as long as repeat_id stays within its reserved
    # block, but this check makes that guarantee explicit rather than assumed.
    if args.policy_noise_repeats is not None:
        seed_to_combo = {}
        for seed in rollout_seeds:
            for repeat_id in repeat_ids:
                derived = derive_policy_noise_seed(args.policy_noise_base_seed, seed, repeat_id)
                if derived in seed_to_combo:
                    raise ValueError(f"derived policy noise seed collision: {derived} produced by both "
                                      f"{seed_to_combo[derived]} and (environment_seed={seed}, repeat_id={repeat_id})")
                seed_to_combo[derived] = (seed, repeat_id)

    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint_dir)

    diagnostic_log_dir = Path(args.diagnostic_log_dir) if args.diagnostic_log_dir else None
    if args.diagnostic_log and diagnostic_log_dir is not None:
        diagnostic_log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in rollout_seeds:
        for repeat_id in repeat_ids:
            if args.policy_noise_repeats is not None:
                derived_policy_seed = derive_policy_noise_seed(args.policy_noise_base_seed, seed, repeat_id)
                print(f"=== Rollout seed {seed} repeat {repeat_id} (derived_policy_seed={derived_policy_seed}) ===")
            else:
                derived_policy_seed = args.policy_noise_seed
                print(f"=== Rollout seed {seed} ===")

            result = run_one_rollout(
                policy, preprocessor, postprocessor, seed, max_steps=args.max_rollout_steps,
                debug_queue_log=args.debug_queue_log, diagnostic_log=args.diagnostic_log,
                policy_noise_seed=derived_policy_seed,
            )
            result["environment_seed"] = seed
            result["repeat_id"] = repeat_id
            result["policy_noise_seed_used"] = derived_policy_seed
            results.append(result)
            print(f"  steps_executed={result['steps_executed']} aborted_early={result['aborted_early']} "
                  f"failure_reason={result['failure_reason']} grasp_ever={result['grasp_was_ever_established']} "
                  f"place_success={result['model_rollout_place_success']}")

            if args.diagnostic_log and diagnostic_log_dir is not None:
                seed_path = (diagnostic_log_dir / f"seed_{seed}_repeat_{repeat_id}.json") if repeat_id is not None else (diagnostic_log_dir / f"seed_{seed}.json")
                with open(seed_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "seed": seed, "repeat_id": repeat_id, "policy_noise_seed_used": derived_policy_seed,
                        "grasp_was_ever_established": result["grasp_was_ever_established"],
                        "model_rollout_place_success": result["model_rollout_place_success"],
                        "min_object_gripper_distance_m": result["min_object_gripper_distance_m"],
                        "steps_executed": result["steps_executed"],
                        "diagnostic_log": result["diagnostic_log"],
                    }, f, indent=2, default=str)
                result["diagnostic_log"] = None  # keep the aggregate summary file lean -- full per-step detail lives in the per-seed file above

    summary = {
        "rollout_seeds": rollout_seeds,
        "max_rollout_steps": args.max_rollout_steps,
        "checkpoint": str(checkpoint_dir),
        "policy_noise_seed": args.policy_noise_seed,
        "policy_noise_repeats": args.policy_noise_repeats,
        "policy_noise_base_seed": args.policy_noise_base_seed if args.policy_noise_repeats is not None else None,
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
