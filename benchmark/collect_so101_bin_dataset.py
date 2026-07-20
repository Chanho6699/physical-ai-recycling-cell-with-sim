"""SO-101 open-top bin pilot dataset collector -- FIRST small-scale
demonstration dataset before bulk collection (see this task's chat
report, "첫 번째 소규모 SO-101 bin demonstration dataset"). Collects a
fixed number of deterministic-seed episodes (default 20, seeds 0..19)
into a single LeRobotDataset, using the existing ±1cm object-position
randomization and the existing scripted expert -- unmodified.

Reuses (does NOT reimplement):
  - benchmark.collect_so101_episode's own make_frame_recorder(),
    write_phase_id_mapping(), verify_dataset() -- the SAME recorder
    pieces validated by benchmark.smoke_so101_bin_dataset_preflight.
  - benchmark.so101_scripted_expert's own run_pick_and_place_episode()
    -- unmodified waypoints/clearances/success criterion.
  - benchmark.evaluate_so101_expert_small_randomization's own
    sample_object_position()/DEFAULT_X_RANGE/DEFAULT_Y_RANGE --
    the SAME deterministic seed -> (x_offset, y_offset) function
    already used by benchmark.benchmark_so101_bin_diagnostic's own
    20-seed production benchmark.

Does NOT touch expert waypoints/clearances/bin geometry/randomization
range/success criterion/failure_reason priority/settle threshold/
action schema/normalization/VLA adapter/VLA training/Panda backend.

Single dataset, single process, one LeRobotDataset.create() call --
each seed gets its OWN backend instance (object_position is a
constructor-time override, so a fresh backend per seed is required;
see benchmark.benchmark_so101_bin_diagnostic's own per-seed backend
construction for the same reason), but all seeds write into the SAME
dataset object (save_episode() per success, clear_episode_buffer() per
failure -- exactly the existing single-episode discard policy, just
looped).

Per-episode contract checks (see this task's chat report, section 4)
run on the IN-MEMORY recorded frames right after each episode
completes -- BEFORE save_episode() is called for that episode. If a
place_success=True episode FAILS its contract check, this is treated
as a genuine bug (not a normal discard): the episode buffer is cleared
without saving, and the ENTIRE collection run aborts immediately with
an explicit non-zero exit and an "aborted" summary -- it does not
silently skip the bad episode and continue.

Run:
  .venv-vla/bin/python -m benchmark.collect_so101_bin_dataset
  .venv-vla/bin/python -m benchmark.collect_so101_bin_dataset --num-episodes 20 \\
    --dataset-root datasets/so101_bin_pilot_20
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    FIXED_BIN_OBJECT_X_RANGE,
    FIXED_BIN_OBJECT_Y_RANGE,
    RANDOMIZATION_MODE_COUPLED_SMALL,
    RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY,
)
from benchmark.collect_so101_episode import (
    make_frame_recorder,
    verify_dataset,
    write_phase_id_mapping,
)
from benchmark.evaluate_so101_expert_small_randomization import (
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    sample_object_position,
)
from benchmark.so101_dataset_schema import (
    SO101_FEATURES,
    SO101_JOINT_NAMES,
    SO101_ROBOT_TYPE,
)
from benchmark.so101_scripted_expert import (
    PHASE_ID_BY_NAME,
    PHASE_SEQUENCE,
    So101ExpertError,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import (
    DEFAULT_SCENE_CONFIG,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_WIDTH,
    InvalidSceneLayoutError,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = "datasets/so101_bin_pilot_20"
DEFAULT_REPO_ID = "local/so101_bin_pilot_20"
DEFAULT_FPS = 10  # matches collect_so101_episode.py's own DEFAULT_FPS
DEFAULT_INSTRUCTION = "Pick up the object and place it in the bin."
DEFAULT_NUM_EPISODES = 20
DEFAULT_SEED_START = 0
SCHEMA_IDENTIFIER = "so101_joint6d_v1"  # see benchmark/so101_dataset_schema.py -- unversioned prior to this task, first identifier assigned here for manifest traceability only.


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_episode_contract(
    recorded_frames: list, result: dict, scene_state: dict, seed: int,
    randomization_mode: str = RANDOMIZATION_MODE_COUPLED_SMALL, object_yaw_rad: float = 0.0,
) -> dict:
    """Per-episode contract checks (this task's chat report, section 4).
    Operates on the IN-MEMORY frames recorded via wrapped_on_step for
    THIS episode only -- does not depend on LeRobotDataset's own
    internal buffering/flush timing. A separate, later, whole-dataset
    disk round-trip (section 5) is done once after all episodes are
    collected."""
    checks = {}
    checks["place_success_true"] = result.get("place_success") is True
    checks["failure_reason_none"] = result.get("failure_reason") is None
    checks["use_bin_true"] = scene_state.get("use_bin") is True
    checks["randomization_mode_matches"] = randomization_mode in (RANDOMIZATION_MODE_COUPLED_SMALL, RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY)
    checks["object_yaw_is_zero"] = object_yaw_rad == 0.0
    checks["layout_validation_passed_true"] = scene_state.get("layout_validation_passed") is True
    debug = result.get("bin_success_debug") or {}
    checks["bin_success_debug_conditions_true"] = bool(debug) and all(
        debug.get(k) for k in (
            "layout_validation_passed", "object_separated", "inside_inner_xy",
            "object_center_below_rim", "object_top_below_rim", "settle_success",
            "manipulation_steps_completed", "place_waypoint_reached",
        )
    )
    checks["frame_count_positive"] = len(recorded_frames) > 0

    arm_targets = np.array([f["arm_joint_targets"] for f in recorded_frames], dtype=np.float64)
    gripper_targets = np.array([f["gripper_target_normalized"] for f in recorded_frames], dtype=np.float64)
    checks["action_shape_is_6"] = arm_targets.ndim == 2 and arm_targets.shape[1] == 5  # 5 arm joints + gripper packed separately below
    action_array = np.concatenate([arm_targets, gripper_targets.reshape(-1, 1) * 100.0], axis=1)
    checks["action_shape_is_6"] = action_array.shape[1] == 6
    checks["action_finite"] = bool(np.all(np.isfinite(action_array)))

    checks["seed_metadata_present"] = seed is not None
    checks["initial_object_position_present"] = scene_state.get("object_position") is not None
    checks["bin_center_present"] = scene_state.get("bin_center") is not None

    phase_ids = [PHASE_ID_BY_NAME[f["phase"]] for f in recorded_frames]
    checks["phase_id_valid"] = all(0 <= pid < len(PHASE_SEQUENCE) for pid in phase_ids)

    checks["pass"] = all(checks.values())
    checks["action_array"] = action_array
    return checks


def collect_episode(
    dataset, seed: int, task: str, episode_index_counter: dict,
    randomization_mode: str = RANDOMIZATION_MODE_COUPLED_SMALL,
    x_range: tuple = None, y_range: tuple = None, bin_center_override_xy: list = None, scene_config: dict = None,
) -> dict:
    x_range = x_range if x_range is not None else DEFAULT_X_RANGE
    y_range = y_range if y_range is not None else DEFAULT_Y_RANGE
    sampled_object_position = sample_object_position(seed, x_range, y_range)

    backend_kwargs = {"gui": False, "use_bin": True, "object_position": sampled_object_position}
    if bin_center_override_xy is not None:
        backend_kwargs["bin_center_override_xy"] = bin_center_override_xy
    if scene_config is not None:
        backend_kwargs["scene_config"] = scene_config
    backend = So101PyBulletBackend(**backend_kwargs)

    recorded_frames = []
    result = None
    failure_reason = None
    scene_state = None
    contract = None
    saved = False

    try:
        try:
            backend.reset()
        except InvalidSceneLayoutError as exc:
            return {
                "seed": seed, "saved": False, "success": False, "randomization_mode": randomization_mode,
                "failure_reason": f"scene_invalid:{exc.failure_type}",
                "frame_count": 0, "aborted": False, "contract": None,
                "sampled_object_position": sampled_object_position,
            }

        scene_state = backend.get_scene_state()
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])

        on_step, frame_counter = make_frame_recorder(dataset, backend, task)

        def wrapped_on_step(phase, arm_joint_targets, gripper_target_normalized):
            recorded_frames.append({
                "phase": phase,
                "arm_joint_targets": list(arm_joint_targets),
                "gripper_target_normalized": gripper_target_normalized,
            })
            on_step(phase, arm_joint_targets, gripper_target_normalized)

        try:
            result = run_pick_and_place_episode(backend, transport_delta_xy, on_step=wrapped_on_step)
            failure_reason = result["failure_reason"]
        except So101ExpertError as exc:
            failure_reason = exc.failure_reason
            result = {"place_success": False, "failure_reason": failure_reason, "bin_success_debug": None}

        place_success = result.get("place_success", False)

        if not place_success:
            dataset.clear_episode_buffer()
            return {
                "seed": seed, "saved": False, "success": False, "randomization_mode": randomization_mode, "failure_reason": failure_reason,
                "frame_count": len(recorded_frames), "aborted": False, "contract": None,
                "sampled_object_position": sampled_object_position,
            }

        contract = validate_episode_contract(
            recorded_frames, result, scene_state, seed, randomization_mode=randomization_mode, object_yaw_rad=0.0,
        )
        if not contract["pass"]:
            dataset.clear_episode_buffer()
            return {
                "seed": seed, "saved": False, "success": True, "randomization_mode": randomization_mode, "failure_reason": failure_reason,
                "frame_count": len(recorded_frames), "aborted": True, "contract": contract,
                "sampled_object_position": sampled_object_position, "scene_state": scene_state, "result": result,
            }

        dataset.save_episode()
        saved = True
        episode_index = episode_index_counter["count"]
        episode_index_counter["count"] += 1

        return {
            "seed": seed, "saved": True, "success": True, "randomization_mode": randomization_mode, "failure_reason": None,
            "frame_count": len(recorded_frames), "aborted": False, "contract": contract,
            "episode_index": episode_index, "sampled_object_position": sampled_object_position,
            "scene_state": scene_state, "result": result,
        }
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--mode", type=str, default=RANDOMIZATION_MODE_COUPLED_SMALL,
        choices=[RANDOMIZATION_MODE_COUPLED_SMALL, RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY],
        help="coupled_small (default, unchanged): bin_center = object_position + offset. "
             "fixed_bin_object_xy: bin stays fixed, only object XY is independently randomized "
             "(see benchmark.benchmark_so101_bin_diagnostic's own --mode of the same name).",
    )
    args = parser.parse_args()

    root = resolve(args.dataset_root)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")

    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=SO101_FEATURES, root=str(root),
        robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )

    episode_index_counter = {"count": 0}
    manifest_records = []
    per_seed_results = []
    aborted = False
    abort_reason = None

    collect_kwargs = {"randomization_mode": args.mode}
    if args.mode == RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY:
        nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
        fixed_bin_center_xy = [
            nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
        ]
        collect_kwargs.update({
            "x_range": FIXED_BIN_OBJECT_X_RANGE, "y_range": FIXED_BIN_OBJECT_Y_RANGE,
            "bin_center_override_xy": fixed_bin_center_xy,
            "scene_config": {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY},
        })
        print(f"--mode fixed_bin_object_xy: fixed_bin_center_xy={fixed_bin_center_xy}, "
              f"x_range={FIXED_BIN_OBJECT_X_RANGE}, y_range={FIXED_BIN_OBJECT_Y_RANGE}")

    try:
        for seed in seeds:
            outcome = collect_episode(dataset, seed, args.instruction, episode_index_counter, **collect_kwargs)
            per_seed_results.append(outcome)

            manifest_records.append({
                "episode_index": outcome.get("episode_index"), "seed": seed, "robot_type": SO101_ROBOT_TYPE,
                "skill": "pick_and_place", "scenario_group": f"bin_pilot_{args.mode}", "expert_policy": "scripted_so101",
                "success": outcome["success"], "frame_count": outcome["frame_count"],
                "place_success": outcome["success"], "failure_reason": outcome["failure_reason"],
                "dataset_action_space": "absolute_joint_position", "state_dimension": 6, "action_dimension": 6,
                "saved": outcome["saved"], "use_bin": True, "randomization_mode": args.mode, "object_yaw_rad": 0.0,
                "bin_center": outcome.get("scene_state", {}).get("bin_center") if outcome.get("scene_state") else None,
                "target_zone_offset_xy": outcome.get("scene_state", {}).get("target_zone_offset_xy") if outcome.get("scene_state") else None,
                "layout_validation_passed": outcome.get("scene_state", {}).get("layout_validation_passed") if outcome.get("scene_state") else None,
                "bin_success_debug": outcome.get("result", {}).get("bin_success_debug") if outcome.get("result") else None,
                "action_representation": "absolute_joint_position_6d", "joint_names": list(SO101_JOINT_NAMES),
                "image_camera_name": "front", "image_resolution": [FRONT_CAMERA_HEIGHT, FRONT_CAMERA_WIDTH],
                "initial_object_position": outcome.get("scene_state", {}).get("object_position") if outcome.get("scene_state") else None,
                "sampled_object_position": outcome.get("sampled_object_position"),
                "schema_identifier": SCHEMA_IDENTIFIER,
            })

            print(f"[seed {seed}] saved={outcome['saved']} success={outcome['success']} "
                  f"frame_count={outcome['frame_count']} failure_reason={outcome['failure_reason']}")

            if outcome["aborted"]:
                aborted = True
                abort_reason = f"seed {seed}: place_success=True but contract validation failed: {outcome['contract']}"
                print(f"\n!!! ABORTING COLLECTION: {abort_reason}\n")
                break
    finally:
        dataset.finalize()

    if episode_index_counter["count"] > 0:
        write_phase_id_mapping(root)

    manifest_path = root / "collection_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record, default=str) + "\n")

    saved_results = [r for r in per_seed_results if r["saved"]]
    discarded_results = [r for r in per_seed_results if not r["saved"]]
    episode_lengths = [r["frame_count"] for r in saved_results]

    failure_reason_counts = {}
    for r in discarded_results:
        key = str(r["failure_reason"])
        failure_reason_counts[key] = failure_reason_counts.get(key, 0) + 1

    summary = {
        "dataset_name": root.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "robot_type": SO101_ROBOT_TYPE,
        "use_bin": True,
        "requested_episode_count": len(seeds),
        "saved_episode_count": len(saved_results),
        "discarded_episode_count": len(discarded_results),
        "seeds_requested": seeds,
        "seeds_saved": [r["seed"] for r in saved_results],
        "seeds_discarded": [r["seed"] for r in discarded_results],
        "randomization_mode": args.mode,
        "randomization_range": {
            "x_range": list(collect_kwargs.get("x_range", DEFAULT_X_RANGE)),
            "y_range": list(collect_kwargs.get("y_range", DEFAULT_Y_RANGE)),
        },
        "fixed_bin_center_xy": collect_kwargs.get("bin_center_override_xy"),
        "action_representation": "absolute_joint_position_6d",
        "action_dimension": 6,
        "joint_names": list(SO101_JOINT_NAMES),
        "camera_name": "front",
        "image_resolution": [FRONT_CAMERA_HEIGHT, FRONT_CAMERA_WIDTH],
        "total_frames": int(sum(episode_lengths)),
        "episode_length_stats": {
            "min": int(np.min(episode_lengths)) if episode_lengths else None,
            "max": int(np.max(episode_lengths)) if episode_lengths else None,
            "mean": float(np.mean(episode_lengths)) if episode_lengths else None,
            "std": float(np.std(episode_lengths)) if episode_lengths else None,
            "unique_values": sorted(set(episode_lengths)),
        },
        "place_success_count": len(saved_results),
        "failure_reason_counts": failure_reason_counts,
        "source_expert": "benchmark.so101_scripted_expert.run_pick_and_place_episode",
        "schema_identifier": SCHEMA_IDENTIFIER,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "dataset_root": str(root),
        "manifest_path": str(manifest_path),
    }

    summary_path = root / "collection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== SO-101 bin pilot dataset collection ===")
    print(f"requested={summary['requested_episode_count']} saved={summary['saved_episode_count']} discarded={summary['discarded_episode_count']}")
    print(f"seeds_saved={summary['seeds_saved']}")
    print(f"seeds_discarded={summary['seeds_discarded']}")
    print(f"total_frames={summary['total_frames']} episode_length_stats={summary['episode_length_stats']}")
    print(f"aborted={aborted}" + (f" ({abort_reason})" if aborted else ""))
    print(f"\nDataset root: {root}")
    print(f"Manifest: {manifest_path}")
    print(f"Collection summary: {summary_path}")

    if aborted:
        sys.exit(1)

    if summary["saved_episode_count"] > 0:
        verification = verify_dataset(root)
        print("\n=== Dataset verification (verify_dataset) ===")
        for k, v in verification.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
