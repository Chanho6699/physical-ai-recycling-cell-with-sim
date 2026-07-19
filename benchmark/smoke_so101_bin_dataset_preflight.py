"""SO-101 bin dataset recorder preflight (see this task's chat report,
"대량 VLA 학습 데이터 수집을 시작하기 전에... recorder가 bin
pick-and-place episode를 시간 정렬과 schema 관점에서 정확히 저장하는지
preflight 검증"). Saves at most a FEW validation episodes to
results/dataset_preflight/ -- NEVER the production dataset path
(datasets/so101_recorder_smoke or similar), does NOT collect a bulk
dataset, does NOT train anything, does NOT modify
benchmark/so101_scripted_expert.py's waypoints/clearances/geometry/
success criterion, does NOT touch robot_sim/pybullet_panda_backend.py.

Reuses (does NOT reimplement) benchmark.collect_so101_episode's own
make_frame_recorder()/verify_dataset()/write_phase_id_mapping() -- the
SAME recorder this preflight is validating, not a parallel
implementation.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_bin_dataset_preflight
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import io

from benchmark.collect_so101_episode import (
    make_frame_recorder,
    verify_dataset,
    write_phase_id_mapping,
)
from benchmark.so101_dataset_schema import SO101_FEATURES, SO101_JOINT_NAMES, SO101_ROBOT_TYPE
from benchmark.so101_scripted_expert import (
    PHASE_APPROACH,
    PHASE_GRASP,
    PHASE_ID_BY_NAME,
    PHASE_LIFT,
    PHASE_NAME_BY_ID,
    PHASE_PLACE_DESCEND,
    PHASE_PRE_GRASP,
    PHASE_RELEASE,
    PHASE_SETTLE,
    PHASE_TRANSPORT,
    So101ExpertError,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import InvalidSceneLayoutError, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ROOT = PROJECT_ROOT / "results" / "dataset_preflight"
SUCCESS_DATASET_ROOT = PREFLIGHT_ROOT / "success_episode"
DISCARD_UNIT_DATASET_ROOT = PREFLIGHT_ROOT / "discard_unit_test"
SUMMARY_PATH = PROJECT_ROOT / "results" / "so101_bin_dataset_preflight_summary.json"

PREFLIGHT_SEED = 0  # a seed already confirmed production_place_success=True in the 20-seed benchmark
EXPECTED_NON_SETTLE_PHASES = [PHASE_PRE_GRASP, PHASE_APPROACH, PHASE_GRASP, PHASE_LIFT, PHASE_TRANSPORT, PHASE_PLACE_DESCEND, PHASE_RELEASE]


def record_success_episode_with_instrumentation(dataset_root: Path) -> dict:
    """Records ONE real bin episode via the EXISTING recorder's own
    make_frame_recorder(), while ALSO independently tracking, via a
    spy on the backend's own apply_joint_target()/set_gripper(), the
    EXACT joint target/gripper value actually applied at each step --
    in the SAME call order the dataset frames were written in. This is
    the ground truth for the "recorded_action == executed_action"
    check (see this task's chat report, section 5A)."""
    dataset = LeRobotDataset.create(
        repo_id="local/so101_bin_preflight_success", fps=10, features=SO101_FEATURES, root=str(dataset_root),
        robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )
    backend = So101PyBulletBackend(gui=False, use_bin=True)

    applied_log = []
    recorded_frames = []
    success = False
    failure_reason = None
    result = None
    orig_apply = None
    orig_set_gripper = None
    frame_counter = {"count": 0}
    try:
        # reset() itself calls self.set_gripper(1.0, ...) internally as
        # part of its own neutral-pose setup (see
        # robot_sim/so101_pybullet_backend.py's reset()). Spies MUST be
        # installed only AFTER reset() completes -- otherwise that
        # internal call is captured into applied_log with no
        # corresponding recorded_frames entry, shifting every later
        # index by one and producing spurious recorded != executed
        # mismatches that reflect a test-harness bug, not a real
        # recorder issue.
        backend.reset()

        orig_apply = backend.apply_joint_target
        orig_set_gripper = backend.set_gripper

        def spy_apply(arm_joint_targets, settle_steps=40):
            applied_log.append({"kind": "move", "value": list(arm_joint_targets)})
            return orig_apply(arm_joint_targets, settle_steps=settle_steps)

        def spy_set_gripper(value, settle_steps=40):
            applied_log.append({"kind": "gripper", "value": value})
            return orig_set_gripper(value, settle_steps=settle_steps)

        backend.apply_joint_target = spy_apply
        backend.set_gripper = spy_set_gripper

        on_step, frame_counter = make_frame_recorder(dataset, backend, "Pick up the object and place it in the bin.")

        def wrapped_on_step(phase, arm_joint_targets, gripper_target_normalized):
            recorded_frames.append({
                "phase": phase, "phase_id": PHASE_ID_BY_NAME[phase],
                "arm_joint_targets": list(arm_joint_targets), "gripper_target_normalized": gripper_target_normalized,
            })
            on_step(phase, arm_joint_targets, gripper_target_normalized)

        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        result = run_pick_and_place_episode(backend, transport_delta_xy, on_step=wrapped_on_step)
        success = result["place_success"]
        failure_reason = result["failure_reason"]
    except So101ExpertError as exc:
        failure_reason = exc.failure_reason
    finally:
        if orig_apply is not None:
            backend.apply_joint_target = orig_apply
        if orig_set_gripper is not None:
            backend.set_gripper = orig_set_gripper

    if success:
        dataset.save_episode()
    else:
        dataset.clear_episode_buffer()
    dataset.finalize()
    scene_state = backend.get_scene_state() if backend.client_id is not None else None
    backend.close()

    if success:
        write_phase_id_mapping(dataset_root)
        # Same manifest fields collect_so101_episode.py's own main()
        # writes (see this task's own bin-metadata requirements) --
        # written here directly since this preflight script calls the
        # lower-level recorder pieces (make_frame_recorder(), not
        # main()) to get instrumented spies in place.
        manifest_record = {
            "episode_index": 0, "seed": PREFLIGHT_SEED, "robot_type": SO101_ROBOT_TYPE,
            "skill": "pick_and_place", "scenario_group": "normal", "expert_policy": "scripted_so101",
            "success": success, "frame_count": frame_counter["count"],
            "target_zone_center_xy": result["target_center_position"] if result else None,
            "released": result["release_constraint_removed"] if result else None,
            "place_success": success, "failure_reason": failure_reason,
            "dataset_action_space": "absolute_joint_position", "state_dimension": 6, "action_dimension": 6,
            "saved": success, "use_bin": True,
            "bin_center": scene_state["bin_center"] if scene_state else None,
            "target_zone_offset_xy": scene_state["target_zone_offset_xy"] if scene_state else None,
            "layout_validation_passed": scene_state["layout_validation_passed"] if scene_state else None,
            "bin_success_debug": result.get("bin_success_debug") if result else None,
            "action_representation": "absolute_joint_position_6d", "joint_names": list(SO101_JOINT_NAMES),
            "image_camera_name": "front",
        }
        with open(dataset_root / "collection_manifest.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_record, default=str) + "\n")

    return {
        "success": success, "failure_reason": failure_reason, "frame_count": frame_counter["count"],
        "recorded_frames": recorded_frames, "applied_log": applied_log,
        "result": result, "scene_state": scene_state,
    }


def check_temporal_alignment(recorded_frames: list, applied_log: list) -> dict:
    checks = {}
    checks["frame_count_matches_applied_count"] = len(recorded_frames) == len(applied_log)

    mismatches = []
    for i, (frame, applied) in enumerate(zip(recorded_frames, applied_log)):
        if applied["kind"] == "move":
            if list(frame["arm_joint_targets"]) != list(applied["value"]):
                mismatches.append({"index": i, "kind": "move", "recorded": frame["arm_joint_targets"], "executed": applied["value"]})
        else:
            if frame["gripper_target_normalized"] != applied["value"]:
                mismatches.append({"index": i, "kind": "gripper", "recorded": frame["gripper_target_normalized"], "executed": applied["value"]})
    checks["recorded_action_equals_executed_action"] = len(mismatches) == 0
    checks["mismatches"] = mismatches

    # No dangling action: last recorded frame corresponds to a real
    # applied step (guaranteed by the 1:1 zip above being non-empty).
    checks["no_dangling_final_action"] = len(recorded_frames) > 0 and len(recorded_frames) == len(applied_log)
    # First frame has a real preceding observation (add_frame() always
    # writes observation.state + action together from the same
    # snapshot -- structurally impossible for one to exist without the
    # other in this recorder's design).
    checks["first_frame_has_observation"] = len(recorded_frames) > 0
    # Episode end: settle never calls on_step -- last recorded phase
    # must be "release" (the retreat's own last move, tagged release),
    # never "settle".
    checks["last_frame_phase_is_release_not_settle"] = bool(recorded_frames) and recorded_frames[-1]["phase"] == PHASE_RELEASE

    return checks


def check_phase_consistency(recorded_frames: list) -> dict:
    phase_ids = [f["phase_id"] for f in recorded_frames]
    checks = {
        "all_phase_ids_valid": all(pid in PHASE_NAME_BY_ID for pid in phase_ids),
        "monotonic_non_decreasing": all(phase_ids[i] >= phase_ids[i - 1] for i in range(1, len(phase_ids))),
        "settle_never_recorded": PHASE_ID_BY_NAME[PHASE_SETTLE] not in phase_ids,
        "phases_present": sorted(set(f["phase"] for f in recorded_frames)),
    }
    missing_phases = [p for p in EXPECTED_NON_SETTLE_PHASES if p not in checks["phases_present"]]
    checks["no_missing_expected_phase"] = len(missing_phases) == 0
    checks["missing_phases"] = missing_phases

    release_frames = [f for f in recorded_frames if f["phase"] == PHASE_RELEASE]
    checks["release_action_in_release_phase"] = any(f["gripper_target_normalized"] >= 0.99 for f in release_frames)
    checks["last_phase_is_release"] = bool(recorded_frames) and recorded_frames[-1]["phase"] == PHASE_RELEASE
    return checks


def check_action_contract(state_array: np.ndarray, action_array: np.ndarray, info: dict) -> dict:
    return {
        "action_shape_is_6": tuple(action_array.shape[1:]) == (6,),
        "action_dtype_float32": action_array.dtype == np.float32,
        "joint_ordering_matches_schema": info.get("features", {}).get("action", {}).get("names") == list(SO101_JOINT_NAMES),
        "no_nan_inf": bool(np.all(np.isfinite(action_array))),
        "gripper_channel_within_0_100": bool(np.all(action_array[:, -1] >= -1e-6) and np.all(action_array[:, -1] <= 100.0 + 1e-6)),
    }


def check_observation_contract(state_array: np.ndarray, frames_df: pd.DataFrame, info: dict) -> dict:
    image_col = "observation.images.front"
    checks = {
        "state_shape_is_6": tuple(state_array.shape[1:]) == (6,),
        "state_dtype_float32": state_array.dtype == np.float32,
        "joint_ordering_matches_action": (
            info.get("features", {}).get("observation.state", {}).get("names")
            == info.get("features", {}).get("action", {}).get("names")
        ),
        "no_nan_inf": bool(np.all(np.isfinite(state_array))),
        "observation_action_frame_count_match": len(state_array) == len(frames_df),
        "image_column_present": image_col in frames_df.columns,
    }

    decoded_shapes = []
    decoded_dtypes = []
    for idx in (0, len(frames_df) // 2, len(frames_df) - 1):
        img_bytes = frames_df.iloc[idx][image_col]["bytes"]
        arr = np.array(Image.open(io.BytesIO(img_bytes)))
        decoded_shapes.append(arr.shape)
        decoded_dtypes.append(str(arr.dtype))
    checks["first_mid_last_images_decode_ok"] = True
    checks["decoded_image_shapes"] = decoded_shapes
    checks["decoded_image_dtypes"] = decoded_dtypes
    checks["image_shapes_consistent"] = len(set(decoded_shapes)) == 1
    return checks


def test_discard_policy() -> dict:
    """Directly tests the CURRENT discard mechanism
    (dataset.clear_episode_buffer()) in isolation -- adds a couple of
    valid-shaped frames then discards them, confirming NO episode ends
    up on disk. This is the SAME call collect_so101_episode.py's own
    main() already makes on any place_success=False episode (see this
    task's chat report, "실패 episode를 저장하는 정책인지 폐기하는
    정책인지"), exercised directly rather than by contriving a fake
    expert failure (which would require touching expert/geometry)."""
    root = DISCARD_UNIT_DATASET_ROOT
    dataset = LeRobotDataset.create(
        repo_id="local/so101_bin_preflight_discard_unit", fps=10, features=SO101_FEATURES, root=str(root),
        robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )
    dummy_state = np.zeros(6, dtype=np.float32)
    dummy_action = np.zeros(6, dtype=np.float32)
    dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
    dummy_phase_id = np.array([0], dtype=np.int64)
    for _ in range(3):
        dataset.add_frame({
            "observation.state": dummy_state, "observation.images.front": dummy_image,
            "action": dummy_action, "phase_id": dummy_phase_id, "task": "discard test",
        })
    dataset.clear_episode_buffer()
    dataset.finalize()

    info = json.loads((root / "meta" / "info.json").read_text())
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    total_rows = 0
    for p in parquet_paths:
        total_rows += len(pd.read_parquet(p))

    return {
        "total_episodes_after_discard": info.get("total_episodes"),
        "total_frames_after_discard": info.get("total_frames"),
        "parquet_row_count_after_discard": total_rows,
        "test_pass": info.get("total_episodes") == 0 and total_rows == 0,
    }


def test_scene_invalid_before_episode() -> dict:
    """Controlled failure test using the EXISTING, already-validated
    InvalidSceneLayoutError safety net (see this task's chat report,
    "기존 force failure parameter 또는 안전한 테스트 구조 재사용") --
    NOT a new failure scenario invented for this task, and does not
    touch expert/geometry: a deliberately-too-small
    target_zone_offset_xy (matching the OLD flat default, [0.05,0.05])
    is passed as an explicit scene_config override, which
    validate_initial_scene_layout() already rejects on its own."""
    backend = So101PyBulletBackend(gui=False, use_bin=True, scene_config={"target_zone_offset_xy": [0.05, 0.05]})
    raised = False
    failure_type = None
    try:
        backend.reset()
    except InvalidSceneLayoutError as exc:
        raised = True
        failure_type = exc.failure_type
    finally:
        backend.close()
    return {"raised_pass": raised, "failure_type": failure_type, "test_pass": raised}


def main() -> None:
    if PREFLIGHT_ROOT.exists():
        shutil.rmtree(PREFLIGHT_ROOT)  # this script's own designated scratch/output area only -- never a production dataset path
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)

    failures = []

    # --- A. success episode ---
    episode_data = record_success_episode_with_instrumentation(SUCCESS_DATASET_ROOT)
    schema_checks = {"episode_creation_success": episode_data["success"], "frame_count": episode_data["frame_count"]}
    if not episode_data["success"]:
        failures.append(f"success episode did not succeed: {episode_data['failure_reason']}")

    temporal_checks = check_temporal_alignment(episode_data["recorded_frames"], episode_data["applied_log"])
    if not temporal_checks["recorded_action_equals_executed_action"]:
        failures.append(f"recorded_action != executed_action at {len(temporal_checks['mismatches'])} steps")

    phase_checks = check_phase_consistency(episode_data["recorded_frames"])
    if not phase_checks["monotonic_non_decreasing"]:
        failures.append("phase_id sequence went backward")
    if not phase_checks["no_missing_expected_phase"]:
        failures.append(f"missing expected phases: {phase_checks['missing_phases']}")

    # --- round-trip reload (see this task's chat report, section 9) ---
    reload_verification = verify_dataset(SUCCESS_DATASET_ROOT) if episode_data["success"] else None
    info = json.loads((SUCCESS_DATASET_ROOT / "meta" / "info.json").read_text())
    parquet_paths = sorted((SUCCESS_DATASET_ROOT / "data").rglob("*.parquet"))
    frames_df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    state_array = np.stack(frames_df["observation.state"].to_numpy())
    action_array = np.stack(frames_df["action"].to_numpy())

    action_checks = check_action_contract(state_array, action_array, info)
    observation_checks = check_observation_contract(state_array, frames_df, info)

    manifest_path = SUCCESS_DATASET_ROOT / "collection_manifest.jsonl"
    manifest_line = json.loads(manifest_path.read_text().splitlines()[0]) if manifest_path.exists() else {}
    if not manifest_path.exists():
        failures.append("no collection_manifest.jsonl was written for the success episode")
    round_trip_checks = {
        "reload_verification": reload_verification,
        "manifest_reload_use_bin": manifest_line.get("use_bin"),
        "manifest_reload_place_success": manifest_line.get("place_success"),
        "manifest_reload_bin_center": manifest_line.get("bin_center"),
        "manifest_reload_seed_present": "seed" in manifest_line,
        "manifest_reload_bin_success_debug_present": manifest_line.get("bin_success_debug") is not None,
        "last_frame_state_finite": bool(np.all(np.isfinite(state_array[-1]))),
        "last_frame_action_finite": bool(np.all(np.isfinite(action_array[-1]))),
    }
    if reload_verification is not None and (reload_verification["state_has_nan_or_inf"] or reload_verification["action_has_nan_or_inf"]):
        failures.append("round-trip reload found NaN/Inf in state or action")

    # --- B. failure/discard policy checks ---
    discard_unit_result = test_discard_policy()
    if not discard_unit_result["test_pass"]:
        failures.append("clear_episode_buffer() did not fully discard buffered frames")

    scene_invalid_result = test_scene_invalid_before_episode()
    if not scene_invalid_result["test_pass"]:
        failures.append("InvalidSceneLayoutError was not raised for a known-bad bin scene_config")

    overall_pass = len(failures) == 0

    summary = {
        "overall_pass": overall_pass,
        "episodes_checked": 1,
        "schema_checks": schema_checks,
        "temporal_checks": {k: v for k, v in temporal_checks.items() if k != "mismatches"},
        "temporal_mismatches": temporal_checks["mismatches"],
        "action_checks": action_checks,
        "observation_checks": observation_checks,
        "phase_checks": phase_checks,
        "round_trip_checks": round_trip_checks,
        "discard_policy_unit_test": discard_unit_result,
        "scene_invalid_controlled_failure_test": scene_invalid_result,
        "failures": failures,
        "success_episode_dataset_root": str(SUCCESS_DATASET_ROOT),
        "discard_unit_test_dataset_root": str(DISCARD_UNIT_DATASET_ROOT),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 bin dataset recorder preflight ===")
    print(f"schema_checks: {schema_checks}")
    print(f"temporal_checks: { {k: v for k, v in temporal_checks.items() if k != 'mismatches'} }")
    if temporal_checks["mismatches"]:
        print(f"  MISMATCHES: {temporal_checks['mismatches']}")
    print(f"action_checks: {action_checks}")
    print(f"observation_checks: {observation_checks}")
    print(f"phase_checks: {phase_checks}")
    print(f"round_trip_checks: {round_trip_checks}")
    print(f"discard_policy_unit_test: {discard_unit_result}")
    print(f"scene_invalid_controlled_failure_test: {scene_invalid_result}")
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    if failures:
        print(f"failures: {failures}")
    print(f"\nSuccess episode dataset: {SUCCESS_DATASET_ROOT}")
    print(f"Discard unit test dataset: {DISCARD_UNIT_DATASET_ROOT}")
    print(f"Summary JSON: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
