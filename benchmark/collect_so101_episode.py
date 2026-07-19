"""SO-101 LeRobot dataset recorder -- minimal, single-episode (see this
task's chat report, "SO-101 Dataset Recorder" and "공통 Expert 모듈
정리"). Entirely independent of Panda's own
benchmark/collect_recycling_dataset.py / collect_v2_dataset.py /
build_v3_dataset.py -- does not import, call, or modify any of them,
and does not touch robot_sim/pybullet_panda_backend.py.

Records ONE fixed-object-position pick-and-place episode from
robot_sim.so101_pybullet_backend.So101PyBulletBackend into a real
LeRobotDataset, using benchmark/so101_dataset_schema.py's 6-D
joint-centric observation.state/action.

As of the "공통 Expert 모듈 정리" task, this file no longer
reimplements the expert phase sequence -- it calls
benchmark/so101_scripted_expert.py::run_pick_and_place_episode(),
the SAME shared Expert benchmark/smoke_so101_pick_and_place.py calls,
with an `on_step` callback that records one dataset frame per
control step. Each step's action is the SAME absolute joint target
the Expert computed via a SINGLE
So101PyBulletBackend.compute_joint_target_from_ee_delta() call and
then actually applied via apply_joint_target() -- no duplicate IK
computation.

Data-integrity rule: every recorded action is a fresh, real target
computed from that frame's own real (pre-action) state -- never a
constant, never copied from a different step, never invented.

Run:
  .venv-vla/bin/python -m benchmark.collect_so101_episode \\
    --dataset-root datasets/so101_recorder_smoke --repo-id local/so101_recorder_smoke --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from benchmark.so101_dataset_schema import (
    SO101_FEATURES,
    SO101_JOINT_NAMES,
    SO101_ROBOT_TYPE,
    pack_action,
    pack_phase_id,
    pack_state,
    validate_image,
)
from benchmark.so101_scripted_expert import (
    PHASE_ID_BY_NAME,
    PHASE_NAME_BY_ID,
    So101ExpertError,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import (
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_WIDTH,
    InvalidSceneLayoutError,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = "datasets/so101_recorder_smoke"
DEFAULT_REPO_ID = "local/so101_recorder_smoke"
DEFAULT_FPS = 10  # matches this project's own V3 (Panda) convention -- see collect_recycling_dataset.py's own DEFAULT_FPS
DEFAULT_INSTRUCTION = "Pick up the object and place it in the target zone."

# Transport delta MUST match scene_config's own target_zone_offset_xy
# default ([0.05, 0.05]) so the object actually lands in the target
# zone the backend built -- same assumption
# smoke_so101_pick_and_place.py's own TRANSPORT_DELTA_XY_GOOD states.
# ONLY used for the flat (use_bin=False, the default) path -- see
# main()'s own bin branch, which reads the backend's OWN resolved
# target_zone_offset_xy after reset() instead (see this task's chat
# report, "SO-101 dataset recorder preflight 검증").
TRANSPORT_DELTA_XY = [0.05, 0.05]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_frame_recorder(dataset, backend: So101PyBulletBackend, task: str):
    """Returns (on_step, frame_counter) -- on_step is handed to
    run_pick_and_place_episode() as its recording hook. Called BEFORE
    each actual arm/gripper apply (see So101ExpertError/move_to_target's
    own docstring in so101_scripted_expert.py for the observation_t ->
    action_t ordering guarantee): reads the REAL current state (not
    fabricated), the front camera image at that same instant, and packs
    the phase's already-computed target as this frame's action."""
    frame_counter = {"count": 0}

    def on_step(phase: str, arm_joint_targets: list, gripper_target_normalized: float) -> None:
        obs = backend.get_observation()
        state = pack_state(obs["joint_positions"], obs["gripper_position_normalized"])
        front_image = backend.render_front_camera()
        validate_image(front_image)
        action = pack_action(arm_joint_targets, gripper_target_normalized)
        phase_id = pack_phase_id(PHASE_ID_BY_NAME[phase])

        dataset.add_frame({
            "observation.state": state,
            "observation.images.front": front_image,
            "action": action,
            "phase_id": phase_id,
            "task": task,
        })
        frame_counter["count"] += 1

    return on_step, frame_counter


def write_phase_id_mapping(dataset_root: Path) -> None:
    """Sidecar JSON (not forced into LeRobotDataset's own auto-managed
    info.json schema) -- see this task's chat report, "phase 저장
    방식". Plain file I/O, so it works regardless of LeRobotDataset's
    internal metadata API."""
    mapping_path = dataset_root / "meta" / "phase_id_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in PHASE_NAME_BY_ID.items()}, f, indent=2)


def run_episode(dataset, backend: So101PyBulletBackend, task: str, transport_delta_xy: list = None) -> dict:
    """`transport_delta_xy=None` (default) -- uses the EXISTING module
    constant TRANSPORT_DELTA_XY, exactly as before this task (flat
    scenes, use_bin=False). A caller running a bin episode passes the
    backend's OWN resolved `scene_config["target_zone_offset_xy"]`
    (read AFTER reset(), same pattern already used by
    benchmark/smoke_so101_bin_place.py) so transport actually lands at
    THIS episode's real bin center, not a hardcoded flat value."""
    delta = transport_delta_xy if transport_delta_xy is not None else TRANSPORT_DELTA_XY
    initial_joint_positions = backend.get_joint_positions()
    object_position, _ = backend.get_object_pose()

    on_step, frame_counter = make_frame_recorder(dataset, backend, task)

    success = False
    released = False
    place_success = False
    failure_reason = None
    target_zone_center_xy = None
    bin_success_debug = None

    try:
        result = run_pick_and_place_episode(backend, delta, on_step=on_step)
        released = result["release_constraint_removed"]
        place_success = result["place_success"]
        failure_reason = result["failure_reason"]
        target_zone_center_xy = result["target_center_position"]
        bin_success_debug = result.get("bin_success_debug")
        success = place_success
    except So101ExpertError as exc:
        failure_reason = exc.failure_reason

    final_joint_positions = backend.get_joint_positions()

    return {
        "success": success, "frame_count": frame_counter["count"], "failure_reason": failure_reason,
        "object_position": object_position, "target_zone_center_xy": target_zone_center_xy,
        "initial_joint_positions": initial_joint_positions, "final_joint_positions": final_joint_positions,
        "released": released, "place_success": place_success, "bin_success_debug": bin_success_debug,
    }


def verify_dataset(dataset_root: Path) -> dict:
    """Minimal post-hoc inspection -- reads back what was actually
    written, does not trust in-memory state."""
    import pandas as pd

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    stats_path = dataset_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    phase_mapping_path = dataset_root / "meta" / "phase_id_mapping.json"
    phase_mapping = json.loads(phase_mapping_path.read_text()) if phase_mapping_path.exists() else None

    parquet_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    frames = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    state_array = np.stack(frames["observation.state"].to_numpy())
    action_array = np.stack(frames["action"].to_numpy())
    phase_id_array = np.stack(frames["phase_id"].to_numpy()) if "phase_id" in frames.columns else None

    result = {
        "robot_type": info.get("robot_type"),
        "observation_state_shape_declared": info.get("features", {}).get("observation.state", {}).get("shape"),
        "action_shape_declared": info.get("features", {}).get("action", {}).get("shape"),
        "feature_names": info.get("features", {}).get("observation.state", {}).get("names"),
        "episode_count": int(frames["episode_index"].nunique()),
        "frame_count": int(len(frames)),
        "state_array_shape": list(state_array.shape),
        "action_array_shape": list(action_array.shape),
        "state_has_nan_or_inf": bool(not np.all(np.isfinite(state_array))),
        "action_has_nan_or_inf": bool(not np.all(np.isfinite(action_array))),
        "action_all_identical": bool(np.all(action_array == action_array[0])),
        "gripper_action_min": float(action_array[:, -1].min()),
        "gripper_action_max": float(action_array[:, -1].max()),
        "gripper_includes_open_and_close": bool(action_array[:, -1].min() < 10.0 and action_array[:, -1].max() > 90.0),
        "stats_present": stats is not None,
        "phase_id_present_in_all_frames": bool(phase_id_array is not None and len(phase_id_array) == len(frames)),
        "phase_id_unique_values": sorted(set(int(v) for v in phase_id_array.flatten())) if phase_id_array is not None else None,
        "phase_id_mapping": phase_mapping,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--use-bin", action="store_true", help="record into the open-top bin scene instead of the flat target marker (see robot_sim.So101PyBulletBackend's own use_bin)")
    args = parser.parse_args()

    root = resolve(args.dataset_root)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=SO101_FEATURES, root=str(root),
        robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )

    manifest_path = root / "collection_manifest.jsonl"
    backend = So101PyBulletBackend(gui=False, use_bin=args.use_bin)
    try:
        backend.reset()
        # Bin scenes need transport to land at THIS episode's actual
        # (production-default-resolved) bin center, not the flat
        # module constant -- read back after reset(), never hardcoded
        # a second time here (same pattern
        # benchmark/smoke_so101_bin_place.py already established).
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"]) if args.use_bin else None
        result = run_episode(dataset, backend, args.instruction, transport_delta_xy=transport_delta_xy)

        if result["success"]:
            dataset.save_episode()
            saved = True
        else:
            dataset.clear_episode_buffer()
            saved = False
        scene_state = backend.get_scene_state()
    finally:
        dataset.finalize()
        backend.close()

    if saved:
        write_phase_id_mapping(root)

    manifest_record = {
        "episode_index": 0, "seed": args.seed, "robot_type": SO101_ROBOT_TYPE,
        "skill": "pick_and_place", "scenario_group": "normal", "expert_policy": "scripted_so101",
        "success": result["success"], "frame_count": result["frame_count"],
        "object_position": result["object_position"], "target_zone_center_xy": result["target_zone_center_xy"],
        "initial_joint_positions": result["initial_joint_positions"], "final_joint_positions": result["final_joint_positions"],
        "released": result["released"], "place_success": result["place_success"],
        "failure_reason": result["failure_reason"],
        "dataset_action_space": "absolute_joint_position", "state_dimension": 6, "action_dimension": 6,
        "saved": saved,
        # --- Pure additions (see this task's chat report, "bin episode
        # metadata 요구사항") -- present for BOTH flat and bin episodes
        # (generically useful, not bin-specific), existing manifest
        # consumers reading only the keys above are unaffected. ---
        "use_bin": args.use_bin,
        "bin_center": scene_state["bin_center"],
        "target_zone_offset_xy": scene_state["target_zone_offset_xy"],
        "surface_footprint_xy": backend.scene_config["surface_footprint_xy"],
        "layout_validation_passed": scene_state["layout_validation_passed"],
        "bin_success_debug": result.get("bin_success_debug"),
        "action_representation": "absolute_joint_position_6d",
        "joint_names": list(SO101_JOINT_NAMES),
        "image_camera_name": "front",
        "image_resolution": [FRONT_CAMERA_HEIGHT, FRONT_CAMERA_WIDTH],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record, default=str) + "\n")

    print("=== SO-101 episode recorder ===")
    print(f"success={result['success']} frame_count={result['frame_count']} failure_reason={result['failure_reason']}")
    print(f"saved={saved}")
    print(f"Dataset root: {root}")
    print(f"Manifest: {manifest_path}")

    if saved:
        verification = verify_dataset(root)
        print("\n=== Dataset verification ===")
        for k, v in verification.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
