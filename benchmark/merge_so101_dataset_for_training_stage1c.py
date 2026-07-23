"""Stage 1C combined-training-dataset builder (see this task's chat
report, "Training dataset merge"). Produces
`datasets/so101_bin_stage1c_training_combined` = original 200 cube
episodes (indices 0-199, copied read-only from
`datasets/so101_bin_main_200`) + Stage 1A's own 70 episodes (indices
200-269, re-collected via LeRobotDataset.resume(), same deterministic
seeds/regions as benchmark.collect_so101_stage1a_xy_reinforcement's own
COLLECTION_PLAN) + Stage 1B's own 100 box episodes (indices 270-369,
re-collected using benchmark.collect_so101_stage1b_box_dataset's own
COLLECTION_PLAN/REGION_DEFS/BOX_FOOTPRINT_XY) + Stage 1C's own 100
cylinder episodes (indices 370-469, re-collected using
benchmark.collect_so101_stage1c_cylinder_dataset's own
POSITION_GROUP_PLAN, via Expert V2.1).

Same re-collection strategy benchmark.merge_so101_dataset_for_training_stage1b.py
already used (LeRobotDataset's own API does not offer a
straightforward "append another already-materialized dataset" -- this
project's own established pattern is deterministic re-collection
instead, which reproduces byte-for-byte since every seed/region/scene_config
is identical to the source dataset's own collector).

`datasets/so101_bin_main_200`, `datasets/so101_bin_stage1a_xy_70`,
`datasets/so101_bin_stage1b_box_100`, and `datasets/so101_bin_stage1c_cylinder_100`
are NEVER opened in write mode -- only read (manifests) or copied
(original, read-only, sha256-verified before/after).

Run:
  .venv-vla/bin/python -m benchmark.merge_so101_dataset_for_training_stage1c
"""

import hashlib
import json
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY,
)
from benchmark.collect_so101_bin_dataset import DEFAULT_INSTRUCTION, collect_episode, resolve
from benchmark.collect_so101_episode import verify_dataset, write_phase_id_mapping
from benchmark.collect_so101_stage1a_xy_reinforcement import COLLECTION_PLAN as STAGE1A_COLLECTION_PLAN, REGION_DEFS as STAGE1A_REGION_DEFS
from benchmark.collect_so101_stage1b_box_dataset import (
    BOX_FOOTPRINT_XY,
    COLLECTION_PLAN as STAGE1B_COLLECTION_PLAN,
    REGION_DEFS as STAGE1B_REGION_DEFS,
)
from benchmark.collect_so101_stage1c_cylinder_dataset import (
    CYLINDER_HEIGHT_M,
    CYLINDER_RADIUS_M,
    POSITION_GROUP_PLAN as STAGE1C_POSITION_GROUP_PLAN,
    collect_episode_v2_1,
)
from benchmark.so101_scripted_expert import PHASE_ID_BY_NAME
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, DEFAULT_SCENE_CONFIG, OBJECT_MASS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_ROOT = "datasets/so101_bin_main_200"
STAGE1A_DATASET_ROOT = "datasets/so101_bin_stage1a_xy_70"
STAGE1B_DATASET_ROOT = "datasets/so101_bin_stage1b_box_100"
STAGE1C_DATASET_ROOT = "datasets/so101_bin_stage1c_cylinder_100"
COMBINED_DATASET_ROOT = "datasets/so101_bin_stage1c_training_combined"
COMBINED_REPO_ID = "local/so101_bin_stage1c_training_combined"


def hash_dataset_files(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def collect_plan_onto(dataset, episode_index_counter, collection_plan, region_defs, scene_config_override, position_manifest, extra_fields_fn):
    """UNCHANGED from benchmark.merge_so101_dataset_for_training_stage1b.py's
    own helper of the same name -- re-collects a V1-driven COLLECTION_PLAN
    (Stage 1A cube or Stage 1B box) onto the given (already-resumed)
    dataset."""
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    for split in ("train", "validation", "test"):
        for region_name, (target, seed_candidates) in collection_plan[split].items():
            region = region_defs[region_name]
            saved_count = 0
            for seed in seed_candidates:
                if saved_count >= target:
                    break
                outcome = collect_episode(
                    dataset, seed, DEFAULT_INSTRUCTION, episode_index_counter,
                    randomization_mode=RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY,
                    x_range=region["x_range"], y_range=region["y_range"],
                    bin_center_override_xy=fixed_bin_center_xy, scene_config=scene_config_override,
                )
                print(f"[combined:{split}/{region_name}] seed={seed} saved={outcome['saved']} ({saved_count + int(outcome['saved'])}/{target})")
                if outcome["saved"]:
                    saved_count += 1
                    sampled = outcome["sampled_object_position"]
                    x_offset = sampled[0] - DEFAULT_OBJECT_POSITION[0]
                    y_offset = sampled[1] - DEFAULT_OBJECT_POSITION[1]
                    record = {
                        "episode_index": outcome["episode_index"], "environment_seed": seed, "split": split,
                        "position_region": region_name, "object_x": sampled[0], "object_y": sampled[1],
                        "x_offset": x_offset, "y_offset": y_offset,
                        "distance_from_center": (x_offset ** 2 + y_offset ** 2) ** 0.5,
                        "is_corner_region": region["is_corner"], "is_negative_x_region": x_offset < 0,
                        "phase_id": PHASE_ID_BY_NAME["release"], "failure_reason": None,
                    }
                    record.update(extra_fields_fn())
                    position_manifest.append(record)
            if saved_count < target:
                raise RuntimeError(f"Combined re-collection shortfall: split={split!r} region={region_name!r} saved {saved_count}/{target}")


def collect_cylinder_plan_onto(dataset, episode_index_counter, position_manifest):
    """Cylinder counterpart of collect_plan_onto() -- drives Expert V2.1
    (collect_episode_v2_1(), NOT V1's collect_episode()) across Stage
    1C's own POSITION_GROUP_PLAN (region-nested-within-group structure,
    see benchmark/collect_so101_stage1c_cylinder_dataset.py)."""
    import math

    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config_override = {
        "surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
        "object_shape": "cylinder", "object_radius": CYLINDER_RADIUS_M, "object_height": CYLINDER_HEIGHT_M,
    }
    from benchmark.collect_so101_stage1b_box_dataset import REGION_DEFS as STAGE1B_REGION_DEFS_FOR_CYLINDER
    from benchmark.collect_so101_stage1c_cylinder_dataset import YAW_CYCLE_DEG

    yaw_cursor = {"i": 0}

    def next_yaw_deg():
        yaw = YAW_CYCLE_DEG[yaw_cursor["i"] % len(YAW_CYCLE_DEG)]
        yaw_cursor["i"] += 1
        return yaw

    for split in ("train", "validation", "test"):
        for position_group, regions in STAGE1C_POSITION_GROUP_PLAN[split].items():
            for region_name, (target, seed_candidates) in regions.items():
                region = STAGE1B_REGION_DEFS_FOR_CYLINDER[region_name]
                saved_count = 0
                for seed in seed_candidates:
                    if saved_count >= target:
                        break
                    yaw_deg = next_yaw_deg()
                    outcome = collect_episode_v2_1(
                        dataset, seed, DEFAULT_INSTRUCTION, episode_index_counter, yaw_deg,
                        region["x_range"], region["y_range"], fixed_bin_center_xy, scene_config_override,
                    )
                    print(f"[combined:{split}/{position_group}/{region_name}] seed={seed} yaw={yaw_deg}deg saved={outcome['saved']} "
                          f"({saved_count + int(outcome['saved'])}/{target})")
                    if outcome["saved"]:
                        saved_count += 1
                        sampled = outcome["sampled_object_position"]
                        x_offset = sampled[0] - DEFAULT_OBJECT_POSITION[0]
                        y_offset = sampled[1] - DEFAULT_OBJECT_POSITION[1]
                        position_manifest.append({
                            "episode_index": outcome["episode_index"], "environment_seed": seed, "split": split,
                            "position_group": position_group, "position_region": region_name,
                            "source": "stage1c", "object_type": "upright_cylinder_v1", "object_shape": "cylinder",
                            "object_radius": CYLINDER_RADIUS_M, "object_height": CYLINDER_HEIGHT_M,
                            "object_x": sampled[0], "object_y": sampled[1], "x_offset": x_offset, "y_offset": y_offset,
                            "distance_from_center": (x_offset ** 2 + y_offset ** 2) ** 0.5,
                            "is_corner_region": region["is_corner"], "is_negative_x_region": x_offset < 0,
                            "yaw": math.radians(yaw_deg), "phase_id": PHASE_ID_BY_NAME["release"],
                            "object_mass": outcome["object_mass"], "lateral_friction": outcome["lateral_friction"],
                            "scripted_expert_success": True, "failure_reason": None,
                        })
                if saved_count < target:
                    raise RuntimeError(f"Combined cylinder re-collection shortfall: split={split!r} group={position_group!r} region={region_name!r} saved {saved_count}/{target}")


def main() -> None:
    original_root = resolve(ORIGINAL_DATASET_ROOT)
    stage1a_root = resolve(STAGE1A_DATASET_ROOT)
    stage1b_root = resolve(STAGE1B_DATASET_ROOT)
    stage1c_root = resolve(STAGE1C_DATASET_ROOT)
    combined_root = resolve(COMBINED_DATASET_ROOT)
    if combined_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {combined_root}")

    original_hashes_before = hash_dataset_files(original_root)
    stage1a_hashes_before = hash_dataset_files(stage1a_root)
    stage1b_hashes_before = hash_dataset_files(stage1b_root)
    stage1c_hashes_before = hash_dataset_files(stage1c_root)
    print(f"Original dataset: {len(original_hashes_before)} files hashed before copy.")
    print(f"Stage 1A/1B/1C source datasets hashed before merge (read-only, never copied): "
          f"{len(stage1a_hashes_before)}/{len(stage1b_hashes_before)}/{len(stage1c_hashes_before)} files.")

    shutil.copytree(original_root, combined_root)
    print(f"Copied {original_root} -> {combined_root}")
    if hash_dataset_files(original_root) != original_hashes_before:
        raise RuntimeError("Original dataset changed during copytree() -- aborting.")

    dataset = LeRobotDataset.resume(repo_id=COMBINED_REPO_ID, root=str(combined_root))
    starting_count = dataset.meta.total_episodes
    print(f"Resumed combined dataset at episode_index={starting_count} (expected 200).")
    if starting_count != 200:
        raise RuntimeError(f"Expected to resume at 200 episodes, got {starting_count}")

    episode_index_counter = {"count": starting_count}
    position_manifest = []

    try:
        # Stage 1A's own 70 episodes -> combined indices 200-269
        stage1a_scene_config = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY}
        collect_plan_onto(
            dataset, episode_index_counter, STAGE1A_COLLECTION_PLAN, STAGE1A_REGION_DEFS, stage1a_scene_config,
            position_manifest, lambda: {"source": "stage1a", "object_type": "cube", "object_shape": "cube"},
        )
        count_after_stage1a = episode_index_counter["count"]
        if count_after_stage1a != 270:
            raise RuntimeError(f"Expected 270 episodes after Stage 1A re-collection, got {count_after_stage1a}")

        # Stage 1B's own 100 box episodes -> combined indices 270-369
        stage1b_scene_config = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY, "object_footprint_xy": BOX_FOOTPRINT_XY}
        collect_plan_onto(
            dataset, episode_index_counter, STAGE1B_COLLECTION_PLAN, STAGE1B_REGION_DEFS, stage1b_scene_config,
            position_manifest, lambda: {
                "source": "stage1b", "object_type": "rectangular_box_v1", "object_shape": "box",
                "object_dimensions": {"half_extent_x_m": BOX_FOOTPRINT_XY[0], "half_extent_y_m": BOX_FOOTPRINT_XY[1],
                                      "half_extent_z_m": DEFAULT_SCENE_CONFIG["object_height"] / 2.0},
                "object_mass": OBJECT_MASS, "object_friction": 0.5, "yaw": 0.0, "scripted_expert_success": True,
            },
        )
        count_after_stage1b = episode_index_counter["count"]
        if count_after_stage1b != 370:
            raise RuntimeError(f"Expected 370 episodes after Stage 1B re-collection, got {count_after_stage1b}")

        # Stage 1C's own 100 cylinder episodes -> combined indices 370-469
        collect_cylinder_plan_onto(dataset, episode_index_counter, position_manifest)
    finally:
        dataset.finalize()

    write_phase_id_mapping(combined_root)

    manifest_path = combined_root / "combined_position_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in position_manifest:
            f.write(json.dumps(record, default=str) + "\n")

    final_count = episode_index_counter["count"]
    print(f"Combined dataset final episode count: {final_count} (expected 470).")
    if final_count != 470:
        raise RuntimeError(f"Expected 470 total episodes, got {final_count}")

    if hash_dataset_files(original_root) != original_hashes_before:
        raise RuntimeError("Original dataset changed during combined-dataset build -- aborting.")
    if hash_dataset_files(stage1a_root) != stage1a_hashes_before:
        raise RuntimeError("Stage 1A dataset changed during combined-dataset build -- aborting.")
    if hash_dataset_files(stage1b_root) != stage1b_hashes_before:
        raise RuntimeError("Stage 1B dataset changed during combined-dataset build -- aborting.")
    if hash_dataset_files(stage1c_root) != stage1c_hashes_before:
        raise RuntimeError("Stage 1C dataset changed during combined-dataset build -- aborting.")
    print("Original + Stage 1A + Stage 1B + Stage 1C source dataset files confirmed byte-identical before/after (sha256).")

    verification = verify_dataset(combined_root)
    print(f"verify_dataset(combined): state_has_nan_or_inf={verification.get('state_has_nan_or_inf')} "
          f"action_has_nan_or_inf={verification.get('action_has_nan_or_inf')}")
    print(f"Combined dataset: {combined_root}")


if __name__ == "__main__":
    main()
