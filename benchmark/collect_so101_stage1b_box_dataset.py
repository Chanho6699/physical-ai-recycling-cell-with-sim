"""Stage 1B rectangular-box dataset collector (see this task's chat
report, "Stage 1B: rectangular-box shape generalization"). Collects 100
NEW episodes (70 train / 15 validation / 15 held-out test) of a
rectangular box -- object_footprint_xy=[0.02, 0.03] (half-extents; full
4cm x 6cm x 4cm, 1.5x aspect ratio, X = the gripper's own closing axis,
confirmed empirically this task) -- spanning a STRATIFIED set of 13
regions covering center/interior, single-axis existing-range
boundaries, the -X corridor, all 4 corners, and 3 bridge directions
(the SAME existing-range/expanded-range band Stage 1A already
validated, 0.015m-0.01875m for the boundary/corner/bridge regions, plus
new interior coverage this task's own object-shape generalization goal
needs that Stage 1A's cube-only collection didn't).

Reuses (does NOT reimplement): benchmark.collect_so101_bin_dataset's
own `collect_episode()` -- object shape is passed via its EXISTING
`scene_config` parameter (`object_footprint_xy` override), no backend
code change needed at all (confirmed this task via direct
introspection: robot_sim/so101_pybullet_backend.py's own reset()
already reads scene_config["object_footprint_xy"]/["object_height"]
for object geometry). Mass/friction are left at their existing
defaults (OBJECT_MASS/PyBullet's own default lateralFriction) --
untouched, so they match the cube exactly. object yaw is not touched
either (stays 0.0, the existing baseline value).

Seed blocks (train=15000s, validation=16000s, test=17000s) are
disjoint from every seed block used anywhere else in this project
(original dataset 0-199; Stage 1A 5000s/6000s/7000s; zero-shot eval's
derived-seed base 500000).

Run:
  .venv-vla/bin/python -m benchmark.collect_so101_stage1b_box_dataset
"""

import datetime
import json
from pathlib import Path

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY,
)
from benchmark.collect_so101_bin_dataset import DEFAULT_INSTRUCTION, collect_episode, resolve
from benchmark.collect_so101_episode import verify_dataset, write_phase_id_mapping
from benchmark.so101_dataset_schema import SO101_FEATURES, SO101_ROBOT_TYPE
from benchmark.so101_scripted_expert import PHASE_ID_BY_NAME
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, DEFAULT_SCENE_CONFIG, OBJECT_MASS
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = "datasets/so101_bin_stage1b_box_100"
REPO_ID = "local/so101_bin_stage1b_box_100"
FPS = 10

OLD_RADIUS_M = 0.015
NEW_RADIUS_M = 0.01875
BOX_FOOTPRINT_XY = [0.02, 0.03]  # half-extents: 4cm(X, grasp axis) x 6cm(Y, long axis)
OBJECT_TYPE = "rectangular_box_v1"
OBJECT_SHAPE = "box"
OBJECT_YAW_RAD = 0.0

REGION_DEFS = {
    "center": {"x_range": (-0.003, 0.003), "y_range": (-0.003, 0.003), "is_corner": False},
    "existing_x_min": {"x_range": (-OLD_RADIUS_M, -0.010), "y_range": (-0.010, 0.010), "is_corner": False},
    "existing_x_max": {"x_range": (0.010, OLD_RADIUS_M), "y_range": (-0.010, 0.010), "is_corner": False},
    "existing_y_min": {"x_range": (-0.010, 0.010), "y_range": (-OLD_RADIUS_M, -0.010), "is_corner": False},
    "existing_y_max": {"x_range": (-0.010, 0.010), "y_range": (0.010, OLD_RADIUS_M), "is_corner": False},
    "x_min_corridor": {"x_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "y_range": (-OLD_RADIUS_M, OLD_RADIUS_M), "is_corner": False},
    "corner_pp": {"x_range": (OLD_RADIUS_M, NEW_RADIUS_M), "y_range": (OLD_RADIUS_M, NEW_RADIUS_M), "is_corner": True},
    "corner_pn": {"x_range": (OLD_RADIUS_M, NEW_RADIUS_M), "y_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "is_corner": True},
    "corner_np": {"x_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "y_range": (OLD_RADIUS_M, NEW_RADIUS_M), "is_corner": True},
    "corner_nn": {"x_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "y_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "is_corner": True},
    "bridge_plus_x": {"x_range": (OLD_RADIUS_M, NEW_RADIUS_M), "y_range": (-0.005, 0.005), "is_corner": False},
    "bridge_minus_y": {"x_range": (-0.005, 0.005), "y_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "is_corner": False},
    "bridge_plus_y": {"x_range": (-0.005, 0.005), "y_range": (OLD_RADIUS_M, NEW_RADIUS_M), "is_corner": False},
}

COLLECTION_PLAN = {
    # train spans 15000-16299 (hundreds-spaced sub-blocks per region)
    "train": {
        "center": (8, list(range(15000, 15020))),
        "existing_x_min": (6, list(range(15100, 15115))),
        "existing_x_max": (6, list(range(15200, 15215))),
        "existing_y_min": (6, list(range(15300, 15315))),
        "existing_y_max": (6, list(range(15400, 15415))),
        "x_min_corridor": (10, list(range(15500, 15525))),
        "corner_pp": (5, list(range(15600, 15615))),
        "corner_pn": (6, list(range(15700, 15716))),
        "corner_np": (5, list(range(15800, 15815))),
        "corner_nn": (5, list(range(15900, 15915))),
        "bridge_plus_x": (3, list(range(16000, 16010))),
        "bridge_minus_y": (2, list(range(16100, 16108))),
        "bridge_plus_y": (2, list(range(16200, 16208))),
    },
    # validation spans 17000-17299 (twenties-spaced sub-blocks -- small
    # per-region counts) -- entirely disjoint from train's 15000-16299.
    "validation": {
        "center": (2, list(range(17000, 17006))),
        "existing_x_min": (1, list(range(17020, 17024))),
        "existing_x_max": (1, list(range(17040, 17044))),
        "existing_y_min": (1, list(range(17060, 17064))),
        "existing_y_max": (1, list(range(17080, 17084))),
        "x_min_corridor": (2, list(range(17100, 17106))),
        "corner_pp": (1, list(range(17120, 17124))),
        "corner_pn": (1, list(range(17140, 17144))),
        "corner_np": (1, list(range(17160, 17164))),
        "corner_nn": (1, list(range(17180, 17184))),
        "bridge_plus_x": (1, list(range(17200, 17204))),
        "bridge_minus_y": (1, list(range(17220, 17224))),
        "bridge_plus_y": (1, list(range(17240, 17244))),
    },
    # test spans 18000-18299 -- entirely disjoint from train and validation.
    "test": {
        "center": (2, list(range(18000, 18006))),
        "existing_x_min": (1, list(range(18020, 18024))),
        "existing_x_max": (1, list(range(18040, 18044))),
        "existing_y_min": (1, list(range(18060, 18064))),
        "existing_y_max": (1, list(range(18080, 18084))),
        "x_min_corridor": (2, list(range(18100, 18106))),
        "corner_pp": (1, list(range(18120, 18124))),
        "corner_pn": (1, list(range(18140, 18144))),
        "corner_np": (1, list(range(18160, 18164))),
        "corner_nn": (1, list(range(18180, 18184))),
        "bridge_plus_x": (1, list(range(18200, 18204))),
        "bridge_minus_y": (1, list(range(18220, 18224))),
        "bridge_plus_y": (1, list(range(18240, 18244))),
    },
}


def main() -> None:
    root = resolve(DATASET_ROOT)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")

    for split, regions in COLLECTION_PLAN.items():
        target_sum = sum(target for target, _ in regions.values())
        expected = {"train": 70, "validation": 15, "test": 15}[split]
        assert target_sum == expected, f"{split} region targets sum to {target_sum}, expected {expected}"

    # seed-block disjointness sanity check (train/validation/test candidate pools must never overlap)
    all_seed_pools = []
    for split, regions in COLLECTION_PLAN.items():
        pool = set()
        for _, seeds in regions.values():
            pool.update(seeds)
        all_seed_pools.append((split, pool))
    for i in range(len(all_seed_pools)):
        for j in range(i + 1, len(all_seed_pools)):
            overlap = all_seed_pools[i][1] & all_seed_pools[j][1]
            assert not overlap, f"seed overlap between {all_seed_pools[i][0]} and {all_seed_pools[j][0]}: {overlap}"

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, features=SO101_FEATURES, root=str(root), robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )

    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config_override = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY, "object_footprint_xy": BOX_FOOTPRINT_XY}

    episode_index_counter = {"count": 0}
    position_manifest = []
    attempts_log = []

    try:
        for split in ("train", "validation", "test"):
            for region_name, (target, seed_candidates) in COLLECTION_PLAN[split].items():
                region = REGION_DEFS[region_name]
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
                    attempts_log.append({
                        "seed": seed, "split": split, "position_region": region_name,
                        "saved": outcome["saved"], "failure_reason": outcome["failure_reason"],
                    })
                    print(f"[{split}/{region_name}] seed={seed} saved={outcome['saved']} "
                          f"failure_reason={outcome['failure_reason']} ({saved_count + int(outcome['saved'])}/{target})")

                    if outcome["saved"]:
                        saved_count += 1
                        sampled = outcome["sampled_object_position"]
                        x_offset = sampled[0] - DEFAULT_OBJECT_POSITION[0]
                        y_offset = sampled[1] - DEFAULT_OBJECT_POSITION[1]
                        position_manifest.append({
                            "episode_index": outcome["episode_index"], "environment_seed": seed, "split": split,
                            "position_region": region_name,
                            "object_type": OBJECT_TYPE, "object_shape": OBJECT_SHAPE,
                            "object_dimensions": {"half_extent_x_m": BOX_FOOTPRINT_XY[0], "half_extent_y_m": BOX_FOOTPRINT_XY[1],
                                                  "half_extent_z_m": DEFAULT_SCENE_CONFIG["object_height"] / 2.0},
                            "object_mass": OBJECT_MASS,
                            "object_friction": 0.5,  # PyBullet default lateralFriction -- unset explicitly, same as cube
                            "object_x": sampled[0], "object_y": sampled[1], "x_offset": x_offset, "y_offset": y_offset,
                            "distance_from_center": (x_offset ** 2 + y_offset ** 2) ** 0.5,
                            "is_corner_region": region["is_corner"], "is_negative_x_region": x_offset < 0,
                            "yaw": OBJECT_YAW_RAD, "phase_id": PHASE_ID_BY_NAME["release"],
                            "scripted_expert_success": True, "failure_reason": None,
                        })
                if saved_count < target:
                    raise RuntimeError(
                        f"Stage 1B box collection shortfall: split={split!r} region={region_name!r} "
                        f"saved only {saved_count}/{target} from {len(seed_candidates)} seed candidates."
                    )
    finally:
        dataset.finalize()

    if episode_index_counter["count"] > 0:
        write_phase_id_mapping(root)

    manifest_path = root / "stage1b_position_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in position_manifest:
            f.write(json.dumps(record, default=str) + "\n")

    attempts_log_path = root / "stage1b_collection_attempts_log.jsonl"
    with open(attempts_log_path, "w", encoding="utf-8") as f:
        for record in attempts_log:
            f.write(json.dumps(record, default=str) + "\n")

    verification = verify_dataset(root)

    summary = {
        "dataset_name": root.name, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "object_type": OBJECT_TYPE, "box_footprint_xy": BOX_FOOTPRINT_XY,
        "total_episodes_saved": episode_index_counter["count"],
        "split_counts": {split: sum(1 for r in position_manifest if r["split"] == split) for split in ("train", "validation", "test")},
        "region_counts": {region_name: sum(1 for r in position_manifest if r["position_region"] == region_name) for region_name in REGION_DEFS},
        "total_attempts": len(attempts_log), "total_discarded": sum(1 for a in attempts_log if not a["saved"]),
        "verify_dataset_result": verification, "dataset_root": str(root),
    }
    with open(root / "collection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"total_episodes_saved: {summary['total_episodes_saved']}/100")
    print(f"split_counts: {summary['split_counts']}")
    print(f"total_discarded: {summary['total_discarded']}")
    print(f"verify_dataset: state_has_nan_or_inf={verification.get('state_has_nan_or_inf')} "
          f"action_has_nan_or_inf={verification.get('action_has_nan_or_inf')}")
    print(f"Dataset root: {root}")


if __name__ == "__main__":
    main()
