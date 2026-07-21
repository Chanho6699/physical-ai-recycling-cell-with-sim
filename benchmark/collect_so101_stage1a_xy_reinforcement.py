"""Stage 1A object-XY targeted reinforcement dataset collector (see this
task's chat report, "Stage 1A object XY 표적 보강 데이터셋 설계"/"...구현").

Collects 70 NEW episodes (50 train / 10 expansion-validation / 10
held-out expansion-test) into a BRAND NEW dataset
(`datasets/so101_bin_stage1a_xy_70`), targeting the weak regions
identified by the prior XY-extrapolation evaluation (corners and the
-X boundary, radius band 0.015m-0.01875m) -- NEVER touches
`datasets/so101_bin_main_200` (only imports read-only constants from
sibling modules, opens no file under that dataset's own directory).

Reuses (does NOT reimplement): benchmark.collect_so101_bin_dataset's
own `collect_episode()` (same scripted-expert call, same discard-on-
failure policy, same contract check) and
benchmark.collect_so101_episode's own `write_phase_id_mapping()`/
`verify_dataset()`. Each region gets its own (x_range, y_range) window
passed straight through to `collect_episode()`'s own existing
x_range/y_range parameters -- sample_object_position(seed, x_range,
y_range) then draws a fresh CONTINUOUS point inside that window per
seed, so no two episodes in a region share the exact same coordinate
and no single coordinate is over-represented (this task's own "특정 한
위치에만 데이터를 몰지 말고... 연속 좌표를 seed 기반으로 샘플링").

Region seed blocks are fully disjoint by construction (train=5000s,
validation=6000s, test=7000s; within each, one hundreds-block per
region) and never overlap the original dataset's own collection seeds
(0-199) or any policy-noise-seed base used elsewhere in this project
(100000/200000/300000/400000).

`collect_episode()` discards (does not save) any episode where the
scripted expert itself fails place_success -- this script does NOT
retry a failed seed; instead each region is given a seed CANDIDATE list
noticeably larger than its target count, and collection stops as soon
as the target number of SAVED episodes is reached for that region
(early-stop). If a region's candidate list is exhausted before its
target is reached, this raises immediately rather than silently
under-collecting (loud failure, not a silent shortfall).

Run:
  .venv-vla/bin/python -m benchmark.collect_so101_stage1a_xy_reinforcement
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
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, DEFAULT_SCENE_CONFIG
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = "datasets/so101_bin_stage1a_xy_70"
REPO_ID = "local/so101_bin_stage1a_xy_70"
FPS = 10  # matches datasets/so101_bin_main_200's own DEFAULT_FPS

OLD_RADIUS_M = 0.015
NEW_RADIUS_M = 0.01875

# (name, x_range, y_range, is_corner) -- x_range/y_range are ABSOLUTE
# OFFSET tuples relative to DEFAULT_OBJECT_POSITION, passed straight to
# collect_episode()'s own x_range/y_range params (same convention as
# FIXED_BIN_OBJECT_X_RANGE/Y_RANGE). Every window lives strictly inside
# the already-evaluated band [0.015, 0.01875] -- nothing here explores
# beyond what the prior XY-extrapolation eval actually tested.
REGION_DEFS = {
    "x_min_corridor": {"x_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "y_range": (-OLD_RADIUS_M, OLD_RADIUS_M), "is_corner": False},
    "corner_pp": {"x_range": (OLD_RADIUS_M, NEW_RADIUS_M), "y_range": (OLD_RADIUS_M, NEW_RADIUS_M), "is_corner": True},
    "corner_pn": {"x_range": (OLD_RADIUS_M, NEW_RADIUS_M), "y_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "is_corner": True},
    "corner_np": {"x_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "y_range": (OLD_RADIUS_M, NEW_RADIUS_M), "is_corner": True},
    "corner_nn": {"x_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "y_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "is_corner": True},
    "bridge_plus_x": {"x_range": (OLD_RADIUS_M, NEW_RADIUS_M), "y_range": (-0.005, 0.005), "is_corner": False},
    "bridge_minus_y": {"x_range": (-0.005, 0.005), "y_range": (-NEW_RADIUS_M, -OLD_RADIUS_M), "is_corner": False},
    "bridge_plus_y": {"x_range": (-0.005, 0.005), "y_range": (OLD_RADIUS_M, NEW_RADIUS_M), "is_corner": False},
}

# {split: {region_name: (target_count, seed_candidates)}} -- disjoint
# seed blocks (train=5xxx, validation=6xxx, test=7xxx; one hundreds-
# sub-block per region) with a generous buffer over target so a
# discard doesn't stall collection. Sums: train=50, validation=10, test=10.
COLLECTION_PLAN = {
    "train": {
        "x_min_corridor": (15, list(range(5000, 5030))),
        "corner_pp": (6, list(range(5100, 5120))),
        "corner_pn": (7, list(range(5200, 5220))),
        "corner_np": (6, list(range(5300, 5320))),
        "corner_nn": (6, list(range(5400, 5420))),
        "bridge_plus_x": (4, list(range(5500, 5510))),
        "bridge_minus_y": (3, list(range(5600, 5610))),
        "bridge_plus_y": (3, list(range(5700, 5710))),
    },
    "validation": {
        "x_min_corridor": (3, list(range(6000, 6010))),
        "corner_pp": (1, list(range(6100, 6105))),
        "corner_pn": (2, list(range(6200, 6206))),
        "corner_np": (1, list(range(6300, 6305))),
        "corner_nn": (1, list(range(6400, 6405))),
        "bridge_plus_x": (1, list(range(6500, 6504))),
        "bridge_minus_y": (1, list(range(6600, 6604))),
        "bridge_plus_y": (0, []),
    },
    "test": {
        "x_min_corridor": (3, list(range(7000, 7010))),
        "corner_pp": (1, list(range(7100, 7105))),
        "corner_pn": (2, list(range(7200, 7206))),
        "corner_np": (1, list(range(7300, 7305))),
        "corner_nn": (1, list(range(7400, 7405))),
        "bridge_plus_x": (1, list(range(7600, 7604))),
        "bridge_minus_y": (0, []),
        "bridge_plus_y": (1, list(range(7500, 7504))),
    },
}


def main() -> None:
    root = resolve(DATASET_ROOT)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")

    for split, regions in COLLECTION_PLAN.items():
        target_sum = sum(target for target, _ in regions.values())
        expected = {"train": 50, "validation": 10, "test": 10}[split]
        assert target_sum == expected, f"{split} region targets sum to {target_sum}, expected {expected}"

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, features=SO101_FEATURES, root=str(root),
        robot_type=SO101_ROBOT_TYPE, use_videos=False,
    )

    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config_override = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY}

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
                            "episode_index": outcome["episode_index"],
                            "environment_seed": seed,
                            "split": split,
                            "position_region": region_name,
                            "object_x": sampled[0], "object_y": sampled[1],
                            "x_offset": x_offset, "y_offset": y_offset,
                            "distance_from_center": (x_offset ** 2 + y_offset ** 2) ** 0.5,
                            "is_corner_region": region["is_corner"],
                            "is_negative_x_region": x_offset < 0,
                            "phase_id": PHASE_ID_BY_NAME["release"],
                            "failure_reason": None,
                        })

                if saved_count < target:
                    raise RuntimeError(
                        f"Stage 1A collection shortfall: split={split!r} region={region_name!r} "
                        f"saved only {saved_count}/{target} from {len(seed_candidates)} seed candidates -- "
                        f"widen the seed candidate block for this region and rerun."
                    )
    finally:
        dataset.finalize()

    if episode_index_counter["count"] > 0:
        write_phase_id_mapping(root)

    manifest_path = root / "stage1a_position_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in position_manifest:
            f.write(json.dumps(record, default=str) + "\n")

    attempts_log_path = root / "stage1a_collection_attempts_log.jsonl"
    with open(attempts_log_path, "w", encoding="utf-8") as f:
        for record in attempts_log:
            f.write(json.dumps(record, default=str) + "\n")

    verification = verify_dataset(root)

    summary = {
        "dataset_name": root.name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "robot_type": SO101_ROBOT_TYPE,
        "use_bin": True,
        "total_episodes_saved": episode_index_counter["count"],
        "split_counts": {
            split: sum(1 for r in position_manifest if r["split"] == split) for split in ("train", "validation", "test")
        },
        "region_counts": {
            region_name: sum(1 for r in position_manifest if r["position_region"] == region_name) for region_name in REGION_DEFS
        },
        "total_attempts": len(attempts_log),
        "total_discarded": sum(1 for a in attempts_log if not a["saved"]),
        "discard_failure_reason_counts": {
            reason: sum(1 for a in attempts_log if not a["saved"] and str(a["failure_reason"]) == reason)
            for reason in {str(a["failure_reason"]) for a in attempts_log if not a["saved"]}
        },
        "old_radius_m": OLD_RADIUS_M, "new_radius_m": NEW_RADIUS_M,
        "fixed_bin_center_xy": fixed_bin_center_xy,
        "verify_dataset_result": verification,
        "dataset_root": str(root),
    }
    with open(root / "collection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"total_episodes_saved: {summary['total_episodes_saved']}/70")
    print(f"split_counts: {summary['split_counts']}")
    print(f"total_discarded: {summary['total_discarded']}")
    print(f"verify_dataset: state_has_nan_or_inf={verification.get('state_has_nan_or_inf')} "
          f"action_has_nan_or_inf={verification.get('action_has_nan_or_inf')}")
    print(f"Dataset root: {root}")


if __name__ == "__main__":
    main()
