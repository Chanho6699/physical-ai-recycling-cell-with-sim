"""Stage 1A combined-training-dataset builder (see this task's chat
report, "Stage 1A object XY 표적 보강 데이터셋 설계"/"...구현"). Produces
a NEW physical dataset (`datasets/so101_bin_stage1a_combined_270`)
containing the original 200 episodes PLUS the 70 new Stage 1A episodes
(270 total), and a train-episode-index allowlist manifest
(`configs/so101_stage1a_train_episodes.json`) that selects exactly the
160 original-train + 50 new-train episodes (210 total) actually used
for fine-tuning.

`datasets/so101_bin_main_200` is NEVER opened in write mode -- only
`shutil.copytree()`'d (read-only source access) into the new combined
directory. This script re-verifies the original's file hashes are
unchanged after the copy (see `hash_dataset_files()`/`main()`'s own
before/after comparison).

The 70 new episodes are NOT copied file-by-file from
`datasets/so101_bin_stage1a_xy_70` -- they are RE-COLLECTED (same
scripted expert, same deterministic seeds, same per-region x_range/
y_range windows, reusing benchmark.collect_so101_stage1a_xy_reinforcement's
own COLLECTION_PLAN/REGION_DEFS so nothing is redefined) directly onto
the combined dataset via LeRobotDataset.resume() -- the library's own
supported "append more episodes to an existing dataset" API -- so the
combined dataset's episode-index/frame-index bookkeeping is exactly
what LeRobotDataset itself expects, rather than hand-editing parquet/
metadata files. Determinism (same seed -> same simulated trajectory)
means the re-collected content is expected to match
`datasets/so101_bin_stage1a_xy_70`'s own 70 episodes; this script's own
integrity checks (see validate_so101_stage1a_dataset.py) confirm the
counts/regions/seeds line up, not the raw bytes.

Run:
  .venv-vla/bin/python -m benchmark.merge_so101_dataset_for_training
"""

import hashlib
import json
import shutil
from pathlib import Path

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY,
)
from benchmark.collect_so101_bin_dataset import DEFAULT_INSTRUCTION, collect_episode, resolve
from benchmark.collect_so101_episode import verify_dataset, write_phase_id_mapping
from benchmark.collect_so101_stage1a_xy_reinforcement import COLLECTION_PLAN, REGION_DEFS
from benchmark.so101_scripted_expert import PHASE_ID_BY_NAME
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, DEFAULT_SCENE_CONFIG
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_ROOT = "datasets/so101_bin_main_200"
COMBINED_DATASET_ROOT = "datasets/so101_bin_stage1a_combined_270"
COMBINED_REPO_ID = "local/so101_bin_stage1a_combined_270"
ORIGINAL_SPLIT_PATH = "results/so101_pipeline_runs/all_20260720_114358/split.json"
TRAIN_ALLOWLIST_PATH = "configs/so101_stage1a_train_episodes.json"


def hash_dataset_files(root: Path) -> dict:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    original_root = resolve(ORIGINAL_DATASET_ROOT)
    combined_root = resolve(COMBINED_DATASET_ROOT)

    if combined_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {combined_root}")

    original_hashes_before = hash_dataset_files(original_root)
    print(f"Original dataset: {len(original_hashes_before)} files hashed before copy.")

    shutil.copytree(original_root, combined_root)
    print(f"Copied {original_root} -> {combined_root}")

    original_hashes_after_copy = hash_dataset_files(original_root)
    if original_hashes_before != original_hashes_after_copy:
        raise RuntimeError("Original dataset files changed during copytree() -- aborting, investigate before proceeding.")

    dataset = LeRobotDataset.resume(repo_id=COMBINED_REPO_ID, root=str(combined_root))
    starting_episode_count = dataset.meta.total_episodes
    print(f"Resumed combined dataset at episode_index={starting_episode_count} (expected 200).")
    if starting_episode_count != 200:
        raise RuntimeError(f"Expected combined dataset to resume at 200 episodes, got {starting_episode_count}")

    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config_override = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY}

    episode_index_counter = {"count": starting_episode_count}
    position_manifest = []

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
                    print(f"[combined:{split}/{region_name}] seed={seed} saved={outcome['saved']} "
                          f"({saved_count + int(outcome['saved'])}/{target})")
                    if outcome["saved"]:
                        saved_count += 1
                        sampled = outcome["sampled_object_position"]
                        x_offset = sampled[0] - DEFAULT_OBJECT_POSITION[0]
                        y_offset = sampled[1] - DEFAULT_OBJECT_POSITION[1]
                        position_manifest.append({
                            "episode_index": outcome["episode_index"], "environment_seed": seed, "split": split,
                            "position_region": region_name, "object_x": sampled[0], "object_y": sampled[1],
                            "x_offset": x_offset, "y_offset": y_offset,
                            "distance_from_center": (x_offset ** 2 + y_offset ** 2) ** 0.5,
                            "is_corner_region": region["is_corner"], "is_negative_x_region": x_offset < 0,
                            "phase_id": PHASE_ID_BY_NAME["release"], "failure_reason": None,
                        })
                if saved_count < target:
                    raise RuntimeError(
                        f"Combined-dataset re-collection shortfall: split={split!r} region={region_name!r} "
                        f"saved only {saved_count}/{target}."
                    )
    finally:
        dataset.finalize()

    write_phase_id_mapping(combined_root)

    manifest_path = combined_root / "stage1a_position_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in position_manifest:
            f.write(json.dumps(record, default=str) + "\n")

    final_episode_count = episode_index_counter["count"]
    print(f"Combined dataset final episode count: {final_episode_count} (expected 270).")
    if final_episode_count != 270:
        raise RuntimeError(f"Expected 270 total episodes in combined dataset, got {final_episode_count}")

    original_hashes_final = hash_dataset_files(original_root)
    if original_hashes_before != original_hashes_final:
        raise RuntimeError("Original dataset files changed during combined-dataset build -- aborting.")
    print("Original dataset files confirmed byte-identical before/after (sha256).")

    verification = verify_dataset(combined_root)
    print(f"verify_dataset(combined): state_has_nan_or_inf={verification.get('state_has_nan_or_inf')} "
          f"action_has_nan_or_inf={verification.get('action_has_nan_or_inf')}")

    # --- Train-episode-index allowlist manifest ---
    original_split = json.loads(resolve(ORIGINAL_SPLIT_PATH).read_text())
    original_train_indices = original_split["train_episodes"]
    original_validation_indices = original_split["validation_episodes"]

    new_train_indices = sorted(r["episode_index"] for r in position_manifest if r["split"] == "train")
    new_validation_indices = sorted(r["episode_index"] for r in position_manifest if r["split"] == "validation")
    new_test_indices = sorted(r["episode_index"] for r in position_manifest if r["split"] == "test")

    assert len(original_train_indices) == 160
    assert len(original_validation_indices) == 40
    assert len(new_train_indices) == 50
    assert len(new_validation_indices) == 10
    assert len(new_test_indices) == 10

    train_allowlist = sorted(original_train_indices + new_train_indices)
    assert len(train_allowlist) == 210
    assert len(set(train_allowlist)) == 210

    manifest = {
        "combined_dataset": COMBINED_DATASET_ROOT,
        "original_dataset": ORIGINAL_DATASET_ROOT,
        "new_dataset_standalone": "datasets/so101_bin_stage1a_xy_70",
        "train_episode_indices": train_allowlist,
        "train_episode_count": len(train_allowlist),
        "excluded_existing_validation_indices": sorted(original_validation_indices),
        "excluded_new_validation_indices": new_validation_indices,
        "excluded_new_test_indices": new_test_indices,
        "original_split_seed": original_split["split_seed"],
        "original_environment_seed_range": "0-199",
        "new_environment_seed_ranges": {
            "train": "5000-5999 (per-region sub-blocks, see benchmark/collect_so101_stage1a_xy_reinforcement.py COLLECTION_PLAN)",
            "validation": "6000-6999",
            "test": "7000-7999",
        },
    }
    allowlist_path = resolve(TRAIN_ALLOWLIST_PATH)
    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(allowlist_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Combined dataset: {combined_root}")
    print(f"Train allowlist ({len(train_allowlist)} episodes): {allowlist_path}")


if __name__ == "__main__":
    main()
