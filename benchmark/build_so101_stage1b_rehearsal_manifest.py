"""Stage 1B rehearsal train-episode manifest builder (see this task's
chat report, "Rehearsal 학습 구성"). Selects 160 total training
episodes for Stage 1B fine-tuning:
  - 45 cube episodes from the ORIGINAL 160-episode train pool
    (`datasets/so101_bin_main_200`) -- deterministic distance-from-
    center TERTILE stratification (near/mid/far from the object's
    nominal center), so the 45 selected still span the interior/center
    region the original random sampling covered, not just one cluster.
  - 45 cube episodes from Stage 1A's own 50 new-train episodes
    (`datasets/so101_bin_stage1a_xy_70`, all boundary/corner/bridge
    regions) -- deterministic PROPORTIONAL region trimming: 1 episode
    dropped (highest episode_index, i.e. last-collected) from each of
    the 5 regions with >=6 episodes (x_min_corridor, corner_pp,
    corner_pn, corner_np, corner_nn), the 3 smaller bridge regions kept
    whole -- preserves every region's representation rather than
    dropping one region entirely.
  - 70 box episodes -- ALL of Stage 1B's own new box train episodes
    (`datasets/so101_bin_stage1b_box_100`, nothing held back further).

Writes `configs/so101_stage1b_train_episodes.json`, an index-based
allowlist manifest resolved against the COMBINED physical training
dataset built by benchmark.merge_so101_dataset_for_training_stage1b
(episode indices there: 0-199 = original 200, 200-269 = Stage 1A's own
70, 270-369 = Stage 1B's own 100 box episodes).

Nothing here writes/modifies any dataset -- purely reads existing
manifests/JSONL files and writes ONE new JSON manifest.

Run:
  .venv-vla/bin/python -m benchmark.build_so101_stage1b_rehearsal_manifest
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COLLECTION_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_main_200" / "collection_manifest.jsonl"
ORIGINAL_SPLIT_PATH = PROJECT_ROOT / "results" / "so101_pipeline_runs" / "all_20260720_114358" / "split.json"
STAGE1A_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1a_xy_70" / "stage1a_position_manifest.jsonl"
STAGE1A_TRAIN_ALLOWLIST_PATH = PROJECT_ROOT / "configs" / "so101_stage1a_train_episodes.json"
STAGE1B_BOX_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1b_box_100" / "stage1b_position_manifest.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "so101_stage1b_train_episodes.json"

# Index offsets in the combined physical dataset (see
# merge_so101_dataset_for_training_stage1b.py) -- original 200 kept at
# 0-199, Stage 1A's own 70 appended at 200-269, Stage 1B's own 100
# appended at 270-369.
STAGE1A_INDEX_OFFSET = 200
STAGE1B_INDEX_OFFSET = 270

STAGE1A_REGIONS_TO_TRIM = {"x_min_corridor", "corner_pp", "corner_pn", "corner_np", "corner_nn"}  # drop 1 each (highest episode_index)


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_original_cube_rehearsal(count: int) -> list:
    """45 (default `count`) deterministic distance-tertile-stratified
    episode indices from the ORIGINAL dataset's own 160-episode train
    split (never touches Stage 1A's episodes)."""
    manifest_rows = {r["episode_index"]: r for r in load_jsonl(ORIGINAL_COLLECTION_MANIFEST_PATH)}
    train_indices = json.loads(ORIGINAL_SPLIT_PATH.read_text())["train_episodes"]

    def distance(idx):
        pos = manifest_rows[idx]["sampled_object_position"]
        # sampled_object_position is stored as [x, y, z]; center distance
        # only needs x/y (z is the fixed drop height, never randomized).
        default_xy = (0.391, 0.0)  # DEFAULT_OBJECT_POSITION[:2] -- matches robot_sim.so101_pybullet_backend's own constant, not re-derived from a live import to keep this script torch/pybullet-free
        return ((pos[0] - default_xy[0]) ** 2 + (pos[1] - default_xy[1]) ** 2) ** 0.5

    sorted_by_distance = sorted(train_indices, key=distance)
    n = len(sorted_by_distance)
    tertile_size = n // 3
    near = sorted_by_distance[:tertile_size]
    mid = sorted_by_distance[tertile_size:2 * tertile_size]
    far = sorted_by_distance[2 * tertile_size:]

    per_tertile = count // 3
    remainder = count - per_tertile * 3
    counts = [per_tertile + (1 if i < remainder else 0) for i in range(3)]

    selected = []
    for tertile, take in zip((near, mid, far), counts):
        selected.extend(sorted(tertile)[:take])
    assert len(selected) == count, f"expected {count} selected, got {len(selected)}"
    return sorted(selected)


def select_stage1a_cube_rehearsal(count: int) -> list:
    """45 (default `count`) deterministic proportionally-trimmed episode
    indices from Stage 1A's own 50-episode new-train set (in the
    COMBINED dataset's index space, i.e. + STAGE1A_INDEX_OFFSET)."""
    rows = [r for r in load_jsonl(STAGE1A_MANIFEST_PATH) if r["split"] == "train"]
    by_region = {}
    for r in rows:
        by_region.setdefault(r["position_region"], []).append(r["episode_index"])

    selected = []
    for region, indices in by_region.items():
        indices = sorted(indices)
        if region in STAGE1A_REGIONS_TO_TRIM:
            indices = indices[:-1]  # drop the highest (last-collected) episode_index in this region
        selected.extend(indices)

    assert len(selected) == count, f"expected {count} selected after trimming, got {len(selected)} -- region counts: {{r: len(v) for r, v in by_region.items()}}"
    return sorted(idx + STAGE1A_INDEX_OFFSET for idx in selected)


def select_stage1b_box_train() -> list:
    rows = [r for r in load_jsonl(STAGE1B_BOX_MANIFEST_PATH) if r["split"] == "train"]
    return sorted(r["episode_index"] + STAGE1B_INDEX_OFFSET for r in rows)


def main() -> None:
    original_cube = select_original_cube_rehearsal(45)
    stage1a_cube = select_stage1a_cube_rehearsal(45)
    box_train = select_stage1b_box_train()

    assert len(original_cube) == 45
    assert len(stage1a_cube) == 45
    assert len(box_train) == 70
    assert len(set(original_cube) & set(stage1a_cube)) == 0
    assert len(set(original_cube) & set(box_train)) == 0
    assert len(set(stage1a_cube) & set(box_train)) == 0

    train_indices = sorted(original_cube + stage1a_cube + box_train)
    assert len(train_indices) == 160
    assert len(set(train_indices)) == 160

    original_split = json.loads(ORIGINAL_SPLIT_PATH.read_text())
    stage1a_allowlist = json.loads(STAGE1A_TRAIN_ALLOWLIST_PATH.read_text())
    box_rows = load_jsonl(STAGE1B_BOX_MANIFEST_PATH)

    excluded_existing_validation = sorted(original_split["validation_episodes"])
    # NOTE: Stage 1A's own train_episodes manifest already stores
    # ABSOLUTE indices within STAGE 1A's OWN 270-episode combined dataset
    # (0-199=original, 200-249=Stage1A new-train, 250-259=Stage1A
    # new-validation, 260-269=Stage1A new-test) -- these happen to line
    # up EXACTLY with this Stage 1B combined dataset's own 200-269 range
    # too (merge_so101_dataset_for_training_stage1b.py re-collects Stage
    # 1A's train/validation/test in that same order right after the
    # original 200), so NO additional STAGE1A_INDEX_OFFSET is added here
    # -- these values are already correct as-is. (STAGE1A_INDEX_OFFSET is
    # only needed for select_stage1a_cube_rehearsal(), which reads
    # STAGE1A-LOCAL indices 0-69 from the position manifest directly.)
    excluded_stage1a_validation = sorted(stage1a_allowlist["excluded_new_validation_indices"])
    excluded_stage1a_test = sorted(stage1a_allowlist["excluded_new_test_indices"])
    excluded_box_validation = sorted(r["episode_index"] + STAGE1B_INDEX_OFFSET for r in box_rows if r["split"] == "validation")
    excluded_box_test = sorted(r["episode_index"] + STAGE1B_INDEX_OFFSET for r in box_rows if r["split"] == "test")

    all_excluded = set(excluded_existing_validation) | set(excluded_stage1a_validation) | set(excluded_stage1a_test) | \
        set(excluded_box_validation) | set(excluded_box_test)
    assert len(set(train_indices) & all_excluded) == 0, "train/excluded overlap detected"

    manifest = {
        "combined_dataset": "datasets/so101_bin_stage1b_training_combined",
        "sources": {
            "original_dataset": "datasets/so101_bin_main_200 (indices 0-199)",
            "stage1a_new_episodes": "datasets/so101_bin_stage1a_xy_70 (indices 200-269 in combined)",
            "stage1b_new_box_episodes": "datasets/so101_bin_stage1b_box_100 (indices 270-369 in combined)",
        },
        "train_episode_indices": train_indices,
        "train_episode_count": len(train_indices),
        "train_composition": {
            "cube_rehearsal_original_range": original_cube,
            "cube_rehearsal_stage1a_boundary_corner": stage1a_cube,
            "box_train": box_train,
            "cube_total": len(original_cube) + len(stage1a_cube),
            "box_total": len(box_train),
        },
        "excluded_existing_validation_indices": excluded_existing_validation,
        "excluded_stage1a_validation_indices": excluded_stage1a_validation,
        "excluded_stage1a_test_indices": excluded_stage1a_test,
        "excluded_box_validation_indices": excluded_box_validation,
        "excluded_box_test_indices": excluded_box_test,
        "original_split_seed": original_split["split_seed"],
        "environment_seed_ranges": {
            "original": "0-199", "stage1a_train": "5000-5999", "stage1a_validation": "6000-6999", "stage1a_test": "7000-7999",
            "stage1b_train": "15000-16299", "stage1b_validation": "17000-17299", "stage1b_test": "18000-18299",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"cube rehearsal (original-range): {len(original_cube)}")
    print(f"cube rehearsal (Stage 1A boundary/corner): {len(stage1a_cube)}")
    print(f"box train: {len(box_train)}")
    print(f"total train episodes: {len(train_indices)}")
    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
