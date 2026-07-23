"""Stage 1C rehearsal train-episode manifest builder (see this task's
chat report, "Rehearsal dataset 구성"). Selects 180 total training
episodes for Stage 1C fine-tuning:
  - 55 cube episodes: 28 from the ORIGINAL 160-episode train pool
    (`datasets/so101_bin_main_200`, same distance-tertile-stratified
    method benchmark.build_so101_stage1b_rehearsal_manifest.py already
    used, just re-sized) + 27 from Stage 1A's own 50 new-train episodes
    (`datasets/so101_bin_stage1a_xy_70`, region-proportional trim, same
    method re-sized). ONLY these datasets' own TRAIN splits are read --
    validation/test rows are never selected (this task's own absolute
    principle "기존 validation/test episode 사용 금지").
  - 55 box episodes: a region-proportional trim of Stage 1B's own
    70-episode box TRAIN split (`datasets/so101_bin_stage1b_box_100`),
    keeping every one of its 13 regions represented (this task's own
    "center/interior뿐 아니라 edge/corner 사례 포함") -- deterministic,
    keeps the LOWEST episode_index(es) within each region (earliest-
    collected), same tie-break convention Stage 1B's own trim used.
  - 70 cylinder episodes -- ALL of Stage 1C's own new train episodes
    (`datasets/so101_bin_stage1c_cylinder_100`), nothing held back.

Writes `configs/so101_stage1c_train_episodes.json`, an index-based
allowlist manifest resolved against the COMBINED physical training
dataset built by benchmark.merge_so101_dataset_for_training_stage1c
(episode indices there: 0-199 = original 200, 200-269 = Stage 1A's own
70, 270-369 = Stage 1B's own 100 box, 370-469 = Stage 1C's own 100
cylinder).

Nothing here writes/modifies any dataset -- purely reads existing
manifests/JSONL files and writes ONE new JSON manifest.

Run:
  .venv-vla/bin/python -m benchmark.build_so101_stage1c_rehearsal_manifest
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COLLECTION_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_main_200" / "collection_manifest.jsonl"
ORIGINAL_SPLIT_PATH = PROJECT_ROOT / "results" / "so101_pipeline_runs" / "all_20260720_114358" / "split.json"
STAGE1A_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1a_xy_70" / "stage1a_position_manifest.jsonl"
STAGE1A_TRAIN_ALLOWLIST_PATH = PROJECT_ROOT / "configs" / "so101_stage1a_train_episodes.json"
STAGE1B_BOX_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1b_box_100" / "stage1b_position_manifest.jsonl"
STAGE1C_CYLINDER_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1c_cylinder_100" / "stage1c_position_manifest.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "so101_stage1c_train_episodes.json"

# Index offsets in the combined physical dataset (see
# merge_so101_dataset_for_training_stage1c.py) -- original 200 kept at
# 0-199, Stage 1A's own 70 appended at 200-269, Stage 1B's own 100 box
# appended at 270-369, Stage 1C's own 100 cylinder appended at 370-469.
STAGE1A_INDEX_OFFSET = 200
STAGE1B_INDEX_OFFSET = 270
STAGE1C_INDEX_OFFSET = 370

# Deterministic region-proportional trim: 50 Stage-1A new-train episodes -> 27
# (drop 23, keep-first-by-ascending-episode_index within each region, every
# region stays represented).
STAGE1A_REGION_KEEP_COUNTS = {
    "x_min_corridor": 8, "corner_pp": 3, "corner_pn": 4, "corner_np": 3, "corner_nn": 3,
    "bridge_plus_x": 2, "bridge_minus_y": 2, "bridge_plus_y": 2,
}
assert sum(STAGE1A_REGION_KEEP_COUNTS.values()) == 27

# Deterministic region-proportional trim: 70 Stage-1B box train episodes -> 55
# (drop 15, keep-first-by-ascending-episode_index within each region, every
# one of the 13 regions stays represented -- this task's own "center/
# interior뿐 아니라 edge/corner 사례 포함").
STAGE1B_BOX_REGION_KEEP_COUNTS = {
    "center": 7, "existing_x_min": 5, "existing_x_max": 5, "existing_y_min": 5, "existing_y_max": 5,
    "x_min_corridor": 5, "corner_pp": 4, "corner_pn": 5, "corner_np": 4, "corner_nn": 4,
    "bridge_plus_x": 2, "bridge_minus_y": 2, "bridge_plus_y": 2,
}
assert sum(STAGE1B_BOX_REGION_KEEP_COUNTS.values()) == 55


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_original_cube_rehearsal(count: int) -> list:
    """Same distance-tertile-stratified method as
    benchmark.build_so101_stage1b_rehearsal_manifest.select_original_cube_rehearsal(),
    re-sized to `count` (28 here, vs. Stage 1B's own 45)."""
    manifest_rows = {r["episode_index"]: r for r in load_jsonl(ORIGINAL_COLLECTION_MANIFEST_PATH)}
    train_indices = json.loads(ORIGINAL_SPLIT_PATH.read_text())["train_episodes"]

    def distance(idx):
        pos = manifest_rows[idx]["sampled_object_position"]
        default_xy = (0.391, 0.0)
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


def select_stage1a_cube_rehearsal() -> list:
    """Region-proportional trim of Stage 1A's own 50-episode new-train set
    down to 27 (STAGE1A_REGION_KEEP_COUNTS), in the COMBINED dataset's
    index space (+ STAGE1A_INDEX_OFFSET)."""
    rows = [r for r in load_jsonl(STAGE1A_MANIFEST_PATH) if r["split"] == "train"]
    by_region = {}
    for r in rows:
        by_region.setdefault(r["position_region"], []).append(r["episode_index"])

    selected = []
    for region, indices in by_region.items():
        indices = sorted(indices)
        keep = STAGE1A_REGION_KEEP_COUNTS[region]
        selected.extend(indices[:keep])

    assert len(selected) == 27, f"expected 27 selected after trimming, got {len(selected)}"
    return sorted(idx + STAGE1A_INDEX_OFFSET for idx in selected)


def select_stage1b_box_rehearsal() -> list:
    """Region-proportional trim of Stage 1B's own 70-episode box TRAIN set
    down to 55 (STAGE1B_BOX_REGION_KEEP_COUNTS), in the COMBINED dataset's
    index space (+ STAGE1B_INDEX_OFFSET). Validation/test rows are never
    read here -- filtered to split=="train" first."""
    rows = [r for r in load_jsonl(STAGE1B_BOX_MANIFEST_PATH) if r["split"] == "train"]
    by_region = {}
    for r in rows:
        by_region.setdefault(r["position_region"], []).append(r["episode_index"])

    selected = []
    for region, indices in by_region.items():
        indices = sorted(indices)
        keep = STAGE1B_BOX_REGION_KEEP_COUNTS[region]
        selected.extend(indices[:keep])

    assert len(selected) == 55, f"expected 55 selected after trimming, got {len(selected)} -- region counts: { {r: len(v) for r, v in by_region.items()} }"
    return sorted(idx + STAGE1B_INDEX_OFFSET for idx in selected)


def select_stage1c_cylinder_train() -> list:
    rows = [r for r in load_jsonl(STAGE1C_CYLINDER_MANIFEST_PATH) if r["split"] == "train"]
    return sorted(r["episode_id"] + STAGE1C_INDEX_OFFSET for r in rows)


def main() -> None:
    original_cube = select_original_cube_rehearsal(28)
    stage1a_cube = select_stage1a_cube_rehearsal()
    box_rehearsal = select_stage1b_box_rehearsal()
    cylinder_train = select_stage1c_cylinder_train()

    assert len(original_cube) == 28
    assert len(stage1a_cube) == 27
    assert len(box_rehearsal) == 55
    assert len(cylinder_train) == 70
    assert len(set(original_cube) & set(stage1a_cube)) == 0
    assert len(set(original_cube) & set(box_rehearsal)) == 0
    assert len(set(original_cube) & set(cylinder_train)) == 0
    assert len(set(stage1a_cube) & set(box_rehearsal)) == 0
    assert len(set(stage1a_cube) & set(cylinder_train)) == 0
    assert len(set(box_rehearsal) & set(cylinder_train)) == 0

    train_indices = sorted(original_cube + stage1a_cube + box_rehearsal + cylinder_train)
    assert len(train_indices) == 180
    assert len(set(train_indices)) == 180

    original_split = json.loads(ORIGINAL_SPLIT_PATH.read_text())
    stage1a_allowlist = json.loads(STAGE1A_TRAIN_ALLOWLIST_PATH.read_text())
    box_rows = load_jsonl(STAGE1B_BOX_MANIFEST_PATH)
    cylinder_rows = load_jsonl(STAGE1C_CYLINDER_MANIFEST_PATH)

    excluded_existing_validation = sorted(original_split["validation_episodes"])
    # Same reasoning as Stage 1B's own rehearsal manifest builder: Stage
    # 1A's own train_episodes manifest already stores ABSOLUTE indices in
    # the 200-269 range (this combined dataset's Stage-1A slot too), so no
    # further offset is added for these two fields.
    excluded_stage1a_validation = sorted(stage1a_allowlist["excluded_new_validation_indices"])
    excluded_stage1a_test = sorted(stage1a_allowlist["excluded_new_test_indices"])
    excluded_box_validation = sorted(r["episode_index"] + STAGE1B_INDEX_OFFSET for r in box_rows if r["split"] == "validation")
    excluded_box_test = sorted(r["episode_index"] + STAGE1B_INDEX_OFFSET for r in box_rows if r["split"] == "test")
    excluded_cylinder_validation = sorted(r["episode_id"] + STAGE1C_INDEX_OFFSET for r in cylinder_rows if r["split"] == "validation")
    excluded_cylinder_test = sorted(r["episode_id"] + STAGE1C_INDEX_OFFSET for r in cylinder_rows if r["split"] == "test")

    all_excluded = (
        set(excluded_existing_validation) | set(excluded_stage1a_validation) | set(excluded_stage1a_test)
        | set(excluded_box_validation) | set(excluded_box_test) | set(excluded_cylinder_validation) | set(excluded_cylinder_test)
    )
    assert len(set(train_indices) & all_excluded) == 0, "train/excluded overlap detected"

    manifest = {
        "combined_dataset": "datasets/so101_bin_stage1c_training_combined",
        "sources": {
            "original_dataset": "datasets/so101_bin_main_200 (indices 0-199)",
            "stage1a_new_episodes": "datasets/so101_bin_stage1a_xy_70 (indices 200-269 in combined)",
            "stage1b_new_box_episodes": "datasets/so101_bin_stage1b_box_100 (indices 270-369 in combined)",
            "stage1c_new_cylinder_episodes": "datasets/so101_bin_stage1c_cylinder_100 (indices 370-469 in combined)",
        },
        "train_episode_indices": train_indices,
        "train_episode_count": len(train_indices),
        "train_composition": {
            "cube_rehearsal_original_range": original_cube,
            "cube_rehearsal_stage1a_boundary_corner": stage1a_cube,
            "box_rehearsal": box_rehearsal,
            "cylinder_train": cylinder_train,
            "cube_total": len(original_cube) + len(stage1a_cube),
            "box_total": len(box_rehearsal),
            "cylinder_total": len(cylinder_train),
        },
        "excluded_existing_validation_indices": excluded_existing_validation,
        "excluded_stage1a_validation_indices": excluded_stage1a_validation,
        "excluded_stage1a_test_indices": excluded_stage1a_test,
        "excluded_box_validation_indices": excluded_box_validation,
        "excluded_box_test_indices": excluded_box_test,
        "excluded_cylinder_validation_indices": excluded_cylinder_validation,
        "excluded_cylinder_test_indices": excluded_cylinder_test,
        "original_split_seed": original_split["split_seed"],
        "environment_seed_ranges": {
            "original": "0-199", "stage1a_train": "5000-5999", "stage1a_validation": "6000-6999", "stage1a_test": "7000-7999",
            "stage1b_train": "15000-16299", "stage1b_validation": "17000-17299", "stage1b_test": "18000-18299",
            "stage1c_train": "20000-20499", "stage1c_validation": "21000-21099", "stage1c_test": "22000-22099",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"cube rehearsal (original-range): {len(original_cube)}")
    print(f"cube rehearsal (Stage 1A boundary/corner): {len(stage1a_cube)}")
    print(f"box rehearsal: {len(box_rehearsal)}")
    print(f"cylinder train: {len(cylinder_train)}")
    print(f"total train episodes: {len(train_indices)}")
    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
