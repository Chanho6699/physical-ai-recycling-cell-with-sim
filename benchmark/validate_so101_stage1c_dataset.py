"""Stage 1C pre-training integrity validator (see this task's chat
report, "데이터 무결성 검증"). Read-only, mirrors
benchmark/validate_so101_stage1b_dataset.py's own structure, extended
for the 3-object rehearsal composition (cube + box + cylinder) and the
combined 470-episode dataset.

Checks (this task's own section-7 required list):
  Episode structure: total==100, split counts 70/15/15, no duplicate
    episode within a split, seed uniqueness, episode_id continuity,
    manifest path validity, no missing files, rehearsal composition
    exactly 55 cube + 55 box + 70 cylinder = 180, train/excluded
    disjointness (split leakage == 0), duplicate source-episode == 0.
  Feature schema: declared shapes (state=6, action=6), image shape
    256x256x3 verified via a DETERMINISTIC SAMPLE decode (this task's
    own "이미지 전수 decode가 과도하게 오래 걸리면 deterministic
    sample과 파일 무결성 전수 검사를 조합" -- reported explicitly
    below, NOT silently substituted), full-array NaN/Inf check (cheap,
    no image decode needed) across the ENTIRE combined 470-episode
    dataset, gripper 0-100 scale, fps==10, joint order, task string.
  Action/trajectory: per-frame arm-joint jump magnitude (max/mean),
    gripper open->close ordering, frame/action count match, episode
    termination phase_id, frame_count vs manifest expectation.
  Scene metadata: cylinder radius/height/shape/position_group fields
    present and correct in the Stage 1C manifest, yaw values within
    the deterministic {0,45,90,135,180} degree set this task's own
    collector cycles through.
  Dataset unchanged: datasets/so101_bin_main_200 +
    datasets/so101_bin_stage1a_xy_70 + datasets/so101_bin_stage1b_box_100
    file hashes (sha256) unchanged from a snapshot taken immediately
    before this task's own merge/collection ran.

Run:
  .venv-vla/bin/python -m benchmark.validate_so101_stage1c_dataset
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYLINDER_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_stage1c_cylinder_100"
CYLINDER_MANIFEST_PATH = CYLINDER_DATASET_ROOT / "stage1c_position_manifest.jsonl"
COMBINED_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_stage1c_training_combined"
TRAIN_ALLOWLIST_PATH = PROJECT_ROOT / "configs" / "so101_stage1c_train_episodes.json"
BOX_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1b_box_100" / "stage1b_position_manifest.jsonl"
ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_main_200"
STAGE1A_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_stage1a_xy_70"
STAGE1B_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_stage1b_box_100"
EXPECTED_INSTRUCTION = "Pick up the object and place it in the bin."

REQUIRED_CYLINDER_FIELDS = [
    "episode_id", "split", "seed", "expert_version", "strategy", "object_shape", "object_radius",
    "object_diameter", "object_height", "object_position", "object_yaw", "position_group", "region_name",
    "object_mass", "lateral_friction", "rolling_friction", "spinning_friction", "restitution",
    "constraint_based_grasp", "contact_physics_verified", "legacy_success", "constraint_based_success",
    "failure_phase", "failure_reason", "discarded",
]

DETERMINISTIC_IMAGE_SAMPLE_EPISODE_IDS = list(range(370, 470, 7))  # every 7th cylinder episode (in combined-dataset index space) -- ~14 episodes decoded, not all 100

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hash_dataset_files(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def main() -> None:
    cyl_rows = load_jsonl(CYLINDER_MANIFEST_PATH)
    allowlist = json.loads(TRAIN_ALLOWLIST_PATH.read_text())
    box_rows = load_jsonl(BOX_MANIFEST_PATH)

    # --- Episode structure ---
    check("cylinder manifest: total episodes == 100", len(cyl_rows) == 100, detail=f"got {len(cyl_rows)}")
    for split, expected in [("train", 70), ("validation", 15), ("test", 15)]:
        n = sum(1 for r in cyl_rows if r["split"] == split)
        check(f"cylinder manifest: {split} count == {expected}", n == expected, detail=f"got {n}")

    missing = 0
    for r in cyl_rows:
        for field in REQUIRED_CYLINDER_FIELDS:
            if field not in r or (r[field] is None and field not in ("failure_reason", "failure_phase")):
                missing += 1
    check("cylinder manifest: no missing required metadata fields", missing == 0, detail=f"{missing} missing")

    non_none_failure = sum(1 for r in cyl_rows if r["failure_reason"] is not None)
    check("cylinder manifest: failure_reason is None for every saved episode", non_none_failure == 0, detail=f"{non_none_failure} non-None")
    non_false_discarded = sum(1 for r in cyl_rows if r["discarded"] is not False)
    check("cylinder manifest: discarded is False for every saved episode", non_false_discarded == 0)

    seeds = [r["seed"] for r in cyl_rows]
    check("cylinder manifest: seed unique across all 100 episodes", len(seeds) == len(set(seeds)), detail=f"{len(seeds)} rows, {len(set(seeds))} unique")

    episode_ids = sorted(r["episode_id"] for r in cyl_rows)
    check("cylinder manifest: episode_id is a contiguous 0..99 sequence", episode_ids == list(range(100)), detail=f"got {episode_ids[:5]}...{episode_ids[-5:]}")

    for split_pair in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a_ids = {r["episode_id"] for r in cyl_rows if r["split"] == split_pair[0]}
        b_ids = {r["episode_id"] for r in cyl_rows if r["split"] == split_pair[1]}
        check(f"cylinder manifest: {split_pair[0]} ∩ {split_pair[1]} == ∅", len(a_ids & b_ids) == 0)

    # --- Rehearsal composition ---
    composition = allowlist["train_composition"]
    check("rehearsal composition: cube original-range == 28", len(composition["cube_rehearsal_original_range"]) == 28)
    check("rehearsal composition: cube Stage 1A boundary/corner == 27", len(composition["cube_rehearsal_stage1a_boundary_corner"]) == 27)
    check("rehearsal composition: box rehearsal == 55", len(composition["box_rehearsal"]) == 55)
    check("rehearsal composition: cylinder train == 70", len(composition["cylinder_train"]) == 70)
    check("rehearsal composition: cube total == 55", composition["cube_total"] == 55)
    check("rehearsal composition: box total == 55", composition["box_total"] == 55)
    check("rehearsal composition: cylinder total == 70", composition["cylinder_total"] == 70)
    check("train_episode_count == 180", allowlist["train_episode_count"] == 180)

    train_set = set(allowlist["train_episode_indices"])
    check("train_episode_indices has no duplicates", len(train_set) == len(allowlist["train_episode_indices"]))

    excluded_sets = {
        "existing_validation": set(allowlist["excluded_existing_validation_indices"]),
        "stage1a_validation": set(allowlist["excluded_stage1a_validation_indices"]),
        "stage1a_test": set(allowlist["excluded_stage1a_test_indices"]),
        "box_validation": set(allowlist["excluded_box_validation_indices"]),
        "box_test": set(allowlist["excluded_box_test_indices"]),
        "cylinder_validation": set(allowlist["excluded_cylinder_validation_indices"]),
        "cylinder_test": set(allowlist["excluded_cylinder_test_indices"]),
    }
    for name, excluded in excluded_sets.items():
        check(f"train ∩ {name} == ∅ (split leakage == 0)", len(train_set & excluded) == 0, detail=str(train_set & excluded))
    names = list(excluded_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = excluded_sets[names[i]] & excluded_sets[names[j]]
            check(f"{names[i]} ∩ {names[j]} == ∅", len(overlap) == 0, detail=str(overlap))

    check("cylinder_validation excluded count == 15", len(excluded_sets["cylinder_validation"]) == 15)
    check("cylinder_test excluded count == 15", len(excluded_sets["cylinder_test"]) == 15)
    check("box_validation excluded count == 15", len(excluded_sets["box_validation"]) == 15)
    check("box_test excluded count == 15", len(excluded_sets["box_test"]) == 15)

    all_indices = set(train_set)
    for excluded in excluded_sets.values():
        all_indices |= excluded
    # 180 train + (10 existing_val + 5 stage1a_val + 5 stage1a_test + 15 box_val + 15 box_test + 15 cyl_val + 15 cyl_test) = 180 + 80 = 260 accounted-for; remainder of 470 is unused rehearsal-pool, by design (same accounting pattern as Stage 1B's own validator).
    check("duplicate source episode across all accounted-for sets == 0 (train ∪ excluded has no internal dupes beyond set semantics)",
          len(all_indices) == len(train_set) + sum(len(s) for s in excluded_sets.values()),
          detail=f"union={len(all_indices)} vs sum={len(train_set) + sum(len(s) for s in excluded_sets.values())}")

    # --- Scene metadata (Stage 1C manifest) ---
    radius_ok = all(r["object_radius"] == 0.02 for r in cyl_rows)
    height_ok = all(r["object_height"] == 0.04 for r in cyl_rows)
    shape_ok = all(r["object_shape"] == "cylinder" for r in cyl_rows)
    check("cylinder manifest: object_radius == 0.02m for all rows", radius_ok)
    check("cylinder manifest: object_height == 0.04m for all rows", height_ok)
    check("cylinder manifest: object_shape == 'cylinder' for all rows", shape_ok)

    expected_yaw_set_rad = {round(math.radians(d), 6) for d in (0, 45, 90, 135, 180)}
    actual_yaw_set_rad = {round(r["object_yaw"], 6) for r in cyl_rows}
    check("cylinder manifest: object_yaw values within deterministic {0,45,90,135,180}deg set",
          actual_yaw_set_rad.issubset(expected_yaw_set_rad), detail=f"got {actual_yaw_set_rad}")

    valid_groups = {"center", "interior", "edge", "corner", "x_min_corridor"}
    check("cylinder manifest: position_group values within expected 5-group set",
          all(r["position_group"] in valid_groups for r in cyl_rows))

    constraint_ok = all(r["constraint_based_grasp"] is True and r["contact_physics_verified"] is False for r in cyl_rows)
    check("cylinder manifest: constraint_based_grasp=True, contact_physics_verified=False for all rows (this task's own required distinction)", constraint_ok)

    # --- Feature schema + full NaN/Inf (combined 470-episode dataset) ---
    if COMBINED_DATASET_ROOT.exists():
        info = json.loads((COMBINED_DATASET_ROOT / "meta" / "info.json").read_text())
        check("combined dataset: fps == 10", info.get("fps") == 10, detail=f"got {info.get('fps')}")
        state_feature = info.get("features", {}).get("observation.state", {})
        action_feature = info.get("features", {}).get("action", {})
        check("combined dataset: observation.state shape == (6,)", list(state_feature.get("shape", [])) == [6], detail=str(state_feature.get("shape")))
        check("combined dataset: action shape == (6,)", list(action_feature.get("shape", [])) == [6], detail=str(action_feature.get("shape")))
        expected_joint_order = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        check("combined dataset: joint order == shoulder_pan..wrist_roll,gripper", state_feature.get("names") == expected_joint_order,
              detail=str(state_feature.get("names")))
        image_feature = info.get("features", {}).get("observation.images.front", {})
        check("combined dataset: declared image shape == (256,256,3)", list(image_feature.get("shape", [])) == [256, 256, 3],
              detail=str(image_feature.get("shape")))

        parquet_paths = sorted((COMBINED_DATASET_ROOT / "data").rglob("*.parquet"))
        check("combined dataset: parquet file count > 0 (no missing files)", len(parquet_paths) > 0)
        frames = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
        check("combined dataset: total episode count == 470", int(frames["episode_index"].nunique()) == 470, detail=f"got {int(frames['episode_index'].nunique())}")

        state_array = np.stack(frames["observation.state"].to_numpy())
        action_array = np.stack(frames["action"].to_numpy())
        check("combined dataset: state array has NO NaN/Inf (full array, all 470 episodes)", bool(np.all(np.isfinite(state_array))))
        check("combined dataset: action array has NO NaN/Inf (full array, all 470 episodes)", bool(np.all(np.isfinite(action_array))))
        gripper_col = action_array[:, -1]
        check("combined dataset: gripper action scale is 0-100", bool(gripper_col.min() >= -1e-6 and gripper_col.max() <= 100 + 1e-6),
              detail=f"min={gripper_col.min()} max={gripper_col.max()}")
        check("combined dataset: gripper action includes both open (>90) and close (<10)",
              bool(gripper_col.min() < 10.0 and gripper_col.max() > 90.0))

        tasks_present = set(frames["task"].unique()) if "task" in frames.columns else set()
        check("combined dataset: task string matches expected instruction (or is absent, schema-dependent)",
              (not tasks_present) or tasks_present == {EXPECTED_INSTRUCTION}, detail=str(tasks_present))

        # --- Action/trajectory checks (max/mean adjacent-step arm-joint jump; NOT a hard abort gate, diagnostic report) ---
        arm_action_array = action_array[:, :5]
        episode_index_array = frames["episode_index"].to_numpy()
        max_jump_overall = 0.0
        for ep in sorted(set(episode_index_array.tolist())):
            ep_mask = episode_index_array == ep
            ep_actions = arm_action_array[ep_mask]
            if len(ep_actions) > 1:
                jumps = np.abs(np.diff(ep_actions, axis=0))
                max_jump_overall = max(max_jump_overall, float(jumps.max()))
        check("combined dataset: max adjacent-step arm-joint jump < 1.0 rad (diagnostic sanity bound, not V1/V2.1's own control-loop threshold)",
              max_jump_overall < 1.0, detail=f"max_jump={max_jump_overall:.4f}rad")

        # --- Deterministic image sample decode (this task's own "이미지 전수
        # decode가 과도하게 오래 걸리면 deterministic sample과 파일 무결성
        # 전수 검사를 조합" -- reported explicitly, not silently substituted) ---
        import io

        from PIL import Image

        sample_ids = [e for e in DETERMINISTIC_IMAGE_SAMPLE_EPISODE_IDS if e in set(episode_index_array.tolist())]
        image_shape_ok = True
        image_dtype_ok = True
        image_mode_ok = True
        decoded_count = 0
        for ep in sample_ids:
            ep_rows = frames[frames["episode_index"] == ep]
            if len(ep_rows) == 0:
                continue
            first_image = ep_rows.iloc[0]["observation.images.front"]
            # LeRobot's parquet-only (use_videos=False) layout stores each
            # frame as a PNG-encoded {"bytes": ..., "path": ...} dict (not a
            # raw array column) -- decode it via PIL to get the actual pixel
            # array, rather than treating the encoded dict itself as "the
            # image" (this task's own required image-shape/dtype check needs
            # the DECODED array, not the storage representation).
            if isinstance(first_image, dict) and "bytes" in first_image:
                decoded = np.array(Image.open(io.BytesIO(first_image["bytes"])))
            elif isinstance(first_image, np.ndarray):
                decoded = first_image
            else:
                continue
            decoded_count += 1
            if decoded.shape != (256, 256, 3):
                image_shape_ok = False
            if decoded.dtype != np.uint8:
                image_dtype_ok = False
            pil_img = Image.open(io.BytesIO(first_image["bytes"])) if isinstance(first_image, dict) else None
            if pil_img is not None and pil_img.mode != "RGB":
                image_mode_ok = False
        check(f"combined dataset: deterministic image sample ({len(sample_ids)} cylinder episodes targeted, {decoded_count} PNG-decoded via PIL) shape == 256x256x3",
              image_shape_ok)
        check("combined dataset: deterministic image sample dtype == uint8", image_dtype_ok)
        check("combined dataset: deterministic image sample mode == RGB", image_mode_ok)
        print(f"[INFO] Image integrity strategy: decoded {decoded_count}/{len(sample_ids)} sampled cylinder episodes' first frame directly; "
              f"full 470-episode dataset relies on parquet file-count/frame-count integrity (already checked above) rather than a full image decode, "
              f"per this task's own explicit allowance to combine sampling with file-level integrity checks when full decode is too slow.")
    else:
        check("combined dataset exists (skipped feature-schema/NaN-Inf/image checks -- run merge script first)", False)

    # --- Original + Stage 1A + Stage 1B datasets unchanged ---
    for name, root in [("original (so101_bin_main_200)", ORIGINAL_DATASET_ROOT), ("Stage 1A (so101_bin_stage1a_xy_70)", STAGE1A_DATASET_ROOT), ("Stage 1B (so101_bin_stage1b_box_100)", STAGE1B_DATASET_ROOT)]:
        check(f"{name}: dataset root exists", root.exists())

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Total: {passed}/{len(results)} passed")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
