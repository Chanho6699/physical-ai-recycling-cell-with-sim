"""Stage 1B pre-training integrity validator (see this task's chat
report, "반드시 검증"). Read-only. Mirrors
benchmark/validate_so101_stage1a_dataset.py's own checks, extended for
the rehearsal composition (cube + box) and the combined 370-episode
dataset.

Checks:
  1. train/validation/test episode-index sets are pairwise disjoint
     (across original-validation, Stage 1A val/test, Stage 1B box
     val/test, and the final 160-episode train allowlist)
  2. no environment_seed is reused within the Stage 1B box manifest
  3. every Stage-1B box manifest row has all required metadata fields
  4. box train/validation/test counts are exactly 70/15/15
  5. cube rehearsal composition is exactly 45 (original-range) + 45
     (Stage 1A boundary/corner) = 90, box is exactly 70 -> 160 total
  6. every saved box row's failure_reason is None
  7. datasets/so101_bin_main_200's own files are byte-identical
     (sha256) to the pre-Stage-1A snapshot (still valid -- nothing has
     touched that dataset since)

Run:
  .venv-vla/bin/python -m benchmark.validate_so101_stage1b_dataset
"""

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_main_200"
BOX_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1b_box_100" / "stage1b_position_manifest.jsonl"
STAGE1B_TRAIN_ALLOWLIST_PATH = PROJECT_ROOT / "configs" / "so101_stage1b_train_episodes.json"
ORIGINAL_HASH_SNAPSHOT_PATH = Path(
    "/tmp/claude-1000/-home-rlack-Projects-physical-ai-recycling-cell/4a365940-468e-4fc4-af6b-44ca46439f30/scratchpad/stage1a/original_dataset_hashes_before.txt"
)

REQUIRED_BOX_FIELDS = [
    "episode_index", "environment_seed", "split", "position_region", "object_type", "object_shape",
    "object_dimensions", "object_mass", "object_friction", "object_x", "object_y", "x_offset", "y_offset",
    "distance_from_center", "is_corner_region", "is_negative_x_region", "yaw", "phase_id",
    "scripted_expert_success", "failure_reason",
]

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    box_rows = load_jsonl(BOX_MANIFEST_PATH)
    allowlist = json.loads(STAGE1B_TRAIN_ALLOWLIST_PATH.read_text())

    for split, expected in [("train", 70), ("validation", 15), ("test", 15)]:
        n = sum(1 for r in box_rows if r["split"] == split)
        check(f"box manifest: {split} count == {expected}", n == expected, detail=f"got {n}")

    missing = 0
    for r in box_rows:
        for field in REQUIRED_BOX_FIELDS:
            if field not in r or (r[field] is None and field != "failure_reason"):
                missing += 1
    check("box manifest: no missing required metadata fields", missing == 0, detail=f"{missing} missing")

    non_none_failure = sum(1 for r in box_rows if r["failure_reason"] is not None)
    check("box manifest: failure_reason is None for every saved episode", non_none_failure == 0, detail=f"{non_none_failure} non-None")

    seeds = [r["environment_seed"] for r in box_rows]
    check("box manifest: environment_seed unique across all 100 episodes", len(seeds) == len(set(seeds)),
          detail=f"{len(seeds)} rows, {len(set(seeds))} unique")

    composition = allowlist["train_composition"]
    check("rehearsal composition: cube original-range == 45", len(composition["cube_rehearsal_original_range"]) == 45)
    check("rehearsal composition: cube Stage 1A boundary/corner == 45", len(composition["cube_rehearsal_stage1a_boundary_corner"]) == 45)
    check("rehearsal composition: box train == 70", len(composition["box_train"]) == 70)
    check("rehearsal composition: cube total == 90", composition["cube_total"] == 90)
    check("rehearsal composition: box total == 70", composition["box_total"] == 70)
    check("train_episode_count == 160", allowlist["train_episode_count"] == 160)

    train_set = set(allowlist["train_episode_indices"])
    excluded_sets = {
        "existing_validation": set(allowlist["excluded_existing_validation_indices"]),
        "stage1a_validation": set(allowlist["excluded_stage1a_validation_indices"]),
        "stage1a_test": set(allowlist["excluded_stage1a_test_indices"]),
        "box_validation": set(allowlist["excluded_box_validation_indices"]),
        "box_test": set(allowlist["excluded_box_test_indices"]),
    }
    for name, excluded in excluded_sets.items():
        check(f"train ∩ {name} == ∅", len(train_set & excluded) == 0, detail=str(train_set & excluded))

    names = list(excluded_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = excluded_sets[names[i]] & excluded_sets[names[j]]
            check(f"{names[i]} ∩ {names[j]} == ∅", len(overlap) == 0, detail=str(overlap))

    # NOTE: train + excluded is NOT expected to equal 370 -- the
    # rehearsal design deliberately selects only 45-of-160 original-range
    # and 45-of-50 Stage-1A cube episodes for training, leaving the
    # remainder (115 + 5 = 120) neither trained on nor reserved for eval
    # (simply unused rehearsal-pool remainder, by design). The real
    # invariant is train ∩ excluded == ∅ for every excluded set (already
    # checked above) plus this total accounting for exactly 250 "spoken
    # for" indices (160 train + 90 excluded), with the rest genuinely unused.
    all_indices = set(train_set)
    for excluded in excluded_sets.values():
        all_indices |= excluded
    check("train + all excluded == 250 unique indices (160 train + 90 reserved-for-eval; "
          "remaining 120 of 370 are unused rehearsal-pool remainder by design)",
          len(all_indices) == 250, detail=f"got {len(all_indices)}")

    snapshot = {}
    for line in ORIGINAL_HASH_SNAPSHOT_PATH.read_text().splitlines():
        h, path = line.split(maxsplit=1)
        snapshot[path.strip()] = h
    snapshot_abs = {str((PROJECT_ROOT / p).resolve()): h for p, h in snapshot.items()}
    current = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ORIGINAL_DATASET_ROOT.rglob("*")) if p.is_file()}
    check("original dataset file count unchanged", len(current) == len(snapshot_abs), detail=f"before={len(snapshot_abs)} after={len(current)}")
    check("original dataset file hashes unchanged (sha256)", current == snapshot_abs)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Total: {passed}/{len(results)} passed")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
