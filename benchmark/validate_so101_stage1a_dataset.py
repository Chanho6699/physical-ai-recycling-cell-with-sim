"""Stage 1A pre-training integrity validator (see this task's chat
report, "실제 학습 전에 코드로 다음을 검증"). Read-only -- never
modifies any dataset, never trains anything. Exits non-zero (and
prints exactly which check failed) if any condition is violated, so a
CI-style caller can gate training on this.

Checks:
  1. train/validation/test episode-index sets are pairwise disjoint
  2. no environment_seed is reused across any two (split, region) rows
  3. every position-manifest row has all required metadata fields present
  4. train/validation/test counts are exactly 50/10/10
  5. every saved row's failure_reason is None (collect_episode() only
     ever saves place_success=True episodes -- this re-confirms it,
     rather than trusting that invariant blindly)
  6. datasets/so101_bin_main_200's own files are byte-identical
     (sha256) to the snapshot taken before Stage 1A work began

Run:
  .venv-vla/bin/python -m benchmark.validate_so101_stage1a_dataset
"""

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_main_200"
STANDALONE_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1a_xy_70" / "stage1a_position_manifest.jsonl"
COMBINED_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1a_combined_270" / "stage1a_position_manifest.jsonl"
TRAIN_ALLOWLIST_PATH = PROJECT_ROOT / "configs" / "so101_stage1a_train_episodes.json"
ORIGINAL_HASH_SNAPSHOT_PATH = Path(
    "/tmp/claude-1000/-home-rlack-Projects-physical-ai-recycling-cell/4a365940-468e-4fc4-af6b-44ca46439f30/scratchpad/stage1a/original_dataset_hashes_before.txt"
)

REQUIRED_FIELDS = [
    "episode_index", "environment_seed", "split", "position_region", "object_x", "object_y",
    "x_offset", "y_offset", "distance_from_center", "is_corner_region", "is_negative_x_region",
    "phase_id", "failure_reason",
]

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def load_manifest(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    standalone_rows = load_manifest(STANDALONE_MANIFEST_PATH)
    combined_rows = load_manifest(COMBINED_MANIFEST_PATH)
    allowlist = json.loads(TRAIN_ALLOWLIST_PATH.read_text())

    # 4. exact counts (standalone manifest, the source of truth for split assignment)
    for split, expected in [("train", 50), ("validation", 10), ("test", 10)]:
        n = sum(1 for r in standalone_rows if r["split"] == split)
        check(f"standalone manifest: {split} count == {expected}", n == expected, detail=f"got {n}")

    # 3. metadata completeness
    missing_count = 0
    for r in standalone_rows:
        for field in REQUIRED_FIELDS:
            if field not in r or (r[field] is None and field != "failure_reason"):
                missing_count += 1
    check("standalone manifest: no missing required metadata fields", missing_count == 0, detail=f"{missing_count} missing")

    # 5. failure_reason None for every saved row (scripted-expert-success-only)
    non_none_failure = sum(1 for r in standalone_rows if r["failure_reason"] is not None)
    check("standalone manifest: failure_reason is None for every saved episode", non_none_failure == 0, detail=f"{non_none_failure} non-None")

    # 2. environment_seed uniqueness across ALL (split, region) rows
    seeds = [r["environment_seed"] for r in standalone_rows]
    check("standalone manifest: environment_seed unique across all 70 episodes", len(seeds) == len(set(seeds)),
          detail=f"{len(seeds)} rows, {len(set(seeds))} unique")

    # 1. train/validation/test episode-index disjointness (combined dataset indices, via allowlist + manifest)
    train_set = set(allowlist["train_episode_indices"])
    new_val_set = set(allowlist["excluded_new_validation_indices"])
    new_test_set = set(allowlist["excluded_new_test_indices"])
    existing_val_set = set(allowlist["excluded_existing_validation_indices"])

    check("train_episode_indices count == 210", len(train_set) == 210, detail=f"got {len(train_set)}")
    check("train ∩ new_validation == ∅", len(train_set & new_val_set) == 0, detail=str(train_set & new_val_set))
    check("train ∩ new_test == ∅", len(train_set & new_test_set) == 0, detail=str(train_set & new_test_set))
    check("train ∩ existing_validation == ∅", len(train_set & existing_val_set) == 0, detail=str(train_set & existing_val_set))
    check("new_validation ∩ new_test == ∅", len(new_val_set & new_test_set) == 0, detail=str(new_val_set & new_test_set))
    check("new_validation ∩ existing_validation == ∅ (disjoint index spaces)", len(new_val_set & existing_val_set) == 0)
    check("new_test ∩ existing_validation == ∅ (disjoint index spaces)", len(new_test_set & existing_val_set) == 0)

    all_270 = train_set | new_val_set | new_test_set | existing_val_set
    check("train + new_val + new_test + existing_val == 270 unique indices", len(all_270) == 270, detail=f"got {len(all_270)}")

    # cross-check: combined dataset manifest's own split-labeled indices match the allowlist's
    combined_train_indices = {r["episode_index"] for r in combined_rows if r["split"] == "train"}
    combined_val_indices = {r["episode_index"] for r in combined_rows if r["split"] == "validation"}
    combined_test_indices = {r["episode_index"] for r in combined_rows if r["split"] == "test"}
    new_train_in_allowlist = set(allowlist["train_episode_indices"]) - set(json.loads(Path(PROJECT_ROOT / "results" / "so101_pipeline_runs" / "all_20260720_114358" / "split.json").read_text())["train_episodes"])
    check("combined-dataset manifest new-train indices == allowlist new-train indices", combined_train_indices == new_train_in_allowlist,
          detail=f"combined={sorted(combined_train_indices)} allowlist={sorted(new_train_in_allowlist)}")
    check("combined-dataset manifest new-validation indices == allowlist", combined_val_indices == new_val_set)
    check("combined-dataset manifest new-test indices == allowlist", combined_test_indices == new_test_set)

    # 6. original dataset unchanged (sha256 snapshot comparison)
    snapshot_lines = ORIGINAL_HASH_SNAPSHOT_PATH.read_text().splitlines()
    snapshot = {}
    for line in snapshot_lines:
        h, path = line.split(maxsplit=1)
        snapshot[path.strip()] = h
    current = {}
    for path in sorted(ORIGINAL_DATASET_ROOT.rglob("*")):
        if path.is_file():
            current[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    # snapshot paths were written relative to the project root (via `find
    # datasets/so101_bin_main_200 ...` run with cwd=PROJECT_ROOT) -- resolve
    # to absolute for a like-for-like comparison against `current` above.
    snapshot_abs = {str((PROJECT_ROOT / p).resolve()): h for p, h in snapshot.items()}
    check("original dataset file count unchanged", len(current) == len(snapshot_abs), detail=f"before={len(snapshot_abs)} after={len(current)}")
    check("original dataset file hashes unchanged (sha256)", current == snapshot_abs,
          detail=f"{sum(1 for k in current if snapshot_abs.get(k) != current.get(k))} files differ")

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Total: {passed}/{len(results)} passed")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
