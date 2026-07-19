"""Collects the train80/validation20 LeRobot v3.0 datasets (see this
task's chat report) using the SAME, already-validated
DummyOpenVLAPolicy + PyBulletPandaBackend + ActionAdapter pipeline as
benchmark/collect_recycling_dataset.py -- this script reuses
collect_recycling_dataset.run_one_episode() UNCHANGED (no expert-policy,
gripper-semantics, or translation/rotation-scale-semantics edits; those
are all out of scope for this task and enforced by not touching those
files at all), only supplying a different (position, seed) plan per
episode: the 4x4-grid train / interpolated-midpoint-cross validation
split from benchmark/train80_validation20_positions.py, pre-verified
for zero coordinate/seed overlap and (in
benchmark/precheck_train80_validation20_positions.py) 100/100 dry-run
success before this script ever writes a dataset frame.

Writes to NEW dataset roots (never collides with the existing
datasets/recycling_lerobot_v0_train20 / _validation5):
  datasets/recycling_train80_v1
  datasets/recycling_validation20_v1

Only SUCCESSFUL episodes are saved (dataset.save_episode()); failed
attempts are dataset.clear_episode_buffer()'d (never written) and
logged separately to results/dataset_build/train80_validation20_failed_episodes.jsonl.
Every action array is checked for NaN/Inf before being written (see
_ActionSanityGuard) -- a hit is treated as an episode failure, not a
crash, and is logged the same way.

Run:
  .venv-vla/bin/python -m benchmark.collect_train80_validation20_dataset
"""

import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from benchmark.collect_recycling_dataset import (
    DEFAULT_FPS,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_MAX_STEPS_PER_EPISODE,
    DEFAULT_OBJECT_TYPE,
    DEFAULT_STEPS_PER_ACTION,
    FEATURES,
    run_one_episode,
)
from benchmark.train80_validation20_positions import build_train_positions, build_validation_positions, verify_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_ROOT = "datasets/recycling_train80_v1"
VALIDATION_ROOT = "datasets/recycling_validation20_v1"
TRAIN_REPO_ID = "local/recycling_cell_train80_v1"
VALIDATION_REPO_ID = "local/recycling_cell_validation20_v1"

FAILED_EPISODES_LOG = "results/dataset_build/train80_validation20_failed_episodes.jsonl"


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class _ActionSanityGuard:
    """Transiently wraps dataset.add_frame() to reject (raise) any frame
    whose 'action' array contains NaN/Inf, without modifying
    collect_recycling_dataset.py's run_one_episode() (which calls
    add_frame() internally) -- same bound-method-wrapping pattern
    collect_recycling_dataset._FrameInstrumentation already uses for
    open_gripper/close_gripper. The caller catches the raised exception
    and treats it exactly like any other episode failure (discard
    buffer, log, don't save)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._original = dataset.add_frame

    def __enter__(self):
        def guarded(frame: dict):
            action = frame.get("action")
            if action is not None and not np.all(np.isfinite(action)):
                raise ValueError(f"NaN/Inf detected in action before add_frame(): {action}")
            return self._original(frame)

        self.dataset.add_frame = guarded
        return self

    def __exit__(self, *exc_info):
        self.dataset.add_frame = self._original


def collect_split(split_name: str, root: str, repo_id: str, positions: list, failed_log_file) -> dict:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=DEFAULT_FPS,
        features=FEATURES,
        root=str(resolve(root)),
        robot_type="franka_panda_pybullet",
        use_videos=False,
    )

    instructions = list(DEFAULT_INSTRUCTIONS.items())
    manifest_path = resolve(root) / "collection_manifest.jsonl"
    timing_dir = resolve(root) / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    frame_timing_path = timing_dir / "frame_timing.jsonl"

    saved = 0
    failed = 0
    print(f"\n=== Collecting {split_name}: {len(positions)} planned episodes -> {root} ===")
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file, \
             open(frame_timing_path, "w", encoding="utf-8") as timing_file:
            for index, p in enumerate(positions):
                instruction_name, instruction = instructions[index % len(instructions)]
                nan_inf_detected = False
                try:
                    with _ActionSanityGuard(dataset):
                        success, num_frames, final_status, final_phase, timing_rows = run_one_episode(
                            dataset, p["position"], instruction, DEFAULT_OBJECT_TYPE,
                            DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION,
                            instruction_name=instruction_name, seed=p["seed"], split=split_name,
                        )
                except ValueError as exc:
                    nan_inf_detected = True
                    success, num_frames, final_status, final_phase, timing_rows = False, 0, "nan_inf_detected", "aborted", []
                    print(f"  [{split_name} {index+1:03d}] NaN/Inf detected, discarding: {exc}")

                if success:
                    dataset.save_episode()
                    episode_index = saved
                    for row in timing_rows:
                        timing_file.write(json.dumps({"episode_index": episode_index, **row}) + "\n")
                    timing_file.flush()
                    saved += 1
                else:
                    dataset.clear_episode_buffer()
                    failed += 1
                    failed_log_file.write(json.dumps({
                        "split": split_name,
                        "anchor_name": p["anchor_name"],
                        "seed": p["seed"],
                        "position": p["position"],
                        "instruction_name": instruction_name,
                        "final_status": final_status,
                        "final_phase": final_phase,
                        "num_frames": num_frames,
                        "nan_inf_detected": nan_inf_detected,
                    }) + "\n")
                    failed_log_file.flush()

                manifest_file.write(json.dumps({
                    "attempt": index + 1,
                    "split": split_name,
                    "anchor_name": p["anchor_name"],
                    "position_name": p["anchor_name"],
                    "position": p["position"],
                    "instruction_name": instruction_name,
                    "instruction": instruction,
                    "seed": p["seed"],
                    "success": success,
                    "final_status": final_status,
                    "final_phase": final_phase,
                    "num_frames": num_frames,
                    "saved": success,
                }) + "\n")
                manifest_file.flush()

                print(
                    f"[{split_name} {index + 1:03d}/{len(positions)}] anchor={p['anchor_name']:16s} "
                    f"seed={p['seed']:6d} success={success} status={final_status:<10} "
                    f"frames={num_frames:3d} saved={saved}/{len(positions)}"
                )
    finally:
        dataset.finalize()

    print(f"=== {split_name} done: {saved} saved, {failed} failed, {len(positions)} planned ===")
    return {"split": split_name, "root": root, "planned": len(positions), "saved": saved, "failed": failed}


def main() -> None:
    train_positions = build_train_positions()
    validation_positions = build_validation_positions()

    split_check = verify_split(train_positions, validation_positions)
    assert split_check["num_exact_coordinate_duplicates"] == 0, "Refusing to collect: train/validation coordinates overlap"
    assert split_check["num_duplicate_seeds"] == 0, "Refusing to collect: train/validation seeds overlap"
    print("=== Pre-collection split verification (re-checked immediately before writing any dataset) ===")
    print(json.dumps(split_check, indent=2))

    for root in (TRAIN_ROOT, VALIDATION_ROOT):
        if resolve(root).exists():
            raise RuntimeError(f"Refusing to overwrite existing dataset root: {resolve(root)}")

    failed_log_path = resolve(FAILED_EPISODES_LOG)
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    with open(failed_log_path, "w", encoding="utf-8") as failed_log_file:
        summaries.append(collect_split("train", TRAIN_ROOT, TRAIN_REPO_ID, train_positions, failed_log_file))
        summaries.append(collect_split("validation", VALIDATION_ROOT, VALIDATION_REPO_ID, validation_positions, failed_log_file))

    print("\n=== Collection summary ===")
    for s in summaries:
        print(s)
    print(f"Failed-episode log: {failed_log_path}")


if __name__ == "__main__":
    main()
