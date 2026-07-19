"""Collects the v2 train160/validation40 LeRobot datasets (see this
task's chat report) using the EXACT same object/bin coordinates, jitter,
and seeds already dry-run-verified 200/200 in
benchmark/dry_run_v2_dataset.py / benchmark/v2_dataset_positions.py --
no split redesign, no coordinate/seed changes here.

Reuses collect_recycling_dataset.run_one_episode() UNCHANGED except for
the bin_position parameter already added (and regression-tested) in the
prior task -- no further expert-policy, gripper-semantics, or
translation/rotation-scale-semantics edits.

Each PLANNED episode is attempted EXACTLY ONCE, in the fixed order
v2_dataset_positions.py already produces -- there is no retry-with-a-
different-seed loop substituting a failure with a fresh attempt (unlike
collect_recycling_dataset.py's own main(), which cycles through
--max-attempts until it collects --episodes successes). A failure here
is recorded immediately (to the failed-episodes log) and reported, not
quietly replaced, per this task's explicit requirement.

Writes to NEW dataset roots (never touches datasets/recycling_train80_v1
or datasets/recycling_validation20_v1):
  datasets/recycling_v2_train160
  datasets/recycling_v2_validation40

Run:
  .venv-vla/bin/python -m benchmark.collect_v2_dataset
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
from benchmark.v2_dataset_positions import (
    BIN_POSITION_NAMES,
    OBJECT_ANCHOR_NAMES,
    build_train_v2_episodes,
    build_validation_v2_episodes,
    split_combination_pools,
)
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_ROOT = "datasets/recycling_v2_train160"
VALIDATION_ROOT = "datasets/recycling_v2_validation40"
TRAIN_REPO_ID = "local/recycling_cell_v2_train160"
VALIDATION_REPO_ID = "local/recycling_cell_v2_validation40"

FAILED_EPISODES_LOG = "results/v2_dataset_build/failed_episodes.jsonl"
BIN_POSE_MISMATCH_TOLERANCE_M = 1e-6


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class _ActionSanityGuard:
    """Transiently wraps dataset.add_frame() to reject (raise) any frame
    whose 'action' array contains NaN/Inf -- same pattern already used
    in benchmark/collect_train80_validation20_dataset.py. Does not
    modify collect_recycling_dataset.py."""

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


def _verify_bin_pose_consistency(bin_position: list) -> None:
    """Independently re-verifies (a fresh backend, not the one the real
    episode runs on) that PyBulletPandaBackend.set_bin_position()
    genuinely places the bin at the intended coordinates before this
    episode's real run even starts -- catches a silent regression in
    that production method itself, not just in this script's own
    bookkeeping."""
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    state = backend.set_bin_position(list(bin_position))
    backend.shutdown()
    actual = state["bin_position"]
    mismatch = max(abs(a - b) for a, b in zip(actual, bin_position))
    if mismatch > BIN_POSE_MISMATCH_TOLERANCE_M:
        raise RuntimeError(
            f"bin pose mismatch: requested {bin_position}, simulator reports {actual} "
            f"(off by {mismatch}m, tolerance {BIN_POSE_MISMATCH_TOLERANCE_M}m)"
        )


def collect_split(split_name: str, root: str, repo_id: str, episodes: list, combination_ids: dict, failed_log_file) -> dict:
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=DEFAULT_FPS, features=FEATURES, root=str(resolve(root)),
        robot_type="franka_panda_pybullet", use_videos=False,
    )
    instructions = list(DEFAULT_INSTRUCTIONS.items())
    manifest_path = resolve(root) / "collection_manifest.jsonl"
    timing_dir = resolve(root) / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    frame_timing_path = timing_dir / "frame_timing.jsonl"

    saved = 0
    failed = 0
    print(f"\n=== Collecting {split_name}: {len(episodes)} planned episodes -> {root} ===")
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file, \
             open(frame_timing_path, "w", encoding="utf-8") as timing_file:
            for index, e in enumerate(episodes):
                instruction_name, instruction = instructions[index % len(instructions)]
                object_anchor_index = OBJECT_ANCHOR_NAMES.index(e["object_anchor_name"])
                combination_id = combination_ids[(e["object_anchor_name"], e["bin_name"])]

                # Bin-pose consistency check BEFORE this episode's own run
                # (see this task's chat report, item 3).
                bin_pose_error = None
                try:
                    _verify_bin_pose_consistency(e["bin_position"])
                except RuntimeError as exc:
                    bin_pose_error = str(exc)

                nan_inf_detected = False
                run_error = None
                if bin_pose_error is None:
                    try:
                        with _ActionSanityGuard(dataset):
                            success, num_frames, final_status, final_phase, timing_rows = run_one_episode(
                                dataset, e["position"], instruction, DEFAULT_OBJECT_TYPE,
                                DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION,
                                instruction_name=instruction_name, seed=e["seed"], split=split_name,
                                bin_position=e["bin_position"],
                            )
                    except ValueError as exc:
                        nan_inf_detected = True
                        run_error = str(exc)
                        success, num_frames, final_status, final_phase, timing_rows = False, 0, "nan_inf_detected", "aborted", []
                else:
                    success, num_frames, final_status, final_phase, timing_rows = False, 0, "bin_pose_mismatch", "aborted", []

                if success:
                    dataset.save_episode()
                    episode_index = saved
                    for row in timing_rows:
                        timing_file.write(json.dumps({
                            "episode_index": episode_index,
                            "object_anchor_name": e["object_anchor_name"],
                            "object_anchor_index": object_anchor_index,
                            "bin_name": e["bin_name"],
                            "bin_position": e["bin_position"],
                            "combination_id": combination_id,
                            **row,
                        }) + "\n")
                    timing_file.flush()
                    saved += 1
                else:
                    dataset.clear_episode_buffer()
                    failed += 1
                    failed_log_file.write(json.dumps({
                        "split": split_name,
                        "planned_index": index,
                        "object_anchor_name": e["object_anchor_name"],
                        "object_anchor_index": object_anchor_index,
                        "bin_name": e["bin_name"],
                        "bin_position": e["bin_position"],
                        "combination_id": combination_id,
                        "seed": e["seed"],
                        "position": e["position"],
                        "final_status": final_status,
                        "final_phase": final_phase,
                        "num_frames": num_frames,
                        "nan_inf_detected": nan_inf_detected,
                        "bin_pose_error": bin_pose_error,
                        "run_error": run_error,
                    }) + "\n")
                    failed_log_file.flush()

                manifest_file.write(json.dumps({
                    "attempt": index + 1,
                    "split": split_name,
                    "object_anchor_name": e["object_anchor_name"],
                    "object_anchor_index": object_anchor_index,
                    "position": e["position"],
                    "bin_name": e["bin_name"],
                    "bin_position": e["bin_position"],
                    "combination_id": combination_id,
                    "instruction_name": instruction_name,
                    "instruction": instruction,
                    "seed": e["seed"],
                    "success": success,
                    "final_status": final_status,
                    "final_phase": final_phase,
                    "num_frames": num_frames,
                    "saved": success,
                }) + "\n")
                manifest_file.flush()

                status = "OK" if success else "FAIL"
                print(
                    f"[{status}] [{split_name} {index + 1:03d}/{len(episodes)}] "
                    f"{e['object_anchor_name']:14s}+{e['bin_name']:7s} seed={e['seed']:7d} "
                    f"success={success} status={final_status:<16s} frames={num_frames:3d} saved={saved}/{len(episodes)}"
                )
    finally:
        dataset.finalize()

    print(f"=== {split_name} done: {saved} saved, {failed} failed, {len(episodes)} planned ===")
    return {"split": split_name, "root": root, "planned": len(episodes), "saved": saved, "failed": failed}


def main() -> None:
    train_episodes = build_train_v2_episodes()
    validation_episodes = build_validation_v2_episodes()

    train_combos, validation_combos = split_combination_pools()
    all_combos = sorted(set(train_combos) | set(validation_combos))
    combination_ids = {combo: f"combo_{i:03d}_{combo[0]}_{combo[1]}" for i, combo in enumerate(all_combos)}

    for root in (TRAIN_ROOT, VALIDATION_ROOT):
        if resolve(root).exists():
            raise RuntimeError(f"Refusing to overwrite existing dataset root: {resolve(root)}")

    failed_log_path = resolve(FAILED_EPISODES_LOG)
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    with open(failed_log_path, "w", encoding="utf-8") as failed_log_file:
        summaries.append(collect_split("train", TRAIN_ROOT, TRAIN_REPO_ID, train_episodes, combination_ids, failed_log_file))
        summaries.append(collect_split("validation", VALIDATION_ROOT, VALIDATION_REPO_ID, validation_episodes, combination_ids, failed_log_file))

    print("\n=== Collection summary ===")
    for s in summaries:
        print(s)
    print(f"Failed-episode log: {failed_log_path}")


if __name__ == "__main__":
    main()
