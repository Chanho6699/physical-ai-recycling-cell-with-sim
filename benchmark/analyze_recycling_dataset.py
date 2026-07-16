"""Quality report + official-reload validation for a dataset produced by
benchmark/collect_recycling_dataset.py (v0).

Reloads the dataset via the official LeRobotDataset(repo_id=..., root=...)
constructor (not a hand-rolled parquet read) for the structural checks,
then reads the raw parquet directly (pandas) for the statistics that
constructor doesn't expose in bulk (per-column min/max/mean/std, image
hash-collision checks). If a collection_manifest.jsonl sidecar is present
(written by collect_recycling_dataset.py's main(), not part of the
official LeRobot schema), also reports position/instruction/seed
distribution and per-failure-mode counts across ALL attempts (including
ones that were discarded, never saved).

Run: .venv-vla/bin/python -m benchmark.analyze_recycling_dataset --root datasets/recycling_lerobot_v0
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="local/recycling_cell_v0")
    return parser.parse_args()


def image_hash(image_cell) -> str:
    return hashlib.sha1(image_cell["bytes"]).hexdigest()


def main() -> None:
    args = parse_args()
    root = resolve(args.root)

    print(f"=== Official reload: LeRobotDataset(repo_id={args.repo_id!r}, root={root}) ===")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=str(root))
    print(f"num_episodes = {dataset.num_episodes}")
    print(f"num_frames   = {dataset.num_frames}")
    print(f"fps          = {dataset.fps}")
    sample = dataset[0]
    print(f"observation.state shape = {tuple(sample['observation.state'].shape)}, dtype = {sample['observation.state'].dtype}")
    print(f"action shape            = {tuple(sample['action'].shape)}, dtype = {sample['action'].dtype}")
    print()

    data_files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    stats_path = root / "meta" / "stats.json"

    print("=== Structural validation ===")
    episode_lengths = df.groupby("episode_index").size()
    print(f"episodes: {len(episode_lengths)}, frames: {len(df)}, tasks: {len(tasks_df)}")

    monotonic_ok = True
    for episode_index, group in df.groupby("episode_index"):
        ts = group.sort_values("frame_index")["timestamp"].tolist()
        if not all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)):
            monotonic_ok = False
            print(f"  NON-MONOTONIC timestamps in episode {episode_index}")
    print(f"timestamp monotonicity per episode: {'OK' if monotonic_ok else 'FAILED'}")

    contiguous_ok = True
    for episode_index, group in df.groupby("episode_index"):
        frame_idx = group.sort_values("frame_index")["frame_index"].tolist()
        if frame_idx != list(range(len(frame_idx))):
            contiguous_ok = False
            print(f"  NON-CONTIGUOUS frame_index in episode {episode_index}: {frame_idx[:5]}...")
    print(f"frame_index contiguity (0..len-1) per episode: {'OK' if contiguous_ok else 'FAILED'}")

    task_index_values = set(df["task_index"].unique().tolist())
    tasks_defined = set(tasks_df["task_index"].unique().tolist())
    print(f"task_index values used in data: {sorted(task_index_values)}, defined in meta/tasks.parquet: {sorted(tasks_defined)}")
    print(f"task_index mapping consistent: {'OK' if task_index_values <= tasks_defined else 'FAILED (undefined task_index used)'}")

    print(f"meta/stats.json exists: {stats_path.exists()}")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        missing = [k for k in ("observation.state", "action") if k not in stats]
        print(f"stats.json covers observation.state/action: {'OK' if not missing else f'MISSING {missing}'}")
    print()

    print("=== Quality report ===")
    print(f"episode length: min={episode_lengths.min()}, mean={episode_lengths.mean():.1f}, max={episode_lengths.max()}")

    state_matrix = np.stack(df["observation.state"].to_numpy())
    action_matrix = np.stack(df["action"].to_numpy())
    state_names = ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz", "gripper_left", "gripper_right"]
    action_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_cmd"]

    print("observation.state per-dim min/max/mean/std:")
    for i, name in enumerate(state_names):
        col = state_matrix[:, i]
        print(f"  {name:<13} min={col.min():+.4f} max={col.max():+.4f} mean={col.mean():+.4f} std={col.std():.4f}")

    print("action per-dim min/max/mean/std:")
    for i, name in enumerate(action_names):
        col = action_matrix[:, i]
        print(f"  {name:<13} min={col.min():+.4f} max={col.max():+.4f} mean={col.mean():+.4f} std={col.std():.4f}")

    gripper_cmd = action_matrix[:, 6]
    close_ratio = float((gripper_cmd >= 0.5).mean())
    print(f"gripper open/close ratio: close={close_ratio:.2%}, open={1 - close_ratio:.2%}")

    main_hashes = [image_hash(cell) for cell in df["observation.images.image"]]
    wrist_hashes = [image_hash(cell) for cell in df["observation.images.image2"]]
    main_dup_ratio = 1 - len(set(main_hashes)) / len(main_hashes)
    wrist_dup_ratio = 1 - len(set(wrist_hashes)) / len(wrist_hashes)
    cross_collisions = sum(1 for m, w in zip(main_hashes, wrist_hashes) if m == w)
    print(f"main-camera duplicate-image ratio (identical PNG bytes across frames): {main_dup_ratio:.2%}")
    print(f"wrist-camera duplicate-image ratio: {wrist_dup_ratio:.2%}")
    print(f"main/wrist same-frame hash collisions (main == wrist, would indicate a wiring bug): {cross_collisions}/{len(df)}")
    print()

    manifest_path = root / "collection_manifest.jsonl"
    if manifest_path.exists():
        print("=== Collection manifest (all attempts, including discarded failures) ===")
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        total = len(records)
        successes = sum(1 for r in records if r["success"])
        print(f"attempts: {total}, successes: {successes}, success_rate: {successes / total:.2%}" if total else "no attempts recorded")

        position_counts = Counter(r["position_name"] for r in records)
        instruction_counts = Counter(r["instruction_name"] for r in records)
        seed_values = [r["seed"] for r in records if r["seed"] is not None]
        print(f"position distribution: {dict(position_counts)}")
        print(f"instruction distribution: {dict(instruction_counts)}")
        print(f"seed distribution: {'no seeds recorded (--seed not set -> jitter disabled)' if not seed_values else f'{len(seed_values)} seeds, range [{min(seed_values)}, {max(seed_values)}]'}")

        failure_reasons = Counter(
            f"{r['final_status']}@{r['final_phase']}" for r in records if not r["success"]
        )
        print(f"failure reasons (final_status@final_phase): {dict(failure_reasons)}")
    else:
        print("(no collection_manifest.jsonl found next to this dataset -- position/instruction/seed/failure-reason "
              "distribution not available; only the official parquet/meta files were analyzed above)")


if __name__ == "__main__":
    main()
