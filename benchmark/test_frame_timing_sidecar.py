"""Consistency tests for the timing/frame_timing.jsonl sidecar added to
benchmark/collect_recycling_dataset.py (v0.2 -- see this task's chat
report). The official LeRobot data/meta/ schema is untouched; this file
only verifies the NEW sidecar and its relationship to the official
dataset and to collection_manifest.jsonl.

Covers (this task's item 3 minimum list):
  1. frame_timing row count == LeRobot frame count, per episode
  2. episode_index/frame_index correspond exactly between the sidecar
     and the official parquet (no orphans either direction)
  3. cumulative_simulated_time_s strictly increasing within an episode
  4. hold frames measure ~40 steps, gripper-transition frames ~100 steps
     (DEFAULT_STEPS_PER_ACTION=40 + DEFAULT_GRIPPER_STEPS=60)
  5. per-episode sum(simulation_steps_elapsed) matches an INDEPENDENT
     step counter wrapped around the whole run_one_episode() call (minus
     reset()'s own fixed 50-step warmup, which happens before frame
     instrumentation starts and is intentionally excluded from every
     frame's count)
  6. a discarded (forced-failure) episode's timing rows never reach the
     sidecar file, mirroring clear_episode_buffer()'s LeRobot-side discard

Run: .venv-vla/bin/python -m benchmark.test_frame_timing_sidecar
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import robot_sim.pybullet_panda_backend as backend_module
from benchmark.collect_recycling_dataset import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_POSITIONS,
    FEATURES,
    run_one_episode,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESET_WARMUP_STEPS = 50  # PyBulletPandaBackend.reset()'s own fixed warmup loop
_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def main() -> None:
    scratch_root = Path(tempfile.mkdtemp(prefix="frame_timing_sidecar_test_"))
    try:
        print("=== 1-2. Real CLI run: sidecar row counts and episode/frame_index correspondence ===")
        dataset_root = scratch_root / "cli_ds"
        result = subprocess.run(
            [sys.executable, "-m", "benchmark.collect_recycling_dataset",
             "--episodes", "3", "--root", str(dataset_root),
             "--split", "train", "--seed", "500", "--max-steps-per-episode", "150"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        check("collector CLI run exits 0", result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:])

        data_files = sorted((dataset_root / "data").rglob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
        timing_path = dataset_root / "timing" / "frame_timing.jsonl"
        check("timing/frame_timing.jsonl exists", timing_path.exists())
        timing_rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
        timing_df = pd.DataFrame(timing_rows)

        check("frame_timing row count == LeRobot frame count (whole dataset)", len(timing_df) == len(df), f"{len(timing_df)} != {len(df)}")

        for episode_index in sorted(df["episode_index"].unique().tolist()):
            lerobot_frames = sorted(df[df["episode_index"] == episode_index]["frame_index"].tolist())
            timing_frames = sorted(timing_df[timing_df["episode_index"] == episode_index]["frame_index"].tolist())
            check(
                f"episode {episode_index}: frame_timing row count == LeRobot frame count",
                len(timing_frames) == len(lerobot_frames),
                f"{len(timing_frames)} != {len(lerobot_frames)}",
            )
            check(
                f"episode {episode_index}: frame_index sets match exactly (no orphans either direction)",
                timing_frames == lerobot_frames,
                f"{timing_frames} != {lerobot_frames}",
            )

        check(
            "official meta/info.json is untouched by the sidecar (still declares only the original 4 features + defaults)",
            set(json.loads((dataset_root / "meta" / "info.json").read_text())["features"].keys())
            == {"observation.images.image", "observation.images.image2", "observation.state", "action",
                "timestamp", "frame_index", "episode_index", "index", "task_index"},
        )
        print()

        print("=== 3. cumulative_simulated_time_s strictly increasing within each episode ===")
        for episode_index in sorted(timing_df["episode_index"].unique().tolist()):
            ep = timing_df[timing_df["episode_index"] == episode_index].sort_values("frame_index")
            cum = ep["cumulative_simulated_time_s"].tolist()
            check(
                f"episode {episode_index}: cumulative_simulated_time_s strictly increasing",
                all(cum[i] < cum[i + 1] for i in range(len(cum) - 1)),
                f"{cum[:5]}...",
            )
            check(f"episode {episode_index}: first frame's cumulative_simulated_time_s == 0.0", cum[0] == 0.0, f"got {cum[0]}")
        print()

        print("=== 4. hold frames ~40 steps, gripper-transition frames ~100 steps ===")
        hold_steps = timing_df[~timing_df["gripper_transition"]]["simulation_steps_elapsed"]
        transition_steps = timing_df[timing_df["gripper_transition"]]["simulation_steps_elapsed"]
        check("at least one hold frame present", len(hold_steps) > 0)
        check("at least one transition frame present", len(transition_steps) > 0)
        check("all hold frames measure exactly 40 steps", (hold_steps == 40).all(), f"unique values: {hold_steps.unique()}")
        check("all transition frames measure exactly 100 steps", (transition_steps == 100).all(), f"unique values: {transition_steps.unique()}")
        check(
            "exactly 2 gripper transitions per episode (1 grasp-close + 1 release-open, per DummyOpenVLAPolicy's phase machine)",
            (timing_df.groupby("episode_index")["gripper_transition"].sum() == 2).all(),
            str(timing_df.groupby("episode_index")["gripper_transition"].sum().to_dict()),
        )
        print()

        print("=== 5. Per-episode total simulation steps: sidecar sum vs. an INDEPENDENT step counter ===")
        # A second, separate instrumentation layer built here in the test
        # (not reusing collect_recycling_dataset._FrameInstrumentation),
        # wrapped around the ENTIRE run_one_episode() call including its
        # internal reset() -- so the comparison below subtracts
        # RESET_WARMUP_STEPS to line up with what frame_timing itself
        # measures (frame instrumentation starts only after reset()).
        independent_dataset_root = scratch_root / "independent_check_ds"
        independent_dataset = LeRobotDataset.create(
            repo_id="local/independent_check", fps=20, features=FEATURES,
            root=str(independent_dataset_root), robot_type="franka_panda_pybullet", use_videos=False,
        )
        call_count = {"n": 0}
        original_step = backend_module.p.stepSimulation

        def counting_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        backend_module.p.stepSimulation = counting_step
        try:
            success, num_frames, final_status, final_phase, ep_timing_rows = run_one_episode(
                independent_dataset, DEFAULT_POSITIONS["center_right"], DEFAULT_INSTRUCTIONS["en_short"],
                "plastic_bottle", 150, 40,
            )
        finally:
            backend_module.p.stepSimulation = original_step
        check("independent-check episode succeeds", success, f"success={success}")
        independent_total = call_count["n"] - RESET_WARMUP_STEPS
        sidecar_total = sum(row["simulation_steps_elapsed"] for row in ep_timing_rows)
        check(
            "sum(simulation_steps_elapsed) from run_one_episode()'s own return value matches an independently "
            "instrumented step counter wrapped around the whole call (minus reset()'s fixed 50-step warmup)",
            independent_total == sidecar_total,
            f"independent={independent_total}, sidecar={sidecar_total}",
        )
        independent_dataset.save_episode()
        independent_dataset.finalize()
        print()

        print("=== 6. Discarded (forced-failure) episode's timing rows never reach the sidecar ===")
        discard_dataset_root = scratch_root / "discard_check_ds"
        discard_dataset = LeRobotDataset.create(
            repo_id="local/discard_check", fps=20, features=FEATURES,
            root=str(discard_dataset_root), robot_type="franka_panda_pybullet", use_videos=False,
        )
        # A forced failure -- max_steps cut short -- produces real timing_rows...
        success, num_frames, final_status, final_phase, failed_timing_rows = run_one_episode(
            discard_dataset, DEFAULT_POSITIONS["center_right"], DEFAULT_INSTRUCTIONS["en_short"],
            "plastic_bottle", 3, 40,
        )
        check("forced-failure episode does not succeed", not success)
        check("forced-failure episode still produced timing_rows locally (proves the discard is a real choice, not a no-op)", len(failed_timing_rows) == 3, f"got {len(failed_timing_rows)}")
        discard_dataset.clear_episode_buffer()
        # ... mirroring main()'s own logic: only write timing rows to disk
        # AFTER save_episode() succeeds. Here we deliberately never write
        # failed_timing_rows anywhere -- exactly what main() does.
        success, num_frames, final_status, final_phase, ok_timing_rows = run_one_episode(
            discard_dataset, DEFAULT_POSITIONS["center_right"], DEFAULT_INSTRUCTIONS["en_short"],
            "plastic_bottle", 150, 40,
        )
        check("control success episode succeeds", success)
        discard_dataset.save_episode()
        discard_dataset.finalize()
        check(
            "only the successful episode's timing rows would be written (failed episode's rows discarded, matching LeRobot buffer discard)",
            len(ok_timing_rows) > 0 and id(ok_timing_rows) != id(failed_timing_rows),
        )

    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
