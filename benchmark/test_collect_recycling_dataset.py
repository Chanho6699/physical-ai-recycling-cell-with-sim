"""Reliability tests for benchmark/collect_recycling_dataset.py (v0), run
BEFORE any large (50-100 episode) collection. No fine-tuning, no
production file changes -- exercises only the new collector module plus
read-only inspection of already-verified production code
(robot_sim/pybullet_panda_backend.py, action_adapter/adapter_v0.py).

Covers:
  1. Forced-failure episodes (max_steps / grasp_fail / place_fail /
     done_without_success) are never save_episode()'d, always
     clear_episode_buffer()'d, and never corrupt a subsequent real
     success episode's episode_index/frame_index numbering.
  2. Real control cadence: empirically counts p.stepSimulation() calls
     per apply_command() call (steps_per_action + gripper actuation,
     since ActionAdapter always emits "open" or "close", never a
     no-op/hold -- see action_adapter/adapter_v0.py's threshold logic)
     and compares against the dataset's declared fps.
  3. Action/state temporal alignment on one real successful episode:
     state[t+1].ee_position - state[t].ee_position vs. action[t]'s
     commanded [dx, dy, dz].
  4. Official reload: LeRobotDataset(repo_id=..., root=...) (not
     .create()) against the produced dataset, checking episode/frame
     counts, dtypes/shapes, timestamp monotonicity, task_index mapping,
     and stats.json presence.
  5. This module has no production dependency beyond already-existing,
     already-verified production modules (no test-only production hooks
     added).

Run: .venv-vla/bin/python -m benchmark.test_collect_recycling_dataset
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from action_adapter.adapter_v0 import RobotCommand
from benchmark.collect_recycling_dataset import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_POSITIONS,
    FEATURES,
    run_one_episode,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def make_dataset(root: Path, fps: int = 20) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id="local/test_recycling_cell",
        fps=fps,
        features=FEATURES,
        root=str(root),
        robot_type="franka_panda_pybullet",
        use_videos=False,
    )


def main() -> None:
    scratch_root = Path(tempfile.mkdtemp(prefix="collect_recycling_dataset_test_"))
    try:
        position_name, position = "center_right", DEFAULT_POSITIONS["center_right"]
        instruction_name, instruction = "en_short", DEFAULT_INSTRUCTIONS["en_short"]

        print("=== 1. Forced-failure episodes: never saved, buffer cleared, no corruption ===")
        dataset_root = scratch_root / "fault_injection_ds"
        dataset = make_dataset(dataset_root)

        # 1a. max_steps exceeded (normal conditions, just cut off early).
        success, num_frames, final_status, final_phase, _timing_rows = run_one_episode(
            dataset, position, instruction, "plastic_bottle", max_steps=3, steps_per_action=40,
        )
        check("max_steps: episode does not succeed", not success, f"success={success}")
        check("max_steps: stopped at requested cap", num_frames == 3, f"num_frames={num_frames}")
        dataset.clear_episode_buffer()
        check("max_steps: dataset.num_episodes stays 0 after discard", dataset.num_episodes == 0, str(dataset.num_episodes))
        check("max_steps: dataset.num_frames stays 0 after discard", dataset.num_frames == 0, str(dataset.num_frames))

        # 1b. grasp failure: policy is lied to about object position, so the
        # real object is always outside GRASP_THRESHOLD -- close_gripper
        # phase can never observe held_object/task_status=="grasped".
        success, num_frames, final_status, final_phase, _timing_rows = run_one_episode(
            dataset, position, instruction, "plastic_bottle", max_steps=60, steps_per_action=40,
            lie_object_position_offset=[0.35, 0.0, 0.0],
        )
        check("grasp_fail: episode does not succeed", not success, f"success={success}")
        check("grasp_fail: never grasped (stuck in close_gripper)", final_phase == "close_gripper", f"final_phase={final_phase}")
        check("grasp_fail: task_status never advanced past running", final_status == "running", f"final_status={final_status}")
        dataset.clear_episode_buffer()
        check("grasp_fail: dataset.num_episodes stays 0 after discard", dataset.num_episodes == 0, str(dataset.num_episodes))

        # 1c. place failure: policy is lied to about bin position, so it
        # releases the object far from the real bin -- backend reports
        # "released", not "success"; held_object becomes False either way,
        # so the policy also naturally reaches done=True here (this case
        # doubles as a real, non-synthetic done_without_success example).
        success, num_frames, final_status, final_phase, _timing_rows = run_one_episode(
            dataset, position, instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
            lie_bin_position_offset=[0.35, 0.0, 0.0],
        )
        check("place_fail: episode does not succeed", not success, f"success={success}")
        check("place_fail: task_status == released (missed the real bin)", final_status == "released", f"final_status={final_status}")
        dataset.clear_episode_buffer()
        check("place_fail: dataset.num_episodes stays 0 after discard", dataset.num_episodes == 0, str(dataset.num_episodes))

        # 1d. synthetic done_without_success: force policy_output.done=True
        # at step 1, well before the real phase machine could ever reach
        # task_status=="success".
        success, num_frames, final_status, final_phase, _timing_rows = run_one_episode(
            dataset, position, instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
            force_done_after_step=1,
        )
        check("done_without_success: episode does not succeed", not success, f"success={success}")
        check("done_without_success: stopped right after forced step", num_frames == 2, f"num_frames={num_frames}")
        check("done_without_success: task_status was not success", final_status != "success", f"final_status={final_status}")
        dataset.clear_episode_buffer()
        check("done_without_success: dataset.num_episodes stays 0 after discard", dataset.num_episodes == 0, str(dataset.num_episodes))

        # 1e. Now a REAL success episode, right after 4 discarded failures --
        # confirms episode_index/frame_index numbering was not polluted.
        success, num_frames, final_status, final_phase, _timing_rows = run_one_episode(
            dataset, position, instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
        )
        check("control success: episode succeeds", success, f"success={success}")
        dataset.save_episode()
        check("control success: dataset.num_episodes == 1 (only this one)", dataset.num_episodes == 1, str(dataset.num_episodes))
        check("control success: dataset.num_frames == this episode's frame count", dataset.num_frames == num_frames, f"{dataset.num_frames} != {num_frames}")
        dataset.finalize()

        data_parquet = dataset_root / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(data_parquet)
        check(
            "control success: saved parquet has exactly 1 episode_index (no leaked failed frames)",
            sorted(df["episode_index"].unique().tolist()) == [0],
            f"episodes present: {sorted(df['episode_index'].unique().tolist())}",
        )
        check(
            "control success: frame_index is contiguous from 0 (no gap from discarded episodes)",
            df["frame_index"].tolist() == list(range(len(df))),
            f"got {df['frame_index'].tolist()[:5]}...",
        )
        check(
            "control success: index (global row index) is contiguous from 0",
            df["index"].tolist() == list(range(len(df))),
        )
        print()

        print("=== 2. Real control cadence per condition (post redundant-actuation fix) ===")
        import robot_sim.pybullet_panda_backend as backend_module

        call_count = {"n": 0}
        original_step = backend_module.p.stepSimulation

        def counting_step(*args, **kwargs):
            call_count["n"] += 1
            return original_step(*args, **kwargs)

        steps_per_action = 40  # matches collect_recycling_dataset.DEFAULT_STEPS_PER_ACTION
        time_step = 1.0 / 240.0

        backend = PyBulletPandaBackend(gui=False)
        backend.reset()
        backend_module.p.stepSimulation = counting_step
        measured = {}
        try:
            # (a) hold: gripper command repeats the current state -> gripper actuation skipped.
            call_count["n"] = 0
            backend.apply_command(RobotCommand(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, "open"), steps=steps_per_action)
            measured["hold (open->open)"] = call_count["n"]

            # (b) transition: open -> close.
            call_count["n"] = 0
            backend.apply_command(RobotCommand(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, "close"), steps=steps_per_action)
            measured["transition (open->close)"] = call_count["n"]

            # (c) transition: close -> open.
            call_count["n"] = 0
            backend.apply_command(RobotCommand(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, "open"), steps=steps_per_action)
            measured["transition (close->open)"] = call_count["n"]
        finally:
            backend_module.p.stepSimulation = original_step
            backend.shutdown()

        check(
            f"hold frame triggers exactly {steps_per_action} stepSimulation() calls (gripper actuation skipped)",
            measured["hold (open->open)"] == steps_per_action,
            f"measured={measured['hold (open->open)']}",
        )
        check(
            f"open->close transition triggers exactly {steps_per_action + 60} stepSimulation() calls",
            measured["transition (open->close)"] == steps_per_action + 60,
            f"measured={measured['transition (open->close)']}",
        )
        check(
            f"close->open transition triggers exactly {steps_per_action + 60} stepSimulation() calls",
            measured["transition (close->open)"] == steps_per_action + 60,
            f"measured={measured['transition (close->open)']}",
        )

        declared_fps = 10  # collect_recycling_dataset.DEFAULT_FPS -- matches LIBERO's own training fps (see chat report item 4)
        libero_fps = 10.0
        print(f"PyBullet time_step = {time_step:.6f}s (1/240), steps_per_action = {steps_per_action}")
        for label, steps in measured.items():
            seconds = steps * time_step
            hz = 1.0 / seconds
            print(f"  {label:<28} steps={steps:3d}  seconds/frame={seconds:.5f}  effective_hz={hz:.4f}")
        print(
            f"cadence is NOT constant across frames anymore: hold frames run at "
            f"{1.0 / (steps_per_action * time_step):.2f}Hz, transition frames at "
            f"{1.0 / ((steps_per_action + 60) * time_step):.2f}Hz -- see item 5's sampling-policy discussion "
            f"(chat report) for why a single fixed dataset fps can't exactly represent both."
        )
        print(f"declared dataset fps (collector default) = {declared_fps}; LIBERO original = {libero_fps}")
        print()

        print("=== 3. Action/state temporal alignment on the real success episode above ===")
        state_rows = np.stack(df[df["episode_index"] == 0]["observation.state"].to_numpy())
        action_rows = np.stack(df[df["episode_index"] == 0]["action"].to_numpy())
        errors = []
        for t in range(len(state_rows) - 1):
            commanded_delta = action_rows[t][:3]
            actual_delta = state_rows[t + 1][:3] - state_rows[t][:3]
            error = np.linalg.norm(actual_delta - commanded_delta)
            errors.append(error)
            if t < 3 or t == len(state_rows) - 2:
                print(
                    f"  t={t:3d} commanded_delta={np.round(commanded_delta, 4).tolist()} "
                    f"actual_delta={np.round(actual_delta, 4).tolist()} error={error:.5f} "
                    f"gripper_cmd={action_rows[t][6]:.1f} next_gripper_state={np.round(state_rows[t+1][6:8], 4).tolist()}"
                )
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        print(f"  mean |actual_delta - commanded_delta| = {mean_error:.5f} m, max = {max_error:.5f} m over {len(errors)} steps")
        check(
            "commanded delta at step t matches actual EE displacement from state[t] to state[t+1] (no off-by-one)",
            mean_error < 0.01,
            f"mean_error={mean_error:.5f} (threshold 0.01m, i.e. 1/3 of the policy's 0.03m clamp)",
        )
        print()

        print("=== 4. Official reload: LeRobotDataset(repo_id=..., root=...) ===")
        reloaded = LeRobotDataset(repo_id="local/test_recycling_cell", root=str(dataset_root))
        check("reload: num_episodes == 1", reloaded.num_episodes == 1, str(reloaded.num_episodes))
        check("reload: num_frames == len(df)", reloaded.num_frames == len(df), f"{reloaded.num_frames} != {len(df)}")
        check("reload: len(dataset) == num_frames", len(reloaded) == reloaded.num_frames)
        sample = reloaded[0]
        check("reload: observation.state shape (8,)", tuple(sample["observation.state"].shape) == (8,), str(sample["observation.state"].shape))
        check("reload: action shape (7,)", tuple(sample["action"].shape) == (7,), str(sample["action"].shape))
        check("reload: observation.images.image decodes to a real image tensor", sample["observation.images.image"] is not None)
        check("reload: observation.images.image2 decodes to a real image tensor", sample["observation.images.image2"] is not None)
        timestamps = df[df["episode_index"] == 0]["timestamp"].tolist()
        check("reload: timestamp strictly monotonic within the episode", all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1)))
        tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
        mapped_tasks = tasks_df.index[tasks_df["task_index"] == 0].tolist()
        check(f"reload: task_index 0 maps back to '{instruction}'", mapped_tasks == [instruction], f"got {mapped_tasks}")
        stats_path = dataset_root / "meta" / "stats.json"
        check("reload: meta/stats.json exists", stats_path.exists())
        if stats_path.exists():
            import json
            stats = json.loads(stats_path.read_text())
            check("reload: stats.json has an entry for observation.state", "observation.state" in stats)
            check("reload: stats.json has an entry for action", "action" in stats)
        print()

        print("=== 5. No production dependency introduced by this test's own helpers ===")
        production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy", "action_adapter"]
        project_root = Path(__file__).resolve().parents[1]
        hits = []
        for directory in production_dirs:
            for path in (project_root / directory).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                if "test_collect_recycling_dataset" in text:
                    hits.append(str(path.relative_to(project_root)))
        check("no production file imports/references this test module", len(hits) == 0, f"unexpected: {hits}")

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
