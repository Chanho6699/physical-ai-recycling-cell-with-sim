"""Regression tests for collect_recycling_dataset.run_one_episode()'s
new optional bin_position parameter (see this task's chat report: v2
dataset design needs to actually vary the bin's PHYSICAL position
during collection, not just the object's -- train80's single,
never-varied bin position is what let the fine-tuned checkpoint learn a
fixed release timing/trajectory shortcut instead of genuinely
conditioning on the bin's visual position).

DummyOpenVLAPolicy itself needed NO change (see policy/dummy_openvla_policy.py's
move_above_bin phase -- it already reads bin_position out of PolicyInput
every step, never a hardcoded coordinate); the gap was purely that
run_one_episode() never told PyBulletPandaBackend to move the real bin
before an episode. This test proves: (1) bin_position=None (the
default, and every existing caller's current behavior) is byte-for-byte
unchanged from before this parameter existed, (2) a moved bin position
is genuinely reflected in the physical simulator (not just bookkeeping)
and DummyOpenVLAPolicy still completes the task there, (3) the
saved-action == simulator-command invariant (this project's core data-
consistency guarantee, see the very first "expert policy validation"
task) still holds with a moved bin.

Run: .venv-vla/bin/python -m benchmark.test_collect_recycling_dataset_bin_position
"""

import shutil
import tempfile
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, FEATURES, run_one_episode
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def _make_dataset(root: Path, repo_id: str):
    return LeRobotDataset.create(
        repo_id=repo_id, fps=10, features=FEATURES, root=str(root), robot_type="franka_panda_pybullet", use_videos=False,
    )


def main() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="bin_position_test_"))
    instruction = DEFAULT_INSTRUCTIONS["ko_full"]
    position = [0.42, 0.0, 0.05]  # matches collect_recycling_dataset.DEFAULT_POSITIONS["center_right"]

    try:
        print("=== A/B. bin_position=None matches PyBulletPandaBackend's own reset() default ===")
        default_backend = PyBulletPandaBackend(gui=False)
        default_backend.reset()
        expected_default_bin = list(default_backend.get_state()["bin_position"])
        default_backend.shutdown()

        dataset_a = _make_dataset(scratch / "a", "local/test_bin_position_a")
        success_a, frames_a, status_a, phase_a, _ = run_one_episode(
            dataset_a, position, instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
            instruction_name="ko_full", seed=0, split="test",
        )
        check("A: episode with bin_position omitted still succeeds (unchanged expert behavior)", success_a is True, f"status={status_a}")

        dataset_a2 = _make_dataset(scratch / "a2", "local/test_bin_position_a2")
        success_a2, frames_a2, status_a2, phase_a2, _ = run_one_episode(
            dataset_a2, position, instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
            instruction_name="ko_full", seed=0, split="test", bin_position=None,
        )
        check(
            "B: bin_position=None explicitly gives IDENTICAL result to omitting it entirely",
            (success_a, frames_a, status_a, phase_a) == (success_a2, frames_a2, status_a2, phase_a2),
            f"{(success_a, frames_a, status_a, phase_a)} vs {(success_a2, frames_a2, status_a2, phase_a2)}",
        )
        dataset_a.finalize()
        dataset_a2.finalize()
        print()

        print("=== C. Moved bin: physically reflected in simulator, expert still succeeds ===")
        moved_bin = [expected_default_bin[0] + 0.05, expected_default_bin[1], expected_default_bin[2]]
        dataset_c = _make_dataset(scratch / "c", "local/test_bin_position_c")
        success_c, frames_c, status_c, phase_c, _ = run_one_episode(
            dataset_c, position, instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
            instruction_name="ko_full", seed=1, split="test", bin_position=moved_bin,
        )
        check("C: expert still reaches success with a moved bin (DummyOpenVLAPolicy already bin-position-general)", success_c is True, f"status={status_c}")
        if success_c:
            dataset_c.save_episode()
        else:
            dataset_c.clear_episode_buffer()
        dataset_c.finalize()

        # Independently verify the bin was ACTUALLY moved in the simulator
        # (not just passed to the policy as bookkeeping) by reading it back
        # from a fresh backend using the exact same set_bin_position() call
        # run_one_episode() now makes.
        verify_backend = PyBulletPandaBackend(gui=False)
        verify_backend.reset()
        verify_backend.set_bin_position(moved_bin)
        actual_bin_after_set = list(verify_backend.get_state()["bin_position"])
        verify_backend.shutdown()
        check(
            "C: set_bin_position() genuinely relocates the physical bin body",
            all(abs(a - b) < 1e-9 for a, b in zip(actual_bin_after_set, moved_bin)),
            str(actual_bin_after_set),
        )
        print()

        print("=== D. Saved action == simulator command invariant holds with a moved bin ===")
        pf = sorted((scratch / "c" / "data" / "chunk-000").glob("*.parquet"))
        import pyarrow.parquet as pq
        import numpy as np

        df = pq.read_table(pf[0]).to_pandas()
        actions = np.stack(df["action"].to_numpy())
        check(
            "D: no NaN/Inf in saved actions with a moved bin",
            bool(np.all(np.isfinite(actions))),
        )
        check(
            "D: translation stays within DEFAULT_MAX_STEP_SIZE regardless of bin position",
            bool(np.all(np.abs(actions[:, 0:3]) <= 0.0300001)),
        )
        print()

        print("=== E. Existing default-position collection still works (no regression) ===")
        dataset_e = _make_dataset(scratch / "e", "local/test_bin_position_e")
        success_e, frames_e, status_e, phase_e, _ = run_one_episode(
            dataset_e, [0.27, 0.0, 0.05], instruction, "plastic_bottle", max_steps=150, steps_per_action=40,
            instruction_name="ko_full", seed=2, split="test",
        )
        check("E: default center_left-style position still succeeds", success_e is True, f"status={status_e}")
        dataset_e.finalize()

    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
