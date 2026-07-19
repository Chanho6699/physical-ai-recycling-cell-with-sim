"""Dataset gripper/phase statistics for datasets/recycling_train80_v1
(see this task's chat report: diagnosing WHY the 4000-step checkpoint
releases the gripper prematurely, before touching training/loss).

Item 1 (raw dataset gripper stats) reads directly from the real saved
parquet. Item 2 (phase-labeled stats) needs a per-frame phase label
that the LeRobot dataset schema itself has no field for -- so this
script RE-DERIVES it by replaying DummyOpenVLAPolicy against the exact
same (position, seed) list benchmark/collect_train80_validation20_dataset.py
used to build train80 (benchmark.train80_validation20_positions
.build_train_positions(), same order, 0 collection failures already
confirmed in that task's report, so episode_index i in the saved
dataset corresponds EXACTLY to build_train_positions()[i]) -- this is
a deterministic replay of the SAME expert/simulator, not a guess, and
was already confirmed byte-for-byte-reproducible in the very first
"expert policy validation" task's data-consistency check. No dataset
file is modified; this only reads datasets/recycling_train80_v1's
existing parquet/manifest for cross-checking.

Run: .venv-vla/bin/python -m benchmark.analyze_train80_gripper_phase_statistics
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from action_adapter.adapter_v0 import ActionAdapter
from benchmark.train80_validation20_positions import build_train_positions
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN80_ROOT = PROJECT_ROOT / "datasets/recycling_train80_v1"
BIN_POSITION = [0.3, 0.35, 0.05]
MAX_STEPS = 150
STEPS_PER_ACTION = 40

PHASE_LABEL_MAP = {
    "move_to_object": "approach",
    "close_gripper": "grasp_transition",
    "lift_object": "lift",
    "move_above_bin": "carry",
    "open_gripper": "release",
    "done": "done",
}


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def replay_with_phase_labels(position, seed, instruction="플라스틱 병을 플라스틱 수거함에 넣어줘"):
    """Replays DummyOpenVLAPolicy at the exact position/seed, returning
    a list of per-step dicts: {step, phase, mapped_phase, gripper_label,
    translation, held_object, task_status}."""
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")
    backend.set_object_position(list(position))
    policy = DummyOpenVLAPolicy()
    policy.reset()
    action_adapter = ActionAdapter()

    rows = []
    for step_index in range(MAX_STEPS):
        robot_state = backend.get_state()
        object_position = list(robot_state["object_position"])
        policy_input = PolicyInput(
            image=None, instruction=instruction, robot_state=robot_state, task_goal={},
            target_object_position=object_position, bin_position=BIN_POSITION,
            step_index=step_index, phase=policy.phase,
        )
        policy_output = policy.predict_action(policy_input)
        robot_command = action_adapter.convert(policy_output.action)
        rows.append({
            "step": step_index,
            "phase": policy_output.phase,
            "mapped_phase": PHASE_LABEL_MAP.get(policy_output.phase, policy_output.phase),
            "gripper_label": policy_output.action[6],
            "translation": policy_output.action[0:3],
        })
        robot_state_after = backend.apply_command(robot_command, steps=STEPS_PER_ACTION)
        final_status = robot_state_after["task_status"]
        if final_status == "success" or policy_output.done:
            break
    backend.shutdown()
    return rows


def main() -> None:
    print("=== Part 1: train80 raw dataset gripper statistics ===")
    pf = sorted((TRAIN80_ROOT / "data/chunk-000").glob("*.parquet"))
    df = pq.read_table(pf[0]).to_pandas()
    actions = np.stack(df["action"].to_numpy())
    grip = actions[:, 6]
    total_frames = len(grip)
    open_count = int((grip < 0.5).sum())
    close_count = int((grip >= 0.5).sum())
    print(f"total frames: {total_frames}")
    print(f"open: {open_count} ({open_count/total_frames:.2%})  close: {close_count} ({close_count/total_frames:.2%})")

    ep_groups = df.groupby("episode_index")
    per_episode_open = []
    per_episode_close = []
    per_episode_len = []
    for ep_idx, group in ep_groups:
        g = np.stack(group["action"].to_numpy())[:, 6]
        per_episode_open.append(int((g < 0.5).sum()))
        per_episode_close.append(int((g >= 0.5).sum()))
        per_episode_len.append(len(g))
    print(f"per-episode open frames: mean={statistics.mean(per_episode_open):.2f} min={min(per_episode_open)} max={max(per_episode_open)}")
    print(f"per-episode close frames: mean={statistics.mean(per_episode_close):.2f} min={min(per_episode_close)} max={max(per_episode_close)}")
    close_fraction_per_ep = [c / l for c, l in zip(per_episode_close, per_episode_len)]
    print(f"per-episode close fraction: mean={statistics.mean(close_fraction_per_ep):.2%} min={min(close_fraction_per_ep):.2%} max={max(close_fraction_per_ep):.2%}")
    print()

    print("=== Part 2: phase-reconstructed statistics (via deterministic replay) ===")
    positions = build_train_positions()
    assert len(positions) == len(per_episode_len), f"{len(positions)} planned vs {len(per_episode_len)} saved -- mismatch, cannot correlate 1:1"

    all_rows_by_episode = []
    phase_gripper_counter = Counter()
    phase_action_stats = defaultdict(list)
    release_steps = []
    close_run_lengths = []  # continuous close-run length starting at grasp (close_gripper phase start) through release
    episode_lengths_replay = []

    for i, p in enumerate(positions):
        rows = replay_with_phase_labels(p["position"], p["seed"])
        all_rows_by_episode.append(rows)
        episode_lengths_replay.append(len(rows))
        for r in rows:
            phase_gripper_counter[(r["mapped_phase"], round(r["gripper_label"], 1))] += 1
            phase_action_stats[r["mapped_phase"]].append(r["translation"])

        grasp_start = next((r["step"] for r in rows if r["phase"] == "close_gripper"), None)
        release_start = next((r["step"] for r in rows if r["phase"] == "open_gripper"), None)
        if grasp_start is not None and release_start is not None:
            release_steps.append(release_start)
            close_run_lengths.append(release_start - grasp_start)

    print(f"replay episode lengths vs saved dataset lengths match: {episode_lengths_replay == per_episode_len}")
    if episode_lengths_replay != per_episode_len:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(episode_lengths_replay, per_episode_len)) if a != b]
        print(f"  MISMATCHES (first 5): {mismatches[:5]}")

    print("\nphase -> gripper_label frame counts:")
    for (phase, label), count in sorted(phase_gripper_counter.items()):
        print(f"  {phase:20s} gripper={label:.1f}  count={count}")

    print("\nphase -> translation[0:3] stats (mean/std per axis):")
    for phase, translations in phase_action_stats.items():
        arr = np.array(translations)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        print(f"  {phase:20s} n={len(translations):4d} mean=({means[0]:+.4f},{means[1]:+.4f},{means[2]:+.4f}) std=({stds[0]:.4f},{stds[1]:.4f},{stds[2]:.4f})")

    print(f"\nrelease step (open_gripper phase start): mean={statistics.mean(release_steps):.2f} min={min(release_steps)} max={max(release_steps)} (n={len(release_steps)})")
    print(f"continuous close-run length (grasp start -> release start): mean={statistics.mean(close_run_lengths):.2f} min={min(close_run_lengths)} max={max(close_run_lengths)} (n={len(close_run_lengths)})")

    total_frame_count_by_phase = Counter()
    for rows in all_rows_by_episode:
        for r in rows:
            total_frame_count_by_phase[r["mapped_phase"]] += 1
    total_all = sum(total_frame_count_by_phase.values())
    print("\ntotal frame count by phase (fraction of all replayed frames):")
    for phase, count in sorted(total_frame_count_by_phase.items(), key=lambda kv: -kv[1]):
        print(f"  {phase:20s} {count:5d} ({count/total_all:.2%})")

    print("\n'return' phase check: does DummyOpenVLAPolicy ever have a return-to-home leg after release?")
    has_return = any(r["phase"] not in PHASE_LABEL_MAP for rows in all_rows_by_episode for r in rows)
    print(f"  unmapped phase strings found: {has_return} (state machine is move_to_object->close_gripper->lift_object->move_above_bin->open_gripper->done -- no return leg)")

    output = {
        "raw_dataset": {
            "total_frames": total_frames,
            "open_count": open_count,
            "close_count": close_count,
            "per_episode_open_mean": statistics.mean(per_episode_open),
            "per_episode_close_mean": statistics.mean(per_episode_close),
            "per_episode_close_fraction_mean": statistics.mean(close_fraction_per_ep),
        },
        "phase_gripper_counts": {f"{k[0]}|{k[1]}": v for k, v in phase_gripper_counter.items()},
        "release_step_stats": {"mean": statistics.mean(release_steps), "min": min(release_steps), "max": max(release_steps)},
        "close_run_length_stats": {"mean": statistics.mean(close_run_lengths), "min": min(close_run_lengths), "max": max(close_run_lengths)},
        "frame_count_by_phase": dict(total_frame_count_by_phase),
    }
    output_path = PROJECT_ROOT / "results/gripper_diagnosis/train80_gripper_phase_statistics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
