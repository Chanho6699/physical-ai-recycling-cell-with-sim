"""V3 End-to-End Training Pipeline -- Dataset Builder (see this task's
chat report). One CLI that calls the two ALREADY-VALIDATED collectors
unchanged:

  - benchmark.collect_recycling_dataset.run_one_episode() for --normal
    episodes (varying object anchor + bin position, no EE-init
    randomization/perturbation/stabilization -- exactly how
    datasets/recycling_v2_train160 was collected).
  - benchmark.collect_v3_recovery_smoke.run_recovery_episode() for
    --recovery episodes (EE-init randomization always on, plus one of
    x/y/diagonal/overshoot perturbation or near-target stabilization,
    evenly split across those 5 types via build_recovery_plan_even5()).

No collector logic is duplicated -- this file only builds episode plans
(anchor/bin/seed combinations) and a shared LeRobotDataset, then calls
the two existing run_*_episode() functions in a single retry-capped
attempt loop, exactly mirroring collect_recycling_dataset.py's/
collect_v3_recovery_smoke.py's own attempt/save/discard pattern.

Run:
  .venv-vla/bin/python -m benchmark.build_v3_dataset --normal 300 --recovery 200
"""

import argparse
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, DEFAULT_OBJECT_TYPE, FEATURES, run_one_episode
from benchmark.collect_v3_recovery_smoke import (
    DEFAULT_MAX_STEPS_PER_EPISODE,
    DEFAULT_STEPS_PER_ACTION,
    build_recovery_plan_even5,
    run_recovery_episode,
)
from benchmark.v2_dataset_positions import BIN_POSITIONS, OBJECT_ANCHOR_NAMES, OBJECT_ANCHORS, OBJECT_Z
from benchmark.train80_validation20_positions import _jitter_xy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = "datasets/recycling_v3_dataset"
DEFAULT_REPO_ID = "local/recycling_cell_v3_dataset"
DEFAULT_FPS = 10
OBJECT_JITTER_RADIUS_M = 0.015
BIN_NAMES = list(BIN_POSITIONS.keys())


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_normal_plan(num_episodes: int, seed_base: int) -> list:
    """Deterministic round-robin over the SAME 16 object anchors x 5 bin
    positions this project's v2 collector already uses -- no new
    coordinate/split design, just enough combinations to reach
    num_episodes with a distinct jitter seed each."""
    combos = [(a, b) for a in OBJECT_ANCHOR_NAMES for b in BIN_NAMES]
    rng = random.Random(seed_base)
    rng.shuffle(combos)
    plan = []
    for i in range(num_episodes):
        anchor_name, bin_name = combos[i % len(combos)]
        seed = seed_base + i * 100
        plan.append({"plan_index": i, "anchor_name": anchor_name, "bin_name": bin_name, "seed": seed})
    return plan


def run_normal_episode_once(dataset, plan_entry: dict, instruction: str, instruction_name: str, object_type: str) -> dict:
    rng = random.Random(plan_entry["seed"])
    anchor_x, anchor_y = OBJECT_ANCHORS[plan_entry["anchor_name"]]
    position = _jitter_xy(anchor_x, anchor_y, OBJECT_Z, rng, OBJECT_JITTER_RADIUS_M)
    bin_position = list(BIN_POSITIONS[plan_entry["bin_name"]])

    success, num_frames, final_status, final_phase, _timing_rows = run_one_episode(
        dataset, position, instruction, object_type, DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION,
        instruction_name=instruction_name, seed=plan_entry["seed"], split="normal", bin_position=bin_position,
    )
    return {
        "success": success, "num_frames": num_frames, "final_status": final_status, "final_phase": final_phase,
        "position": position, "bin_position": bin_position,
    }


def collect_group(
    dataset, mode: str, plan: list, target: int, max_attempts: int, instruction: str, instruction_name: str,
    object_type: str, manifest_file, failed_file, episode_index_start: int,
) -> dict:
    saved = 0
    attempt = 0
    next_episode_index = episode_index_start
    plan_iter = iter(plan)
    print(f"\n=== Collecting {target} '{mode}' episodes ===")
    while saved < target and attempt < max_attempts:
        try:
            plan_entry = next(plan_iter)
        except StopIteration:
            reseeded = [{**p, "seed": p["seed"] + 777 * (attempt + 1)} for p in plan]
            plan_iter = iter(reseeded)
            plan_entry = next(plan_iter)

        attempt += 1
        try:
            if mode == "normal":
                result = run_normal_episode_once(dataset, plan_entry, instruction, instruction_name, object_type)
            else:
                result = run_recovery_episode(dataset, plan_entry, instruction, instruction_name, DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION, object_type)
        except (ValueError, RuntimeError) as exc:
            dataset.clear_episode_buffer()
            failed_file.write(json.dumps({"mode": mode, "attempt": attempt, "plan_index": plan_entry["plan_index"], "reason": str(exc)}) + "\n")
            failed_file.flush()
            print(f"[{mode}] [attempt {attempt:04d}] plan={plan_entry['plan_index']:3d} CRASH: {exc}")
            continue

        if result["success"]:
            dataset.save_episode()
            record = {
                "collection_mode": mode, "attempt": attempt, "episode_index": next_episode_index,
                "scenario_group": plan_entry.get("scenario_group", "normal"),
                "perturbation_type": plan_entry.get("perturbation_type"),
                "object_anchor_name": plan_entry["anchor_name"], "bin_name": plan_entry["bin_name"],
                "position": result["object_position"] if mode == "recovery" else result["position"],
                "bin_position": result["bin_position"], "seed": plan_entry["seed"],
                "instruction_name": instruction_name, "instruction": instruction,
                "success": True, "final_status": result["final_status"], "final_phase": result["final_phase"],
                "num_frames": result["num_frames"], "saved": True,
            }
            if mode == "recovery":
                record.update({
                    "ee_init_requested_offset": result["ee_init"]["requested_offset"],
                    "ee_init_actual_position": result["ee_init"]["actual_initial_ee_position"],
                    "ee_init_settle_error_m": result["ee_init"]["settle_error_m"],
                    "perturbation": result["perturbation"],
                    "max_distance_after_perturbation": result["max_distance_after_perturbation"],
                    "recovery_completion_step": result["recovery_completion_step"],
                    "correction_step_count": result["correction_step_count"],
                    "stabilization_steps": result["stabilization_steps"],
                    "near_target_entry_step": result["near_target_entry_step"],
                    "close_step": result["close_step"], "close_distance": result["close_distance"],
                })
            manifest_file.write(json.dumps(record) + "\n")
            manifest_file.flush()
            saved += 1
            next_episode_index += 1
        else:
            dataset.clear_episode_buffer()
            failed_file.write(json.dumps({
                "mode": mode, "attempt": attempt, "plan_index": plan_entry["plan_index"],
                "reason": f"episode_did_not_succeed: final_status={result['final_status']}",
            }) + "\n")
            failed_file.flush()

        print(f"[{mode}] [attempt {attempt:04d}] plan={plan_entry['plan_index']:3d} success={result['success']} saved={saved}/{target}")

    return {"mode": mode, "target": target, "saved": saved, "attempts": attempt, "next_episode_index": next_episode_index}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=int, default=0, help="Number of successful NORMAL episodes to collect.")
    parser.add_argument("--recovery", type=int, default=0, help="Number of successful RECOVERY episodes to collect.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--seed-base-normal", type=int, default=700000)
    parser.add_argument("--seed-base-recovery", type=int, default=900000)
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--max-attempts-multiplier", type=float, default=1.4)
    args = parser.parse_args()

    if args.normal <= 0 and args.recovery <= 0:
        raise ValueError("At least one of --normal/--recovery must be > 0.")

    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]
    root = resolve(args.root)
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset root: {root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=DEFAULT_FPS, features=FEATURES, root=str(root),
        robot_type="franka_panda_pybullet", use_videos=False,
    )

    manifest_path = root / "collection_manifest.jsonl"
    failed_path = root / "failed_attempts.jsonl"
    group_summaries = []
    next_episode_index = 0
    try:
        with open(manifest_path, "w", encoding="utf-8") as manifest_file, \
             open(failed_path, "w", encoding="utf-8") as failed_file:
            if args.normal > 0:
                normal_plan = build_normal_plan(args.normal, args.seed_base_normal)
                max_attempts = int(args.normal * args.max_attempts_multiplier) + 10
                summary = collect_group(
                    dataset, "normal", normal_plan, args.normal, max_attempts,
                    instruction, args.instruction_name, DEFAULT_OBJECT_TYPE, manifest_file, failed_file,
                    next_episode_index,
                )
                group_summaries.append(summary)
                next_episode_index = summary["next_episode_index"]
            if args.recovery > 0:
                recovery_plan = build_recovery_plan_even5(args.recovery, args.seed_base_recovery)
                max_attempts = int(args.recovery * args.max_attempts_multiplier) + 10
                summary = collect_group(
                    dataset, "recovery", recovery_plan, args.recovery, max_attempts,
                    instruction, args.instruction_name, DEFAULT_OBJECT_TYPE, manifest_file, failed_file,
                    next_episode_index,
                )
                group_summaries.append(summary)
                next_episode_index = summary["next_episode_index"]
    finally:
        dataset.finalize()

    total_saved = sum(g["saved"] for g in group_summaries)
    total_attempts = sum(g["attempts"] for g in group_summaries)
    run_summary = {
        "root": str(root), "repo_id": args.repo_id,
        "requested_normal": args.normal, "requested_recovery": args.recovery,
        "group_summaries": group_summaries,
        "total_saved_episodes": total_saved, "total_attempts": total_attempts,
    }
    with open(root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"\n=== Done: {total_saved} episodes saved ({total_attempts} attempts) -> {root} ===")
    for g in group_summaries:
        print(f"  {g['mode']}: {g['saved']}/{g['target']} saved, {g['attempts']} attempts")
    print(f"Manifest: {manifest_path}")
    print(f"Run summary: {root / 'run_summary.json'}")


if __name__ == "__main__":
    main()
