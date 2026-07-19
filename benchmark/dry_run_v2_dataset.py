"""Pre-collection dry-run for all 200 planned v2 dataset episodes (see
this task's chat report) -- no LeRobotDataset is created, no frame is
ever saved; this only runs the (already-general) DummyOpenVLAPolicy +
PyBulletPandaBackend + ActionAdapter pipeline to confirm every planned
(object position, bin position, seed) triple is actually reachable/
collision-free/gripper-feasible BEFORE spending time on the real
200-episode collection.

Reuses benchmark/evaluate_expert_policy_benchmark.run_episode() (now
fixed to also relocate the PHYSICAL simulator bin, not just the
policy's target -- see this task's chat report) unchanged.

Run:
  .venv-vla/bin/python -m benchmark.dry_run_v2_dataset \\
    --output results/v2_dataset_dry_run.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_OBJECT_TYPE, DEFAULT_STEPS_PER_ACTION
from benchmark.evaluate_expert_policy_benchmark import DEFAULT_WORKSPACE_BOUNDS_STR, run_episode
from benchmark.run_full_recycling_cell_demo import parse_workspace_bounds
from benchmark.v2_dataset_positions import build_train_v2_episodes, build_validation_v2_episodes

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dry_run_split(split_name: str, episodes: list, instruction: str, workspace_bounds) -> list:
    results = []
    for index, e in enumerate(episodes):
        episode = run_episode(
            f"{split_name}:{e['object_anchor_name']}+{e['bin_name']}", e["position"], instruction, "ko_full",
            e["bin_position"], e["seed"], DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION,
            DEFAULT_OBJECT_TYPE, workspace_bounds,
        )
        timed_out = episode["final_task_status"] == "running" and episode["num_steps"] >= DEFAULT_MAX_STEPS_PER_EPISODE
        passed = (
            episode["success"]
            and episode["pick_success"]
            and episode["workspace_violations"] == 0
            and episode["ik_residual_violations"] == 0
            and not timed_out
        )
        result = {
            "split": split_name,
            "object_anchor_name": e["object_anchor_name"],
            "bin_name": e["bin_name"],
            "bin_position": e["bin_position"],
            "seed": e["seed"],
            "position": e["position"],
            "task_success": episode["success"],
            "pick_success": episode["pick_success"],
            "place_success": episode["success"],
            "workspace_violations": episode["workspace_violations"],
            "ik_residual_violations": episode["ik_residual_violations"],
            "timed_out": timed_out,
            "num_steps": episode["num_steps"],
            "failure_reason": episode["failure_reason"],
            "dry_run_passed": passed,
        }
        results.append(result)
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] [{index+1:03d}/{len(episodes)}] {split_name}:{e['object_anchor_name']:14s}+{e['bin_name']:7s} "
            f"seed={e['seed']:7d} success={episode['success']} pick={episode['pick_success']} "
            f"ws_viol={episode['workspace_violations']} ik_viol={episode['ik_residual_violations']} "
            f"timeout={timed_out} steps={episode['num_steps']:3d} reason={episode['failure_reason']}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    workspace_bounds = parse_workspace_bounds(DEFAULT_WORKSPACE_BOUNDS_STR)
    instruction = DEFAULT_INSTRUCTIONS["ko_full"]

    train_episodes = build_train_v2_episodes()
    validation_episodes = build_validation_v2_episodes()

    print(f"=== v2 dataset dry-run: {len(train_episodes)} train + {len(validation_episodes)} validation ===")
    train_results = _dry_run_split("train", train_episodes, instruction, workspace_bounds)
    validation_results = _dry_run_split("validation", validation_episodes, instruction, workspace_bounds)

    all_results = train_results + validation_results
    failures = [r for r in all_results if not r["dry_run_passed"]]

    per_bin_success = Counter()
    per_bin_total = Counter()
    for r in all_results:
        per_bin_total[r["bin_name"]] += 1
        if r["dry_run_passed"]:
            per_bin_success[r["bin_name"]] += 1

    per_combo_failures = [
        {"object": r["object_anchor_name"], "bin": r["bin_name"], "seed": r["seed"], "reason": r["failure_reason"]}
        for r in failures
    ]

    output = {
        "num_train_checked": len(train_results),
        "num_validation_checked": len(validation_results),
        "num_failures": len(failures),
        "per_bin_success_rate": {b: f"{per_bin_success[b]}/{per_bin_total[b]}" for b in per_bin_total},
        "failures": per_combo_failures,
        "train_results": train_results,
        "validation_results": validation_results,
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print()
    print(f"=== Done: {len(all_results)} checked, {len(failures)} failure(s) ===")
    print("per-bin-position success rate:")
    for b, rate in output["per_bin_success_rate"].items():
        print(f"  {b:8s} {rate}")
    if failures:
        print("\nFAILING COMBINATIONS:")
        for f in per_combo_failures:
            print(f"  {f}")
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
