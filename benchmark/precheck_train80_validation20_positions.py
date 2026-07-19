"""Pre-collection dry-run validation for every planned train80/
validation20 episode (see this task's chat report, item 3: "수집 전
expert 검증"). Writes NO dataset -- runs DummyOpenVLAPolicy against
each of the exact 100 planned (position, seed) pairs from
benchmark/train80_validation20_positions.py using the same run_episode()
already used to validate the expert's operating envelope
(benchmark/evaluate_expert_policy_benchmark.py), with the SAME
max_steps_per_episode/steps_per_action collect_recycling_dataset.py
itself uses -- so a pass here is a deterministic prediction of the real
collection outcome for that exact (position, seed) pair (DummyOpenVLAPolicy
+ PyBulletPandaBackend + ActionAdapter have no other source of
randomness).

Run:
  .venv-vla/bin/python -m benchmark.precheck_train80_validation20_positions \\
    --output results/dataset_build/train80_validation20_precheck.json
"""

import argparse
import json
from pathlib import Path

from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS, DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_OBJECT_TYPE, DEFAULT_STEPS_PER_ACTION
from benchmark.evaluate_expert_policy_benchmark import DEFAULT_BIN_POSITION, DEFAULT_WORKSPACE_BOUNDS_STR, run_episode
from benchmark.run_full_recycling_cell_demo import parse_workspace_bounds
from benchmark.train80_validation20_positions import build_train_positions, build_validation_positions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _precheck_split(split_name: str, positions: list, instruction: str, workspace_bounds) -> list:
    results = []
    for index, p in enumerate(positions):
        episode = run_episode(
            f"{split_name}:{p['anchor_name']}", p["position"], instruction, "ko_full", DEFAULT_BIN_POSITION,
            p["seed"], DEFAULT_MAX_STEPS_PER_EPISODE, DEFAULT_STEPS_PER_ACTION, DEFAULT_OBJECT_TYPE, workspace_bounds,
        )
        timed_out = episode["final_task_status"] == "running" and episode["num_steps"] >= DEFAULT_MAX_STEPS_PER_EPISODE
        passed = (
            episode["success"]
            and episode["workspace_violations"] == 0
            and episode["ik_residual_violations"] == 0
            and not timed_out
        )
        results.append({
            "split": split_name,
            "anchor_name": p["anchor_name"],
            "seed": p["seed"],
            "position": p["position"],
            "task_success": episode["success"],
            "pick_success": episode["pick_success"],
            "workspace_violations": episode["workspace_violations"],
            "ik_residual_violations": episode["ik_residual_violations"],
            "timed_out": timed_out,
            "num_steps": episode["num_steps"],
            "failure_reason": episode["failure_reason"],
            "precheck_passed": passed,
        })
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {split_name}:{p['anchor_name']:16s} seed={p['seed']:6d} "
            f"success={episode['success']} pick={episode['pick_success']} "
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

    train_positions = build_train_positions()
    validation_positions = build_validation_positions()

    print(f"=== Pre-collection dry-run: {len(train_positions)} train + {len(validation_positions)} validation ===")
    train_results = _precheck_split("train", train_positions, instruction, workspace_bounds)
    validation_results = _precheck_split("validation", validation_positions, instruction, workspace_bounds)

    all_results = train_results + validation_results
    failures = [r for r in all_results if not r["precheck_passed"]]

    output = {
        "num_train_checked": len(train_results),
        "num_validation_checked": len(validation_results),
        "num_failures": len(failures),
        "failures": failures,
        "train_results": train_results,
        "validation_results": validation_results,
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print()
    print(f"=== Done: {len(all_results)} checked, {len(failures)} failure(s) ===")
    if failures:
        print("FAILING COORDINATES:")
        for f in failures:
            print(f"  {f['split']}:{f['anchor_name']} seed={f['seed']} position={f['position']} reason={f['failure_reason']}")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
