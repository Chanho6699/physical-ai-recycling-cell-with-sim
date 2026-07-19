"""Aggregates the 5 checkpoint-comparison result JSONs (A-E, see this
task's chat report) produced by
benchmark/run_checkpoint_comparison_benchmark.py into: (1) an overall
comparison table, (2) a train-distribution vs validation-distribution
breakdown per model, (3) representative trajectory extraction (success /
approach-failure / gripper-timing-failure) per model.

Run:
  .venv-vla/bin/python -m benchmark.analyze_checkpoint_comparison \\
    --inputs results/checkpoint_comparison/A_zero_shot.json ... \\
    --output results/checkpoint_comparison/summary.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_episodes(episodes: list) -> dict:
    n = len(episodes)
    if n == 0:
        return {}
    phase_of_failure = Counter(e["rows"][-1]["phase"] if e["rows"] else None for e in episodes if not e["success"])
    return {
        "num_episodes": n,
        "task_success_rate": sum(1 for e in episodes if e["success"]) / n,
        "pick_success_rate": sum(1 for e in episodes if e["pick_success"]) / n,
        "mean_final_distance_to_object": _mean([e["final_distance_to_object"] for e in episodes]),
        "mean_distance_improvement": _mean([e["distance_improvement"] for e in episodes]),
        "mean_cosine_commanded_vs_object": _mean([e["mean_cosine_commanded_vs_object"] for e in episodes]),
        "gripper_open_count": sum(e["gripper_open_count"] for e in episodes),
        "gripper_close_count": sum(e["gripper_close_count"] for e in episodes),
        "mean_gripper_close_ratio": _mean([e["gripper_close_ratio"] for e in episodes]),
        "mean_first_close_command_step": _mean([e["first_close_command_step"] for e in episodes]),
        "mean_distance_at_first_close_command": _mean([e["distance_at_first_close_command"] for e in episodes]),
        "timeout_count": sum(1 for e in episodes if e["timed_out"]),
        "workspace_violation_count": sum(e["workspace_violations"] for e in episodes),
        "mean_episode_length": _mean([e["num_steps"] for e in episodes]),
        "failure_reason_distribution": dict(Counter(e["failure_reason"] for e in episodes)),
        "failure_phase_distribution": dict(phase_of_failure),
    }


def pick_representative_episodes(episodes: list) -> dict:
    successes = [e for e in episodes if e["success"]]
    approach_failures = [
        e for e in episodes if not e["success"] and not e["pick_success"] and e["distance_improvement"] is not None
    ]
    approach_failures.sort(key=lambda e: e["distance_improvement"])  # worst approach first
    gripper_timing_failures = [
        e for e in episodes if not e["success"] and e["pick_success"] and e["failure_reason"] != "none (success)"
    ]

    def _trim(ep):
        if ep is None:
            return None
        trimmed = dict(ep)
        trimmed["rows"] = ep["rows"]  # keep full per-step trace for trajectory diagnosis
        return trimmed

    return {
        "success_example": _trim(successes[0]) if successes else None,
        "approach_failure_example": _trim(approach_failures[0]) if approach_failures else None,
        "gripper_timing_failure_example": _trim(gripper_timing_failures[0]) if gripper_timing_failures else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="One result JSON per model (A-E)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    comparison = {}
    representative = {}
    for input_path in args.inputs:
        data = json.load(open(resolve(input_path), encoding="utf-8"))
        label = data["label"]
        episodes = data["episodes"]
        train_eps = [e for e in episodes if e["split_tag"] == "train_distribution"]
        validation_eps = [e for e in episodes if e["split_tag"] == "validation_distribution"]

        comparison[label] = {
            "model_id_or_path": data["model_id_or_path"],
            "overall": summarize_episodes(episodes),
            "train_distribution": summarize_episodes(train_eps),
            "validation_distribution": summarize_episodes(validation_eps),
        }
        representative[label] = pick_representative_episodes(episodes)

        print(f"=== {label} ===")
        print(f"  overall success={comparison[label]['overall']['task_success_rate']:.2%} "
              f"pick={comparison[label]['overall']['pick_success_rate']:.2%}")
        print(f"  train success={comparison[label]['train_distribution']['task_success_rate']:.2%} "
              f"validation success={comparison[label]['validation_distribution']['task_success_rate']:.2%}")

    output = {"comparison": comparison, "representative_episodes": representative}
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
