"""V3 End-to-End Training Pipeline -- Checkpoint Rollout Evaluation (see
this task's chat report). Runs a FIXED rollout evaluation suite via
benchmark.run_checkpoint_comparison_benchmark.run_episode() UNCHANGED,
then derives the pipeline-stage metrics this task asks for (approach@10cm,
approach@5cm, close, close_distance, pick, place, overall success, mean
minimum distance, mean rollout length, failure-stage breakdown) from each
episode's existing per-step 'rows' / precomputed 'failure_reason' -- no
new rollout loop, no new failure classifier (reuses
run_checkpoint_comparison_benchmark._classify_failure_reason() indirectly
via episode['failure_reason'], already computed by run_episode()).

Suite fixing (this task's chat report item 3): by default the 40-episode
benchmark.checkpoint_eval_positions set is used (20 train_distribution +
20 validation_distribution, ALREADY deterministic/fixed -- the same set
every other checkpoint comparison this session has used). Pass
--suite-file to instead load a saved JSON suite (positions/seeds/bin
position/max_steps), so every checkpoint in a comparison run is
GUARANTEED to see byte-identical episodes read from one frozen file,
not just "the same deterministic function called again". Use
--save-suite-to to freeze the current suite to such a file once, up
front, before evaluating any checkpoint.

Requires a vla_server already /load_model'd with the checkpoint under
test (same pattern as every other checkpoint benchmark this session).

Run:
  .venv-vla/bin/python -m benchmark.evaluate_v3_checkpoint \\
    --label v3_step1000 --step 1000 \\
    --suite-file results/v3_pipeline/eval_suite.json \\
    --output results/v3_pipeline/rollout_eval_step1000.json \\
    --csv results/v3_pipeline/rollout_eval.csv
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmark.checkpoint_eval_positions import build_train_eval_positions, build_validation_eval_positions
from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS
from benchmark.run_checkpoint_comparison_benchmark import (
    DEFAULT_BIN_POSITION,
    DEFAULT_MAX_POLICY_STEPS,
    DEFAULT_STEPS_PER_ACTION,
    DEFAULT_WORKSPACE_BOUNDS_STR,
    run_episode,
)
from benchmark.run_full_recycling_cell_demo import parse_workspace_bounds
from benchmark.run_vla_action_direction_diagnostic import resolve
from policy.real_vla_policy_client import RealVLAPolicyClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = [
    "label", "step", "model_id_or_path", "timestamp", "num_episodes",
    "approach_10cm_rate", "approach_5cm_rate", "close_rate", "close_distance_mean",
    "pick_rate", "place_rate", "overall_success_rate", "mean_min_distance", "mean_rollout_length",
]


def build_default_suite(instruction_name: str, max_policy_steps: int, bin_position: list) -> dict:
    train_eval_positions = build_train_eval_positions()
    validation_eval_positions = build_validation_eval_positions()
    all_positions = [{"split_tag": "train_distribution", **p} for p in train_eval_positions] + [
        {"split_tag": "validation_distribution", **p} for p in validation_eval_positions
    ]
    return {
        "instruction_name": instruction_name, "max_policy_steps": max_policy_steps,
        "bin_position": bin_position, "num_episodes": len(all_positions), "episodes": all_positions,
    }


def load_suite(suite_file: str) -> dict:
    return json.loads(Path(suite_file).read_text(encoding="utf-8"))


def episode_stage_flags(episode: dict) -> dict:
    rows = episode["rows"]
    ever_within_10cm = any(r["distance_to_object"] <= 0.10 for r in rows)
    ever_within_5cm = any(r["distance_to_object"] <= 0.05 for r in rows)
    ever_closed = any(r["gripper_command"] == "close" for r in rows)
    min_distance = min(r["distance_to_object"] for r in rows)
    return {
        "approach_10cm": ever_within_10cm,
        "approach_5cm": ever_within_5cm,
        "close": ever_closed,
        "close_distance": episode.get("distance_at_first_close_command"),
        "pick": episode["pick_success"],
        "place": episode["release_step"] is not None,
        "success": episode["success"],
        "min_distance": min_distance,
        "rollout_length": episode["num_steps"],
        "failure_reason": episode["failure_reason"],
    }


def run_evaluation(
    label: str, real_vla_config: str, object_type: str, strict: bool,
    suite: dict = None, instruction_name: str = "ko_full", max_policy_steps: int = DEFAULT_MAX_POLICY_STEPS,
    steps_per_action: int = DEFAULT_STEPS_PER_ACTION,
) -> dict:
    if suite is None:
        suite = build_default_suite(instruction_name, max_policy_steps, list(DEFAULT_BIN_POSITION))
    instruction_name = suite["instruction_name"]
    max_policy_steps = suite["max_policy_steps"]
    bin_position = suite["bin_position"]
    instruction = DEFAULT_INSTRUCTIONS[instruction_name]
    workspace_bounds = parse_workspace_bounds(DEFAULT_WORKSPACE_BOUNDS_STR)

    policy = RealVLAPolicyClient(config_path=resolve(real_vla_config), fallback_policy=None)
    health = policy.check_health()
    if health.get("model_status") != "loaded":
        raise RuntimeError(f"Server model_status={health.get('model_status')!r}, expected 'loaded'.")
    if not (health.get("compatibility") or {}).get("passed"):
        raise RuntimeError(f"Server compatibility.passed is not True: {health.get('compatibility')}")
    model_id_or_path = health.get("model_id_or_path")

    episodes = []
    suite_episodes = suite["episodes"]
    for n, p in enumerate(suite_episodes, start=1):
        episode = run_episode(
            policy, p["split_tag"], p["anchor_name"], p["position"], p["seed"], instruction, instruction_name,
            bin_position, max_policy_steps, steps_per_action, object_type, strict, label, workspace_bounds,
        )
        episodes.append(episode)
        flags = episode_stage_flags(episode)
        print(f"[{n:02d}/{len(suite_episodes)}] {p['anchor_name']:16s} seed={p['seed']:7d} " + " ".join(f"{k}={v}" for k, v in flags.items() if k != "failure_reason") + f" fail={flags['failure_reason']}")

    flags_all = [episode_stage_flags(e) for e in episodes]
    n = len(flags_all)
    close_distances = [f["close_distance"] for f in flags_all if f["close_distance"] is not None]
    failure_reason_counts = Counter(f["failure_reason"] for f in flags_all)

    summary = {
        "label": label, "model_id_or_path": model_id_or_path, "num_episodes": n,
        "approach_10cm_rate": sum(f["approach_10cm"] for f in flags_all) / n,
        "approach_5cm_rate": sum(f["approach_5cm"] for f in flags_all) / n,
        "close_rate": sum(f["close"] for f in flags_all) / n,
        "close_distance_mean": float(np.mean(close_distances)) if close_distances else None,
        "pick_rate": sum(f["pick"] for f in flags_all) / n,
        "place_rate": sum(f["place"] for f in flags_all) / n,
        "overall_success_rate": sum(f["success"] for f in flags_all) / n,
        "mean_min_distance": float(np.mean([f["min_distance"] for f in flags_all])),
        "mean_rollout_length": float(np.mean([f["rollout_length"] for f in flags_all])),
        "failure_stage_counts": dict(failure_reason_counts),
        "episodes": episodes,
    }
    return summary


def append_csv_row(csv_path: Path, label: str, step, summary: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "label": label, "step": step, "model_id_or_path": summary["model_id_or_path"],
            "timestamp": datetime.now().isoformat(), "num_episodes": summary["num_episodes"],
            "approach_10cm_rate": summary["approach_10cm_rate"], "approach_5cm_rate": summary["approach_5cm_rate"],
            "close_rate": summary["close_rate"], "close_distance_mean": summary["close_distance_mean"],
            "pick_rate": summary["pick_rate"], "place_rate": summary["place_rate"],
            "overall_success_rate": summary["overall_success_rate"],
            "mean_min_distance": summary["mean_min_distance"], "mean_rollout_length": summary["mean_rollout_length"],
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--step", type=int, default=None, help="Training step this checkpoint corresponds to (recorded in the CSV row).")
    parser.add_argument("--output", type=str, default=None, help="Required unless --save-suite-to is used.")
    parser.add_argument("--csv", type=str, default="results/v3_pipeline/rollout_eval.csv")
    parser.add_argument("--suite-file", type=str, default=None, help="Load a frozen eval suite JSON instead of regenerating positions.")
    parser.add_argument("--save-suite-to", type=str, default=None, help="Write the (default, on-the-fly-generated) suite to this path and exit -- does not run any evaluation.")
    parser.add_argument("--real-vla-config", type=str, default="configs/real_vla_backend_config.json")
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--max-policy-steps", type=int, default=DEFAULT_MAX_POLICY_STEPS)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args()

    if args.save_suite_to:
        suite = build_default_suite(args.instruction_name, args.max_policy_steps, list(DEFAULT_BIN_POSITION))
        suite_path = resolve(args.save_suite_to)
        suite_path.parent.mkdir(parents=True, exist_ok=True)
        with open(suite_path, "w", encoding="utf-8") as f:
            json.dump(suite, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved fixed eval suite ({suite['num_episodes']} episodes) -> {suite_path}")
        return

    if not args.output:
        raise SystemExit("--output is required unless --save-suite-to is used.")

    suite = load_suite(args.suite_file) if args.suite_file else None

    summary = run_evaluation(
        args.label, args.real_vla_config, args.object_type, args.strict,
        suite=suite, instruction_name=args.instruction_name, max_policy_steps=args.max_policy_steps,
        steps_per_action=args.steps_per_action,
    )
    summary["step"] = args.step
    summary["timestamp"] = datetime.now().isoformat()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    append_csv_row(csv_path, args.label, args.step, summary)

    print(f"\n=== {args.label} (step={args.step}): "
          f"approach10cm={summary['approach_10cm_rate']:.2%} approach5cm={summary['approach_5cm_rate']:.2%} "
          f"close={summary['close_rate']:.2%} close_dist={summary['close_distance_mean']} "
          f"pick={summary['pick_rate']:.2%} place={summary['place_rate']:.2%} "
          f"success={summary['overall_success_rate']:.2%} mean_min_dist={summary['mean_min_distance']:.4f} "
          f"mean_len={summary['mean_rollout_length']:.1f} ===")
    print(f"failure stages: {summary['failure_stage_counts']}")
    print(f"Result JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
