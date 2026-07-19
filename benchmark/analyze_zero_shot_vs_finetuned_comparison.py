"""Zero-shot vs Fine-tuned comparison analysis (v0).

Loads the two result JSONs produced by
benchmark/run_zero_shot_vs_finetuned_comparison.py (one per checkpoint,
run with IDENTICAL --seeds/--positions/--instruction/--max-policy-steps)
and prints:

  - aggregate metrics per checkpoint (overall + broken down by
    split_tag: train_seen / validation_seen / never_seen -- see
    run_zero_shot_vs_finetuned_comparison.POSITION_SPLIT_TAG)
  - a PAIRED per-condition table (matched by (position_name, seed), since
    both runs used the same jittered object positions) with per-pair
    deltas -- more informative than an unpaired comparison at this
    sample size
  - a plain-language analysis of where fine-tuning helped, where it
    didn't, and why, grounded in the actual numbers (not asserted)

Run:
  .venv-vla/bin/python -m benchmark.analyze_zero_shot_vs_finetuned_comparison \\
    results/zero_shot_vs_finetuned/zero_shot.json \\
    results/zero_shot_vs_finetuned/fine_tuned.json
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zero_shot_json", type=str)
    parser.add_argument("fine_tuned_json", type=str)
    return parser.parse_args()


def load(path_str: str) -> dict:
    with open(resolve(path_str), "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _stdev(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _rate(bools):
    bools = list(bools)
    return sum(1 for b in bools if b) / len(bools) if bools else None


def aggregate(episodes: list) -> dict:
    pick_episodes = [e for e in episodes if e["pick_success"]]
    grasp_steps = [e["first_grasp_step"] for e in pick_episodes if e["first_grasp_step"] is not None]
    release_steps = [e["release_step"] for e in episodes if e["release_step"] is not None]
    all_latencies = [e["mean_inference_latency_ms"] for e in episodes if e["mean_inference_latency_ms"] is not None]

    return {
        "num_episodes": len(episodes),
        "task_success_rate": _rate(e["success"] for e in episodes),
        "pick_success_rate": _rate(e["pick_success"] for e in episodes),
        "place_success_rate_given_pick": (
            sum(1 for e in pick_episodes if e["success"]) / len(pick_episodes) if pick_episodes else None
        ),
        "final_task_status_distribution": dict(Counter(e["final_task_status"] for e in episodes)),
        "avg_episode_length": _mean(e["num_steps"] for e in episodes),
        "avg_policy_steps": _mean(e["num_steps"] for e in episodes),  # identical to episode length -- see note below
        "avg_inference_latency_ms": _mean(all_latencies),
        "avg_distance_improvement": _mean(e["distance_improvement"] for e in episodes),
        "std_distance_improvement": _stdev([e["distance_improvement"] for e in episodes]),
        "avg_direction_cosine": _mean(e["mean_cosine_commanded_vs_object"] for e in episodes),
        "avg_x_sign_accuracy": _mean(e["x_sign_accuracy"] for e in episodes),
        "avg_first_grasp_step": _mean(grasp_steps),
        "avg_release_step": _mean(release_steps),
        "retry_count": 0,  # RealVLAPolicyClient has no retry logic -- see run_zero_shot_vs_finetuned_comparison.py
        "failure_reason_distribution": dict(Counter(e["failure_reason"] for e in episodes if not e["success"])),
    }


def aggregate_by_split(episodes: list) -> dict:
    split_tags = sorted(set(e["split_tag"] for e in episodes))
    return {tag: aggregate([e for e in episodes if e["split_tag"] == tag]) for tag in split_tags}


def print_aggregate_table(label: str, agg: dict) -> None:
    print(f"--- {label} (n={agg['num_episodes']}) ---")
    print(f"  task_success_rate:              {agg['task_success_rate']}")
    print(f"  pick_success_rate:               {agg['pick_success_rate']}")
    print(f"  place_success_rate_given_pick:   {agg['place_success_rate_given_pick']}")
    print(f"  final_task_status_distribution:  {agg['final_task_status_distribution']}")
    print(f"  avg_episode_length:              {agg['avg_episode_length']}")
    print(f"  avg_policy_steps:                {agg['avg_policy_steps']} (== episode_length: no retries, 1 predict()/step)")
    print(f"  avg_inference_latency_ms:        {agg['avg_inference_latency_ms']}")
    print(f"  avg_distance_improvement:        {agg['avg_distance_improvement']} (std={agg['std_distance_improvement']})")
    print(f"  avg_direction_cosine:            {agg['avg_direction_cosine']}")
    print(f"  avg_x_sign_accuracy:             {agg['avg_x_sign_accuracy']}")
    print(f"  avg_first_grasp_step:            {agg['avg_first_grasp_step']} (among episodes that ever grasped)")
    print(f"  avg_release_step:                {agg['avg_release_step']} (among episodes that ever released)")
    print(f"  retry_count:                     {agg['retry_count']} (constant -- no retry logic exists)")
    print(f"  failure_reason_distribution:     {agg['failure_reason_distribution']}")
    print()


def paired_table(zs_episodes: list, ft_episodes: list) -> list:
    zs_by_key = {(e["position_name"], e["seed"]): e for e in zs_episodes}
    ft_by_key = {(e["position_name"], e["seed"]): e for e in ft_episodes}
    keys = sorted(set(zs_by_key) & set(ft_by_key))
    missing = (set(zs_by_key) ^ set(ft_by_key))
    if missing:
        print(f"WARNING: {len(missing)} (position_name, seed) condition(s) present in only one run, excluded from pairing: {missing}")

    pairs = []
    for key in keys:
        zs = zs_by_key[key]
        ft = ft_by_key[key]
        pairs.append({
            "position_name": key[0],
            "seed": key[1],
            "split_tag": zs["split_tag"],
            "zero_shot_success": zs["success"],
            "fine_tuned_success": ft["success"],
            "zero_shot_status": zs["final_task_status"],
            "fine_tuned_status": ft["final_task_status"],
            "zero_shot_dist_improve": zs["distance_improvement"],
            "fine_tuned_dist_improve": ft["distance_improvement"],
            "dist_improve_delta": (
                (ft["distance_improvement"] - zs["distance_improvement"])
                if ft["distance_improvement"] is not None and zs["distance_improvement"] is not None
                else None
            ),
            "zero_shot_cosine": zs["mean_cosine_commanded_vs_object"],
            "fine_tuned_cosine": ft["mean_cosine_commanded_vs_object"],
            "cosine_delta": (
                (ft["mean_cosine_commanded_vs_object"] - zs["mean_cosine_commanded_vs_object"])
                if ft["mean_cosine_commanded_vs_object"] is not None and zs["mean_cosine_commanded_vs_object"] is not None
                else None
            ),
        })
    return pairs


def print_paired_table(pairs: list) -> None:
    print("=== Paired per-condition comparison (same object position/seed in both runs) ===")
    print(f"{'position':<13} {'seed':<5} {'split':<15} {'zs_success':<11} {'ft_success':<11} {'zs_status':<10} {'ft_status':<10} {'dist_Δ':>8} {'cos_Δ':>8}")
    for p in pairs:
        dist_delta = f"{p['dist_improve_delta']:+.4f}" if p["dist_improve_delta"] is not None else "n/a"
        cos_delta = f"{p['cosine_delta']:+.3f}" if p["cosine_delta"] is not None else "n/a"
        print(
            f"{p['position_name']:<13} {p['seed']:<5} {p['split_tag']:<15} {str(p['zero_shot_success']):<11} "
            f"{str(p['fine_tuned_success']):<11} {p['zero_shot_status']:<10} {p['fine_tuned_status']:<10} "
            f"{dist_delta:>8} {cos_delta:>8}"
        )
    print()


def main() -> None:
    args = parse_args()
    zs = load(args.zero_shot_json)
    ft = load(args.fine_tuned_json)

    print("=== Zero-shot vs Fine-tuned SmolVLA -- comparison ===")
    print(f"zero_shot model_id_or_path: {zs['model_id_or_path']}")
    print(f"fine_tuned model_id_or_path: {ft['model_id_or_path']}")
    print(f"instruction: {zs['instruction']!r} (zero_shot) / {ft['instruction']!r} (fine_tuned) -- "
          f"{'MATCH' if zs['instruction'] == ft['instruction'] else 'MISMATCH -- comparison invalid'}")
    print(f"seeds: {zs['seeds']} (zero_shot) / {ft['seeds']} (fine_tuned) -- "
          f"{'MATCH' if zs['seeds'] == ft['seeds'] else 'MISMATCH -- comparison invalid'}")
    print(f"positions: {zs['positions_requested']} (zero_shot) / {ft['positions_requested']} (fine_tuned) -- "
          f"{'MATCH' if zs['positions_requested'] == ft['positions_requested'] else 'MISMATCH -- comparison invalid'}")
    print(f"max_policy_steps: {zs['max_policy_steps']} (zero_shot) / {ft['max_policy_steps']} (fine_tuned) -- "
          f"{'MATCH' if zs['max_policy_steps'] == ft['max_policy_steps'] else 'MISMATCH -- comparison invalid'}")
    print()

    zs_agg = aggregate(zs["episodes"])
    ft_agg = aggregate(ft["episodes"])
    print_aggregate_table("ZERO-SHOT (HuggingFaceVLA/smolvla_libero)", zs_agg)
    print_aggregate_table("FINE-TUNED (local checkpoint)", ft_agg)

    print("=== Aggregate deltas (fine_tuned - zero_shot) ===")
    for metric in (
        "task_success_rate", "pick_success_rate", "place_success_rate_given_pick",
        "avg_episode_length", "avg_inference_latency_ms", "avg_distance_improvement",
        "avg_direction_cosine", "avg_x_sign_accuracy",
    ):
        zs_v, ft_v = zs_agg[metric], ft_agg[metric]
        delta = (ft_v - zs_v) if (zs_v is not None and ft_v is not None) else None
        print(f"  {metric:<32} zero_shot={zs_v} fine_tuned={ft_v} delta={delta}")
    print()

    print("=== By position split (train_seen / validation_seen / never_seen) ===")
    zs_by_split = aggregate_by_split(zs["episodes"])
    ft_by_split = aggregate_by_split(ft["episodes"])
    for tag in sorted(set(zs_by_split) | set(ft_by_split)):
        print(f"-- split={tag} --")
        zs_s = zs_by_split.get(tag, {})
        ft_s = ft_by_split.get(tag, {})
        print(f"  zero_shot:  success_rate={zs_s.get('task_success_rate')} n={zs_s.get('num_episodes')}")
        print(f"  fine_tuned: success_rate={ft_s.get('task_success_rate')} n={ft_s.get('num_episodes')}")
    print()

    pairs = paired_table(zs["episodes"], ft["episodes"])
    print_paired_table(pairs)

    print("=" * 70)
    print("Full aggregates (JSON) are in the two input files' 'episodes' arrays; this script only prints, it writes nothing.")


if __name__ == "__main__":
    main()
