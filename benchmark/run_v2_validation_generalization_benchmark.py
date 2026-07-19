"""Benchmark B for the v2-2000->4000 resume-training task (see chat
report): generalization eval using the v2 validation40 split's ACTUAL
held-out (object_anchor, bin_name) combinations, object position/jitter,
bin_position, and seed -- as opposed to benchmark A
(run_checkpoint_comparison_benchmark.py), which always uses the fixed
center bin position on a separate, bin-invariant position set.

Reuses run_checkpoint_comparison_benchmark.run_episode() UNCHANGED (it
already accepts a per-episode bin_position argument) -- no expert,
decoder, threshold, or dataset changes; this is read-only evaluation
against benchmark/v2_dataset_positions.py's existing, already-collected
validation40 combination pool.

Run ONCE PER CHECKPOINT against a server already /load_model'd with
that checkpoint:

  .venv-vla/bin/python -m benchmark.run_v2_validation_generalization_benchmark \\
    --label v2_2500step_valgen --output results/checkpoint_comparison/v2_2500step_valgen.json
"""

import argparse
import json
import time
from datetime import datetime

from benchmark.collect_recycling_dataset import DEFAULT_INSTRUCTIONS
from benchmark.run_checkpoint_comparison_benchmark import (
    DEFAULT_MAX_POLICY_STEPS,
    DEFAULT_STEPS_PER_ACTION,
    DEFAULT_WORKSPACE_BOUNDS_STR,
    run_episode,
)
from benchmark.run_full_recycling_cell_demo import parse_workspace_bounds
from benchmark.run_vla_action_direction_diagnostic import resolve
from benchmark.v2_dataset_positions import build_validation_v2_episodes
from policy.real_vla_policy_client import RealVLAPolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--real-vla-config", type=str, default="configs/real_vla_backend_config.json")
    parser.add_argument("--instruction-name", type=str, default="ko_full", choices=list(DEFAULT_INSTRUCTIONS.keys()))
    parser.add_argument("--max-policy-steps", type=int, default=DEFAULT_MAX_POLICY_STEPS)
    parser.add_argument("--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION)
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instruction = DEFAULT_INSTRUCTIONS[args.instruction_name]
    workspace_bounds = parse_workspace_bounds(DEFAULT_WORKSPACE_BOUNDS_STR)

    validation_episodes = build_validation_v2_episodes()

    policy = RealVLAPolicyClient(config_path=resolve(args.real_vla_config), fallback_policy=None)
    health = policy.check_health()
    print(f"=== v2 validation40 generalization benchmark -- label={args.label!r} ===")
    print(f"server health: {health}")
    if health.get("model_status") != "loaded":
        raise RuntimeError(f"Server model_status={health.get('model_status')!r}, expected 'loaded'.")
    if not (health.get("compatibility") or {}).get("passed"):
        raise RuntimeError(f"Server compatibility.passed is not True: {health.get('compatibility')}")
    model_id_or_path = health.get("model_id_or_path")
    print(f"model_id_or_path: {model_id_or_path}")
    print(f"validation40 episodes: {len(validation_episodes)}")

    episodes = []
    start_time = time.time()
    for n, e in enumerate(validation_episodes, start=1):
        episode = run_episode(
            policy, "v2_validation40", e["object_anchor_name"], e["position"], e["seed"], instruction,
            args.instruction_name, e["bin_position"], args.max_policy_steps, args.steps_per_action,
            args.object_type, args.strict, args.label, workspace_bounds,
        )
        episode["bin_name"] = e["bin_name"]
        episode["bin_position"] = e["bin_position"]
        episodes.append(episode)
        print(
            f"[{n:02d}/{len(validation_episodes)}] anchor={e['object_anchor_name']:14s} bin={e['bin_name']:7s} "
            f"seed={e['seed']:7d} success={episode['success']} status={episode['final_task_status']:<10} "
            f"steps={episode['num_steps']:3d} pick={episode['pick_success']} "
            f"dist_improve={episode['distance_improvement']:.4f} mean_cos={episode['mean_cosine_commanded_vs_object']}"
        )

    elapsed_s = time.time() - start_time
    result = {
        "label": args.label,
        "model_id_or_path": model_id_or_path,
        "server_health_at_start": health,
        "instruction_name": args.instruction_name,
        "instruction": instruction,
        "max_policy_steps": args.max_policy_steps,
        "steps_per_action": args.steps_per_action,
        "object_type": args.object_type,
        "strict": args.strict,
        "num_episodes": len(episodes),
        "wall_clock_s": elapsed_s,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    success_rate = sum(1 for e in episodes if e["success"]) / len(episodes)
    pick_rate = sum(1 for e in episodes if e["pick_success"]) / len(episodes)
    print(f"\n=== Done: {len(episodes)} episodes, success_rate={success_rate:.2%}, pick_rate={pick_rate:.2%}, wall_clock={elapsed_s:.1f}s ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
