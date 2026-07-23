"""Stage 1C cylinder validation / held-out test closed-loop evaluator
(see this task's chat report, "Cylinder validation"). Mirrors
benchmark/evaluate_so101_stage1b_box_expansion.py's own structure
exactly (manifest-driven positions, derived policy-noise seed formula
`base_seed + position_index*1000 + repeat_id`, reuses
run_one_rollout()/load_policy_and_processors() unchanged) -- the ONLY
difference is (a) the default manifest path points at Stage 1C's own
cylinder position manifest instead of Stage 1B's box one (different
field names: `episode_id`/`object_position`/`position_group`/
`region_name` vs. box's `episode_index`/`object_x`/`object_y`/
`position_region`), and (b) every rollout passes
`object_shape_override="cylinder", object_radius_override=0.02` (this
task's own additive parameters on build_rollout_backend()/
run_one_rollout(), added this task) so the scene actually spawns the
cylinder, not a cube/box.

Per-position-group breakdown (center/interior/edge/corner/
x_min_corridor) is reported in addition to the overall rate, since this
task's own checkpoint-selection criteria require both.

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_stage1c_expansion \\
    --checkpoint-dir <dir> --split validation --policy-noise-repeats 5 \\
    --policy-noise-base-seed 900000 --output-path results/.../cylinder_validation.json
"""

import argparse
import json
from pathlib import Path

from benchmark.so101_smolvla_rollout import MAX_ROLLOUT_STEPS, load_policy_and_processors, run_one_rollout
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1c_cylinder_100" / "stage1c_position_manifest.jsonl"
CYLINDER_RADIUS_M = 0.02
POLICY_NOISE_SEED_BLOCK_SIZE = 1000


def load_positions(manifest_path: Path, split: str) -> list:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    split_rows = [r for r in rows if r["split"] == split]
    split_rows.sort(key=lambda r: r["episode_id"])
    return split_rows


def derive_cylinder_policy_noise_seed(base_seed: int, position_index: int, repeat_id: int) -> int:
    if not (0 <= repeat_id < POLICY_NOISE_SEED_BLOCK_SIZE):
        raise ValueError(f"repeat_id={repeat_id} out of supported range [0, {POLICY_NOISE_SEED_BLOCK_SIZE})")
    return base_seed + position_index * POLICY_NOISE_SEED_BLOCK_SIZE + repeat_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--split", type=str, required=True, choices=["validation", "test"])
    parser.add_argument("--policy-noise-repeats", type=int, required=True)
    parser.add_argument("--policy-noise-base-seed", type=int, required=True)
    parser.add_argument("--max-rollout-steps", type=int, default=MAX_ROLLOUT_STEPS)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    manifest_path = Path(args.manifest_path)
    output_path = Path(args.output_path)

    positions = load_positions(manifest_path, args.split)
    print(f"Loaded {len(positions)} '{args.split}' cylinder positions from {manifest_path}")

    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint_dir)

    results = []
    for position_index, pos in enumerate(positions):
        object_position = [pos["object_position"][0], pos["object_position"][1], DEFAULT_OBJECT_POSITION[2]]
        for repeat_id in range(args.policy_noise_repeats):
            derived_seed = derive_cylinder_policy_noise_seed(args.policy_noise_base_seed, position_index, repeat_id)
            print(f"=== position_index={position_index} (group={pos['position_group']}, region={pos['region_name']}, "
                  f"episode_id={pos['episode_id']}, x_offset={pos['x_offset']:+.5f} y_offset={pos['y_offset']:+.5f}) "
                  f"repeat={repeat_id} (derived_policy_seed={derived_seed}) ===")
            result = run_one_rollout(
                policy, preprocessor, postprocessor, seed=position_index, max_steps=args.max_rollout_steps,
                policy_noise_seed=derived_seed, object_position_override=object_position,
                object_shape_override="cylinder", object_radius_override=CYLINDER_RADIUS_M,
            )
            result["position_index"] = position_index
            result["position_group"] = pos["position_group"]
            result["region_name"] = pos["region_name"]
            result["source_episode_id"] = pos["episode_id"]
            result["environment_seed"] = pos["seed"]
            result["x_offset"] = pos["x_offset"]
            result["y_offset"] = pos["y_offset"]
            result["repeat_id"] = repeat_id
            result["policy_noise_seed_used"] = derived_seed
            results.append(result)
            print(f"  place_success={result['model_rollout_place_success']} grasp_ever={result['grasp_was_ever_established']} "
                  f"min_dist={result['min_object_gripper_distance_m']} failure_reason={result['failure_reason']} "
                  f"clamp={result['any_joint_limit_clamp_triggered']}")

    summary = {
        "checkpoint": str(checkpoint_dir), "split": args.split, "manifest_path": str(manifest_path),
        "cylinder_radius_m": CYLINDER_RADIUS_M,
        "num_positions": len(positions), "policy_noise_repeats": args.policy_noise_repeats,
        "policy_noise_base_seed": args.policy_noise_base_seed, "max_rollout_steps": args.max_rollout_steps,
        "results": results,
        "success_count": sum(1 for r in results if r["model_rollout_place_success"]),
        "num_rollouts": len(results),
        "clamp_count": sum(1 for r in results if r["any_joint_limit_clamp_triggered"]),
    }
    for group in ("center", "interior", "edge", "corner", "x_min_corridor"):
        group_results = [r for r in results if r["position_group"] == group]
        summary[f"{group}_success_count"] = sum(1 for r in group_results if r["model_rollout_place_success"])
        summary[f"{group}_total"] = len(group_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"success_count: {summary['success_count']}/{summary['num_rollouts']}")
    print(f"clamp_count: {summary['clamp_count']}")
    for group in ("center", "interior", "edge", "corner", "x_min_corridor"):
        print(f"  {group}: {summary[f'{group}_success_count']}/{summary[f'{group}_total']}")
    print(f"Results JSON: {output_path}")


if __name__ == "__main__":
    main()
