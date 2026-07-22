"""Stage 1B box validation / held-out box-test closed-loop evaluator
(see this task's chat report, "Stage 1B: rectangular-box shape
generalization"). Mirrors benchmark/evaluate_so101_stage1a_expansion.py's
own structure exactly (manifest-driven positions, derived policy-noise
seed formula `base_seed + position_index*1000 + repeat_id`, reuses
run_one_rollout()/load_policy_and_processors() unchanged) -- the ONLY
difference is (a) the default manifest path points at Stage 1B's own
box position manifest instead of Stage 1A's cube one, and (b) every
rollout passes `object_footprint_xy_override=BOX_FOOTPRINT_XY` so the
scene actually spawns the box, not a cube.

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_box_expansion \\
    --checkpoint-dir <dir> --split validation --policy-noise-repeats 5 \\
    --policy-noise-base-seed 600000 --output-path results/.../box_validation.json
"""

import argparse
import json
from pathlib import Path

from benchmark.collect_so101_stage1b_box_dataset import BOX_FOOTPRINT_XY
from benchmark.so101_smolvla_rollout import MAX_ROLLOUT_STEPS, load_policy_and_processors, run_one_rollout
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "datasets" / "so101_bin_stage1b_box_100" / "stage1b_position_manifest.jsonl"
POLICY_NOISE_SEED_BLOCK_SIZE = 1000


def load_positions(manifest_path: Path, split: str) -> list:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    split_rows = [r for r in rows if r["split"] == split]
    split_rows.sort(key=lambda r: r["episode_index"])
    return split_rows


def derive_box_policy_noise_seed(base_seed: int, position_index: int, repeat_id: int) -> int:
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
    print(f"Loaded {len(positions)} '{args.split}' box positions from {manifest_path}")

    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint_dir)

    results = []
    for position_index, pos in enumerate(positions):
        object_position = [pos["object_x"], pos["object_y"], DEFAULT_OBJECT_POSITION[2]]
        for repeat_id in range(args.policy_noise_repeats):
            derived_seed = derive_box_policy_noise_seed(args.policy_noise_base_seed, position_index, repeat_id)
            print(f"=== position_index={position_index} (region={pos['position_region']}, episode_index={pos['episode_index']}, "
                  f"x_offset={pos['x_offset']:+.5f} y_offset={pos['y_offset']:+.5f}) repeat={repeat_id} "
                  f"(derived_policy_seed={derived_seed}) ===")
            result = run_one_rollout(
                policy, preprocessor, postprocessor, seed=position_index, max_steps=args.max_rollout_steps,
                policy_noise_seed=derived_seed, object_position_override=object_position,
                object_footprint_xy_override=BOX_FOOTPRINT_XY,
            )
            result["position_index"] = position_index
            result["position_region"] = pos["position_region"]
            result["source_episode_index"] = pos["episode_index"]
            result["environment_seed"] = pos["environment_seed"]
            result["x_offset"] = pos["x_offset"]
            result["y_offset"] = pos["y_offset"]
            result["is_corner_region"] = pos["is_corner_region"]
            result["is_negative_x_region"] = pos["is_negative_x_region"]
            result["repeat_id"] = repeat_id
            result["policy_noise_seed_used"] = derived_seed
            results.append(result)
            print(f"  place_success={result['model_rollout_place_success']} grasp_ever={result['grasp_was_ever_established']} "
                  f"min_dist={result['min_object_gripper_distance_m']} failure_reason={result['failure_reason']}")

    summary = {
        "checkpoint": str(checkpoint_dir), "split": args.split, "manifest_path": str(manifest_path),
        "box_footprint_xy": BOX_FOOTPRINT_XY,
        "num_positions": len(positions), "policy_noise_repeats": args.policy_noise_repeats,
        "policy_noise_base_seed": args.policy_noise_base_seed, "max_rollout_steps": args.max_rollout_steps,
        "results": results,
        "success_count": sum(1 for r in results if r["model_rollout_place_success"]),
        "num_rollouts": len(results),
        "clamp_count": sum(1 for r in results if r["any_joint_limit_clamp_triggered"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"success_count: {summary['success_count']}/{summary['num_rollouts']}")
    print(f"clamp_count: {summary['clamp_count']}")
    print(f"Results JSON: {output_path}")


if __name__ == "__main__":
    main()
