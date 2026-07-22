"""Stage 1B zero-shot rectangular-box evaluation (see this task's chat
report, "Stage 1B: rectangular-box shape generalization"). Evaluates
the Stage 1A OFFICIAL checkpoint (never fine-tuned on box shape) against
a rectangular box swapped in via object_footprint_xy_override, at 15
fixed positions spanning Stage 1A's validated XY range (center, existing-
range single-axis boundaries + 2 corners, expanded-range single-axis
boundaries + all 4 corners). This is BEFORE any box data collection or
training -- a pure before/after reference point.

Box shape: object_footprint_xy=[0.02, 0.03] (half-extents; full 4cm x
6cm x 4cm, 1.5x aspect ratio, X = grasp axis matching the gripper's
own closing direction -- confirmed empirically this task, see chat
report) via benchmark.so101_smolvla_rollout's own
object_footprint_xy_override (this task's own minimal addition to that
file). Reuses run_one_rollout()/load_policy_and_processors() unchanged.

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_box_zeroshot
"""

import json
from pathlib import Path

from benchmark.so101_smolvla_rollout import MAX_ROLLOUT_STEPS, load_policy_and_processors, run_one_rollout
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1A_CHECKPOINT = PROJECT_ROOT / "outputs" / "train" / "so101_stage1a_xy_reinforcement_20260721_231829" / "checkpoints" / "007500" / "pretrained_model"
BOX_FOOTPRINT_XY = [0.02, 0.03]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101_stage1b_reinforcement" / "zeroshot_box_on_stage1a_checkpoint" / "results.json"

OLD_RADIUS_M = 0.015
NEW_RADIUS_M = 0.01875
POLICY_NOISE_REPEATS = 5
POLICY_NOISE_BASE_SEED = 500000

POSITIONS = [
    {"position_id": "center", "x_offset": 0.0, "y_offset": 0.0},
    {"position_id": "existing_x_min", "x_offset": -OLD_RADIUS_M, "y_offset": 0.0},
    {"position_id": "existing_x_max", "x_offset": OLD_RADIUS_M, "y_offset": 0.0},
    {"position_id": "existing_y_min", "x_offset": 0.0, "y_offset": -OLD_RADIUS_M},
    {"position_id": "existing_y_max", "x_offset": 0.0, "y_offset": OLD_RADIUS_M},
    {"position_id": "existing_corner_pp", "x_offset": OLD_RADIUS_M, "y_offset": OLD_RADIUS_M},
    {"position_id": "existing_corner_nn", "x_offset": -OLD_RADIUS_M, "y_offset": -OLD_RADIUS_M},
    {"position_id": "expanded_x_min", "x_offset": -NEW_RADIUS_M, "y_offset": 0.0},
    {"position_id": "expanded_x_max", "x_offset": NEW_RADIUS_M, "y_offset": 0.0},
    {"position_id": "expanded_y_min", "x_offset": 0.0, "y_offset": -NEW_RADIUS_M},
    {"position_id": "expanded_y_max", "x_offset": 0.0, "y_offset": NEW_RADIUS_M},
    {"position_id": "expanded_corner_pp", "x_offset": NEW_RADIUS_M, "y_offset": NEW_RADIUS_M},
    {"position_id": "expanded_corner_pn", "x_offset": NEW_RADIUS_M, "y_offset": -NEW_RADIUS_M},
    {"position_id": "expanded_corner_np", "x_offset": -NEW_RADIUS_M, "y_offset": NEW_RADIUS_M},
    {"position_id": "expanded_corner_nn", "x_offset": -NEW_RADIUS_M, "y_offset": -NEW_RADIUS_M},
]
assert len(POSITIONS) == 15


def main() -> None:
    policy, preprocessor, postprocessor = load_policy_and_processors(STAGE1A_CHECKPOINT)

    results = []
    for position_index, pos in enumerate(POSITIONS):
        object_position = [
            DEFAULT_OBJECT_POSITION[0] + pos["x_offset"], DEFAULT_OBJECT_POSITION[1] + pos["y_offset"], DEFAULT_OBJECT_POSITION[2],
        ]
        for repeat_id in range(POLICY_NOISE_REPEATS):
            derived_seed = POLICY_NOISE_BASE_SEED + position_index * 1000 + repeat_id
            print(f"=== position={pos['position_id']} (x_offset={pos['x_offset']:+.5f} y_offset={pos['y_offset']:+.5f}) "
                  f"repeat={repeat_id} (derived_policy_seed={derived_seed}) ===")
            result = run_one_rollout(
                policy, preprocessor, postprocessor, seed=position_index, max_steps=MAX_ROLLOUT_STEPS,
                policy_noise_seed=derived_seed, object_position_override=object_position,
                object_footprint_xy_override=BOX_FOOTPRINT_XY,
            )
            result["position_id"] = pos["position_id"]
            result["x_offset"] = pos["x_offset"]
            result["y_offset"] = pos["y_offset"]
            result["repeat_id"] = repeat_id
            result["policy_noise_seed_used"] = derived_seed
            results.append(result)
            print(f"  place_success={result['model_rollout_place_success']} grasp_ever={result['grasp_was_ever_established']} "
                  f"min_dist={result['min_object_gripper_distance_m']} failure_reason={result['failure_reason']} "
                  f"scene_invalid={result.get('failure_reason') == 'scene_invalid'}")

    summary = {
        "checkpoint": str(STAGE1A_CHECKPOINT), "box_footprint_xy": BOX_FOOTPRINT_XY,
        "num_positions": len(POSITIONS), "policy_noise_repeats": POLICY_NOISE_REPEATS,
        "policy_noise_base_seed": POLICY_NOISE_BASE_SEED, "results": results,
        "success_count": sum(1 for r in results if r["model_rollout_place_success"]),
        "grasp_count": sum(1 for r in results if r["grasp_was_ever_established"]),
        "num_rollouts": len(results),
        "clamp_count": sum(1 for r in results if r["any_joint_limit_clamp_triggered"]),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"success_count: {summary['success_count']}/{summary['num_rollouts']}")
    print(f"grasp_count: {summary['grasp_count']}/{summary['num_rollouts']}")
    print(f"clamp_count: {summary['clamp_count']}")
    print(f"Results JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
