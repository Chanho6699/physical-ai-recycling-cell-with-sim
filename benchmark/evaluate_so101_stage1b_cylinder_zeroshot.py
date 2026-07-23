"""Stage 1B checkpoint zero-shot cylinder evaluation (see this task's
chat report, "Stage 1B cylinder zero-shot 평가"). Evaluates the Stage
1B OFFICIAL checkpoint (never trained on any cylinder shape) against an
upright cylinder swapped in via object_shape_override="cylinder" +
object_radius_override=0.02 (this task's own minimal, additive changes
to benchmark/so101_smolvla_rollout.py's build_rollout_backend()/
run_one_rollout() -- None defaults preserve every existing cube/box
call site unchanged).

Cylinder shape: radius=0.02m, height=0.04m -- the SAME candidate
selected and Expert-validated in
benchmark/evaluate_so101_expert_v2_cylinder.py (125/125 = 100% legacy
AND physical success there). This is a pure before/any-training
reference point -- run ONCE, checkpoint is NOT re-selected based on
this result (this task's own explicit instruction).

Position groups (5, matching this task's own section 6/8 categories,
reusing the SAME position definitions
evaluate_so101_stage1b_box_zeroshot.py already established for Stage
1A/1B's own known/expanded/edge/corner distinction):
  known/interior  -> center + existing_x_min/x_max/y_min/y_max (within the ORIGINAL validated XY range)
  expanded        -> expanded_x_min/x_max/y_min/y_max (edge of the EXPANDED range)
  corner          -> existing_corner_pp/nn + expanded_corner_pp/pn/np/nn (true held-out corners)

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_stage1b_cylinder_zeroshot
"""

import json
from pathlib import Path

from benchmark.so101_smolvla_rollout import MAX_ROLLOUT_STEPS, load_policy_and_processors, run_one_rollout
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1B_CHECKPOINT = PROJECT_ROOT / "outputs" / "train" / "so101_stage1b_box_reinforcement_20260722_101757" / "checkpoints" / "007500" / "pretrained_model"
CYLINDER_RADIUS_M = 0.02
CYLINDER_HEIGHT_M = 0.04
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101_stage1c_cylinder_feasibility" / "zeroshot_cylinder_on_stage1b_checkpoint" / "results.json"

OLD_RADIUS_M = 0.015
NEW_RADIUS_M = 0.01875
POLICY_NOISE_REPEATS = 5
POLICY_NOISE_BASE_SEED = 800000

POSITIONS = [
    {"position_id": "center", "group": "known_interior", "x_offset": 0.0, "y_offset": 0.0},
    {"position_id": "existing_x_min", "group": "known_interior", "x_offset": -OLD_RADIUS_M, "y_offset": 0.0},
    {"position_id": "existing_x_max", "group": "known_interior", "x_offset": OLD_RADIUS_M, "y_offset": 0.0},
    {"position_id": "existing_y_min", "group": "known_interior", "x_offset": 0.0, "y_offset": -OLD_RADIUS_M},
    {"position_id": "existing_y_max", "group": "known_interior", "x_offset": 0.0, "y_offset": OLD_RADIUS_M},
    {"position_id": "existing_corner_pp", "group": "corner", "x_offset": OLD_RADIUS_M, "y_offset": OLD_RADIUS_M},
    {"position_id": "existing_corner_nn", "group": "corner", "x_offset": -OLD_RADIUS_M, "y_offset": -OLD_RADIUS_M},
    {"position_id": "expanded_x_min", "group": "expanded_edge", "x_offset": -NEW_RADIUS_M, "y_offset": 0.0},
    {"position_id": "expanded_x_max", "group": "expanded_edge", "x_offset": NEW_RADIUS_M, "y_offset": 0.0},
    {"position_id": "expanded_y_min", "group": "expanded_edge", "x_offset": 0.0, "y_offset": -NEW_RADIUS_M},
    {"position_id": "expanded_y_max", "group": "expanded_edge", "x_offset": 0.0, "y_offset": NEW_RADIUS_M},
    {"position_id": "expanded_corner_pp", "group": "corner", "x_offset": NEW_RADIUS_M, "y_offset": NEW_RADIUS_M},
    {"position_id": "expanded_corner_pn", "group": "corner", "x_offset": NEW_RADIUS_M, "y_offset": -NEW_RADIUS_M},
    {"position_id": "expanded_corner_np", "group": "corner", "x_offset": -NEW_RADIUS_M, "y_offset": NEW_RADIUS_M},
    {"position_id": "expanded_corner_nn", "group": "corner", "x_offset": -NEW_RADIUS_M, "y_offset": -NEW_RADIUS_M},
]
assert len(POSITIONS) == 15


def main() -> None:
    policy, preprocessor, postprocessor = load_policy_and_processors(STAGE1B_CHECKPOINT)

    results = []
    for position_index, pos in enumerate(POSITIONS):
        object_position = [
            DEFAULT_OBJECT_POSITION[0] + pos["x_offset"], DEFAULT_OBJECT_POSITION[1] + pos["y_offset"], DEFAULT_OBJECT_POSITION[2],
        ]
        for repeat_id in range(POLICY_NOISE_REPEATS):
            derived_seed = POLICY_NOISE_BASE_SEED + position_index * 1000 + repeat_id
            print(f"=== position={pos['position_id']} (group={pos['group']}, x_offset={pos['x_offset']:+.5f} y_offset={pos['y_offset']:+.5f}) "
                  f"repeat={repeat_id} (derived_policy_seed={derived_seed}) ===")
            result = run_one_rollout(
                policy, preprocessor, postprocessor, seed=position_index, max_steps=MAX_ROLLOUT_STEPS,
                policy_noise_seed=derived_seed, object_position_override=object_position,
                object_shape_override="cylinder", object_radius_override=CYLINDER_RADIUS_M,
            )
            result["position_id"] = pos["position_id"]
            result["group"] = pos["group"]
            result["x_offset"] = pos["x_offset"]
            result["y_offset"] = pos["y_offset"]
            result["repeat_id"] = repeat_id
            result["policy_noise_seed_used"] = derived_seed
            results.append(result)
            print(f"  place_success={result['model_rollout_place_success']} grasp_ever={result['grasp_was_ever_established']} "
                  f"min_dist={result['min_object_gripper_distance_m']} failure_reason={result['failure_reason']} "
                  f"scene_invalid={result.get('failure_reason') == 'scene_invalid'} clamp={result['any_joint_limit_clamp_triggered']}")

    summary = {
        "checkpoint": str(STAGE1B_CHECKPOINT), "object_shape": "cylinder",
        "cylinder_radius_m": CYLINDER_RADIUS_M, "cylinder_height_m": CYLINDER_HEIGHT_M,
        "num_positions": len(POSITIONS), "policy_noise_repeats": POLICY_NOISE_REPEATS,
        "policy_noise_base_seed": POLICY_NOISE_BASE_SEED, "results": results,
        "success_count": sum(1 for r in results if r["model_rollout_place_success"]),
        "grasp_count": sum(1 for r in results if r["grasp_was_ever_established"]),
        "num_rollouts": len(results),
        "clamp_count": sum(1 for r in results if r["any_joint_limit_clamp_triggered"]),
        "scene_invalid_count": sum(1 for r in results if r.get("failure_reason") == "scene_invalid"),
    }
    for group in ("known_interior", "expanded_edge", "corner"):
        group_results = [r for r in results if r["group"] == group]
        summary[f"{group}_success_count"] = sum(1 for r in group_results if r["model_rollout_place_success"])
        summary[f"{group}_total"] = len(group_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"overall success_count: {summary['success_count']}/{summary['num_rollouts']}")
    print(f"grasp_count: {summary['grasp_count']}/{summary['num_rollouts']}")
    print(f"clamp_count: {summary['clamp_count']}  scene_invalid_count: {summary['scene_invalid_count']}")
    for group in ("known_interior", "expanded_edge", "corner"):
        print(f"  {group}: {summary[f'{group}_success_count']}/{summary[f'{group}_total']}")
    print(f"Results JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
