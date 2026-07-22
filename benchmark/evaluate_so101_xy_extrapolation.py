"""SO-101 object-XY extrapolation evaluation (see this task's chat
report, "25,000-step 공식 checkpoint의 XY 외삽 성능을 학습 없이 평가"). A
NO-TRAINING, NO-DATASET-GENERATION decision-support eval: does the
official 25000-step checkpoint's closed-loop success rate hold up if
the object's spawn XY is pushed outside the existing dataset's
FIXED_BIN_OBJECT_X_RANGE/FIXED_BIN_OBJECT_Y_RANGE (both currently
+/-0.015m -- see benchmark/benchmark_so101_bin_diagnostic.py)?

Reuses (does NOT reimplement): benchmark.so101_smolvla_rollout's own
load_policy_and_processors()/run_one_rollout() -- specifically this
task's own minimal addition to run_one_rollout()/build_rollout_backend(),
`object_position_override`, which lets a caller spawn the object at an
EXACT XY instead of a seed-sampled one, without touching anything else
(control loop, success criterion, action-chunk handling, noise-seed
mechanism are all byte-identical to the official 400-rollout baseline
eval this project has been using since the 10k-step checkpoint).

Fixed evaluation positions (25 total -- see this task's chat report for
the derivation): the shared center, the EXISTING range's own 4
boundary points + 4 corners (already-validated in-distribution
reference), the SAME 4 boundary points + 4 corners at a 25%-WIDER
radius (extrapolation), and the 4 boundary points + 4 corners at the
radius exactly BETWEEN the two (transition midpoints). Every position
gets its own `policy_noise_seed` per (position_index, repeat_id),
deriving from a SEPARATE base seed (200000) from the validation-seed
sweep's own base (100000) purely so the two experiments' derived seeds
never numerically collide/get confused with each other -- the
DERIVATION FORMULA itself is identical in spirit to
so101_smolvla_rollout.derive_policy_noise_seed().

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_xy_extrapolation
"""

import json
from pathlib import Path

import numpy as np

from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE
from benchmark.so101_smolvla_rollout import MAX_ROLLOUT_STEPS, load_policy_and_processors, run_one_rollout
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, InvalidSceneLayoutError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "so101_xy_extrapolation"
# The current official best checkpoint (see this task's chat report,
# "25,000-step 공식 checkpoint") -- deliberately NOT the module default
# so101_smolvla_rollout.CHECKPOINT_DIR re-exports (that constant still
# points at the old sanity-training checkpoint from an earlier phase of
# this project, kept there for backward compatibility with other
# scripts that import it).
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "train" / "all_20260720_114358_resume_25000" / "checkpoints" / "025000" / "pretrained_model"

OLD_RADIUS_M = FIXED_BIN_OBJECT_X_RANGE[1]  # 0.015 -- same for X and Y in this project's existing range
NEW_RADIUS_M = round(OLD_RADIUS_M * 1.25, 6)  # 25% wider, within this task's requested 20-30% band
MID_RADIUS_M = round((OLD_RADIUS_M + NEW_RADIUS_M) / 2, 6)  # transition point between existing and expanded range

NUM_REPEATS = 5
POLICY_NOISE_BASE_SEED = 200000  # deliberately separate from the validation-seed sweep's own base=100000
POLICY_NOISE_SEED_BLOCK_SIZE = 1000  # same convention as so101_smolvla_rollout.derive_policy_noise_seed()


def derive_xy_policy_noise_seed(position_index: int, repeat_id: int) -> int:
    if not (0 <= repeat_id < POLICY_NOISE_SEED_BLOCK_SIZE):
        raise ValueError(f"repeat_id={repeat_id} out of supported range [0, {POLICY_NOISE_SEED_BLOCK_SIZE})")
    return POLICY_NOISE_BASE_SEED + position_index * POLICY_NOISE_SEED_BLOCK_SIZE + repeat_id


def build_positions() -> list:
    """Returns a list of {"position_id", "category", "radius_m", "x_offset", "y_offset"} dicts, 25 total."""
    positions = [{"position_id": "center", "category": "center", "radius_m": 0.0, "x_offset": 0.0, "y_offset": 0.0}]

    for radius, category in [(OLD_RADIUS_M, "existing_range"), (NEW_RADIUS_M, "expanded_range"), (MID_RADIUS_M, "transition_midpoint")]:
        positions += [
            {"position_id": f"{category}_x_min", "category": category, "radius_m": radius, "x_offset": -radius, "y_offset": 0.0},
            {"position_id": f"{category}_x_max", "category": category, "radius_m": radius, "x_offset": radius, "y_offset": 0.0},
            {"position_id": f"{category}_y_min", "category": category, "radius_m": radius, "x_offset": 0.0, "y_offset": -radius},
            {"position_id": f"{category}_y_max", "category": category, "radius_m": radius, "x_offset": 0.0, "y_offset": radius},
            {"position_id": f"{category}_corner_pp", "category": category, "radius_m": radius, "x_offset": radius, "y_offset": radius},
            {"position_id": f"{category}_corner_pn", "category": category, "radius_m": radius, "x_offset": radius, "y_offset": -radius},
            {"position_id": f"{category}_corner_np", "category": category, "radius_m": radius, "x_offset": -radius, "y_offset": radius},
            {"position_id": f"{category}_corner_nn", "category": category, "radius_m": radius, "x_offset": -radius, "y_offset": -radius},
        ]
    return positions


def main() -> None:
    positions = build_positions()
    assert len(positions) == 25, f"expected 25 positions, got {len(positions)}"

    policy, preprocessor, postprocessor = load_policy_and_processors(CHECKPOINT_DIR)

    results = []
    for position_index, pos in enumerate(positions):
        object_position = [
            DEFAULT_OBJECT_POSITION[0] + pos["x_offset"],
            DEFAULT_OBJECT_POSITION[1] + pos["y_offset"],
            DEFAULT_OBJECT_POSITION[2],
        ]
        for repeat_id in range(NUM_REPEATS):
            derived_seed = derive_xy_policy_noise_seed(position_index, repeat_id)
            print(f"=== position={pos['position_id']} (x_offset={pos['x_offset']:+.5f} y_offset={pos['y_offset']:+.5f}) "
                  f"repeat={repeat_id} (derived_policy_seed={derived_seed}) ===")
            try:
                result = run_one_rollout(
                    policy, preprocessor, postprocessor, seed=position_index, max_steps=MAX_ROLLOUT_STEPS,
                    policy_noise_seed=derived_seed, object_position_override=object_position,
                )
                result["scene_invalid"] = False
                result["scene_invalid_reason"] = None
            except InvalidSceneLayoutError as exc:
                # Case D candidate (see this task's chat report, "물리적
                # 도달 불가능... workspace 제한으로 분류") -- a genuinely
                # invalid scene layout (e.g. object/bin overlap), NOT a
                # model-generalization failure. Recorded, not raised, so
                # the sweep continues past this one position.
                result = {
                    "seed": position_index, "steps_executed": 0, "aborted_early": True,
                    "failure_reason": f"scene_invalid:{exc.failure_type}", "grasp_was_ever_established": False,
                    "any_joint_limit_clamp_triggered": False, "joint_limit_clamp_count": 0,
                    "min_object_gripper_distance_m": None, "model_rollout_success_debug": None,
                    "model_rollout_place_success": False, "queue_debug_log": None, "diagnostic_log": None,
                    "scene_invalid": True, "scene_invalid_reason": exc.failure_type,
                }

            result["position_id"] = pos["position_id"]
            result["category"] = pos["category"]
            result["radius_m"] = pos["radius_m"]
            result["x_offset"] = pos["x_offset"]
            result["y_offset"] = pos["y_offset"]
            result["object_position"] = object_position
            result["repeat_id"] = repeat_id
            result["policy_noise_seed_used"] = derived_seed
            results.append(result)
            print(f"  place_success={result['model_rollout_place_success']} "
                  f"grasp_ever={result['grasp_was_ever_established']} "
                  f"min_dist={result['min_object_gripper_distance_m']} "
                  f"failure_reason={result['failure_reason']} scene_invalid={result['scene_invalid']}")

    summary = {
        "checkpoint": str(CHECKPOINT_DIR),
        "old_radius_m": OLD_RADIUS_M, "new_radius_m": NEW_RADIUS_M, "mid_radius_m": MID_RADIUS_M,
        "num_repeats": NUM_REPEATS, "policy_noise_base_seed": POLICY_NOISE_BASE_SEED,
        "num_positions": len(positions), "num_rollouts": len(results),
        "results": results,
        "success_count": sum(1 for r in results if r["model_rollout_place_success"]),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "xy_extrapolation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"success_count: {summary['success_count']}/{len(results)}")
    print(f"Results JSON: {output_path}")


if __name__ == "__main__":
    main()
