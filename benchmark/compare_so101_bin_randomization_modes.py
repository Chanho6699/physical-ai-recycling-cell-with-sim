"""Compares the existing "coupled_small" bin randomization mode against
the new "fixed_bin_object_xy" mode (see this task's chat report,
"randomization 설계 변경"). Reads the existing 20-seed coupled
diagnostic (results/so101_bin_diagnostic_20seeds.json, NOT overwritten)
and the new 20-seed fixed-bin diagnostic
(results/so101_bin_independent_randomization_20seeds.json, produced by
`benchmark.benchmark_so101_bin_diagnostic --mode fixed_bin_object_xy`)
for success-rate/contact/waypoint-error comparisons, and additionally
re-runs BOTH modes' 20 seeds with a lightweight action-trajectory
recorder (reusing benchmark.so101_scripted_expert's own
run_pick_and_place_episode() on_step hook -- no dataset write, no
recorder/schema involved) to compute a directly comparable pairwise
action-trajectory RMS diversity metric, since the diagnostic JSONs only
store waypoint/summary-level fields, not full per-step trajectories.

Does NOT modify expert waypoints/success criterion/settle threshold.
Does NOT collect a dataset. Does NOT train anything.

Run:
  .venv-vla/bin/python -m benchmark.compare_so101_bin_randomization_modes
"""

import json
from pathlib import Path

import numpy as np

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    FIXED_BIN_OBJECT_X_RANGE,
    FIXED_BIN_OBJECT_Y_RANGE,
    RANDOMIZATION_MODE_COUPLED_SMALL,
    RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY,
)
from benchmark.evaluate_so101_expert_small_randomization import DEFAULT_X_RANGE, DEFAULT_Y_RANGE, sample_object_position
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, So101PyBulletBackend
from benchmark.so101_scripted_expert import run_pick_and_place_episode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUPLED_DIAGNOSTIC_PATH = PROJECT_ROOT / "results" / "so101_bin_diagnostic_20seeds.json"
FIXED_BIN_DIAGNOSTIC_PATH = PROJECT_ROOT / "results" / "so101_bin_independent_randomization_20seeds.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101_bin_randomization_comparison.json"
SEEDS = list(range(20))


def record_action_trajectory(mode: str, seed: int) -> np.ndarray:
    if mode == RANDOMIZATION_MODE_COUPLED_SMALL:
        sampled_object_position = sample_object_position(seed, DEFAULT_X_RANGE, DEFAULT_Y_RANGE)
        backend = So101PyBulletBackend(gui=False, use_bin=True, object_position=sampled_object_position)
    else:
        sampled_object_position = sample_object_position(seed, FIXED_BIN_OBJECT_X_RANGE, FIXED_BIN_OBJECT_Y_RANGE)
        nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
        fixed_bin_center_xy = [nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1]]
        backend = So101PyBulletBackend(
            gui=False, use_bin=True, object_position=sampled_object_position, bin_center_override_xy=fixed_bin_center_xy,
            scene_config={"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY},
        )
    trajectory = []
    try:
        backend.reset()
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])

        def on_step(phase, arm_joint_targets, gripper_target_normalized):
            trajectory.append(list(arm_joint_targets) + [gripper_target_normalized])

        run_pick_and_place_episode(backend, transport_delta_xy, on_step=on_step)
    finally:
        backend.close()
    return np.array(trajectory)


def pairwise_rms(trajectories: dict) -> dict:
    seeds = sorted(trajectories.keys())
    values = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = trajectories[seeds[i]], trajectories[seeds[j]]
            if a.shape == b.shape:
                values.append(float(np.sqrt(np.mean((a - b) ** 2))))
    return {
        "mean": float(np.mean(values)) if values else None, "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None, "count": len(values),
    }


def object_bin_offset_stats(records: list) -> dict:
    offsets = np.array([r["A_initial_scene"]["target_zone_offset_xy"] for r in records if "A_initial_scene" in r])
    return {
        "mean_xy": offsets.mean(axis=0).tolist(), "std_xy": offsets.std(axis=0).tolist(),
        "min_xy": offsets.min(axis=0).tolist(), "max_xy": offsets.max(axis=0).tolist(),
    }


def object_position_stats(records: list) -> dict:
    positions = np.array([r["A_initial_scene"]["object_initial_xyz"][:2] for r in records if "A_initial_scene" in r])
    return {"mean_xy": positions.mean(axis=0).tolist(), "std_xy": positions.std(axis=0).tolist(), "range_xy": (positions.max(axis=0) - positions.min(axis=0)).tolist()}


def main() -> None:
    coupled = json.loads(COUPLED_DIAGNOSTIC_PATH.read_text())
    fixed_bin = json.loads(FIXED_BIN_DIAGNOSTIC_PATH.read_text())

    print("Recording action trajectories for both modes (20 seeds each, no dataset write)...")
    coupled_trajectories = {seed: record_action_trajectory(RANDOMIZATION_MODE_COUPLED_SMALL, seed) for seed in SEEDS}
    fixed_bin_trajectories = {seed: record_action_trajectory(RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY, seed) for seed in SEEDS}

    comparison = {
        "coupled_small": {
            "source_file": str(COUPLED_DIAGNOSTIC_PATH),
            "production_place_success_rate": coupled["summary"]["production_place_success_rate"],
            "production_failure_reason_counts": coupled["summary"]["production_failure_reason_counts"],
            "scene_invalid_count": coupled["summary"]["scene_invalid_count"],
            "meaningful_contact_seeds": coupled["summary"]["meaningful_contact_seeds"],
            "stats_waypoint_error_m": coupled["summary"]["stats_waypoint_error_m"],
            "object_bin_offset_xy": object_bin_offset_stats(coupled["records"]),
            "object_position_xy": object_position_stats(coupled["records"]),
            "action_trajectory_pairwise_rms": pairwise_rms(coupled_trajectories),
        },
        "fixed_bin_object_xy": {
            "source_file": str(FIXED_BIN_DIAGNOSTIC_PATH),
            "production_place_success_rate": fixed_bin["summary"]["production_place_success_rate"],
            "production_failure_reason_counts": fixed_bin["summary"]["production_failure_reason_counts"],
            "scene_invalid_count": fixed_bin["summary"]["scene_invalid_count"],
            "meaningful_contact_seeds": fixed_bin["summary"]["meaningful_contact_seeds"],
            "stats_waypoint_error_m": fixed_bin["summary"]["stats_waypoint_error_m"],
            "object_bin_offset_xy": object_bin_offset_stats(fixed_bin["records"]),
            "object_position_xy": object_position_stats(fixed_bin["records"]),
            "action_trajectory_pairwise_rms": pairwise_rms(fixed_bin_trajectories),
        },
    }

    comparison["object_bin_offset_std_increased"] = (
        sum(comparison["fixed_bin_object_xy"]["object_bin_offset_xy"]["std_xy"])
        > sum(comparison["coupled_small"]["object_bin_offset_xy"]["std_xy"])
    )
    comparison["action_trajectory_diversity_increased"] = (
        comparison["fixed_bin_object_xy"]["action_trajectory_pairwise_rms"]["mean"]
        > comparison["coupled_small"]["action_trajectory_pairwise_rms"]["mean"]
    )
    comparison["success_rate_maintained"] = comparison["fixed_bin_object_xy"]["production_place_success_rate"] >= 0.90

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    print("=== SO-101 bin randomization mode comparison ===")
    print(f"coupled_small: success_rate={comparison['coupled_small']['production_place_success_rate']}, "
          f"offset_std_xy={comparison['coupled_small']['object_bin_offset_xy']['std_xy']}, "
          f"trajectory_rms_mean={comparison['coupled_small']['action_trajectory_pairwise_rms']['mean']:.5f}")
    print(f"fixed_bin_object_xy: success_rate={comparison['fixed_bin_object_xy']['production_place_success_rate']}, "
          f"offset_std_xy={comparison['fixed_bin_object_xy']['object_bin_offset_xy']['std_xy']}, "
          f"trajectory_rms_mean={comparison['fixed_bin_object_xy']['action_trajectory_pairwise_rms']['mean']:.5f}")
    print(f"object_bin_offset_std_increased: {comparison['object_bin_offset_std_increased']}")
    print(f"action_trajectory_diversity_increased: {comparison['action_trajectory_diversity_increased']}")
    print(f"success_rate_maintained (>=90%): {comparison['success_rate_maintained']}")
    print(f"\nComparison JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
