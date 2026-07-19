"""SO-101 full pick-and-place episode smoke test (see this task's chat
report). Completes the episode this session's SO-101 work has been
building toward:

  approach -> grasp -> lift -> transport -> release -> settle

Reuses robot_sim/so101_pybullet_backend.So101PyBulletBackend's existing
interface. As of this task ("공통 Expert 모듈 정리"), the actual phase
sequence, waypoint/threshold constants, IK target computation, gripper
open/close, and success judgment all live in
benchmark/so101_scripted_expert.py::run_pick_and_place_episode() --
this file no longer reimplements any of that (previously duplicated
with benchmark/collect_so101_episode.py); it only calls the shared
Expert and does its own JSON-report bookkeeping.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_pick_and_place
"""

import argparse
import json
import math
from pathlib import Path

from benchmark.so101_scripted_expert import (
    LIFT_DISTANCE_M,
    So101ExpertError,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "pick_and_place_smoke.json"

# Positive case's transport delta MUST match scene_config's own
# target_zone_offset_xy default ([0.05, 0.05]) so the object actually
# lands in the target zone the backend built -- this is the one place
# that assumption is stated explicitly, not silently assumed.
TRANSPORT_DELTA_XY_GOOD = [0.05, 0.05]
# Negative case: deliberately released outside the (fixed) target zone
# -- mirrored x-direction from the good case (same magnitude as the
# already-validated transport delta, so this stays a reachability-safe
# move, not an unreachable one). Distance from the target zone center
# this produces (~0.10m) is well beyond TARGET_XY_ERROR_PASS_M (0.03m).
TRANSPORT_DELTA_XY_BAD = [-0.05, 0.05]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def run_episode(backend: So101PyBulletBackend, transport_delta_xy: list) -> dict:
    return run_pick_and_place_episode(backend, transport_delta_xy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    crashed = False
    crash_reason = None
    results = {}

    backend = So101PyBulletBackend(gui=False)
    try:
        # === Positive case: full episode, transport INTO the target zone ===
        backend.reset()
        episode = run_episode(backend, TRANSPORT_DELTA_XY_GOOD)
        results["episode"] = episode

        # Reset cleanup check
        backend.reset()
        results["reset_cleanup_pass"] = (not backend.is_grasped()) and (backend.get_grasp_state()["grasp_constraint_id"] is None)

        # === Negative case: full episode, transport AWAY FROM the target zone ===
        negative_episode = run_episode(backend, TRANSPORT_DELTA_XY_BAD)
        results["negative_episode"] = negative_episode
        results["negative_case_place_success"] = negative_episode["place_success"]
        # The negative case's OWN success is that place_success is
        # correctly False despite the object settling normally (not
        # crashing/exploding) -- a True here would be the exact false-
        # positive this task's negative case exists to catch.
        results["negative_case_pass"] = (negative_episode["place_success"] is False)

        results["finite_values_pass"] = (
            all_finite(episode["object_final_position"])
            and all_finite(episode["object_final_orientation"])
            and all_finite(negative_episode["object_final_position"])
        )
        results["joint_limits_pass"] = True  # run_pick_and_place_episode() raises immediately on any violation -- reaching here means none occurred

    except (Exception, So101ExpertError) as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    episode = results.get("episode", {})
    overall_pass = (
        not crashed
        and episode.get("lift", {}).get("grasp_maintained_all_steps", False)
        and episode.get("lift", {}).get("constraint_valid_all_steps", False)
        and episode.get("transport", {}).get("grasp_maintained_all_steps", False)
        and episode.get("transport", {}).get("constraint_valid_all_steps", False)
        and episode.get("place_descend", {}).get("grasp_maintained_all_steps", False)
        and episode.get("place_descend", {}).get("constraint_valid_all_steps", False)
        and episode.get("release_constraint_removed", False)
        and episode.get("grasp_state_after_release", {}).get("is_grasped") is False
        and episode.get("object_in_target_zone", False)
        and episode.get("resting_height_ok", False)
        and episode.get("object_stably_settled", False)
        and episode.get("place_success", False)
        and results.get("reset_cleanup_pass", False)
        and results.get("negative_case_pass", False)
        and results.get("finite_values_pass", False)
        and results.get("joint_limits_pass", False)
    )

    output = {
        "crashed": crashed, "crash_reason": crash_reason,
        "initial_object_position": episode.get("initial_object_position"),
        "grasp_position": episode.get("grasp_position"),
        "lift_distance_commanded_m": LIFT_DISTANCE_M,
        "ee_transport_target_position": episode.get("transport", {}).get("target") if isinstance(episode.get("transport"), dict) else None,
        "object_release_position": episode.get("object_release_position"),
        "target_center_position": episode.get("target_center_position"),
        "release_constraint_removed": episode.get("release_constraint_removed"),
        "grasp_state_after_release": episode.get("grasp_state_after_release"),
        "object_final_position": episode.get("object_final_position"),
        "object_final_orientation": episode.get("object_final_orientation"),
        "object_target_xy_error_m": episode.get("object_target_xy_error_m"),
        "object_resting_height_error_m": episode.get("object_resting_height_error_m"),
        "object_final_linear_speed_mps": episode.get("object_final_linear_speed_mps"),
        "object_final_angular_speed_radps": episode.get("object_final_angular_speed_radps"),
        "object_recent_settle_drift_m": episode.get("object_recent_settle_drift_m"),
        "object_in_target_zone": episode.get("object_in_target_zone"),
        "object_stably_settled": episode.get("object_stably_settled"),
        "place_success": episode.get("place_success"),
        "failure_reason": episode.get("failure_reason"),
        "negative_case_place_success": results.get("negative_case_place_success"),
        "reset_cleanup_pass": results.get("reset_cleanup_pass"),
        "finite_values_pass": results.get("finite_values_pass"),
        "joint_limits_pass": results.get("joint_limits_pass"),
        "overall_pass": overall_pass,
        "full_results": results,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 full pick-and-place episode smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        print(f"initial_object_position: {output['initial_object_position']}")
        print(f"target_center_position: {output['target_center_position']}")
        print(f"object_release_position: {output['object_release_position']}")
        print(f"release_constraint_removed: {output['release_constraint_removed']}")
        print(f"object_final_position: {output['object_final_position']}")
        print(f"object_target_xy_error_m: {output['object_target_xy_error_m']:.4f}")
        print(f"object_resting_height_error_m: {output['object_resting_height_error_m']:.4f}")
        print(f"object_final_linear_speed_mps: {output['object_final_linear_speed_mps']:.5f}")
        print(f"object_final_angular_speed_radps: {output['object_final_angular_speed_radps']:.5f}")
        print(f"object_recent_settle_drift_m: {output['object_recent_settle_drift_m']:.5f}")
        print(f"object_in_target_zone: {output['object_in_target_zone']}, object_stably_settled: {output['object_stably_settled']}")
        print(f"place_success: {output['place_success']} (failure_reason: {output['failure_reason']})")
        print(f"negative_case_place_success: {output['negative_case_place_success']} (must be False)")
        print(f"reset_cleanup_pass: {output['reset_cleanup_pass']}")
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
