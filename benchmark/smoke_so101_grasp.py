"""SO-101 minimal grasp smoke test (see this task's chat report).

Verifies robot_sim/so101_pybullet_backend.So101PyBulletBackend's grasp
trigger: reset -> gripper open -> pre-grasp -> approach -> gripper
close -> grasp judged True + PyBullet fixed constraint created -> a few
settle steps confirming stable EE-object relative position -> reset ->
grasp state and constraint gone. Also runs a NEGATIVE case (close far
from the object) confirming no grasp is created.

No lift, no bin/place, no camera, no orientation IK, no expert-policy/
SmolVLA wiring, no ROS2. Reuses the same stepped-approach helper pattern
already established in benchmark/smoke_so101_object_approach.py (small,
per-axis-clamped command_end_effector_delta() calls) -- not duplicated
verbatim, but the same design, since this is a distinct smoke test file.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_grasp
"""

import argparse
import json
import math
from pathlib import Path

from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "grasp_smoke.json"

PRE_GRASP_OFFSET_M = [0.0, 0.0, 0.08]
APPROACH_OFFSET_M = [0.0, 0.0, 0.03]
FAR_OFFSET_M = [0.0, 0.0, 0.20]  # negative case: close 20cm above the object -- well beyond GRASP_DISTANCE_THRESHOLD_M (0.04)

MAX_STEP_M = 0.02
MAX_STEPS = 50
CONVERGENCE_TOLERANCE_M = 0.005
STEP_ERROR_FAILURE_THRESHOLD_M = 0.03

POST_GRASP_SETTLE_CHECKS = 5
POST_GRASP_SETTLE_STEPS_PER_CHECK = 20
RELATIVE_DRIFT_PASS_M = 0.002


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def move_to_target(backend: So101PyBulletBackend, target_position: list, stage_label: str, events: list) -> dict:
    for step_index in range(MAX_STEPS):
        current_ee_position, _ = backend.get_end_effector_pose()
        remaining = [target_position[i] - current_ee_position[i] for i in range(3)]
        remaining_norm = math.sqrt(sum(c ** 2 for c in remaining))
        if remaining_norm <= CONVERGENCE_TOLERANCE_M:
            break
        clamped_delta = [max(-MAX_STEP_M, min(MAX_STEP_M, c)) for c in remaining]
        obs = backend.command_end_effector_delta(clamped_delta)
        if not all_finite(obs["end_effector_position"]):
            raise RuntimeError(f"[{stage_label}] non-finite EE position at step {step_index}")
        if obs["ee_delta_position_error"] > STEP_ERROR_FAILURE_THRESHOLD_M:
            raise RuntimeError(f"[{stage_label}] abnormal IK step error {obs['ee_delta_position_error']:.4f}m at step {step_index}")
    final_ee_position, _ = backend.get_end_effector_pose()
    final_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))
    return {"target": target_position, "final_ee_position": final_ee_position, "error": final_error}


def relative_offset(ee_position: list, object_position: list) -> list:
    return [object_position[i] - ee_position[i] for i in range(3)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    events = []
    crashed = False
    crash_reason = None
    stage_results = {}
    joint_limit_violations = []

    backend = So101PyBulletBackend(gui=False)
    try:
        # === Positive case ===
        obs = backend.reset()
        object_position, _ = backend.get_object_pose()
        stage_results["object_initial_position"] = object_position

        backend.set_gripper(1.0)
        stage_results["pre_grasp"] = move_to_target(backend, [object_position[i] + PRE_GRASP_OFFSET_M[i] for i in range(3)], "pre_grasp", events)
        stage_results["approach"] = move_to_target(backend, [object_position[i] + APPROACH_OFFSET_M[i] for i in range(3)], "approach", events)

        ee_before_close, _ = backend.get_end_effector_pose()
        gripper_state_before_close = backend.get_observation()["gripper_position_normalized"]
        close_obs = backend.set_gripper(0.0)
        stage_results["gripper_state_at_trigger"] = close_obs["gripper_position_normalized"]

        grasp_state = backend.get_grasp_state()
        stage_results["grasp_state"] = grasp_state
        stage_results["grasp_distance"] = grasp_state["grasp_distance_at_trigger"]
        stage_results["is_grasped_after_close"] = backend.is_grasped()
        stage_results["constraint_id_valid"] = grasp_state["grasp_constraint_id"] is not None and grasp_state["grasp_constraint_id"] >= 0

        if not backend.is_grasped():
            events.append({"stage": "positive_case", "issue": "grasp was NOT created after approach + close (expected success)"})

        # Settle checks: relative EE-object offset should stay stable
        # while the constraint holds (object follows the EE, not the
        # other way around -- so we track OFFSET stability, not absolute
        # position, since nothing is commanding further motion here).
        ee_now, _ = backend.get_end_effector_pose()
        object_now = backend.get_object_position()
        initial_relative_offset = relative_offset(ee_now, object_now)
        max_drift = 0.0
        for check_index in range(POST_GRASP_SETTLE_CHECKS):
            backend.step(POST_GRASP_SETTLE_STEPS_PER_CHECK)
            ee_now, _ = backend.get_end_effector_pose()
            object_now = backend.get_object_position()
            if not (all_finite(ee_now) and all_finite(object_now)):
                raise RuntimeError(f"non-finite EE/object position during post-grasp settle check {check_index}")
            current_relative_offset = relative_offset(ee_now, object_now)
            drift = math.sqrt(sum((current_relative_offset[i] - initial_relative_offset[i]) ** 2 for i in range(3)))
            max_drift = max(max_drift, drift)
        stage_results["max_relative_drift_m"] = max_drift

        for name, pos in zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], backend.get_joint_positions()):
            info = backend.joint_info_by_name[name]
            if pos < info["lower"] - 1e-6 or pos > info["upper"] + 1e-6:
                joint_limit_violations.append({"stage": "positive_case", "joint": name, "position": pos})

        # Reset -- grasp state and constraint must be gone.
        backend.reset()
        stage_results["is_grasped_after_reset"] = backend.is_grasped()
        stage_results["constraint_id_after_reset"] = backend.get_grasp_state()["grasp_constraint_id"]

        # === Negative case: close far from the object ===
        object_position_2, _ = backend.get_object_pose()
        backend.set_gripper(1.0)
        far_result = move_to_target(backend, [object_position_2[i] + FAR_OFFSET_M[i] for i in range(3)], "negative_case_far", events)
        stage_results["negative_case_far_target"] = far_result
        far_ee_position, _ = backend.get_end_effector_pose()
        far_distance = math.sqrt(sum((far_ee_position[i] - object_position_2[i]) ** 2 for i in range(3)))
        stage_results["negative_case_distance"] = far_distance

        backend.set_gripper(0.0)
        stage_results["negative_case_is_grasped"] = backend.is_grasped()
        if backend.is_grasped():
            events.append({"stage": "negative_case", "issue": "grasp was created despite being far from the object (false positive)"})

    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    passed = (
        not crashed
        and not events
        and not joint_limit_violations
        and stage_results.get("is_grasped_after_close", False)
        and stage_results.get("constraint_id_valid", False)
        and stage_results.get("max_relative_drift_m", 999) <= RELATIVE_DRIFT_PASS_M
        and stage_results.get("is_grasped_after_reset", True) is False
        and stage_results.get("constraint_id_after_reset") is None
        and stage_results.get("negative_case_is_grasped", True) is False
    )

    result = {
        "crashed": crashed, "crash_reason": crash_reason,
        "stage_results": stage_results,
        "joint_limit_violations": joint_limit_violations,
        "numeric_issues": events,
        "all_passed": passed,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 grasp smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        print(f"object_initial_position: {stage_results['object_initial_position']}")
        print(f"pre_grasp error: {stage_results['pre_grasp']['error']:.4f}m")
        print(f"approach error: {stage_results['approach']['error']:.4f}m")
        print(f"gripper_state_at_trigger: {stage_results['gripper_state_at_trigger']:.4f}")
        print(f"grasp_distance: {stage_results['grasp_distance']}")
        print(f"is_grasped_after_close: {stage_results['is_grasped_after_close']}")
        print(f"constraint_id_valid: {stage_results['constraint_id_valid']}")
        print(f"max_relative_drift_m: {stage_results['max_relative_drift_m']:.5f}")
        print(f"is_grasped_after_reset: {stage_results['is_grasped_after_reset']}, constraint_id_after_reset: {stage_results['constraint_id_after_reset']}")
        print(f"negative_case_distance: {stage_results['negative_case_distance']:.4f}m, negative_case_is_grasped: {stage_results['negative_case_is_grasped']}")
    print(f"joint_limit_violations: {len(joint_limit_violations)}")
    print(f"numeric_issues: {events}")
    print(f"\n=== ALL PASSED: {passed} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
