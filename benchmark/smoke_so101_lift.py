"""SO-101 minimal lift smoke test (see this task's chat report).

Verifies robot_sim/so101_pybullet_backend.So101PyBulletBackend's grasp
holds through a stepped VERTICAL-ONLY lift: reset -> gripper open ->
pre-grasp -> approach -> gripper close -> confirmed grasp -> record
grasp-time EE/object z -> repeated small +dz command_end_effector_delta()
calls (dx=dy=0 every step) -> final displacement/clearance/drift
judged against grasp-time positions, NEVER a hardcoded absolute z.

No backend changes were needed for this task -- get_end_effector_pose(),
get_object_position()/get_object_pose(), is_grasped(), get_grasp_state(),
and get_scene_state() (for surface_height) already covered everything
lift verification needs. The lift stepping loop itself lives here, in
this smoke test file, not as a new backend method (same "caller drives
via existing command_end_effector_delta()" pattern already used in
benchmark/smoke_so101_object_approach.py and smoke_so101_grasp.py, kept
independent per this task's "대규모 공통화 리팩터링은 하지 않는다").

No lift-then-bin-move, no place/release, no camera, no orientation IK,
no expert-policy/SmolVLA wiring, no ROS2.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_lift
"""

import argparse
import json
import math
from pathlib import Path

import pybullet as p

from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "lift_smoke.json"

PRE_GRASP_OFFSET_M = [0.0, 0.0, 0.08]
APPROACH_OFFSET_M = [0.0, 0.0, 0.03]
FAR_OFFSET_M = [0.0, 0.0, 0.20]  # negative case: gripper closes far above the object -- no grasp expected

MAX_STEP_M = 0.02               # per-command clamp, reused for approach AND lift steps
CONVERGENCE_TOLERANCE_M = 0.005
STEP_ERROR_FAILURE_THRESHOLD_M = 0.03
MAX_MOVE_STEPS = 50

LIFT_DISTANCE_M = 0.08
LIFT_STEP_DZ_M = 0.015          # within the recommended 0.01-0.02m band
LIFT_MAX_STEPS = 20

VERTICAL_DISPLACEMENT_PASS_M = 0.01     # both EE and object final vertical displacement must be within this of LIFT_DISTANCE_M
MIN_CLEARANCE_FROM_SURFACE_M = 0.03
RELATIVE_DRIFT_PASS_M = 0.002
LATERAL_DRIFT_PASS_M = 0.01              # not given an exact number in the task -- chosen conservatively, stated explicitly here
NEGATIVE_CASE_OBJECT_DISPLACEMENT_PASS_M = 0.002

JOINT_LIMIT_EPS = 1e-6
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def object_offset_in_ee_frame(ee_position: list, ee_orientation: list, object_position: list) -> list:
    """Object position expressed in the EE link's OWN local frame --
    NOT a plain world-frame subtraction. A fixed constraint (see
    So101PyBulletBackend._maybe_trigger_grasp()) holds the object rigid
    in exactly this frame, so if the grasp is genuinely solid, this
    offset stays constant EVEN WHILE THE EE ROTATES during the lift
    (position-only IK does not hold orientation fixed -- verified this
    changes noticeably step-to-step during a real lift). A world-frame
    offset difference would (incorrectly) register that normal,
    harmless wrist rotation as 'drift' -- this is exactly the bug this
    task's own regression check caught before landing."""
    ee_pos_inv, ee_orn_inv = p.invertTransform(ee_position, ee_orientation)
    local_position, _local_orientation = p.multiplyTransforms(ee_pos_inv, ee_orn_inv, object_position, [0, 0, 0, 1])
    return list(local_position)


def check_joint_limits(backend: So101PyBulletBackend, joint_positions: list) -> list:
    violations = []
    for name, pos in zip(ARM_JOINT_NAMES, joint_positions):
        info = backend.joint_info_by_name[name]
        if pos < info["lower"] - JOINT_LIMIT_EPS or pos > info["upper"] + JOINT_LIMIT_EPS:
            violations.append({"joint": name, "position": pos, "lower": info["lower"], "upper": info["upper"]})
    return violations


def move_to_target(backend: So101PyBulletBackend, target_position: list, stage_label: str) -> dict:
    for step_index in range(MAX_MOVE_STEPS):
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
        violations = check_joint_limits(backend, obs["joint_positions"])
        if violations:
            raise RuntimeError(f"[{stage_label}] joint limit violation at step {step_index}: {violations}")
    final_ee_position, _ = backend.get_end_effector_pose()
    final_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))
    return {"target": target_position, "final_ee_position": final_ee_position, "error": final_error}


def run_lift(backend: So101PyBulletBackend, lift_distance_m: float, require_grasp_maintained: bool) -> dict:
    """Repeated, vertical-ONLY (dx=dy=0) small +dz command_end_effector_delta()
    calls -- target z is grasp_ee_z + lift_distance_m, never a hardcoded
    absolute value (see this task's chat report, item 2)."""
    ee_start, ee_orientation_start = backend.get_end_effector_pose()
    target_ee_z = ee_start[2] + lift_distance_m

    grasp_constraint_id_start = backend.get_grasp_state()["grasp_constraint_id"]
    max_relative_drift = 0.0
    max_lateral_drift = 0.0
    grasp_maintained_all_steps = True
    constraint_valid_all_steps = True

    object_position_start = backend.get_object_position()
    # EE-LOCAL-FRAME offset (see object_offset_in_ee_frame()'s own
    # docstring for why this, not a world-frame subtraction, is the
    # correct "did the grasp actually slip" measure).
    initial_relative_offset = object_offset_in_ee_frame(ee_start, ee_orientation_start, object_position_start)

    remaining_z = target_ee_z - ee_start[2]
    step_index = 0
    while remaining_z > CONVERGENCE_TOLERANCE_M and step_index < LIFT_MAX_STEPS:
        dz = min(LIFT_STEP_DZ_M, remaining_z)
        obs = backend.command_end_effector_delta([0.0, 0.0, dz])
        if not all_finite(obs["end_effector_position"]):
            raise RuntimeError(f"[lift] non-finite EE position at step {step_index}")
        if obs["ee_delta_position_error"] > STEP_ERROR_FAILURE_THRESHOLD_M:
            raise RuntimeError(f"[lift] abnormal IK step error {obs['ee_delta_position_error']:.4f}m at step {step_index} (immediate failure, no retry)")
        violations = check_joint_limits(backend, obs["joint_positions"])
        if violations:
            raise RuntimeError(f"[lift] joint limit violation at step {step_index}: {violations}")

        object_position_now = backend.get_object_position()
        if not all_finite(object_position_now):
            raise RuntimeError(f"[lift] non-finite object position at step {step_index}")

        is_grasped_now = backend.is_grasped()
        if require_grasp_maintained and not is_grasped_now:
            grasp_maintained_all_steps = False
        constraint_id_now = backend.get_grasp_state()["grasp_constraint_id"]
        if require_grasp_maintained and constraint_id_now != grasp_constraint_id_start:
            constraint_valid_all_steps = False

        ee_now = obs["end_effector_position"]
        _ee_now_unused, ee_orientation_now = backend.get_end_effector_pose()
        current_relative_offset = object_offset_in_ee_frame(ee_now, ee_orientation_now, object_position_now)
        drift = math.sqrt(sum((current_relative_offset[i] - initial_relative_offset[i]) ** 2 for i in range(3)))
        max_relative_drift = max(max_relative_drift, drift)

        lateral_drift = math.sqrt((ee_now[0] - ee_start[0]) ** 2 + (ee_now[1] - ee_start[1]) ** 2)
        max_lateral_drift = max(max_lateral_drift, lateral_drift)

        remaining_z = target_ee_z - ee_now[2]
        step_index += 1

    ee_final, _ = backend.get_end_effector_pose()
    object_position_final = backend.get_object_position()

    return {
        "ee_start_position": ee_start, "ee_final_position": ee_final,
        "object_start_position": object_position_start, "object_final_position": object_position_final,
        "ee_vertical_displacement_m": ee_final[2] - ee_start[2],
        "object_vertical_displacement_m": object_position_final[2] - object_position_start[2],
        "max_relative_drift_m": max_relative_drift, "max_lateral_drift_m": max_lateral_drift,
        "grasp_maintained_all_steps": grasp_maintained_all_steps,
        "constraint_valid_all_steps": constraint_valid_all_steps,
        "num_steps": step_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    crashed = False
    crash_reason = None
    results = {}

    backend = So101PyBulletBackend(gui=False)
    try:
        # === Positive case: grasp then lift ===
        backend.reset()
        object_position, _ = backend.get_object_pose()
        scene = backend.get_scene_state()
        surface_height = scene["table_top_z"]
        object_half_height = backend.scene_config["object_height"] / 2.0

        backend.set_gripper(1.0)
        move_to_target(backend, [object_position[i] + PRE_GRASP_OFFSET_M[i] for i in range(3)], "pre_grasp")
        move_to_target(backend, [object_position[i] + APPROACH_OFFSET_M[i] for i in range(3)], "approach")
        backend.set_gripper(0.0)

        if not backend.is_grasped():
            raise RuntimeError("grasp was not established before lift -- cannot proceed")

        lift_result = run_lift(backend, LIFT_DISTANCE_M, require_grasp_maintained=True)
        results["lift"] = lift_result

        object_bottom_z = lift_result["object_final_position"][2] - object_half_height
        clearance = object_bottom_z - surface_height
        results["object_clearance_from_surface_m"] = clearance
        results["surface_height"] = surface_height
        results["object_half_height"] = object_half_height

        results["is_grasped_after_lift"] = backend.is_grasped()
        results["grasp_state_after_lift"] = backend.get_grasp_state()

        ee_disp_ok = abs(lift_result["ee_vertical_displacement_m"] - LIFT_DISTANCE_M) <= VERTICAL_DISPLACEMENT_PASS_M
        object_disp_ok = abs(lift_result["object_vertical_displacement_m"] - LIFT_DISTANCE_M) <= VERTICAL_DISPLACEMENT_PASS_M
        clearance_ok = clearance >= MIN_CLEARANCE_FROM_SURFACE_M
        drift_ok = lift_result["max_relative_drift_m"] <= RELATIVE_DRIFT_PASS_M
        lateral_ok = lift_result["max_lateral_drift_m"] <= LATERAL_DRIFT_PASS_M

        results["ee_displacement_pass"] = ee_disp_ok
        results["object_displacement_pass"] = object_disp_ok
        results["clearance_pass"] = clearance_ok
        results["relative_drift_pass"] = drift_ok
        results["lateral_drift_pass"] = lateral_ok

        # Reset cleanup check
        backend.reset()
        results["reset_cleanup_pass"] = (not backend.is_grasped()) and (backend.get_grasp_state()["grasp_constraint_id"] is None)

        # === Negative case: no grasp established, lift anyway ===
        object_position_2, _ = backend.get_object_pose()
        backend.set_gripper(1.0)
        move_to_target(backend, [object_position_2[i] + FAR_OFFSET_M[i] for i in range(3)], "negative_case_far")
        backend.set_gripper(0.0)  # far from object -- must NOT grasp
        negative_case_grasped_after_close = backend.is_grasped()

        negative_lift_result = run_lift(backend, LIFT_DISTANCE_M, require_grasp_maintained=False)
        results["negative_case"] = negative_lift_result
        results["negative_case_grasped_after_close"] = negative_case_grasped_after_close
        results["negative_case_object_displacement_m"] = abs(negative_lift_result["object_vertical_displacement_m"])
        results["negative_case_grasped_during_lift"] = backend.is_grasped()
        results["negative_case_pass"] = (
            not negative_case_grasped_after_close
            and not results["negative_case_grasped_during_lift"]
            and results["negative_case_object_displacement_m"] <= NEGATIVE_CASE_OBJECT_DISPLACEMENT_PASS_M
        )

        finite_values_pass = all(
            all_finite(v) for v in [
                lift_result["ee_start_position"], lift_result["ee_final_position"],
                lift_result["object_start_position"], lift_result["object_final_position"],
            ]
        )
        results["finite_values_pass"] = finite_values_pass
        results["joint_limits_pass"] = True  # move_to_target/run_lift raise immediately on any violation -- reaching here means none occurred

    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    overall_pass = (
        not crashed
        and results.get("is_grasped_after_lift", False)
        and results.get("lift", {}).get("grasp_maintained_all_steps", False)
        and results.get("lift", {}).get("constraint_valid_all_steps", False)
        and results.get("ee_displacement_pass", False)
        and results.get("object_displacement_pass", False)
        and results.get("clearance_pass", False)
        and results.get("relative_drift_pass", False)
        and results.get("lateral_drift_pass", False)
        and results.get("reset_cleanup_pass", False)
        and results.get("negative_case_pass", False)
        and results.get("finite_values_pass", False)
        and results.get("joint_limits_pass", False)
    )

    output = {
        "crashed": crashed, "crash_reason": crash_reason,
        "lift_distance_commanded_m": LIFT_DISTANCE_M,
        "ee_start_position": results.get("lift", {}).get("ee_start_position"),
        "ee_final_position": results.get("lift", {}).get("ee_final_position"),
        "ee_vertical_displacement_m": results.get("lift", {}).get("ee_vertical_displacement_m"),
        "object_start_position": results.get("lift", {}).get("object_start_position"),
        "object_final_position": results.get("lift", {}).get("object_final_position"),
        "object_vertical_displacement_m": results.get("lift", {}).get("object_vertical_displacement_m"),
        "object_clearance_from_surface_m": results.get("object_clearance_from_surface_m"),
        "max_object_ee_relative_drift_m": results.get("lift", {}).get("max_relative_drift_m"),
        "max_ee_lateral_drift_m": results.get("lift", {}).get("max_lateral_drift_m"),
        "grasp_maintained_all_steps": results.get("lift", {}).get("grasp_maintained_all_steps"),
        "constraint_valid_all_steps": results.get("lift", {}).get("constraint_valid_all_steps"),
        "negative_case_object_displacement_m": results.get("negative_case_object_displacement_m"),
        "negative_case_grasped": results.get("negative_case_grasped_after_close"),
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

    print("=== SO-101 lift smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        print(f"lift_distance_commanded_m: {LIFT_DISTANCE_M}")
        print(f"ee_vertical_displacement_m: {output['ee_vertical_displacement_m']:.4f}")
        print(f"object_vertical_displacement_m: {output['object_vertical_displacement_m']:.4f}")
        print(f"object_clearance_from_surface_m: {output['object_clearance_from_surface_m']:.4f}")
        print(f"grasp_maintained_all_steps: {output['grasp_maintained_all_steps']}")
        print(f"constraint_valid_all_steps: {output['constraint_valid_all_steps']}")
        print(f"max_object_ee_relative_drift_m: {output['max_object_ee_relative_drift_m']:.5f}")
        print(f"max_ee_lateral_drift_m: {output['max_ee_lateral_drift_m']:.5f}")
        print(f"negative_case_object_displacement_m: {output['negative_case_object_displacement_m']:.5f}")
        print(f"negative_case_grasped: {output['negative_case_grasped']}")
        print(f"reset_cleanup_pass: {output['reset_cleanup_pass']}")
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
