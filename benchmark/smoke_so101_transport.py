"""SO-101 minimal transport smoke test (see this task's chat report).

Verifies robot_sim/so101_pybullet_backend.So101PyBulletBackend's grasp
survives a LATERAL move after a completed lift: reset -> gripper open
-> pre-grasp -> approach -> gripper close -> confirmed grasp -> vertical
lift (0.08m) -> lateral transport target (lift-completion EE position +
[dx, dy, 0], NEVER a hardcoded absolute coordinate) -> stepped transport
-> final displacement/drift/height-hold judged against transport-start
positions.

No backend changes were needed for this task (same interface already
covered lift: get_end_effector_pose(), get_object_position()/
get_object_pose(), is_grasped(), get_grasp_state(), get_scene_state()).
The lift+transport stepping loop lives here, in ONE unified helper
(move_with_grasp_tracking()) used for both phases, rather than copying
benchmark/smoke_so101_lift.py's own loop verbatim into a second file.

Drift is measured in the EE's LOCAL frame (p.invertTransform/
multiplyTransforms), matching the frame the grasp constraint itself is
defined in -- NOT a world-frame position subtraction (see
smoke_so101_lift.py's own chat report for why a world-frame subtraction
falsely flagged normal wrist-orientation drift as "slippage").

No release/place, no bin, no camera, no orientation IK, no expert-
policy/SmolVLA wiring, no ROS2.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_transport
"""

import argparse
import json
import math
from pathlib import Path

import pybullet as p

from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "transport_smoke.json"

PRE_GRASP_OFFSET_M = [0.0, 0.0, 0.08]
APPROACH_OFFSET_M = [0.0, 0.0, 0.03]
FAR_OFFSET_M = [0.0, 0.0, 0.20]
LIFT_DISTANCE_M = 0.08
LIFT_STEP_M = 0.015

# Transport target = lift-completion EE position + this delta -- a single,
# named test constant, never repeated/hardcoded elsewhere in this file.
TRANSPORT_DELTA_XY = [0.05, 0.05]

MAX_STEP_M = 0.02
CONVERGENCE_TOLERANCE_M = 0.005
STEP_ERROR_FAILURE_THRESHOLD_M = 0.03
MAX_MOVE_STEPS = 50
LIFT_MAX_STEPS = 20
TRANSPORT_MAX_STEPS = 30

EE_POSITION_ERROR_PASS_M = 0.01
OBJECT_EE_LATERAL_MATCH_PASS_M = 0.01
RELATIVE_DRIFT_PASS_M = 0.002
EE_VERTICAL_DEVIATION_PASS_M = 0.01
MIN_CLEARANCE_FROM_SURFACE_M = 0.03
NEGATIVE_CASE_OBJECT_DISPLACEMENT_PASS_M = 0.002

JOINT_LIMIT_EPS = 1e-6
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def object_offset_in_ee_frame(ee_position: list, ee_orientation: list, object_position: list) -> list:
    """Object position in the EE link's OWN local frame -- the frame the
    grasp constraint is actually defined in, so this stays constant for a
    genuinely rigid grasp regardless of EE orientation changes (position-
    only IK does not hold orientation fixed). See this module's own
    docstring / smoke_so101_lift.py's chat report for why a world-frame
    subtraction is the wrong metric here."""
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


def move_to_target(backend: So101PyBulletBackend, target_position: list, stage_label: str, max_steps: int = MAX_MOVE_STEPS) -> dict:
    """Plain stepped move, no grasp tracking -- used for pre_grasp/
    approach/negative-case movement, where there either is no grasp yet
    or grasp state isn't being judged."""
    for step_index in range(max_steps):
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


def move_with_grasp_tracking(backend: So101PyBulletBackend, target_position: list, stage_label: str, max_steps: int, track_grasp: bool) -> dict:
    """Unified stepper used for BOTH the vertical lift AND the lateral
    transport phases (see this module's own docstring) -- per-axis-
    clamped remaining-vector steps via the EXISTING
    command_end_effector_delta(), immediate failure (no retry, no
    tolerance relaxation) on a non-finite result, an abnormally large
    single-step error, or a joint-limit violation."""
    ee_start, ee_orientation_start = backend.get_end_effector_pose()
    object_start = backend.get_object_position()
    grasp_constraint_id_start = backend.get_grasp_state()["grasp_constraint_id"]
    initial_relative_offset = object_offset_in_ee_frame(ee_start, ee_orientation_start, object_start) if track_grasp else None

    max_relative_drift = 0.0
    max_vertical_deviation = 0.0
    grasp_maintained_all_steps = True
    constraint_valid_all_steps = True

    step_index = 0
    while step_index < max_steps:
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
            raise RuntimeError(f"[{stage_label}] abnormal IK step error {obs['ee_delta_position_error']:.4f}m at step {step_index} (immediate failure, no retry)")
        violations = check_joint_limits(backend, obs["joint_positions"])
        if violations:
            raise RuntimeError(f"[{stage_label}] joint limit violation at step {step_index}: {violations}")

        object_now = backend.get_object_position()
        if not all_finite(object_now):
            raise RuntimeError(f"[{stage_label}] non-finite object position at step {step_index}")

        if track_grasp:
            if not backend.is_grasped():
                grasp_maintained_all_steps = False
            constraint_id_now = backend.get_grasp_state()["grasp_constraint_id"]
            if constraint_id_now != grasp_constraint_id_start:
                constraint_valid_all_steps = False

            ee_now = obs["end_effector_position"]
            _ee_now_unused, ee_orientation_now = backend.get_end_effector_pose()
            current_relative_offset = object_offset_in_ee_frame(ee_now, ee_orientation_now, object_now)
            drift = math.sqrt(sum((current_relative_offset[i] - initial_relative_offset[i]) ** 2 for i in range(3)))
            max_relative_drift = max(max_relative_drift, drift)

        vertical_deviation = abs(obs["end_effector_position"][2] - ee_start[2])
        max_vertical_deviation = max(max_vertical_deviation, vertical_deviation)
        step_index += 1

    ee_final, _ = backend.get_end_effector_pose()
    object_final = backend.get_object_position()

    return {
        "ee_start_position": ee_start, "ee_final_position": ee_final,
        "object_start_position": object_start, "object_final_position": object_final,
        "target_position": target_position,
        "ee_position_error_m": math.sqrt(sum((ee_final[i] - target_position[i]) ** 2 for i in range(3))),
        "max_relative_drift_m": max_relative_drift, "max_vertical_deviation_m": max_vertical_deviation,
        "grasp_maintained_all_steps": grasp_maintained_all_steps if track_grasp else None,
        "constraint_valid_all_steps": constraint_valid_all_steps if track_grasp else None,
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
        # === Positive case: grasp -> lift -> transport ===
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

        ee_pre_lift, _ = backend.get_end_effector_pose()
        lift_target = [ee_pre_lift[0], ee_pre_lift[1], ee_pre_lift[2] + LIFT_DISTANCE_M]
        lift_result = move_with_grasp_tracking(backend, lift_target, "lift", LIFT_MAX_STEPS, track_grasp=True)
        results["lift"] = lift_result
        if not backend.is_grasped():
            raise RuntimeError("grasp was lost during lift -- cannot proceed to transport")

        # Transport target: lift-completion EE position + [dx, dy, 0] --
        # never a hardcoded absolute coordinate (see this task's chat report).
        ee_lift_final = lift_result["ee_final_position"]
        transport_target = [ee_lift_final[0] + TRANSPORT_DELTA_XY[0], ee_lift_final[1] + TRANSPORT_DELTA_XY[1], ee_lift_final[2]]
        transport_result = move_with_grasp_tracking(backend, transport_target, "transport", TRANSPORT_MAX_STEPS, track_grasp=True)
        results["transport"] = transport_result

        ee_start = transport_result["ee_start_position"]
        ee_final = transport_result["ee_final_position"]
        object_start = transport_result["object_start_position"]
        object_final = transport_result["object_final_position"]

        ee_lateral_displacement = math.sqrt((ee_final[0] - ee_start[0]) ** 2 + (ee_final[1] - ee_start[1]) ** 2)
        object_lateral_displacement = math.sqrt((object_final[0] - object_start[0]) ** 2 + (object_final[1] - object_start[1]) ** 2)
        object_bottom_z = object_final[2] - object_half_height
        clearance = object_bottom_z - surface_height

        results["ee_lateral_displacement_m"] = ee_lateral_displacement
        results["object_lateral_displacement_m"] = object_lateral_displacement
        results["object_surface_clearance_m"] = clearance
        results["surface_height"] = surface_height

        ee_error_ok = transport_result["ee_position_error_m"] <= EE_POSITION_ERROR_PASS_M
        lateral_match_ok = abs(object_lateral_displacement - ee_lateral_displacement) <= OBJECT_EE_LATERAL_MATCH_PASS_M
        drift_ok = transport_result["max_relative_drift_m"] <= RELATIVE_DRIFT_PASS_M
        vertical_ok = transport_result["max_vertical_deviation_m"] <= EE_VERTICAL_DEVIATION_PASS_M
        clearance_ok = clearance >= MIN_CLEARANCE_FROM_SURFACE_M

        results["ee_error_pass"] = ee_error_ok
        results["lateral_match_pass"] = lateral_match_ok
        results["relative_drift_pass"] = drift_ok
        results["vertical_deviation_pass"] = vertical_ok
        results["clearance_pass"] = clearance_ok
        results["is_grasped_after_transport"] = backend.is_grasped()

        # Reset cleanup check
        backend.reset()
        results["reset_cleanup_pass"] = (not backend.is_grasped()) and (backend.get_grasp_state()["grasp_constraint_id"] is None)

        # === Negative case: no grasp, EE-only transport along the SAME path ===
        object_position_2, _ = backend.get_object_pose()
        backend.set_gripper(1.0)
        move_to_target(backend, [object_position_2[i] + FAR_OFFSET_M[i] for i in range(3)], "negative_case_far")
        backend.set_gripper(0.0)  # far from object -- must NOT grasp
        negative_case_grasped_after_close = backend.is_grasped()

        ee_pre_lift_neg, _ = backend.get_end_effector_pose()
        neg_lift_target = [ee_pre_lift_neg[0], ee_pre_lift_neg[1], ee_pre_lift_neg[2] + LIFT_DISTANCE_M]
        move_with_grasp_tracking(backend, neg_lift_target, "negative_case_lift", LIFT_MAX_STEPS, track_grasp=False)
        ee_lift_final_neg, _ = backend.get_end_effector_pose()
        neg_transport_target = [ee_lift_final_neg[0] + TRANSPORT_DELTA_XY[0], ee_lift_final_neg[1] + TRANSPORT_DELTA_XY[1], ee_lift_final_neg[2]]
        negative_transport_result = move_with_grasp_tracking(backend, neg_transport_target, "negative_case_transport", TRANSPORT_MAX_STEPS, track_grasp=False)

        negative_object_start = negative_transport_result["object_start_position"]
        negative_object_final = negative_transport_result["object_final_position"]
        negative_case_object_displacement = math.sqrt(sum((negative_object_final[i] - negative_object_start[i]) ** 2 for i in range(3)))

        results["negative_case_grasped_after_close"] = negative_case_grasped_after_close
        results["negative_case_object_displacement_m"] = negative_case_object_displacement
        results["negative_case_grasped_during_transport"] = backend.is_grasped()
        results["negative_case_pass"] = (
            not negative_case_grasped_after_close
            and not results["negative_case_grasped_during_transport"]
            and negative_case_object_displacement <= NEGATIVE_CASE_OBJECT_DISPLACEMENT_PASS_M
        )

        results["finite_values_pass"] = all(
            all_finite(v) for v in [ee_start, ee_final, object_start, object_final]
        )
        results["joint_limits_pass"] = True  # move_to_target/move_with_grasp_tracking raise immediately on any violation -- reaching here means none occurred

    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    overall_pass = (
        not crashed
        and results.get("is_grasped_after_transport", False)
        and results.get("transport", {}).get("grasp_maintained_all_steps", False)
        and results.get("transport", {}).get("constraint_valid_all_steps", False)
        and results.get("ee_error_pass", False)
        and results.get("lateral_match_pass", False)
        and results.get("relative_drift_pass", False)
        and results.get("vertical_deviation_pass", False)
        and results.get("clearance_pass", False)
        and results.get("reset_cleanup_pass", False)
        and results.get("negative_case_pass", False)
        and results.get("finite_values_pass", False)
        and results.get("joint_limits_pass", False)
    )

    transport = results.get("transport", {})
    output = {
        "crashed": crashed, "crash_reason": crash_reason,
        "transport_delta_commanded": TRANSPORT_DELTA_XY,
        "ee_start_position": transport.get("ee_start_position"),
        "ee_target_position": transport.get("target_position"),
        "ee_final_position": transport.get("ee_final_position"),
        "ee_position_error_m": transport.get("ee_position_error_m"),
        "ee_lateral_displacement_m": results.get("ee_lateral_displacement_m"),
        "ee_max_vertical_deviation_m": transport.get("max_vertical_deviation_m"),
        "object_start_position": transport.get("object_start_position"),
        "object_final_position": transport.get("object_final_position"),
        "object_lateral_displacement_m": results.get("object_lateral_displacement_m"),
        "object_surface_clearance_m": results.get("object_surface_clearance_m"),
        "max_object_ee_local_relative_drift_m": transport.get("max_relative_drift_m"),
        "grasp_maintained_all_steps": transport.get("grasp_maintained_all_steps"),
        "constraint_valid_all_steps": transport.get("constraint_valid_all_steps"),
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

    print("=== SO-101 transport smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        print(f"transport_delta_commanded: {TRANSPORT_DELTA_XY}")
        print(f"ee_position_error_m: {output['ee_position_error_m']:.4f}")
        print(f"ee_lateral_displacement_m: {output['ee_lateral_displacement_m']:.4f}")
        print(f"object_lateral_displacement_m: {output['object_lateral_displacement_m']:.4f}")
        print(f"ee_max_vertical_deviation_m: {output['ee_max_vertical_deviation_m']:.5f}")
        print(f"object_surface_clearance_m: {output['object_surface_clearance_m']:.4f}")
        print(f"grasp_maintained_all_steps: {output['grasp_maintained_all_steps']}")
        print(f"constraint_valid_all_steps: {output['constraint_valid_all_steps']}")
        print(f"max_object_ee_local_relative_drift_m: {output['max_object_ee_local_relative_drift_m']:.5f}")
        print(f"negative_case_object_displacement_m: {output['negative_case_object_displacement_m']:.5f}")
        print(f"negative_case_grasped: {output['negative_case_grasped']}")
        print(f"reset_cleanup_pass: {output['reset_cleanup_pass']}")
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
