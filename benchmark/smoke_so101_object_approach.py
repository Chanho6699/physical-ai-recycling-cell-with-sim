"""SO-101 object-approach smoke test (see this task's chat report).

Verifies robot_sim/so101_pybullet_backend.So101PyBulletBackend's new
table+object scene and a stepped, two-stage approach (pre-grasp, then
approach) driven entirely by repeated, small, axis-clamped calls to the
EXISTING command_end_effector_delta() -- no new backend "approach"
method was added; the step-limiting/remaining-vector/abnormal-IK-
failure logic lives here, in the caller, matching this task's own
"현재 command_end_effector_delta()를 사용해 반복적으로 목표에 접근한다"
framing. No grasp constraint, no lift, no bin/place, no camera,
no orientation IK, no expert policy, no SmolVLA, no training.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_object_approach
"""

import argparse
import json
import math
from pathlib import Path

from robot_sim.so101_pybullet_backend import (
    MIN_EE_HEIGHT_M,
    OBJECT_HALF_EXTENTS,
    TABLE_TOP_Z,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "object_approach_smoke.json"

PRE_GRASP_OFFSET_M = [0.0, 0.0, 0.08]
APPROACH_OFFSET_M = [0.0, 0.0, 0.03]

MAX_STEP_M = 0.02          # per-axis clamp on any single command_end_effector_delta() call
MAX_STEPS = 50
CONVERGENCE_TOLERANCE_M = 0.005
STEP_ERROR_FAILURE_THRESHOLD_M = 0.03  # a single small (<=MAX_STEP_M) step's own resulting error should never be this large

PRE_GRASP_ERROR_PASS_M = 0.01
APPROACH_ERROR_PASS_M = 0.01
OBJECT_DISPLACEMENT_PASS_M = 0.002
JOINT_LIMIT_EPS = 1e-6


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def all_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def move_to_target(backend: So101PyBulletBackend, target_position: list, events: list, stage_label: str) -> dict:
    """Repeatedly calls the EXISTING command_end_effector_delta() with a
    small, per-axis-clamped remaining-vector step each iteration -- see
    this module's docstring. Raises immediately on a non-finite EE
    position or an abnormally large single-step error (this smoke
    test's own definition of 'IK 결과가 비정상' for SMALL, bounded steps
    -- a large delta intentionally requested elsewhere, e.g.
    benchmark/smoke_so101_ik.py's unreachable-target case, is a
    different scenario and is not what this check is judging)."""
    step_errors = []
    joint_limit_violations = []
    for step_index in range(MAX_STEPS):
        current_ee_position, _ = backend.get_end_effector_pose()
        remaining = [target_position[i] - current_ee_position[i] for i in range(3)]
        remaining_norm = math.sqrt(sum(c ** 2 for c in remaining))
        if remaining_norm <= CONVERGENCE_TOLERANCE_M:
            break
        clamped_delta = [max(-MAX_STEP_M, min(MAX_STEP_M, c)) for c in remaining]

        obs = backend.command_end_effector_delta(clamped_delta)
        if not all_finite(obs["end_effector_position"]):
            raise RuntimeError(f"[{stage_label}] non-finite EE position at step {step_index}: {obs['end_effector_position']}")
        step_errors.append(obs["ee_delta_position_error"])
        if obs["ee_delta_position_error"] > STEP_ERROR_FAILURE_THRESHOLD_M:
            raise RuntimeError(
                f"[{stage_label}] abnormal IK step error {obs['ee_delta_position_error']:.4f}m at step {step_index} "
                f"(commanded delta {clamped_delta}) -- treated as immediate failure, not retried."
            )
        if obs["end_effector_position"][2] < MIN_EE_HEIGHT_M - 1e-4:
            events.append({"stage": stage_label, "issue": f"EE z={obs['end_effector_position'][2]:.4f} below MIN_EE_HEIGHT_M={MIN_EE_HEIGHT_M}"})

        for name, pos in zip(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], obs["joint_positions"]):
            info = backend.joint_info_by_name[name]
            if pos < info["lower"] - JOINT_LIMIT_EPS or pos > info["upper"] + JOINT_LIMIT_EPS:
                joint_limit_violations.append({"stage": stage_label, "step": step_index, "joint": name, "position": pos})

    final_ee_position, _ = backend.get_end_effector_pose()
    final_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))
    return {
        "target": target_position, "final_ee_position": final_ee_position, "error": final_error,
        "num_steps": len(step_errors), "step_errors": step_errors, "joint_limit_violations": joint_limit_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    events = []
    crashed = False
    crash_reason = None
    stage_results = {}
    all_joint_limit_violations = []

    backend = So101PyBulletBackend(gui=False)
    try:
        # 1. reset, scene sanity
        obs = backend.reset()
        scene = backend.get_scene_state()
        object_initial_position, object_initial_orientation = backend.get_object_pose()

        stage_results["object_initial_pose"] = {"position": object_initial_position, "orientation": object_initial_orientation}
        stage_results["table_top_z"] = TABLE_TOP_Z
        stage_results["object_pose_finite"] = all_finite(object_initial_position) and all_finite(object_initial_orientation)

        object_bottom_z = object_initial_position[2] - OBJECT_HALF_EXTENTS[2]
        stage_results["object_on_table"] = abs(object_bottom_z - TABLE_TOP_Z) < 0.005  # resting on the table top, not floating or sunk in
        stage_results["object_z"] = object_initial_position[2]

        initial_ee_position = obs["end_effector_position"]
        initial_distance_ee_to_object = math.sqrt(sum((initial_ee_position[i] - object_initial_position[i]) ** 2 for i in range(3)))
        stage_results["initial_ee_object_distance_m"] = initial_distance_ee_to_object
        stage_results["initial_no_overlap"] = initial_distance_ee_to_object > (OBJECT_HALF_EXTENTS[0] + 0.01)

        if not stage_results["object_pose_finite"]:
            events.append({"stage": "reset", "issue": "object pose not finite"})
        if not stage_results["object_on_table"]:
            events.append({"stage": "reset", "issue": f"object not resting on table top (bottom_z={object_bottom_z:.4f}, table_top_z={TABLE_TOP_Z})"})
        if not stage_results["initial_no_overlap"]:
            events.append({"stage": "reset", "issue": "initial EE and object positions overlap"})

        # 2. gripper open (explicit)
        open_obs = backend.set_gripper(1.0)
        stage_results["gripper_open_normalized"] = open_obs["gripper_position_normalized"]

        # 3. pre-grasp
        pre_grasp_target = [object_initial_position[i] + PRE_GRASP_OFFSET_M[i] for i in range(3)]
        pre_grasp_result = move_to_target(backend, pre_grasp_target, events, "pre_grasp")
        stage_results["pre_grasp"] = pre_grasp_result
        all_joint_limit_violations += pre_grasp_result["joint_limit_violations"]

        object_after_pre_grasp = backend.get_object_position()

        # 4. approach
        approach_target = [object_initial_position[i] + APPROACH_OFFSET_M[i] for i in range(3)]
        approach_result = move_to_target(backend, approach_target, events, "approach")
        stage_results["approach"] = approach_result
        all_joint_limit_violations += approach_result["joint_limit_violations"]

        object_after_approach = backend.get_object_position()

        object_displacement_pre_grasp = math.sqrt(sum((object_after_pre_grasp[i] - object_initial_position[i]) ** 2 for i in range(3)))
        object_displacement_approach = math.sqrt(sum((object_after_approach[i] - object_initial_position[i]) ** 2 for i in range(3)))
        stage_results["object_displacement_after_pre_grasp_m"] = object_displacement_pre_grasp
        stage_results["object_displacement_after_approach_m"] = object_displacement_approach
        stage_results["max_object_displacement_m"] = max(object_displacement_pre_grasp, object_displacement_approach)

        # 5. reset reproducibility
        reset2_obs = backend.reset()
        reset2_object_position, _ = backend.get_object_pose()
        stage_results["reset_reproducible_ee"] = math.sqrt(
            sum((reset2_obs["end_effector_position"][i] - initial_ee_position[i]) ** 2 for i in range(3))
        ) < 1e-6
        stage_results["reset_reproducible_object"] = math.sqrt(
            sum((reset2_object_position[i] - object_initial_position[i]) ** 2 for i in range(3))
        ) < 1e-4  # slightly looser than EE -- object settling under gravity is not bit-exact deterministic re-solve like a pure kinematic reset

    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    passed = (
        not crashed
        and not events
        and not all_joint_limit_violations
        and stage_results.get("object_pose_finite", False)
        and stage_results.get("object_on_table", False)
        and stage_results.get("initial_no_overlap", False)
        and stage_results.get("pre_grasp", {}).get("error", 999) <= PRE_GRASP_ERROR_PASS_M
        and stage_results.get("approach", {}).get("error", 999) <= APPROACH_ERROR_PASS_M
        and stage_results.get("max_object_displacement_m", 999) <= OBJECT_DISPLACEMENT_PASS_M
        and stage_results.get("reset_reproducible_ee", False)
        and stage_results.get("reset_reproducible_object", False)
    )

    result = {
        "crashed": crashed, "crash_reason": crash_reason,
        "stage_results": stage_results,
        "joint_limit_violations": all_joint_limit_violations,
        "numeric_issues": events,
        "all_passed": passed,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 object-approach smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        print(f"object_initial_pose: {stage_results['object_initial_pose']}")
        print(f"object_on_table: {stage_results['object_on_table']} (z={stage_results['object_z']:.4f}, table_top_z={TABLE_TOP_Z})")
        print(f"initial_no_overlap: {stage_results['initial_no_overlap']} (distance={stage_results['initial_ee_object_distance_m']:.4f}m)")
        pg = stage_results["pre_grasp"]
        print(f"pre_grasp: target={pg['target']} final={pg['final_ee_position']} error={pg['error']:.4f}m steps={pg['num_steps']}")
        ap = stage_results["approach"]
        print(f"approach: target={ap['target']} final={ap['final_ee_position']} error={ap['error']:.4f}m steps={ap['num_steps']}")
        print(f"max_object_displacement_m: {stage_results['max_object_displacement_m']:.5f}")
        print(f"reset_reproducible_ee: {stage_results['reset_reproducible_ee']}, reset_reproducible_object: {stage_results['reset_reproducible_object']}")
    print(f"joint_limit_violations: {len(all_joint_limit_violations)}")
    print(f"numeric_issues: {events}")
    print(f"\n=== ALL PASSED: {passed} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
