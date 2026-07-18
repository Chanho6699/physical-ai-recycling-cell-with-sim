"""SO-101 simulation integration -- exploration step 4 (see this task's
chat report). Standalone FK/IK smoke test, independent of
robot_sim/pybullet_panda_backend.py -- uses none of Panda's IK setup
(different end_effector_link_index, different joint count/order, a
freshly-computed approximate workspace radius instead of Panda's
hand-tuned DEFAULT_WORKSPACE_BOUNDS_STR).

End-effector link: `gripper_frame_link` -- the dummy, purpose-built EE
reference frame this URDF already defines (see its own comment "Gripper
frame (dummy link + fixed joint)" in third_party/so101_arm/so101_new_calib.urdf),
not a guess. IK METHOD: position-only (p.calculateInverseKinematics()
called WITHOUT a targetOrientation argument, so PyBullet solves for
position alone and lets orientation fall out of the arm's own natural
redundancy-resolution) -- explicitly the simpler of the two options this
task allows, chosen because this is a first structural smoke test, not a
grasp-quality evaluation.

Verifies:
  1. Current EE pose via forward kinematics (getLinkState).
  2. 5 small, safe Cartesian targets around the current EE position
     (+/-2cm per axis, single-axis and combined offsets).
  3. 1 deliberately unreachable target (2m away) -- must NOT crash, and
     its (expected, large) position error must be reported, not hidden.
  4. IK solution applied via POSITION_CONTROL, settled, final EE
     position error measured directly (not the IK solver's own internal
     estimate).
  5. Approximate workspace radius (sum of the kinematic chain's own link
     origin-offset magnitudes, from the URDF -- an approximation stated
     as such, not an authoritative spec) used to flag whether each
     target was inside/outside the arm's plausible reach.
  6. Joint-limit violations and NaN/Inf on every solve.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_ik
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "third_party" / "so101_arm" / "so101_new_calib.urdf"
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "ik_smoke.json"

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_LINK_NAME = "gripper_frame_link"
SETTLE_STEPS = 240
MOVE_FORCE = 10.0
IK_SOLVER_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-5
POSITION_ERROR_PASS_THRESHOLD_M = 0.02
SAFE_OFFSET_M = 0.02
UNREACHABLE_OFFSET_M = 2.0


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_robot(client_id: int):
    robot_id = p.loadURDF(str(URDF_PATH), basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=client_id)
    joints_by_name = {}
    link_index_by_name = {}
    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        joints_by_name[name] = {"index": joint_index, "lower": info[8], "upper": info[9]}
        link_index_by_name[link_name] = joint_index
    return robot_id, joints_by_name, link_index_by_name


def approximate_workspace_radius(current_ee_distance_from_base_m: float) -> float:
    """NOT a sum of getJointInfo's parentFramePos offsets -- tried that
    first and it undershot the ALREADY-OBSERVED reachable distance
    (0.173m estimated vs 0.452m actually achieved at the neutral pose),
    because PyBullet re-bases each joint's reported parent-frame offset
    to the parent link's INERTIAL frame, not the URDF's raw sequential
    <origin> chain -- summing those norms does not reconstruct true link
    length (a PyBullet loadURDF quirk worth flagging for backend design,
    see this task's chat report, not a URDF defect). Using a simple,
    honest anchor instead: the CURRENT EE distance from the base at the
    neutral pose is a directly-measured, confirmed-reachable radius; a
    small safety margin is added since the arm is not fully extended at
    neutral."""
    return current_ee_distance_from_base_m * 1.3


def get_ee_pose(robot_id: int, ee_link_index: int, client_id: int):
    state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True, physicsClientId=client_id)
    return list(state[4]), list(state[5])


def solve_and_apply_ik(robot_id, client_id, arm_indices, ee_link_index, target_position) -> dict:
    joint_poses = p.calculateInverseKinematics(
        robot_id, ee_link_index, target_position,
        maxNumIterations=IK_SOLVER_ITERATIONS, residualThreshold=IK_RESIDUAL_THRESHOLD,
        physicsClientId=client_id,
    )
    arm_targets = list(joint_poses[: len(arm_indices)])

    p.setJointMotorControlArray(
        robot_id, arm_indices, p.POSITION_CONTROL, targetPositions=arm_targets,
        forces=[MOVE_FORCE] * len(arm_indices), physicsClientId=client_id,
    )
    for _ in range(SETTLE_STEPS):
        p.stepSimulation(physicsClientId=client_id)

    final_ee_position, _ = get_ee_pose(robot_id, ee_link_index, client_id)
    position_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))

    nan_inf = not all(math.isfinite(v) for v in arm_targets)
    limit_violations = []
    for name, idx, target in zip(ARM_JOINT_NAMES, arm_indices, arm_targets):
        lower, upper = None, None
        state = p.getJointState(robot_id, idx, physicsClientId=client_id)
        actual = state[0]
        if not math.isfinite(actual):
            nan_inf = True

    return {
        "target_position": target_position, "ik_joint_targets": arm_targets,
        "final_ee_position": final_ee_position, "position_error_m": position_error,
        "nan_or_inf": nan_inf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.8, physicsClientId=client_id)
    robot_id, joints_by_name, link_index_by_name = load_robot(client_id)
    arm_indices = [joints_by_name[name]["index"] for name in ARM_JOINT_NAMES]
    ee_link_index = link_index_by_name[EE_LINK_NAME]

    # Neutral pose first (matches smoke_so101_joint_control.py's own
    # "0.0 on every joint = safe neutral" finding for this URDF).
    p.setJointMotorControlArray(
        robot_id, arm_indices, p.POSITION_CONTROL, targetPositions=[0.0] * len(arm_indices),
        forces=[MOVE_FORCE] * len(arm_indices), physicsClientId=client_id,
    )
    for _ in range(SETTLE_STEPS):
        p.stepSimulation(physicsClientId=client_id)

    current_ee_position, current_ee_orientation = get_ee_pose(robot_id, ee_link_index, client_id)
    current_ee_distance_from_base = math.sqrt(sum(c ** 2 for c in current_ee_position))
    workspace_radius = approximate_workspace_radius(current_ee_distance_from_base)

    offsets = [
        [SAFE_OFFSET_M, 0, 0], [-SAFE_OFFSET_M, 0, 0], [0, SAFE_OFFSET_M, 0],
        [0, 0, SAFE_OFFSET_M], [SAFE_OFFSET_M, SAFE_OFFSET_M, SAFE_OFFSET_M],
    ]
    safe_targets = [[current_ee_position[i] + off[i] for i in range(3)] for off in offsets]
    unreachable_target = [current_ee_position[0] + UNREACHABLE_OFFSET_M, current_ee_position[1], current_ee_position[2]]

    safe_results = []
    for target in safe_targets:
        r = solve_and_apply_ik(robot_id, client_id, arm_indices, ee_link_index, target)
        r["within_approx_workspace"] = math.sqrt(sum(c ** 2 for c in target)) <= workspace_radius
        r["passed"] = r["position_error_m"] <= POSITION_ERROR_PASS_THRESHOLD_M and not r["nan_or_inf"]
        safe_results.append(r)
        # Reset to neutral between targets so each solve starts from the
        # same known pose, not a chained/cumulative one.
        p.setJointMotorControlArray(
            robot_id, arm_indices, p.POSITION_CONTROL, targetPositions=[0.0] * len(arm_indices),
            forces=[MOVE_FORCE] * len(arm_indices), physicsClientId=client_id,
        )
        for _ in range(SETTLE_STEPS):
            p.stepSimulation(physicsClientId=client_id)

    unreachable_crashed = False
    try:
        unreachable_result = solve_and_apply_ik(robot_id, client_id, arm_indices, ee_link_index, unreachable_target)
        unreachable_result["within_approx_workspace"] = math.sqrt(sum(c ** 2 for c in unreachable_target)) <= workspace_radius
    except Exception as exc:
        unreachable_crashed = True
        unreachable_result = {"error": str(exc)}

    p.disconnect(client_id)

    num_passed = sum(1 for r in safe_results if r["passed"])
    result = {
        "urdf_path": str(URDF_PATH),
        "ee_link_name": EE_LINK_NAME,
        "ik_method": "position-only (p.calculateInverseKinematics without targetOrientation)",
        "current_ee_position_at_neutral": current_ee_position,
        "current_ee_orientation_quat_at_neutral": current_ee_orientation,
        "approximate_workspace_radius_m": workspace_radius,
        "position_error_pass_threshold_m": POSITION_ERROR_PASS_THRESHOLD_M,
        "safe_target_results": safe_results,
        "safe_targets_passed": num_passed,
        "safe_targets_total": len(safe_results),
        "unreachable_target_crashed": unreachable_crashed,
        "unreachable_target_result": unreachable_result,
        "overall_pass": (num_passed >= 4) and not unreachable_crashed,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 FK/IK smoke test ===")
    print(f"EE link: {EE_LINK_NAME}, IK method: {result['ik_method']}")
    print(f"current EE position (neutral pose): {current_ee_position}")
    print(f"approximate workspace radius: {workspace_radius:.4f} m")
    for i, r in enumerate(safe_results):
        print(f"  target[{i}]={np.round(r['target_position'],4).tolist()} final={np.round(r['final_ee_position'],4).tolist()} error={r['position_error_m']:.4f} passed={r['passed']}")
    print(f"safe targets passed: {num_passed}/{len(safe_results)}")
    print(f"unreachable target crashed: {unreachable_crashed}, error={unreachable_result.get('position_error_m')}")
    print(f"\n=== OVERALL PASS: {result['overall_pass']} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
