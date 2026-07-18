"""SO-101 simulation integration -- exploration step 3 (see this task's
chat report). Standalone joint-control smoke test, independent of
robot_sim/pybullet_panda_backend.py -- no Panda joint indices/EE link/
home pose/workspace are reused (this arm has a completely different
joint count, order, and axis convention; see the backend design table
in the chat report's final report).

Verifies, in DIRECT mode (deterministic, no GUI needed):
  1. The arm reaches a safe neutral pose. Per third_party/so101_arm/
     README.md, so101_new_calib.urdf's "new calibration" already defines
     0.0 on every joint as the middle of that joint's own range -- so
     "command everything to 0.0" IS the safe neutral pose here, not an
     assumption borrowed from Panda's READY_JOINT_POSITIONS.
  2. Each of the 5 arm joints (shoulder_pan/lift, elbow_flex, wrist_flex/
     roll) is moved individually, one small step within its own limits,
     holding the others at neutral -- commanded vs measured position is
     compared after settling.
  3. The gripper joint is driven to both limits (open/close).
  4. NaN/Inf, joint-limit violations, and "did position control actually
     converge" (not a free-fall test like inspect_so101_urdf.py's -- this
     one applies real POSITION_CONTROL, the realistic operating mode) are
     all checked automatically and written to a pass/fail JSON.

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_joint_control
"""

import argparse
import json
import math
from pathlib import Path

import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "third_party" / "so101_arm" / "so101_new_calib.urdf"
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "joint_control_smoke.json"

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
SETTLE_STEPS = 240
MOVE_FORCE = 10.0  # matches this URDF's own <limit effort="10" .../> for every actuated joint
POSITION_TOLERANCE_RAD = 0.03  # ~1.7 degrees -- convergence bar for "did it actually get there"
SMALL_STEP_FRACTION = 0.25  # fraction of each joint's own half-range used for the individual-move test


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_robot(client_id: int) -> tuple:
    robot_id = p.loadURDF(str(URDF_PATH), basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=client_id)
    joints_by_name = {}
    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        joints_by_name[info[1].decode("utf-8")] = {
            "index": joint_index, "lower": info[8], "upper": info[9], "type": info[2],
        }
    return robot_id, joints_by_name


def command_positions(robot_id: int, client_id: int, joint_indices: list, targets: list, settle_steps: int = SETTLE_STEPS) -> dict:
    p.setJointMotorControlArray(
        robot_id, joint_indices, p.POSITION_CONTROL, targetPositions=targets,
        forces=[MOVE_FORCE] * len(joint_indices), physicsClientId=client_id,
    )
    for _ in range(settle_steps):
        p.stepSimulation(physicsClientId=client_id)

    measured = {}
    for idx in joint_indices:
        state = p.getJointState(robot_id, idx, physicsClientId=client_id)
        measured[idx] = {"position": state[0], "velocity": state[1]}
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.8, physicsClientId=client_id)
    robot_id, joints_by_name = load_robot(client_id)

    arm_indices = [joints_by_name[name]["index"] for name in ARM_JOINT_NAMES]
    gripper_info = joints_by_name[GRIPPER_JOINT_NAME]
    gripper_index = gripper_info["index"]

    issues = []
    nan_inf_detected = False
    joint_limit_violations = []

    # --- 1. Neutral pose ---
    neutral_targets = [0.0] * len(arm_indices)
    measured_neutral = command_positions(robot_id, client_id, arm_indices, neutral_targets)
    neutral_errors = {}
    for name, idx in zip(ARM_JOINT_NAMES, arm_indices):
        pos = measured_neutral[idx]["position"]
        vel = measured_neutral[idx]["velocity"]
        if not (math.isfinite(pos) and math.isfinite(vel)):
            nan_inf_detected = True
        neutral_errors[name] = abs(pos - 0.0)
    neutral_converged = all(err <= POSITION_TOLERANCE_RAD for err in neutral_errors.values())

    # --- 2. Individual joint moves ---
    individual_move_results = []
    for name, idx in zip(ARM_JOINT_NAMES, arm_indices):
        lower, upper = joints_by_name[name]["lower"], joints_by_name[name]["upper"]
        half_range = (upper - lower) / 2.0
        target = min(upper, max(lower, 0.0 + SMALL_STEP_FRACTION * half_range))

        full_targets = list(neutral_targets)
        full_targets[arm_indices.index(idx)] = target
        measured = command_positions(robot_id, client_id, arm_indices, full_targets)

        pos, vel = measured[idx]["position"], measured[idx]["velocity"]
        if not (math.isfinite(pos) and math.isfinite(vel)):
            nan_inf_detected = True
        if pos < lower - 1e-6 or pos > upper + 1e-6:
            joint_limit_violations.append({"joint": name, "position": pos, "lower": lower, "upper": upper})

        error = abs(pos - target)
        individual_move_results.append({
            "joint": name, "commanded": target, "measured": pos, "error": error,
            "converged": error <= POSITION_TOLERANCE_RAD, "final_velocity": vel,
        })

        # Reset back to neutral before testing the next joint, so each
        # joint's test starts from the same known-good pose (isolates
        # one joint's behavior at a time, not a cumulative chain).
        command_positions(robot_id, client_id, arm_indices, neutral_targets)

    # --- 3. Gripper open/close ---
    gripper_results = []
    for label, target in [("close", gripper_info["lower"]), ("open", gripper_info["upper"])]:
        measured = command_positions(robot_id, client_id, [gripper_index], [target])
        pos, vel = measured[gripper_index]["position"], measured[gripper_index]["velocity"]
        if not (math.isfinite(pos) and math.isfinite(vel)):
            nan_inf_detected = True
        if pos < gripper_info["lower"] - 1e-6 or pos > gripper_info["upper"] + 1e-6:
            joint_limit_violations.append({"joint": GRIPPER_JOINT_NAME, "position": pos, "lower": gripper_info["lower"], "upper": gripper_info["upper"]})
        error = abs(pos - target)
        gripper_results.append({
            "label": label, "commanded": target, "measured": pos, "error": error,
            "converged": error <= POSITION_TOLERANCE_RAD, "final_velocity": vel,
        })

    p.disconnect(client_id)

    all_converged = neutral_converged and all(r["converged"] for r in individual_move_results) and all(r["converged"] for r in gripper_results)
    passed = all_converged and not nan_inf_detected and not joint_limit_violations

    result = {
        "urdf_path": str(URDF_PATH),
        "arm_joint_names": ARM_JOINT_NAMES,
        "gripper_joint_name": GRIPPER_JOINT_NAME,
        "position_tolerance_rad": POSITION_TOLERANCE_RAD,
        "neutral_pose_errors": neutral_errors,
        "neutral_pose_converged": neutral_converged,
        "individual_joint_moves": individual_move_results,
        "gripper_moves": gripper_results,
        "nan_or_inf_detected": nan_inf_detected,
        "joint_limit_violations": joint_limit_violations,
        "all_passed": passed,
    }

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 joint-control smoke test ===")
    print(f"neutral_pose_converged: {neutral_converged} (errors: {neutral_errors})")
    for r in individual_move_results:
        print(f"  {r['joint']:14s} commanded={r['commanded']:+.4f} measured={r['measured']:+.4f} error={r['error']:.4f} converged={r['converged']}")
    for r in gripper_results:
        print(f"  gripper {r['label']:6s} commanded={r['commanded']:+.4f} measured={r['measured']:+.4f} error={r['error']:.4f} converged={r['converged']}")
    print(f"nan_or_inf_detected: {nan_inf_detected}")
    print(f"joint_limit_violations: {joint_limit_violations}")
    print(f"\n=== ALL PASSED: {passed} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
