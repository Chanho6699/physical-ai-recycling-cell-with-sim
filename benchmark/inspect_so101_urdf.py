"""SO-101 simulation integration -- exploration step 2 (see this task's
chat report). Standalone URDF inspection, entirely independent of
robot_sim/pybullet_panda_backend.py and every other production file --
does not import or touch anything Panda-specific, does not register any
new backend, does not load any SmolVLA checkpoint.

Loads third_party/so101_arm/so101_new_calib.urdf (the official
TheRobotStudio/SO-ARM100 asset, vendored unmodified -- see
third_party/so101_arm/SOURCE.md) into a FRESH, isolated PyBullet client
and reports its structure: joint list (type/parent/child/axis/limits),
controllable-joint auto-detection, link states at the URDF's own default
joint values, and a basic post-load stability check (a few
p.stepSimulation() calls under gravity, checking for NaN/Inf or a
large unexplained velocity spike -- NOT a claim about the real robot's
physical behavior, just "did loading this URDF into PyBullet blow up").

Run:
  .venv-vla/bin/python -m benchmark.inspect_so101_urdf
  .venv-vla/bin/python -m benchmark.inspect_so101_urdf --gui
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SO101_DIR = PROJECT_ROOT / "third_party" / "so101_arm"
URDF_PATH = SO101_DIR / "so101_new_calib.urdf"
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "urdf_inspection.json"

JOINT_TYPE_NAMES = {
    p.JOINT_REVOLUTE: "revolute", p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical", p.JOINT_PLANAR: "planar", p.JOINT_FIXED: "fixed",
}

STABILITY_STEPS = 120
STABILITY_VELOCITY_THRESHOLD_RAD_S = 5.0  # a joint drifting this fast under pure gravity (no motors driving it) after load flags a modeling problem, not a real robot claim


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def inspect_joints(robot_id: int, client_id: int) -> list:
    joints = []
    for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
        joint_type = JOINT_TYPE_NAMES.get(info[2], f"unknown({info[2]})")
        joints.append({
            "index": joint_index,
            "name": info[1].decode("utf-8"),
            "type": joint_type,
            "lower_limit": info[8],
            "upper_limit": info[9],
            "max_force": info[10],
            "max_velocity": info[11],
            "link_name": info[12].decode("utf-8"),
            "axis": list(info[13]),
            "parent_index": info[16],
        })
    return joints


def find_link_index(joints: list, link_name: str):
    for j in joints:
        if j["link_name"] == link_name:
            return j["index"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--urdf", type=str, default=str(URDF_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    urdf_path = resolve(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"SO-101 URDF not found at {urdf_path} -- see third_party/so101_arm/SOURCE.md")

    connection_mode = p.GUI if args.gui else p.DIRECT
    client_id = p.connect(connection_mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.8, physicsClientId=client_id)

    load_error = None
    robot_id = None
    try:
        # useFixedBase=True: this is a tabletop-mounted arm (matches the
        # real SO-101 follower's mounting), not a free-floating body --
        # same convention robot_sim/pybullet_panda_backend.py uses for
        # Panda, chosen independently here for the same physical reason,
        # not copied from it.
        robot_id = p.loadURDF(str(urdf_path), basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=client_id)
    except p.error as exc:
        load_error = str(exc)

    result = {
        "urdf_path": str(urdf_path),
        "load_succeeded": robot_id is not None,
        "load_error": load_error,
    }

    if robot_id is not None:
        joints = inspect_joints(robot_id, client_id)
        controllable = [j for j in joints if j["type"] in ("revolute", "prismatic")]
        fixed = [j for j in joints if j["type"] == "fixed"]

        num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=client_id)

        link_states = []
        for j in joints:
            state = p.getLinkState(robot_id, j["index"], physicsClientId=client_id)
            link_states.append({
                "link_index": j["index"], "link_name": j["link_name"],
                "world_position": list(state[4]), "world_orientation_quat": list(state[5]),
            })

        # Candidate EE links: any link whose name suggests it's the
        # intended end-effector reference frame (matches this URDF's own
        # explicit "gripper frame (dummy link)" comment) plus the two
        # physical links nearest the gripper mechanism, so a human
        # reviewer sees all reasonable options rather than one guess.
        ee_candidates = [
            j["link_name"] for j in joints
            if any(kw in j["link_name"] for kw in ("gripper_frame", "gripper_link", "moving_jaw"))
        ]

        gripper_joints = [j for j in joints if "gripper" in j["name"].lower()]

        joint_limit_issues = [
            {"joint": j["name"], "lower": j["lower_limit"], "upper": j["upper_limit"]}
            for j in controllable if j["lower_limit"] > j["upper_limit"]
        ]

        # Stability check: step under gravity with NO motor control
        # applied (velocity control mode with 0 force = "free" -- so this
        # measures whether the URDF's own mass/inertia/joint definitions
        # are self-consistent, not whether a real controller would hold
        # position, which is a separate, later question).
        p.setJointMotorControlArray(
            robot_id, [j["index"] for j in controllable], p.VELOCITY_CONTROL,
            forces=[0] * len(controllable), physicsClientId=client_id,
        )
        max_abs_velocity = 0.0
        nan_detected = False
        for _ in range(STABILITY_STEPS):
            p.stepSimulation(physicsClientId=client_id)
            for j in controllable:
                joint_state = p.getJointState(robot_id, j["index"], physicsClientId=client_id)
                position, velocity = joint_state[0], joint_state[1]
                if not (math.isfinite(position) and math.isfinite(velocity)):
                    nan_detected = True
                max_abs_velocity = max(max_abs_velocity, abs(velocity))

        result.update({
            "num_joints_total": num_joints,
            "base_position": list(base_pos),
            "base_orientation_quat": list(base_orn),
            "joints": joints,
            "num_controllable_joints": len(controllable),
            "controllable_joint_names": [j["name"] for j in controllable],
            "num_fixed_joints": len(fixed),
            "fixed_joint_names": [j["name"] for j in fixed],
            "link_states_at_default_pose": link_states,
            "end_effector_link_candidates": ee_candidates,
            "gripper_joints": [{"name": j["name"], "index": j["index"], "lower": j["lower_limit"], "upper": j["upper_limit"]} for j in gripper_joints],
            "gripper_has_mimic_joint": False,  # confirmed by inspection: only ONE revolute joint drives the moving jaw; the opposing "finger" is part of the static gripper_link mesh, not a second joint
            "joint_limit_issues": joint_limit_issues,
            "stability_check": {
                "steps": STABILITY_STEPS,
                "max_abs_joint_velocity_rad_s": max_abs_velocity,
                "nan_or_inf_detected": nan_detected,
                "exceeds_threshold": max_abs_velocity > STABILITY_VELOCITY_THRESHOLD_RAD_S,
                "note": "Free-fall-under-gravity check (zero motor force) -- flags URDF mass/inertia/limit inconsistencies, not real-robot controller behavior.",
            },
        })

        p.disconnect(client_id)
    else:
        result.update({
            "num_joints_total": None, "joints": [], "num_controllable_joints": None,
        })

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"=== SO-101 URDF inspection: {urdf_path} ===")
    print(f"load_succeeded: {result['load_succeeded']}")
    if result["load_succeeded"]:
        print(f"num_joints_total: {result['num_joints_total']}")
        print(f"controllable joints ({result['num_controllable_joints']}): {result['controllable_joint_names']}")
        print(f"fixed joints ({result['num_fixed_joints']}): {result['fixed_joint_names']}")
        print(f"end_effector_link_candidates: {result['end_effector_link_candidates']}")
        print(f"gripper_joints: {[g['name'] for g in result['gripper_joints']]}")
        print(f"joint_limit_issues: {result['joint_limit_issues']}")
        print(f"stability_check: {result['stability_check']}")
    else:
        print(f"load_error: {result['load_error']}")
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
