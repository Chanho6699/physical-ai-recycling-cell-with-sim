"""Cross-verifies robosuite's Panda base-frame axis convention against
this project's PyBulletPandaBackend -- via actual simulation results,
not a config-file read. Two independent checks:

  1. Delta-application: apply a +X/+Y/+Z-only OSC_POSE-style action
     independently in each simulator, record the real EE displacement
     vector, and confirm each is dominated by the intended axis (not a
     config comparison -- these are live physics results).
  2. Forward-kinematics: set both simulators' Panda arm to the exact
     same joint angles (this project's READY_JOINT_POSITIONS) and
     compare the resulting EE position relative to the robot's own
     base, with no controller/action involved at all -- a purely
     geometric, deterministic check.

Needs robosuite + mujoco installed (this project's .venv-vla has both,
added for this verification -- see docs/panda_axis_cross_verification.md
for the exact install commands and version pin, mujoco==3.3.0 specifically,
since robosuite 1.5.2 is incompatible with mujoco>=3.4 as installed here).

Run: .venv-vla/bin/python -m benchmark.verify_panda_axis_convention

Writes docs/panda_axis_cross_verification.json (machine-readable) and
docs/panda_axis_cross_verification.md (human-readable) -- both feed
directly into whether policy_semantics/manifest.py's
axis_convention_verified may be set True.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Tolerances (explicit, not implicit):
# - Delta test: the intended axis's displacement must be at least this
#   many times larger than the largest cross-axis component, and must be
#   positive (same sign as the commanded +1.0 input).
DELTA_TEST_MIN_DOMINANCE_RATIO = 5.0
# - FK test: EE-relative-to-base position must match within this many
#   meters -- generous enough to cover the two simulators' slightly
#   different EE reference points (PyBullet's panda_grasptarget vs.
#   robosuite's gripper0_right_grip_site are not defined at exactly the
#   same point along the gripper's approach axis).
FK_TEST_TOLERANCE_M = 0.02

READY_JOINT_POSITIONS = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

REPORT_JSON_PATH = Path("docs/panda_axis_cross_verification.json")
REPORT_MD_PATH = Path("docs/panda_axis_cross_verification.md")


def _run_pybullet_delta_test():
    from action_adapter.adapter_v0 import RobotCommand
    from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

    results = {}
    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        backend = PyBulletPandaBackend(gui=False)
        state = backend.reset()
        ee_before = np.array(state["end_effector_position"])

        deltas = [0.0, 0.0, 0.0]
        deltas[axis_index] = 0.05  # matches TRANSLATION_SCALE_M -- see smolvla_libero_adapter.py
        command = RobotCommand(
            target_dx=deltas[0],
            target_dy=deltas[1],
            target_dz=deltas[2],
            target_droll=0.0,
            target_dpitch=0.0,
            target_dyaw=0.0,
            gripper_command="open",
        )
        state_after = backend.apply_command(command, steps=60)
        ee_after = np.array(state_after["end_effector_position"])
        backend.shutdown()

        results[axis_name] = (ee_after - ee_before).tolist()
    return results


def _run_pybullet_fk_test():
    from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

    backend = PyBulletPandaBackend(gui=False)
    state = backend.reset()
    ee_pos = np.array(state["end_effector_position"])
    backend.shutdown()
    # PyBulletPandaBackend's Panda base is loaded at basePosition=[0,0,0]
    # (see its reset()) -- EE-relative-to-base is just ee_pos itself.
    return ee_pos.tolist()


def _make_robosuite_env():
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(controller=None, robot="Panda")
    return robosuite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )


def _run_robosuite_delta_test():
    results = {}
    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        env = _make_robosuite_env()
        obs = env.reset()
        ee_before = np.array(obs["robot0_eef_pos"])

        action = np.zeros(7)
        action[axis_index] = 1.0  # OSC_POSE native [-1, 1] input -- +1.0 on the intended axis only
        action[6] = -1.0  # gripper no-op-ish (open, per PandaGripper.format_action() convention)
        for _ in range(1):
            obs, _reward, _done, _info = env.step(action)
        ee_after = np.array(obs["robot0_eef_pos"])
        env.close()

        results[axis_name] = (ee_after - ee_before).tolist()
    return results


def _run_robosuite_fk_test():
    env = _make_robosuite_env()
    env.reset()
    robot = env.robots[0]
    sim = env.sim
    joint_indexes = robot._ref_joint_pos_indexes
    for idx, angle in zip(joint_indexes, READY_JOINT_POSITIONS):
        sim.data.qpos[idx] = angle
    sim.forward()  # recompute kinematics without stepping physics/controllers

    grip_site = np.array(sim.data.get_site_xpos("gripper0_right_grip_site"))
    base_pos = np.array(robot.base_pos)
    env.close()
    return (grip_site - base_pos).tolist()


def _evaluate_delta_test(pybullet_results: dict, robosuite_results: dict) -> dict:
    checks = {}
    for axis_index, axis_name in enumerate(["x", "y", "z"]):
        pb_delta = pybullet_results[axis_name]
        rs_delta = robosuite_results[axis_name]

        pb_intended = pb_delta[axis_index]
        pb_cross_max = max(abs(pb_delta[i]) for i in range(3) if i != axis_index)
        rs_intended = rs_delta[axis_index]
        rs_cross_max = max(abs(rs_delta[i]) for i in range(3) if i != axis_index)

        pb_dominant = pb_intended > 0 and (pb_cross_max == 0 or pb_intended / pb_cross_max >= DELTA_TEST_MIN_DOMINANCE_RATIO)
        rs_dominant = rs_intended > 0 and (rs_cross_max == 0 or rs_intended / rs_cross_max >= DELTA_TEST_MIN_DOMINANCE_RATIO)
        same_sign = (pb_intended > 0) == (rs_intended > 0)

        checks[axis_name] = {
            "pybullet_delta": pb_delta,
            "robosuite_delta": rs_delta,
            "pybullet_dominant_and_positive": pb_dominant,
            "robosuite_dominant_and_positive": rs_dominant,
            "same_sign": same_sign,
            "passed": pb_dominant and rs_dominant and same_sign,
        }
    return checks


def _evaluate_fk_test(pybullet_ee: list, robosuite_ee: list) -> dict:
    diff = [abs(pybullet_ee[i] - robosuite_ee[i]) for i in range(3)]
    return {
        "pybullet_ee_relative_to_base": pybullet_ee,
        "robosuite_ee_relative_to_base": robosuite_ee,
        "abs_diff": diff,
        "tolerance_m": FK_TEST_TOLERANCE_M,
        "passed": all(d <= FK_TEST_TOLERANCE_M for d in diff),
    }


def main() -> dict:
    print("=== Delta-application test (real simulation, both simulators) ===")
    pybullet_delta_results = _run_pybullet_delta_test()
    robosuite_delta_results = _run_robosuite_delta_test()
    delta_checks = _evaluate_delta_test(pybullet_delta_results, robosuite_delta_results)
    for axis_name, check in delta_checks.items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] +{axis_name.upper()}: pybullet={check['pybullet_delta']}, robosuite={check['robosuite_delta']}")

    print()
    print("=== Forward-kinematics test (same joint angles, no controller) ===")
    pybullet_fk = _run_pybullet_fk_test()
    robosuite_fk = _run_robosuite_fk_test()
    fk_check = _evaluate_fk_test(pybullet_fk, robosuite_fk)
    status = "PASS" if fk_check["passed"] else "FAIL"
    print(f"[{status}] EE relative to base -- pybullet={pybullet_fk}, robosuite={robosuite_fk}, diff={fk_check['abs_diff']}")

    all_passed = all(c["passed"] for c in delta_checks.values()) and fk_check["passed"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerances": {
            "delta_test_min_dominance_ratio": DELTA_TEST_MIN_DOMINANCE_RATIO,
            "fk_test_tolerance_m": FK_TEST_TOLERANCE_M,
        },
        "ready_joint_positions": READY_JOINT_POSITIONS,
        "delta_application_test": delta_checks,
        "forward_kinematics_test": fk_check,
        "axis_convention_verified": all_passed,
    }

    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2))

    md_lines = [
        "# Panda Base-Frame Axis Cross-Verification (robosuite vs. PyBullet)",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Both checks are *real simulation results*, not a config-file comparison.",
        "",
        "## 1. Delta-application test",
        "",
        f"Tolerance: intended-axis displacement must be positive and at least "
        f"{DELTA_TEST_MIN_DOMINANCE_RATIO}x larger than the largest cross-axis component.",
        "",
        "| Axis | PyBullet delta (m) | robosuite delta (m) | Same sign | Passed |",
        "|---|---|---|---|---|",
    ]
    for axis_name, check in delta_checks.items():
        md_lines.append(
            f"| +{axis_name.upper()} | {check['pybullet_delta']} | {check['robosuite_delta']} | "
            f"{check['same_sign']} | {check['passed']} |"
        )
    md_lines += [
        "",
        "## 2. Forward-kinematics test",
        "",
        f"Both simulators set to identical READY_JOINT_POSITIONS={READY_JOINT_POSITIONS}, "
        f"EE position read relative to robot base with no controller involved. "
        f"Tolerance: {FK_TEST_TOLERANCE_M} m per axis.",
        "",
        f"- PyBullet EE relative to base: {pybullet_fk}",
        f"- robosuite EE relative to base: {robosuite_fk}",
        f"- abs diff: {fk_check['abs_diff']}",
        f"- Passed: {fk_check['passed']}",
        "",
        "## Verdict",
        "",
        f"**axis_convention_verified = {all_passed}**",
    ]
    REPORT_MD_PATH.write_text("\n".join(md_lines) + "\n")

    print()
    print(f"Report written to {REPORT_JSON_PATH} and {REPORT_MD_PATH}")
    print("=" * 60)
    print(f"axis_convention_verified = {all_passed}")

    return report


if __name__ == "__main__":
    main()
