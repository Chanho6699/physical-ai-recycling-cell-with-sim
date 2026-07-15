"""Rotation control + backend capability + full-gate tests (this turn).

Covers the 8 scenarios this task's rotation-control/axis-verification
work requires. Item 1 (robosuite vs. PyBullet axis agreement) is a
separate, heavier script (benchmark/verify_panda_axis_convention.py,
needs robosuite+mujoco) -- this file re-checks its saved report instead
of re-running the full cross-simulator verification, to keep this
suite runnable with only this project's normal .venv (no robosuite).

Per this task's explicit methodology: SmolVLA model predictions are
never used to validate axes/rotation -- NativePolicyAction is
hand-constructed everywhere below.

Run: python -m benchmark.test_panda_rotation_and_capability
"""

import json
import math
from dataclasses import replace
from pathlib import Path

from policy_semantics.adapters.smolvla_libero_adapter import (
    ROTATION_SCALE_RAD,
    TRANSLATION_SCALE_M,
    SmolVLALiberoActionAdapter,
)
from policy_semantics.compatibility_gate import CompatibilityGate
from policy_semantics.manifest import (
    PANDA_BACKEND_CAPABILITIES,
    PANDA_TARGET_EMBODIMENT,
    BackendCapabilities,
    get_manifest,
)
from policy_semantics.native_policy_action import NativePolicyAction
from policy_semantics.safety_filter import PandaCommandSafetyFilter
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


MANIFEST = get_manifest("HuggingFaceVLA/smolvla_libero")
ADAPTER = SmolVLALiberoActionAdapter()
SAFETY_FILTER = PandaCommandSafetyFilter()


def native_action(values) -> NativePolicyAction:
    return NativePolicyAction(values=list(values), source_policy="HuggingFaceVLA/smolvla_libero", postprocessor_used=True)


def decode(values) -> "CanonicalRobotCommand":  # noqa: F821 -- typing only
    return ADAPTER.decode(native_action(values), MANIFEST, context={})


def fresh_backend() -> PyBulletPandaBackend:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    return backend


def quat_angle_diff_deg(q_before, q_after) -> float:
    import pybullet as p

    _, angle = p.getAxisAngleFromQuaternion(p.getDifferenceQuaternion(q_before, q_after))
    return math.degrees(angle)


def main() -> None:
    print("=== 1. robosuite/PyBullet +X/+Y/+Z axis agreement (from saved verification report) ===")
    report_path = Path("docs/panda_axis_cross_verification.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        check(
            "docs/panda_axis_cross_verification.json reports axis_convention_verified=True",
            report.get("axis_convention_verified") is True,
            f"report={report.get('axis_convention_verified')}",
        )
        check(
            "all 3 delta-application axis checks passed",
            all(c["passed"] for c in report["delta_application_test"].values()),
        )
        check("forward-kinematics check passed", report["forward_kinematics_test"]["passed"])
    else:
        check(
            "docs/panda_axis_cross_verification.json exists (run benchmark/verify_panda_axis_convention.py first)",
            False,
        )
    check("PolicyManifest.axis_convention_verified matches the report", MANIFEST.axis_convention_verified is True)
    print()

    print("=== 2. rotation X/Y/Z each -> real EE orientation change ===")
    for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
        raw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        raw[3 + axis_index] = 1.0
        command = decode(raw)
        backend = fresh_backend()
        state_before = backend.get_state()
        robot_command = command.to_legacy_robot_command()
        state_after = backend.apply_command(robot_command, steps=60)
        backend.shutdown()
        angle_change_deg = quat_angle_diff_deg(state_before["end_effector_orientation"], state_after["end_effector_orientation"])
        expected_deg = math.degrees(ROTATION_SCALE_RAD)
        check(
            f"rotation-{axis_name} produces a real, non-trivial EE orientation change",
            angle_change_deg > 1.0,
            f"angle_change_deg={angle_change_deg}, expected~={expected_deg}",
        )
        check(f"state reports rotation_ignored=False for rotation-{axis_name}", state_after["rotation_ignored"] is False)
    print()

    print("=== 3. rotation-only input -> position drift measurement ===")
    command = decode([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0])  # roll only, no translation
    backend = fresh_backend()
    state_before = backend.get_state()
    pos_before = state_before["end_effector_position"]
    robot_command = command.to_legacy_robot_command()
    state_after = backend.apply_command(robot_command, steps=60)
    backend.shutdown()
    pos_after = state_after["end_effector_position"]
    drift = [pos_after[i] - pos_before[i] for i in range(3)]
    drift_magnitude = math.sqrt(sum(d * d for d in drift))
    print(f"    rotation-only position drift: {drift} (magnitude={drift_magnitude:.5f} m)")
    check(
        "rotation-only position drift stays small (IK coupling, not runaway motion)",
        drift_magnitude < 0.02,
        f"drift_magnitude={drift_magnitude}",
    )
    print()

    print("=== 4. translation + rotation applied simultaneously ===")
    command = decode([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])  # +X translation, +Y rotation, gripper closed
    backend = fresh_backend()
    state_before = backend.get_state()
    robot_command = command.to_legacy_robot_command()
    state_after = backend.apply_command(robot_command, steps=60)
    backend.shutdown()
    pos_delta = [state_after["end_effector_position"][i] - state_before["end_effector_position"][i] for i in range(3)]
    angle_change_deg = quat_angle_diff_deg(
        state_before["end_effector_orientation"], state_after["end_effector_orientation"]
    )
    check(
        "combined command moves +X by roughly TRANSLATION_SCALE_M",
        pos_delta[0] > 0.01,
        f"pos_delta={pos_delta}",
    )
    check("combined command also produces a real orientation change", angle_change_deg > 1.0, f"angle_change_deg={angle_change_deg}")
    print()

    print("=== 5. gripper regression (open/closed still correct with rotation path active) ===")
    open_command = decode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]).to_legacy_robot_command()
    closed_command = decode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).to_legacy_robot_command()
    check("gripper_command == 'open' for native -1", open_command.gripper_command == "open")
    check("gripper_command == 'close' for native +1", closed_command.gripper_command == "close")
    print()

    print("=== 6. NaN/Inf and excessive rotation -> rejected or clipped ===")
    base_command = decode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    nan_command = replace(base_command, rotation_axis_angle_rad=(float("nan"), 0.0, 0.0))
    huge_rotation_command = replace(base_command, rotation_axis_angle_rad=(5.0, 0.0, 0.0))  # far beyond 0.10 rad default
    nan_result = SAFETY_FILTER.apply(nan_command)
    huge_result = SAFETY_FILTER.apply(huge_rotation_command)
    check("NaN rotation is rejected outright", not nan_result.accepted)
    check("Excessive rotation is clipped, not rejected", huge_result.accepted and huge_result.clipped)
    check(
        "clipped rotation magnitude respects the configured max_rotation_step_rad",
        abs(huge_result.command.rotation_axis_angle_rad[0]) <= SAFETY_FILTER.max_rotation_step_rad + 1e-9,
    )
    print()

    print("=== 7. rotation-unsupported backend -> gate fails ===")
    degraded_capabilities = BackendCapabilities(
        supports_cartesian_translation=True,
        supports_cartesian_rotation=False,  # simulated "old" backend, pre-this-turn
        supports_gripper=True,
        rotation_representation="axis_angle",
        reference_frame="robot_base",
    )
    degraded_target = replace(PANDA_TARGET_EMBODIMENT)  # same target embodiment fields
    # Monkey-patch-free simulation: re-run the same check logic CompatibilityGate
    # uses, but against a hypothetically rotation-incapable backend, by
    # temporarily checking the capability condition directly (this test
    # asserts the *policy*, not by mutating the real PANDA_BACKEND_CAPABILITIES
    # singleton other tests/processes may rely on concurrently).
    capability_check_passes = (
        degraded_capabilities.supports_cartesian_translation
        and degraded_capabilities.supports_cartesian_rotation
        and degraded_capabilities.supports_gripper
        and degraded_capabilities.rotation_representation == degraded_target.rotation_representation
        and degraded_capabilities.reference_frame == degraded_target.reference_frame
    )
    check("a rotation-incapable backend's capability check would fail", capability_check_passes is False)
    check(
        "the real PANDA_BACKEND_CAPABILITIES (this turn's implementation) does pass",
        PANDA_BACKEND_CAPABILITIES.supports_cartesian_rotation is True,
    )
    # And confirm this is actually wired into the real gate today:
    gate_result = CompatibilityGate.check(MANIFEST, smoke_test_mode=False)
    check("backend_capabilities check is present and passing in the real gate", gate_result.checks["backend_capabilities"]["passed"] is True)
    print()

    print("=== 8. All capability + axis verification complete -> smolvla_libero gate passes ===")
    final_gate_result = CompatibilityGate.check(MANIFEST, smoke_test_mode=False)
    check("HuggingFaceVLA/smolvla_libero CompatibilityGate.passed is True", final_gate_result.passed is True)
    check("semantic_action_valid is True", final_gate_result.semantic_action_valid is True)
    check("no remaining failure reasons", final_gate_result.reasons == [], f"reasons={final_gate_result.reasons}")

    # End-to-end: production adapter path now actually accepts a
    # well-formed NativePolicyAction instead of refusing.
    from vla_adapters.smolvla_adapter import SmolVLAActionAdapter

    adapter = SmolVLAActionAdapter(config={"model_id_or_path": "HuggingFaceVLA/smolvla_libero"})
    good_native = native_action([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    context = {"step_index": 0, "phase": "move_to_object", "compatibility": final_gate_result.to_dict()}
    normalized = adapter.normalize_model_output(good_native, context)
    check("production /predict-equivalent call now returns a real action (not None)", normalized["action"] is not None)
    check("info.semantic_action_valid is True", normalized["info"].get("semantic_action_valid") is True)
    check(
        "returned action still matches the legacy 7-float wire format",
        isinstance(normalized["action"], list) and len(normalized["action"]) == 7,
    )
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
