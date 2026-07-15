"""Axis/rotation/gripper/safety injection tests for SmolVLALiberoActionAdapter
(v0). Matches this repo's benchmark/*.py convention -- standalone script,
plain assertions, PASS/FAIL summary, no pytest.

Per this task's explicit methodology: NativePolicyAction is injected
directly into the adapter, separate from any real model prediction --
these tests are about whether the *conversion* (scale/frame/rotation/
gripper) is correct, not about model quality. Where possible, the
resulting CanonicalRobotCommand is bridged all the way to a real
(headless) PyBulletPandaBackend to observe the actual end-effector
movement, not just intermediate numbers.

Run: python -m benchmark.test_smolvla_libero_action_adapter
"""

import math

from policy_semantics.adapters.smolvla_libero_adapter import (
    ROTATION_SCALE_RAD,
    TRANSLATION_SCALE_M,
    SmolVLALiberoActionAdapter,
)
from policy_semantics.canonical_command import CanonicalRobotCommand
from policy_semantics.manifest import get_manifest
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


def native_action(values, postprocessor_used=True) -> NativePolicyAction:
    return NativePolicyAction(
        values=list(values), source_policy="HuggingFaceVLA/smolvla_libero", postprocessor_used=postprocessor_used
    )


def decode(values, postprocessor_used=True):
    return ADAPTER.decode(native_action(values, postprocessor_used), MANIFEST, context={})


def fresh_backend() -> PyBulletPandaBackend:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    return backend


def apply_and_get_delta(backend: PyBulletPandaBackend, command: CanonicalRobotCommand):
    state_before = backend.get_state()
    ee_before = state_before["end_effector_position"]
    robot_command = command.to_legacy_robot_command()
    state_after = backend.apply_command(robot_command, steps=60)
    ee_after = state_after["end_effector_position"]
    delta = [ee_after[i] - ee_before[i] for i in range(3)]
    return delta, state_after


def main() -> None:
    print("=== 1. translation +X input -> Panda base frame +X movement ===")
    command = decode([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # gripper=-1 (open, no-op for this test)
    expected_dx = 1.0 * TRANSLATION_SCALE_M
    check(
        "CanonicalRobotCommand.translation_m[0] == +TRANSLATION_SCALE_M",
        math.isclose(command.translation_m[0], expected_dx, rel_tol=1e-6),
        f"got {command.translation_m}",
    )
    backend = fresh_backend()
    delta, _ = apply_and_get_delta(backend, command)
    backend.shutdown()
    check(
        "PyBulletPandaBackend actually moved +X by ~TRANSLATION_SCALE_M",
        delta[0] > 0.01 and abs(delta[0] - expected_dx) < 0.01,
        f"delta={delta}, expected_dx={expected_dx}",
    )
    check("no significant Y/Z drift", abs(delta[1]) < 0.01 and abs(delta[2]) < 0.01, f"delta={delta}")
    print()

    print("=== 2. translation +Y input -> Panda base frame +Y movement ===")
    command = decode([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    expected_dy = 1.0 * TRANSLATION_SCALE_M
    backend = fresh_backend()
    delta, _ = apply_and_get_delta(backend, command)
    backend.shutdown()
    check(
        "PyBulletPandaBackend actually moved +Y by ~TRANSLATION_SCALE_M",
        delta[1] > 0.01 and abs(delta[1] - expected_dy) < 0.01,
        f"delta={delta}, expected_dy={expected_dy}",
    )
    print()

    print("=== 3. translation +Z input -> Panda base frame +Z movement ===")
    command = decode([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    expected_dz = 1.0 * TRANSLATION_SCALE_M
    backend = fresh_backend()
    delta, _ = apply_and_get_delta(backend, command)
    backend.shutdown()
    check(
        "PyBulletPandaBackend actually moved +Z by ~TRANSLATION_SCALE_M",
        delta[2] > 0.01 and abs(delta[2] - expected_dz) < 0.01,
        f"delta={delta}, expected_dz={expected_dz}",
    )
    print()

    print("=== 4. rotation X/Y/Z each -> CanonicalRobotCommand scaled correctly ===")
    for axis_index, axis_name in enumerate(["X", "Y", "Z"]):
        raw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        raw[3 + axis_index] = 1.0
        command = decode(raw)
        expected = 1.0 * ROTATION_SCALE_RAD
        check(
            f"rotation_axis_angle_rad[{axis_name}] == +ROTATION_SCALE_RAD",
            math.isclose(command.rotation_axis_angle_rad[axis_index], expected, rel_tol=1e-6),
            f"got {command.rotation_axis_angle_rad}",
        )
        others = [i for i in range(3) if i != axis_index]
        check(
            f"other rotation axes stay 0 for {axis_name}-only input",
            all(command.rotation_axis_angle_rad[i] == 0.0 for i in others),
            f"got {command.rotation_axis_angle_rad}",
        )
    # PyBulletPandaBackend v1 ignores orientation deltas (see its module
    # docstring/apply_command comment) -- this is a known backend
    # limitation, not an adapter bug, so rotation is verified at the
    # CanonicalRobotCommand/RobotCommand level only, not via observed EE
    # orientation change.
    robot_command = decode([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]).to_legacy_robot_command()
    check(
        "RobotCommand.target_droll carries the scaled rotation value through",
        math.isclose(robot_command.target_droll, ROTATION_SCALE_RAD, rel_tol=1e-6),
        f"got {robot_command.target_droll}",
    )
    print()

    print("=== 5. gripper min/max -> open/close direction ===")
    command_open = decode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # robosuite: -1 = open
    command_closed = decode([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # robosuite: +1 = closed
    check("gripper=-1 (native open) -> gripper_opening_01 == 1.0 (open)", command_open.gripper_opening_01 == 1.0)
    check("gripper=+1 (native closed) -> gripper_opening_01 == 0.0 (closed)", command_closed.gripper_opening_01 == 0.0)
    robot_command_open = command_open.to_legacy_robot_command()
    robot_command_closed = command_closed.to_legacy_robot_command()
    check("legacy RobotCommand.gripper_command == 'open' for native -1", robot_command_open.gripper_command == "open")
    check(
        "legacy RobotCommand.gripper_command == 'close' for native +1", robot_command_closed.gripper_command == "close"
    )
    backend = fresh_backend()
    state_before = backend.get_state()
    state_after_open = backend.apply_command(robot_command_open, steps=30)
    state_after_close = backend.apply_command(robot_command_closed, steps=30)
    backend.shutdown()
    check(
        "PyBulletPandaBackend actually widens gripper on open command",
        state_after_open["gripper_width"] >= state_before["gripper_width"] - 1e-6,
        f"before={state_before['gripper_width']}, after_open={state_after_open['gripper_width']}",
    )
    check(
        "PyBulletPandaBackend actually narrows gripper on close command",
        state_after_close["gripper_width"] < state_after_open["gripper_width"],
        f"after_open={state_after_open['gripper_width']}, after_close={state_after_close['gripper_width']}",
    )
    print()

    print("=== 6. Too-large translation -> SafetyFilter clips ===")
    huge_command = decode([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    # Force an oversized value directly (bypassing the adapter's own
    # native [-1,1] clip) to specifically exercise PandaCommandSafetyFilter's
    # physical-unit clipping, independent of the adapter's own bound.
    from dataclasses import replace

    oversized = replace(huge_command, translation_m=(10.0, 0.0, 0.0))
    safety_filter = PandaCommandSafetyFilter(max_translation_step_m=0.03, max_rotation_step_rad=0.10)
    result = safety_filter.apply(oversized)
    check("oversized translation is accepted but clipped", result.accepted and result.clipped)
    check("clipped translation_m[0] == max_translation_step_m", result.command.translation_m[0] == 0.03)
    check("safety_clipped flag set on returned command", result.command.safety_clipped is True)
    print()

    print("=== 7. NaN/Inf -> execution refused ===")
    from dataclasses import replace as _replace

    nan_command = _replace(huge_command, translation_m=(float("nan"), 0.0, 0.0))
    inf_command = _replace(huge_command, rotation_axis_angle_rad=(0.0, float("inf"), 0.0))
    nan_result = safety_filter.apply(nan_command)
    inf_result = safety_filter.apply(inf_command)
    check("NaN translation is rejected outright (not clipped)", not nan_result.accepted and nan_result.command is None)
    check("Inf rotation is rejected outright (not clipped)", not inf_result.accepted and inf_result.command is None)
    check("NaN rejection reason mentions nan_or_inf", "nan_or_inf" in (nan_result.rejected_reason or ""))
    print()

    print("=== 8. UNKNOWN semantics -> production execution refused (unregistered checkpoint) ===")
    from vla_adapters.smolvla_adapter import SmolVLAActionAdapter

    unknown_adapter = SmolVLAActionAdapter(config={"model_id_or_path": "some/unregistered-checkpoint"})
    unknown_native = native_action([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], postprocessor_used=True)
    # An unregistered model_id gets get_manifest()'s all-UNKNOWN fallback
    # manifest (see policy_semantics/manifest.py) -- unlike
    # HuggingFaceVLA/smolvla_libero (see
    # benchmark/test_panda_rotation_and_capability.py for that gate now
    # legitimately passing as of the rotation-control turn), this one has
    # no registered semantics at all and must still refuse.
    from policy_semantics.compatibility_gate import CompatibilityGate
    from policy_semantics.manifest import get_manifest as _get_manifest

    unknown_manifest = _get_manifest("some/unregistered-checkpoint")
    gate_result = CompatibilityGate.check(unknown_manifest, smoke_test_mode=False)
    check("gate refuses an unregistered/UNKNOWN-semantics checkpoint", gate_result.passed is False)
    context = {"step_index": 0, "phase": "move_to_object", "compatibility": gate_result.to_dict()}
    normalized = unknown_adapter.normalize_model_output(unknown_native, context)
    check(
        "production call refuses even with a perfectly well-formed NativePolicyAction",
        normalized["action"] is None,
    )
    check(
        "rejection reason mentions compatibility_gate_rejected",
        "compatibility_gate_rejected" in normalized["info"].get("reason", ""),
    )
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
