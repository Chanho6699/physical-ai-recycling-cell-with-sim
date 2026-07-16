"""Real PyBullet observation tests for HuggingFaceVLA/smolvla_libero (v0).

Covers this task's 9 required scenarios. Items 1-6 exercise
PyBulletPandaBackend directly (no GPU/model needed); items 7-8 exercise
vla_server/model_loader.py's _build_smolvla_libero_images()/
_build_smolvla_libero_state() directly (pure functions, no model
either); item 9 re-runs the three prior policy_semantics/rotation/
mock-action suites to confirm no regression.

Run: python -m benchmark.test_libero_real_observation
"""

import numpy as np

from robot_sim.pybullet_panda_backend import (
    LIBERO_CAMERA_HEIGHT,
    LIBERO_CAMERA_WIDTH,
    PyBulletPandaBackend,
)
from action_adapter.adapter_v0 import RobotCommand
from vla_server.model_loader import _build_smolvla_libero_images, _build_smolvla_libero_state

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def fresh_backend() -> PyBulletPandaBackend:
    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    return backend


def main() -> None:
    print("=== 1. Panda move -> 8D state EE position actually changes ===")
    backend = fresh_backend()
    state_before = backend.get_libero_observation_state()
    command = RobotCommand(
        target_dx=0.05, target_dy=0.0, target_dz=0.0, target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
        gripper_command="open",
    )
    backend.apply_command(command, steps=60)
    state_after = backend.get_libero_observation_state()
    pos_delta = [state_after[i] - state_before[i] for i in range(3)]
    check("EE position[0] (+X) actually changed by ~0.05m", pos_delta[0] > 0.01, f"pos_delta={pos_delta}")
    check("EE position[1]/[2] stayed roughly still", abs(pos_delta[1]) < 0.01 and abs(pos_delta[2]) < 0.01, f"pos_delta={pos_delta}")
    backend.shutdown()
    print()

    print("=== 2. Panda rotation -> 8D state axis-angle actually changes ===")
    backend = fresh_backend()
    state_before = backend.get_libero_observation_state()
    command = RobotCommand(
        target_dx=0.0, target_dy=0.0, target_dz=0.0, target_droll=0.3, target_dpitch=0.0, target_dyaw=0.0,
        gripper_command="open",
    )
    backend.apply_command(command, steps=60)
    state_after = backend.get_libero_observation_state()
    orn_delta = [state_after[3 + i] - state_before[3 + i] for i in range(3)]
    orn_delta_mag = sum(d * d for d in orn_delta) ** 0.5
    check("axis-angle state component actually changed", orn_delta_mag > 0.05, f"orn_delta={orn_delta}")
    backend.shutdown()
    print()

    print("=== 3. gripper open/close -> both finger qpos change ===")
    backend = fresh_backend()
    state_open = backend.get_libero_observation_state()
    close_command = RobotCommand(
        target_dx=0.0, target_dy=0.0, target_dz=0.0, target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
        gripper_command="close",
    )
    backend.apply_command(close_command, steps=30)
    state_closed = backend.get_libero_observation_state()
    left_delta = state_open[6] - state_closed[6]
    # state[7] (the second gripper channel) is negated in
    # get_libero_observation_state() to match
    # HuggingFaceVLA/smolvla_libero's training-time sign convention (see
    # that method's docstring) -- it moves from strongly negative (open,
    # ~-0.04) toward ~0 (closed), i.e. it INCREASES on close, the
    # opposite direction of the (unnegated) left channel.
    right_delta = state_closed[7] - state_open[7]
    check("left finger qpos decreased on close", left_delta > 0.01, f"open={state_open[6]}, closed={state_closed[6]}")
    check("right finger qpos (negated) increased on close", right_delta > 0.01, f"open={state_open[7]}, closed={state_closed[7]}")
    backend.shutdown()
    print()

    print("=== 4. main and wrist images are not identical ===")
    backend = fresh_backend()
    main_img = backend.render_main_camera()
    wrist_img = backend.render_wrist_camera()
    check("main image != wrist image", not np.array_equal(main_img, wrist_img))
    backend.shutdown()
    print()

    print("=== 5. wrist camera view changes as the robot moves ===")
    backend = fresh_backend()
    wrist_before = backend.render_wrist_camera()
    command = RobotCommand(
        target_dx=0.05, target_dy=0.05, target_dz=0.0, target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
        gripper_command="open",
    )
    backend.apply_command(command, steps=60)
    wrist_after = backend.render_wrist_camera()
    wrist_diff = np.abs(wrist_after.astype(int) - wrist_before.astype(int)).mean()
    check("wrist image changed after robot movement", not np.array_equal(wrist_before, wrist_after))
    check("wrist image change is non-trivial (not a 1-pixel fluke)", wrist_diff > 0.5, f"wrist_diff={wrist_diff}")
    backend.shutdown()
    print()

    print("=== 6. observation shape/dtype matches SmolVLA processor requirements ===")
    backend = fresh_backend()
    main_img = backend.render_main_camera()
    wrist_img = backend.render_wrist_camera()
    state8d = backend.get_libero_observation_state()
    check("main image is (256, 256, 3) uint8", main_img.shape == (LIBERO_CAMERA_HEIGHT, LIBERO_CAMERA_WIDTH, 3) and main_img.dtype == np.uint8, f"shape={main_img.shape}, dtype={main_img.dtype}")
    check("wrist image is (256, 256, 3) uint8", wrist_img.shape == (LIBERO_CAMERA_HEIGHT, LIBERO_CAMERA_WIDTH, 3) and wrist_img.dtype == np.uint8, f"shape={wrist_img.shape}, dtype={wrist_img.dtype}")
    check("8D state has exactly 8 float components", len(state8d) == 8 and all(isinstance(v, float) for v in state8d))
    backend.shutdown()
    print()

    print("=== 7. real 2-camera + 8D state -> degraded_input=False ===")
    backend = fresh_backend()
    main_img = backend.render_main_camera()
    wrist_img = backend.render_wrist_camera()
    state8d = backend.get_libero_observation_state()
    robot_state = {
        "ee_position": state8d[0:3],
        "ee_orientation_axis_angle": state8d[3:6],
        "gripper_qpos": state8d[6:8],
    }
    model_input = {
        "images_by_role": {"main": main_img, "wrist": wrist_img},
        "robot_state": robot_state,
        "instruction": "pick up the bottle",
    }
    images, images_degraded, images_source = _build_smolvla_libero_images(model_input)
    state, state_degraded, state_source = _build_smolvla_libero_state(model_input)
    check("images not degraded with real main+wrist frames", images_degraded is False, f"source={images_source}")
    check("state not degraded with real 8D robot_state", state_degraded is False, f"source={state_source}")
    check(
        "the two observation.images.* arrays are genuinely different (not duplicated)",
        not np.array_equal(images["observation.images.image"], images["observation.images.image2"]),
    )
    backend.shutdown()
    print()

    print("=== 8. legacy single-image / missing-state path -> degraded_input=True ===")
    legacy_model_input = {"image": main_img, "robot_state": {}, "instruction": "pick up the bottle"}
    legacy_images, legacy_images_degraded, legacy_images_source = _build_smolvla_libero_images(legacy_model_input)
    legacy_state, legacy_state_degraded, legacy_state_source = _build_smolvla_libero_state(legacy_model_input)
    check("legacy single-image path is marked degraded", legacy_images_degraded is True, f"source={legacy_images_source}")
    check(
        "legacy path duplicates the single image across both keys (documented degraded behavior)",
        np.array_equal(legacy_images["observation.images.image"], legacy_images["observation.images.image2"]),
    )
    check("missing robot_state fields -> state marked degraded (zero-filled)", legacy_state_degraded is True, f"source={legacy_state_source}")
    check("degraded state is all zero", all(v == 0.0 for v in legacy_state))

    no_image_input = {"robot_state": {}, "instruction": "pick up the bottle"}
    none_images, none_degraded, none_source = _build_smolvla_libero_images(no_image_input)
    check("no image at all -> zero placeholder, still marked degraded", none_degraded is True, f"source={none_source}")
    print()

    print("=== 9. regression: policy_semantics / rotation / mock-action suites ===")
    import subprocess
    import sys as _sys

    for module in (
        "benchmark.test_policy_semantics",
        "benchmark.test_smolvla_libero_action_adapter",
        "benchmark.test_panda_rotation_and_capability",
    ):
        result = subprocess.run([_sys.executable, "-m", module], capture_output=True, text=True)
        passed = "ALL CHECKS PASSED" in result.stdout
        check(f"{module} -- ALL CHECKS PASSED", passed, result.stdout[-800:] if not passed else "")
    print()

    print("=== 10. gripper channel-2 sign convention (production fix regression) ===")
    # Confirms get_libero_observation_state()'s minimal fix: the second
    # gripper channel (state[-1]) is negated to match
    # HuggingFaceVLA/smolvla_libero's training-time convention (verified
    # directly against real HuggingFaceVLA/libero dataset samples --
    # see that method's docstring and
    # benchmark/run_gripper_channel_sign_ab_experiment.py), while the
    # first channel (state[-2]) and everything else (field order/shape,
    # action adapter, gripper open/close execution) stay unchanged.
    backend = fresh_backend()
    state_open = backend.get_libero_observation_state()
    check("shape is still exactly 8", len(state_open) == 8, f"got {len(state_open)}")
    check(
        "open: state[-2] > 0 and state[-1] < 0 (first channel unnegated, second negated)",
        state_open[-2] > 0 and state_open[-1] < 0,
        f"state[-2:]={state_open[-2:]}",
    )
    check(
        "open: abs(state[-2]) and abs(state[-1]) correspond within tolerance "
        "(both fingers move symmetrically -- only the sign of the second differs)",
        abs(abs(state_open[-2]) - abs(state_open[-1])) < 1e-3,
        f"abs(state[-2])={abs(state_open[-2])} abs(state[-1])={abs(state_open[-1])}",
    )

    close_command = RobotCommand(
        target_dx=0.0, target_dy=0.0, target_dz=0.0, target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
        gripper_command="close",
    )
    backend.apply_command(close_command, steps=30)
    state_closed = backend.get_libero_observation_state()
    check(
        "closed: the two channels still have opposite signs (or the negated one is ~0, never positive-and-large)",
        state_closed[-2] >= -1e-3 and state_closed[-1] <= 1e-3,
        f"state[-2:]={state_closed[-2:]}",
    )
    check(
        "closed: abs(state[-2]) and abs(state[-1]) still correspond within tolerance",
        abs(abs(state_closed[-2]) - abs(state_closed[-1])) < 1e-3,
        f"abs(state[-2])={abs(state_closed[-2])} abs(state[-1])={abs(state_closed[-1])}",
    )
    check(
        "existing gripper open/close EXECUTION is unchanged -- backend.get_state()['gripper_state'] still "
        "reports 'close' (this fix only touches the observation-building return value, never joint control)",
        backend.get_state()["gripper_state"] == "close",
    )
    backend.shutdown()

    fresh_reset_backend = fresh_backend()
    reset_state = fresh_reset_backend.get_libero_observation_state()
    check(
        "sign convention already holds immediately after reset() (not just after some motion)",
        reset_state[-2] > 0 and reset_state[-1] < 0,
        f"state[-2:]={reset_state[-2:]}",
    )
    fresh_reset_backend.shutdown()

    # action_adapter/gripper-command EXECUTION path is untouched: convert
    # a canned 7-float action through the same, unmodified ActionAdapter
    # and confirm the resulting RobotCommand/gripper_command still work
    # exactly as before (this fix never touches action_adapter/adapter_v0.py).
    from action_adapter.adapter_v0 import ActionAdapter

    action_adapter = ActionAdapter()
    close_action_command = action_adapter.convert([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    open_action_command = action_adapter.convert([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    check(
        "ActionAdapter's own gripper-command conversion is unaffected (still 1.0=close, 0.0=open)",
        close_action_command.gripper_command == "close" and open_action_command.gripper_command == "open",
    )

    # production 코드가 diagnostic helper에 의존하지 않음 (production code does not
    # depend on any diagnostic helper) -- grep-based, not "trust me".
    import re
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    production_dirs = ["robot_sim", "vla_server", "policy_semantics", "vla_adapters", "policy"]
    diagnostic_module_pattern = re.compile(r"\bbenchmark\.(run|test)_[a-z_]*diagnostic[a-z_]*\b")
    hits = []
    for directory in production_dirs:
        for path in (project_root / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if ("import " in line and diagnostic_module_pattern.search(line)) or "apply_gripper_condition(" in line or "apply_coordinate_hypothesis(" in line:
                    hits.append(f"{path.relative_to(project_root)}:{line_no}")
    check("no production file imports/calls any diagnostic-only helper", len(hits) == 0, f"unexpected: {hits}")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
