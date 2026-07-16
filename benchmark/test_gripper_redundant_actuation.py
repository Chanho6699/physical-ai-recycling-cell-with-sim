"""Regression tests for the redundant-gripper-actuation fix in
robot_sim/pybullet_panda_backend.py's PyBulletPandaBackend.apply_command()
(see this task's chat report for the discovery: ActionAdapter.convert()
always emits "open" or "close", never a no-op/hold, so apply_command()
used to re-run a full 60-step open_gripper()/close_gripper() actuation on
EVERY call regardless of whether the gripper was already in that state --
70 stepSimulation() calls/frame, ~3.43Hz real cadence against a declared
20Hz metadata).

The fix (apply_command()'s new _is_gripper_open() gate) only skips
actuation when the command wouldn't change anything; this file checks
that skip decision directly by isolating gripper-only physics steps
(passing steps=0 to apply_command() so the arm itself contributes zero
stepSimulation() calls, leaving only whatever open_gripper()/
close_gripper() might add).

Covers (this task's item 3 minimum list):
  1. reset -> open command -> 0 gripper steps (already open)
  2. reset -> close command -> 60 gripper steps (real transition)
  3. close -> close command -> 0 gripper steps (already closed)
  4. close -> open command -> 60 gripper steps (real transition)
  5. open -> open command -> 0 gripper steps (already open)
  6. finger qpos actually reaches the commanded target on real transitions
  7. closing on a grasped object (finger qpos stuck at the object's
     half-width, never reaching FINGER_CLOSE_POSITION) is still correctly
     recognized as "closed" -- re-issuing close skips actuation instead of
     re-running it forever during lift/move-above-bin phases
  8. arm-movement/IK call args are untouched by this change (regression)

Run: .venv-vla/bin/python -m benchmark.test_gripper_redundant_actuation
"""

from pathlib import Path

from action_adapter.adapter_v0 import RobotCommand
from robot_sim.pybullet_panda_backend import (
    FINGER_CLOSE_POSITION,
    FINGER_OPEN_POSITION,
    GRIPPER_OPEN_QPOS_THRESHOLD,
    PyBulletPandaBackend,
)
import robot_sim.pybullet_panda_backend as backend_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def count_gripper_only_steps(backend: PyBulletPandaBackend, gripper_command: str) -> int:
    """steps=0 means move_end_effector_to()'s own loop contributes zero
    stepSimulation() calls, so every counted call comes from
    open_gripper()/close_gripper() (or none, if apply_command() skips
    them)."""
    call_count = {"n": 0}
    original_step = backend_module.p.stepSimulation

    def counting_step(*args, **kwargs):
        call_count["n"] += 1
        return original_step(*args, **kwargs)

    backend_module.p.stepSimulation = counting_step
    try:
        command = RobotCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_command)
        backend.apply_command(command, steps=0)
    finally:
        backend_module.p.stepSimulation = original_step
    return call_count["n"]


def finger_qpos(backend: PyBulletPandaBackend):
    import pybullet as p
    states = p.getJointStates(backend.robot_id, backend.finger_joint_indices, physicsClientId=backend.client_id)
    return states[0][0], states[1][0]


def main() -> None:
    print("=== 1-5. Gripper actuation is skipped iff the command matches the current state ===")

    backend_a = PyBulletPandaBackend(gui=False)
    backend_a.reset()
    steps = count_gripper_only_steps(backend_a, "open")
    check("reset -> open command -> 0 gripper steps (already open)", steps == 0, f"got {steps}")
    backend_a.shutdown()

    backend_b = PyBulletPandaBackend(gui=False)
    backend_b.reset()
    steps = count_gripper_only_steps(backend_b, "close")
    check("reset -> close command -> 60 gripper steps (real transition)", steps == 60, f"got {steps}")
    left, right = finger_qpos(backend_b)
    check(
        "after close transition, finger qpos reached FINGER_CLOSE_POSITION (no object nearby)",
        abs(left - FINGER_CLOSE_POSITION) < 1e-3 and abs(right - FINGER_CLOSE_POSITION) < 1e-3,
        f"got left={left}, right={right}",
    )

    steps = count_gripper_only_steps(backend_b, "close")
    check("close -> close command -> 0 gripper steps (already closed)", steps == 0, f"got {steps}")

    steps = count_gripper_only_steps(backend_b, "open")
    check("close -> open command -> 60 gripper steps (real transition)", steps == 60, f"got {steps}")
    left, right = finger_qpos(backend_b)
    check(
        "after open transition, finger qpos reached FINGER_OPEN_POSITION",
        abs(left - FINGER_OPEN_POSITION) < 1e-3 and abs(right - FINGER_OPEN_POSITION) < 1e-3,
        f"got left={left}, right={right}",
    )

    steps = count_gripper_only_steps(backend_b, "open")
    check("open -> open command -> 0 gripper steps (already open)", steps == 0, f"got {steps}")
    backend_b.shutdown()
    print()

    print("=== 6. No flapping across repeated identical commands ===")
    backend_c = PyBulletPandaBackend(gui=False)
    backend_c.reset()
    repeat_steps = [count_gripper_only_steps(backend_c, "close") for _ in range(5)]
    check(
        "5x repeated 'close' command: only the first actuates, the rest are skipped",
        repeat_steps == [60, 0, 0, 0, 0],
        f"got {repeat_steps}",
    )
    backend_c.shutdown()
    print()

    print("=== 7. Closing on a grasped object is recognized as 'closed' (qpos != FINGER_CLOSE_POSITION) ===")
    backend_d = PyBulletPandaBackend(gui=False)
    backend_d.reset()
    # Move the object right under the gripper so a close command actually grasps it
    # (GRASP_THRESHOLD=0.05 -- see PyBulletPandaBackend), then close.
    ee_position, _ = backend_d._get_ee_pose()
    backend_d.set_object_position([ee_position[0], ee_position[1], ee_position[2] - 0.01])
    steps = count_gripper_only_steps(backend_d, "close")
    check("first close near the object actuates (60 steps)", steps == 60, f"got {steps}")
    check("object was actually grasped (held_object True)", backend_d._held_object, str(backend_d._held_object))
    left, right = finger_qpos(backend_d)
    check(
        "grasped object stops fingers short of FINGER_CLOSE_POSITION (object half-width, not 0)",
        left > FINGER_CLOSE_POSITION + 1e-3,
        f"left qpos={left} (expected clearly above {FINGER_CLOSE_POSITION})",
    )
    check(
        "but still correctly classified as 'not open' (below GRIPPER_OPEN_QPOS_THRESHOLD)",
        left < GRIPPER_OPEN_QPOS_THRESHOLD,
        f"left qpos={left}, threshold={GRIPPER_OPEN_QPOS_THRESHOLD}",
    )
    steps = count_gripper_only_steps(backend_d, "close")
    check(
        "re-issuing close while holding the object skips actuation (0 steps) -- the fix that actually matters "
        "during lift_object/move_above_bin phases",
        steps == 0,
        f"got {steps}",
    )
    check("object is still held after the skipped call (nothing was disturbed)", backend_d._held_object)
    backend_d.shutdown()
    print()

    print("=== 8. Arm-movement/IK call signature unaffected (regression) ===")
    import inspect
    sig = inspect.signature(PyBulletPandaBackend.move_end_effector_to)
    expected_params = ["self", "target_position", "target_orientation", "steps", "safety_callback",
                        "action_name", "safety_check_interval", "trajectory_callback",
                        "trajectory_record_interval", "step_delay"]
    check(
        "move_end_effector_to()'s signature is unchanged",
        list(sig.parameters.keys()) == expected_params,
        f"got {list(sig.parameters.keys())}",
    )
    backend_e = PyBulletPandaBackend(gui=False)
    backend_e.reset()
    ee_before, _ = backend_e._get_ee_pose()
    command = RobotCommand(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, "open")  # no gripper transition
    backend_e.apply_command(command, steps=40)
    ee_after, _ = backend_e._get_ee_pose()
    check(
        "arm still moves normally when gripper actuation is skipped",
        abs((ee_after[0] - ee_before[0]) - 0.02) < 0.005,
        f"delta_x={ee_after[0] - ee_before[0]}",
    )
    backend_e.shutdown()

    print()
    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
