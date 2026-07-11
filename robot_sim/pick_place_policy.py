"""Reusable dynamic pick-and-place policy for PyBulletBackend.

Builds each RobotCommand from the *current* backend state (object/bin
position) rather than a fixed command sequence, so it works regardless of
where the recyclable object was placed -- e.g. via Real2Sim mapping,
where the object rarely ends up at the default spawn position
[0.5, 0.0, 0.53] that a fixed sequence would assume.
"""

from action_adapter.adapter_v0 import RobotCommand


def make_move_command(current_position: list, target_position: list, gripper: str) -> RobotCommand:
    dx = target_position[0] - current_position[0]
    dy = target_position[1] - current_position[1]
    dz = target_position[2] - current_position[2]

    return RobotCommand(
        target_dx=dx,
        target_dy=dy,
        target_dz=dz,
        target_droll=0.0,
        target_dpitch=0.0,
        target_dyaw=0.0,
        gripper_command=gripper,
    )


def run_dynamic_pick_place(backend) -> dict:
    print("\n=== Step 1: approach_object ===")
    state_before = backend.get_state()
    command = make_move_command(
        state_before["end_effector_position"], state_before["object_position"], gripper="open"
    )
    state_after = backend.apply_command(command)
    print(f"state_after: {state_after}")

    print("\n=== Step 2: grasp_object ===")
    state_before = backend.get_state()
    ee_pos = state_before["end_effector_position"]
    command = make_move_command(ee_pos, ee_pos, gripper="close")
    state_after = backend.apply_command(command)
    print(f"state_after: {state_after}")

    print("\n=== Step 3: carry_object_to_bin ===")
    state_before = backend.get_state()
    command = make_move_command(
        state_before["end_effector_position"], state_before["bin_position"], gripper="close"
    )
    state_after = backend.apply_command(command)
    print(f"state_after: {state_after}")

    print("\n=== Step 4: place_object ===")
    state_before = backend.get_state()
    ee_pos = state_before["end_effector_position"]
    command = make_move_command(ee_pos, ee_pos, gripper="open")
    state_after = backend.apply_command(command)
    print(f"state_after: {state_after}")

    return state_after
