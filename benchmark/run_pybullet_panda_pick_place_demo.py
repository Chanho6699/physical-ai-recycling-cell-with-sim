"""Franka Panda URDF pick-and-place smoke test (v1).

  reset -> move to object -> close gripper -> move to bin -> open gripper

Uses fixed, hand-picked object/bin coordinates (not Real2Sim-mapped yet --
that wiring is a future step, see docs/07_pybullet_panda_backend.md).
"""

import argparse
import time

from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

# How far above the bin's stored position to hover before opening the
# gripper. The bin is a solid box (not a hollow container); descending to
# its exact z would push the held object into its collision volume and
# stall the IK-driven motion, so we approach from just above it instead.
BIN_APPROACH_CLEARANCE = 0.05

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    parser.add_argument("--print-joint-info", action="store_true")

    parser.add_argument("--object-x", type=float, default=0.45)
    parser.add_argument("--object-y", type=float, default=0.0)
    parser.add_argument("--object-z", type=float, default=0.05)

    parser.add_argument("--bin-x", type=float, default=0.3)
    parser.add_argument("--bin-y", type=float, default=0.35)
    parser.add_argument("--bin-z", type=float, default=0.05)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    object_position = [args.object_x, args.object_y, args.object_z]
    bin_position = [args.bin_x, args.bin_y, args.bin_z]
    bin_approach_position = [args.bin_x, args.bin_y, args.bin_z + BIN_APPROACH_CLEARANCE]

    backend = PyBulletPandaBackend(gui=args.gui)
    try:
        state = backend.reset()
        print("=== Reset State ===")
        print(state)

        if args.print_joint_info:
            backend.print_joint_info()

        backend.set_object_position(object_position)
        backend.set_bin_position(bin_position)

        print("\n=== Move to object ===")
        state = backend.move_end_effector_to(object_position)
        print(state)

        print("\n=== Close gripper ===")
        state = backend.close_gripper()
        print(state)

        print("\n=== Move to bin ===")
        state = backend.move_end_effector_to(bin_approach_position)
        print(state)

        print("\n=== Open gripper ===")
        state = backend.open_gripper()
        print(state)

        print("\n=== Final State ===")
        print(state)

        final_status = state["task_status"]
        print(f"\n=== Demo finished: task_status={final_status} ===")
        print("PASS" if final_status == "success" else "FAIL")

        if KEEP_GUI_OPEN and args.gui:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
