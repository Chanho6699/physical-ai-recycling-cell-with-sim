"""Direct RobotCommand pick/place demo for PyBulletBackend.

No RuleBasedTaskParser, no Dummy OpenVLA server, no ActionAdapter here -
a fixed RobotCommand sequence is applied directly to PyBulletBackend to
check that the distance-based grasp/place state machine works.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from action_adapter.adapter_v0 import RobotCommand
from robot_sim.camera_utils import capture_pybullet_camera, save_rgb_image
from robot_sim.pybullet_backend import PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "pybullet_pick_place_demo_log.jsonl"
FINAL_CAMERA_PATH = PROJECT_ROOT / "results" / "camera" / "pybullet_pick_place_final.png"

GUI_MODE = True
KEEP_GUI_OPEN = True
KEEP_SECONDS = 30

# (description, RobotCommand) sequence:
#   1. approach the object (gripper open)
#   2. close the gripper on the object -> grasp
#   3. carry the object to the bin (gripper stays closed so it keeps following)
#   4. open the gripper over the bin -> place
STEP_SEQUENCE = [
    (
        "approach_object",
        RobotCommand(
            target_dx=0.5, target_dy=0.0, target_dz=0.03,
            target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
            gripper_command="open",
        ),
    ),
    (
        "grasp_object",
        RobotCommand(
            target_dx=0.0, target_dy=0.0, target_dz=0.0,
            target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
            gripper_command="close",
        ),
    ),
    (
        "carry_to_bin",
        RobotCommand(
            target_dx=-0.5, target_dy=0.6, target_dz=-0.43,
            target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
            gripper_command="close",
        ),
    ),
    (
        "place_in_bin",
        RobotCommand(
            target_dx=0.0, target_dy=0.0, target_dz=0.0,
            target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0,
            gripper_command="open",
        ),
    ),
]


def append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_demo() -> None:
    backend = PyBulletBackend(gui=GUI_MODE)
    try:
        state_before_all = backend.reset()
        print("=== Reset State ===")
        print(state_before_all)

        state_before = state_before_all
        for step_index, (description, robot_command) in enumerate(STEP_SEQUENCE, start=1):
            print(f"\n=== Step {step_index}: {description} ===")
            print(f"RobotCommand: {robot_command}")

            state_after = backend.apply_command(robot_command)

            print(f"state_before: {state_before}")
            print(f"state_after:  {state_after}")
            print(f"task_status: {state_after['task_status']}")
            print(f"last_event:  {state_after['last_event']}")

            append_log(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "simulator_backend": "pybullet",
                    "step": step_index,
                    "description": description,
                    "robot_command": asdict(robot_command),
                    "state_before": state_before,
                    "state_after": state_after,
                    "task_status": state_after["task_status"],
                    "last_event": state_after["last_event"],
                }
            )

            state_before = state_after

        final_status = state_before["task_status"]
        print(f"\n=== Demo finished: task_status={final_status} ===")
        print("PASS" if final_status == "success" else "FAIL")

        final_image = capture_pybullet_camera(physics_client_id=backend.client_id)
        saved_path = save_rgb_image(final_image, str(FINAL_CAMERA_PATH))
        print(f"Saved final camera image to: {saved_path}")

        if KEEP_GUI_OPEN:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        backend.close()

    print(f"\nSaved log to: {LOG_PATH}")


if __name__ == "__main__":
    run_demo()
