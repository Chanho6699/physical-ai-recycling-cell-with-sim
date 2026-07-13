"""Safety Pause/Resume v0 probe.

Confirms the mock-timed hand-intrusion pause/resume state machine used by
run_full_recycling_cell_demo.py --safety-mode pause-resume in isolation:
no YOLO, no ArUco, no external camera -- object_position is given
directly and only a DummyOpenVLAPolicy control loop runs against
PyBulletPandaBackend, with a MockHandIntrusionMonitor firing between
--mock-hand-start-step and --mock-hand-end-step.

Verifies:
  - the robot never applies a command while the mock hand is "present"
    (or while still waiting out --safety-resume-stable-steps of clear
    checks before resuming)
  - the episode still reaches task_status=success afterward, i.e. pausing
    and resuming does not abort or fail the run
"""

import argparse
from pathlib import Path

from action_adapter.adapter_v0 import ActionAdapter
from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput
from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from safety.mock_hand_intrusion_monitor import MockHandIntrusionMonitor
from safety.safety_gate import SafetyGate

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-position", type=float, nargs=3, default=[0.40, -0.10, 0.05])
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--max-policy-steps", type=int, default=80)
    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--mock-hand-start-step", type=int, default=5)
    parser.add_argument("--mock-hand-end-step", type=int, default=10)
    parser.add_argument("--safety-resume-stable-steps", type=int, default=3)

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_position = list(args.object_position)
    task_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    print("=== Safety Pause/Resume Probe ===")

    backend = PyBulletPandaBackend(gui=args.gui)
    try:
        try:
            backend.reset()
        except Exception as exc:
            print(f"Panda backend reset failed: {exc}")
            print("FAIL")
            return

        backend.set_object_type(args.object_type)
        state = backend.set_object_position(object_position)
        bin_position = state["bin_position"]

        action_adapter = ActionAdapter()
        policy = DummyOpenVLAPolicy()
        policy.reset()

        hand_intrusion_gate = SafetyGate(
            MockHandIntrusionMonitor(start_step=args.mock_hand_start_step, end_step=args.mock_hand_end_step)
        )

        task_goal_dict = {
            "instruction": DEFAULT_INSTRUCTION,
            "action": "pick_and_place",
            "target_object": args.object_type,
            "target_bin": "plastic_bin",
        }

        safety_state = "running"
        stable_clear_steps = 0
        safety_pause_count = 0
        safety_resume_count = 0
        paused_steps = 0
        robot_action_applied_during_pause = False

        final_state = state
        for step_index in range(args.max_policy_steps):
            robot_state = backend.get_state()

            hand_intrusion_gate.safety_monitor.set_step(step_index)
            hand_gate_result = hand_intrusion_gate.check(task_frame, "safety_check")
            hand_detected = hand_gate_result.decision.emergency_stop

            paused_this_step = False
            if hand_detected:
                stable_clear_steps = 0
                if safety_state == "running":
                    safety_pause_count += 1
                safety_state = "paused_by_safety"
                paused_steps += 1
                paused_this_step = True
                print(f"[step {step_index:02d}] safety_pause/safety_still_paused hand_detected=True")
            elif safety_state in ("paused_by_safety", "resuming"):
                stable_clear_steps += 1
                paused_steps += 1
                paused_this_step = True
                if stable_clear_steps >= args.safety_resume_stable_steps:
                    safety_state = "running"
                    safety_resume_count += 1
                    stable_clear_steps = 0
                    print(f"[step {step_index:02d}] safety_resume hand_detected=False")
                else:
                    safety_state = "resuming"
                    print(f"[step {step_index:02d}] safety_still_paused (resuming) hand_detected=False")

            if paused_this_step:
                continue

            policy_input = PolicyInput(
                image=task_frame,
                instruction=DEFAULT_INSTRUCTION,
                robot_state=robot_state,
                task_goal=task_goal_dict,
                target_object_position=object_position,
                bin_position=bin_position,
                step_index=step_index,
                phase=policy.phase,
            )
            policy_output = policy.predict_action(policy_input)
            robot_command = action_adapter.convert(policy_output.action)
            final_state = backend.apply_command(robot_command, steps=args.steps_per_action)

            if final_state["task_status"] == "success" or policy_output.done:
                break

        final_status = final_state["task_status"]
        success = final_status == "success"

        print(f"safety_pause_count: {safety_pause_count}")
        print(f"safety_resume_count: {safety_resume_count}")
        print(f"paused_steps: {paused_steps}")
        print(f"robot_action_applied_during_pause: {robot_action_applied_during_pause}")
        print(f"final_status: {final_status}")
        print("PASS" if success and safety_pause_count >= 1 and safety_resume_count >= 1 else "FAIL")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
