"""PyBullet safety-gate demo.

  frame = frame_source.get_frame()
  decision = safety_monitor.check(frame)
  if decision.emergency_stop: block the command
  else: backend.apply_command(command)

MockSafetyMonitor stands in for a future YOLOSafetyMonitor -- no real
YOLO/ONNX/TensorRT, no real OpenVLA, no ROS 2, no Real2Sim mapping here
yet. Goal is just to validate the gate structure itself.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from benchmark.run_pybullet_pick_place_demo import STEP_SEQUENCE
from robot_sim.pybullet_backend import PyBulletBackend
from safety.mock_safety_monitor import MockSafetyMonitor
from vision.sim_camera_source import SimCameraSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "pybullet_safety_gate_demo_log.jsonl"

GUI_MODE = True
KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_demo(simulate_hazard: bool, monitor: str = "mock") -> None:
    backend = PyBulletBackend(gui=GUI_MODE)
    frame_source = None
    try:
        state = backend.reset()
        print("=== Reset State ===")
        print(state)

        frame_source = SimCameraSource(physics_client_id=backend.client_id)

        if monitor == "yolo":
            from safety.yolo_safety_monitor import YOLOSafetyMonitor

            safety_monitor = YOLOSafetyMonitor()
            if simulate_hazard:
                print("--simulate-hazard is ignored with --monitor yolo (mock-only option).")
        elif monitor == "onnx":
            from safety.onnx_yolo_safety_monitor import ONNXRuntimeYOLOSafetyMonitor

            safety_monitor = ONNXRuntimeYOLOSafetyMonitor()
            if simulate_hazard:
                print("--simulate-hazard is ignored with --monitor onnx (mock-only option).")
        else:
            safety_monitor = MockSafetyMonitor(simulate_hazard=simulate_hazard)

        state_before = state
        for step_index, (description, robot_command) in enumerate(STEP_SEQUENCE, start=1):
            print(f"\n=== Step {step_index}: {description} ===")

            frame = frame_source.get_frame()
            decision = safety_monitor.check(frame)
            print(f"safety_decision: emergency_stop={decision.emergency_stop}, reason={decision.reason}")

            command_applied = False
            state_after = state_before

            if decision.emergency_stop:
                print(f"!!! EMERGENCY STOP at step '{description}' ({decision.reason}) - command blocked !!!")
            else:
                state_after = backend.apply_command(robot_command)
                command_applied = True
                print(f"state_after: {state_after}")

            append_log(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "step_name": description,
                    "safety_decision": {
                        "emergency_stop": decision.emergency_stop,
                        "reason": decision.reason,
                        "detections": decision.detections,
                    },
                    "command_applied": command_applied,
                    "state_before": state_before,
                    "state_after": state_after,
                }
            )

            if decision.emergency_stop:
                break

            state_before = state_after

        final_status = state_before.get("task_status")
        print(f"\n=== Demo finished: task_status={final_status} ===")

        if KEEP_GUI_OPEN:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        if frame_source is not None:
            frame_source.close()
        backend.close()

    print(f"\nSaved log to: {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-hazard", action="store_true")
    parser.add_argument("--monitor", choices=["mock", "yolo", "onnx"], default="mock")
    args = parser.parse_args()

    run_demo(simulate_hazard=args.simulate_hazard, monitor=args.monitor)
