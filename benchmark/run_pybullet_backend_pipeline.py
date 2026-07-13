# Deprecated:
# This demo used the old /predict_action endpoint, which no longer
# exists on openvla_server_dummy/dummy_server.py (replaced by /predict,
# /health, /reset -- see policy/fastapi_vla_policy_client.py). It also
# predates TaskGoal/Real2Sim/PyBulletPandaBackend and still drives the
# early sphere-based PyBulletBackend directly.
# Use benchmark/probe_fastapi_vla_policy_client.py or
# run_full_recycling_cell_demo.py with --policy-backend fastapi-dummy instead.

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from action_adapter.adapter_v0 import ActionAdapter
from llm_agent.rule_based_parser import RuleBasedTaskParser
from robot_sim.pybullet_backend import PyBulletBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "pybullet_backend_pipeline_log.jsonl"

SERVER_HEALTH_URL = "http://localhost:8000/health"
PREDICT_ACTION_URL = "http://localhost:8000/predict_action"

DEFAULT_COMMAND = "플라스틱 컵을 플라스틱 수거함에 넣어줘"

# Set to True to watch the PyBullet GUI window instead of running headless.
GUI_MODE = True
# If True, hold the GUI window open (Enter to close, or a timed fallback)
# instead of closing it right after the single command is applied.
KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def check_server_health() -> None:
    response = requests.get(SERVER_HEALTH_URL, timeout=5)
    response.raise_for_status()

    print("=== Server Health ===")
    print(response.json())


def request_action(instruction: str, image_path: Optional[str] = None) -> dict:
    payload = {
        "instruction": instruction,
        "image_path": image_path,
    }

    start = time.perf_counter()
    response = requests.post(PREDICT_ACTION_URL, json=payload, timeout=10)
    http_round_trip_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    data = response.json()
    data["http_round_trip_ms"] = http_round_trip_ms

    return data


def append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(user_command: str) -> None:
    parser = RuleBasedTaskParser()
    task_goal = parser.parse(user_command)

    print("=== Task Goal ===")
    print(json.dumps(asdict(task_goal), ensure_ascii=False, indent=2))

    check_server_health()

    print("\n=== Request Action ===")
    action_response = request_action(instruction=task_goal.vla_instruction)

    action = action_response["action"]
    print(f"VLA instruction: {task_goal.vla_instruction}")
    print(f"Model: {action_response['model']}")
    print(f"Action: {action}")
    print(f"Server inference ms: {action_response['inference_ms']:.4f}")
    print(f"HTTP round-trip ms: {action_response['http_round_trip_ms']:.4f}")

    print("\n=== Convert Action ===")
    adapter = ActionAdapter(position_scale=1.0, rotation_scale=1.0)
    robot_command = adapter.convert(action)
    print(robot_command)

    print("\n=== Apply to PyBullet Backend ===")
    backend = PyBulletBackend(gui=GUI_MODE)
    try:
        state_before = backend.reset()
        state_after = backend.apply_command(robot_command)

        if KEEP_GUI_OPEN:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        backend.close()

    print(f"State before: {state_before}")
    print(f"State after:  {state_after}")

    log_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "simulator_backend": "pybullet",
        "user_command": user_command,
        "task_goal": asdict(task_goal),
        "action": action,
        "robot_command": asdict(robot_command),
        "state_before": state_before,
        "state_after": state_after,
        "server_inference_ms": action_response["inference_ms"],
        "http_round_trip_ms": action_response["http_round_trip_ms"],
    }

    append_log(log_record)

    print(f"\nSaved log to: {LOG_PATH}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_command = " ".join(sys.argv[1:])
    else:
        user_command = input(
            f'한국어 명령을 입력하세요 (엔터만 누르면 기본값 사용: "{DEFAULT_COMMAND}"): '
        ).strip()
        if not user_command:
            user_command = DEFAULT_COMMAND

    run_pipeline(user_command)
