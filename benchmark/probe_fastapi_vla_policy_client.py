"""FastAPI VLA policy client probe (v0).

Checks that a running openvla_server_dummy.dummy_server is reachable
and returns a well-formed 7-DoF action before wiring
FastAPIVLAPolicyClient into run_full_recycling_cell_demo.py
(--policy-backend fastapi-dummy). No PyBullet, no YOLO, no ArUco here --
just the HTTP round trip with a synthetic PolicyInput.
"""

import argparse

import numpy as np

from policy.fastapi_vla_policy_client import FastAPIVLAPolicyClient
from policy.policy_types import PolicyInput

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/predict"
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-server-url", type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument("--policy-request-timeout", type=float, default=5.0)
    parser.add_argument("--with-image", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== FastAPI VLA Policy Client Probe ===")
    print(f"policy_server_url: {args.policy_server_url}")

    client = FastAPIVLAPolicyClient(server_url=args.policy_server_url, timeout=args.policy_request_timeout)

    try:
        health = client.check_health()
    except RuntimeError as exc:
        print(str(exc))
        print("FAIL")
        return

    print(f"health: {health.get('status')}")
    print(f"health_response: {health}")

    client.reset()

    dummy_image = None
    if args.with_image:
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        dummy_image[100:140, 140:180] = [80, 140, 220]

    policy_input = PolicyInput(
        image=dummy_image,
        instruction=DEFAULT_INSTRUCTION,
        robot_state={
            "end_effector_position": [0.5, 0.0, 0.5],
            "held_object": False,
            "task_status": "running",
        },
        task_goal={"action": "pick_and_place", "target_object": "plastic_bottle", "target_bin": "plastic_bin"},
        target_object_position=[0.4, -0.1, 0.05],
        bin_position=[0.3, 0.35, 0.05],
        step_index=0,
        phase="move_to_object",
        observation_source="wrist" if dummy_image is not None else None,
    )

    try:
        policy_output = client.predict_action(policy_input)
    except RuntimeError as exc:
        print(str(exc))
        print("FAIL")
        return

    action_len = len(policy_output.action)
    inference_latency_ms = (policy_output.info or {}).get("inference_latency_ms")

    print(f"action: {policy_output.action}")
    print(f"action_len: {action_len}")
    print(f"phase: {policy_output.phase}")
    print(f"inference_latency_ms: {inference_latency_ms}")

    success = health.get("status") == "ok" and action_len == 7
    print("PASS" if success else "FAIL")


if __name__ == "__main__":
    main()
