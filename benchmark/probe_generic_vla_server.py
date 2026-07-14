"""Generic VLA Backend server probe (v0).

Checks that a running vla_server.generic_vla_server instance is
reachable through RealVLAPolicyClient and reports which model_family/
model_status/adapter it's running, regardless of whether that family
is "mock-action" (should always succeed), "smolvla" (succeeds once a
real checkpoint is loaded, gracefully load_failed otherwise), or
"openvla" (currently always structured-errors with
openvla_action_adapter_required by design -- see
vla_adapters/openvla_adapter.py). In every non-mock-action case where
the server can't produce an executable action, RealVLAPolicyClient's
fallback should still produce a valid 7-DoF action.
"""

import argparse

import numpy as np
import requests

from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.fastapi_vla_policy_client import FastAPIVLAPolicyClient
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient

DEFAULT_CONFIG = "configs/vla_backend_smolvla_config.json"
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fallback-backend", choices=["none", "local-dummy", "fastapi-dummy"], default="local-dummy"
    )
    parser.add_argument("--policy-server-url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--policy-request-timeout", type=float, default=5.0)
    parser.add_argument("--with-image", action="store_true")
    return parser.parse_args()


def build_fallback_policy(args):
    if args.fallback_backend == "none":
        return None
    if args.fallback_backend == "fastapi-dummy":
        return FastAPIVLAPolicyClient(server_url=args.policy_server_url, timeout=args.policy_request_timeout)
    return DummyOpenVLAPolicy()


def main() -> None:
    args = parse_args()

    print("=== Generic VLA Server Probe ===")
    print(f"vla_config: {args.vla_config}")

    fallback_policy = build_fallback_policy(args)
    try:
        client = RealVLAPolicyClient(config_path=args.vla_config, fallback_policy=fallback_policy)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc))
        print("FAIL")
        return

    print(f"server_url: {client.server_url}")

    model_family = None
    model_status = None
    adapter = None
    health_ok = False
    try:
        response = requests.get(client.health_url, timeout=client.timeout)
        response.raise_for_status()
        health = response.json()
        health_ok = health.get("status") == "ok"
        model_family = health.get("model_family")
        model_status = health.get("model_status")
        adapter = health.get("adapter")
        print(f"health: {health.get('status')}")
        print(f"model_family: {model_family}")
        print(f"model_status: {model_status}")
        print(f"model_id_or_path: {health.get('model_id_or_path')}")
        print(f"adapter: {adapter}")
    except requests.exceptions.RequestException as exc:
        print(f"health check failed: {exc}")
        print("health: unreachable")

    client.reset()

    dummy_image = None
    if args.with_image:
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        dummy_image[100:140, 140:180] = [80, 140, 220]

    policy_input = PolicyInput(
        image=dummy_image,
        instruction=DEFAULT_INSTRUCTION,
        robot_state={"end_effector_position": [0.5, 0.0, 0.5], "held_object": False, "task_status": "running"},
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

    info = policy_output.info or {}
    action = policy_output.action or []
    action_len = len(action)
    fallback_used = bool(info.get("fallback_used", False))

    print(f"policy_backend: {info.get('policy_backend')}")
    print(f"action_len: {action_len}")
    print(f"fallback_used: {fallback_used}")
    print(f"inference_latency_ms: {info.get('inference_latency_ms')}")
    if fallback_used:
        print(f"fallback_backend: {args.fallback_backend}")
        print(f"fallback_reason: {info.get('fallback_reason')}")

    if action_len == 7 and not fallback_used:
        print("PASS")
    elif action_len == 7 and fallback_used:
        print("PASS_WITH_FALLBACK (adapter_required or model_not_loaded -- expected until a real checkpoint is loaded)")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
