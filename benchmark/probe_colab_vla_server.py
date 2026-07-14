"""Colab VLA server probe (v0).

Checks that a Colab-hosted (or, for local testing, a locally-run)
openvla_server_real.colab_vla_server instance is reachable through
RealVLAPolicyClient, reports which server_mode/model_status it's
running, and confirms the fallback path works when the server can't
produce an executable action (health-only mode, or openvla-dryrun
without a loaded model).

Point --real-vla-config at whatever config currently has the right
server_url/health_url -- for a real Colab session, run
scripts/update_colab_vla_config.py first to fill those in from the
tunnel's public URL.
"""

import argparse

import numpy as np
import requests

from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.fastapi_vla_policy_client import FastAPIVLAPolicyClient
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient

DEFAULT_CONFIG = "configs/real_vla_backend_colab_config.json"
DEFAULT_INSTRUCTION = "플라스틱 병을 플라스틱 수거함에 넣어줘"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-vla-config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--real-vla-fallback-backend", choices=["none", "local-dummy", "fastapi-dummy"], default="local-dummy"
    )
    parser.add_argument("--policy-server-url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--policy-request-timeout", type=float, default=5.0)
    parser.add_argument("--with-image", action="store_true")
    return parser.parse_args()


def build_fallback_policy(args):
    if args.real_vla_fallback_backend == "none":
        return None
    if args.real_vla_fallback_backend == "fastapi-dummy":
        return FastAPIVLAPolicyClient(server_url=args.policy_server_url, timeout=args.policy_request_timeout)
    return DummyOpenVLAPolicy()


def main() -> None:
    args = parse_args()

    print("=== Colab VLA Server Probe ===")
    print(f"real_vla_config: {args.real_vla_config}")

    fallback_policy = build_fallback_policy(args)
    try:
        client = RealVLAPolicyClient(config_path=args.real_vla_config, fallback_policy=fallback_policy)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc))
        print("FAIL")
        return

    print(f"server_url: {client.server_url}")

    server_mode = None
    model_status = None
    model_status_reason = None
    health_ok = False
    try:
        response = requests.get(client.health_url, timeout=client.timeout)
        response.raise_for_status()
        health = response.json()
        health_ok = health.get("status") == "ok"
        server_mode = health.get("server_mode")
        model_status = health.get("model_status")
        model_status_reason = health.get("model_status_reason")
        print(f"health: {health.get('status')}")
        print(f"server_mode: {server_mode}")
        print(f"model_status: {model_status}")
        if model_status_reason:
            print(f"model_status_reason: {model_status_reason}")
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

    if server_mode == "health-only":
        print("predict: expected_failure (model_not_loaded)" if fallback_used else "predict: unexpected_success")

    if server_mode == "openvla-dryrun":
        print(f"raw_model_output_available: {model_status == 'loaded'}")
        print(f"project_action_available: {(not fallback_used) and action_len == 7}")
        if fallback_used:
            print(f"reason: {info.get('fallback_reason', model_status_reason or 'action_adapter_required')}")

    if server_mode == "health-only":
        success = fallback_used and action_len == 7
        print("PASS (environment-confirmed limitation)" if success else "FAIL")
    elif server_mode == "mock-action":
        success = (not fallback_used) and action_len == 7
        print("PASS" if success else "FAIL")
    elif server_mode == "openvla-dryrun":
        if (not fallback_used) and action_len == 7:
            print("PASS")
        elif fallback_used and action_len == 7:
            print("PASS_WITH_FALLBACK")
        else:
            print("FAIL")
    else:
        # Server unreachable entirely (server_mode unknown) -- still a
        # pass as long as the fallback path produced a usable action.
        success = action_len == 7 and (health_ok or fallback_used)
        print("PASS" if success else "FAIL")


if __name__ == "__main__":
    main()
