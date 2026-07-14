"""Real VLA policy client probe (v0).

Checks that a running openvla_server_dummy.real_vla_compatible_server
(or, eventually, a real VLA server speaking the same schema) is
reachable through RealVLAPolicyClient and returns a well-formed 7-DoF
action, before wiring --policy-backend real-vla into
run_full_recycling_cell_demo.py. Also exercises the fallback path: if
the server is unreachable and a fallback backend is configured,
predict_action() should still return a valid action (via the fallback
policy) instead of raising -- run this same command with the server
stopped to see that path.
"""

import argparse

import numpy as np

from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.fastapi_vla_policy_client import FastAPIVLAPolicyClient
from policy.policy_types import PolicyInput
from policy.real_vla_policy_client import RealVLAPolicyClient

DEFAULT_CONFIG = "configs/real_vla_backend_config.json"
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

    print("=== Real VLA Policy Client Probe ===")
    print(f"real_vla_config: {args.real_vla_config}")
    print(f"real_vla_fallback_backend: {args.real_vla_fallback_backend}")

    fallback_policy = build_fallback_policy(args)
    try:
        client = RealVLAPolicyClient(config_path=args.real_vla_config, fallback_policy=fallback_policy)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc))
        print("FAIL")
        return

    print(f"server_url: {client.server_url}")

    health_ok = False
    try:
        health = client.check_health()
        health_ok = health.get("status") == "ok"
        print(f"health: {health.get('status')}")
        print(f"health_response: {health}")
    except RuntimeError as exc:
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
    action_len = len(policy_output.action)
    fallback_used = bool(info.get("fallback_used", False))
    real_vla_request_failed = bool(info.get("real_vla_request_failed", False))

    print(f"action: {policy_output.action}")
    print(f"action_len: {action_len}")
    print(f"phase: {policy_output.phase}")
    print(f"policy_backend: {info.get('policy_backend')}")
    print(f"model: {info.get('model')}")
    print(f"real_vla_request_failed: {real_vla_request_failed}")
    print(f"fallback_used: {fallback_used}")
    if fallback_used:
        print(f"fallback_backend: {args.real_vla_fallback_backend}")
    print(f"inference_latency_ms: {info.get('inference_latency_ms')}")
    print(f"image_encoding_latency_ms: {info.get('image_encoding_latency_ms')}")
    if "action_postprocess" in info:
        print(f"postprocess_ok: True")
        print(f"action_postprocess: {info['action_postprocess']}")

    success = action_len == 7 and (health_ok or fallback_used)
    print("PASS" if success else "FAIL")


if __name__ == "__main__":
    main()
