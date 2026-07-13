# Deprecated:
# This demo used the old /predict_action endpoint, which no longer
# exists on openvla_server_dummy/dummy_server.py (replaced by /predict,
# /health, /reset -- see policy/fastapi_vla_policy_client.py).
# Use benchmark/probe_fastapi_vla_policy_client.py or
# run_full_recycling_cell_demo.py with --policy-backend fastapi-dummy instead.

import time
import requests


SERVER_URL = "http://localhost:8000/predict_action"


def request_action(instruction: str, image_path: str | None = None):
    payload = {
        "instruction": instruction,
        "image_path": image_path,
    }

    start = time.perf_counter()
    response = requests.post(SERVER_URL, json=payload, timeout=10)
    total_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    data = response.json()

    print("=== OpenVLA Client Result ===")
    print(f"Instruction: {data['instruction']}")
    print(f"Model: {data['model']}")
    print(f"Action: {data['action']}")
    print(f"Server inference ms: {data['inference_ms']:.4f}")
    print(f"HTTP round-trip ms: {total_ms:.4f}")

    return data


if __name__ == "__main__":
    request_action(
        instruction="Pick the plastic cup and place it in the plastic recycling bin.",
        image_path=None,
    )
