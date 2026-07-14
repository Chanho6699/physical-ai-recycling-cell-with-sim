"""FastAPI VLA policy backend client (v0).

Talks to openvla_server_dummy.dummy_server's /predict endpoint the same
way run_full_recycling_cell_demo.py's local-dummy backend talks to
DummyOpenVLAPolicy directly -- same PolicyInput in, same PolicyOutput
out (implements BasePolicy) -- so the control loop doesn't need to know
or care which backend is actually driving the arm.

No real OpenVLA model here; this is the network-shaped placeholder side
of the same v0. A real VLA inference server could sit behind the exact
same request/response contract without this client changing at all.
"""

import base64
import io
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput

DEFAULT_SERVER_URL = "http://127.0.0.1:8000/predict"
DEFAULT_TIMEOUT = 5.0
DEFAULT_JPEG_QUALITY = 80


class FastAPIVLAPolicyClient(BasePolicy):
    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        timeout: float = DEFAULT_TIMEOUT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self.phase = "move_to_object"
        self.last_info: dict = {}

        # /predict -> /health, /predict -> /reset (sibling endpoints on
        # the same dummy_server.py app).
        base_url = server_url.rsplit("/", 1)[0]
        self.health_url = f"{base_url}/health"
        self.reset_url = f"{base_url}/reset"

    def check_health(self) -> dict:
        try:
            response = requests.get(self.health_url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to FastAPI VLA policy server at {self.health_url}. "
                "Is it running? (uvicorn openvla_server_dummy.dummy_server:app --port 8000)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"FastAPI VLA policy server health check timed out after {self.timeout}s "
                f"at {self.health_url}."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"FastAPI VLA policy server health check failed: {exc}") from exc

    def reset(self) -> None:
        self.phase = "move_to_object"
        self.last_info = {}
        try:
            response = requests.post(self.reset_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"Warning: could not reset FastAPI VLA policy server at {self.reset_url}: {exc}")

    def _encode_image(self, image) -> Optional[dict]:
        if image is None:
            return None
        array = np.asarray(image)
        pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        encoded_data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"encoding": "jpg_base64", "shape": list(array.shape), "data": encoded_data}

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        payload = {
            "instruction": policy_input.instruction,
            "robot_state": policy_input.robot_state,
            "task_goal": policy_input.task_goal,
            "target_object_position": policy_input.target_object_position,
            "bin_position": policy_input.bin_position,
            "step_index": policy_input.step_index,
            "phase": policy_input.phase if policy_input.phase is not None else self.phase,
            "observation_source": policy_input.observation_source,
            "visual_observation": policy_input.visual_observation,
            "image": self._encode_image(policy_input.image),
        }

        start = time.perf_counter()
        try:
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to FastAPI VLA policy server at {self.server_url}. "
                "Is it running? (uvicorn openvla_server_dummy.dummy_server:app --port 8000)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"FastAPI VLA policy server at {self.server_url} did not respond within "
                f"{self.timeout}s (--policy-request-timeout)."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"FastAPI VLA policy server request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"FastAPI VLA policy server returned an error ({response.status_code}): {response.text}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"FastAPI VLA policy server returned invalid JSON: {exc}") from exc

        if "action" not in data:
            raise RuntimeError(f"FastAPI VLA policy server response is missing 'action': {data}")

        self.phase = data.get("phase", self.phase)
        info = dict(data.get("info") or {})
        info["inference_latency_ms"] = round(latency_ms, 3)
        self.last_info = info

        return PolicyOutput(
            action=data["action"],
            phase=data.get("phase", self.phase),
            done=bool(data.get("done", False)),
            info=info,
        )
