"""Real VLA-compatible policy backend adapter (v0).

Talks to a Real VLA-compatible external server -- which may eventually
be an actual OpenVLA/other VLA model server, or (today)
openvla_server_dummy/real_vla_compatible_server.py, an adapter-test
mock included in this repo -- over the request/response contract
described in configs/real_vla_backend_config.json. Implements the same
BasePolicy/PolicyBackend interface as DummyOpenVLAPolicy and
FastAPIVLAPolicyClient, so run_full_recycling_cell_demo.py's control
loop does not change between --policy-backend local-dummy /
fastapi-dummy / real-vla -- only create_policy_backend() picks a
different implementation.

No real OpenVLA model is loaded or required here. This class is the
adapter layer a real VLA server would sit behind: config-driven image
preprocessing (policy/vla_image_preprocessor.py), action validation/
postprocessing (policy/vla_action_postprocessor.py), latency logging,
and an optional fallback policy for local development without a
running real VLA server. See docs/hardware_portability.md.
"""

import json
import time
from pathlib import Path
from typing import Optional, Union

import requests

from policy.base_policy import BasePolicy
from policy.policy_types import PolicyInput, PolicyOutput
from policy.vla_action_postprocessor import validate_and_postprocess_vla_action
from policy.vla_image_preprocessor import encode_policy_image_for_vla

DEFAULT_CONFIG_PATH = "configs/real_vla_backend_config.json"


class RealVLAPolicyClient(BasePolicy):
    def __init__(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
        fallback_policy: Optional[BasePolicy] = None,
    ):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Real VLA backend config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.backend_name = self.config.get("backend_name", "real_vla_adapter_v0")
        if "server_url" not in self.config:
            raise RuntimeError(f"Real VLA backend config {config_path} is missing required key 'server_url'.")
        self.server_url = self.config["server_url"]
        self.health_url = self.config.get("health_url") or f"{self.server_url.rsplit('/', 1)[0]}/health"
        self.reset_url = f"{self.server_url.rsplit('/', 1)[0]}/reset"
        self.timeout = float(self.config.get("timeout_sec", 10.0))
        self.request_schema = self.config.get("request_schema", {}) or {}
        self.action_schema = self.config.get("action_schema", {}) or {}
        self.fallback_policy = fallback_policy

        self.phase = "move_to_object"
        self.last_info: dict = {}
        self.fallback_used_count = 0

    def check_health(self) -> dict:
        try:
            response = requests.get(self.health_url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to Real VLA server at {self.health_url}. "
                "Is it running? (uvicorn openvla_server_dummy.real_vla_compatible_server:app --port 9000)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Real VLA server health check timed out after {self.timeout}s at {self.health_url}."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Real VLA server health check failed: {exc}") from exc

    def reset(self) -> None:
        self.phase = "move_to_object"
        self.last_info = {}
        if self.fallback_policy is not None:
            self.fallback_policy.reset()
        try:
            response = requests.post(self.reset_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"Warning: could not reset Real VLA server at {self.reset_url}: {exc}")

    def _build_payload(self, policy_input: PolicyInput, image_payload: Optional[dict]) -> dict:
        payload = {
            "instruction": policy_input.instruction,
            "step_index": policy_input.step_index,
            "phase": policy_input.phase if policy_input.phase is not None else self.phase,
            "observation_source": policy_input.observation_source,
            "image": image_payload,
            "action_schema": self.action_schema,
        }
        if self.request_schema.get("include_robot_state", True):
            payload["robot_state"] = policy_input.robot_state
        if self.request_schema.get("include_task_goal", True):
            payload["task_goal"] = policy_input.task_goal
        if self.request_schema.get("include_visual_observation", True):
            payload["visual_observation"] = policy_input.visual_observation
        if self.request_schema.get("include_target_positions", True):
            payload["target_object_position"] = policy_input.target_object_position
            payload["bin_position"] = policy_input.bin_position
        return payload

    def _fallback(self, policy_input: PolicyInput, reason: str) -> PolicyOutput:
        if self.fallback_policy is None:
            raise RuntimeError(
                f"Real VLA server request failed ({reason}) and no fallback backend is configured "
                f"(--real-vla-fallback-backend none). Start the server at {self.server_url}, or set "
                "--real-vla-fallback-backend local-dummy/fastapi-dummy to develop without it."
            )
        self.fallback_used_count += 1
        policy_output = self.fallback_policy.predict_action(policy_input)
        info = dict(policy_output.info or {})
        info["real_vla_request_failed"] = True
        info["fallback_used"] = True
        info["fallback_reason"] = reason
        info["policy_backend"] = "real-vla"
        policy_output.info = info
        self.phase = policy_output.phase
        self.last_info = info
        return policy_output

    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        image_payload, image_debug = encode_policy_image_for_vla(policy_input.image, self.config)
        payload = self._build_payload(policy_input, image_payload)

        start = time.perf_counter()
        try:
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            return self._fallback(policy_input, f"connection_error: {exc}")
        except requests.exceptions.Timeout as exc:
            return self._fallback(policy_input, f"timeout: {exc}")
        except requests.exceptions.RequestException as exc:
            return self._fallback(policy_input, f"request_error: {exc}")
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            return self._fallback(policy_input, f"http_error_{response.status_code}: {exc}")

        try:
            data = response.json()
        except ValueError as exc:
            return self._fallback(policy_input, f"invalid_json: {exc}")

        if "action" not in data:
            return self._fallback(policy_input, f"missing_action_field: {data}")

        try:
            postprocessed_action, postprocess_debug = validate_and_postprocess_vla_action(data["action"], self.config)
        except RuntimeError as exc:
            return self._fallback(policy_input, f"invalid_action: {exc}")

        self.phase = data.get("phase", self.phase)
        info = dict(data.get("info") or {})
        info["policy_backend"] = "real-vla"
        info["inference_latency_ms"] = round(latency_ms, 3)
        info["image_encoding_latency_ms"] = image_debug["encoding_latency_ms"]
        info["fallback_used"] = False
        info["real_vla_request_failed"] = False
        info["used_image_input"] = image_payload is not None
        info["action_postprocess"] = {
            "translation_clipped": postprocess_debug["translation_clipped"],
            "rotation_clipped": postprocess_debug["rotation_clipped"],
            "gripper_normalized": postprocess_debug["gripper_normalized"],
        }
        self.last_info = info

        return PolicyOutput(
            action=postprocessed_action,
            phase=data.get("phase", self.phase),
            done=bool(data.get("done", False)),
            info=info,
        )
