"""Real VLA-compatible mock server (v0).

Adapter-test server for policy/real_vla_policy_client.py's
RealVLAPolicyClient -- a different purpose from
openvla_server_dummy/dummy_server.py (the plain --policy-backend
fastapi-dummy path used since earlier turns). This server exists to
validate the exact request/response schema a real OpenVLA/VLA model
server would need to speak (configs/real_vla_backend_config.json)
before one exists, by internally reusing the identical
DummyOpenVLAPolicy phase engine dummy_server.py already uses.

  POST /predict  <- Real VLA schema JSON (see configs/real_vla_backend_config.json)
                 -> PolicyOutput-shaped JSON, info.model="real-vla-compatible-mock"
  POST /reset    <- start a new episode (resets phase back to move_to_object)
  GET  /health   <- {"status": "ok", "model": "real-vla-compatible-mock", "version": "v0"}

No real OpenVLA model, no GPU inference, no learned visual reasoning on
the uploaded image here -- see docs/hardware_portability.md for what
actually plugging in a real VLA model server would replace.
"""

import base64
import io
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput

app = FastAPI(title="Real VLA-Compatible Mock Server (v0)")

_policy = DummyOpenVLAPolicy()


class ImagePayload(BaseModel):
    encoding: str = "jpg_base64"
    shape: List[int]
    data: str


class PredictRequest(BaseModel):
    instruction: str
    robot_state: Optional[dict] = None
    task_goal: Optional[dict] = None
    target_object_position: Optional[List[float]] = None
    bin_position: Optional[List[float]] = None
    step_index: int = 0
    phase: Optional[str] = None
    observation_source: Optional[str] = None
    visual_observation: Optional[dict] = None
    image: Optional[ImagePayload] = None
    action_schema: Optional[dict] = None


class PredictResponse(BaseModel):
    action: List[float]
    phase: str
    done: bool = False
    info: dict = {}


def _decode_image(image_payload: Optional[ImagePayload]) -> Optional[np.ndarray]:
    if image_payload is None:
        return None
    image_bytes = base64.b64decode(image_payload.data)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image, dtype=np.uint8)


@app.get("/health")
def health():
    return {"status": "ok", "model": "real-vla-compatible-mock", "version": "v0"}


@app.post("/reset")
def reset():
    _policy.reset()
    return {"status": "reset", "phase": _policy.phase}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    image_array = _decode_image(req.image)

    if req.phase is not None:
        _policy.phase = req.phase

    policy_input = PolicyInput(
        image=image_array,
        instruction=req.instruction,
        robot_state=req.robot_state or {},
        task_goal=req.task_goal or {},
        target_object_position=req.target_object_position,
        bin_position=req.bin_position,
        step_index=req.step_index,
        phase=_policy.phase,
        observation_source=req.observation_source,
        visual_observation=req.visual_observation,
    )

    policy_output = _policy.predict_action(policy_input)

    server_inference_ms = (time.perf_counter() - start) * 1000
    info = dict(policy_output.info or {})
    info["model"] = "real-vla-compatible-mock"
    info["policy_backend"] = "real-vla"
    info["server_inference_ms"] = round(server_inference_ms, 3)

    return PredictResponse(
        action=list(policy_output.action),
        phase=policy_output.phase,
        done=bool(policy_output.done),
        info=info,
    )
