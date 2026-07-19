"""FastAPI dummy VLA policy server (v0).

Hosts the exact same DummyOpenVLAPolicy phase state machine used by
run_full_recycling_cell_demo.py's local-dummy backend, just behind an
HTTP endpoint shaped like a real VLA inference server would be:

  POST /predict  <- PolicyInput-shaped JSON (image as base64 JPEG)
                 -> PolicyOutput-shaped JSON (action/phase/done/info)
  POST /reset    <- start a new episode (resets phase back to move_to_object)
  GET  /health   <- {"status": "ok", "model": "dummy-openvla-fastapi", "version": "v0"}

Reusing DummyOpenVLAPolicy itself (not a reimplementation of its phase
logic) is deliberate: it's what makes --policy-backend fastapi-dummy
produce the same final_status=success as --policy-backend local-dummy
in run_full_recycling_cell_demo.py -- both are driven by the identical
phase/action code, just called in-process vs. over HTTP.

No real OpenVLA model, no GPU inference, no learned visual reasoning on
the uploaded image here -- this is a network-shaped placeholder so a
real VLA inference server can be swapped in behind the same /predict
contract later.

Single global policy instance -- fine for this v0's "one client driving
one episode at a time" use case, not meant for concurrent multi-episode
serving.
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

app = FastAPI(title="Dummy OpenVLA Policy Server (v0)")

_policy = DummyOpenVLAPolicy()


class ImagePayload(BaseModel):
    encoding: str = "jpg_base64"
    shape: List[int]
    data: str


class PredictRequest(BaseModel):
    instruction: str
    robot_state: dict
    task_goal: dict
    target_object_position: Optional[List[float]] = None
    bin_position: Optional[List[float]] = None
    step_index: int = 0
    phase: Optional[str] = None
    observation_source: Optional[str] = None
    visual_observation: Optional[dict] = None
    image: Optional[ImagePayload] = None


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
    return {"status": "ok", "model": "dummy-openvla-fastapi", "version": "v0"}


@app.post("/reset")
def reset():
    _policy.reset()
    return {"status": "reset", "phase": _policy.phase}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    image_array = _decode_image(req.image)

    # Keep the server's own phase authoritative but accept the client's
    # reported phase as a defensive resync (matters if a request were
    # ever retried/replayed out of order; a no-op in the normal
    # sequential single-episode case since it already matches what this
    # same instance returned last response).
    if req.phase is not None:
        _policy.phase = req.phase

    policy_input = PolicyInput(
        image=image_array,
        instruction=req.instruction,
        robot_state=req.robot_state,
        task_goal=req.task_goal,
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
    info["policy_backend"] = "fastapi-dummy"
    info["server_inference_ms"] = round(server_inference_ms, 3)

    return PredictResponse(
        action=list(policy_output.action),
        phase=policy_output.phase,
        done=bool(policy_output.done),
        info=info,
    )
