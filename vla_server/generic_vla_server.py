"""Generic VLA Backend server (v0).

The model-agnostic counterpart to openvla_server_real/colab_vla_server.py
(left as-is, not modified/replaced -- see docs/generic_vla_backend.md
for why this file is now the recommended path). The local robot
control loop -- RealVLAPolicyClient and everything upstream of it --
never needs to know which model_family is actually running behind this
server: it always speaks the same /predict request/response schema
whether model_family is "mock-action", "smolvla", or "openvla".

  GET  /health       always instant -- reports model_family/model_status/
                      model_id_or_path/adapter without loading anything
  POST /load_model    the only thing that (for smolvla/openvla) ever
                      triggers a real model load; mock-action is marked
                      loaded immediately at startup since it has no
                      real model to download
  POST /predict       decodes the request the same way regardless of
                      model_family, dispatches to
                      vla_server.model_loader.run_inference(), then
                      ALWAYS passes the raw output through this
                      family's adapter.normalize_model_output() before
                      returning anything -- raw model output is never
                      handed back as an executable action directly.
                      If the adapter returns action=None (not loaded,
                      or "I'm not confident in this output"), responds
                      with a structured error so the local
                      RealVLAPolicyClient falls back instead of
                      executing a guess.
  POST /reset         resets whatever per-episode state this family's
                      loaded model keeps (currently only mock-action's
                      DummyOpenVLAPolicy phase).

Model selection (model_family, model_id_or_path, local_files_only,
device, dtype) is entirely env-var/config driven -- see
vla_server/model_loader.py's module docstring. Nothing here is
hardcoded to any one model.

VLA_BACKEND_CONFIG_PATH (optional) points at one of the
configs/vla_backend_*.json files -- if set, its "action_postprocess"
block (clip ranges, gripper threshold) is passed into the adapter, so
editing that file actually changes server-side clipping instead of the
adapter silently using its own hardcoded defaults. This is deliberately
redundant with RealVLAPolicyClient's own client-side
policy/vla_action_postprocessor.py -- defense in depth, so a raw
unclipped/unvalidated action never leaves this server even if some
other, less careful client ever talks to it.
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from vla_server import model_loader
from vla_server.model_registry import get_adapter

CONFIG_PATH_ENV_VAR = "VLA_BACKEND_CONFIG_PATH"


def _load_backend_config() -> dict:
    config_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"Warning: {CONFIG_PATH_ENV_VAR}={config_path!r} does not exist -- using adapter defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


_BACKEND_CONFIG = _load_backend_config()
_MODEL_FAMILY = model_loader.resolve_model_family(_BACKEND_CONFIG)
_ADAPTER = get_adapter(_MODEL_FAMILY, config=_BACKEND_CONFIG)

app = FastAPI(title=f"Generic VLA Backend v0 [{_MODEL_FAMILY}]")

if _MODEL_FAMILY == "mock-action":
    # mock-action has no real model to download -- mark it loaded
    # immediately so /predict works without ever requiring a
    # POST /load_model call, matching every existing mock-action demo's
    # expectations (see docs/generic_vla_backend.md).
    _mock_model_id_or_path = model_loader.resolve_model_id_or_path(_MODEL_FAMILY, _BACKEND_CONFIG)
    model_loader.load_model_once(_MODEL_FAMILY, _mock_model_id_or_path, False)


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
    # Multi-camera observation, keyed by role (e.g. "main", "wrist") --
    # see policy/policy_types.py's PolicyInput.images_by_role. Additive:
    # a client that only sends `image` still works unchanged (legacy/
    # degraded single-camera path); this is only present when the client
    # actually has independent camera renders to send (e.g.
    # PyBulletPandaBackend.render_main_camera()/render_wrist_camera()).
    images: Optional[Dict[str, ImagePayload]] = None
    action_schema: Optional[dict] = None


class PredictResponse(BaseModel):
    action: Optional[List[float]] = None
    phase: str = "move_to_object"
    done: bool = False
    info: dict = {}


def decode_request_image(image_payload: Optional[ImagePayload]) -> Optional[np.ndarray]:
    if image_payload is None:
        return None
    image_bytes = base64.b64decode(image_payload.data)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image, dtype=np.uint8)


@app.get("/health")
def health():
    state = model_loader.get_state()
    compatibility_result = model_loader.get_compatibility_result()
    return {
        "status": "ok",
        "model_family": _MODEL_FAMILY,
        "model_status": state["status"],
        "model_status_reason": state["reason"],
        "model_id_or_path": model_loader.resolve_model_id_or_path(_MODEL_FAMILY, _BACKEND_CONFIG),
        "local_files_only": model_loader.resolve_local_files_only(),
        "allow_vlm_fallback": model_loader.resolve_allow_vlm_fallback(),
        "smoke_test_mode": model_loader.resolve_smoke_test_mode(),
        "compatibility": compatibility_result.to_dict() if compatibility_result is not None else None,
        "adapter": type(_ADAPTER).__name__,
        "version": "v0",
    }


@app.post("/load_model")
def load_model():
    if _MODEL_FAMILY == "mock-action":
        return {"model_status": "loaded", "model_status_reason": None}

    state = model_loader.get_state()
    if state["model_family"] == _MODEL_FAMILY and state["status"] in ("loading", "loaded"):
        return {"model_status": state["status"], "model_status_reason": state["reason"]}

    model_id_or_path = model_loader.resolve_model_id_or_path(_MODEL_FAMILY, _BACKEND_CONFIG)
    local_files_only = model_loader.resolve_local_files_only()
    result = model_loader.load_model_once(_MODEL_FAMILY, model_id_or_path, local_files_only)
    return {"model_status": result["status"], "model_status_reason": result["reason"]}


@app.post("/reset")
def reset():
    if _MODEL_FAMILY == "mock-action":
        model_loader.reset_mock_policy()
    return {"status": "reset", "model_family": _MODEL_FAMILY}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    image_array = decode_request_image(req.image)
    images_by_role = (
        {role: decode_request_image(payload) for role, payload in req.images.items()} if req.images else None
    )
    policy_input_dict = {
        "instruction": req.instruction,
        "image": image_array,
        "images_by_role": images_by_role,
        "robot_state": req.robot_state or {},
        "task_goal": req.task_goal or {},
        "target_object_position": req.target_object_position,
        "bin_position": req.bin_position,
        "step_index": req.step_index,
        "phase": req.phase,
        "observation_source": req.observation_source,
        "visual_observation": req.visual_observation,
    }

    model_input = _ADAPTER.build_model_input(policy_input_dict)

    try:
        raw_output = model_loader.run_inference(_MODEL_FAMILY, model_input)
    except RuntimeError as exc:
        state = model_loader.get_state()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "model_family": _MODEL_FAMILY,
                "model_status": state["status"],
                "reason": state["reason"] or str(exc),
            },
        )
    except Exception as exc:  # noqa: BLE001 -- inference failure is reported, never a crash
        raise HTTPException(
            status_code=503,
            detail={"error": "inference_failed", "model_family": _MODEL_FAMILY, "reason": str(exc)},
        )

    compatibility_result = model_loader.get_compatibility_result()
    context = {
        "step_index": req.step_index,
        "phase": req.phase,
        "compatibility": compatibility_result.to_dict() if compatibility_result is not None else None,
        "smoke_test_mode": model_loader.resolve_smoke_test_mode(),
    }
    normalized = _ADAPTER.normalize_model_output(raw_output, context)

    server_inference_ms = (time.perf_counter() - start) * 1000
    info = dict(normalized.get("info") or {})
    info["server_inference_ms"] = round(server_inference_ms, 3)
    info.setdefault("policy_backend", "real-vla")

    if normalized.get("action") is None:
        raise HTTPException(
            status_code=503,
            detail={"error": info.get("reason", "action_adapter_required"), "model_family": _MODEL_FAMILY, **info},
        )

    return PredictResponse(
        action=normalized["action"],
        phase=normalized.get("phase", "move_to_object"),
        done=bool(normalized.get("done", False)),
        info=info,
    )
