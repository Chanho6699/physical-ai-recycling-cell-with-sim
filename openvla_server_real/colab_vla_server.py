"""Colab VLA server spike (v0).

A FastAPI app meant to run inside a Google Colab runtime (see
notebooks/colab_vla_server_spike_v0.ipynb) and be exposed to a local
RealVLAPolicyClient over an ngrok/cloudflared HTTPS tunnel. It is
importable and runnable locally too (this file has no Colab-specific
code) -- that's how it's tested in this repo without an actual Colab
session.

Non-goals, deliberately:
  - This server never touches a robot. It only proposes a 7-DoF action
    (or, in openvla-dryrun, may refuse to -- see below); the caller's
    local action postprocessor -> SafetySupervisor -> RobotBackend
    chain is what decides whether anything actually moves.
  - This is a spike/test environment, not a long-running production
    server -- see docs/colab_vla_server_spike.md for Colab session/GPU
    limitations.
  - Real OpenVLA raw output is never auto-converted into an executable
    delta_ee_7dof action without a verified, dedicated adapter. Until
    one exists, openvla-dryrun surfaces the raw output for inspection
    and returns action_adapter_required instead of a fabricated action.

Three server modes (set via the COLAB_VLA_SERVER_MODE env var, default
"health-only"):

  health-only     No model loaded. /health reports model_status=
                  not_loaded. /predict always fails with
                  model_not_loaded (503) -- exists purely to validate
                  that the tunnel/HTTP plumbing works end to end before
                  trying anything else.
  mock-action     No real model -- reuses DummyOpenVLAPolicy (the same
                  phase engine local-dummy/fastapi-dummy/
                  real-vla-compatible-mock already use) to return a
                  deterministic, safe 7-DoF action. This is what
                  RealVLAPolicyClient/probe_colab_vla_server.py should
                  be tested against first.
  openvla-dryrun  Best-effort attempt to load a real OpenVLA model
                  (only if torch+transformers+a CUDA GPU are all
                  available -- never required, never auto-installed).
                  If loading fails for any reason (no GPU in this
                  Colab runtime, out of VRAM, missing dependency, model
                  download failure, ...), that is recorded as an
                  environment limitation (model_status=not_loaded,
                  model_status_reason=...), not a crash, and /predict
                  responds action_adapter_required so the local client
                  falls back. If the model does load, /predict returns
                  the raw model output for inspection but still does
                  NOT return an executable action -- see module
                  docstring above.

"openvla-direct" (a mode that would apply raw OpenVLA output straight
to the robot) is intentionally not implemented.
"""

import base64
import io
import os
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from policy.dummy_openvla_policy import DummyOpenVLAPolicy
from policy.policy_types import PolicyInput

SERVER_MODE_ENV_VAR = "COLAB_VLA_SERVER_MODE"
MODEL_NAME_ENV_VAR = "COLAB_VLA_MODEL_NAME"
DEFAULT_MODEL_NAME = "openvla/openvla-7b"
VALID_SERVER_MODES = ("health-only", "mock-action", "openvla-dryrun")
DEFAULT_SERVER_MODE = "health-only"


def _resolve_server_mode() -> str:
    mode = os.environ.get(SERVER_MODE_ENV_VAR, DEFAULT_SERVER_MODE)
    if mode not in VALID_SERVER_MODES:
        raise ValueError(
            f"Unknown {SERVER_MODE_ENV_VAR}={mode!r}. Expected one of {VALID_SERVER_MODES} "
            "(\"openvla-direct\" is intentionally not implemented)."
        )
    return mode


SERVER_MODE = _resolve_server_mode()

app = FastAPI(title=f"Colab VLA Server Spike v0 [{SERVER_MODE}]")

_mock_policy = DummyOpenVLAPolicy() if SERVER_MODE == "mock-action" else None

_model_state = {"status": "not_loaded", "reason": "server_mode is not openvla-dryrun"}
_openvla_model = None
_openvla_processor = None


def _try_load_openvla_model() -> None:
    """Best-effort, never raises. Any failure (missing torch/
    transformers, no CUDA GPU, OOM, model download failure, ...) is
    recorded in _model_state as an environment limitation, not raised
    as an error -- this is what lets /health and /predict report
    model_status=not_loaded cleanly instead of the server crashing on
    import or on startup."""
    global _openvla_model, _openvla_processor

    _model_state["status"] = "loading"
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:
        _model_state["status"] = "not_loaded"
        _model_state["reason"] = f"missing_dependency: {exc}"
        return

    if not torch.cuda.is_available():
        _model_state["status"] = "not_loaded"
        _model_state["reason"] = (
            "no_cuda_gpu_available (openvla-dryrun needs a Colab GPU runtime: "
            "Runtime > Change runtime type > T4 GPU or better)"
        )
        return

    model_name = os.environ.get(MODEL_NAME_ENV_VAR, DEFAULT_MODEL_NAME)
    try:
        _openvla_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _openvla_model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda")
        _model_state["status"] = "loaded"
        _model_state["reason"] = None
    except Exception as exc:  # noqa: BLE001 -- any load failure is an environment limitation, never a crash
        _model_state["status"] = "not_loaded"
        _model_state["reason"] = f"model_load_failed: {exc}"
        _openvla_model = None
        _openvla_processor = None


if SERVER_MODE == "openvla-dryrun":
    _try_load_openvla_model()


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
    action: Optional[List[float]] = None
    phase: str = "move_to_object"
    done: bool = False
    info: dict = {}


def _decode_image(image_payload: Optional[ImagePayload]) -> Optional[np.ndarray]:
    if image_payload is None:
        return None
    image_bytes = base64.b64decode(image_payload.data)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image, dtype=np.uint8)


def _to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


@app.get("/health")
def health():
    return {
        "status": "ok",
        "server_mode": SERVER_MODE,
        "model_status": _model_state["status"],
        "model_status_reason": _model_state.get("reason"),
        "version": "v0",
    }


@app.post("/reset")
def reset():
    if _mock_policy is not None:
        _mock_policy.reset()
    return {"status": "reset", "server_mode": SERVER_MODE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    if SERVER_MODE == "health-only":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "server_mode": SERVER_MODE,
                "reason": (
                    "health-only mode never loads a model or generates actions -- restart the "
                    "server with COLAB_VLA_SERVER_MODE=mock-action or openvla-dryrun instead."
                ),
            },
        )

    if SERVER_MODE == "mock-action":
        image_array = _decode_image(req.image)
        if req.phase is not None:
            _mock_policy.phase = req.phase

        policy_input = PolicyInput(
            image=image_array,
            instruction=req.instruction,
            robot_state=req.robot_state or {},
            task_goal=req.task_goal or {},
            target_object_position=req.target_object_position,
            bin_position=req.bin_position,
            step_index=req.step_index,
            phase=_mock_policy.phase,
            observation_source=req.observation_source,
            visual_observation=req.visual_observation,
        )
        policy_output = _mock_policy.predict_action(policy_input)

        server_inference_ms = (time.perf_counter() - start) * 1000
        info = dict(policy_output.info or {})
        info["model"] = "colab-mock-action"
        info["policy_backend"] = "real-vla"
        info["server_mode"] = SERVER_MODE
        info["model_status"] = "not_loaded"
        info["server_inference_ms"] = round(server_inference_ms, 3)

        return PredictResponse(
            action=list(policy_output.action),
            phase=policy_output.phase,
            done=bool(policy_output.done),
            info=info,
        )

    # openvla-dryrun
    if _model_state["status"] != "loaded":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "action_adapter_required",
                "server_mode": SERVER_MODE,
                "model_status": _model_state["status"],
                "reason": _model_state.get("reason") or "OpenVLA model is not loaded in this Colab runtime.",
            },
        )

    image_array = _decode_image(req.image)
    if image_array is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "image_required", "reason": "openvla-dryrun mode requires an input image."},
        )

    try:
        import torch

        pil_image = Image.fromarray(image_array)
        inputs = _openvla_processor(req.instruction, pil_image).to("cuda", dtype=torch.bfloat16)
        raw_output = _openvla_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    except Exception as exc:  # noqa: BLE001 -- inference failure is reported, not a 500 crash
        raise HTTPException(
            status_code=503,
            detail={"error": "openvla_inference_failed", "reason": str(exc)},
        )

    server_inference_ms = (time.perf_counter() - start) * 1000

    # Deliberately do NOT return this as `action`: OpenVLA's own action
    # space/normalization (unnorm_key, gripper convention, frame
    # convention) is not verified here to match this project's
    # delta_ee_7dof action_schema. Surfacing the raw output lets a
    # human/future adapter inspect it, but the local
    # RealVLAPolicyClient must fall back rather than execute it blind.
    return PredictResponse(
        action=None,
        phase=req.phase or "move_to_object",
        done=False,
        info={
            "model": "openvla-dryrun",
            "policy_backend": "real-vla",
            "server_mode": SERVER_MODE,
            "model_status": "loaded",
            "raw_model_output_available": True,
            "raw_model_output": _to_jsonable(raw_output),
            "project_action_available": False,
            "reason": "action_adapter_required",
            "server_inference_ms": round(server_inference_ms, 3),
        },
    )
