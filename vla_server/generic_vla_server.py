"""Generic VLA Backend server (v0).

Desktop용 Expert-Replay VLA Server (see this task's chat report,
"Desktop용 Expert-Replay VLA Server"): model_family in ("dummy",
"expert_replay") is a NEW, entirely additive, session-aware code path
living alongside the original mock-action/smolvla/openvla path below --
it never touches vla_server/model_loader.py, vla_server/model_registry.py,
vla_adapters/base_vla_adapter.py, or any of the 3 original adapters, and
none of the original endpoints' existing behavior changes for those 3
families. This exists so the Laptop track (ROS2 / Task Manager /
Execution Monitor / Safety) can integrate against the FINAL intended
HTTP contract (session_id, request_id, action_chunk, trajectory_finished,
etc.) before a real SmolVLA backend for the SO-101 recycling-cell
embodiment exists -- expert_replay REPLAYS a pre-generated SO-101
scripted-expert trajectory (see benchmark/generate_so101_expert_replay_trajectory.py)
one step per /predict call, per session; it never re-runs the scripted
expert live. dummy is a trivial fixed-action stand-in speaking the exact
same schema, used to prove the schema itself doesn't depend on which
backend is behind it. See vla_server/session_store.py,
vla_adapters/expert_replay_adapter.py, vla_adapters/dummy_session_adapter.py.

To swap in a real SmolVLA backend for this embodiment later: add a
`vla_adapters/so101_smolvla_session_adapter.py` implementing the same
3-attribute/2-method shape ExpertReplayAdapter/DummySessionAdapter
already implement (`model_id`, `action_space_metadata`, `num_steps`,
`get_step(position)`, `check_initial_conditions(caller_conditions)`) --
`get_step()` would run real inference instead of an file lookup -- then
add one line to `_build_session_backend()` below and one new value to
SESSION_AWARE_FAMILIES. Nothing in generic_vla_server.py's request/
response schema, session_store.py, or any Laptop-side client needs to
change: the whole point of this design is that the wire contract is
identical whether the session-aware backend behind it is dummy,
expert_replay, or a real model.

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
from vla_server.session_store import SessionStore, StepOrderError, resolve_step_outcome

CONFIG_PATH_ENV_VAR = "VLA_BACKEND_CONFIG_PATH"

# See this file's own docstring, "Desktop용 Expert-Replay VLA Server".
SESSION_AWARE_FAMILIES = ("dummy", "expert_replay")


def _build_session_backend(model_family: str, config: dict):
    if model_family == "expert_replay":
        from vla_adapters.expert_replay_adapter import ExpertReplayAdapter

        return ExpertReplayAdapter(config=config)
    if model_family == "dummy":
        from vla_adapters.dummy_session_adapter import DummySessionAdapter

        return DummySessionAdapter(config=config)
    raise ValueError(f"not a session-aware family: {model_family!r}")


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

if _MODEL_FAMILY in SESSION_AWARE_FAMILIES:
    # New session-aware path -- never calls get_adapter()/model_registry.py,
    # so VALID_MODEL_FAMILIES and the 3 original adapter classes are
    # completely untouched (see this file's own docstring).
    _ADAPTER = None
    _SESSION_BACKEND = _build_session_backend(_MODEL_FAMILY, _BACKEND_CONFIG)
    _SESSION_STORE = SessionStore()
else:
    _ADAPTER = get_adapter(_MODEL_FAMILY, config=_BACKEND_CONFIG)
    _SESSION_BACKEND = None
    _SESSION_STORE = None

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
    # Optional per-step RNG seed (see policy/policy_types.py's
    # PolicyInput.seed) -- only meaningful to model_family="smolvla"'s
    # _run_smolvla_libero_inference(), which calls torch.manual_seed(seed)
    # right before sampling if present. None (default): no seeding.
    seed: Optional[int] = None

    # --- Session-aware fields (see this file's own docstring, "Desktop용
    # Expert-Replay VLA Server") -- all optional/additive; a caller that
    # never sends any of these still gets the exact pre-existing
    # mock-action/smolvla/openvla behavior unchanged. Only read when
    # model_family is in SESSION_AWARE_FAMILIES.
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    chunk_index: Optional[int] = None  # alias for step_index -- see session_store.resolve_step_outcome()
    observation: Optional[dict] = None  # freeform alternative to `image`/`images` for callers without an encoded frame ready yet
    target_info: Optional[dict] = None
    timestamp: Optional[float] = None


class PredictResponse(BaseModel):
    action: Optional[List[float]] = None
    phase: str = "move_to_object"
    done: bool = False
    info: dict = {}

    # --- Session-aware fields -- see PredictRequest's own comment above.
    # Left at their defaults (None/False/"ok") for the original 3
    # families' responses, which never set them.
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    step_index: Optional[int] = None
    chunk_index: Optional[int] = None
    action_chunk: Optional[List[List[float]]] = None
    action_space_metadata: Optional[dict] = None
    status: str = "ok"
    trajectory_finished: bool = False
    backend_type: Optional[str] = None
    model_id: Optional[str] = None
    latency_ms: Optional[float] = None
    failure_reason: Optional[str] = None


def decode_request_image(image_payload: Optional[ImagePayload]) -> Optional[np.ndarray]:
    if image_payload is None:
        return None
    image_bytes = base64.b64decode(image_payload.data)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image, dtype=np.uint8)


@app.get("/health")
def health():
    if _MODEL_FAMILY in SESSION_AWARE_FAMILIES:
        return {
            "status": "ok",
            "model_family": _MODEL_FAMILY,
            "model_status": "loaded",
            "model_status_reason": None,
            "model_id_or_path": _SESSION_BACKEND.model_id,
            "backend_type": _SESSION_BACKEND.backend_type,
            "num_sessions": _SESSION_STORE.num_sessions(),
            "trajectory_num_steps": _SESSION_BACKEND.num_steps,
            "action_space_metadata": _SESSION_BACKEND.action_space_metadata,
            "adapter": type(_SESSION_BACKEND).__name__,
            "version": "v0",
        }

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
    if _MODEL_FAMILY in SESSION_AWARE_FAMILIES:
        # Trajectory/dummy config is loaded synchronously in
        # _build_session_backend() at import time -- nothing to do here,
        # mirrors mock-action's own "no real model, always loaded" contract.
        return {"model_status": "loaded", "model_status_reason": None}
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


class SessionResetRequest(BaseModel):
    session_id: str
    # Caller's own actual scene conditions (e.g. object_position,
    # bin_center_xy) -- compared against the loaded trajectory's own
    # generation-time conditions; see ExpertReplayAdapter.check_initial_conditions().
    initial_conditions: Optional[dict] = None


@app.post("/session/reset")
def session_reset(req: SessionResetRequest):
    if _MODEL_FAMILY not in SESSION_AWARE_FAMILIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "session_reset_not_supported",
                "model_family": _MODEL_FAMILY,
                "reason": f"POST /session/reset is only supported for model_family in {SESSION_AWARE_FAMILIES}",
            },
        )
    _SESSION_STORE.reset_session(req.session_id, req.initial_conditions)
    warning = _SESSION_BACKEND.check_initial_conditions(req.initial_conditions)
    return {
        "status": "reset",
        "session_id": req.session_id,
        "backend_type": _SESSION_BACKEND.backend_type,
        "model_id": _SESSION_BACKEND.model_id,
        "trajectory_num_steps": _SESSION_BACKEND.num_steps,
        "action_space_metadata": _SESSION_BACKEND.action_space_metadata,
        "expected_initial_conditions": _SESSION_BACKEND.initial_conditions,
        "warning": warning,
    }


def _completed_response_dict(req: PredictRequest) -> dict:
    """The safe completion response (see this task's chat report, "완료
    응답") -- action_chunk/action are ALWAYS None here, never the last
    real action repeated. Used both when the backend itself reports
    status="completed" and when a session that already completed on a
    PRIOR call receives yet another new request_id (see
    _handle_session_predict's own "already completed" short-circuit,
    which never calls the backend again once state.completed is True)."""
    return {
        "request_id": req.request_id, "session_id": req.session_id,
        "step_index": None, "chunk_index": req.chunk_index,
        "action_chunk": None, "action_space_metadata": _SESSION_BACKEND.action_space_metadata,
        "status": "completed", "trajectory_finished": True,
        "backend_type": _SESSION_BACKEND.backend_type, "model_id": _SESSION_BACKEND.model_id,
        "latency_ms": None, "failure_reason": None,
        "action": None, "phase": "completed", "done": True, "info": {},
    }


def _handle_session_predict(req: PredictRequest, start: float):
    """Session-aware /predict dispatch -- see this file's own docstring.
    Never reaches model_loader.run_inference()/_ADAPTER (both are None
    for these families); only touches _SESSION_STORE/_SESSION_BACKEND.

    Calls ONLY `_SESSION_BACKEND.predict(req, state)` -- the common
    request-based interface every session-aware backend implements (see
    this task's chat report, "session backend 공통 인터페이스... request
    기반 추론 인터페이스로 정리"). This function never calls
    get_step()/check_initial_conditions() directly and never inspects
    which concrete backend class it's holding -- a future real SmolVLA
    backend slots in here with zero changes to this function."""
    if req.session_id is None or req.request_id is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "session_id_and_request_id_required",
                "model_family": _MODEL_FAMILY,
                "reason": "session-aware backends (dummy/expert_replay) require both session_id and request_id on every /predict call.",
            },
        )

    state = _SESSION_STORE.get_or_create(req.session_id)

    # Idempotent replay of the immediately-preceding request_id --
    # unconditional, even once completed (this task's own "완료 이후 같은
    # request_id 재전송: 기존 idempotency 규칙에 따라 동일 completed 응답
    # 반환").
    if req.request_id == state.last_request_id:
        cached = dict(state.last_response)
        cached["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return PredictResponse(**cached)

    if state.completed:
        # Already finished on a prior call -- ANY new request_id gets a
        # fresh completed response; step order is not re-validated and
        # the backend is not called again (this task's own "완료 이후
        # 새로운 request_id로 다음 step 요청: 동일하게 completed 응답 반환,
        # 실제 action은 절대 반환하지 않음").
        response_dict = _completed_response_dict(req)
        response_dict["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        _SESSION_STORE.record_response(
            req.session_id, req.request_id, response_dict, advance=False, trajectory_finished=True, completed=True,
        )
        return PredictResponse(**response_dict)

    step_index = req.chunk_index if req.chunk_index is not None else req.step_index
    try:
        resolve_step_outcome(state, req.request_id, step_index)
    except StepOrderError as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return PredictResponse(
            request_id=req.request_id, session_id=req.session_id,
            step_index=step_index, chunk_index=req.chunk_index,
            action_chunk=None, action_space_metadata=_SESSION_BACKEND.action_space_metadata,
            status="error", trajectory_finished=state.trajectory_finished,
            backend_type=_SESSION_BACKEND.backend_type, model_id=_SESSION_BACKEND.model_id,
            latency_ms=latency_ms,
            failure_reason=f"step_order_mismatch: expected_step_index={exc.expected_step_index}, received={exc.received_step_index}",
        )

    outcome = _SESSION_BACKEND.predict(req, state)
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    if outcome["status"] == "completed":
        response_dict = _completed_response_dict(req)
        response_dict["latency_ms"] = latency_ms
        _SESSION_STORE.record_response(
            req.session_id, req.request_id, response_dict, advance=False, trajectory_finished=True, completed=True,
        )
        return PredictResponse(**response_dict)

    warning = outcome.get("warning")
    response_dict = {
        "request_id": req.request_id, "session_id": req.session_id,
        "step_index": state.position, "chunk_index": req.chunk_index,
        "action_chunk": outcome["action_chunk"], "action_space_metadata": _SESSION_BACKEND.action_space_metadata,
        "status": "ok" if warning is None else "warning",
        "trajectory_finished": outcome["trajectory_finished"],
        "backend_type": _SESSION_BACKEND.backend_type, "model_id": _SESSION_BACKEND.model_id,
        "latency_ms": latency_ms,
        "failure_reason": warning,
        # Old-schema fields, kept populated too so a client reading only
        # the pre-existing PredictResponse shape still gets something
        # sensible (see this file's own docstring).
        "action": outcome["action_chunk"][0], "phase": outcome.get("phase"), "done": outcome["trajectory_finished"], "info": {},
    }
    _SESSION_STORE.record_response(
        req.session_id, req.request_id, response_dict, advance=True, trajectory_finished=outcome["trajectory_finished"],
    )
    return PredictResponse(**response_dict)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()

    if _MODEL_FAMILY in SESSION_AWARE_FAMILIES:
        return _handle_session_predict(req, start)

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
        "seed": req.seed,
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
