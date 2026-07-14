"""Generic VLA model loader (v0).

Owns the lazy load lifecycle AND the actual inference call for
whichever model_family is configured. vla_adapters/*.py only ever
transform data (request -> model input, raw output -> normalized
action) and never touch a model object or the network -- this module
is the only place that does. That split is what lets swapping SmolVLA
for OpenVLA (or anything else) later mean "change what this file
loads/calls for that family", never a change to the server, the
adapters, RealVLAPolicyClient, or the local robot control loop.

Model selection is read from environment variables (never hardcoded),
falling back to a config dict (e.g. from configs/vla_backend_*.json)
if the env var isn't set:

  VLA_MODEL_FAMILY       "mock-action" | "smolvla" | "openvla"
                         (default: mock-action)
  VLA_MODEL_ID_OR_PATH   HF Hub repo id or local directory
                         (default depends on model_family)
  VLA_LOCAL_FILES_ONLY   "1" to force local_files_only=True
                         (skip any network call entirely)
  VLA_DEVICE             "cuda" | "cpu" (default: cuda if available, else cpu)
  VLA_DTYPE              a torch dtype name, e.g. "bfloat16" | "float16" | "float32"
  VLA_ALLOW_VLM_FALLBACK "1" to allow _load_smolvla() to fall back to a plain
                         transformers.AutoModelForImageTextToText load (e.g.
                         HuggingFaceTB/SmolVLM2-500M-Video-Instruct) when none
                         of the LeRobot SmolVLA policy import candidates match
                         the installed LeRobot version. Default: disabled --
                         that fallback downloads the VLM backbone SmolVLA is
                         built on top of, NOT the SmolVLA action policy
                         itself, which isn't useful for confirming the real
                         SmolVLA import path and can be a large, unwanted
                         download (see docs/smolvla_cloud_loading_spike.md).

Import-time behavior: importing this module NEVER loads a model or
downloads anything -- only load_model_once() does, mirroring
openvla_server_real/colab_vla_server.py's lazy-load contract (that
file is left as-is; this module generalizes the same pattern instead
of replacing it). mock-action is the one exception: it has no real
model to download, so it's marked "loaded" as soon as
load_model_once("mock-action", ...) is called (generic_vla_server.py
does this once at startup), not lazily on the first /predict.
"""

import os
import threading
from typing import Any, Dict, Optional

MODEL_FAMILY_ENV_VAR = "VLA_MODEL_FAMILY"
MODEL_ID_OR_PATH_ENV_VAR = "VLA_MODEL_ID_OR_PATH"
LOCAL_FILES_ONLY_ENV_VAR = "VLA_LOCAL_FILES_ONLY"
DEVICE_ENV_VAR = "VLA_DEVICE"
DTYPE_ENV_VAR = "VLA_DTYPE"
ALLOW_VLM_FALLBACK_ENV_VAR = "VLA_ALLOW_VLM_FALLBACK"

DEFAULT_MODEL_FAMILY = "mock-action"
DEFAULT_MODEL_ID_BY_FAMILY = {
    "smolvla": "lerobot/smolvla_base",
    "openvla": "openvla/openvla-7b",
}
DEFAULT_DTYPE_NAME = "bfloat16"

VALID_MODEL_STATUSES = ("not_loaded", "loading", "loaded", "load_failed")

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "status": "not_loaded",
    "reason": "model load has not been requested",
    "model_family": None,
    "model": None,
    "processor": None,
}


def resolve_model_family(config: Optional[dict] = None) -> str:
    env_value = os.environ.get(MODEL_FAMILY_ENV_VAR)
    if env_value:
        return env_value
    if config and config.get("model_family"):
        return config["model_family"]
    return DEFAULT_MODEL_FAMILY


def resolve_model_id_or_path(model_family: str, config: Optional[dict] = None) -> str:
    env_value = os.environ.get(MODEL_ID_OR_PATH_ENV_VAR)
    if env_value:
        return env_value
    if config and config.get("model_id_or_path"):
        return config["model_id_or_path"]
    return DEFAULT_MODEL_ID_BY_FAMILY.get(model_family, model_family)


def resolve_local_files_only() -> bool:
    return os.environ.get(LOCAL_FILES_ONLY_ENV_VAR) == "1"


def resolve_device() -> str:
    env_value = os.environ.get(DEVICE_ENV_VAR)
    if env_value:
        return env_value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_dtype_name() -> str:
    return os.environ.get(DTYPE_ENV_VAR, DEFAULT_DTYPE_NAME)


def resolve_allow_vlm_fallback() -> bool:
    return os.environ.get(ALLOW_VLM_FALLBACK_ENV_VAR) == "1"


def get_state() -> dict:
    with _lock:
        return {
            "status": _state["status"],
            "reason": _state["reason"],
            "model_family": _state["model_family"],
        }


def reset_mock_policy() -> None:
    """mock-action-specific: resets the loaded DummyOpenVLAPolicy's
    phase back to move_to_object for a new episode. No-op for any
    other family (or if mock-action hasn't been loaded)."""
    with _lock:
        model = _state.get("model") if _state.get("model_family") == "mock-action" else None
    if model is not None and hasattr(model, "reset"):
        model.reset()


def load_model_once(model_family: str, model_id_or_path: str, local_files_only: bool) -> dict:
    """Idempotent and thread-safe: the first caller for a given
    model_family actually attempts the load; any concurrent/later
    caller just observes whatever state that first attempt reached (or
    is still reaching) instead of starting a second redundant load.
    Never raises -- every failure mode is recorded into _state and
    returned as a plain dict instead."""
    with _lock:
        if _state["model_family"] == model_family and _state["status"] in ("loading", "loaded"):
            return get_state()
        _state["status"] = "loading"
        _state["reason"] = None
        _state["model_family"] = model_family

    if model_family == "mock-action":
        return _load_mock_action()
    if model_family == "smolvla":
        return _load_smolvla(model_id_or_path, local_files_only)
    if model_family == "openvla":
        return _load_openvla(model_id_or_path, local_files_only)

    return _fail(f"unknown_model_family: {model_family}")


def _fail(reason: str) -> dict:
    with _lock:
        _state["status"] = "load_failed"
        _state["reason"] = reason
    return get_state()


def _load_mock_action() -> dict:
    from policy.dummy_openvla_policy import DummyOpenVLAPolicy

    with _lock:
        _state["model"] = DummyOpenVLAPolicy()
        _state["processor"] = None
        _state["status"] = "loaded"
        _state["reason"] = None
    return get_state()


# Tried in order -- LeRobot has reorganized its policies package layout
# across versions, so more than one import path is plausible depending
# on which LeRobot release is installed. Each entry is
# "module.path:ClassName"; _try_import_smolvla_policy_class() records
# exactly which candidates were tried and why each one failed, so a
# "SmolVLA import path needs update" failure is actionable instead of
# a bare ImportError.
_SMOLVLA_IMPORT_CANDIDATES = [
    "lerobot.common.policies.smolvla.modeling_smolvla:SmolVLAPolicy",
    "lerobot.policies.smolvla.modeling_smolvla:SmolVLAPolicy",
    "lerobot.common.policies.smolvla.smolvla:SmolVLAPolicy",
]


def _try_import_smolvla_policy_class():
    """Never raises. Returns (policy_class_or_None, attempts) where
    attempts is a list of {"candidate": str, "ok": bool, "error": str}
    dicts -- one per import path tried, in order, stopping at the
    first success."""
    attempts = []
    for candidate in _SMOLVLA_IMPORT_CANDIDATES:
        module_path, _, class_name = candidate.partition(":")
        try:
            module = __import__(module_path, fromlist=[class_name])
            policy_class = getattr(module, class_name)
        except Exception as exc:  # noqa: BLE001 -- record and try the next candidate
            attempts.append({"candidate": candidate, "ok": False, "error": str(exc)})
            continue
        attempts.append({"candidate": candidate, "ok": True, "error": None})
        return policy_class, attempts
    return None, attempts


def _context_suffix(model_id_or_path: str, local_files_only: bool) -> str:
    return (
        f"model_id_or_path={model_id_or_path}, local_files_only={local_files_only}, "
        f"device={resolve_device()}, dtype={resolve_dtype_name()}"
    )


def _load_smolvla(model_id_or_path: str, local_files_only: bool) -> dict:
    """Best-effort: SmolVLA ships as a LeRobot policy checkpoint, so the
    LeRobot policy loader is tried first, across several plausible
    import paths (see _SMOLVLA_IMPORT_CANDIDATES -- LeRobot's package
    layout has moved between releases).

    If none of those import candidates match the installed LeRobot
    version, the default behavior is to fail immediately with
    model_status=load_failed -- NOT to fall back to a plain
    transformers.AutoModelForImageTextToText load. That fallback
    resolves to downloading the VLM backbone SmolVLA is built on top of
    (e.g. HuggingFaceTB/SmolVLM2-500M-Video-Instruct), not the SmolVLA
    action policy itself, which is a large, unwanted download for what
    this spike is actually trying to confirm. Set
    VLA_ALLOW_VLM_FALLBACK=1 to opt into that fallback path explicitly
    (see docs/smolvla_cloud_loading_spike.md)."""
    try:
        import torch
    except ImportError as exc:
        return _fail(f"missing_dependency: torch not installed ({exc}). {_context_suffix(model_id_or_path, local_files_only)}")

    dtype_name = resolve_dtype_name()
    dtype = getattr(torch, dtype_name, torch.float32)
    device = resolve_device()
    context = _context_suffix(model_id_or_path, local_files_only)

    policy_class, lerobot_attempts = _try_import_smolvla_policy_class()
    processor = None

    if policy_class is not None:
        try:
            model = policy_class.from_pretrained(model_id_or_path, local_files_only=local_files_only)
            model = model.to(device)
        except Exception as exc:  # noqa: BLE001 -- any load failure is an environment limitation, never a crash
            return _fail(f"model_load_failed via {policy_class.__module__}.{policy_class.__name__} ({context}): {exc}")
    elif not resolve_allow_vlm_fallback():
        # Default path: none of the LeRobot import candidates worked,
        # and the VLM fallback is disabled -- fail immediately instead
        # of ever attempting a transformers/SmolVLM2 import or download.
        tried = ", ".join(a["candidate"] for a in lerobot_attempts)
        return _fail(
            f"SmolVLA policy import failed; VLM fallback disabled -- tried [{tried}] (all failed). "
            f"Set {ALLOW_VLM_FALLBACK_ENV_VAR}=1 to opt into the transformers/SmolVLM2 backbone fallback "
            f"instead (not the SmolVLA action policy). {context}"
        )
    else:
        # VLA_ALLOW_VLM_FALLBACK=1: none of the LeRobot import
        # candidates worked -- fall back to a plain transformers load
        # before giving up entirely. This downloads the VLM backbone
        # SmolVLA is built on (e.g. HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
        # not the SmolVLA action policy -- opt-in only.
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            tried = ", ".join(a["candidate"] for a in lerobot_attempts)
            return _fail(
                f"missing_dependency: SmolVLA import path needs update -- tried [{tried}] (all failed) "
                f"and transformers fallback also unavailable ({exc}). {context}"
            )

        try:
            processor = AutoProcessor.from_pretrained(
                model_id_or_path, local_files_only=local_files_only, trust_remote_code=True
            )
            model = AutoModelForImageTextToText.from_pretrained(
                model_id_or_path,
                local_files_only=local_files_only,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)
        except Exception as exc:  # noqa: BLE001 -- any load failure is an environment limitation, never a crash
            tried = ", ".join(a["candidate"] for a in lerobot_attempts)
            return _fail(
                f"model_load_failed via transformers.AutoModelForImageTextToText fallback "
                f"(LeRobot candidates tried and failed: [{tried}]; {context}): {exc}"
            )

    with _lock:
        _state["model"] = model
        _state["processor"] = processor
        _state["status"] = "loaded"
        _state["reason"] = None
    return get_state()


def _load_openvla(model_id_or_path: str, local_files_only: bool) -> dict:
    """Mirrors openvla_server_real/colab_vla_server.py's OpenVLA load
    (same libraries/calling convention) so this family is genuinely
    swappable, not just a stub -- but per this v0's explicit scope,
    OpenVLAActionAdapter always refuses to return an executable action
    regardless of whether this load succeeds (see
    vla_adapters/openvla_adapter.py), so a working load here is not
    required for the rest of the system to behave correctly."""
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:
        return _fail(f"missing_dependency: {exc}. {_context_suffix(model_id_or_path, local_files_only)}")

    if not torch.cuda.is_available():
        return _fail(
            "no_cuda_gpu_available (openvla needs a CUDA GPU runtime -- see "
            f"docs/colab_vla_server_spike.md for the Colab-specific experiment for this family). "
            f"{_context_suffix(model_id_or_path, local_files_only)}"
        )

    dtype_name = resolve_dtype_name()
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    context = _context_suffix(model_id_or_path, local_files_only)

    try:
        processor = AutoProcessor.from_pretrained(
            model_id_or_path, local_files_only=local_files_only, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_id_or_path,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to("cuda")
    except Exception as exc:  # noqa: BLE001 -- any load failure is an environment limitation, never a crash
        return _fail(f"model_load_failed ({context}): {exc}")

    with _lock:
        _state["model"] = model
        _state["processor"] = processor
        _state["status"] = "loaded"
        _state["reason"] = None
    return get_state()


def run_inference(model_family: str, model_input: dict) -> Any:
    """Dispatches to the already-loaded model for model_family and
    returns its raw output, using whatever calling convention that
    model actually needs. Raises RuntimeError if the model for this
    family isn't loaded, or if the forward pass itself fails --
    generic_vla_server.py turns that into a structured error response
    rather than a 500/crash."""
    with _lock:
        if _state["model_family"] != model_family or _state["status"] != "loaded":
            raise RuntimeError(f"Model for family={model_family!r} is not loaded (status={_state['status']!r}).")
        model = _state["model"]
        processor = _state["processor"]

    if model_family == "mock-action":
        policy_input = model_input["policy_input"]
        if policy_input.phase is not None:
            model.phase = policy_input.phase
        else:
            policy_input.phase = model.phase
        return model.predict_action(policy_input)

    if model_family == "smolvla":
        return _run_smolvla_inference(model, processor, model_input)

    if model_family == "openvla":
        return _run_openvla_inference(model, processor, model_input)

    raise RuntimeError(f"No inference dispatch implemented for model_family={model_family!r}.")


def _run_smolvla_inference(model, processor, model_input: dict) -> Any:
    """Tries each plausible calling convention for the loaded model in
    turn, raising a RuntimeError naming exactly which ones were tried
    if none apply -- generic_vla_server.py turns that into a structured
    inference_failed error rather than a crash, and the reason string
    is specific enough to say what needs updating."""
    import torch

    batch = {
        "observation.image": model_input.get("image"),
        "observation.state": model_input.get("robot_state"),
        "task": model_input.get("instruction"),
    }

    with torch.no_grad():
        if hasattr(model, "select_action"):
            # LeRobot policy interface: select_action(batch) -> action tensor/dict.
            return model.select_action(batch)

        if hasattr(model, "predict_action"):
            # Some LeRobot-adjacent policies expose predict_action(batch) instead.
            return model.predict_action(batch)

        if processor is not None and hasattr(model, "generate"):
            # transformers-style processor+generate fallback.
            dtype = getattr(torch, resolve_dtype_name(), torch.float32)
            inputs = processor(model_input.get("instruction", ""), model_input.get("image")).to(
                resolve_device(), dtype=dtype
            )
            return model.generate(**inputs)

    raise RuntimeError(
        f"Loaded SmolVLA model ({type(model).__module__}.{type(model).__name__}) has none of the expected "
        "inference methods (select_action, predict_action, generate) -- the model_loader.py dispatch needs "
        "updating for this model's actual API."
    )


def _run_openvla_inference(model, processor, model_input: dict) -> Any:
    import torch

    dtype = getattr(torch, resolve_dtype_name(), torch.bfloat16)
    inputs = processor(model_input.get("instruction", ""), model_input.get("image")).to(
        resolve_device(), dtype=dtype
    )
    return model.predict_action(**inputs, unnorm_key=model_input.get("unnorm_key", "bridge_orig"), do_sample=False)
