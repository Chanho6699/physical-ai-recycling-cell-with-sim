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
  VLA_SMOKE_TEST_MODE    "1" to allow adapters to fall back to a shape-only
                         (no verified semantic meaning) action when
                         CompatibilityGate rejects a checkpoint for
                         production -- see policy_semantics/compatibility_gate.py.
                         Default: disabled. Never set this in production; it
                         exists solely to smoke-test the serving pipeline
                         (does a forward pass run end to end and produce
                         *some* 7-length vector) independent of whether that
                         vector means anything for this project's robot.

Compatibility gating: right after a model_family="smolvla" load succeeds,
load_model_once() looks up this model_id_or_path's PolicyManifest (see
policy_semantics/manifest.py) and runs CompatibilityGate.check() against
this project's Panda target embodiment, storing the CompatibilityResult
in _state so generic_vla_server.py can expose it via /health and pass it
into each adapter call's context -- vla_adapters/smolvla_adapter.py
refuses to produce a production action whenever this didn't pass (see
that module's docstring for why lerobot/smolvla_base never will).

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

from policy_semantics.compatibility_gate import CompatibilityGate, CompatibilityResult
from policy_semantics.manifest import get_manifest

MODEL_FAMILY_ENV_VAR = "VLA_MODEL_FAMILY"
MODEL_ID_OR_PATH_ENV_VAR = "VLA_MODEL_ID_OR_PATH"
LOCAL_FILES_ONLY_ENV_VAR = "VLA_LOCAL_FILES_ONLY"
DEVICE_ENV_VAR = "VLA_DEVICE"
DTYPE_ENV_VAR = "VLA_DTYPE"
ALLOW_VLM_FALLBACK_ENV_VAR = "VLA_ALLOW_VLM_FALLBACK"
SMOKE_TEST_MODE_ENV_VAR = "VLA_SMOKE_TEST_MODE"

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
    "compatibility_result": None,  # CompatibilityResult|None, set at load time -- see _run_compatibility_gate()
    "preprocessor_pipeline": None,  # official LeRobot PolicyProcessorPipeline|None -- see _load_official_smolvla_processors()
    "postprocessor_pipeline": None,
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


def resolve_smoke_test_mode() -> bool:
    return os.environ.get(SMOKE_TEST_MODE_ENV_VAR) == "1"


def get_state() -> dict:
    with _lock:
        return {
            "status": _state["status"],
            "reason": _state["reason"],
            "model_family": _state["model_family"],
        }


def get_compatibility_result() -> Optional[CompatibilityResult]:
    """The CompatibilityResult computed at load time (see
    _run_compatibility_gate()), or None if no model has finished loading
    yet (e.g. mock-action, or before /load_model has run for smolvla).
    generic_vla_server.py passes this into each adapter call's context so
    vla_adapters/*.py can refuse production output without reaching into
    model_loader's internals itself (keeping the adapter/loader boundary
    documented in vla_adapters/base_vla_adapter.py intact)."""
    with _lock:
        return _state.get("compatibility_result")


def _run_compatibility_gate(model_id_or_path: str) -> CompatibilityResult:
    """Runs once, right after a real model load succeeds (see
    _load_smolvla()) -- looks up this model_id_or_path's PolicyManifest
    and checks it against this project's Panda target embodiment.
    Never raises: an unregistered model_id still gets a (failing)
    CompatibilityResult via get_manifest()'s all-UNKNOWN fallback rather
    than crashing the load."""
    manifest = get_manifest(model_id_or_path)
    return CompatibilityGate.check(manifest, smoke_test_mode=resolve_smoke_test_mode())


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
        _state["compatibility_result"] = None
        _state["preprocessor_pipeline"] = None
        _state["postprocessor_pipeline"] = None

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
            model = model.to(device=device, dtype=dtype)
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

    preprocessor_pipeline, postprocessor_pipeline = _load_official_smolvla_processors(
        model_id_or_path, local_files_only
    )
    _log_load_provenance(model_id_or_path, model)

    compatibility_result = _run_compatibility_gate(model_id_or_path)

    with _lock:
        _state["model"] = model
        _state["processor"] = processor
        _state["preprocessor_pipeline"] = preprocessor_pipeline
        _state["postprocessor_pipeline"] = postprocessor_pipeline
        _state["status"] = "loaded"
        _state["reason"] = None
        _state["compatibility_result"] = compatibility_result
    return get_state()


# Only checkpoints explicitly confirmed to ship their own
# policy_preprocessor.json/policy_postprocessor.json (+ baked-in
# normalizer/unnormalizer .safetensors) get this treatment -- confirmed
# for HuggingFaceVLA/smolvla_libero this session via
# `HfApi().list_repo_files(...)`. Never attempted for an arbitrary/
# unregistered model_id (a checkpoint without these files would just
# fail the from_pretrained() call below, so this list is a deliberate
# allowlist, not a strict requirement of the loader itself).
_MODELS_WITH_OFFICIAL_PROCESSOR_FILES = ("HuggingFaceVLA/smolvla_libero",)


def _load_official_smolvla_processors(model_id_or_path: str, local_files_only: bool):
    """Loads this checkpoint's own official pre/post-processor pipelines
    (LeRobot's PolicyProcessorPipeline.from_pretrained(), the same
    mechanism lerobot-eval itself uses) if this exact model_id is known
    to ship them -- see _MODELS_WITH_OFFICIAL_PROCESSOR_FILES. This is
    what lets _run_smolvla_libero_inference() build a NativePolicyAction
    with postprocessor_used=True (real unnormalization, not this
    project's own manual batch-building from earlier turns) -- see
    policy_semantics/native_policy_action.py. Returns (None, None) if
    this model_id isn't in the allowlist, or if loading fails for any
    reason (never raises; a missing/broken official processor just
    means official_processor_wired stays effectively False for this
    load, which CompatibilityGate already accounts for)."""
    if model_id_or_path not in _MODELS_WITH_OFFICIAL_PROCESSOR_FILES:
        return None, None
    try:
        from lerobot.processor import PolicyProcessorPipeline

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            model_id_or_path, config_filename="policy_preprocessor.json", local_files_only=local_files_only
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            model_id_or_path, config_filename="policy_postprocessor.json", local_files_only=local_files_only
        )
        return preprocessor, postprocessor
    except Exception as exc:  # noqa: BLE001 -- an official-processor load failure degrades to the
        # manual-batch path (official_processor_wired effectively False for this load), never a crash.
        print(f"[model_loader] official pre/post-processor load failed for {model_id_or_path!r}: {exc}")
        return None, None


def _resolve_loaded_revision(model_id_or_path: str, model) -> str:
    """Best-effort resolved commit hash for model_id_or_path -- read from
    the local HF hub cache's own on-disk layout first (snapshot
    directories are literally named after the resolved commit hash, so
    this works offline and reflects exactly what was actually loaded),
    falling back to an HfApi() network lookup, then "unknown". Never
    raises."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id_or_path:
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revisions:
                    return revisions[0].commit_hash
    except Exception:  # noqa: BLE001 -- fall through to the network lookup below
        pass

    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id_or_path).sha or "unknown"
    except Exception:  # noqa: BLE001 -- offline, private, or any other lookup failure
        return "unknown"


def _log_load_provenance(model_id_or_path: str, model) -> None:
    """Prints the exact lerobot version and checkpoint revision actually
    loaded -- required so results are reproducible/auditable, not just
    'it worked once'. Never raises (best-effort introspection only)."""
    try:
        import lerobot

        lerobot_version = getattr(lerobot, "__version__", "unknown")
    except ImportError:
        lerobot_version = "lerobot not importable"

    revision = _resolve_loaded_revision(model_id_or_path, model)

    print(
        f"[model_loader] loaded model_id_or_path={model_id_or_path!r} "
        f"lerobot_version={lerobot_version!r} revision={revision!r}"
    )


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
        preprocessor_pipeline = _state["preprocessor_pipeline"]
        postprocessor_pipeline = _state["postprocessor_pipeline"]

    if model_family == "mock-action":
        policy_input = model_input["policy_input"]
        if policy_input.phase is not None:
            model.phase = policy_input.phase
        else:
            policy_input.phase = model.phase
        return model.predict_action(policy_input)

    if model_family == "smolvla":
        if preprocessor_pipeline is not None and postprocessor_pipeline is not None:
            # Official processors are loaded for this checkpoint (see
            # _load_official_smolvla_processors()) -- run the real
            # LeRobot pre/post-processor pipeline instead of this
            # module's own manual batch-building, and return a
            # NativePolicyAction (postprocessor_used=True) rather than a
            # raw tensor. vla_adapters/smolvla_adapter.py detects this
            # type and routes to SmolVLALiberoActionAdapter accordingly.
            return _run_smolvla_libero_inference(model, preprocessor_pipeline, postprocessor_pipeline, model_input)
        return _run_smolvla_inference(model, processor, model_input)

    if model_family == "openvla":
        return _run_openvla_inference(model, processor, model_input)

    raise RuntimeError(f"No inference dispatch implemented for model_family={model_family!r}.")


# Used only when the loaded policy's own config doesn't expose a resolvable
# visual feature schema (see _extract_visual_feature_specs) -- matches the
# shape LeRobot's SmolVLA checkpoints have been observed to expect.
_DEFAULT_IMAGE_CHW = (3, 256, 256)


def _feature_shape(feature) -> tuple:
    shape = getattr(feature, "shape", None)
    if shape:
        try:
            return tuple(int(s) for s in shape)
        except (TypeError, ValueError):
            pass
    return _DEFAULT_IMAGE_CHW


def _is_visual_feature(key: str, feature) -> bool:
    # FeatureType enum members stringify as e.g. "FeatureType.VISUAL"
    # regardless of which lerobot module they were imported from, so this
    # doesn't need its own multi-candidate import like _SMOLVLA_IMPORT_CANDIDATES.
    type_name = str(getattr(feature, "type", "")).upper()
    if "VISUAL" in type_name or "IMAGE" in type_name:
        return True
    # Fall back to LeRobot's own naming convention if `.type` isn't a
    # recognizable FeatureType (older/alternate config shapes).
    return "image" in key.lower()


def _extract_visual_feature_specs(model) -> Dict[str, tuple]:
    """Reads the loaded SmolVLA policy's own declared input feature schema
    (policy.config.image_features / input_features / features -- the exact
    attribute name has moved across LeRobot versions, same as
    _SMOLVLA_IMPORT_CANDIDATES) to find every visual/image feature key this
    specific checkpoint actually expects (e.g. observation.images.camera1/2/3),
    instead of hardcoding a single "observation.image" key. Returns {} if none
    of those attributes exist or resolve to anything -- callers fall back to
    one best-guess key/shape in that case rather than failing outright."""
    config = getattr(model, "config", None)
    if config is None:
        return {}

    image_features = getattr(config, "image_features", None)
    if isinstance(image_features, dict) and image_features:
        return {key: _feature_shape(feature) for key, feature in image_features.items()}

    for attr_name in ("input_features", "features"):
        features = getattr(config, attr_name, None)
        if not isinstance(features, dict):
            continue
        visual = {key: feature for key, feature in features.items() if _is_visual_feature(key, feature)}
        if visual:
            return {key: _feature_shape(feature) for key, feature in visual.items()}

    return {}


def _image_array_to_tensor(image_array, device: str, dtype):
    """HWC uint8 (or already-CHW) numpy array -> a single [1, 3, H, W] tensor
    on the target device/dtype. Returns None if no image was provided at all
    (e.g. a request with no visual observation)."""
    import numpy as np
    import torch

    if image_array is None:
        return None

    array = np.asarray(image_array)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        chw = array  # already channel-first
    else:
        chw = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(np.ascontiguousarray(chw)).to(dtype=torch.float32)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)  # add the batch dimension -> [1, 3, H, W]
    return tensor.to(device=device, dtype=dtype)


def _resize_image_tensor(tensor, target_chw: tuple):
    import torch

    if tensor is None:
        return None
    target_hw = tuple(int(s) for s in target_chw[-2:])
    if tuple(tensor.shape[-2:]) != target_hw:
        tensor = torch.nn.functional.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return tensor


def _build_visual_batch(image_tensor, visual_feature_specs: Dict[str, tuple]):
    """Replicates the single available camera image across every expected
    visual feature key (resizing per-key if that key's declared shape
    differs) -- this project has one camera, so "one camera image satisfies
    N expected image inputs" is the only reasonable default until multi-
    camera support exists. Returns (visual_batch, effective_specs) where
    effective_specs is what was actually used (falls back to one best-guess
    key if visual_feature_specs was empty, so debug_info -- see
    _run_smolvla_inference -- always reflects what was actually sent)."""
    if not visual_feature_specs:
        visual_feature_specs = {"observation.image": _DEFAULT_IMAGE_CHW}
    visual_batch = {
        key: _resize_image_tensor(image_tensor, shape) for key, shape in visual_feature_specs.items()
    }
    return visual_batch, visual_feature_specs


# Last-resort placeholder shape if the policy's own config doesn't expose
# observation.state's declared shape at all -- essentially never hit in
# practice since LeRobot policies always declare this feature, but keeps
# _build_state_tensor() from crashing if one somehow doesn't.
_DEFAULT_STATE_SHAPE = (1,)


def _extract_state_feature_shape(model) -> tuple:
    """Reads the loaded SmolVLA policy's own config for the declared shape of
    its observation.state input feature, the same way
    _extract_visual_feature_specs reads the visual features -- so the
    placeholder zero tensor below (_build_state_tensor) has the right number
    of dims instead of guessing one."""
    config = getattr(model, "config", None)
    if config is None:
        return _DEFAULT_STATE_SHAPE

    for attr_name in ("input_features", "features"):
        features = getattr(config, attr_name, None)
        if not isinstance(features, dict):
            continue
        feature = features.get("observation.state")
        if feature is None:
            # Some configs key it differently -- fall back to any feature
            # whose type stringifies to STATE (same trick as
            # _is_visual_feature, no version-specific FeatureType import).
            for key, candidate in features.items():
                if "STATE" in str(getattr(candidate, "type", "")).upper():
                    feature = candidate
                    break
        if feature is not None:
            shape = getattr(feature, "shape", None)
            if shape:
                try:
                    return tuple(int(s) for s in shape)
                except (TypeError, ValueError):
                    pass

    return _DEFAULT_STATE_SHAPE


def _ensure_batch_dim(value):
    """Adds a leading batch dimension to a 1-D state vector (list/tuple of
    numbers, numpy array, or torch tensor). Returns None for None -- callers
    that need a tensor regardless (SmolVLA's observation.state) use
    _build_state_tensor() instead of this directly."""
    import numpy as np
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0) if value.dim() == 1 else value
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, (int, float)) for item in value):
        return torch.tensor(value, dtype=torch.float32).unsqueeze(0)
    return value


def _build_state_tensor(robot_state, state_shape: tuple, device: str, dtype):
    """SmolVLA's observation.state must be a tensor -- RealVLAPolicyClient's
    robot_state is currently always a dict (e.g.
    {"end_effector_position": [...], "held_object": bool, ...}), not a flat
    vector matching this policy's actual per-index state convention, and
    mapping that dict onto the real state schema is a separate task from
    this fix. _ensure_batch_dim() converts robot_state when it's *already* a
    plain numeric vector/array/tensor (some future caller might pass one
    directly); a dict (or anything else _ensure_batch_dim can't convert)
    falls back to a zero tensor of the policy's own declared state shape --
    enough to let the forward pass run end to end instead of crashing on
    'dict' object has no attribute 'ndim', without pretending to be a
    correct state reading."""
    import torch

    converted = _ensure_batch_dim(robot_state)
    if isinstance(converted, torch.Tensor):
        return converted.to(device=device, dtype=dtype)
    return torch.zeros((1, *state_shape), device=device, dtype=dtype)


# Nested attribute paths tried, in order, if `model.parameters()` itself
# comes up empty/unavailable (e.g. `model` isn't an nn.Module, only a thin
# wrapper around one) -- nn.Module.parameters() already recurses through
# every child module on its own, so this is only a fallback for the case
# where the top-level object genuinely has no working .parameters().
_NESTED_PARAMETER_ATTR_PATHS = ("model", "model.vlm_with_expert", "vlm_with_expert", "model.vlm_with_expert.expert")


def _find_reference_parameter(model):
    """Returns the first torch.nn.Parameter found on `model` -- never
    hardcodes a dtype/device, always reads it off whatever the model
    actually is. Tried via model.parameters() first (covers any nn.Module,
    including SmolVLAPolicy, since parameters() already walks every nested
    submodule); if that's empty or unavailable, walks a few plausible nested
    attribute paths (see _NESTED_PARAMETER_ATTR_PATHS) looking for a
    sub-object with its own .parameters(). Returns None if nothing is found
    anywhere -- callers fall back to the configured VLA_DTYPE/VLA_DEVICE."""
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters())
        except StopIteration:
            pass

    for path in _NESTED_PARAMETER_ATTR_PATHS:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            continue
        nested_parameters = getattr(obj, "parameters", None)
        if callable(nested_parameters):
            try:
                return next(nested_parameters())
            except StopIteration:
                continue
    return None


def _resolve_model_dtype_device(model, fallback_dtype, fallback_device: str):
    """(dtype, device) to build/align every batch tensor against, read from
    the loaded model's own parameters rather than trusting the configured
    VLA_DTYPE/VLA_DEVICE to have actually stuck (_load_smolvla's .to(...)
    call can fail to reach every submodule). Falls back to the configured
    values only if the model has no parameters to inspect at all."""
    import torch

    parameter = _find_reference_parameter(model)
    if parameter is not None:
        return parameter.dtype, parameter.device
    return fallback_dtype, torch.device(fallback_device)


def _align_batch_dtype_device(batch: dict, target_dtype, target_device) -> dict:
    """Final normalization pass on the whole batch right before calling the
    model: every torch.Tensor is moved to the model's actual device, and --
    only if it's a floating-point tensor (images, observation.state) -- cast
    to the model's actual dtype too. Non-floating tensors
    (observation.language.tokens is int64, its attention_mask is bool) keep
    whatever dtype the tokenizer already gave them; SmolVLA's embedding/
    attention-mask lookups need those to stay integer/bool regardless of the
    model's own floating-point compute dtype ('mat1 and mat2 must have the
    same dtype' is a floating-point-only concern)."""
    import torch

    aligned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                aligned[key] = value.to(device=target_device, dtype=target_dtype)
            else:
                aligned[key] = value.to(device=target_device)
        else:
            aligned[key] = value
    return aligned


def _describe_batch_value(value) -> dict:
    """Cheap type/shape description of one batch value for debug_info --
    e.g. {"type": "Tensor", "shape": [1, 6]} or {"type": "list", "length": 1}
    -- never the value's actual contents."""
    description = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        description["shape"] = list(value.shape)
    elif isinstance(value, (list, tuple)):
        description["length"] = len(value)
    return description


# Tried in order, each a dotted attribute path off the loaded SmolVLAPolicy
# instance. The first entry is confirmed against an actually-installed
# lerobot==0.6.0 checkpoint via:
#   grep -R "tokenizer" .venv-vla/lib/python3.12/site-packages/lerobot/policies/smolvla
#   -> smolvlm_with_expert.py:  self.processor = AutoProcessor.from_pretrained(model_id)
#   -> modeling_smolvla.py:     self.vlm_with_expert.processor.tokenizer.fake_image_token_id
# i.e. policy.model.vlm_with_expert.processor.tokenizer is the real HF
# tokenizer tied to this exact checkpoint's VLM backbone (config.vlm_model_name)
# -- never a new tokenizer constructed from a hardcoded model id. The
# remaining entries are plausible fallbacks in case this nesting moves again,
# same spirit as _SMOLVLA_IMPORT_CANDIDATES.
_SMOLVLA_TOKENIZER_ATTR_PATHS = [
    "model.vlm_with_expert.processor.tokenizer",
    "vlm_with_expert.processor.tokenizer",
    "model.vlm_with_expert.tokenizer",
    "processor.tokenizer",
    "tokenizer",
    "language_tokenizer",
    "vlm_processor.tokenizer",
]


def _find_smolvla_tokenizer(model):
    """Never raises. Returns (tokenizer_or_None, attempts) where attempts is
    a list of {"path": str, "ok": bool} dicts -- one per attribute path
    tried, in order, stopping at the first one that resolves to something
    callable (a tokenizer instance is called directly, e.g.
    tokenizer(text_list, ...)), so a caller can tell exactly which nesting
    worked (or that none did) without re-deriving it."""
    attempts = []
    for path in _SMOLVLA_TOKENIZER_ATTR_PATHS:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and callable(obj):
            attempts.append({"path": path, "ok": True})
            return obj, attempts
        attempts.append({"path": path, "ok": False})
    return None, attempts


def _tokenize_language(tokenizer, model_config, instructions: list, device: str):
    """Tokenizes a batch of instruction strings the same way LeRobot's own
    TokenizerProcessorStep does (lerobot/processor/tokenizer_processor.py),
    so the resulting observation.language.tokens/attention_mask match what
    SmolVLAPolicy.select_action() reads out of the batch -- including
    appending a trailing newline the same way NewLineTaskProcessorStep does
    (some tokenizers, e.g. PaliGemma's, expect one) and reading
    tokenizer_max_length/pad_language_to from the policy's own config
    instead of guessing padding/truncation settings."""
    import torch

    padded_instructions = [text if text.endswith("\n") else f"{text}\n" for text in instructions]
    max_length = getattr(model_config, "tokenizer_max_length", 48) if model_config is not None else 48
    padding = getattr(model_config, "pad_language_to", "longest") if model_config is not None else "longest"

    encoded = tokenizer(
        padded_instructions,
        max_length=max_length,
        truncation=True,
        padding=padding,
        padding_side="right",
        return_tensors="pt",
    )
    tokens = encoded["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.bool) if "attention_mask" in encoded else None
    return tokens, attention_mask


# HuggingFaceVLA/smolvla_libero's real input_features (confirmed via its
# config.json, see policy_semantics/manifest.py's _SMOLVLA_LIBERO_MANIFEST):
# two camera keys, 8-dim state. This project has one camera today, so
# that single image is replicated across both keys (same "one camera
# satisfies N expected image inputs" approach _build_visual_batch() uses
# for smolvla_base) -- and there is no real 8-dim EE-pose+gripper-qpos
# state plumbed from this project's robot_state dict yet, so
# observation.state is a zero vector, explicitly marked degraded_input
# in the resulting CanonicalRobotCommand rather than silently pretended
# to be a real reading.
_SMOLVLA_LIBERO_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
_SMOLVLA_LIBERO_STATE_DIM = 8


def _run_smolvla_libero_inference(model, preprocessor_pipeline, postprocessor_pipeline, model_input: dict):
    """Real official-processor path for HuggingFaceVLA/smolvla_libero (or
    any future model_id added to _MODELS_WITH_OFFICIAL_PROCESSOR_FILES):
    builds the plain-dict observation LeRobot's own preprocessor pipeline
    expects, runs it, adds the batch dimension the loaded saved-config
    pipeline does not add on its own for image/state tensors (confirmed
    empirically this session -- observation.language.tokens/
    attention_mask already come back batched from the tokenizer step,
    but observation.images.*/observation.state do not), calls
    policy.select_action(), then runs the real official postprocessor
    (unnormalizes using this checkpoint's own baked-in dataset stats) --
    never this module's own guessed normalization. Returns a
    NativePolicyAction with postprocessor_used=True."""
    import numpy as np
    import torch

    from policy_semantics.native_policy_action import NativePolicyAction

    device = resolve_device()

    image_array = model_input.get("image")
    if image_array is None:
        image_chw_float = np.zeros((3, 256, 256), dtype=np.float32)
    else:
        # This checkpoint's saved preprocessor pipeline expects CHW
        # float32 in [0, 1] as input (confirmed empirically: an HWC
        # uint8 array reaches its internal resize_with_pad() bilinear
        # interpolation and fails with "upsample_bilinear2d_out_frame
        # not implemented for 'Byte'" -- this saved config apparently
        # has no dtype/layout-conversion step of its own, unlike a raw
        # LeRobot dataset's camera frame).
        array = np.asarray(image_array)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
        image_chw_float = array.astype(np.float32)
        if image_chw_float.max() > 1.0:
            image_chw_float = image_chw_float / 255.0

    instruction = model_input.get("instruction", "")

    observation = {key: image_chw_float for key in _SMOLVLA_LIBERO_IMAGE_KEYS}
    observation["observation.state"] = np.zeros((_SMOLVLA_LIBERO_STATE_DIM,), dtype=np.float32)
    observation["task"] = instruction

    processed = preprocessor_pipeline(observation)

    batch = {}
    for key in (*_SMOLVLA_LIBERO_IMAGE_KEYS, "observation.state"):
        value = processed[key]
        if isinstance(value, torch.Tensor) and value.dim() in (1, 3):
            value = value.unsqueeze(0)
        batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    for key in ("observation.language.tokens", "observation.language.attention_mask", "task"):
        value = processed[key]
        batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value

    with torch.inference_mode():
        raw_action = model.select_action(batch)

    postprocessed = postprocessor_pipeline({"action": raw_action})
    action_tensor = postprocessed["action"] if isinstance(postprocessed, dict) else postprocessed
    values = action_tensor.detach().cpu().flatten().tolist()

    return NativePolicyAction(
        values=values,
        source_policy=model_input.get("model_id_or_path", "HuggingFaceVLA/smolvla_libero"),
        postprocessor_used=True,
        metadata={
            "degraded_input": True,  # observation.state is a zero placeholder, see function docstring
            "raw_model_action": raw_action.detach().cpu().flatten().tolist(),
        },
    )


def _run_smolvla_inference(model, processor, model_input: dict) -> Any:
    """Tries each plausible calling convention for the loaded model in
    turn, raising a RuntimeError naming exactly which ones were tried
    if none apply -- generic_vla_server.py turns that into a structured
    inference_failed error rather than a crash, and the reason string
    is specific enough to say what needs updating.

    The batch's visual feature keys are read from the loaded policy's own
    config (see _extract_visual_feature_specs) rather than hardcoded, since
    LeRobot's SmolVLAPolicy expects one key per camera the checkpoint was
    trained with (e.g. observation.images.camera1/2/3) and rejects a plain
    "observation.image" key with an "All image features are missing from the
    batch" error. This project has a single camera, so that one image is
    replicated across every expected key. observation.state is likewise
    built as a tensor (see _build_state_tensor) rather than passed through as
    the raw robot_state dict, which LeRobot's policy code indexes/reshapes as
    if it were already a tensor (`'dict' object has no attribute 'ndim'`).
    task is kept as the plain natural-language instruction string (wrapped in
    a length-1 list for batch size 1) -- never the TaskGoal dict -- and is
    additionally tokenized into observation.language.tokens/attention_mask
    (see _find_smolvla_tokenizer/_tokenize_language) using the tokenizer tied
    to this exact loaded checkpoint, since SmolVLAPolicy.select_action()
    reads those two keys directly (KeyError: 'observation.language.tokens'
    if they're missing) rather than tokenizing the plain "task" string
    itself. Right before the model is actually called, every tensor in the
    batch is re-aligned to the model's own actual (dtype, device) (see
    _resolve_model_dtype_device/_align_batch_dtype_device) -- floating
    tensors (images, state) get cast, integer/bool tensors (language tokens/
    attention mask) only get moved, never cast -- so a per-key dtype drift
    anywhere upstream can't produce a 'mat1 and mat2 must have the same
    dtype' mismatch."""
    import torch

    device = resolve_device()
    configured_dtype = getattr(torch, resolve_dtype_name(), torch.float32)
    dtype, target_device = _resolve_model_dtype_device(model, configured_dtype, device)

    visual_feature_specs = _extract_visual_feature_specs(model)
    image_tensor = _image_array_to_tensor(model_input.get("image"), target_device, dtype)
    visual_batch, visual_feature_specs = _build_visual_batch(image_tensor, visual_feature_specs)

    state_shape = _extract_state_feature_shape(model)
    state_tensor = _build_state_tensor(model_input.get("robot_state"), state_shape, target_device, dtype)

    instruction = model_input.get("instruction", "")
    task_texts = [instruction] if isinstance(instruction, str) else list(instruction)

    tokenizer, tokenizer_attempts = _find_smolvla_tokenizer(model)
    language_tokens = None
    language_attention_mask = None
    if tokenizer is not None:
        try:
            language_tokens, language_attention_mask = _tokenize_language(
                tokenizer, getattr(model, "config", None), task_texts, target_device
            )
        except Exception as exc:  # noqa: BLE001 -- tokenization failure is reported via
            # debug_info/the eventual select_action KeyError below, not a crash here.
            print(f"[model_loader] SmolVLA language tokenization failed: {exc}")

    batch = {
        **visual_batch,
        "observation.state": state_tensor,
        "task": task_texts,
    }
    if language_tokens is not None:
        batch["observation.language.tokens"] = language_tokens
    if language_attention_mask is not None:
        batch["observation.language.attention_mask"] = language_attention_mask

    # Final normalization pass, right before the model is called: whatever
    # dtype/device each tensor above was actually built with, every floating
    # tensor ends up on the model's own (dtype, device) and every
    # integer/bool tensor ends up on the model's device only -- see
    # _align_batch_dtype_device's docstring for why floating vs. non-floating
    # are handled differently.
    batch = _align_batch_dtype_device(batch, dtype, target_device)

    debug_info = {
        "batch_keys": list(batch.keys()),
        "expected_visual_feature_keys": list(visual_feature_specs.keys()),
        "expected_state_shape": list(state_shape),
        "tensor_shapes": {
            key: (list(value.shape) if hasattr(value, "shape") else None) for key, value in batch.items()
        },
        "value_types": {key: _describe_batch_value(value) for key, value in batch.items()},
        "tensor_info": {
            key: (
                {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
                if isinstance(value, torch.Tensor)
                else None
            )
            for key, value in batch.items()
        },
        "tokenizer_found": tokenizer is not None,
        "tokenizer_attempts": tokenizer_attempts,
        "language_tokens_present": "observation.language.tokens" in batch,
        "language_tokens_shape": list(language_tokens.shape) if language_tokens is not None else None,
        "language_tokens_dtype": str(language_tokens.dtype) if language_tokens is not None else None,
        "language_attention_mask_present": "observation.language.attention_mask" in batch,
        "language_attention_mask_shape": (
            list(language_attention_mask.shape) if language_attention_mask is not None else None
        ),
    }

    with torch.inference_mode():
        try:
            if hasattr(model, "select_action"):
                # LeRobot policy interface: select_action(batch) -> action tensor/dict.
                return model.select_action(batch)

            if hasattr(model, "predict_action"):
                # Some LeRobot-adjacent policies expose predict_action(batch) instead.
                return model.predict_action(batch)

            if processor is not None and hasattr(model, "generate"):
                # transformers-style processor+generate fallback.
                inputs = processor(model_input.get("instruction", ""), model_input.get("image")).to(
                    target_device, dtype=dtype
                )
                return model.generate(**inputs)
        except Exception as exc:  # noqa: BLE001 -- re-raised (not as RuntimeError, so
            # generic_vla_server.py buckets this as inference_failed, not
            # model_not_loaded) with the batch debug info attached, and also
            # printed here so it shows up in the server's own log even if a
            # client never reads the structured response.
            print(f"[model_loader] SmolVLA /predict inference failed: {exc}. debug_info={debug_info}")
            raise Exception(f"SmolVLA inference failed: {exc} | debug_info={debug_info}") from exc

    raise RuntimeError(
        f"Loaded SmolVLA model ({type(model).__module__}.{type(model).__name__}) has none of the expected "
        "inference methods (select_action, predict_action, generate) -- the model_loader.py dispatch needs "
        f"updating for this model's actual API. debug_info={debug_info}"
    )


def _run_openvla_inference(model, processor, model_input: dict) -> Any:
    import torch

    dtype = getattr(torch, resolve_dtype_name(), torch.bfloat16)
    inputs = processor(model_input.get("instruction", ""), model_input.get("image")).to(
        resolve_device(), dtype=dtype
    )
    return model.predict_action(**inputs, unnorm_key=model_input.get("unnorm_key", "bridge_orig"), do_sample=False)
