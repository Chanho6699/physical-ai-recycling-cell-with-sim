"""SmolVLA action adapter (v0.1 -- manifest/gate aware).

Normalizes SmolVLA's raw model output into this project's normalized
7-DoF action ([dx, dy, dz, droll, dpitch, dyaw, gripper]) -- but only
when policy_semantics.compatibility_gate says this specific checkpoint's
action semantics have actually been verified against this project's
Panda target embodiment. See policy_semantics/manifest.py's
_SMOLVLA_BASE_MANIFEST for the concrete story this replaces: an earlier
version of this file treated a 6-number raw output as "7D Cartesian
delta minus a gripper channel" and padded it with a neutral gripper
value -- array length matched, but those 6 numbers are actually
SO-100/SO-101 joint-space values (a different robot's joint positions),
not this project's [dx, dy, dz, droll, dpitch, dyaw] at all. That filler
is now isolated in policy_semantics/adapters/legacy_shape_only_adapter.py
and only reachable via VLA_SMOKE_TEST_MODE=1 -- never on the production
path, regardless of what raw_output's shape looks like.

Production flow per /predict call, using the CompatibilityResult
vla_server/model_loader.py computed at load time (passed in via
context["compatibility"], see generic_vla_server.py):

  1. context["compatibility"]["passed"] is True (this checkpoint's
     manifest matched the Panda target on every CompatibilityGate check)
     -> raw_output must be a NativePolicyAction (model_loader.py already
     ran this checkpoint's official postprocessor -- see
     _run_smolvla_libero_inference()) -> SmolVLALiberoActionAdapter.decode()
     builds a CanonicalRobotCommand -> PandaCommandSafetyFilter clips/
     validates it -> bridged to the legacy flat-list wire format via
     CanonicalRobotCommand.to_legacy_action_list(). This is the real,
     meaning-preserving path -- currently reachable only once
     HuggingFaceVLA/smolvla_libero's remaining UNKNOWN manifest fields
     (see policy_semantics/manifest.py) are resolved.
  2. Otherwise, only if context["compatibility"]["shape_only_allowed"]
     is True (computed once at load time from VLA_SMOKE_TEST_MODE, see
     vla_server/model_loader.py) -- run the legacy shape-only filler,
     loudly marked semantic_action_valid=False.
  3. Otherwise -- refuse (action=None) with the gate's specific
     failure reasons, regardless of raw_output's shape. Matching array
     length alone is never treated as sufficient (this task's explicit
     requirement) -- lerobot/smolvla_base will never reach case 1 no
     matter what shape its output has, because its manifest fails on
     source_embodiment/action_space/rotation/frame/gripper convention,
     not on dimension.

SmolVLA's raw output shape itself can still vary by how it was loaded/
called (a plain vector, a {"action": ...}/{"actions": ...} dict, a
chunked/batched list, numpy/torch instead of plain list) --
_extract_raw_action()/_peel_to_vector() handle that shape-decoding step
the same way regardless of which of the three cases above applies;
valid_lengths controls whether 6 is accepted (smoke-test only) or only
7 (production).
"""

import math
import warnings
from typing import Any, Optional

from policy_semantics.adapters.legacy_shape_only_adapter import (
    ADAPTER_NAME as LEGACY_ADAPTER_NAME,
    ADAPTER_VERSION as LEGACY_ADAPTER_VERSION,
    fill_to_seven,
)
from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
from policy_semantics.manifest import get_manifest
from policy_semantics.native_policy_action import NativePolicyAction
from policy_semantics.safety_filter import PandaCommandSafetyFilter
from vla_adapters.base_vla_adapter import BaseVLAAdapter

DEFAULT_MAX_TRANSLATION_STEP = 0.03
DEFAULT_MAX_ROTATION_STEP = 0.10
DEFAULT_GRIPPER_THRESHOLD = 0.5


class SmolVLAActionAdapter(BaseVLAAdapter):
    model_family = "smolvla"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        postprocess = self.config.get("action_postprocess", {}) or {}
        max_translation_step = abs(float(postprocess.get("max_translation_step", DEFAULT_MAX_TRANSLATION_STEP)))
        max_rotation_step = abs(float(postprocess.get("max_rotation_step", DEFAULT_MAX_ROTATION_STEP)))
        self.translation_clip = (-max_translation_step, max_translation_step)
        self.rotation_clip = (-max_rotation_step, max_rotation_step)
        self.gripper_threshold = float(postprocess.get("gripper_threshold", DEFAULT_GRIPPER_THRESHOLD))
        self.clip_action = bool(postprocess.get("clip_action", True))
        # Production (compatibility.passed) path only -- see
        # normalize_model_output(). Currently the only checkpoint that can
        # ever reach compatibility.passed=True is HuggingFaceVLA/smolvla_libero
        # (see policy_semantics/manifest.py); if a second, differently-scaled
        # checkpoint is registered later, this single hardcoded adapter
        # instance needs to become model_id-dispatched too.
        self._libero_action_adapter = SmolVLALiberoActionAdapter()
        self._safety_filter = PandaCommandSafetyFilter(
            max_translation_step_m=max_translation_step, max_rotation_step_rad=max_rotation_step
        )

    def build_model_input(self, policy_input_dict: dict) -> dict:
        # The actual tensor/batch construction a real SmolVLA forward
        # pass needs is vla_server/model_loader.py's job (it owns the
        # loaded processor/model) -- this adapter only needs enough
        # context (instruction, image, step_index) to interpret
        # whatever raw output comes back in normalize_model_output().
        return {
            "instruction": policy_input_dict.get("instruction", ""),
            "image": policy_input_dict.get("image"),
            # Multi-camera observation (e.g. {"main": ..., "wrist": ...}),
            # passed through unchanged -- see
            # vla_server/model_loader.py's _run_smolvla_libero_inference()
            # for where this actually gets used instead of the legacy
            # single-image duplication.
            "images_by_role": policy_input_dict.get("images_by_role"),
            "robot_state": policy_input_dict.get("robot_state") or {},
            "step_index": policy_input_dict.get("step_index", 0),
            "phase": policy_input_dict.get("phase"),
            "seed": policy_input_dict.get("seed"),
            "model_id_or_path": self.config.get("model_id_or_path", "unknown"),
        }

    MAX_PEEL_DEPTH = 4

    def normalize_model_output(self, raw_output: Any, context: dict) -> dict:
        step_index = context.get("step_index", 0)
        phase = context.get("phase") or "move_to_object"
        compatibility = context.get("compatibility") or {}
        # compatibility["model_id"] (see policy_semantics.compatibility_gate
        # .CompatibilityResult) is the model_id_or_path vla_server/model_loader.py
        # actually resolved and loaded at load_model_once() time (env-var
        # driven, see model_loader.resolve_model_id_or_path()) -- the
        # authoritative source. self.config["model_id_or_path"] is only
        # ever populated when a VLA_BACKEND_CONFIG_PATH JSON file
        # separately declares that same key (most server launches, incl.
        # the plain env-var-only ones this task's benchmarks use, never
        # set one), so falling back to it first would silently resolve
        # to "unknown" -> get_manifest("unknown") -> an all-UNKNOWN
        # manifest -- harmless before this task's native-gripper-range
        # fix (the old decode() never consulted manifest fields for its
        # fixed (-1,1) formula), but decode() now genuinely depends on
        # manifest.native_gripper_range being correct, so this ordering
        # bug had to be fixed together with it.
        model_id = compatibility.get("model_id") or self.config.get("model_id_or_path", "unknown")

        if compatibility.get("passed"):
            # Reached only once a checkpoint's manifest has actually
            # matched this project's Panda target on every
            # CompatibilityGate check (source embodiment, action space,
            # frame, rotation representation, gripper convention,
            # axis_convention_verified, official processor wired, ...) --
            # see policy_semantics/manifest.py for exactly which manifest
            # (if any) currently reaches this. raw_output here must be a
            # NativePolicyAction (model_loader.py's official-processor
            # path) -- a raw model tensor is refused, never guessed at.
            if not isinstance(raw_output, NativePolicyAction):
                return self._reject(
                    phase,
                    f"compatibility_gate_passed_but_raw_output_not_postprocessed: expected NativePolicyAction, "
                    f"got {type(raw_output).__name__} -- vla_server/model_loader.py must run this checkpoint's "
                    "official postprocessor before compatibility.passed can be trusted",
                    raw_output,
                    compatibility,
                )

            manifest = get_manifest(model_id)
            canonical_command = self._libero_action_adapter.decode(
                raw_output, manifest, context={"degraded_input": raw_output.metadata.get("degraded_input", False)}
            )
            if canonical_command is None:
                return self._reject(
                    phase,
                    "action_adapter_declined: SmolVLALiberoActionAdapter.decode() returned None "
                    "(non-postprocessed input or wrong action length)",
                    raw_output,
                    compatibility,
                )

            # Captured before PandaCommandSafetyFilter.apply() below so
            # the response can show both sides of that clip -- this is
            # the SmolVLALiberoActionAdapter-decoded command (translation/
            # rotation already scaled into meters/radians, gripper
            # already converted to gripper_opening_01), not yet clipped
            # to this project's own configured per-step step-size limits.
            pre_safety_filter_command = canonical_command.to_info_dict()

            filter_result = self._safety_filter.apply(canonical_command)
            if not filter_result.accepted:
                return self._reject(
                    phase, f"safety_filter_rejected: {filter_result.rejected_reason}", raw_output, compatibility
                )
            canonical_command = filter_result.command

            action = canonical_command.to_legacy_action_list()
            debug = {
                "canonical_command": canonical_command.to_info_dict(),
                "canonical_command_pre_safety_filter": pre_safety_filter_command,
                "safety_filter_clipped": filter_result.clipped,
                "safety_clipped": canonical_command.safety_clipped,
                "degraded_input": canonical_command.degraded_input,
            }
            return self._accept(phase, action, debug, compatibility, semantic_action_valid=True)

        if compatibility.get("shape_only_allowed"):
            # shape_only_allowed is already exactly "was smoke_test_mode on
            # when CompatibilityGate ran at load time" (see
            # policy_semantics/compatibility_gate.py) -- the single source
            # of truth, not re-derived from the (informational-only)
            # per-request smoke_test_mode flag below.
            try:
                raw_action = self._extract_raw_action(raw_output, step_index, valid_lengths=(6, 7))
            except ValueError as exc:
                return self._reject(phase, str(exc), raw_output, compatibility)

            action_7d = fill_to_seven(raw_action)
            if len(raw_action) == 6:
                warnings.warn(
                    f"{LEGACY_ADAPTER_NAME}: filling missing gripper channel for {model_id!r} "
                    "(smoke_test_mode only -- see policy_semantics/adapters/legacy_shape_only_adapter.py)",
                    stacklevel=2,
                )
                print(
                    f"[SmolVLAActionAdapter] SMOKE TEST ONLY: {LEGACY_ADAPTER_NAME} filled a 6D raw "
                    f"action from {model_id!r} with a neutral gripper value -- semantic_action_valid=False"
                )

            try:
                action, debug = self._validate_and_clip(action_7d)
            except ValueError as exc:
                return self._reject(phase, str(exc), raw_output, compatibility)
            debug["semantic_action_valid"] = False
            debug["raw_action_length"] = len(raw_action)
            debug["legacy_adapter"] = f"{LEGACY_ADAPTER_NAME} {LEGACY_ADAPTER_VERSION}"
            return self._accept(
                phase, action, debug, compatibility, semantic_action_valid=False, smoke_test_mode=True
            )

        reasons = compatibility.get("reasons") or [
            "no compatibility_result available -- model_loader.get_compatibility_result() returned nothing "
            "(model not loaded via load_model_once(), or CompatibilityGate never ran for this model_family)"
        ]
        return self._reject(
            phase,
            f"compatibility_gate_rejected: model_id={model_id!r} is not verified-compatible with the "
            f"Panda target embodiment -- {reasons}. Set VLA_SMOKE_TEST_MODE=1 to run a shape-only "
            "(semantic_action_valid=false) smoke test instead of refusing outright.",
            raw_output,
            compatibility,
        )

    def _accept(
        self,
        phase: str,
        action: list,
        debug: dict,
        compatibility: dict,
        semantic_action_valid: bool,
        smoke_test_mode: bool = False,
    ) -> dict:
        info = {
            "model_family": self.model_family,
            "adapter_used": "SmolVLAActionAdapter",
            "raw_model_output_available": True,
            "action_postprocess": debug,
            "compatibility": compatibility,
            "semantic_action_valid": semantic_action_valid,
        }
        if "degraded_input" in debug:
            # Also surfaced at the top level (not just nested in
            # action_postprocess) so it's as easy to check as
            # semantic_action_valid -- and so it survives
            # RealVLAPolicyClient.predict_action() overwriting
            # info["action_postprocess"] with its own client-side
            # postprocess debug (see that method's merge, which now
            # preserves this dict's other keys but replaces this exact
            # sub-dict wholesale).
            info["degraded_input"] = debug["degraded_input"]
        if smoke_test_mode:
            info["smoke_test_mode"] = True
        return {"action": action, "phase": phase, "done": False, "info": info}

    def _extract_raw_action(self, raw_output: Any, step_index: int, valid_lengths: tuple = (7,)) -> list:
        """Unwraps a {"action": ...}/{"actions": ...} dict first (if
        present), then peels arbitrarily-nested batch/chunk dimensions
        via _peel_to_vector() -- see module docstring for the shapes
        this covers. Raises ValueError (never crashes) for anything it
        can't resolve down to a flat vector of one of valid_lengths --
        production calls this with valid_lengths=(7,) only; the
        smoke-test-only path is the one that allows (6, 7)."""
        value = self._to_plain(raw_output)

        if isinstance(value, dict):
            if "action" in value:
                value = value["action"]
            elif "actions" in value:
                value = value["actions"]
            else:
                raise ValueError(f"smolvla_raw_output_missing_action_field: keys={list(value.keys())}")

        return self._peel_to_vector(value, step_index, depth=0, valid_lengths=valid_lengths)

    def _peel_to_vector(self, value: Any, step_index: int, depth: int, valid_lengths: tuple = (7,)) -> list:
        """Recursively selects one nesting level at a time (batch
        dim, then chunk/time dim, ...) using step_index (clamped) as
        the selector at every level, until a flat vector of one of
        valid_lengths is reached or MAX_PEEL_DEPTH is exceeded. This is
        a deliberate simplification -- without explicit shape metadata
        from the model, [B, N] and [T, N] are indistinguishable, so both
        are resolved the same way (index by step_index). A real
        integration that knows the exact SmolVLA output shape should
        prefer indexing it directly rather than relying on this."""
        value = self._to_plain(value)

        if isinstance(value, (list, tuple)):
            if len(value) in valid_lengths and all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
            ):
                return list(value)

            if depth >= self.MAX_PEEL_DEPTH:
                raise ValueError(
                    f"smolvla_raw_output_shape_too_deep: exceeded {self.MAX_PEEL_DEPTH} nesting levels "
                    f"without reaching a vector of length in {valid_lengths}; last shape={self._shape_of(value)}"
                )
            if len(value) == 0:
                raise ValueError("smolvla_raw_output_empty_sequence")

            first = self._to_plain(value[0])
            if isinstance(first, (list, tuple)):
                index = min(max(step_index, 0), len(value) - 1)
                return self._peel_to_vector(value[index], step_index, depth + 1, valid_lengths=valid_lengths)

            # A flat sequence whose length isn't in valid_lengths -- not interpretable.
            raise ValueError(
                f"smolvla_raw_output_wrong_length: expected a length in {valid_lengths} "
                f"at nesting depth {depth}, got length {len(value)}: {value}"
            )

        raise ValueError(f"smolvla_raw_output_unrecognized_shape: {type(value)!r}")

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if hasattr(value, "detach"):  # torch tensor
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):  # numpy array
            value = value.tolist()
        return value

    @classmethod
    def _shape_of(cls, value: Any):
        if isinstance(value, (list, tuple)):
            return [len(value)] + (cls._shape_of(value[0]) if len(value) > 0 else [])
        return []

    def _summarize_raw_output(self, raw_output: Any) -> dict:
        """Cheap, safe-to-log summary of an unrecognized raw output --
        never dumps the full tensor/array contents."""
        summary = {"type": type(raw_output).__name__}
        if hasattr(raw_output, "shape"):
            summary["shape"] = list(raw_output.shape)
        elif isinstance(raw_output, dict):
            summary["dict_keys"] = list(raw_output.keys())
        elif isinstance(raw_output, (list, tuple)):
            summary["shape"] = self._shape_of(raw_output)
        return summary

    def _validate_and_clip(self, raw_action: list):
        if len(raw_action) != 7:
            raise ValueError(
                "smolvla_action_wrong_length: expected 7 ([dx, dy, dz, droll, dpitch, dyaw, gripper]), "
                f"got {len(raw_action)}: {raw_action}"
            )

        for index, value in enumerate(raw_action):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"smolvla_action_non_numeric_value at index {index}: {raw_action}")
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"smolvla_action_nan_or_inf at index {index}: {raw_action}")

        action = [float(value) for value in raw_action]
        translation_clipped = False
        rotation_clipped = False

        if self.clip_action:
            for index in range(3):
                clipped = max(self.translation_clip[0], min(self.translation_clip[1], action[index]))
                if clipped != action[index]:
                    translation_clipped = True
                action[index] = clipped
            for index in range(3, 6):
                clipped = max(self.rotation_clip[0], min(self.rotation_clip[1], action[index]))
                if clipped != action[index]:
                    rotation_clipped = True
                action[index] = clipped

        raw_gripper = action[6]
        normalized_gripper = 1.0 if raw_gripper >= self.gripper_threshold else 0.0
        gripper_normalized = normalized_gripper != raw_gripper
        action[6] = normalized_gripper

        debug = {
            "raw_action": list(raw_action),
            "postprocessed_action": action,
            "translation_clipped": translation_clipped,
            "rotation_clipped": rotation_clipped,
            "gripper_normalized": gripper_normalized,
        }
        return action, debug

    def _reject(self, phase: str, reason: str, raw_output: Any = None, compatibility: Optional[dict] = None) -> dict:
        return {
            "action": None,
            "phase": phase,
            "done": False,
            "info": {
                "model_family": self.model_family,
                "adapter_used": "SmolVLAActionAdapter",
                "raw_model_output_available": True,
                "raw_output_summary": self._summarize_raw_output(raw_output),
                "project_action_available": False,
                "semantic_action_valid": False,
                "compatibility": compatibility or {},
                "reason": reason,
            },
        }

    def health_info(self) -> dict:
        return {
            "model_family": self.model_family,
            "adapter": "SmolVLAActionAdapter",
            "translation_clip": list(self.translation_clip),
            "rotation_clip": list(self.rotation_clip),
            "gripper_threshold": self.gripper_threshold,
        }
