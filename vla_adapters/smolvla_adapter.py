"""SmolVLA action adapter (v0).

Normalizes SmolVLA's raw model output into this project's normalized
7-DoF action ([dx, dy, dz, droll, dpitch, dyaw, gripper]). SmolVLA's
actual raw output shape can vary depending on how it was loaded/called:

  - a plain 7-number action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  - a dict with an "action" key wrapping any of the shapes below
  - a dict with an "actions" key (chunked/action-horizon policies)
  - a chunked action with no dict wrapper: [T, 7]
  - a batched chunk: [B, T, 7] (or [B, 7])
  - any of the above as a numpy array or torch tensor instead of a
    plain list/tuple

This adapter tries each of those shapes in turn (via a small bounded
recursive "peel one dimension, select by step_index" pass) rather than
assuming one, and rejects -- with a structured, fallback-triggering
reason plus a `raw_output_summary` (type/shape, never the full tensor)
-- anything it can't confidently resolve down to a flat 7-number
vector instead of guessing.
"""

import math
from typing import Any, Optional

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

    def build_model_input(self, policy_input_dict: dict) -> dict:
        # The actual tensor/batch construction a real SmolVLA forward
        # pass needs is vla_server/model_loader.py's job (it owns the
        # loaded processor/model) -- this adapter only needs enough
        # context (instruction, image, step_index) to interpret
        # whatever raw output comes back in normalize_model_output().
        return {
            "instruction": policy_input_dict.get("instruction", ""),
            "image": policy_input_dict.get("image"),
            "robot_state": policy_input_dict.get("robot_state") or {},
            "step_index": policy_input_dict.get("step_index", 0),
            "phase": policy_input_dict.get("phase"),
        }

    MAX_PEEL_DEPTH = 4

    def normalize_model_output(self, raw_output: Any, context: dict) -> dict:
        step_index = context.get("step_index", 0)
        phase = context.get("phase") or "move_to_object"

        try:
            raw_action = self._extract_raw_action(raw_output, step_index)
            action, debug = self._validate_and_clip(raw_action)
        except ValueError as exc:
            return self._reject(phase, str(exc), raw_output)

        return {
            "action": action,
            "phase": phase,
            "done": False,
            "info": {
                "model_family": self.model_family,
                "adapter_used": "SmolVLAActionAdapter",
                "raw_model_output_available": True,
                "action_postprocess": debug,
            },
        }

    def _extract_raw_action(self, raw_output: Any, step_index: int) -> list:
        """Unwraps a {"action": ...}/{"actions": ...} dict first (if
        present), then peels arbitrarily-nested batch/chunk dimensions
        via _peel_to_vector() -- see module docstring for the shapes
        this covers. Raises ValueError (never crashes) for anything it
        can't resolve down to a flat 7-number vector."""
        value = self._to_plain(raw_output)

        if isinstance(value, dict):
            if "action" in value:
                value = value["action"]
            elif "actions" in value:
                value = value["actions"]
            else:
                raise ValueError(f"smolvla_raw_output_missing_action_field: keys={list(value.keys())}")

        return self._peel_to_vector(value, step_index, depth=0)

    def _peel_to_vector(self, value: Any, step_index: int, depth: int) -> list:
        """Recursively selects one nesting level at a time (batch
        dim, then chunk/time dim, ...) using step_index (clamped) as
        the selector at every level, until a flat 7-number vector is
        reached or MAX_PEEL_DEPTH is exceeded. This is a deliberate
        simplification -- without explicit shape metadata from the
        model, [B, 7] and [T, 7] are indistinguishable, so both are
        resolved the same way (index by step_index). A real
        integration that knows the exact SmolVLA output shape should
        prefer indexing it directly rather than relying on this."""
        value = self._to_plain(value)

        if isinstance(value, (list, tuple)):
            if len(value) == 7 and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
                return list(value)

            if depth >= self.MAX_PEEL_DEPTH:
                raise ValueError(
                    f"smolvla_raw_output_shape_too_deep: exceeded {self.MAX_PEEL_DEPTH} nesting levels "
                    f"without reaching a 7-number vector; last shape={self._shape_of(value)}"
                )
            if len(value) == 0:
                raise ValueError("smolvla_raw_output_empty_sequence")

            first = self._to_plain(value[0])
            if isinstance(first, (list, tuple)):
                index = min(max(step_index, 0), len(value) - 1)
                return self._peel_to_vector(value[index], step_index, depth + 1)

            # A flat sequence that isn't length 7 -- not interpretable.
            raise ValueError(
                f"smolvla_raw_output_wrong_length: expected 7 numbers ([dx, dy, dz, droll, dpitch, dyaw, gripper]) "
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

    def _reject(self, phase: str, reason: str, raw_output: Any = None) -> dict:
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
