"""OpenVLA action adapter skeleton (v0).

Deliberately does NOT convert OpenVLA's raw output into an executable
action yet -- OpenVLA's own action space/normalization (unnorm_key,
gripper convention, frame convention) has not been verified against
this project's normalized 7-DoF action_schema. Every
normalize_model_output() call returns action=None,
project_action_available=False, reason="openvla_action_adapter_required"
regardless of what raw_output contains, so RealVLAPolicyClient always
falls back for this family until a real decoder is written and wired
in below (see _decode_openvla_action()).

This exists so swapping SmolVLA for OpenVLA later means "point
model_registry.py at this adapter (already done) and implement
_decode_openvla_action()", not a server/client rewrite -- the
interface is ready even though the decoder isn't.

Loading OpenVLA itself (vla_server/model_loader.py) is unaffected by
this file -- see openvla_server_real/colab_vla_server.py for the
existing, separate OpenVLA-specific dryrun/Drive-cache experiment this
does not replace.
"""

from typing import Any, Optional

from vla_adapters.base_vla_adapter import BaseVLAAdapter


def _to_jsonable(value):
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):  # numpy array
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class OpenVLAActionAdapter(BaseVLAAdapter):
    model_family = "openvla"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.unnorm_key = self.config.get("unnorm_key", "bridge_orig")

    def build_model_input(self, policy_input_dict: dict) -> dict:
        return {
            "instruction": policy_input_dict.get("instruction", ""),
            "image": policy_input_dict.get("image"),
            "unnorm_key": self.unnorm_key,
        }

    def normalize_model_output(self, raw_output: Any, context: dict) -> dict:
        phase = context.get("phase") or "move_to_object"

        # TODO (future work, not this v0): once OpenVLA's raw action
        # space is verified against this project's normalized 7-DoF
        # schema (units, frame convention, gripper convention,
        # unnorm_key correctness), implement _decode_openvla_action(
        # raw_output) here and return a real action the same way
        # SmolVLAActionAdapter does. Until then, always refuse -- see
        # module docstring.
        return {
            "action": None,
            "phase": phase,
            "done": False,
            "info": {
                "model_family": self.model_family,
                "adapter_used": "OpenVLAActionAdapter",
                "raw_model_output_available": raw_output is not None,
                "raw_model_output": _to_jsonable(raw_output) if raw_output is not None else None,
                "project_action_available": False,
                "reason": "openvla_action_adapter_required",
            },
        }

    def health_info(self) -> dict:
        return {
            "model_family": self.model_family,
            "adapter": "OpenVLAActionAdapter",
            "status": "raw_output_only (no verified action decoder yet -- see module docstring)",
        }
