"""Mock VLA adapter (v0).

Wraps the existing DummyOpenVLAPolicy phase engine (the same one
local-dummy/fastapi-dummy/real-vla-compatible-mock/
colab_vla_server.py's mock-action already use) behind the same
BaseVLAAdapter interface every other model family uses, so
model_registry.py/generic_vla_server.py can treat
model_family="mock-action" exactly like "smolvla"/"openvla" -- no
special-casing anywhere else in the server. Existing mock-action demos
keep working unchanged; this is a new home for the same behavior, not
a reimplementation of it.

Unlike a real model, DummyOpenVLAPolicy.predict_action() already does
"build input -> run inference -> produce output" in one call (there's
no separate model forward pass to decouple from adapter logic for a
scripted policy) -- vla_server/model_loader.py's run_inference() calls
predict_action() directly and hands this adapter the resulting
PolicyOutput as `raw_output`.
"""

from typing import Any, Optional

from policy.policy_types import PolicyInput
from vla_adapters.base_vla_adapter import BaseVLAAdapter


class MockVLAAdapter(BaseVLAAdapter):
    model_family = "mock-action"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

    def build_model_input(self, policy_input_dict: dict) -> dict:
        return {
            "policy_input": PolicyInput(
                image=policy_input_dict.get("image"),
                instruction=policy_input_dict.get("instruction", ""),
                robot_state=policy_input_dict.get("robot_state") or {},
                task_goal=policy_input_dict.get("task_goal") or {},
                target_object_position=policy_input_dict.get("target_object_position"),
                bin_position=policy_input_dict.get("bin_position"),
                step_index=policy_input_dict.get("step_index", 0),
                phase=policy_input_dict.get("phase"),
                observation_source=policy_input_dict.get("observation_source"),
                visual_observation=policy_input_dict.get("visual_observation"),
            )
        }

    def normalize_model_output(self, raw_output: Any, context: dict) -> dict:
        policy_output = raw_output  # a policy.policy_types.PolicyOutput
        info = dict(policy_output.info or {})
        info["model"] = "generic-mock-action"
        info["model_family"] = self.model_family
        info["adapter_used"] = "MockVLAAdapter"
        info["raw_model_output_available"] = True
        return {
            "action": list(policy_output.action),
            "phase": policy_output.phase,
            "done": bool(policy_output.done),
            "info": info,
        }

    def health_info(self) -> dict:
        return {
            "model_family": self.model_family,
            "adapter": "MockVLAAdapter",
            "note": "deterministic DummyOpenVLAPolicy phase engine, no real model",
        }
