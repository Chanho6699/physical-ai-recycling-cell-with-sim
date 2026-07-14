"""Base interface for model-specific VLA adapters (v0).

A VLA adapter's only job is data transformation: given a decoded
request (build_model_input) or a model's raw output plus some context
(normalize_model_output), produce the next shape in the pipeline. An
adapter NEVER calls a model itself (that's vla_server/model_loader.py's
job) and NEVER talks HTTP (that's vla_server/generic_vla_server.py's
job). That boundary is what lets swapping SmolVLA for OpenVLA (or
anything else) mean "point model_registry.py at a different adapter +
model_loader.py at a different loader", never a rewrite of the server,
RealVLAPolicyClient, or the local robot control loop.

    request -> build_model_input() -> model_loader.run_inference()
            -> raw_output -> normalize_model_output() -> {action, phase, done, info}

`normalize_model_output()` is the safety-relevant boundary: it must
return `action=None` (never a fabricated 7-DoF guess) whenever it
isn't confident the raw output maps cleanly onto
[dx, dy, dz, droll, dpitch, dyaw, gripper]. generic_vla_server.py turns
`action=None` into a structured error response so the local
RealVLAPolicyClient falls back instead of executing a guess.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseVLAAdapter(ABC):
    model_family: str = "base"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def build_model_input(self, policy_input_dict: dict) -> dict:
        """Turns the server's already-decoded request (image array
        under "image", instruction, robot_state, task_goal,
        target_object_position, bin_position, step_index, phase,
        observation_source, visual_observation) into whatever shape
        this family's model/processor actually expects. Must not call
        a model or touch the network."""
        raise NotImplementedError

    @abstractmethod
    def normalize_model_output(self, raw_output: Any, context: dict) -> dict:
        """Turns a model's raw output (whatever shape run_inference()
        returned) into:

            {
              "action": [dx, dy, dz, droll, dpitch, dyaw, gripper] or None,
              "phase": str,
              "done": bool,
              "info": {
                "model_family": self.model_family,
                "adapter_used": <class name>,
                "raw_model_output_available": bool,
                ... (adapter-specific debug/reason fields)
              },
            }

        `action` MUST be None whenever this adapter cannot confidently
        map raw_output onto the project's normalized 7-DoF action --
        include a "reason" key in info explaining why, so the caller
        can surface a structured, fallback-triggering error."""
        raise NotImplementedError

    @abstractmethod
    def health_info(self) -> dict:
        """Cheap, side-effect-free info for /health -- must never load
        a model or touch the network/disk."""
        raise NotImplementedError
