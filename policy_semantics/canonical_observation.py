"""CanonicalObservation -- this project's one normalized observation
shape, the input-side counterpart of CanonicalRobotCommand.

vla_server/generic_vla_server.py already decodes an HTTP request into a
plain dict (image array, instruction, robot_state, step_index, phase --
see its policy_input_dict). CanonicalObservation is that same
information re-expressed by *camera role* (e.g. "wrist", "main") rather
than a single positional image, so a manifest-aware ObservationAdapter
(see interfaces.py) can tell which of a checkpoint's
required_camera_roles it actually has real data for, and mark the rest
as missing (contributing to CanonicalRobotCommand.degraded_input)
instead of silently zero-filling.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CanonicalObservation:
    images_by_role: Dict[str, Any]  # e.g. {"wrist": np.ndarray} -- missing roles simply absent, not None-filled
    instruction: str
    robot_state: dict
    step_index: int = 0
    phase: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def missing_camera_roles(self, required_roles) -> list:
        return [role for role in required_roles if role not in self.images_by_role]
