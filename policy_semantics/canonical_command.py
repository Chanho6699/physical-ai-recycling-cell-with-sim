"""CanonicalRobotCommand -- this project's one normalized command shape.

Every checkpoint integration (SmolVLA, OpenVLA, a future one) must
translate its raw output into this shape, or refuse -- never into a
bare float list interpreted purely by array position. This is the
"meaning" layer that policy_semantics exists to add on top of the
existing shape-only wire format.

Gripper polarity note (read before touching gripper conversion code):
this project's *existing* wire format -- action_adapter/adapter_v0.py's
RobotCommand, policy/vla_action_postprocessor.py, and
vla_adapters/smolvla_adapter.py's [dx, dy, dz, droll, dpitch, dyaw,
gripper] float -- uses gripper >= threshold(0.5) means "close" (1.0 =
closed). CanonicalRobotCommand uses the opposite, more intuitive
convention (gripper_opening_01: 1.0 = fully open, 0.0 = fully closed).
to_legacy_action_list()/to_legacy_robot_command() below invert between
the two explicitly -- do not assume the two floats mean the same thing
just because both are near 0.0/1.0.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from action_adapter.adapter_v0 import RobotCommand

CONTROL_MODE_CARTESIAN_DELTA = "cartesian_delta"
TARGET_FRAME_ROBOT_BASE = "robot_base"

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = 0.0

# The existing wire format's polarity (action_adapter/adapter_v0.py,
# policy/vla_action_postprocessor.py, vla_adapters/smolvla_adapter.py):
# 1.0 = close. Opposite of CanonicalRobotCommand's gripper_opening_01.
_LEGACY_WIRE_GRIPPER_CLOSE_VALUE = 1.0
_LEGACY_WIRE_GRIPPER_OPEN_VALUE = 0.0


@dataclass
class CanonicalRobotCommand:
    translation_m: Tuple[float, float, float]  # [dx, dy, dz], meters
    rotation_axis_angle_rad: Tuple[float, float, float]  # [rx, ry, rz], radians
    gripper_opening_01: float  # 0.0 closed, 1.0 open
    duration_s: float  # how long this command is meant to act for (control period)
    source_policy: str
    adapter_name: str
    adapter_version: str
    control_mode: str = CONTROL_MODE_CARTESIAN_DELTA
    target_frame: str = TARGET_FRAME_ROBOT_BASE
    fallback_used: bool = False
    safety_clipped: bool = False
    degraded_input: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_action_list(self) -> list:
        """Bridges to the existing flat [dx, dy, dz, droll, dpitch, dyaw,
        gripper] wire format -- policy/vla_action_postprocessor.py and
        RealVLAPolicyClient still consume exactly this shape over HTTP,
        unchanged. Inverts gripper polarity (see module docstring)."""
        wire_gripper = (
            _LEGACY_WIRE_GRIPPER_CLOSE_VALUE
            if self.gripper_opening_01 <= 0.5
            else _LEGACY_WIRE_GRIPPER_OPEN_VALUE
        )
        return [
            float(self.translation_m[0]),
            float(self.translation_m[1]),
            float(self.translation_m[2]),
            float(self.rotation_axis_angle_rad[0]),
            float(self.rotation_axis_angle_rad[1]),
            float(self.rotation_axis_angle_rad[2]),
            wire_gripper,
        ]

    def to_legacy_robot_command(self) -> RobotCommand:
        """Bridges directly to action_adapter/adapter_v0.py's RobotCommand
        -- the type robot_sim/pybullet_panda_backend.py's apply_command()
        actually consumes -- for the local, in-process production path
        (PandaCommandSafetyFilter -> PyBulletPandaBackend). Deliberately
        does NOT go through action_adapter.adapter_v0.ActionAdapter.convert()
        (that class interprets a flat list purely by array position, the
        exact assume-nothing-about-meaning pattern this package replaces)
        -- this method is the meaning-preserving equivalent, built from
        already-verified CanonicalRobotCommand fields instead. adapter_v0
        itself is untouched/still used by other (legacy/mock) call sites."""
        gripper_command = "close" if self.gripper_opening_01 <= 0.5 else "open"
        return RobotCommand(
            target_dx=float(self.translation_m[0]),
            target_dy=float(self.translation_m[1]),
            target_dz=float(self.translation_m[2]),
            target_droll=float(self.rotation_axis_angle_rad[0]),
            target_dpitch=float(self.rotation_axis_angle_rad[1]),
            target_dyaw=float(self.rotation_axis_angle_rad[2]),
            gripper_command=gripper_command,
        )

    def to_info_dict(self) -> dict:
        """Cheap, JSON-safe summary for the /predict response's info dict
        -- so the full semantic record (source_policy, adapter identity,
        fallback/clip/degraded flags) survives even though the actual
        executed command downstream is still the flat legacy list."""
        return {
            "control_mode": self.control_mode,
            "target_frame": self.target_frame,
            "translation_m": list(self.translation_m),
            "rotation_axis_angle_rad": list(self.rotation_axis_angle_rad),
            "gripper_opening_01": self.gripper_opening_01,
            "duration_s": self.duration_s,
            "source_policy": self.source_policy,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "fallback_used": self.fallback_used,
            "safety_clipped": self.safety_clipped,
            "degraded_input": self.degraded_input,
            "metadata": self.metadata,
        }
