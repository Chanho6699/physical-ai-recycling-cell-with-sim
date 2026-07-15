"""PandaCommandSafetyFilter -- the last check before a CanonicalRobotCommand
reaches PyBulletPandaBackend, per this task's data flow:

    CanonicalRobotCommand -> PandaCommandSafetyFilter -> PyBulletPandaBackend

Two, deliberately separate, responsibilities:
  1. Reject outright anything non-finite (NaN/Inf) -- never execute it,
     regardless of magnitude.
  2. Clip translation/rotation magnitude to this project's existing
     configured per-step limits (the same max_translation_step/
     max_rotation_step already used by
     policy/vla_action_postprocessor.py and
     vla_adapters/smolvla_adapter.py's action_postprocess config, reused
     here rather than inventing a second set of limits) and mark
     safety_clipped=True on the result.

This is independent from (and runs in addition to) SafetyGate/
SafetySupervisor (robot_core/) deciding whether the resulting motion is
currently safe to execute at all -- same two-separate-checks split
policy/vla_action_postprocessor.py's docstring already documents for the
legacy wire format.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

from policy_semantics.canonical_command import CanonicalRobotCommand

DEFAULT_MAX_TRANSLATION_STEP_M = 0.03
DEFAULT_MAX_ROTATION_STEP_RAD = 0.10


@dataclass
class SafetyFilterResult:
    command: Optional[CanonicalRobotCommand]  # None if rejected outright
    accepted: bool
    rejected_reason: Optional[str]
    clipped: bool


class PandaCommandSafetyFilter:
    def __init__(
        self,
        max_translation_step_m: float = DEFAULT_MAX_TRANSLATION_STEP_M,
        max_rotation_step_rad: float = DEFAULT_MAX_ROTATION_STEP_RAD,
    ):
        self.max_translation_step_m = abs(max_translation_step_m)
        self.max_rotation_step_rad = abs(max_rotation_step_rad)

    def apply(self, command: CanonicalRobotCommand) -> SafetyFilterResult:
        all_values = list(command.translation_m) + list(command.rotation_axis_angle_rad) + [
            command.gripper_opening_01
        ]
        for index, value in enumerate(all_values):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return SafetyFilterResult(
                    command=None,
                    accepted=False,
                    rejected_reason=f"non_numeric_value at flat index {index}: {all_values}",
                    clipped=False,
                )
            if math.isnan(value) or math.isinf(value):
                return SafetyFilterResult(
                    command=None,
                    accepted=False,
                    rejected_reason=f"nan_or_inf_value at flat index {index}: {all_values}",
                    clipped=False,
                )

        clipped = False
        translation = list(command.translation_m)
        for index in range(3):
            bounded = max(-self.max_translation_step_m, min(self.max_translation_step_m, translation[index]))
            if bounded != translation[index]:
                clipped = True
            translation[index] = bounded

        rotation = list(command.rotation_axis_angle_rad)
        for index in range(3):
            bounded = max(-self.max_rotation_step_rad, min(self.max_rotation_step_rad, rotation[index]))
            if bounded != rotation[index]:
                clipped = True
            rotation[index] = bounded

        gripper = max(0.0, min(1.0, command.gripper_opening_01))
        if gripper != command.gripper_opening_01:
            clipped = True

        filtered = replace(
            command,
            translation_m=tuple(translation),
            rotation_axis_angle_rad=tuple(rotation),
            gripper_opening_01=gripper,
            safety_clipped=command.safety_clipped or clipped,
        )
        return SafetyFilterResult(command=filtered, accepted=True, rejected_reason=None, clipped=clipped)
