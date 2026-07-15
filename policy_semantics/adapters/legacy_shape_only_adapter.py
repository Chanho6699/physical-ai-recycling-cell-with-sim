"""LegacyShapeOnlyAdapter -- the old 6D->7D gripper-filler, isolated.

This is exactly the logic vla_adapters/smolvla_adapter.py used to run
unconditionally in production: given a raw action of length 6 ([dx, dy,
dz, droll, dpitch, dyaw] with no gripper) or 7, pad a 6-length one with
a neutral gripper value so downstream code always gets 7 numbers.

That padding was only ever a *shape* fix, never a *meaning* fix -- it
was written back when lerobot/smolvla_base's 6-number output was
(incorrectly) assumed to be "Cartesian delta EE, gripper channel
missing". Manifest-level investigation (see
policy_semantics/manifest.py's _SMOLVLA_BASE_MANIFEST) found the real
answer: those 6 numbers are SO-100/SO-101 joint-space values, a
different robot's joint positions, not this project's Cartesian delta
schema minus one channel. Filling a 7th slot cannot fix that -- the
first 6 numbers themselves don't mean [dx, dy, dz, droll, dpitch, dyaw]
either.

This class is kept only for shape-only smoke testing (confirming a
forward pass runs end to end and produces *some* 7-length numeric
vector, e.g. to validate the serving pipeline plumbing) -- it must
never be reachable from a production code path. Every command it
produces has semantic_action_valid=False and a loud warning attached;
callers are responsible for only invoking this when
CompatibilityGate.check(...).shape_only_allowed is True (i.e.
smoke_test_mode was explicitly requested).
"""

import warnings
from typing import Optional

from policy_semantics.canonical_command import CanonicalRobotCommand

GRIPPER_NEUTRAL_VALUE = 0.0
ADAPTER_NAME = "LegacyShapeOnlyAdapter"
ADAPTER_VERSION = "v0-smoke-test-only"


def fill_to_seven(raw_action: list) -> list:
    """Pads a 6-length raw action with a neutral gripper value, or
    returns a 7-length one unchanged. Raises ValueError for anything
    else -- never guesses beyond that."""
    length = len(raw_action)
    if length == 7:
        return list(raw_action)
    if length == 6:
        return list(raw_action) + [GRIPPER_NEUTRAL_VALUE]
    raise ValueError(f"LegacyShapeOnlyAdapter expected length 6 or 7, got {length}: {raw_action}")


def build_shape_only_command(raw_action: list, source_policy: str) -> CanonicalRobotCommand:
    """Builds a CanonicalRobotCommand purely from array position, with
    no claim that the numbers mean [dx, dy, dz, droll, dpitch, dyaw,
    gripper] for the source checkpoint -- degraded_input=True and
    metadata.semantic_action_valid=False mark that explicitly. Emits a
    Python warning as well, so this is loud even if a caller ignores the
    returned metadata."""
    warnings.warn(
        f"{ADAPTER_NAME}: producing a SHAPE-ONLY command from {source_policy!r}'s raw output -- "
        "array position only, no verified semantic meaning. Never use this outside smoke_test_mode.",
        stacklevel=2,
    )

    action_7d = fill_to_seven(raw_action)
    gripper_filled = len(raw_action) == 6

    return CanonicalRobotCommand(
        translation_m=(action_7d[0], action_7d[1], action_7d[2]),
        rotation_axis_angle_rad=(action_7d[3], action_7d[4], action_7d[5]),
        gripper_opening_01=action_7d[6],
        duration_s=0.0,  # unknown/meaningless for a shape-only smoke-test command
        source_policy=source_policy,
        adapter_name=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        degraded_input=True,
        metadata={
            "semantic_action_valid": False,
            "raw_action_length": len(raw_action),
            "gripper_filled": gripper_filled,
            "gripper_fill_strategy": (f"neutral_fill_{GRIPPER_NEUTRAL_VALUE}" if gripper_filled else "none"),
            "warning": "shape-only mapping, not verified against source checkpoint's real action semantics",
        },
    )
