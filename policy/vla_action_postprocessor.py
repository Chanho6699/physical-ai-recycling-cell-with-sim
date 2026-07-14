"""VLA action postprocessing (v0).

A VLA server's response is never applied to the robot verbatim -- this
module validates the raw action (shape, NaN/inf) and then clips/
normalizes it per configs/real_vla_backend_config.json's
"action_postprocess" section before RealVLAPolicyClient hands it back
as a PolicyOutput. This runs upstream of (and is independent from)
SafetyGate/SafetySupervisor deciding whether the resulting command may
be applied at all -- two separate checks: is this action even
well-formed, and is it currently safe to execute.
"""

import math
from typing import Tuple


def validate_and_postprocess_vla_action(raw_action, config: dict) -> Tuple[list, dict]:
    """Raises RuntimeError (with a readable message, not a bare
    traceback) if raw_action is missing, the wrong length, or contains
    NaN/inf/non-numeric values. Otherwise returns
    (postprocessed_action, debug)."""
    if raw_action is None:
        raise RuntimeError("Real VLA server response is missing 'action' (or it is null).")

    action = list(raw_action)
    if len(action) != 7:
        raise RuntimeError(
            "Real VLA server action must have length 7 ([dx, dy, dz, droll, dpitch, dyaw, gripper]), "
            f"got length {len(action)}: {action}"
        )

    for index, value in enumerate(action):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"Real VLA server action has a non-numeric value at index {index}: {action}")
        if math.isnan(value) or math.isinf(value):
            raise RuntimeError(f"Real VLA server action contains NaN/inf at index {index}: {action}")

    postprocess_config = config.get("action_postprocess", {}) or {}
    max_translation_step = float(postprocess_config.get("max_translation_step", 0.03))
    max_rotation_step = float(postprocess_config.get("max_rotation_step", 0.10))
    gripper_threshold = float(postprocess_config.get("gripper_threshold", 0.5))
    clip_action = bool(postprocess_config.get("clip_action", True))

    postprocessed = [float(value) for value in action]
    translation_clipped = False
    rotation_clipped = False

    if clip_action:
        for index in range(3):
            clipped = max(-max_translation_step, min(max_translation_step, postprocessed[index]))
            if clipped != postprocessed[index]:
                translation_clipped = True
            postprocessed[index] = clipped
        for index in range(3, 6):
            clipped = max(-max_rotation_step, min(max_rotation_step, postprocessed[index]))
            if clipped != postprocessed[index]:
                rotation_clipped = True
            postprocessed[index] = clipped

    raw_gripper = postprocessed[6]
    normalized_gripper = 1.0 if raw_gripper >= gripper_threshold else 0.0
    gripper_normalized = normalized_gripper != raw_gripper
    postprocessed[6] = normalized_gripper

    debug = {
        "raw_action": action,
        "postprocessed_action": postprocessed,
        "translation_clipped": translation_clipped,
        "rotation_clipped": rotation_clipped,
        "gripper_normalized": gripper_normalized,
    }
    return postprocessed, debug
