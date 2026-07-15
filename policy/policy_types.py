from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PolicyInput:
    image: Any
    instruction: str
    robot_state: dict
    task_goal: dict
    target_object_position: Optional[list] = None
    bin_position: Optional[list] = None
    step_index: int = 0
    phase: Optional[str] = None
    # Where `image` came from this step ("wrist", "external", None) and,
    # when it's a camera frame that's already been through some vision
    # processing (e.g. PyBulletWristCamera segmentation/depth), a small
    # dict of what was found there (object_visible, estimated_world_position,
    # ...). Both optional and unused by DummyOpenVLAPolicy's action logic
    # today -- they exist so a real VLA/visual policy can be dropped in
    # later without changing PolicyInput's shape again.
    observation_source: Optional[str] = None
    visual_observation: Optional[dict] = None
    # Multi-camera observation, keyed by role (e.g. "main", "wrist") --
    # np.ndarray HWC uint8 per role, e.g. from
    # PyBulletPandaBackend.render_main_camera()/render_wrist_camera().
    # None (the default) means "legacy single-image path" -- `image`
    # above is still what gets sent in that case. Added for
    # HuggingFaceVLA/smolvla_libero's two-camera requirement (see
    # policy_semantics/manifest.py's _SMOLVLA_LIBERO_MANIFEST) without
    # changing what single-camera callers (mock-action, smolvla_base)
    # already do.
    images_by_role: Optional[dict] = None


@dataclass
class PolicyOutput:
    action: list
    phase: str
    done: bool = False
    info: Optional[dict] = None
