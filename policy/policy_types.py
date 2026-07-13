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


@dataclass
class PolicyOutput:
    action: list
    phase: str
    done: bool = False
    info: Optional[dict] = None
