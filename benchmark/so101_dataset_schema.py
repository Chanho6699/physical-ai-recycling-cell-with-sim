"""SO-101 LeRobot dataset schema (see this task's chat report, "SO-101
Dataset Recorder"). Entirely separate from Panda's own FEATURES
(benchmark/collect_recycling_dataset.py) -- this file does not import or
modify that one, and vice versa.

Follows LeRobot's OWN native SO-101 convention as closely as possible --
verified directly against the installed lerobot==0.6.0 package, not
guessed:
  - lerobot/robots/so_follower/so_follower.py: motor dict declaration
    order is shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper (6 motors) -- SO101_JOINT_NAMES below is built
    FROM robot_sim.so101_pybullet_backend's own ARM_JOINT_NAMES +
    GRIPPER_JOINT_NAME constants (not retyped), so state/action joint
    order can never silently diverge from the backend's own.
  - lerobot/utils/feature_utils.py::hw_to_dataset_features(): packs a
    robot's per-joint dict into a SINGLE packed tensor feature
    (`observation.state` / `action`, shape=(num_joints,),
    names=list(joint_keys)) -- exactly the single-vector-per-modality
    shape this schema uses, not a per-joint dict of scalars.
  - lerobot/robots/so_follower/so_follower.py's gripper motor uses
    MotorNormMode.RANGE_0_100 (0=closed, 100=open) -- this schema's
    gripper channel uses that same 0-100 scale (see
    gripper_unit_01_to_lerobot_0_100() below), NOT the backend's own
    internal 0.0-1.0 normalized convention.

Arm joint channels (shoulder_pan..wrist_roll) are stored in RADIANS --
robot_sim/so101_pybullet_backend.py's own native, already-validated unit
(URDF joint limits, IK output) -- no degree conversion is applied here,
since this project has no independently-verified radians<->degrees
mapping beyond a plain multiply, and the task's own instruction only
explicitly calls out the GRIPPER channel's unit as needing a documented
conversion.
"""

import math

import numpy as np

from robot_sim.so101_pybullet_backend import (
    ARM_JOINT_NAMES,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_WIDTH,
    GRIPPER_JOINT_NAME,
)

SO101_JOINT_NAMES = list(ARM_JOINT_NAMES) + [GRIPPER_JOINT_NAME]
SO101_STATE_DIM = len(SO101_JOINT_NAMES)
SO101_ACTION_DIM = len(SO101_JOINT_NAMES)
SO101_ROBOT_TYPE = "so101_pybullet"

SO101_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (SO101_STATE_DIM,), "names": list(SO101_JOINT_NAMES)},
    "action": {"dtype": "float32", "shape": (SO101_ACTION_DIM,), "names": list(SO101_JOINT_NAMES)},
    "observation.images.front": {
        "dtype": "image", "shape": (FRONT_CAMERA_HEIGHT, FRONT_CAMERA_WIDTH, 3),
        "names": ["height", "width", "channel"],
    },
    # Analysis-only bookkeeping (see benchmark/so101_scripted_expert.py's
    # own PHASE_* constants) -- NEVER an SmolVLA observation/action input.
    # int64 to match this project's own existing scalar-feature
    # convention (e.g. info.json's own "episode_index"/"frame_index"
    # fields, both int64 shape (1,)). phase_id -> phase_name mapping is
    # written as a sidecar JSON (meta/phase_id_mapping.json) rather than
    # forced into info.json's own auto-managed schema -- see
    # collect_so101_episode.py's own write_phase_id_mapping().
    "phase_id": {"dtype": "int64", "shape": (1,), "names": None},
}


def gripper_unit_01_to_lerobot_0_100(value_01: float) -> float:
    """robot_sim/so101_pybullet_backend.py's own gripper_normalized_to_radians()/
    gripper_radians_to_normalized() use a 0.0(closed)-1.0(open) convention.
    LeRobot's real so_follower hardware (lerobot/robots/so_follower/so_follower.py,
    MotorNormMode.RANGE_0_100) uses 0(closed)-100(open). Both already
    agree on polarity (0=closed, max=open) -- this is a pure linear
    rescale, not a sign flip or offset."""
    return value_01 * 100.0


def _validate_vector(vector: list, expected_dim: int, label: str) -> np.ndarray:
    if len(vector) != expected_dim:
        raise ValueError(f"{label} must have exactly {expected_dim} elements, got {len(vector)}: {vector}")
    array = np.array(vector, dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ValueError(f"{label} array shape mismatch: expected ({expected_dim},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite value(s): {vector}")
    return array


def pack_state(joint_positions_rad: list, gripper_normalized_01: float) -> np.ndarray:
    """[shoulder_pan..wrist_roll in radians] + [gripper in LeRobot's 0-100
    scale] -- SO101_STATE_DIM (6) elements, float32, finite-checked."""
    if len(joint_positions_rad) != len(ARM_JOINT_NAMES):
        raise ValueError(f"Expected {len(ARM_JOINT_NAMES)} arm joint positions, got {len(joint_positions_rad)}")
    if not math.isfinite(gripper_normalized_01):
        raise ValueError(f"Non-finite gripper_normalized_01: {gripper_normalized_01}")
    vector = list(joint_positions_rad) + [gripper_unit_01_to_lerobot_0_100(gripper_normalized_01)]
    return _validate_vector(vector, SO101_STATE_DIM, "observation.state")


def pack_action(arm_joint_targets_rad: list, gripper_target_normalized_01: float) -> np.ndarray:
    """Same shape/unit convention as pack_state() -- see this module's
    own docstring for why the action is the ABSOLUTE joint target about
    to be applied, not a delta."""
    if len(arm_joint_targets_rad) != len(ARM_JOINT_NAMES):
        raise ValueError(f"Expected {len(ARM_JOINT_NAMES)} arm joint targets, got {len(arm_joint_targets_rad)}")
    if not math.isfinite(gripper_target_normalized_01):
        raise ValueError(f"Non-finite gripper_target_normalized_01: {gripper_target_normalized_01}")
    vector = list(arm_joint_targets_rad) + [gripper_unit_01_to_lerobot_0_100(gripper_target_normalized_01)]
    return _validate_vector(vector, SO101_ACTION_DIM, "action")


def pack_phase_id(phase_id: int) -> np.ndarray:
    """Analysis-only scalar (see SO101_FEATURES's own "phase_id" entry) --
    shape (1,), int64, finite-checked the same way pack_state()/
    pack_action() are."""
    if not math.isfinite(phase_id):
        raise ValueError(f"Non-finite phase_id: {phase_id}")
    return np.array([int(phase_id)], dtype=np.int64)


def validate_image(image: np.ndarray, label: str = "observation.images.front") -> None:
    expected_shape = SO101_FEATURES["observation.images.front"]["shape"]
    if image.shape != expected_shape:
        raise ValueError(f"{label} shape mismatch: expected {expected_shape}, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{label} dtype mismatch: expected uint8, got {image.dtype}")


if __name__ == "__main__":
    print(f"SO101_JOINT_NAMES: {SO101_JOINT_NAMES}")
    print(f"SO101_STATE_DIM: {SO101_STATE_DIM}, SO101_ACTION_DIM: {SO101_ACTION_DIM}")
    print(f"SO101_ROBOT_TYPE: {SO101_ROBOT_TYPE}")
    print(f"SO101_FEATURES: {SO101_FEATURES}")
