"""SO-101 PyBullet backend (minimal, v0) -- see this task's chat report
("So101PyBulletBackend 최소 버전 구현") and docs/so101_backend_design_proposal.md
for the interface rationale and the full Panda-vs-SO101 comparison.

Standalone: does not import, get imported by, or modify
robot_sim/pybullet_panda_backend.py, does not touch any V2/V3 pipeline
file, does not load any SmolVLA checkpoint. Reuses the exact URDF path,
joint names, EE link, neutral pose, IK settings, and move force already
validated in benchmark/inspect_so101_urdf.py / smoke_so101_joint_control.py
/ smoke_so101_ik.py (this task does not re-derive any of that).

Scope of this "minimal version": reset -> observe -> command joint
positions / EE-delta (position-only IK) -> gripper open/close, all on a
bare arm with no object/bin/camera in the scene yet (see chat report
item 6 for exactly what's still missing before this could replace
PyBulletPandaBackend in the real recycling-cell loop). No grasp
physics, no is_object_grasped() -- not asked for this task, not added.
"""

import math
from pathlib import Path
from typing import Optional

import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = PROJECT_ROOT / "third_party" / "so101_arm" / "so101_new_calib.urdf"

# Identified BY NAME at load time in reset() -- never a hardcoded
# PyBullet joint index (see this task's requirement 3). This list is
# just which NAMES to look up, not an index assumption.
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
EE_LINK_NAME = "gripper_frame_link"

# Validated in benchmark/smoke_so101_joint_control.py: 0.0 on every arm
# joint converges cleanly and IS this URDF's own "new calibration"
# neutral (zero at the middle of each joint's range) -- not a Panda-style
# hand-picked home pose.
NEUTRAL_ARM_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0]

MOVE_FORCE = 10.0  # matches this URDF's own <limit effort="10" .../> on every actuated joint
DEFAULT_SETTLE_STEPS = 40  # chosen independently for SO-101 (not copied from Panda's DEFAULT_MOVE_STEPS=120) -- same order of magnitude validated in the smoke tests
IK_SOLVER_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-5

# --- Scene: table + single object (see this task's chat report) ---
# Table centered directly under the arm's own neutral EE position
# ([0.391, 0, 0.226], see benchmark/smoke_so101_ik.py's own measurement)
# so the default reach requires mostly a vertical descent, not a large
# lateral move -- deliberate for a first object-approach smoke test, not
# a workspace-coverage claim. x/y half-extent (0.15m) keeps the table
# well clear of the robot base at the origin (table spans x in
# [0.241, 0.541], base link's own mesh sits within ~0.1m of the origin).
TABLE_POSITION = [0.391, 0.0, 0.025]
TABLE_HALF_EXTENTS = [0.15, 0.15, 0.025]
TABLE_TOP_Z = TABLE_POSITION[2] + TABLE_HALF_EXTENTS[2]  # 0.05

OBJECT_HALF_EXTENTS = [0.02, 0.02, 0.02]
OBJECT_MASS = 0.05
# Sits exactly on the table top, directly under neutral EE -- a
# DEFAULT, not a hardcoded requirement: reset(object_position=...)
# overrides this, keeping randomization a caller-side decision (see
# this task's "구조를 분리" requirement) rather than baked into reset().
DEFAULT_OBJECT_POSITION = [TABLE_POSITION[0], TABLE_POSITION[1], TABLE_TOP_Z + OBJECT_HALF_EXTENTS[2]]
OBJECT_SETTLE_STEPS = 60  # lets the object settle onto the table (tiny initial-contact wobble) before it's treated as "initial pose"

# Safety floor for command_end_effector_delta()'s target z -- independent
# of any specific approach/pre-grasp offset the caller uses, so this
# holds regardless of who's driving the arm.
MIN_EE_HEIGHT_M = TABLE_TOP_Z + 0.01


def gripper_normalized_to_radians(value: float, lower: float, upper: float) -> float:
    """0.0=closed, 1.0=open (this task's external convention) -> this
    URDF's native revolute-joint radians. Kept as its own free function,
    independent of the class, specifically so it can be swapped for a
    real-hardware 0-100 command mapping later (see the vendored SO-101
    README's note that this URDF/MJCF pair does not yet reflect that
    hardware convention) without touching backend control-flow code."""
    value = max(0.0, min(1.0, value))
    return lower + value * (upper - lower)


def gripper_radians_to_normalized(radians: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return max(0.0, min(1.0, (radians - lower) / (upper - lower)))


class So101PyBulletBackend:
    def __init__(self, gui: bool = False, urdf_path=None, time_step: float = 1.0 / 240.0, object_position: Optional[list] = None):
        self.gui = gui
        self.urdf_path = Path(urdf_path) if urdf_path else DEFAULT_URDF_PATH
        self.time_step = time_step
        # None (default) -> DEFAULT_OBJECT_POSITION is used in reset();
        # a caller passing an explicit list here is the separation point
        # for future object-position randomization (see this task's
        # chat report) -- reset() itself never rolls its own randomness.
        self._object_position_override = list(object_position) if object_position is not None else None

        self.client_id = None
        self.robot_id = None
        self.joint_info_by_name = {}
        self.arm_joint_indices = []
        self.gripper_joint_index = None
        self.gripper_lower = None
        self.gripper_upper = None
        self.ee_link_index = None
        self.table_id = None
        self.object_id = None
        self._object_initial_pose = None

    def reset(self) -> dict:
        if self.client_id is not None:
            p.disconnect(self.client_id)

        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        if not self.urdf_path.exists():
            raise FileNotFoundError(f"SO-101 URDF not found at {self.urdf_path} -- see third_party/so101_arm/SOURCE.md")
        self.robot_id = p.loadURDF(str(self.urdf_path), basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)

        self.joint_info_by_name = {}
        self.ee_link_index = None
        for joint_index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.client_id)
            name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self.joint_info_by_name[name] = {"index": joint_index, "lower": info[8], "upper": info[9]}
            if link_name == EE_LINK_NAME:
                self.ee_link_index = joint_index

        missing = [n for n in ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME] if n not in self.joint_info_by_name]
        if missing:
            raise RuntimeError(f"Expected joint name(s) not found in URDF {self.urdf_path}: {missing}")
        if self.ee_link_index is None:
            raise RuntimeError(f"EE link '{EE_LINK_NAME}' not found in URDF {self.urdf_path}")

        self.arm_joint_indices = [self.joint_info_by_name[n]["index"] for n in ARM_JOINT_NAMES]
        gripper_info = self.joint_info_by_name[GRIPPER_JOINT_NAME]
        self.gripper_joint_index = gripper_info["index"]
        self.gripper_lower = gripper_info["lower"]
        self.gripper_upper = gripper_info["upper"]

        # Neutral pose, then gripper fully open -- an explicit, well-
        # defined starting state (mirrors PyBulletPandaBackend.reset()'s
        # own "gripper starts open" convention, chosen independently here
        # for the same reason: an ambiguous mid-range resting angle would
        # make every downstream observation harder to interpret).
        self.command_joint_positions(NEUTRAL_ARM_POSITIONS, settle_steps=DEFAULT_SETTLE_STEPS * 3)
        self.set_gripper(1.0, settle_steps=DEFAULT_SETTLE_STEPS)

        # --- Scene: table (fixed) + single object (see this task's chat
        # report) -- built AFTER the arm is at its known neutral pose, so
        # there is no risk of the object being spawned inside/overlapping
        # a not-yet-settled arm pose. ---
        self.table_id = self._create_box(TABLE_HALF_EXTENTS, TABLE_POSITION, color=[0.55, 0.35, 0.2, 1.0], mass=0.0)
        object_position = self._object_position_override if self._object_position_override is not None else list(DEFAULT_OBJECT_POSITION)
        self.object_id = self._create_box(OBJECT_HALF_EXTENTS, object_position, color=[0.2, 0.6, 1.0, 1.0], mass=OBJECT_MASS)
        self.step(OBJECT_SETTLE_STEPS)  # let any initial-contact wobble settle before recording "initial pose"
        self._object_initial_pose = self.get_object_pose()

        return self.get_observation()

    def _create_box(self, half_extents: list, position: list, color: list, mass: float = 0.0) -> int:
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client_id)
        return p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
            basePosition=position, physicsClientId=self.client_id,
        )

    def _get_ee_pose(self):
        state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True, physicsClientId=self.client_id)
        return list(state[4]), list(state[5])

    def get_joint_positions(self) -> list:
        states = p.getJointStates(self.robot_id, self.arm_joint_indices, physicsClientId=self.client_id)
        return [s[0] for s in states]

    def get_end_effector_pose(self) -> tuple:
        return self._get_ee_pose()

    def command_joint_positions(self, positions: list, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        if len(positions) != len(self.arm_joint_indices):
            raise ValueError(f"Expected {len(self.arm_joint_indices)} arm joint positions, got {len(positions)}")

        clipped = []
        for name, raw_position in zip(ARM_JOINT_NAMES, positions):
            if not math.isfinite(raw_position):
                raise ValueError(f"Non-finite joint position commanded for '{name}': {raw_position}")
            info = self.joint_info_by_name[name]
            clipped.append(max(info["lower"], min(info["upper"], raw_position)))

        p.setJointMotorControlArray(
            self.robot_id, self.arm_joint_indices, p.POSITION_CONTROL, targetPositions=clipped,
            forces=[MOVE_FORCE] * len(self.arm_joint_indices), physicsClientId=self.client_id,
        )
        self.step(settle_steps)
        return self.get_observation()

    def command_end_effector_delta(self, delta_position: list, delta_orientation=None, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        """Position-only IK (see docs/so101_backend_design_proposal.md --
        delta_orientation is accepted for interface symmetry with a
        future full-6DOF version, but is currently ignored, not silently
        misapplied)."""
        current_ee_position, _current_ee_orientation = self._get_ee_pose()
        target_position = [current_ee_position[i] + delta_position[i] for i in range(3)]
        if not all(math.isfinite(v) for v in target_position):
            raise ValueError(f"Non-finite EE target computed from delta {delta_position}: {target_position}")

        # Safety floor: never command the EE below the table top + margin,
        # regardless of which delta produced this target (see this task's
        # "테이블 아래로 내려가지 않도록 최소 높이 제한" requirement) --
        # holds for every caller, not just the approach-smoke-test's own
        # step loop.
        target_position[2] = max(target_position[2], MIN_EE_HEIGHT_M)

        joint_poses = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index, target_position,
            maxNumIterations=IK_SOLVER_ITERATIONS, residualThreshold=IK_RESIDUAL_THRESHOLD,
            physicsClientId=self.client_id,
        )
        arm_targets = list(joint_poses[: len(self.arm_joint_indices)])

        # command_joint_positions() clips to each joint's own limit and
        # checks finiteness -- satisfies "IK 결과를 arm joint limit 안으로
        # 제한한다" without duplicating that logic here.
        observation = self.command_joint_positions(arm_targets, settle_steps=settle_steps)

        final_ee_position, _ = self._get_ee_pose()
        position_error = math.sqrt(sum((final_ee_position[i] - target_position[i]) ** 2 for i in range(3)))
        observation["ee_delta_target_position"] = target_position
        observation["ee_delta_position_error"] = position_error
        return observation

    def set_gripper(self, normalized_value: float, settle_steps: int = DEFAULT_SETTLE_STEPS) -> dict:
        if not math.isfinite(normalized_value):
            raise ValueError(f"Non-finite gripper command: {normalized_value}")
        radians = gripper_normalized_to_radians(normalized_value, self.gripper_lower, self.gripper_upper)
        p.setJointMotorControlArray(
            self.robot_id, [self.gripper_joint_index], p.POSITION_CONTROL, targetPositions=[radians],
            forces=[MOVE_FORCE], physicsClientId=self.client_id,
        )
        self.step(settle_steps)
        return self.get_observation()

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_observation(self) -> dict:
        arm_positions = self.get_joint_positions()
        ee_position, ee_orientation = self._get_ee_pose()
        gripper_state = p.getJointState(self.robot_id, self.gripper_joint_index, physicsClientId=self.client_id)
        gripper_radians = gripper_state[0]
        gripper_normalized = gripper_radians_to_normalized(gripper_radians, self.gripper_lower, self.gripper_upper)

        return {
            "simulator": "pybullet_so101",
            "joint_positions": arm_positions,
            "end_effector_position": ee_position,
            "end_effector_orientation": ee_orientation,
            "gripper_position_normalized": gripper_normalized,
            "gripper_position_radians": gripper_radians,
        }

    def get_object_position(self) -> list:
        position, _orientation = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return list(position)

    def get_object_pose(self) -> tuple:
        position, orientation = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return list(position), list(orientation)

    def get_scene_state(self) -> dict:
        """Superset of get_observation() -- adds object pose and static
        table metadata, all finite-checkable by the caller the same way
        get_observation()'s own fields already are."""
        observation = self.get_observation()
        object_position, object_orientation = self.get_object_pose()
        return {
            **observation,
            "object_position": object_position,
            "object_orientation": object_orientation,
            "table_position": list(TABLE_POSITION),
            "table_half_extents": list(TABLE_HALF_EXTENTS),
            "table_top_z": TABLE_TOP_Z,
        }

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None
