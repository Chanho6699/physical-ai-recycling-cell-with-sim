"""PyBullet Franka Panda URDF backend (v1).

Unlike robot_sim/pybullet_backend.py (a plain end-effector sphere with
distance-based teleport), this backend loads PyBullet's bundled Franka
Panda URDF and drives it with real inverse kinematics + joint motor
control. Grasping is still distance-based (no contact-force grasp
physics yet): closing the gripper near the object attaches it to the
end-effector link with a fixed constraint, so it moves with the arm
without fighting friction/contact instability.

Panda URDF layout (from pybullet_data/franka_panda/panda.urdf, confirmed
via print_joint_info()):
  joint_index=0..6  panda_joint1..7      (revolute)   -- arm
  joint_index=7     panda_joint8          (fixed)
  joint_index=8     panda_hand_joint      (fixed)      -- hand link
  joint_index=9,10  panda_finger_joint1/2 (prismatic)  -- gripper fingers, range [0, 0.04] each
  joint_index=11    panda_grasptarget_hand (fixed)     -- virtual point between the fingertips

end_effector_link_index = 11 (panda_grasptarget) is used for IK, since it
sits at the actual grasp point between the fingers rather than at the
hand's own frame (link 8).
"""

import math

import pybullet as p
import pybullet_data

from action_adapter.adapter_v0 import RobotCommand
from robot_sim.backend_interface import SimulatorBackend

GRASP_THRESHOLD = 0.05
PLACE_THRESHOLD = 0.08

ARM_JOINT_FORCES = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
FINGER_FORCE = 20.0
FINGER_OPEN_POSITION = 0.04
FINGER_CLOSE_POSITION = 0.0

# Franka's common "ready" joint configuration (arm joints 1-7, radians).
READY_JOINT_POSITIONS = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

DEFAULT_MOVE_STEPS = 120
DEFAULT_GRIPPER_STEPS = 60


def _evaluate_safety_result(result):
    """Accepts either a SafetyGateResult or a SafetyDecision (duck-typed,
    so this module doesn't need to import the safety package) and returns
    (should_interrupt, reason)."""
    if result is None:
        return False, None
    if hasattr(result, "allowed"):
        return (not result.allowed), getattr(result, "reason", None)
    if hasattr(result, "emergency_stop"):
        return result.emergency_stop, getattr(result, "reason", None)
    return False, None


class PyBulletPandaBackend(SimulatorBackend):
    def __init__(self, gui: bool = True, time_step: float = 1.0 / 240.0):
        self.gui = gui
        self.time_step = time_step

        self.client_id = None
        self.robot_id = None

        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.finger_joint_indices = [9, 10]
        self.end_effector_link_index = 11  # panda_grasptarget

        self.default_orientation = None

        self._table_id = None
        self._object_id = None
        self._bin_id = None

        self._table_position = [0.35, 0.15, 0.015]
        self._object_position = [0.45, 0.0, 0.05]
        self._bin_position = [0.3, 0.35, 0.05]
        self._object_type = "unknown"

        self._gripper_state = "open"
        self._held_object = False
        self.grasp_constraint_id = None
        self._task_status = "running"
        self._last_event = "none"
        self.last_safety_reason = None
        self.last_blocked_action = None

    def reset(self) -> dict:
        if self.client_id is not None:
            p.disconnect(self.client_id)

        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        print(
            f"[PyBulletPandaBackend.reset] gui={self.gui}, connection_mode={connection_mode}, "
            f"client_id={self.client_id}, isConnected={p.isConnected(self.client_id)}"
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        self._table_id = self._create_box(
            half_extents=[0.35, 0.45, 0.015],
            position=self._table_position,
            color=[0.55, 0.35, 0.2, 1.0],
        )

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client_id,
        )

        for joint_index, angle in zip(self.arm_joint_indices, READY_JOINT_POSITIONS):
            p.resetJointState(self.robot_id, joint_index, angle, physicsClientId=self.client_id)
        for joint_index in self.finger_joint_indices:
            p.resetJointState(self.robot_id, joint_index, FINGER_OPEN_POSITION, physicsClientId=self.client_id)

        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client_id)

        _, self.default_orientation = self._get_ee_pose()

        self._object_id = self._create_box(
            half_extents=[0.02, 0.02, 0.02],
            position=self._object_position,
            color=[0.2, 0.6, 1.0, 1.0],
            mass=0.05,
        )
        self._object_type = "unknown"

        self._bin_id = self._create_box(
            half_extents=[0.06, 0.06, 0.03],
            position=self._bin_position,
            color=[0.2, 0.8, 0.2, 1.0],
        )

        self._gripper_state = "open"
        self._held_object = False
        self.grasp_constraint_id = None
        self._task_status = "running"
        self._last_event = "none"
        self.last_safety_reason = None
        self.last_blocked_action = None

        return self.get_state()

    def apply_command(self, command: RobotCommand) -> dict:
        ee_position, ee_orientation = self._get_ee_pose()

        target_position = [
            ee_position[0] + command.target_dx,
            ee_position[1] + command.target_dy,
            ee_position[2] + command.target_dz,
        ]
        # v1: orientation deltas (target_droll/dpitch/dyaw) are ignored --
        # keep the current end-effector orientation as-is.
        self.move_end_effector_to(target_position, ee_orientation)

        if command.gripper_command == "open":
            self.open_gripper()
        elif command.gripper_command == "close":
            self.close_gripper()

        return self.get_state()

    def move_end_effector_to(
        self,
        target_position: list,
        target_orientation: list = None,
        steps: int = DEFAULT_MOVE_STEPS,
        safety_callback=None,
        action_name: str = "move_end_effector_to",
        safety_check_interval: int = 10,
        trajectory_callback=None,
        trajectory_record_interval: int = 10,
    ) -> dict:
        if target_orientation is None:
            target_orientation = self.default_orientation

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link_index,
            target_position,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.client_id,
        )
        arm_target_positions = joint_poses[: len(self.arm_joint_indices)]

        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=arm_target_positions,
            forces=ARM_JOINT_FORCES,
            physicsClientId=self.client_id,
        )

        for step_index in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

            if (
                trajectory_callback is not None
                and trajectory_record_interval > 0
                and step_index % trajectory_record_interval == 0
            ):
                trajectory_callback(
                    action_name=action_name,
                    step_index=step_index,
                    robot_state=self.get_state(),
                )

            if (
                safety_callback is not None
                and safety_check_interval > 0
                and (step_index + 1) % safety_check_interval == 0
            ):
                result = safety_callback(action_name)
                should_interrupt, reason = _evaluate_safety_result(result)
                if should_interrupt:
                    self._interrupt_motion(action_name, reason)
                    break

        return self.get_state()

    def _interrupt_motion(self, action_name: str, reason) -> None:
        current_joint_states = p.getJointStates(
            self.robot_id, self.arm_joint_indices, physicsClientId=self.client_id
        )
        current_positions = [s[0] for s in current_joint_states]

        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=current_positions,
            forces=ARM_JOINT_FORCES,
            physicsClientId=self.client_id,
        )

        self._task_status = "interrupted_by_safety"
        self._last_event = f"safety_interrupted:{action_name}"
        self.last_safety_reason = reason
        self.last_blocked_action = action_name

    def open_gripper(self, steps: int = DEFAULT_GRIPPER_STEPS) -> dict:
        p.setJointMotorControlArray(
            self.robot_id,
            self.finger_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[FINGER_OPEN_POSITION, FINGER_OPEN_POSITION],
            forces=[FINGER_FORCE, FINGER_FORCE],
            physicsClientId=self.client_id,
        )
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

        self._gripper_state = "open"

        if self._held_object:
            object_position, _ = p.getBasePositionAndOrientation(self._object_id, physicsClientId=self.client_id)
            distance_to_bin = self._distance(object_position, self._bin_position)

            if self.grasp_constraint_id is not None:
                p.removeConstraint(self.grasp_constraint_id, physicsClientId=self.client_id)
                self.grasp_constraint_id = None
            self._held_object = False

            if distance_to_bin <= PLACE_THRESHOLD:
                p.resetBasePositionAndOrientation(
                    self._object_id, self._bin_position, [0, 0, 0, 1], physicsClientId=self.client_id
                )
                self._task_status = "success"
                self._last_event = "object_placed_in_bin"
            else:
                self._task_status = "released"
                self._last_event = "object_released"

        return self.get_state()

    def close_gripper(self, steps: int = DEFAULT_GRIPPER_STEPS) -> dict:
        p.setJointMotorControlArray(
            self.robot_id,
            self.finger_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[FINGER_CLOSE_POSITION, FINGER_CLOSE_POSITION],
            forces=[FINGER_FORCE, FINGER_FORCE],
            physicsClientId=self.client_id,
        )
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

        self._gripper_state = "close"

        if not self._held_object:
            ee_position, ee_orientation = self._get_ee_pose()
            object_position, object_orientation = p.getBasePositionAndOrientation(
                self._object_id, physicsClientId=self.client_id
            )

            if self._distance(ee_position, object_position) < GRASP_THRESHOLD:
                # Attach at the object's *current* offset from the end
                # effector (rather than [0, 0, 0]) so grasping doesn't
                # snap the object to the link origin.
                ee_pos_inv, ee_orn_inv = p.invertTransform(ee_position, ee_orientation)
                frame_pos, frame_orn = p.multiplyTransforms(
                    ee_pos_inv, ee_orn_inv, object_position, object_orientation
                )

                self.grasp_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=self.end_effector_link_index,
                    childBodyUniqueId=self._object_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=frame_pos,
                    parentFrameOrientation=frame_orn,
                    childFramePosition=[0, 0, 0],
                    childFrameOrientation=[0, 0, 0, 1],
                    physicsClientId=self.client_id,
                )
                self._held_object = True
                self._task_status = "grasped"
                self._last_event = "object_grasped"

        return self.get_state()

    def set_object_position(self, position: list) -> dict:
        self._object_position = list(position)
        p.resetBasePositionAndOrientation(
            self._object_id, self._object_position, [0, 0, 0, 1], physicsClientId=self.client_id
        )
        return self.get_state()

    def set_object_type(self, object_type: str) -> None:
        self._object_type = object_type

    def set_bin_position(self, position: list) -> dict:
        self._bin_position = list(position)
        p.resetBasePositionAndOrientation(
            self._bin_id, self._bin_position, [0, 0, 0, 1], physicsClientId=self.client_id
        )
        return self.get_state()

    def get_state(self) -> dict:
        joint_states = p.getJointStates(
            self.robot_id,
            self.arm_joint_indices + self.finger_joint_indices,
            physicsClientId=self.client_id,
        )
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]

        ee_position, ee_orientation = self._get_ee_pose()

        finger_states = p.getJointStates(self.robot_id, self.finger_joint_indices, physicsClientId=self.client_id)
        gripper_width = finger_states[0][0] + finger_states[1][0]

        object_position, _ = p.getBasePositionAndOrientation(self._object_id, physicsClientId=self.client_id)

        return {
            "simulator": "pybullet_panda",
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_position": ee_position,
            "end_effector_orientation": ee_orientation,
            "gripper_width": gripper_width,
            "gripper_state": self._gripper_state,
            "object_position": list(object_position),
            "object_type": self._object_type,
            "bin_position": list(self._bin_position),
            "held_object": self._held_object,
            "task_status": self._task_status,
            "last_event": self._last_event,
            "safety_reason": self.last_safety_reason,
            "blocked_action": self.last_blocked_action,
        }

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def print_joint_info(self) -> None:
        if self.robot_id is None:
            print("Robot not loaded yet. Call reset() first.")
            return

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        print(f"num_joints={num_joints}")
        for joint_index in range(num_joints):
            info = p.getJointInfo(self.robot_id, joint_index, physicsClientId=self.client_id)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            link_name = info[12].decode("utf-8")
            print(
                f"joint_index={joint_index}, joint_name={joint_name}, "
                f"joint_type={joint_type}, link_name={link_name}"
            )
        print(f"arm_joint_indices={self.arm_joint_indices}")
        print(f"finger_joint_indices={self.finger_joint_indices}")
        print(f"end_effector_link_index={self.end_effector_link_index} (panda_grasptarget)")

    def _get_ee_pose(self):
        state = p.getLinkState(
            self.robot_id, self.end_effector_link_index, computeForwardKinematics=True, physicsClientId=self.client_id
        )
        return list(state[4]), list(state[5])

    @staticmethod
    def _distance(a, b) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def _create_box(self, half_extents, position, color, mass: float = 0.0) -> int:
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client_id
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client_id,
        )


if __name__ == "__main__":
    backend = PyBulletPandaBackend(gui=False)
    print("Reset state:", backend.reset())
    backend.print_joint_info()
    backend.close()
