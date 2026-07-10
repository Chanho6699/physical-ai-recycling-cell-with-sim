"""PyBullet-based lightweight simulation backend (v0).

The end-effector is a plain sphere directly teleported to the commanded
position (no arm URDF, no IK, no real grasp physics). The table,
recyclable object, and bin are static props. Pick/place is a distance-based
rule: closing the gripper within GRASP_THRESHOLD of the object "grasps" it
(it then follows the end-effector), and opening the gripper within
PLACE_THRESHOLD of the bin "places" it; opening elsewhere drops it. This is
enough to validate simple task state transitions end to end on local
hardware while Isaac Sim remains a future backend.
"""

import math

import pybullet as p
import pybullet_data

from action_adapter.adapter_v0 import RobotCommand
from robot_sim.backend_interface import SimulatorBackend

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.05, 1.5

SIMULATION_STEPS_PER_COMMAND = 10

GRASP_THRESHOLD = 0.1
PLACE_THRESHOLD = 0.15


class PyBulletBackend(SimulatorBackend):
    def __init__(self, gui: bool = True, time_step: float = 1.0 / 240.0):
        self.gui = gui
        self.time_step = time_step

        self._client_id = None
        self._table_id = None
        self._object_id = None
        self._bin_id = None
        self._ee_id = None

        self._table_position = [0.5, 0.0, 0.25]
        self._object_position = [0.5, 0.0, 0.53]
        self._bin_position = [0.0, 0.6, 0.1]

        self._ee_position = [0.0, 0.0, 0.5]
        self._ee_orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self._gripper_state = "open"
        self._object_type = "unknown"

    @property
    def client_id(self):
        return self._client_id

    def reset(self) -> dict:
        if self._client_id is not None:
            p.disconnect(self._client_id)

        connection_mode = p.GUI if self.gui else p.DIRECT
        self._client_id = p.connect(connection_mode)

        print(
            f"[PyBulletBackend.reset] gui={self.gui}, connection_mode={connection_mode}, "
            f"client_id={self._client_id}, isConnected={p.isConnected(self._client_id)}"
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client_id)
        p.setTimeStep(self.time_step, physicsClientId=self._client_id)
        p.loadURDF("plane.urdf", physicsClientId=self._client_id)

        self._table_position = [0.5, 0.0, 0.25]
        self._object_position = [0.5, 0.0, 0.53]
        self._bin_position = [0.0, 0.6, 0.1]

        self._table_id = self._create_box(
            half_extents=[0.3, 0.3, 0.25],
            position=self._table_position,
            color=[0.55, 0.35, 0.2, 1.0],
        )
        self._object_id = self._create_box(
            half_extents=[0.03, 0.03, 0.03],
            position=self._object_position,
            color=[0.2, 0.6, 1.0, 1.0],
        )
        self._bin_id = self._create_box(
            half_extents=[0.15, 0.15, 0.1],
            position=self._bin_position,
            color=[0.2, 0.8, 0.2, 1.0],
        )

        self._ee_position = [0.0, 0.0, 0.5]
        self._ee_orientation = [0.0, 0.0, 0.0]
        self._gripper_state = "open"
        self._ee_id = self._create_sphere(
            radius=0.03,
            position=self._ee_position,
            color=[1.0, 0.0, 0.0, 1.0],
        )

        self._held_object_id = None
        self._task_status = "running"
        self._last_event = "none"
        self._object_type = "unknown"

        for _ in range(SIMULATION_STEPS_PER_COMMAND):
            p.stepSimulation(physicsClientId=self._client_id)

        return self.get_state()

    def apply_command(self, robot_command: RobotCommand) -> dict:
        x = self._clamp(self._ee_position[0] + robot_command.target_dx, X_MIN, X_MAX)
        y = self._clamp(self._ee_position[1] + robot_command.target_dy, Y_MIN, Y_MAX)
        z = self._clamp(self._ee_position[2] + robot_command.target_dz, Z_MIN, Z_MAX)
        self._ee_position = [x, y, z]

        self._ee_orientation[0] += robot_command.target_droll
        self._ee_orientation[1] += robot_command.target_dpitch
        self._ee_orientation[2] += robot_command.target_dyaw

        orientation_quaternion = p.getQuaternionFromEuler(self._ee_orientation)
        p.resetBasePositionAndOrientation(
            self._ee_id,
            self._ee_position,
            orientation_quaternion,
            physicsClientId=self._client_id,
        )

        self._gripper_state = robot_command.gripper_command

        # Grasp: closing the gripper near the object picks it up.
        if self._gripper_state == "close" and self._held_object_id is None:
            if self._distance(self._ee_position, self._object_position) <= GRASP_THRESHOLD:
                self._held_object_id = self._object_id
                self._task_status = "grasped"
                self._last_event = "object_grasped"

        # While held and still closed, the object follows the end-effector.
        if self._held_object_id is not None and self._gripper_state == "close":
            self._object_position = [
                self._ee_position[0],
                self._ee_position[1],
                self._ee_position[2] - 0.05,
            ]
            p.resetBasePositionAndOrientation(
                self._object_id,
                self._object_position,
                [0, 0, 0, 1],
                physicsClientId=self._client_id,
            )

        # Opening the gripper while holding the object either places it (near
        # the bin) or drops it (anywhere else).
        if self._gripper_state == "open" and self._held_object_id is not None:
            distance_object_to_bin = self._distance(self._object_position, self._bin_position)
            distance_ee_to_bin = self._distance(self._ee_position, self._bin_position)

            if min(distance_object_to_bin, distance_ee_to_bin) <= PLACE_THRESHOLD:
                self._object_position = list(self._bin_position)
                p.resetBasePositionAndOrientation(
                    self._object_id,
                    self._object_position,
                    [0, 0, 0, 1],
                    physicsClientId=self._client_id,
                )
                self._held_object_id = None
                self._task_status = "success"
                self._last_event = "object_placed_in_bin"
            else:
                self._held_object_id = None
                self._task_status = "failed"
                self._last_event = "object_dropped"

        for _ in range(SIMULATION_STEPS_PER_COMMAND):
            p.stepSimulation(physicsClientId=self._client_id)

        return self.get_state()

    def set_object_position(self, position: list) -> dict:
        # v0: assumes the object is not currently held. If it were, this
        # would fight with the grasp-follow logic in apply_command(); not
        # handled here since the Real2Sim demo only calls this right after
        # reset(), before any grasp happens.
        self._object_position = list(position)
        p.resetBasePositionAndOrientation(
            self._object_id,
            self._object_position,
            [0, 0, 0, 1],
            physicsClientId=self._client_id,
        )
        return self.get_state()

    def set_object_type(self, object_type: str) -> None:
        self._object_type = object_type

    def get_state(self) -> dict:
        return {
            "simulator": "pybullet",
            "end_effector_position": list(self._ee_position),
            "gripper_state": self._gripper_state,
            "object_position": list(self._object_position),
            "object_type": self._object_type,
            "bin_position": list(self._bin_position),
            "held_object": self._held_object_id is not None,
            "task_status": self._task_status,
            "last_event": self._last_event,
        }

    def close(self) -> None:
        if self._client_id is not None:
            p.disconnect(self._client_id)
            self._client_id = None

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return min(max(value, min_value), max_value)

    @staticmethod
    def _distance(a, b) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def _create_box(self, half_extents, position, color, mass: float = 0.0) -> int:
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._client_id
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self._client_id
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self._client_id,
        )

    def _create_sphere(self, radius, position, color, mass: float = 0.0) -> int:
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=radius, physicsClientId=self._client_id
        )
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=self._client_id
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self._client_id,
        )


if __name__ == "__main__":
    backend = PyBulletBackend(gui=False)
    print("Reset state:", backend.reset())

    dummy_command = RobotCommand(
        target_dx=0.05,
        target_dy=0.0,
        target_dz=0.0,
        target_droll=0.0,
        target_dpitch=0.0,
        target_dyaw=0.0,
        gripper_command="close",
    )
    print("State after command:", backend.apply_command(dummy_command))
    backend.close()
