"""Hardware-portable robot backend interface (v0).

RobotBackend is the boundary this project's control loop (see
run_full_recycling_cell_demo.py) is meant to depend on, not any
specific simulator. PyBulletPandaBackend implements it today;
RealRobotBackend/ROS2RobotBackend (see real_robot_backend.py,
ros2_robot_backend.py) are unimplemented skeletons showing exactly
which methods a hardware integration needs to fill in to become a
drop-in replacement.

This intentionally overlaps with robot_sim/backend_interface.py's
older, narrower SimulatorBackend (reset/apply_command/get_state/close)
-- that interface predates wrist-camera/grasp-refinement code that
calls move_end_effector_to()/open_gripper()/close_gripper() directly,
so those methods were already de facto part of PyBulletPandaBackend's
public surface without being declared anywhere. RobotBackend just
makes that surface explicit. PyBulletPandaBackend implements both
interfaces side by side (see robot_sim/pybullet_panda_backend.py) --
existing code depending on SimulatorBackend keeps working unchanged.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class RobotBackend(ABC):
    @abstractmethod
    def reset(self) -> dict:
        """Reset the robot (and, in simulation, the scene) to its
        starting state. Returns the same shape of dict get_state() does."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        """Return the current robot/task state dict -- at minimum
        end_effector_position, end_effector_orientation, gripper_width,
        gripper_state, held_object, task_status, last_event (see
        PyBulletPandaBackend.get_state() for the full current shape)."""
        raise NotImplementedError

    @abstractmethod
    def apply_command(self, command: Any, steps: int = 1, **kwargs) -> dict:
        """Apply one RobotCommand (see action_adapter/adapter_v0.py) --
        a relative end-effector delta plus an optional gripper command --
        and return the resulting state."""
        raise NotImplementedError

    @abstractmethod
    def move_end_effector_to(
        self, target_position: list, target_orientation: Optional[list] = None, **kwargs
    ) -> dict:
        """Move the end effector to an absolute Cartesian target
        (world/base frame [x, y, z] meters, optional orientation
        quaternion). Returns the resulting state."""
        raise NotImplementedError

    @abstractmethod
    def open_gripper(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def close_gripper(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release any resources (simulator connection, hardware/ROS2
        session, etc). Idempotent -- safe to call more than once."""
        raise NotImplementedError
