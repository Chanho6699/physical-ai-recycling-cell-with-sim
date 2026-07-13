"""Skeleton for a future real hardware robot arm backend (v0).

This class intentionally does not control a real robot yet -- every
method raises NotImplementedError. It exists to make the hardware
integration surface concrete: implement these methods (bridging to a
vendor SDK, a robot controller's HTTP/gRPC API, or a ROS2 node -- see
ros2_robot_backend.py for that specific path) and RealRobotBackend
becomes a drop-in replacement for PyBulletPandaBackend everywhere
run_full_recycling_cell_demo.py depends on RobotBackend.

No real hardware control code lives here. See docs/hardware_portability.md
for the full list of what has to change (calibration, safety wiring,
etc) when swapping this in.
"""

from typing import Any, Optional

from robot_core.robot_backend import RobotBackend

NOT_IMPLEMENTED_MESSAGE = (
    "RealRobotBackend.{method}() is not implemented -- this is a hardware-"
    "portability skeleton, not a working hardware controller. Implement it "
    "against your robot's SDK/controller API. See docs/hardware_portability.md."
)


class RealRobotBackend(RobotBackend):
    """Skeleton for future hardware robot arm integration.

    Implementations should bridge to a vendor SDK, a robot controller
    API, or (see ros2_robot_backend.py) ROS2 topics/actions/services.
    """

    def __init__(self, robot_config: Optional[dict] = None):
        """robot_config: implementation-defined connection/calibration
        info (e.g. controller IP/port, joint limits, end-effector-frame
        offset, gripper type). Not validated or used here."""
        self.robot_config = robot_config or {}

    def reset(self) -> dict:
        """Expected to: home the arm to a known safe joint configuration,
        open the gripper, and return get_state()'s shape. Real hardware
        should almost certainly require an explicit operator
        confirmation step here before moving, unlike the simulator."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="reset"))

    def get_state(self) -> dict:
        """Expected to read the controller's current joint/end-effector
        state (e.g. via a /joint_states-equivalent read or SDK call) and
        return a dict with at least: end_effector_position (list[float],
        meters, robot base frame), end_effector_orientation (quaternion),
        gripper_width (float), gripper_state ("open"/"close"),
        held_object (bool, if the gripper/force sensor can tell),
        task_status (str), last_event (str)."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="get_state"))

    def apply_command(self, command: Any, steps: int = 1, **kwargs) -> dict:
        """command is a RobotCommand (action_adapter/adapter_v0.py):
        relative end-effector deltas (target_dx/dy/dz/droll/dpitch/dyaw)
        plus an optional gripper_command ("open"/"close"/None). Expected
        to convert this into whatever the real controller needs (joint
        trajectory, Cartesian velocity command, etc), block until motion
        settles (or the configured timeout elapses), and return
        get_state(). MUST check a hardware/software e-stop signal before
        (and ideally during) motion -- SafetySupervisor deciding an
        action may be applied is necessary but not sufficient; the
        backend itself should refuse to move if a hardware interlock is
        tripped."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="apply_command"))

    def move_end_effector_to(
        self, target_position: list, target_orientation: Optional[list] = None, **kwargs
    ) -> dict:
        """Absolute Cartesian target (robot base frame, meters) plus
        optional orientation quaternion. Expected to run inverse
        kinematics (controller-side or via MoveIt2, see
        ros2_robot_backend.py) and execute the resulting trajectory."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="move_end_effector_to"))

    def open_gripper(self) -> dict:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="open_gripper"))

    def close_gripper(self) -> dict:
        """Real hardware should also report whether the grasp actually
        succeeded (e.g. via gripper feedback/force sensing) rather than
        the simulator's distance-based heuristic -- see get_state()'s
        held_object."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="close_gripper"))

    def shutdown(self) -> None:
        """Expected to: stop any in-flight motion, release the
        controller connection/session, and leave the arm in a safe
        resting state."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="shutdown"))
