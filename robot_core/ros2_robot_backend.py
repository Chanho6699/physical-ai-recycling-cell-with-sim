"""ROS2RobotBackend is a hardware-portability skeleton, not a working
ROS2 controller yet.

It does not start rclpy, spin a node, or publish/subscribe to anything.
rclpy is imported lazily (only inside __init__, only if this class is
actually instantiated) so that importing this module -- or the rest of
this project -- never requires ROS2 to be installed. Every method
raises NotImplementedError until a real ROS2 integration fills them in.

Expected future integration points (topics/actions/services a real
implementation would use -- Franka/UR/generic MoveIt2-style stack):

  /joint_states                                   (subscribe, get_state)
  /joint_trajectory_controller/follow_joint_trajectory  (action, move_end_effector_to via MoveIt2)
  /gripper_controller                             (action/topic, open_gripper/close_gripper)
  /tf                                             (robot base <-> end-effector <-> camera frames)
  /camera/color/image_raw                         (external camera feed -- see vision/camera_backend.py's
                                                    ROS2CameraBackend skeleton)
  /safety_state                                   (publish SafetyDecision-derived hardware e-stop signal)
  MoveIt2 planning service/action (e.g. /compute_ik, /move_action)

See docs/hardware_portability.md for the full calibration/integration
checklist.
"""

from typing import Any, Optional

from robot_core.robot_backend import RobotBackend

NOT_IMPLEMENTED_MESSAGE = (
    "ROS2RobotBackend.{method}() is not implemented -- this is a hardware-"
    "portability skeleton, not a working ROS2 controller. See the module "
    "docstring for expected topics/actions and docs/hardware_portability.md."
)

RCLPY_MISSING_MESSAGE = (
    "rclpy is not installed (ROS2 is not required for any currently working "
    "demo in this project). Install a ROS2 distribution (e.g. Humble/Iron) "
    "and source its setup.bash before instantiating ROS2RobotBackend."
)


class ROS2RobotBackend(RobotBackend):
    """Skeleton for future ROS2-based hardware integration.

    Not a working ROS2 node: instantiating this class does not start
    rclpy, create a node, or open any topic/action/service client. Real
    work is left to a future implementation; this class only fixes the
    shape (RobotBackend) that implementation must satisfy, and documents
    where each method should plug into ROS2.
    """

    def __init__(self, node_name: str = "physical_ai_recycling_cell_robot_backend", ros_args: Optional[list] = None):
        try:
            import rclpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(RCLPY_MISSING_MESSAGE) from exc

        self.node_name = node_name
        self.ros_args = ros_args or []
        self._node = None  # would hold the rclpy Node once implemented

    def reset(self) -> dict:
        """Expected: send the arm to a known home configuration via the
        follow_joint_trajectory action, open the gripper, and block
        until /joint_states confirms the arm is settled."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="reset"))

    def get_state(self) -> dict:
        """Expected: read the latest /joint_states and /tf (end-effector
        pose) messages and translate them into the same state dict shape
        PyBulletPandaBackend.get_state() returns."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="get_state"))

    def apply_command(self, command: Any, steps: int = 1, **kwargs) -> dict:
        """Expected: convert the RobotCommand's relative delta into a
        Cartesian target, call MoveIt2's IK service (or an equivalent
        planning service/action), send the resulting joint trajectory to
        follow_joint_trajectory, and drive the gripper action if
        command.gripper_command is set. Should also subscribe to
        /safety_state (or receive a hardware e-stop callback) and abort
        the in-flight trajectory if it trips mid-motion."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="apply_command"))

    def move_end_effector_to(
        self, target_position: list, target_orientation: Optional[list] = None, **kwargs
    ) -> dict:
        """Expected: call MoveIt2's planning service/action (e.g.
        /compute_ik then /move_action, or moveit_py equivalent) for the
        given Cartesian target and execute the resulting trajectory via
        follow_joint_trajectory."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="move_end_effector_to"))

    def open_gripper(self) -> dict:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="open_gripper"))

    def close_gripper(self) -> dict:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="close_gripper"))

    def shutdown(self) -> None:
        """Expected: destroy the node and call rclpy.shutdown()."""
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.format(method="shutdown"))
