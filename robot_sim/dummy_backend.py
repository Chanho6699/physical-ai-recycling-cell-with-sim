from action_adapter.adapter_v0 import RobotCommand
from robot_sim.backend_interface import SimulatorBackend
from robot_sim.dummy_robot import DummyRobotSimulator


class DummyRobotBackend(SimulatorBackend):
    """SimulatorBackend wrapper around DummyRobotSimulator."""

    def __init__(self):
        self._simulator = DummyRobotSimulator()

    def reset(self) -> dict:
        self._simulator = DummyRobotSimulator()
        return self._simulator.as_dict()

    def apply_command(self, robot_command: RobotCommand) -> dict:
        self._simulator.apply_command(robot_command)
        return self._simulator.as_dict()

    def get_state(self) -> dict:
        return self._simulator.as_dict()

    def close(self) -> None:
        pass


if __name__ == "__main__":
    backend = DummyRobotBackend()
    print("Reset state:", backend.reset())

    dummy_command = RobotCommand(
        target_dx=0.01,
        target_dy=0.0,
        target_dz=0.02,
        target_droll=0.0,
        target_dpitch=0.0,
        target_dyaw=0.0,
        gripper_command="close",
    )
    print("State after command:", backend.apply_command(dummy_command))
    print("get_state():", backend.get_state())
    backend.close()
