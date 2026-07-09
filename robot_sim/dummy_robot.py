from dataclasses import asdict, dataclass

from action_adapter.adapter_v0 import RobotCommand

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.5


@dataclass
class RobotState:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    gripper_state: str


class DummyRobotSimulator:
    def __init__(self):
        self.state = RobotState(
            x=0.0,
            y=0.0,
            z=0.5,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            gripper_state="open",
        )

    def apply_command(self, robot_command: RobotCommand) -> RobotState:
        x = self.state.x + robot_command.target_dx
        y = self.state.y + robot_command.target_dy
        z = self.state.z + robot_command.target_dz

        self.state.x = min(max(x, X_MIN), X_MAX)
        self.state.y = min(max(y, Y_MIN), Y_MAX)
        self.state.z = min(max(z, Z_MIN), Z_MAX)

        self.state.roll += robot_command.target_droll
        self.state.pitch += robot_command.target_dpitch
        self.state.yaw += robot_command.target_dyaw

        self.state.gripper_state = robot_command.gripper_command

        return self.state

    def as_dict(self) -> dict:
        return asdict(self.state)


if __name__ == "__main__":
    simulator = DummyRobotSimulator()
    print("Initial state:", simulator.as_dict())

    dummy_command = RobotCommand(
        target_dx=0.01,
        target_dy=0.0,
        target_dz=0.02,
        target_droll=0.0,
        target_dpitch=0.0,
        target_dyaw=0.0,
        gripper_command="close",
    )
    simulator.apply_command(dummy_command)
    print("Updated state:", simulator.as_dict())
