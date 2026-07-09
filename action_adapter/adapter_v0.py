from dataclasses import dataclass
from typing import List


@dataclass
class RobotCommand:
    target_dx: float
    target_dy: float
    target_dz: float
    target_droll: float
    target_dpitch: float
    target_dyaw: float
    gripper_command: str


class ActionAdapter:
    def __init__(self, position_scale: float = 1.0, rotation_scale: float = 1.0):
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale

    def convert(self, action: List[float]) -> RobotCommand:
        if len(action) != 7:
            raise ValueError(f"Expected 7-DoF action, got {len(action)} values")

        dx, dy, dz, droll, dpitch, dyaw, gripper = action

        gripper_command = "close" if gripper >= 0.5 else "open"

        return RobotCommand(
            target_dx=dx * self.position_scale,
            target_dy=dy * self.position_scale,
            target_dz=dz * self.position_scale,
            target_droll=droll * self.rotation_scale,
            target_dpitch=dpitch * self.rotation_scale,
            target_dyaw=dyaw * self.rotation_scale,
            gripper_command=gripper_command,
        )


if __name__ == "__main__":
    adapter = ActionAdapter(position_scale=1.0, rotation_scale=1.0)

    dummy_action = [0.01, 0.00, 0.02, 0.00, 0.00, 0.00, 1.0]
    command = adapter.convert(dummy_action)

    print(command)
