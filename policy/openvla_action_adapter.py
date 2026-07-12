"""Dataset action -> OpenVLA-style 7-DoF action vector adapter (v0).

  {"delta_ee_position": [dx, dy, dz], "gripper_action": "hold"|"close"|"open"}
  -> [dx, dy, dz, droll, dpitch, dyaw, gripper]

No real OpenVLA model is loaded here -- this only reshapes the exported
dataset action format into the vector shape a real OpenVLA policy would
output, so the downstream action_adapter.adapter_v0.ActionAdapter /
PyBulletPandaBackend execution path can be smoke-tested ahead of time.

rotation deltas (droll/dpitch/dyaw) are always 0.0 in v0 -- the recorded
dataset actions don't carry orientation deltas (see
data_collection/lerobot_dataset_exporter.py).

The adapter is stateful: "hold" has no meaning on its own, so it repeats
the *previous* gripper value it produced (starting from
default_gripper_value if "hold" is the very first action seen).
"""

from typing import List


class OpenVLAActionAdapter:
    def __init__(self, default_gripper_value: float = 0.0):
        self.previous_gripper_value = default_gripper_value

    def dataset_action_to_openvla_action(self, dataset_action: dict) -> List[float]:
        dx, dy, dz = dataset_action["delta_ee_position"]
        gripper_action = dataset_action["gripper_action"]

        if gripper_action == "close":
            gripper_value = 1.0
        elif gripper_action == "open":
            gripper_value = 0.0
        else:  # "hold"
            gripper_value = self.previous_gripper_value

        self.previous_gripper_value = gripper_value

        return [dx, dy, dz, 0.0, 0.0, 0.0, gripper_value]
