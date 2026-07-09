from abc import ABC, abstractmethod

from action_adapter.adapter_v0 import RobotCommand


class SimulatorBackend(ABC):
    """Common interface for robot simulation backends.

    Task pipeline code (parser -> VLA -> ActionAdapter) should depend only
    on this interface, not on a specific simulator. This lets the same
    pipeline run against DummyRobotBackend today and swap in a PyBulletBackend
    or IsaacSimBackend later without changing pipeline code.
    """

    @abstractmethod
    def reset(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def apply_command(self, robot_command: RobotCommand) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
