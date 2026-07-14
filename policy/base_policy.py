from abc import ABC, abstractmethod

from policy.policy_types import PolicyInput, PolicyOutput


class BasePolicy(ABC):
    """Common interface for anything that turns a PolicyInput into a
    PolicyOutput (7-DoF OpenVLA-style action). DummyOpenVLAPolicy
    implements this today with scripted phase logic; a real OpenVLA
    policy adapter (or a FastAPI dummy-server client) can implement the
    same interface later without changing the control loop around it.
    """

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def predict_action(self, policy_input: PolicyInput) -> PolicyOutput:
        ...
