"""Safety gate helper: wraps a SafetyMonitor to allow/block named actions.

Does not change the SafetyMonitor interface -- this is a thin wrapper
that attaches an action name to a SafetyDecision so callers can log/block
per named high-level action (e.g. "move_panda_to_object", "close_gripper").
"""

from dataclasses import dataclass

import numpy as np

from safety.safety_monitor import SafetyMonitor
from safety.safety_types import SafetyDecision


@dataclass
class SafetyGateResult:
    allowed: bool
    decision: SafetyDecision
    action_name: str
    reason: str


class SafetyGate:
    def __init__(self, safety_monitor: SafetyMonitor):
        self.safety_monitor = safety_monitor

    def check(self, frame: np.ndarray, action_name: str) -> SafetyGateResult:
        decision = self.safety_monitor.check(frame)
        return SafetyGateResult(
            allowed=not decision.emergency_stop,
            decision=decision,
            action_name=action_name,
            reason=decision.reason,
        )
