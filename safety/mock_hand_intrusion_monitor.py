import numpy as np

from safety.safety_monitor import SafetyMonitor
from safety.safety_types import SafetyDecision


class MockHandIntrusionMonitor(SafetyMonitor):
    """Mock-timed hand-intrusion signal for Safety Pause/Resume v0.

    Ignores the frame content entirely (like MockSafetyMonitor) and
    instead compares an externally-driven step counter (see set_step())
    against [start_step, end_step) to simulate a hand entering and
    leaving the workspace at fixed control-loop steps. This lets the
    pause/resume state machine in run_full_recycling_cell_demo.py be
    validated before any real hand/person detector exists.
    """

    def __init__(self, start_step: int, end_step: int, active: bool = True):
        self.start_step = start_step
        self.end_step = end_step
        self.active = active
        self.current_step = 0

    def set_step(self, step_index: int) -> None:
        self.current_step = step_index

    def check(self, frame: np.ndarray) -> SafetyDecision:
        hand_detected = self.active and self.start_step <= self.current_step < self.end_step
        if hand_detected:
            return SafetyDecision(
                emergency_stop=True,
                reason="mock_hand_intrusion",
                detections=[{"label": "mock_hand", "confidence": 1.0}],
            )

        return SafetyDecision(emergency_stop=False, reason="safe", detections=[])
