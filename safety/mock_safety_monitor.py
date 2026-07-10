import numpy as np

from safety.safety_monitor import SafetyMonitor
from safety.safety_types import SafetyDecision


class MockSafetyMonitor(SafetyMonitor):
    """Placeholder for a future YOLO-based SafetyMonitor.

    Ignores the frame content entirely and always returns the same fixed
    decision, so the safety-gate wiring (frame -> check -> emergency_stop
    -> block/allow apply_command) can be validated before a real detector
    exists.
    """

    def __init__(self, simulate_hazard: bool = False):
        self.simulate_hazard = simulate_hazard

    def check(self, frame: np.ndarray) -> SafetyDecision:
        if self.simulate_hazard:
            return SafetyDecision(
                emergency_stop=True,
                reason="simulated_hazard",
                detections=[{"label": "simulated_person", "confidence": 1.0}],
            )

        return SafetyDecision(emergency_stop=False, reason="safe", detections=[])
