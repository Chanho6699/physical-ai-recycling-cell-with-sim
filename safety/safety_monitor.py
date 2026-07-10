from abc import ABC, abstractmethod

import numpy as np

from safety.safety_types import SafetyDecision


class SafetyMonitor(ABC):
    """Common interface for anything that judges a frame safe or not.

    Task pipeline code should depend only on this interface so
    MockSafetyMonitor today can be swapped for a real YOLOSafetyMonitor
    later without changing the gate logic around apply_command().
    """

    @abstractmethod
    def check(self, frame: np.ndarray) -> SafetyDecision:
        pass
