from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    """Common interface for anything that can hand over an RGB frame.

    Lets SafetyMonitor (and later a real perception pipeline) consume
    frames from PyBullet's virtual camera today and swap in a real webcam
    or Isaac Sim camera later without changing that code.
    """

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """
        Return RGB image as np.ndarray with shape (H, W, 3).
        """
        pass

    def close(self) -> None:
        pass
