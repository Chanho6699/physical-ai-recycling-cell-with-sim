import numpy as np

from robot_sim.camera_utils import capture_pybullet_camera
from vision.frame_source import FrameSource


class SimCameraSource(FrameSource):
    """FrameSource backed by PyBullet's virtual camera."""

    def __init__(self, width: int = 640, height: int = 480, physics_client_id: int = 0):
        self.width = width
        self.height = height
        self.physics_client_id = physics_client_id

    def get_frame(self) -> np.ndarray:
        return capture_pybullet_camera(
            width=self.width,
            height=self.height,
            physics_client_id=self.physics_client_id,
        )
