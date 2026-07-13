"""Hardware-portable camera backend interface (v0).

CameraBackend is a thin, uniform read()/shutdown() wrapper over the
various frame sources this project already has (external webcam/iVCam
relay, a static test image, PyBullet's virtual wrist camera) plus a
skeleton for a future ROS2 camera topic, so a caller can depend on one
interface regardless of where frames actually come from.

v0 does not force run_full_recycling_cell_demo.py to route through
these wrappers everywhere -- it still calls WebcamSource/PyBulletWristCamera
directly in most places, since those already work and are well-tested.
These wrappers exist to make the swap explicit and available (see
create_external_camera_backend() in run_full_recycling_cell_demo.py,
which does use WebcamCameraBackend/StaticImageCameraBackend for the
initial detection frame) without risking the existing per-step wrist
camera / hand-safety live-frame code paths, which are already
well-exercised through PyBulletWristCamera/WebcamSource directly.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image


class CameraBackend(ABC):
    @abstractmethod
    def read(self) -> tuple:
        """Returns (ok: bool, frame: np.ndarray | None, debug: dict).

        ok=False means the read failed (frame is None); debug always
        carries at least {"source": <backend-specific string>}."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release any underlying resource (camera handle, ROS2
        subscription, etc). Idempotent -- safe to call more than once."""
        raise NotImplementedError


class WebcamCameraBackend(CameraBackend):
    """Wraps vision/webcam_source.py's WebcamSource (external
    webcam/iVCam relay) behind CameraBackend. Does not change
    WebcamSource itself."""

    def __init__(self, camera_index: int = 0, camera_url: Optional[str] = None, warmup_frames: int = 10):
        from vision.webcam_source import WebcamSource

        self._source = WebcamSource(camera_index=camera_index, camera_url=camera_url)
        self._source.warmup(warmup_frames)
        self.camera_index = camera_index
        self.camera_url = camera_url

    def read(self) -> tuple:
        try:
            frame = self._source.get_frame()
            return True, frame, {"source": "webcam", "camera_url": self.camera_url, "camera_index": self.camera_index}
        except RuntimeError as exc:
            return False, None, {"source": "webcam", "error": str(exc)}

    def shutdown(self) -> None:
        self._source.close()


class StaticImageCameraBackend(CameraBackend):
    """Wraps a fixed --image-path frame behind CameraBackend. Loads the
    image once at construction time; every read() returns the same
    array (a static image never changes between calls, unlike a live
    camera)."""

    def __init__(self, image_path: Union[str, Path]):
        self.image_path = Path(image_path)
        self._frame = np.array(Image.open(self.image_path).convert("RGB"), dtype=np.uint8)

    def read(self) -> tuple:
        return True, self._frame, {"source": "static_image", "image_path": str(self.image_path)}

    def shutdown(self) -> None:
        pass


class PyBulletWristCameraBackend(CameraBackend):
    """Wraps robot_sim/pybullet_wrist_camera.py's PyBulletWristCamera
    behind CameraBackend. read() returns the rendered frame's "rgb"
    array (PyBulletWristCamera.render() itself returns a richer dict
    with rgb/depth/segmentation -- that full dict is still available via
    self.wrist_camera.render() directly for callers that need depth/seg,
    e.g. grasp refinement)."""

    def __init__(self, wrist_camera):
        self.wrist_camera = wrist_camera

    def read(self) -> tuple:
        frame, render_debug = self.wrist_camera.render()
        return True, frame["rgb"], {"source": "pybullet_wrist_camera", **render_debug}

    def shutdown(self) -> None:
        pass


ROS2_CAMERA_MISSING_MESSAGE = (
    "rclpy is not installed. Install a ROS2 distribution and source its "
    "setup.bash before instantiating ROS2CameraBackend."
)


class ROS2CameraBackend(CameraBackend):
    """Skeleton for a future ROS2 camera topic (e.g.
    /camera/color/image_raw, sensor_msgs/Image). Not a working
    subscriber yet -- rclpy is imported lazily so this module (and the
    rest of the project) can be imported without ROS2 installed."""

    def __init__(self, topic: str = "/camera/color/image_raw", node_name: str = "physical_ai_camera_backend"):
        try:
            import rclpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(ROS2_CAMERA_MISSING_MESSAGE) from exc

        self.topic = topic
        self.node_name = node_name

    def read(self) -> tuple:
        raise NotImplementedError(
            "ROS2CameraBackend.read() is not implemented -- subscribe to "
            f"{self.topic} (sensor_msgs/Image), convert via cv_bridge, and "
            "return the latest frame. See docs/hardware_portability.md."
        )

    def shutdown(self) -> None:
        raise NotImplementedError("ROS2CameraBackend.shutdown() is not implemented -- destroy the node/subscription.")
