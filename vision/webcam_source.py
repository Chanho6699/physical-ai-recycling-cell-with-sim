import cv2
import numpy as np

from vision.frame_source import FrameSource

WEBCAM_ERROR_MESSAGE = (
    "Webcam could not be opened. If you are running inside WSL, check "
    "whether /dev/video0 exists. You may need to run webcam capture from "
    "Windows Python or configure USB/IP. Try a different --camera-index "
    "(e.g. 0 or 1) as well."
)

CAMERA_URL_ERROR_MESSAGE = (
    "Check that the relay server (e.g. camera_stream_server.py on "
    "Windows) is running and reachable from WSL at that address, and "
    "that the URL includes the stream path (e.g. http://<host>:<port>/video)."
)


class WebcamSource(FrameSource):
    """FrameSource backed by either a local OpenCV camera index or an
    MJPEG-style camera URL (e.g. a Windows-side relay server forwarding
    an iVCam/phone camera into WSL, since WSL usually has no /dev/video*
    device of its own). camera_url takes priority over camera_index when
    both are given.
    """

    def __init__(
        self,
        camera_index: int = 0,
        camera_url: str = None,
        width: int = 640,
        height: int = 480,
    ):
        self.camera_index = camera_index
        self.camera_url = camera_url
        self.width = width
        self.height = height

        source = camera_url if camera_url else camera_index
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            if camera_url:
                error_message = f"Could not open camera URL: {camera_url}\n{CAMERA_URL_ERROR_MESSAGE}"
            else:
                error_message = f"Could not open webcam index {camera_index}\n{WEBCAM_ERROR_MESSAGE}"
            print(error_message)
            raise RuntimeError(error_message)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def warmup(self, num_frames: int = 10) -> None:
        """Read and discard a few frames so auto-exposure/auto-focus can
        settle before a frame is actually used (common webcam quirk --
        the first frame or two is often dark/blurry)."""
        for _ in range(num_frames):
            self.cap.read()

    def get_frame(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            if self.camera_url:
                raise RuntimeError(
                    f"Failed to read frame from camera URL: {self.camera_url} "
                    f"(connection opened but returned no frame). {CAMERA_URL_ERROR_MESSAGE}"
                )
            raise RuntimeError(
                "Failed to read frame from webcam (camera opened but returned no frame). "
                "Try a different --camera-index (e.g. 0 or 1) as well."
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb.astype(np.uint8)

    def close(self) -> None:
        self.cap.release()
