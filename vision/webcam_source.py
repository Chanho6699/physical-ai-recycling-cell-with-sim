import cv2
import numpy as np

from vision.frame_source import FrameSource

WEBCAM_ERROR_MESSAGE = (
    "Webcam could not be opened. If you are running inside WSL, check "
    "whether /dev/video0 exists. You may need to run webcam capture from "
    "Windows Python or configure USB/IP."
)


class WebcamSource(FrameSource):
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(WEBCAM_ERROR_MESSAGE)
            raise RuntimeError(WEBCAM_ERROR_MESSAGE)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb.astype(np.uint8)

    def close(self) -> None:
        self.cap.release()
