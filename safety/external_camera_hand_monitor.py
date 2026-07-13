"""External-camera hand/arm intrusion SafetyMonitor (v1).

Detects a hand/arm entering the ArUco-derived (or config ROI fallback)
workspace polygon in a real external-camera frame, using MediaPipe's
HandLandmarker (Tasks API) for hand landmark detection. This is a
hand/arm intrusion detector, not a person detector -- for a recycling
table, what matters is a hand/arm crossing into the workspace, not a
whole person being visible in frame.

Plugs into the exact same duck-typed SafetyMonitor interface (check(),
plus the set_step()/set_workspace_polygon() convention already used by
safety/mock_hand_intrusion_monitor.py) that
run_full_recycling_cell_demo.py's --safety-mode pause-resume control
loop was built against, so swapping --hand-safety-source mock ->
external-camera requires no control-loop changes -- only which
SafetyMonitor gets constructed.

No real hand-model training here: HandLandmarker uses Google's public
pretrained hand_landmarker.task model bundle (see
docs/demo_commands.md for the download step), not a custom-trained
detector.
"""

import json
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

from safety.safety_monitor import SafetyMonitor
from safety.safety_types import SafetyDecision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_SAFETY_CONFIG = "configs/hand_safety_config.json"

MEDIAPIPE_INSTALL_MESSAGE = (
    "MediaPipe is required for ExternalCameraHandSafetyMonitor.\n"
    "Install it with: pip install mediapipe"
)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _point_in_polygon(point, polygon_px) -> bool:
    contour = np.array(polygon_px, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


class ExternalCameraHandSafetyMonitor(SafetyMonitor):
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        config_path = _resolve(config_path) if config_path else _resolve(DEFAULT_HAND_SAFETY_CONFIG)
        if not config_path.exists():
            raise FileNotFoundError(f"Hand safety config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.detector_backend = self.config.get("detector_backend", "mediapipe")
        self.model_asset_path = self.config.get("model_asset_path", "weights/hand_landmarker.task")
        self.min_detection_confidence = self.config.get("min_detection_confidence", 0.5)
        self.min_tracking_confidence = self.config.get("min_tracking_confidence", 0.5)
        self.pause_if_hand_in_workspace = self.config.get("pause_if_hand_in_workspace", True)
        self.roi_config = self.config.get("roi", {"enabled": False})

        self._landmarker = None  # lazy mediapipe HandLandmarker instance
        self.current_step = 0
        self.workspace_polygon_px = None
        self.last_debug = None

    def set_step(self, step_index: int) -> None:
        self.current_step = step_index

    def set_workspace_polygon(self, polygon_px: Optional[list]) -> None:
        """polygon_px: 4 (or more) [x, y] pixel points tracing the
        workspace perimeter (e.g. the 4 ArUco marker centers, in
        perimeter order), or None if no valid workspace could be
        derived this run (falls back to config ROI, if enabled)."""
        self.workspace_polygon_px = polygon_px

    def _ensure_detector(self) -> None:
        if self._landmarker is not None:
            return
        if self.detector_backend != "mediapipe":
            raise ValueError(f"Unsupported detector_backend: {self.detector_backend!r}")

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(MEDIAPIPE_INSTALL_MESSAGE) from exc

        model_path = _resolve(self.model_asset_path)
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe hand landmark model not found: {model_path}\n"
                "Download it with:\n"
                "curl -sL -o weights/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/latest/hand_landmarker.task"
            )

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _resolve_workspace_polygon(self, workspace_polygon_px):
        if workspace_polygon_px is not None:
            return workspace_polygon_px, True
        if self.workspace_polygon_px is not None:
            return self.workspace_polygon_px, True
        if self.roi_config.get("enabled"):
            roi = self.roi_config
            polygon = [
                [roi["x_min"], roi["y_min"]],
                [roi["x_max"], roi["y_min"]],
                [roi["x_max"], roi["y_max"]],
                [roi["x_min"], roi["y_max"]],
            ]
            return polygon, True
        return None, False

    def check_frame(
        self,
        frame: np.ndarray,
        step_index: int,
        workspace_polygon_px: Optional[list] = None,
    ) -> tuple:
        self._ensure_detector()
        polygon, workspace_valid = self._resolve_workspace_polygon(workspace_polygon_px)

        height, width = frame.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = self._landmarker.detect(mp_image)

        hand_detected = bool(result.hand_landmarks)
        hand_in_workspace = False
        hand_landmarks_px = []
        intrusion_points_px = []
        all_points = []
        confidence = None

        for hand_landmarks in result.hand_landmarks:
            points = [[landmark.x * width, landmark.y * height] for landmark in hand_landmarks]
            hand_landmarks_px.append(points)
            all_points.extend(points)

            if polygon is not None:
                for point in points:
                    if _point_in_polygon(point, polygon):
                        hand_in_workspace = True
                        intrusion_points_px.append(point)

        if result.handedness:
            confidence = max(
                category.score for categories in result.handedness for category in categories
            )

        hand_bbox_px = None
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            hand_bbox_px = [min(xs), min(ys), max(xs), max(ys)]

        emergency_stop = bool(hand_in_workspace and workspace_valid and self.pause_if_hand_in_workspace)

        decision = SafetyDecision(
            emergency_stop=emergency_stop,
            reason="hand_in_workspace" if emergency_stop else "safe",
            detections=(
                [{"label": "hand", "confidence": confidence or 1.0, "bbox_xyxy": hand_bbox_px}]
                if hand_detected
                else []
            ),
            severity="high" if emergency_stop else "none",
        )

        debug = {
            "hand_detected": hand_detected,
            "hand_in_workspace": hand_in_workspace,
            "hand_landmarks_px": hand_landmarks_px,
            "hand_bbox_px": hand_bbox_px,
            "workspace_polygon_px": polygon,
            "workspace_valid": workspace_valid,
            "intrusion_points_px": intrusion_points_px,
            "detector_backend": self.detector_backend,
            "confidence": confidence,
            "step_index": step_index,
        }
        self.last_debug = debug
        return decision, debug

    def check(self, frame: np.ndarray) -> SafetyDecision:
        decision, _debug = self.check_frame(frame, self.current_step, self.workspace_polygon_px)
        return decision


def draw_hand_safety_debug_image(frame: np.ndarray, debug: dict) -> np.ndarray:
    pil_image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    polygon = debug.get("workspace_polygon_px")
    if polygon:
        points = [tuple(point) for point in polygon]
        draw.line(points + [points[0]], fill=(0, 200, 255), width=2)

    for hand_points in debug.get("hand_landmarks_px", []):
        for x, y in hand_points:
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 0))

    if debug.get("hand_bbox_px"):
        x1, y1, x2, y2 = debug["hand_bbox_px"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=2)

    for x, y in debug.get("intrusion_points_px", []):
        radius = 5
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 0, 0), width=2)

    status_text = "HAND_IN_WORKSPACE" if debug.get("hand_in_workspace") else "SAFE"
    status_color = (255, 0, 0) if debug.get("hand_in_workspace") else (0, 200, 0)
    draw.text((5, 5), f"status: {status_text}", fill=status_color)
    draw.text((5, pil_image.height - 20), f"step: {debug.get('step_index')}", fill=(255, 128, 0))

    return np.array(pil_image)


def save_hand_safety_debug_image(frame: np.ndarray, debug: dict, output_dir: Union[str, Path]) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_image = draw_hand_safety_debug_image(frame, debug)
    step_index = debug.get("step_index", 0)
    output_path = output_dir / f"hand_safety_step_{step_index:06d}.png"
    Image.fromarray(debug_image).save(output_path)
    return str(output_path)
