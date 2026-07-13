"""ArUco table-plane homography image -> Panda sim mapping (v0).

Four ArUco markers are taped to the four corners of the table's usable
work area. Each marker's *known* sim (x, y) position (from calibration
config) is paired with where that marker's center is actually seen in
the current camera frame, and cv2.findHomography() turns those 4
correspondences into a homography that maps any other image pixel
(the detected object's bbox center) onto the table plane in sim
coordinates.

This is still not real camera calibration (no intrinsics/distortion
model, no 3D pose) and still assumes a flat table plane (object_z is a
fixed constant, not measured) -- but unlike CalibratedImageToSimMapper's
manually-tuned ROI + linear-stretch mapping, this homography is
re-derived from the markers every single frame. So if the camera moves
or is re-angled slightly, the mapping recalibrates itself automatically
as long as the four markers are still visible, instead of requiring a
person to re-edit configs/real2sim_webcam_calibration.json by hand.

No ArUco pose estimation (rvec/tvec, needs camera intrinsics -- not
available here), no Eye-in-Hand wrist camera correction, no depth
estimation. Those are later steps toward the full architecture
(see docs/architecture.md).
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_MAPPING_MODE = "aruco_table_plane_homography"

ARUCO_UNAVAILABLE_MESSAGE = "cv2.aruco is not available. Install opencv-contrib-python in the active venv."


class ArUcoTableMapper:
    def __init__(self, calibration_path: Union[str, Path]):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(ARUCO_UNAVAILABLE_MESSAGE)

        calibration_path = Path(calibration_path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"ArUco calibration config not found: {calibration_path}")

        with open(calibration_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.mapping_mode = config.get("mapping_mode", DEFAULT_MAPPING_MODE)

        aruco_config = config["aruco"]
        self.dictionary_name = aruco_config["dictionary"]
        self.required_marker_ids = list(aruco_config["required_marker_ids"])
        self.min_required_markers = aruco_config.get("min_required_markers", len(self.required_marker_ids))

        self.table_markers = config["table_markers"]
        workspace = config["sim_workspace"]
        self.object_z = float(workspace["object_z"])
        self.debug_config = config.get("debug", {"draw_marker_axes": False, "draw_table_polygon": True})

        self.workspace_bounds = None
        if all(key in workspace for key in ("x_min", "x_max", "y_min", "y_max")):
            self.workspace_bounds = {
                "x_min": float(workspace["x_min"]),
                "x_max": float(workspace["x_max"]),
                "y_min": float(workspace["y_min"]),
                "y_max": float(workspace["y_max"]),
            }

        self.out_of_bounds_policy = config.get("out_of_bounds_policy", "reject")
        if self.out_of_bounds_policy not in ("reject", "clamp", "allow"):
            raise ValueError(f"Unknown out_of_bounds_policy: {self.out_of_bounds_policy!r}")

        if not hasattr(cv2.aruco, self.dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary_name}")

        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary_name))
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    def detect_markers(self, image: np.ndarray) -> dict:
        """Returns {marker_id: {"center": [u, v], "corners": [[x, y], ...4]}}
        for every marker actually seen in `image` (an RGB array)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        detections = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                points = marker_corners[0]
                center = points.mean(axis=0)
                detections[int(marker_id)] = {
                    "center": [float(center[0]), float(center[1])],
                    "corners": [[float(x), float(y)] for x, y in points],
                }
        return detections

    def compute_homography(self, detected_markers: dict) -> Optional[np.ndarray]:
        image_points = []
        sim_points = []
        for marker_id in self.required_marker_ids:
            if marker_id not in detected_markers:
                return None
            image_points.append(detected_markers[marker_id]["center"])
            sim_points.append(self.table_markers[str(marker_id)]["sim_xy"])

        image_points = np.array(image_points, dtype=np.float32)
        sim_points = np.array(sim_points, dtype=np.float32)
        homography, _mask = cv2.findHomography(image_points, sim_points)
        return homography

    def map_detection(self, detection, image: np.ndarray) -> Tuple[Optional[list], dict]:
        """detection is a perception.detection_types.Detection (uses its
        bbox_xyxy); image is the full RGB frame the detection came from
        -- needed here (unlike CalibratedImageToSimMapper) because the
        homography itself has to be re-derived from markers seen in this
        specific frame.
        """
        marker_detections = self.detect_markers(image)
        detected_marker_ids = sorted(marker_detections.keys())

        x1, y1, x2, y2 = detection.bbox_xyxy
        bbox_center_px = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        missing_marker_ids = [m for m in self.required_marker_ids if m not in marker_detections]
        homography_valid = not missing_marker_ids and len(marker_detections) >= self.min_required_markers

        debug = {
            "mapping_mode": self.mapping_mode,
            "detected_marker_ids": detected_marker_ids,
            "required_marker_ids": list(self.required_marker_ids),
            "marker_centers_px": {
                str(marker_id): info["center"] for marker_id, info in marker_detections.items()
            },
            "marker_sim_xy": {
                str(marker_id): self.table_markers[str(marker_id)]["sim_xy"]
                for marker_id in self.required_marker_ids
            },
            "bbox_center_px": bbox_center_px,
            "mapped_position": None,
            "homography_valid": False,
        }

        if not homography_valid:
            debug["error"] = (
                f"ArUco mapping failed: required markers {self.required_marker_ids}, "
                f"detected {detected_marker_ids}"
            )
            return None, debug

        homography = self.compute_homography(marker_detections)
        if homography is None:
            debug["error"] = "cv2.findHomography could not compute a valid homography from the detected markers."
            return None, debug

        point = np.array([[bbox_center_px]], dtype=np.float32)
        sim_xy = cv2.perspectiveTransform(point, homography)[0][0]
        mapped_position_raw = [float(sim_xy[0]), float(sim_xy[1]), self.object_z]

        debug["homography_valid"] = True
        debug["mapped_position_raw"] = mapped_position_raw
        debug["workspace_bounds"] = self.workspace_bounds
        debug["out_of_bounds_policy"] = self.out_of_bounds_policy
        debug["clamped"] = False

        out_of_bounds = False
        if self.workspace_bounds is not None:
            bounds = self.workspace_bounds
            out_of_bounds = not (
                bounds["x_min"] <= mapped_position_raw[0] <= bounds["x_max"]
                and bounds["y_min"] <= mapped_position_raw[1] <= bounds["y_max"]
            )
        debug["out_of_bounds"] = out_of_bounds

        if not out_of_bounds:
            debug["mapped_position"] = mapped_position_raw
            return mapped_position_raw, debug

        if self.out_of_bounds_policy == "clamp":
            bounds = self.workspace_bounds
            clamped_position = [
                min(max(mapped_position_raw[0], bounds["x_min"]), bounds["x_max"]),
                min(max(mapped_position_raw[1], bounds["y_min"]), bounds["y_max"]),
                self.object_z,
            ]
            debug["clamped"] = True
            debug["mapped_position"] = clamped_position
            return clamped_position, debug

        if self.out_of_bounds_policy == "allow":
            debug["mapped_position"] = mapped_position_raw
            return mapped_position_raw, debug

        # reject (default): debug still shows what the position would have
        # been, but the returned mapped_position is None so callers treat
        # this the same as a failed mapping (no PyBullet execution).
        bounds = self.workspace_bounds
        debug["mapped_position"] = mapped_position_raw
        debug["rejection_reason"] = "mapped position is outside sim workspace"
        debug["error"] = (
            f"ArUco mapping rejected: mapped_position "
            f"[{mapped_position_raw[0]:.4f}, {mapped_position_raw[1]:.4f}, {mapped_position_raw[2]:.4f}] "
            f"is outside sim workspace x=[{bounds['x_min']},{bounds['x_max']}], "
            f"y=[{bounds['y_min']},{bounds['y_max']}]. Place the object inside the marker quadrilateral."
        )
        return None, debug


def print_aruco_mapping_debug(debug: dict) -> None:
    print("=== ArUco Real2Sim Mapping Debug ===")
    print(f"mapping_mode: {debug['mapping_mode']}")
    print(f"detected_marker_ids: {debug['detected_marker_ids']}")
    print(f"required_marker_ids: {debug['required_marker_ids']}")
    print(f"marker_centers_px: {debug['marker_centers_px']}")
    print(f"marker_sim_xy: {debug['marker_sim_xy']}")
    print(f"bbox_center_px: {debug['bbox_center_px']}")
    print(f"homography_valid: {debug['homography_valid']}")
    if "mapped_position_raw" in debug:
        raw = debug["mapped_position_raw"]
        print(f"mapped_position_raw: [{raw[0]:.4f}, {raw[1]:.4f}, {raw[2]:.4f}]")
    if debug.get("workspace_bounds") is not None:
        print(f"workspace_bounds: {debug['workspace_bounds']}")
    if "out_of_bounds" in debug:
        print(f"out_of_bounds: {debug['out_of_bounds']}")
    if "out_of_bounds_policy" in debug:
        print(f"out_of_bounds_policy: {debug['out_of_bounds_policy']}")
    if debug.get("clamped"):
        print(f"clamped: {debug['clamped']}")
    if debug["mapped_position"] is not None:
        mapped = debug["mapped_position"]
        print(f"mapped_position: [{mapped[0]:.4f}, {mapped[1]:.4f}, {mapped[2]:.4f}]")
    if "error" in debug:
        print(debug["error"])


def draw_aruco_debug_image(
    image: np.ndarray,
    marker_detections: dict,
    required_marker_ids: list,
    detection=None,
    mapped_position: Optional[list] = None,
    task_goal=None,
    summary: str = "",
    draw_table_polygon: bool = True,
) -> np.ndarray:
    """Draws detected marker outlines/IDs, the table polygon connecting
    the required markers' centers (in required_marker_ids order, which
    is expected to already trace the table perimeter -- see
    configs/real2sim_aruco_table_calibration.json's front_left ->
    front_right -> back_right -> back_left corner naming), the selected
    detection's bbox/center, and the mapped position text.
    """
    pil_image = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    if draw_table_polygon:
        polygon_points = [
            tuple(marker_detections[marker_id]["center"])
            for marker_id in required_marker_ids
            if marker_id in marker_detections
        ]
        if len(polygon_points) >= 2:
            draw.line(polygon_points + [polygon_points[0]], fill=(0, 200, 255), width=2)

    for marker_id, info in marker_detections.items():
        draw.polygon([tuple(corner) for corner in info["corners"]], outline=(255, 165, 0), width=2)
        center_x, center_y = info["center"]
        draw.text((center_x - 6, center_y - 6), str(marker_id), fill=(255, 165, 0))

    if detection is not None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        center_x, center_y = detection.center_xy
        radius = 5
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill=(255, 255, 0),
        )
        draw.text((x1, max(y1 - 20, 0)), f"{detection.label} {detection.confidence:.2f}", fill=(255, 0, 0))

    if task_goal is not None:
        draw.text((5, 5), f"goal: {task_goal.target_object} -> {task_goal.target_bin}", fill=(255, 0, 0))

    if mapped_position is not None:
        pos_text = f"sim_pos=({mapped_position[0]:.2f}, {mapped_position[1]:.2f}, {mapped_position[2]:.2f})"
    else:
        pos_text = "sim_pos=N/A (ArUco mapping failed)"
    draw.text((5, pil_image.height - 20), pos_text, fill=(255, 128, 0))

    if summary:
        draw.text((5, pil_image.height - 36), summary, fill=(255, 128, 0))

    return np.array(pil_image)
