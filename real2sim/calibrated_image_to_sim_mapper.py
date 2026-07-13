"""Calibrated, ROI-aware image -> Panda table-plane mapping (v1).

Still NOT camera calibration in the intrinsics/extrinsics/depth-estimation
sense -- no monocular depth model, no RealSense/depth camera, no ArUco/
AprilTag, no Eye-in-Hand wrist camera. The assumption stays the same one
the plain ImageToSimMapper (v0) makes: the camera is fixed and the object
sits on a flat table plane, so a single 2D pixel position is enough to
place it. Those are the next architectural steps (see docs/architecture.md)
-- this module only makes the *existing* 2D mapping configurable and
debuggable.

What changes from v0:

  1. A configurable image ROI (image_roi) lets you exclude the part of
     the frame that isn't the table (e.g. wall/background above it),
     instead of stretching the *entire* frame (including background)
     onto the workspace.
  2. Axis mapping is explicit and configurable instead of hardcoded.
     In particular, moving an object further from a camera that looks
     across/down at a table mostly shows up as a change in image *y*
     (it moves up/down in frame, not just left/right) -- so by default
     this maps image_y -> sim_x (the Panda's forward/depth axis) and
     image_x -> sim_y (its left/right axis), which is why "closer vs.
     farther" now visibly changes the mapped sim position instead of
     mostly changing the object's apparent size in the old x->x/y->y
     mapping.
  3. Every mapping call also returns a debug dict with the bbox/ROI/
     normalized-coordinate breakdown, so a bad calibration can actually
     be diagnosed (see benchmark/probe_real2sim_mapping.py) instead of
     only ever seeing the final [x, y, z].

Config shape (see configs/real2sim_webcam_calibration.json):

  {
    "mapping_mode": "roi_linear_table_plane",
    "image_roi": {"x_min": 0, "x_max": 640, "y_min": 120, "y_max": 480},
    "sim_workspace": {"x_min": 0.25, "x_max": 0.55, "y_min": -0.25, "y_max": 0.25, "object_z": 0.05},
    "axis_mapping": {
      "image_x_to_sim_y": true, "image_y_to_sim_x": true,
      "invert_image_x": false, "invert_image_y": false
    },
    "clamp_to_roi": true
  }
"""

import json
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_MAPPING_MODE = "roi_linear_table_plane"

DEFAULT_CALIBRATION = {
    "mapping_mode": DEFAULT_MAPPING_MODE,
    "image_roi": {"x_min": 0, "x_max": 640, "y_min": 120, "y_max": 480},
    "sim_workspace": {"x_min": 0.25, "x_max": 0.55, "y_min": -0.25, "y_max": 0.25, "object_z": 0.05},
    "axis_mapping": {
        "image_x_to_sim_y": True,
        "image_y_to_sim_x": True,
        "invert_image_x": False,
        "invert_image_y": False,
    },
    "clamp_to_roi": True,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class CalibratedImageToSimMapper:
    def __init__(self, config: dict):
        self.mapping_mode = config.get("mapping_mode", DEFAULT_MAPPING_MODE)

        roi = config["image_roi"]
        self.roi = {
            "x_min": float(roi["x_min"]),
            "x_max": float(roi["x_max"]),
            "y_min": float(roi["y_min"]),
            "y_max": float(roi["y_max"]),
        }

        workspace = config["sim_workspace"]
        self.sim_x_min, self.sim_x_max = float(workspace["x_min"]), float(workspace["x_max"])
        self.sim_y_min, self.sim_y_max = float(workspace["y_min"]), float(workspace["y_max"])
        self.object_z = float(workspace["object_z"])

        axis_mapping = config.get("axis_mapping", {})
        self.image_y_to_sim_x = axis_mapping.get("image_y_to_sim_x", True)
        self.image_x_to_sim_y = axis_mapping.get("image_x_to_sim_y", True)
        self.invert_image_x = axis_mapping.get("invert_image_x", False)
        self.invert_image_y = axis_mapping.get("invert_image_y", False)

        self.clamp_to_roi = config.get("clamp_to_roi", True)

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "CalibratedImageToSimMapper":
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Calibration config not found: {config_path}. Using built-in default calibration.")
            return cls(DEFAULT_CALIBRATION)

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return cls(config)

    def map_bbox_to_sim(self, bbox_xyxy: list, image_width: int, image_height: int) -> Tuple[list, dict]:
        x1, y1, x2, y2 = bbox_xyxy
        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_y = (y1 + y2) / 2.0
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        bbox_area_ratio = bbox_area / (image_width * image_height) if image_width and image_height else 0.0

        roi = self.roi
        u = (bbox_center_x - roi["x_min"]) / (roi["x_max"] - roi["x_min"])
        v = (bbox_center_y - roi["y_min"]) / (roi["y_max"] - roi["y_min"])

        clamped = (u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0)
        if clamped:
            print(
                f"Warning: bbox center ({bbox_center_x:.1f}, {bbox_center_y:.1f}) falls outside "
                f"image_roi (x: {roi['x_min']}-{roi['x_max']}, y: {roi['y_min']}-{roi['y_max']})"
                + (
                    "; clamping to the nearest ROI edge."
                    if self.clamp_to_roi
                    else "; clamp_to_roi is false, so the mapped position may fall outside sim_workspace."
                )
            )
        if self.clamp_to_roi:
            u = _clamp01(u)
            v = _clamp01(v)

        if self.invert_image_x:
            u = 1.0 - u
        if self.invert_image_y:
            v = 1.0 - v

        value_for_sim_x = v if self.image_y_to_sim_x else u
        value_for_sim_y = u if self.image_x_to_sim_y else v

        mapped_x = self.sim_x_min + value_for_sim_x * (self.sim_x_max - self.sim_x_min)
        mapped_y = self.sim_y_min + value_for_sim_y * (self.sim_y_max - self.sim_y_min)
        mapped_z = self.object_z
        mapped_position = [mapped_x, mapped_y, mapped_z]

        debug = {
            "mapping_mode": self.mapping_mode,
            "image_size": [image_width, image_height],
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_center": [bbox_center_x, bbox_center_y],
            "bbox_size": [bbox_width, bbox_height],
            "bbox_area": bbox_area,
            "bbox_area_ratio": bbox_area_ratio,
            "image_roi": dict(roi),
            "normalized_center": [u, v],
            "clamped": clamped,
            "mapped_position": mapped_position,
        }

        return mapped_position, debug


def print_mapping_debug(debug: dict) -> None:
    print("=== Real2Sim Mapping Debug ===")
    print(f"mapping_mode: {debug['mapping_mode']}")
    print(f"image_size: {debug['image_size'][0]}x{debug['image_size'][1]}")
    print(f"bbox_xyxy: {debug['bbox_xyxy']}")
    print(f"bbox_center: [{debug['bbox_center'][0]:.2f}, {debug['bbox_center'][1]:.2f}]")
    print(f"bbox_size: [{debug['bbox_size'][0]:.2f}, {debug['bbox_size'][1]:.2f}]")
    print(f"bbox_area: {debug['bbox_area']:.1f}")
    print(f"bbox_area_ratio: {debug['bbox_area_ratio']:.4f}")
    print(f"image_roi: {debug['image_roi']}")
    print(f"normalized_center: [{debug['normalized_center'][0]:.4f}, {debug['normalized_center'][1]:.4f}]")
    print(f"clamped: {debug['clamped']}")
    mapped = debug["mapped_position"]
    print(f"mapped_position: [{mapped[0]:.4f}, {mapped[1]:.4f}, {mapped[2]:.4f}]")


def draw_roi_rectangle(frame: np.ndarray, image_roi: dict, color=(0, 140, 255)) -> np.ndarray:
    """Draws the calibration ROI onto a copy of frame. Meant to be called
    before an existing bbox-drawing helper (e.g. draw_debug_image from
    run_task_goal_real2sim_panda_interrupt_demo) so the ROI shows up
    underneath the detection bbox/label/text instead of replacing them.
    """
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [image_roi["x_min"], image_roi["y_min"], image_roi["x_max"], image_roi["y_max"]],
        outline=color,
        width=2,
    )
    draw.text((image_roi["x_min"] + 4, image_roi["y_min"] + 2), "image_roi", fill=color)
    return np.array(image)
