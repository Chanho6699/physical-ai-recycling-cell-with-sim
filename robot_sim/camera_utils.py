"""PyBullet virtual camera utilities.

Renders the current PyBullet scene to an RGB numpy array and saves it as a
PNG. First step toward feeding simulated camera images into a perception /
VLA pipeline later -- no YOLO, no real OpenVLA, no ROS 2 here yet.

Requires numpy and Pillow (`pip install pillow`).
"""

import os
from typing import List, Optional

import numpy as np
import pybullet as p
from PIL import Image


def capture_pybullet_camera(
    width: int = 640,
    height: int = 480,
    camera_eye: Optional[List[float]] = None,
    camera_target: Optional[List[float]] = None,
    camera_up: Optional[List[float]] = None,
    fov: float = 60.0,
    near_val: float = 0.1,
    far_val: float = 5.0,
    physics_client_id: int = 0,
) -> np.ndarray:
    if camera_eye is None:
        camera_eye = [1.0, -1.0, 1.0]
    if camera_target is None:
        camera_target = [0.2, 0.25, 0.25]
    if camera_up is None:
        camera_up = [0.0, 0.0, 1.0]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_target,
        cameraUpVector=camera_up,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near_val,
        farVal=far_val,
    )

    _, _, rgba_pixels, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        physicsClientId=physics_client_id,
    )

    rgba_array = np.array(rgba_pixels, dtype=np.uint8).reshape((height, width, 4))
    return rgba_array[:, :, :3]


def save_rgb_image(image: np.ndarray, output_path: str) -> str:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(image, mode="RGB").save(output_path)
    return output_path
