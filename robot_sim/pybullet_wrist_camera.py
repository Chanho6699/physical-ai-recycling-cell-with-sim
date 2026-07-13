"""PyBullet Eye-in-Hand virtual wrist camera (v0).

A second PyBullet camera, rigidly attached to the Panda end-effector
link instead of fixed in world space (see robot_sim/camera_utils.py /
vision/sim_camera_source.py for the external, world-fixed one). Its
pose is recomputed every render() call from the end-effector's current
world pose plus a fixed local offset/forward/up (configs/wrist_camera_config.json),
the same way a camera bolted to a real robot wrist would move with it.

This is the "local, robot-centric perception" half of the two-stage
architecture (see docs/architecture.md): the external ArUco camera
gives a coarse *global* object position; this wrist camera looks at the
object *up close* once the arm is already near it, using segmentation +
depth to re-estimate its position more precisely.

v1 adds refine_target_with_wrist_camera(): a grasp-target correction on
top of the v0 observe-only estimate, gated by a visibility/pixel-count/
delta-size trust check that falls back to the original (coarse) target
whenever the wrist estimate looks unreliable. No RGB-D sensor
model/noise, no real camera intrinsics/distortion, no ROS 2, no Isaac
Sim here.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pybullet as p

DEFAULT_CONFIG = {
    "camera_name": "panda_wrist_camera_v0",
    "end_effector_link_index": 11,
    "width": 320,
    "height": 240,
    "fov": 70.0,
    "near": 0.01,
    "far": 2.0,
    "camera_local_position": [0.04, 0.0, 0.06],
    "camera_forward_local": [0.0, 0.0, -1.0],
    "camera_up_local": [0.0, -1.0, 0.0],
    "save_depth_colormap": True,
    "save_segmentation_mask": True,
}

# PyBullet's segmentation buffer packs (bodyUniqueId, linkIndex+1) into a
# single int when ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX is requested;
# the low 24 bits are the body id. Background pixels are -1 and must be
# excluded *before* masking (masking -1 itself yields a bogus positive
# number, not a real body id).
BODY_ID_MASK = (1 << 24) - 1


class PyBulletWristCamera:
    def __init__(self, client_id: int, robot_id: int, config_path: Optional[Union[str, Path]] = None):
        self.client_id = client_id
        self.robot_id = robot_id

        config = dict(DEFAULT_CONFIG)
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as config_file:
                    config.update(json.load(config_file))
            else:
                print(f"Wrist camera config not found: {config_path}. Using built-in defaults.")

        self.camera_name = config["camera_name"]
        self.end_effector_link_index = int(config["end_effector_link_index"])
        self.width = int(config["width"])
        self.height = int(config["height"])
        self.fov = float(config["fov"])
        self.near = float(config["near"])
        self.far = float(config["far"])
        self.camera_local_position = np.array(config["camera_local_position"], dtype=np.float64)
        self.camera_forward_local = np.array(config["camera_forward_local"], dtype=np.float64)
        self.camera_up_local = np.array(config["camera_up_local"], dtype=np.float64)
        self.save_depth_colormap = bool(config.get("save_depth_colormap", True))
        self.save_segmentation_mask = bool(config.get("save_segmentation_mask", True))

        self._last_camera_pose: Optional[dict] = None

    def _compute_camera_pose(self) -> dict:
        link_state = p.getLinkState(
            self.robot_id,
            self.end_effector_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        link_world_position = np.array(link_state[4], dtype=np.float64)
        link_world_orientation = link_state[5]

        rotation_matrix = np.array(
            p.getMatrixFromQuaternion(link_world_orientation), dtype=np.float64
        ).reshape(3, 3)

        forward_world = rotation_matrix @ self.camera_forward_local
        forward_world = forward_world / np.linalg.norm(forward_world)

        up_hint_world = rotation_matrix @ self.camera_up_local
        right_world = np.cross(forward_world, up_hint_world)
        right_world = right_world / np.linalg.norm(right_world)
        up_world = np.cross(right_world, forward_world)

        camera_world_position = link_world_position + rotation_matrix @ self.camera_local_position
        camera_target_world_position = camera_world_position + forward_world

        return {
            "camera_world_position": camera_world_position,
            "camera_target_world_position": camera_target_world_position,
            "camera_up_world": up_world,
            "forward_world": forward_world,
            "right_world": right_world,
        }

    def render(self) -> Tuple[dict, dict]:
        pose = self._compute_camera_pose()
        self._last_camera_pose = pose

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pose["camera_world_position"].tolist(),
            cameraTargetPosition=pose["camera_target_world_position"].tolist(),
            cameraUpVector=pose["camera_up_world"].tolist(),
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
        )

        width, height, rgba_pixels, depth_buffer, segmentation = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.client_id,
        )

        rgba_array = np.array(rgba_pixels, dtype=np.uint8).reshape((height, width, 4))
        rgb = rgba_array[:, :, :3]

        depth_buffer_array = np.array(depth_buffer, dtype=np.float64).reshape((height, width))
        depth_meters = (self.near * self.far) / (
            self.far - (self.far - self.near) * depth_buffer_array
        )

        segmentation_array = np.array(segmentation, dtype=np.int64).reshape((height, width))

        frame = {
            "rgb": rgb,
            "depth_buffer": depth_buffer_array,
            "depth_meters": depth_meters,
            "segmentation": segmentation_array,
        }

        debug = {
            "camera_name": self.camera_name,
            "camera_world_position": pose["camera_world_position"].tolist(),
            "camera_target_world_position": pose["camera_target_world_position"].tolist(),
            "camera_up_world": pose["camera_up_world"].tolist(),
            "view_matrix": list(view_matrix),
            "projection_matrix": list(projection_matrix),
            "width": self.width,
            "height": self.height,
            "fov": self.fov,
            "near": self.near,
            "far": self.far,
        }

        return frame, debug

    def estimate_object_position_from_segmentation(
        self, frame: dict, object_body_id: int
    ) -> Tuple[Optional[list], dict]:
        segmentation = frame["segmentation"]
        depth_meters = frame["depth_meters"]

        visible_pixels = segmentation != -1
        body_ids = np.where(visible_pixels, segmentation & BODY_ID_MASK, -1)
        mask = body_ids == object_body_id

        object_pixel_count = int(mask.sum())
        if object_pixel_count == 0:
            return None, {
                "object_visible": False,
                "object_pixel_count": 0,
                "object_bbox_px": None,
                "object_center_px": None,
                "object_depth_median": None,
                "estimated_world_position": None,
            }

        ys, xs = np.nonzero(mask)
        bbox_px = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        center_px = [(bbox_px[0] + bbox_px[2]) / 2.0, (bbox_px[1] + bbox_px[3]) / 2.0]
        depth_median = float(np.median(depth_meters[mask]))

        if self._last_camera_pose is None:
            # render() must run before this -- keep it non-fatal since a
            # caller could reasonably call estimate_* right after render()
            # in the same frame, but not before it at all.
            raise RuntimeError("estimate_object_position_from_segmentation() called before render().")

        estimated_world_position = self._unproject_pixel(center_px[0], center_px[1], depth_median)

        debug = {
            "object_visible": True,
            "object_pixel_count": object_pixel_count,
            "object_bbox_px": bbox_px,
            "object_center_px": center_px,
            "object_depth_median": depth_median,
            "estimated_world_position": estimated_world_position,
        }
        return estimated_world_position, debug

    def _unproject_pixel(self, u: float, v: float, depth_value: float) -> list:
        """Unprojects an (u, v) pixel with a known along-view-axis depth
        (in meters, e.g. from depth_meters) back to a world-space point,
        using the camera's own basis vectors and vertical FOV rather than
        inverting the full projection matrix -- simpler and exact for
        this pinhole/no-distortion model.
        """
        pose = self._last_camera_pose
        forward_world = pose["forward_world"]
        right_world = pose["right_world"]
        up_world = pose["camera_up_world"]
        camera_world_position = pose["camera_world_position"]

        fov_y_rad = math.radians(self.fov)
        tan_half_fov_y = math.tan(fov_y_rad / 2.0)
        tan_half_fov_x = tan_half_fov_y * (self.width / self.height)

        ndc_x = 2.0 * (u + 0.5) / self.width - 1.0
        ndc_y = 1.0 - 2.0 * (v + 0.5) / self.height

        x_cam = ndc_x * tan_half_fov_x * depth_value
        y_cam = ndc_y * tan_half_fov_y * depth_value

        world_point = (
            camera_world_position
            + depth_value * forward_world
            + x_cam * right_world
            + y_cam * up_world
        )
        return [float(world_point[0]), float(world_point[1]), float(world_point[2])]


def save_wrist_camera_outputs(
    frame: dict,
    debug: dict,
    output_dir: Union[str, Path],
    timestamp: Optional[str] = None,
    save_depth_colormap: bool = True,
    save_segmentation_mask: bool = True,
    extra_debug: Optional[dict] = None,
) -> dict:
    """Saves rgb/depth/segmentation PNGs + a debug JSON to output_dir,
    named wrist_{rgb,depth,seg,debug}_<timestamp>.{png,json}."""
    from datetime import datetime

    from data_collection.trajectory_recorder import to_jsonable
    from robot_sim.camera_utils import save_rgb_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_paths = {}

    rgb_path = output_dir / f"wrist_rgb_{timestamp}.png"
    save_rgb_image(frame["rgb"], str(rgb_path))
    saved_paths["rgb"] = str(rgb_path)

    if save_depth_colormap:
        depth = frame["depth_meters"]
        depth_max = float(np.percentile(depth, 99)) if depth.size else 1.0
        depth_clipped = np.clip(depth, 0.0, max(depth_max, 1e-6))
        depth_norm = depth_clipped / max(depth_clipped.max(), 1e-6)
        depth_gray = (depth_norm * 255).astype(np.uint8)
        depth_rgb = np.stack([depth_gray] * 3, axis=-1)
        depth_path = output_dir / f"wrist_depth_{timestamp}.png"
        save_rgb_image(depth_rgb, str(depth_path))
        saved_paths["depth"] = str(depth_path)

    if save_segmentation_mask:
        segmentation = frame["segmentation"]
        visible_pixels = segmentation != -1
        body_ids = np.where(visible_pixels, segmentation & BODY_ID_MASK, -1)
        unique_ids = np.unique(body_ids)
        rng = np.random.default_rng(42)
        seg_vis = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for body_id in unique_ids:
            color = (0, 0, 0) if body_id == -1 else tuple(int(c) for c in rng.integers(50, 255, size=3))
            seg_vis[body_ids == body_id] = color
        seg_path = output_dir / f"wrist_seg_{timestamp}.png"
        save_rgb_image(seg_vis, str(seg_path))
        saved_paths["segmentation"] = str(seg_path)

    debug_payload = dict(debug)
    if extra_debug:
        debug_payload.update(extra_debug)
    debug_path = output_dir / f"wrist_debug_{timestamp}.json"
    with open(debug_path, "w", encoding="utf-8") as debug_file:
        json.dump(to_jsonable(debug_payload), debug_file, indent=2)
    saved_paths["debug"] = str(debug_path)

    return saved_paths


DEFAULT_MIN_OBJECT_PIXELS = 50
DEFAULT_MAX_REFINEMENT_DELTA = 0.08


def refine_target_with_wrist_camera(
    backend,
    wrist_camera: "PyBulletWristCamera",
    current_target_position: list,
    object_body_id: int,
    mode: str = "blend",
    blend_alpha: float = 0.7,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    max_refinement_delta: float = DEFAULT_MAX_REFINEMENT_DELTA,
    frame: Optional[dict] = None,
) -> Tuple[list, dict]:
    """Renders the wrist camera from wherever the end effector currently
    is (the caller is expected to only call this once it's already near
    the object -- see run_full_recycling_cell_demo.py's refine trigger),
    estimates the object's position from segmentation+depth, and either
    corrects current_target_position toward that estimate or falls back
    to it unchanged if the estimate doesn't pass the trust checks below.

    `backend` isn't used directly (wrist_camera already carries its own
    client_id/robot_id) -- kept in the signature for symmetry with the
    rest of this module's helpers and in case a future revision needs it
    (e.g. re-querying object velocity for a moving-object case).

    `frame`: pass an already-rendered wrist_camera.render() result (e.g.
    from this same control-loop step's --policy-observation-source wrist
    render) to skip rendering again; renders fresh when omitted.

    mode:
      none      always falls back (observe-only, no correction)
      blend     refined_xy = (1-alpha)*coarse_xy + alpha*wrist_xy
      override  refined_xy = wrist_xy

    In both blend and override, z is left as current_target_position's
    z (not the wrist estimate's) -- the wrist depth-based z estimate is
    the least reliable axis here (a single median-depth sample, no
    multi-view or filtering), while the existing object_z is already a
    known constant from the Real2Sim mapping.
    """
    if frame is None:
        frame, _render_debug = wrist_camera.render()
    estimated_position, estimate_debug = wrist_camera.estimate_object_position_from_segmentation(
        frame, object_body_id
    )

    xy_delta_from_coarse = None
    if estimated_position is not None:
        xy_delta_from_coarse = math.sqrt(
            (estimated_position[0] - current_target_position[0]) ** 2
            + (estimated_position[1] - current_target_position[1]) ** 2
        )

    debug = {
        "coarse_target_position": list(current_target_position),
        "wrist_estimated_position": estimated_position,
        "refinement_policy": mode,
        "blend_alpha": blend_alpha,
        "object_visible": estimate_debug["object_visible"],
        "object_pixel_count": estimate_debug["object_pixel_count"],
        "xy_delta_from_coarse": xy_delta_from_coarse,
        "refinement_applied": False,
        "refined_target_position": list(current_target_position),
    }

    if mode == "none":
        debug["fallback_reason"] = "policy_none"
    elif not estimate_debug["object_visible"]:
        debug["fallback_reason"] = "object_not_visible"
    elif estimate_debug["object_pixel_count"] < min_object_pixels:
        debug["fallback_reason"] = "too_few_pixels"
    elif xy_delta_from_coarse is None:
        debug["fallback_reason"] = "no_estimate"
    elif xy_delta_from_coarse > max_refinement_delta:
        debug["fallback_reason"] = "delta_too_large"
    elif mode not in ("blend", "override"):
        debug["fallback_reason"] = f"unknown_policy:{mode}"
    else:
        if mode == "override":
            refined_xy = [estimated_position[0], estimated_position[1]]
        else:
            refined_xy = [
                (1.0 - blend_alpha) * current_target_position[0] + blend_alpha * estimated_position[0],
                (1.0 - blend_alpha) * current_target_position[1] + blend_alpha * estimated_position[1],
            ]
        refined_target_position = [refined_xy[0], refined_xy[1], current_target_position[2]]
        debug["refinement_applied"] = True
        debug["refined_target_position"] = refined_target_position

    return debug["refined_target_position"], debug


def print_wrist_refinement_debug(debug: dict) -> None:
    print("=== Wrist Camera Grasp Refinement ===")
    print(f"coarse_target_position: {debug['coarse_target_position']}")
    print(f"wrist_estimated_position: {debug['wrist_estimated_position']}")
    print(f"refinement_policy: {debug['refinement_policy']}")
    print(f"blend_alpha: {debug['blend_alpha']}")
    print(f"xy_delta_from_coarse: {debug['xy_delta_from_coarse']}")
    print(f"object_visible: {debug['object_visible']}")
    print(f"object_pixel_count: {debug['object_pixel_count']}")
    print(f"refinement_applied: {debug['refinement_applied']}")
    print(f"refined_target_position: {debug['refined_target_position']}")
    if "fallback_reason" in debug:
        print(f"fallback_reason: {debug['fallback_reason']}")


def build_wrist_observation_metadata(step_index: int, frame: dict, render_debug: dict, estimate_debug: dict) -> dict:
    """Per-step VLA-ready observation record: reuses render()'s and
    estimate_object_position_from_segmentation()'s outputs directly (no
    separate vision logic here) so a control loop can attach a summary
    of "what the wrist camera saw this step" to PolicyInput/episode
    recording without recomputing anything.
    """
    return {
        "step_index": step_index,
        "observation_source": "wrist",
        "image_shape": list(frame["rgb"].shape),
        "object_visible": estimate_debug["object_visible"],
        "object_pixel_count": estimate_debug["object_pixel_count"],
        "object_bbox_px": estimate_debug["object_bbox_px"],
        "object_center_px": estimate_debug["object_center_px"],
        "estimated_world_position": estimate_debug["estimated_world_position"],
        "camera_world_position": render_debug["camera_world_position"],
    }
