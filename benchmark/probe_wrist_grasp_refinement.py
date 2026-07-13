"""Wrist camera grasp-refinement probe (v1).

Confirms refine_target_with_wrist_camera() actually corrects a
deliberately-offset "coarse" target back toward the true object
position -- entirely inside PyBullet, no external camera, no YOLO, no
ArUco, no online control loop. The offset stands in for the kind of
error a real Real2Sim mapping (ROI/ArUco) can leave uncorrected.

  object placed at object_position (ground truth)
  -> coarse_target = object_position + coarse_offset
  -> move end-effector to just above coarse_target (like move_to_object
     would have gotten it, before the wrist-camera refine trigger fires)
  -> refine_target_with_wrist_camera()
  -> compare error_before_xy (coarse vs. ground truth) against
     error_after_xy (refined vs. ground truth)
"""

import argparse
import math
import time
from pathlib import Path

from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from robot_sim.pybullet_wrist_camera import (
    PyBulletWristCamera,
    refine_target_with_wrist_camera,
    save_wrist_camera_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRIST_CAMERA_CONFIG = "configs/wrist_camera_config.json"
DEFAULT_OUTPUT_DIR = "results/wrist_camera"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-position", type=float, nargs=3, default=[0.40, -0.10, 0.05])
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--coarse-offset", type=float, nargs=2, default=[0.04, -0.03])
    parser.add_argument("--policy", choices=["none", "blend", "override"], default="blend")
    parser.add_argument("--blend-alpha", type=float, default=0.7)
    parser.add_argument("--min-object-pixels", type=int, default=50)
    parser.add_argument("--max-refinement-delta", type=float, default=0.08)
    # 0.15, not the ~0.08 the real refine trigger uses: the coarse_offset
    # pushes the object off to the side of the wrist camera's view, and
    # at only 0.08 depth a default-sized coarse_offset sits right at the
    # edge of the FOV, where visibility becomes fragile/inconsistent
    # (tiny IK/orientation differences flip a pixel or two of the object
    # in or out of frame). 0.15 keeps it comfortably inside the frustum.
    parser.add_argument("--approach-height", type=float, default=0.15)

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--wrist-camera-config", type=str, default=DEFAULT_WRIST_CAMERA_CONFIG)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    object_position = list(args.object_position)
    coarse_target_position = [
        object_position[0] + args.coarse_offset[0],
        object_position[1] + args.coarse_offset[1],
        object_position[2],
    ]

    print("=== Wrist Grasp Refinement Probe ===")
    print(f"gt_object_position: {object_position}")
    print(f"coarse_target_position: {coarse_target_position}")

    backend = PyBulletPandaBackend(gui=args.gui)
    try:
        try:
            backend.reset()
        except Exception as exc:
            print(f"Panda backend reset failed: {exc}")
            print("FAIL")
            return

        backend.set_object_type(args.object_type)
        backend.set_object_position(object_position)

        approach_position = [
            coarse_target_position[0],
            coarse_target_position[1],
            coarse_target_position[2] + args.approach_height,
        ]
        backend.move_end_effector_to(approach_position)

        wrist_camera = PyBulletWristCamera(
            client_id=backend.client_id,
            robot_id=backend.robot_id,
            config_path=resolve(args.wrist_camera_config),
        )

        refined_target_position, refinement_debug = refine_target_with_wrist_camera(
            backend,
            wrist_camera,
            coarse_target_position,
            backend._object_id,
            mode=args.policy,
            blend_alpha=args.blend_alpha,
            min_object_pixels=args.min_object_pixels,
            max_refinement_delta=args.max_refinement_delta,
        )

        print(f"wrist_estimated_position: {refinement_debug['wrist_estimated_position']}")
        print(f"refined_target_position: {refined_target_position}")

        error_before_xy = math.sqrt(
            (coarse_target_position[0] - object_position[0]) ** 2
            + (coarse_target_position[1] - object_position[1]) ** 2
        )
        error_after_xy = math.sqrt(
            (refined_target_position[0] - object_position[0]) ** 2
            + (refined_target_position[1] - object_position[1]) ** 2
        )
        print(f"error_before_xy: {error_before_xy:.4f}")
        print(f"error_after_xy: {error_after_xy:.4f}")
        print(f"refinement_applied: {refinement_debug['refinement_applied']}")
        if "fallback_reason" in refinement_debug:
            print(f"fallback_reason: {refinement_debug['fallback_reason']}")

        if args.save_images:
            frame, render_debug = wrist_camera.render()
            saved_paths = save_wrist_camera_outputs(
                frame,
                {**render_debug, **refinement_debug},
                resolve(args.output_dir),
                save_depth_colormap=wrist_camera.save_depth_colormap,
                save_segmentation_mask=wrist_camera.save_segmentation_mask,
                extra_debug={
                    "object_position_gt": object_position,
                    "error_before_xy": error_before_xy,
                    "error_after_xy": error_after_xy,
                },
            )
            print(f"Saved wrist camera outputs: {saved_paths}")

        success = (
            refinement_debug["object_visible"]
            and refinement_debug["refinement_applied"]
            and error_after_xy < error_before_xy
        )
        print("PASS" if success else "FAIL")

        if KEEP_GUI_OPEN and args.gui:
            print(f"Keeping PyBullet GUI open (up to {KEEP_SECONDS}s if no input)...")
            try:
                input("Press Enter to close PyBullet GUI...")
            except EOFError:
                time.sleep(KEEP_SECONDS)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
