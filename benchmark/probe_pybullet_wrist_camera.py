"""PyBullet Eye-in-Hand wrist camera probe (v0).

PyBulletPandaBackend only -- no external camera, no YOLO, no ArUco.
Confirms the wrist camera (rigidly attached to the end-effector link)
can see and estimate the position of an object placed directly in the
sim, before wiring it into run_full_recycling_cell_demo.py's
--wrist-camera-mode.

  PyBulletPandaBackend.reset() -> set_object_position()
  -> move_end_effector_to(above the object)
  -> PyBulletWristCamera.render()
  -> estimate_object_position_from_segmentation()
  -> compare estimated vs ground-truth object_position
"""

import argparse
import math
import time
from pathlib import Path

from robot_sim.pybullet_panda_backend import PyBulletPandaBackend
from robot_sim.pybullet_wrist_camera import PyBulletWristCamera, save_wrist_camera_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRIST_CAMERA_CONFIG = "configs/wrist_camera_config.json"
DEFAULT_OUTPUT_DIR = "results/wrist_camera"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-position", type=float, nargs=3, default=[0.40, -0.10, 0.05])
    parser.add_argument("--object-type", type=str, default="plastic_bottle")
    parser.add_argument("--observe-height", type=float, default=0.25)
    parser.add_argument("--position-error-threshold", type=float, default=0.05)

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

    print("=== Wrist Camera Probe ===")
    print(f"object_position_gt: {object_position}")

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

        observe_position = [
            object_position[0],
            object_position[1],
            object_position[2] + args.observe_height,
        ]
        backend.move_end_effector_to(observe_position)

        wrist_camera = PyBulletWristCamera(
            client_id=backend.client_id,
            robot_id=backend.robot_id,
            config_path=resolve(args.wrist_camera_config),
        )
        frame, render_debug = wrist_camera.render()
        estimated_position, estimate_debug = wrist_camera.estimate_object_position_from_segmentation(
            frame, backend._object_id
        )

        print(f"object_visible: {estimate_debug['object_visible']}")

        error_xy = None
        if estimate_debug["object_visible"]:
            print(f"object_pixel_count: {estimate_debug['object_pixel_count']}")
            print(f"object_center_px: {estimate_debug['object_center_px']}")
            print(f"estimated_world_position: {estimated_position}")

            error_xy = math.sqrt(
                (estimated_position[0] - object_position[0]) ** 2
                + (estimated_position[1] - object_position[1]) ** 2
            )
            error_3d = math.sqrt(
                sum((estimated_position[axis] - object_position[axis]) ** 2 for axis in range(3))
            )
            print(f"position_error_xy: {error_xy:.4f}")
            print(f"position_error_3d: {error_3d:.4f}")
        else:
            print("Wrist camera could not see the object from the observe position.")

        if args.save_images:
            saved_paths = save_wrist_camera_outputs(
                frame,
                {**render_debug, **estimate_debug},
                resolve(args.output_dir),
                save_depth_colormap=wrist_camera.save_depth_colormap,
                save_segmentation_mask=wrist_camera.save_segmentation_mask,
                extra_debug={"object_position_gt": object_position},
            )
            print(f"Saved wrist camera outputs: {saved_paths}")

        success = (
            estimate_debug["object_visible"]
            and error_xy is not None
            and error_xy <= args.position_error_threshold
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
