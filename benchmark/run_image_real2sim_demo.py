"""Image-file-based Real2Sim demo (v0).

  image file
  -> ONNXYOLODetector
  -> recyclable object detection (bottle/cup)
  -> bbox center
  -> ImageToSimMapper (image coords -> approximate PyBullet table coords)
  -> PyBulletBackend.set_object_position()
  -> pick-and-place command sequence built dynamically from the mapped
     object_position (NOT the fixed STEP_SEQUENCE from
     run_pybullet_pick_place_demo.py, which assumes the object always
     spawns at [0.5, 0.0, 0.53] and would miss the grasp otherwise)

No real webcam, no precise Real2Sim calibration, no OpenVLA, no ROS 2,
no TensorRT, no YOLO training, no latency benchmark here yet.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from action_adapter.adapter_v0 import RobotCommand
from perception.detection_types import Detection
from perception.onnx_yolo_detector import ONNXYOLODetector
from real2sim.image_to_sim_mapper import ImageToSimMapper
from real2sim.recyclable_object_mapper import RecyclableObjectMapper
from robot_sim.camera_utils import save_rgb_image
from robot_sim.pybullet_backend import PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE_PATH = PROJECT_ROOT / "results" / "camera" / "image_real2sim_debug.png"

KEEP_GUI_OPEN = True
KEEP_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="weights/yolo26n.onnx")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--save-debug-image", action="store_true")
    parser.add_argument("--gui", action="store_true", default=True)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_move_command(current_position: list, target_position: list, gripper: str) -> RobotCommand:
    dx = target_position[0] - current_position[0]
    dy = target_position[1] - current_position[1]
    dz = target_position[2] - current_position[2]

    return RobotCommand(
        target_dx=dx,
        target_dy=dy,
        target_dz=dz,
        target_droll=0.0,
        target_dpitch=0.0,
        target_dyaw=0.0,
        gripper_command=gripper,
    )


def draw_debug_image(frame: np.ndarray, detection: Detection, sim_position: list) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    x1, y1, x2, y2 = detection.bbox_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    center_x, center_y = detection.center_xy
    radius = 5
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=(255, 255, 0),
    )

    label_text = f"{detection.label} {detection.confidence:.2f}"
    sim_text = f"sim_pos=({sim_position[0]:.2f}, {sim_position[1]:.2f}, {sim_position[2]:.2f})"
    draw.text((x1, max(y1 - 24, 0)), label_text, fill=(255, 0, 0))
    draw.text((x1, min(y2 + 4, image.height - 12)), sim_text, fill=(255, 0, 0))

    return np.array(image)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        print("Check --image-path and try again.")
        return

    model_path = resolve(args.model_path)
    if not model_path.exists():
        print(f"ONNX model not found: {model_path}")
        print("Run tools/export_yolo_to_onnx.py first to create it.")
        return

    frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    print(f"frame shape: {frame.shape}")
    print(f"frame dtype: {frame.dtype}")

    detector = ONNXYOLODetector(model_path=str(model_path), confidence_threshold=args.confidence_threshold)
    detections = detector.detect(frame)
    print(f"detections: {detections}")

    if not detections:
        print("No detections found in the image. Try a lower --confidence-threshold or a different image.")
        return

    recyclable_mapper = RecyclableObjectMapper()
    best = recyclable_mapper.select_best_recyclable(detections)
    if best is None:
        print("No bottle/cup candidates found among the detections.")
        return

    detection, sim_object_type = best
    print(
        f"Selected recyclable candidate: {detection.label} "
        f"(confidence={detection.confidence:.2f}) -> {sim_object_type}"
    )

    image_height, image_width = frame.shape[:2]
    sim_mapper = ImageToSimMapper(image_width=image_width, image_height=image_height)
    center_x, center_y = detection.center_xy
    sim_position = sim_mapper.image_point_to_sim_position(center_x, center_y)
    print(f"Mapped sim position: {sim_position}")

    backend = PyBulletBackend(gui=args.gui)
    try:
        state = backend.reset()
        print("=== Reset State ===")
        print(state)

        backend.set_object_type(sim_object_type)
        state = backend.set_object_position(sim_position)
        print("=== State After Real2Sim Mapping ===")
        print(state)

        if args.save_debug_image:
            debug_image = draw_debug_image(frame, detection, sim_position)
            saved_path = save_rgb_image(debug_image, str(DEBUG_IMAGE_PATH))
            print(f"Saved debug image to: {saved_path}")

        # Dynamic command sequence built from the *current* state at each
        # step, not the fixed STEP_SEQUENCE from run_pybullet_pick_place_demo.py
        # (which assumes the object always spawns at [0.5, 0.0, 0.53]).
        print("\n=== Step 1: approach_real2sim_object ===")
        state_before = backend.get_state()
        command = make_move_command(
            state_before["end_effector_position"], state_before["object_position"], gripper="open"
        )
        state_after = backend.apply_command(command)
        print(f"state_after: {state_after}")

        print("\n=== Step 2: grasp_real2sim_object ===")
        state_before = backend.get_state()
        ee_pos = state_before["end_effector_position"]
        command = make_move_command(ee_pos, ee_pos, gripper="close")
        state_after = backend.apply_command(command)
        print(f"state_after: {state_after}")

        print("\n=== Step 3: carry_real2sim_object_to_bin ===")
        state_before = backend.get_state()
        command = make_move_command(
            state_before["end_effector_position"], state_before["bin_position"], gripper="close"
        )
        state_after = backend.apply_command(command)
        print(f"state_after: {state_after}")

        print("\n=== Step 4: place_real2sim_object ===")
        state_before = backend.get_state()
        ee_pos = state_before["end_effector_position"]
        command = make_move_command(ee_pos, ee_pos, gripper="open")
        state_after = backend.apply_command(command)
        print(f"state_after: {state_after}")

        final_status = state_after["task_status"]
        print(f"\n=== Demo finished: task_status={final_status} ===")

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
