"""Webcam probe script (v0).

Confirms a real webcam is reachable and returns usable frames before
wiring it into run_full_recycling_cell_demo.py. Deliberately standalone
(no YOLO, no PyBullet, no Real2Sim) so it can be run first on its own to
debug camera access issues -- WSL in particular often doesn't expose
/dev/video0 out of the box.

  WebcamSource.warmup() -> WebcamSource.get_frame() -> (optional) save

No hand detection, no MediaPipe, no safety interrupt, no OpenVLA, no
FastAPI server here -- just confirms the camera itself works.
"""

import argparse
from datetime import datetime
from pathlib import Path

from robot_sim.camera_utils import save_rgb_image
from vision.webcam_source import WebcamSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "results/webcam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-url", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--save-frame", action="store_true")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def print_open_failure_hint(args: argparse.Namespace) -> None:
    if args.camera_url:
        print(f"Could not open camera URL: {args.camera_url}")
        print(
            "Check that the relay server (e.g. camera_stream_server.py on Windows) is "
            "running and reachable from WSL at that address."
        )
        return

    print(f"Could not open webcam index {args.camera_index}.")
    print(
        f"If running under WSL, check whether /dev/video{args.camera_index} is available "
        "or run from a Windows Python environment."
    )
    print("Try a different --camera-index (for example 0 or 1) as well.")


def main() -> None:
    args = parse_args()

    if args.camera_url:
        print(f"camera_url={args.camera_url}")
    else:
        print(f"camera_index={args.camera_index}")

    source = None
    try:
        source = WebcamSource(camera_index=args.camera_index, camera_url=args.camera_url)
    except RuntimeError:
        print("opened=False")
        print_open_failure_hint(args)
        print("FAIL")
        return

    print("opened=True")

    try:
        # num_frames covers the warmup reads *and* the final frame that
        # gets reported/saved below, so a fresh camera settles by the
        # time we actually look at a frame.
        source.warmup(max(args.num_frames - 1, 0))
        frame = source.get_frame()
    except RuntimeError as exc:
        print(f"error={exc}")
        print_open_failure_hint(args)
        print("FAIL")
        return
    finally:
        source.close()

    print(f"frame_shape={frame.shape}")

    saved_frame = None
    if args.save_frame:
        output_dir = resolve(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"webcam_probe_{timestamp}.jpg"
        saved_frame = save_rgb_image(frame, str(output_path))

    print(f"saved_frame={saved_frame}")
    print("PASS")


if __name__ == "__main__":
    main()
