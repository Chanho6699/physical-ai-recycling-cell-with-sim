"""ArUco marker PNG generator (v0).

  cv2.aruco.getPredefinedDictionary(dictionary)
  -> cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
  -> save PNG per marker id

Standalone and offline -- no camera, no YOLO, no PyBullet. Print the
generated PNGs and tape them to the four corners of the table's usable
work area before running benchmark/probe_aruco_real2sim_mapping.py or
run_full_recycling_cell_demo.py --real2sim-mode aruco.
"""

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "results/aruco_markers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    parser.add_argument("--marker-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--marker-size-px", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    if not hasattr(cv2, "aruco"):
        print("cv2.aruco is not available. Install opencv-contrib-python in the active venv.")
        print("FAIL")
        return

    if not hasattr(cv2.aruco, args.dictionary):
        print(f"Unknown ArUco dictionary: {args.dictionary}")
        print("FAIL")
        return

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary_slug = args.dictionary.lower().replace("dict_", "aruco_")

    for marker_id in args.marker_ids:
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, args.marker_size_px)
        output_path = output_dir / f"{dictionary_slug}_id_{marker_id}.png"
        cv2.imwrite(str(output_path), marker_image)
        print(f"{output_path}")

    print("PASS")


if __name__ == "__main__":
    main()
