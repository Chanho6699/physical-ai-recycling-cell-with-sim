"""LeRobot-compatible JSONL dataset export CLI (v0).

  datasets/raw_episodes/episode_.../episode.json
  -> LeRobotDatasetExporter
  -> datasets/lerobot_recycling_cell_v0/{meta,data,videos_or_frames}/

No real HuggingFace Hub upload, no OpenVLA fine-tuning, no ROS 2, no
TensorRT, no Isaac Sim, no VLA policy here yet.
"""

import argparse
from pathlib import Path

from data_collection.lerobot_dataset_exporter import LeRobotDatasetExporter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-episodes-dir", type=str, default="datasets/raw_episodes")
    parser.add_argument("--output-dir", type=str, default="datasets/lerobot_recycling_cell_v0")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    raw_episodes_dir = resolve(args.raw_episodes_dir)
    output_dir = resolve(args.output_dir)

    if not raw_episodes_dir.exists():
        print(f"raw episodes directory not found: {raw_episodes_dir}")
        print("Record at least one episode first, e.g.:")
        print("  python -m benchmark.run_record_panda_episode_demo --image-path ... --headless")
        return

    exporter = LeRobotDatasetExporter()
    result = exporter.export(
        raw_episodes_dir=str(raw_episodes_dir),
        output_dir=str(output_dir),
        include_failed=args.include_failed,
        copy_images=not args.no_copy_images,
        max_episodes=args.max_episodes,
    )

    print("Export finished")
    print(f"raw_episodes_seen: {result['raw_episodes_seen']}")
    print(f"episodes_exported: {result['episodes_exported']}")
    print(f"episodes_skipped: {result['episodes_skipped']}")
    print(f"samples_exported: {result['samples_exported']}")
    print(f"output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
