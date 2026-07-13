"""Recorded episode inspector (v0).

Reads back a raw episode saved by run_full_recycling_cell_demo.py
(--record) and prints a short summary of the whole perception-to-action
chain it captured -- external detection, Real2Sim mapping, wrist camera
refinement, robot execution, final result -- plus which step (if any)
the wrist camera refinement happened at.

Prefers the standalone metadata.json (see
data_collection/perception_episode_schema.py) written alongside
episode.json; falls back to episode.json's own "metadata" field for
older episodes recorded before metadata.json existed.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from data_collection.perception_episode_schema import load_episode_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=str, required=True)
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_wrist_refinement_step(episode: dict) -> Optional[dict]:
    for step in episode.get("steps", []):
        extra = step.get("extra") or {}
        if extra.get("event_type") == "wrist_refinement":
            return {"step_index": step["step_index"], "phase": step["phase"], **extra}
    return None


def main() -> None:
    args = parse_args()

    episode_dir = resolve(args.episode_dir)
    episode_path = episode_dir / "episode.json"
    if not episode_path.exists():
        print(f"episode.json not found: {episode_path}")
        print("Check --episode-dir and try again.")
        print("FAIL")
        return

    with open(episode_path, "r", encoding="utf-8") as episode_file:
        episode = json.load(episode_file)

    metadata = load_episode_metadata(episode_dir)
    if metadata is None:
        # Older episode recorded before metadata.json existed -- fall
        # back to whatever was passed into start_episode()'s metadata.
        metadata = episode.get("metadata") or {}

    print("=== Episode Summary ===")
    print(f"episode_id: {episode.get('episode_id')}")
    print(f"instruction: {episode.get('instruction')}")

    real2sim = metadata.get("real2sim", {})
    print(f"real2sim_mode: {real2sim.get('mode')}")
    print(f"mapped_position: {real2sim.get('mapped_position')}")
    if real2sim.get("mode") == "aruco":
        print(f"homography_valid: {real2sim.get('homography_valid')}")
        print(f"out_of_bounds: {real2sim.get('out_of_bounds')}")

    wrist_camera = metadata.get("wrist_camera", {})
    print(f"wrist_camera_mode: {wrist_camera.get('mode')}")
    if wrist_camera.get("mode") not in (None, "off"):
        print(f"wrist_refinement_attempted: {wrist_camera.get('refinement_attempted')}")
        print(f"wrist_refinement_applied: {wrist_camera.get('refinement_applied')}")

    refinement_step = find_wrist_refinement_step(episode)
    if refinement_step is not None:
        print(f"wrist_refinement_step_index: {refinement_step['step_index']}")

    robot = metadata.get("robot", {})
    policy_steps = robot.get("policy_steps", episode.get("num_steps"))
    final_status = robot.get("final_status", episode.get("status"))
    print(f"policy_steps: {policy_steps}")
    print(f"final_status: {final_status}")

    result = metadata.get("result", {})
    if result:
        print(f"success: {result.get('success')}")
        print(f"final_object_position: {result.get('final_object_position')}")
        print(f"bin_position: {result.get('bin_position')}")

    success = bool(result.get("success", episode.get("success")))
    print("PASS" if success else "FAIL")


if __name__ == "__main__":
    main()
