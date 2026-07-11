"""LeRobot-style raw episode -> JSONL dataset exporter (v0).

Converts datasets/raw_episodes/episode_.../episode.json (produced by
TrajectoryRecorder) into a LeRobot-compatible-*shaped* local JSONL
dataset:

  <output_dir>/
      meta/info.json
      meta/episodes.jsonl
      data/episodes.jsonl
      videos_or_frames/episode_.../frame_XXXXXX.png

No real HuggingFace/LeRobot library dependency and no parquet yet -- just
a locally inspectable JSONL structure to build an actual LeRobotDataset
export on top of later. Each output sample's action is derived from a
pair of adjacent raw steps (there is no recorded OpenVLA-style action
vector in the raw episode), so an episode with N raw steps yields N-1
samples.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

# Real Panda finger open/close swings ~0.038m (0.08 <-> ~0.04, minus
# object width). Once the gripper is holding something, PyBullet's
# position-controlled fingers settle with noise up to ~1e-3m step to
# step (measured empirically) -- a threshold of 1e-4 misreads that noise
# as spurious "open"/"close" events, which breaks replay (the object
# gets released mid-carry). 0.005 sits comfortably between the noise
# ceiling and a real transition.
GRIPPER_ACTION_THRESHOLD = 0.005

OBSERVATION_KEYS = [
    "image_path",
    "joint_positions",
    "joint_velocities",
    "end_effector_position",
    "end_effector_orientation",
    "gripper_width",
]
ACTION_KEYS = ["delta_ee_position", "gripper_action"]


def _gripper_action(current_width: float, next_width: float) -> str:
    delta = next_width - current_width
    if delta < -GRIPPER_ACTION_THRESHOLD:
        return "close"
    if delta > GRIPPER_ACTION_THRESHOLD:
        return "open"
    return "hold"


class LeRobotDatasetExporter:
    def export(
        self,
        raw_episodes_dir: str,
        output_dir: str,
        include_failed: bool = False,
        copy_images: bool = True,
        max_episodes: Optional[int] = None,
    ) -> dict:
        raw_root = Path(raw_episodes_dir)
        output_root = Path(output_dir)

        meta_dir = output_root / "meta"
        data_dir = output_root / "data"
        frames_root = output_root / "videos_or_frames"
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        frames_root.mkdir(parents=True, exist_ok=True)

        episode_dirs = sorted(
            p for p in raw_root.iterdir() if p.is_dir() and (p / "episode.json").exists()
        ) if raw_root.exists() else []

        raw_episodes_seen = len(episode_dirs)
        episodes_exported = 0
        episodes_skipped = 0
        samples_exported = 0

        episode_summaries = []
        data_samples = []

        for episode_dir in episode_dirs:
            if max_episodes is not None and episodes_exported >= max_episodes:
                break

            with (episode_dir / "episode.json").open("r", encoding="utf-8") as f:
                episode = json.load(f)

            success = bool(episode.get("success", False))
            status = episode.get("status", "unknown")

            if not include_failed and not (success and status == "success"):
                episodes_skipped += 1
                continue

            episode_id = episode["episode_id"]
            instruction = episode.get("instruction", "")
            task_goal = episode.get("task_goal") or {}
            final_state = episode.get("final_state") or {}
            raw_metadata = episode.get("metadata") or {}
            steps = sorted(episode.get("steps", []), key=lambda s: s.get("step_index", 0))

            episode_frames_dir = frames_root / episode_id
            if copy_images:
                episode_frames_dir.mkdir(parents=True, exist_ok=True)
                # Copy every step image that exists, regardless of whether
                # it ends up referenced by an output sample below -- the
                # last raw step's image is only ever used as the "next"
                # state for the final sample's action, never as its own
                # observation.
                for step in steps:
                    raw_image_path = step.get("image_path")
                    if raw_image_path:
                        source_path = episode_dir / raw_image_path
                        if source_path.exists():
                            shutil.copy2(source_path, episode_frames_dir / Path(raw_image_path).name)

            failure_reason = None if success else final_state.get("last_event")

            num_samples_this_episode = 0
            for i in range(len(steps) - 1):
                current_step = steps[i]
                next_step = steps[i + 1]

                current_state = current_step.get("robot_state") or {}
                next_state = next_step.get("robot_state") or {}

                current_ee = current_state.get("end_effector_position", [0.0, 0.0, 0.0])
                next_ee = next_state.get("end_effector_position", current_ee)
                delta_ee_position = [next_ee[axis] - current_ee[axis] for axis in range(3)]

                current_width = current_state.get("gripper_width", 0.0)
                next_width = next_state.get("gripper_width", current_width)

                image_path = None
                raw_image_path = current_step.get("image_path")
                if raw_image_path:
                    image_path = f"videos_or_frames/{episode_id}/{Path(raw_image_path).name}"

                sample = {
                    "episode_id": episode_id,
                    "frame_index": current_step.get("step_index", i),
                    "timestamp_index": i,
                    "instruction": instruction,
                    "observation": {
                        "image_path": image_path,
                        "state": {
                            "joint_positions": current_state.get("joint_positions", []),
                            "joint_velocities": current_state.get("joint_velocities", []),
                            "end_effector_position": current_state.get("end_effector_position", []),
                            "end_effector_orientation": current_state.get("end_effector_orientation", []),
                            "gripper_width": current_state.get("gripper_width"),
                        },
                    },
                    "action": {
                        "delta_ee_position": delta_ee_position,
                        "gripper_action": _gripper_action(current_width, next_width),
                    },
                    "task": {
                        "target_object": task_goal.get("target_object"),
                        "target_bin": task_goal.get("target_bin"),
                    },
                    "success": success,
                    "status": status,
                }
                if not success:
                    sample["failure_reason"] = failure_reason

                data_samples.append(sample)
                num_samples_this_episode += 1

            episode_summaries.append(
                {
                    "episode_id": episode_id,
                    "success": success,
                    "status": status,
                    "num_raw_steps": len(steps),
                    "num_samples": num_samples_this_episode,
                    "instruction": instruction,
                    # Carried through from the raw episode so a replay
                    # script can re-initialize the sim (object_type,
                    # mapped_sim_position, bin_position) without needing
                    # to re-run detection/Real2Sim mapping. Purely
                    # additive -- existing consumers that only read the
                    # other keys are unaffected.
                    "metadata": raw_metadata,
                }
            )

            episodes_exported += 1
            samples_exported += num_samples_this_episode

        with (data_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for sample in data_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for summary in episode_summaries:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        info = {
            "dataset_name": output_root.name,
            "format": "lerobot_compatible_jsonl_v0",
            "robot_type": "franka_panda_pybullet",
            "observation_keys": OBSERVATION_KEYS,
            "action_keys": ACTION_KEYS,
            "num_episodes": episodes_exported,
            "num_samples": samples_exported,
            "include_failed": include_failed,
        }
        with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        return {
            "raw_episodes_seen": raw_episodes_seen,
            "episodes_exported": episodes_exported,
            "episodes_skipped": episodes_skipped,
            "samples_exported": samples_exported,
            "output_dir": str(output_root),
        }
