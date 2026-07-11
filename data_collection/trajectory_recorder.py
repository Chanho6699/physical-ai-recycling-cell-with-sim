"""Raw trajectory recorder for Panda manipulation episodes (v0).

Saves one folder per episode under an output directory:

  <output_dir>/episode_<timestamp>_<suffix>/
      episode.json
      frames/frame_000000.png ...

This is NOT a LeRobotDataset export -- just a stable internal raw format
to build on later. No OpenVLA action vector is synthesized here; a future
exporter can compute one from adjacent robot_state entries, e.g.:

  action_t = ee_position[t+1] - ee_position[t]
  gripper_action = gripper_width[t+1] < gripper_width[t]
"""

import json
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from robot_sim.camera_utils import save_rgb_image

DEFAULT_OUTPUT_DIR = "datasets/raw_episodes"


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays, dataclasses, tuples, etc.
    into plain JSON-serializable Python types.

    numpy.generic is checked *before* the native bool/int/float/str check
    because numpy.float64 (unlike int64/bool_) is actually a subclass of
    Python's built-in float -- checking native types first would let it
    slip through unconverted.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return str(obj)


class TrajectoryRecorder:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

        self._episode_id: Optional[str] = None
        self._episode_dir: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        self._instruction: Optional[str] = None
        self._task_goal = None
        self._metadata: dict = {}
        self._steps: list = []
        self._frame_index = 0

    def start_episode(self, instruction: str, task_goal: dict, metadata: Optional[dict] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        self._episode_id = f"episode_{timestamp}_{suffix}"

        self._episode_dir = self.output_dir / self._episode_id
        self._frames_dir = self._episode_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

        self._instruction = instruction
        self._task_goal = to_jsonable(task_goal)
        self._metadata = to_jsonable(metadata) if metadata is not None else {}
        self._steps = []
        self._frame_index = 0

        return self._episode_id

    def record_step(
        self,
        phase: str,
        action_name: str,
        robot_state: dict,
        action: Optional[dict] = None,
        safety: Optional[dict] = None,
        image=None,
        extra: Optional[dict] = None,
    ) -> None:
        if self._episode_dir is None:
            raise RuntimeError("start_episode() must be called before record_step().")

        image_path = None
        if image is not None:
            frame_name = f"frame_{self._frame_index:06d}.png"
            save_rgb_image(np.asarray(image), str(self._frames_dir / frame_name))
            image_path = f"frames/{frame_name}"
            self._frame_index += 1

        self._steps.append(
            {
                "step_index": len(self._steps),
                "phase": phase,
                "action_name": action_name,
                "robot_state": to_jsonable(robot_state),
                "action": to_jsonable(action) if action is not None else {},
                "safety": to_jsonable(safety) if safety is not None else {},
                "image_path": image_path,
                "extra": to_jsonable(extra) if extra is not None else {},
            }
        )

    def finish_episode(self, final_state: dict, success: bool, status: str, extra: Optional[dict] = None) -> dict:
        if self._episode_dir is None:
            raise RuntimeError("start_episode() must be called before finish_episode().")

        episode_record = {
            "episode_id": self._episode_id,
            "instruction": self._instruction,
            "task_goal": self._task_goal,
            "metadata": self._metadata,
            "steps": self._steps,
            "final_state": to_jsonable(final_state),
            "success": bool(success),
            "status": status,
            "num_steps": len(self._steps),
            "extra": to_jsonable(extra) if extra is not None else {},
        }

        episode_path = self._episode_dir / "episode.json"
        with episode_path.open("w", encoding="utf-8") as f:
            json.dump(episode_record, f, ensure_ascii=False, indent=2)

        episode_record["episode_dir"] = str(self._episode_dir)
        episode_record["episode_path"] = str(episode_path)
        return episode_record
