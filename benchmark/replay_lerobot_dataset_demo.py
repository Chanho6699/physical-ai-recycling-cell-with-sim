"""Dataset Replay Validator (v0).

Reads datasets/<dataset>/{meta,data}/episodes.jsonl and replays each
episode's exported (delta_ee_position, gripper_action) action sequence on
PyBulletPandaBackend, to check whether the exported action labels are
actually executable and lead back to a successful placement -- not just
whether the *original* recording succeeded.

No OpenVLA fine-tuning, no LeRobot official parquet/video conversion, no
Hugging Face Hub upload, no ROS 2, no TensorRT, no Isaac Sim, no VLA
policy inference here yet.
"""

import argparse
import json
import math
from pathlib import Path

from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="datasets/lerobot_recycling_cell_v0")
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)

    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", dest="gui", action="store_true")
    gui_group.add_argument("--headless", dest="gui", action="store_false")
    parser.set_defaults(gui=True)

    parser.add_argument("--steps-per-action", type=int, default=10)
    parser.add_argument("--gripper-hold-policy", choices=["keep", "ignore"], default="keep")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--final-distance-threshold", type=float, default=0.08)
    parser.add_argument(
        "--report-path",
        type=str,
        default="results/replay/lerobot_recycling_cell_v0_replay_report.json",
    )
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def distance(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def replay_episode(episode_id: str, samples: list, meta: dict, args: argparse.Namespace):
    metadata = meta.get("metadata") or {}
    mapped_sim_position = metadata.get("mapped_sim_position")

    if mapped_sim_position is None:
        print(f"Episode {episode_id}: metadata.mapped_sim_position missing -- cannot replay, skipping.")
        print("(Re-export the dataset with the metadata-carrying exporter to fix this.)")
        return None

    object_type = metadata.get("object_type")
    bin_position = metadata.get("bin_position")

    backend = PyBulletPandaBackend(gui=args.gui)
    try:
        state = backend.reset()

        if object_type:
            backend.set_object_type(object_type)
        state = backend.set_object_position(mapped_sim_position)

        if bin_position:
            state = backend.set_bin_position(bin_position)

        expected_bin_position = bin_position if bin_position is not None else state["bin_position"]

        ordered_samples = sorted(samples, key=lambda s: s.get("timestamp_index", s.get("frame_index", 0)))

        for sample in ordered_samples:
            action = sample.get("action") or {}
            delta = action.get("delta_ee_position", [0.0, 0.0, 0.0])
            gripper_action = action.get("gripper_action", "hold")

            current_ee = backend.get_state()["end_effector_position"]
            target = [current_ee[axis] + delta[axis] * args.position_scale for axis in range(3)]

            state = backend.move_end_effector_to(target, steps=args.steps_per_action)

            if gripper_action == "close":
                # Only actually call close/open when the gripper isn't
                # already in that state -- calling every sample would be
                # wasteful and, more importantly, close_gripper()/
                # open_gripper() each run a full simulation step loop.
                if state["gripper_state"] != "close":
                    state = backend.close_gripper()
            elif gripper_action == "open":
                if state["gripper_state"] != "open":
                    state = backend.open_gripper()
            elif args.gripper_hold_policy == "keep":
                # Re-issue the current gripper command so PyBullet keeps
                # actively driving it (and so grasp/place conditions get
                # re-evaluated in case motion drifted the arm into range).
                if state["gripper_state"] == "close":
                    state = backend.close_gripper()
                else:
                    state = backend.open_gripper()
            # else "ignore": do nothing for hold samples.

        final_state = backend.get_state()
    finally:
        backend.close()

    final_object_position = final_state["object_position"]
    final_distance_to_bin = distance(final_object_position, expected_bin_position)
    final_task_status = final_state["task_status"]
    replay_success = (final_task_status == "success") or (final_distance_to_bin <= args.final_distance_threshold)

    return {
        "episode_id": episode_id,
        "expected_status": meta.get("status", "unknown"),
        "replay_status": final_task_status,
        "final_object_position": final_object_position,
        "expected_bin_position": expected_bin_position,
        "final_distance_to_bin": final_distance_to_bin,
        "replay_success": replay_success,
        "num_samples_replayed": len(ordered_samples),
    }


def main() -> None:
    args = parse_args()

    dataset_dir = resolve(args.dataset_dir)
    meta_path = dataset_dir / "meta" / "episodes.jsonl"
    data_path = dataset_dir / "data" / "episodes.jsonl"

    if not meta_path.exists() or not data_path.exists():
        print(f"Dataset not found or incomplete at: {dataset_dir}")
        print("Run benchmark.export_lerobot_dataset_demo first.")
        return

    episode_meta_list = read_jsonl(meta_path)
    episode_meta = {m["episode_id"]: m for m in episode_meta_list}

    data_samples = read_jsonl(data_path)
    episode_samples: dict = {}
    for sample in data_samples:
        episode_samples.setdefault(sample["episode_id"], []).append(sample)

    if args.episode_id:
        if args.episode_id not in episode_meta:
            print(f"Episode not found in dataset meta: {args.episode_id!r}")
            print(f"Available episode ids: {list(episode_meta.keys())}")
            return
        target_episode_ids = [args.episode_id]
    else:
        target_episode_ids = list(episode_meta.keys())

    episodes_seen = len(target_episode_ids)
    if args.max_episodes is not None:
        target_episode_ids = target_episode_ids[: args.max_episodes]

    episodes_replayed = 0
    episodes_skipped = 0
    replay_success_count = 0
    episode_reports = []

    for episode_id in target_episode_ids:
        meta = episode_meta[episode_id]
        samples = episode_samples.get(episode_id, [])

        if not samples:
            print(f"Episode {episode_id}: no samples found in data/episodes.jsonl -- skipping.")
            episodes_skipped += 1
            continue

        print(f"\n=== Replaying episode: {episode_id} ===")
        result = replay_episode(episode_id, samples, meta, args)

        if result is None:
            episodes_skipped += 1
            continue

        episodes_replayed += 1
        if result["replay_success"]:
            replay_success_count += 1

        print(f"Episode {episode_id}")
        print(f"expected_status={result['expected_status']}")
        print(f"replay_status={result['replay_status']}")
        print(f"final_distance_to_bin={result['final_distance_to_bin']:.4f}")
        print(f"replay_success={result['replay_success']}")

        episode_reports.append(result)

    replay_success_rate = (replay_success_count / episodes_replayed) if episodes_replayed > 0 else 0.0

    print("\n=== Replay Summary ===")
    print(f"episodes_seen: {episodes_seen}")
    print(f"episodes_replayed: {episodes_replayed}")
    print(f"episodes_skipped: {episodes_skipped}")
    print(f"replay_success_count: {replay_success_count}")
    print(f"replay_success_rate: {replay_success_rate:.2f}")

    report = {
        "dataset_dir": str(dataset_dir),
        "episodes_seen": episodes_seen,
        "episodes_replayed": episodes_replayed,
        "episodes_skipped": episodes_skipped,
        "replay_success_count": replay_success_count,
        "replay_success_rate": replay_success_rate,
        "episodes": episode_reports,
    }

    report_path = resolve(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved replay report to: {report_path}")


if __name__ == "__main__":
    main()
