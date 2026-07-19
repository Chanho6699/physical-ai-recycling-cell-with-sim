"""SO-101 scripted expert small-randomization evaluation (see this
task's chat report, "고정 큐브 위치를 작은 범위에서 랜덤화했을 때
scripted expert의 반복 안정성 평가"). Purely a PERFORMANCE MEASUREMENT
script -- does NOT collect or save any LeRobotDataset, does NOT touch
benchmark/collect_so101_episode.py, and does NOT modify
benchmark/so101_scripted_expert.py's waypoints/thresholds/step limits.

For each seed, samples a small (default +-1cm) x/y offset around the
backend's own default object position (z and yaw untouched, target
zone and initial joint pose untouched -- see this task's own scope
limits), runs
benchmark/so101_scripted_expert.py::run_pick_and_place_episode()
UNCHANGED, and records success/failure diagnostics. Each episode gets
its own fresh So101PyBulletBackend instance (created and closed within
the episode) so no physics state carries over between seeds.

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_expert_small_randomization \\
    --num-episodes 20 --seed-start 0
"""

import argparse
import json
import math
import random
from pathlib import Path

from benchmark.so101_scripted_expert import So101ExpertError, run_pick_and_place_episode
from robot_sim.so101_pybullet_backend import DEFAULT_OBJECT_POSITION, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = "results/so101_expert_small_randomization.json"
DEFAULT_NUM_EPISODES = 20
DEFAULT_SEED_START = 0
DEFAULT_X_RANGE = (-0.01, 0.01)
DEFAULT_Y_RANGE = (-0.01, 0.01)

# Same TRANSPORT_DELTA_XY as smoke_so101_pick_and_place.py's own
# positive case / collect_so101_episode.py -- unchanged, not retuned.
TRANSPORT_DELTA_XY = [0.05, 0.05]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_range(range_str: str) -> tuple:
    low_str, high_str = range_str.split(",")
    return (float(low_str), float(high_str))


def sample_object_position(seed: int, x_range: tuple, y_range: tuple) -> list:
    """Deterministic: random.Random(seed) is a fresh generator seeded
    ONLY by `seed` -- no shared/global RNG state, no dependency on call
    order or how many episodes ran before it. The SAME seed always
    yields the SAME (x_offset, y_offset). z is left at the backend's
    own default object height (not touched); yaw is not sampled at
    all (this task explicitly excludes yaw randomization)."""
    rng = random.Random(seed)
    x_offset = rng.uniform(x_range[0], x_range[1])
    y_offset = rng.uniform(y_range[0], y_range[1])
    return [
        DEFAULT_OBJECT_POSITION[0] + x_offset,
        DEFAULT_OBJECT_POSITION[1] + y_offset,
        DEFAULT_OBJECT_POSITION[2],
    ]


def run_single_episode(seed: int, x_range: tuple, y_range: tuple) -> dict:
    sampled_object_position = sample_object_position(seed, x_range, y_range)

    backend = So101PyBulletBackend(gui=False, object_position=sampled_object_position)
    try:
        backend.reset()
        # Settled position (after reset()'s own OBJECT_SETTLE_STEPS) --
        # the real position the episode actually ran against, same
        # convention benchmark/collect_so101_episode.py's manifest uses.
        settled_object_position, _ = backend.get_object_pose()
        target_zone_center_xy = backend.get_scene_state()["target_zone_center_xy"]

        success = False
        failure_reason = None
        failure_phase = None
        released = False
        place_success = False
        move_step_count = None
        object_xy_error_from_target_m = None
        linear_speed = angular_speed = recent_drift = None

        try:
            result = run_pick_and_place_episode(backend, TRANSPORT_DELTA_XY)
            released = result["release_constraint_removed"]
            place_success = result["place_success"]
            failure_reason = result["failure_reason"]
            failure_phase = result.get("failure_phase")
            object_xy_error_from_target_m = result["object_target_xy_error_m"]
            linear_speed = result["object_final_linear_speed_mps"]
            angular_speed = result["object_final_angular_speed_radps"]
            recent_drift = result["object_recent_settle_drift_m"]
            move_step_count = sum(
                result[k]["num_steps"] for k in ("pre_grasp", "approach", "lift", "transport", "place_descend")
            )
            success = place_success
        except So101ExpertError as exc:
            failure_reason = exc.failure_reason
            failure_phase = exc.phase

        # Always capture the actual final object position -- even on a
        # mid-episode exception, the object/backend are still alive
        # (not yet closed), so this is real, not fabricated.
        final_object_position = backend.get_object_position()
        if object_xy_error_from_target_m is None:
            object_xy_error_from_target_m = math.sqrt(
                (final_object_position[0] - target_zone_center_xy[0]) ** 2
                + (final_object_position[1] - target_zone_center_xy[1]) ** 2
            )

        return {
            "seed": seed,
            "object_position": settled_object_position,
            "success": success,
            "failure_reason": failure_reason,
            "failure_phase": failure_phase,
            "released": released,
            "place_success": place_success,
            "move_step_count": move_step_count,
            "final_object_position": final_object_position,
            "target_zone_center_xy": target_zone_center_xy,
            "object_xy_error_from_target_m": object_xy_error_from_target_m,
            "object_final_linear_speed_mps": linear_speed,
            "object_final_angular_speed_radps": angular_speed,
            "object_recent_settle_drift_m": recent_drift,
        }
    finally:
        backend.close()


def summarize(episodes: list) -> dict:
    total = len(episodes)
    successes = [ep for ep in episodes if ep["success"]]
    failures = [ep for ep in episodes if not ep["success"]]

    failure_reason_counts = {}
    failure_phase_counts = {}
    for ep in failures:
        failure_reason_counts[ep["failure_reason"]] = failure_reason_counts.get(ep["failure_reason"], 0) + 1
        failure_phase_counts[ep["failure_phase"]] = failure_phase_counts.get(ep["failure_phase"], 0) + 1

    def xy_range(subset):
        if not subset:
            return None
        xs = [ep["object_position"][0] for ep in subset]
        ys = [ep["object_position"][1] for ep in subset]
        return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}

    return {
        "total_episodes": total,
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": (len(successes) / total) if total else None,
        "failure_reason_counts": failure_reason_counts,
        "failure_phase_counts": failure_phase_counts,
        "failed_seeds": [ep["seed"] for ep in failures],
        "successful_object_position_range": xy_range(successes),
        "sampled_object_position_range": xy_range(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--x-range", type=str, default=f"{DEFAULT_X_RANGE[0]},{DEFAULT_X_RANGE[1]}")
    parser.add_argument("--y-range", type=str, default=f"{DEFAULT_Y_RANGE[0]},{DEFAULT_Y_RANGE[1]}")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    x_range = parse_range(args.x_range)
    y_range = parse_range(args.y_range)

    episodes = []
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        episode_result = run_single_episode(seed, x_range, y_range)
        episodes.append(episode_result)
        status = "SUCCESS" if episode_result["success"] else f"FAIL ({episode_result['failure_reason']} @ {episode_result['failure_phase']})"
        print(f"[seed {seed}] {status} -- object_position={episode_result['object_position']}")

    summary = summarize(episodes)

    output = {
        "config": {
            "num_episodes": args.num_episodes, "seed_start": args.seed_start,
            "x_range": list(x_range), "y_range": list(y_range),
            "transport_delta_xy": TRANSPORT_DELTA_XY,
        },
        "episodes": episodes,
        "summary": summary,
    }

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== SO-101 expert small-randomization evaluation summary ===")
    print(f"total_episodes: {summary['total_episodes']}")
    print(f"success_count: {summary['success_count']}")
    print(f"failure_count: {summary['failure_count']}")
    print(f"success_rate: {summary['success_rate']:.2%}" if summary["success_rate"] is not None else "success_rate: N/A")
    print(f"failure_reason_counts: {summary['failure_reason_counts']}")
    print(f"failure_phase_counts: {summary['failure_phase_counts']}")
    print(f"failed_seeds: {summary['failed_seeds']}")
    print(f"successful_object_position_range: {summary['successful_object_position_range']}")
    print(f"sampled_object_position_range: {summary['sampled_object_position_range']}")
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
