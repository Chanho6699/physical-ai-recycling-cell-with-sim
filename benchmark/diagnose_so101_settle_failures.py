"""SO-101 settle diagnosis (see this task's chat report, "settle
진단 스크립트가 새로운 판정 방식과 충돌하지 않는지 확인"). Does NOT
change the expert's waypoints, step counts, gripper values, or ANY
threshold.

Under the OLD fixed-360-step single-instant check, this script used to
extend the observation window (`extra_settle_steps`) to ask "would this
have passed given more time?" That parameter no longer exists --
run_pick_and_place_episode()'s new CONTINUOUS-STABILITY settle judgment
(see benchmark/so101_scripted_expert.py) already runs up to
MAX_SETTLE_STEPS by default and answers that exact question natively
via "settle_success"/"settle_timeout"/"settle_steps_used"/
"max_consecutive_stable_steps" -- no special diagnostic call is needed
anymore for the basic A-vs-B question this script originally existed
to answer. This script is kept (updated, not deleted) to report those
native fields for a fixed list of seeds, optionally with the full
per-step trace (`record_settle_trace=True`) for extra detail.

Reuses benchmark.evaluate_so101_expert_small_randomization's own
sample_object_position() (same seed -> same object position, not
reimplemented).

Run:
  .venv-vla/bin/python -m benchmark.diagnose_so101_settle_failures
"""

import argparse
import json
from pathlib import Path

from benchmark.evaluate_so101_expert_small_randomization import (
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    TRANSPORT_DELTA_XY,
    sample_object_position,
)
from benchmark.so101_scripted_expert import So101ExpertError, run_pick_and_place_episode
from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = "results/so101_settle_diagnosis.json"

# These seeds were the settle_failed / comparison set under the OLD
# fixed-360-step judgment (see this task's chat report's own prior
# turn) -- kept as the same fixed seed list for continuity/comparability.
PREVIOUSLY_FAILED_SEEDS = [1, 5, 14, 15, 18, 19]
COMPARISON_SUCCESS_SEEDS = [2, 3, 4]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def diagnose_seed(seed: int, record_trace: bool) -> dict:
    object_position = sample_object_position(seed, DEFAULT_X_RANGE, DEFAULT_Y_RANGE)
    backend = So101PyBulletBackend(gui=False, object_position=object_position)
    try:
        backend.reset()
        settled_object_position, _ = backend.get_object_pose()

        try:
            result = run_pick_and_place_episode(backend, TRANSPORT_DELTA_XY, record_settle_trace=record_trace)
        except So101ExpertError as exc:
            return {
                "seed": seed, "object_position": settled_object_position, "pre_release_failure": True,
                "failure_reason": exc.failure_reason, "failure_phase": exc.phase,
            }

        return {
            "seed": seed,
            "object_position": settled_object_position,
            "pre_release_failure": False,
            "place_success": result["place_success"],
            "failure_reason": result["failure_reason"],
            "settle_success": result["settle_success"],
            "settle_timeout": result["settle_timeout"],
            "settle_steps_used": result["settle_steps_used"],
            "continuous_stable_steps_required": result["continuous_stable_steps_required"],
            "max_consecutive_stable_steps": result["max_consecutive_stable_steps"],
            "final_consecutive_stable_steps": result["final_consecutive_stable_steps"],
            "settle_check_count": result["settle_check_count"],
            "final_linear_speed": result["final_linear_speed"],
            "final_angular_speed": result["final_angular_speed"],
            "final_drift": result["final_drift"],
            "final_object_position": result["final_object_position"],
            "final_xy_error": result["final_xy_error"],
            "target_zone_center_xy": result["target_center_position"],
            "settle_trace": result["settle_trace"],
        }
    finally:
        backend.close()


def summarize(diagnoses: list) -> dict:
    with_data = [d for d in diagnoses if not d.get("pre_release_failure")]
    settled = [d for d in with_data if d["settle_success"]]
    timed_out = [d for d in with_data if d["settle_timeout"]]

    return {
        "settle_steps_used_by_seed": {str(d["seed"]): d["settle_steps_used"] for d in with_data},
        "settled_count": len(settled),
        "timed_out_count": len(timed_out),
        "timed_out_seeds": [d["seed"] for d in timed_out],
        "place_success_by_seed": {str(d["seed"]): d["place_success"] for d in with_data},
        "failure_reason_by_seed": {str(d["seed"]): d["failure_reason"] for d in with_data},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--record-trace", action="store_true", help="also record the full per-step settle_trace")
    args = parser.parse_args()

    seeds = PREVIOUSLY_FAILED_SEEDS + COMPARISON_SUCCESS_SEEDS
    diagnoses = []
    for seed in seeds:
        d = diagnose_seed(seed, args.record_trace)
        diagnoses.append(d)
        if d.get("pre_release_failure"):
            print(f"[seed {seed}] PRE-RELEASE FAILURE ({d['failure_reason']} @ {d['failure_phase']})")
        else:
            print(
                f"[seed {seed}] settle_success={d['settle_success']} settle_steps_used={d['settle_steps_used']} "
                f"place_success={d['place_success']} failure_reason={d['failure_reason']}"
            )

    summary = summarize(diagnoses)

    output = {
        "config": {"previously_failed_seeds": PREVIOUSLY_FAILED_SEEDS, "comparison_success_seeds": COMPARISON_SUCCESS_SEEDS},
        "diagnoses": diagnoses,
        "summary": summary,
    }

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== settle diagnosis summary (new continuous-stability judgment) ===")
    print(f"settle_steps_used_by_seed: {summary['settle_steps_used_by_seed']}")
    print(f"settled_count: {summary['settled_count']}  timed_out_count: {summary['timed_out_count']}")
    print(f"timed_out_seeds: {summary['timed_out_seeds']}")
    print(f"place_success_by_seed: {summary['place_success_by_seed']}")
    print(f"failure_reason_by_seed: {summary['failure_reason_by_seed']}")
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
