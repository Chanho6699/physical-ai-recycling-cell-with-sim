"""One-off re-evaluation of the SO-101 SmolVLA 2000-step checkpoint on
validation seeds [0, 3, 7] AFTER the action-chunk consumption fix in
benchmark/so101_smolvla_rollout.py (see this task's chat report,
"SmolVLA action chunk 실행 방식... 확인"). Reuses run_one_rollout()
unmodified in call signature -- only its own internal per-step
predict_action_in_rollout() call path changed. Does NOT retrain.

Run:
  .venv-vla/bin/python -m benchmark.so101_smolvla_rollout_chunk_fix_reeval
"""

import json
from pathlib import Path

from benchmark.so101_smolvla_checkpoint_inference_eval import CHECKPOINT_DIR, load_policy_and_processors
from benchmark.so101_smolvla_rollout import run_one_rollout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "rollout_results_2000step_chunk_fixed.json"
SEEDS = [0, 3, 7]


def main() -> None:
    policy, preprocessor, postprocessor = load_policy_and_processors(CHECKPOINT_DIR)

    results = []
    for seed in SEEDS:
        print(f"=== Rollout seed {seed} (chunk-fixed, 2000-step checkpoint) ===")
        r = run_one_rollout(policy, preprocessor, postprocessor, seed)
        results.append(r)
        print(f"  steps_executed={r['steps_executed']} grasp_ever={r['grasp_was_ever_established']} "
              f"place_success={r['model_rollout_place_success']} min_obj_gripper_dist={r['min_object_gripper_distance_m']:.4f} "
              f"clamp_count={r['joint_limit_clamp_count']} aborted={r['aborted_early']} failure_reason={r['failure_reason']}")

    summary = {
        "checkpoint": str(CHECKPOINT_DIR),
        "seeds": SEEDS,
        "results": results,
        "grasp_established_count": sum(1 for r in results if r["grasp_was_ever_established"]),
        "place_success_count": sum(1 for r in results if r["model_rollout_place_success"]),
        "any_nan_inf": any(r["failure_reason"] and "nan" in r["failure_reason"] for r in results),
        "total_clamp_count": sum(r["joint_limit_clamp_count"] for r in results),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"grasp_established_count: {summary['grasp_established_count']}/3")
    print(f"place_success_count: {summary['place_success_count']}/3")
    print(f"\nResults JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
