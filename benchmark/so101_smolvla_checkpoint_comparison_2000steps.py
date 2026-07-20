"""SO-101 SmolVLA 2000-step checkpoint comparison (see this task's chat
report: "30 episode 데이터로 충분히 반복 학습했을 때, 모델이 제한된
fixed-bin 환경에서 실제 grasp를 형성하는가?"). Evaluates checkpoints at
steps 500/1000/2000 (resumed from the existing 200-step sanity
checkpoint -- same train/validation split, seed=0,
freeze_vision_encoder=True, train_expert_only=True, use_amp=False,
batch_size=1, no LoRA) against the SAME validation seeds [0, 3, 7] used
for the 200-step rollout.

Reuses (does NOT reimplement):
  - benchmark.so101_smolvla_checkpoint_inference_eval's own
    load_policy_and_processors()/offline_prediction_evaluation().
  - benchmark.so101_smolvla_rollout's own run_one_rollout() (now with
    min object-gripper distance + clamp count tracking added).

Does NOT retrain here (training itself run via a separate lerobot-train
resume invocation, see this task's chat report item 1), does NOT touch
expert/backend/dataset schema/recorder, does NOT apply any additional
analysis beyond the requested metrics (per this task's own "지나친
추가 분석은 하지 않는다").

Run:
  .venv-vla/bin/python -m benchmark.so101_smolvla_checkpoint_comparison_2000steps
"""

import json
import re
from pathlib import Path

from benchmark.analyze_so101_bin_pilot_dataset import load_frames
from benchmark.so101_smolvla_checkpoint_inference_eval import DATASET_ROOT, SPLIT_PATH, load_policy_and_processors, offline_prediction_evaluation
from benchmark.so101_smolvla_rollout import NUM_ROLLOUT_SEEDS, run_one_rollout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOG_PATH = Path("/tmp/claude-1000/-home-rlack-Projects-physical-ai-recycling-cell/4a365940-468e-4fc4-af6b-44ca46439f30/scratchpad/train_2000.log")
CHECKPOINTS_ROOT = PROJECT_ROOT / "outputs" / "train" / "so101_smolvla_sanity_v0" / "checkpoints"
CHECKPOINT_STEPS = [500, 1000, 2000]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "checkpoint_comparison_2000steps.json"
PREVIOUS_200STEP_ROLLOUT_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "rollout_results.json"
PREVIOUS_200STEP_OFFLINE_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "offline_predictions.json"


def _parse_step_token(token: str) -> int:
    """lerobot's own trainer log abbreviates round-thousand steps as
    e.g. "1K"/"2K" (see ot_train.py's own step:{step} formatting) --
    handled explicitly here, not just a bare \\d+ regex, which silently
    truncated "2K" to step 2 in an earlier version of this function."""
    if token.endswith("K"):
        return int(float(token[:-1]) * 1000)
    return int(token)


def extract_loss_log(step_target: int) -> dict:
    """Parses the training log for the loss/grad_norm trajectory up to
    (and including) step_target -- read-only, no re-training."""
    text = TRAIN_LOG_PATH.read_text()
    steps = [_parse_step_token(x) for x in re.findall(r"step:(\d+\.?\d*K?)", text)]
    losses = [float(x) for x in re.findall(r"loss:([0-9.]+)", text)]
    pairs = [(s, l) for s, l in zip(steps, losses) if s <= step_target]
    return {
        "loss_at_last_logged_step": pairs[-1][1] if pairs else None,
        "last_logged_step": pairs[-1][0] if pairs else None,
        "min_loss_up_to_step": min(l for _, l in pairs) if pairs else None,
        "loss_log_up_to_step": pairs,
    }


def evaluate_checkpoint(step: int) -> dict:
    checkpoint_dir = CHECKPOINTS_ROOT / f"{step:06d}" / "pretrained_model"
    print(f"=== Evaluating checkpoint step {step} ({checkpoint_dir}) ===")

    split = json.loads(SPLIT_PATH.read_text())
    val_episodes = split["validation_episodes"]
    frames = load_frames(DATASET_ROOT)

    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint_dir)

    offline = offline_prediction_evaluation(policy, preprocessor, postprocessor, frames, val_episodes)
    arm_joint_mae = {k: v for k, v in offline["joint_mae"].items() if k != "gripper"}

    rollout_seeds = val_episodes[:NUM_ROLLOUT_SEEDS]
    rollout_results = []
    for seed in rollout_seeds:
        r = run_one_rollout(policy, preprocessor, postprocessor, seed)
        rollout_results.append(r)
        print(f"  seed {seed}: grasp_ever={r['grasp_was_ever_established']} place_success={r['model_rollout_place_success']} "
              f"min_obj_gripper_dist={r['min_object_gripper_distance_m']:.4f} clamp_count={r['joint_limit_clamp_count']} "
              f"aborted={r['aborted_early']}")

    return {
        "checkpoint_step": step, "checkpoint_dir": str(checkpoint_dir),
        "loss_trajectory": extract_loss_log(step),
        "arm_joint_mae": arm_joint_mae,
        "arm_joint_mae_mean": sum(arm_joint_mae.values()) / len(arm_joint_mae),
        "gripper_open_close_accuracy": offline["gripper_open_close_accuracy"],
        "constant_output_across_validation_set": offline["constant_output_across_validation_set"],
        "rollout_results": rollout_results,
        "grasp_established_count": sum(1 for r in rollout_results if r["grasp_was_ever_established"]),
        "place_success_count": sum(1 for r in rollout_results if r["model_rollout_place_success"]),
        "any_nan_inf": any(r["failure_reason"] and "nan" in r["failure_reason"] for r in rollout_results),
        "total_clamp_count": sum(r["joint_limit_clamp_count"] for r in rollout_results),
        "min_object_gripper_distance_m_per_seed": {r["seed"]: r["min_object_gripper_distance_m"] for r in rollout_results},
    }


def main() -> None:
    previous_200step_rollout = json.loads(PREVIOUS_200STEP_ROLLOUT_PATH.read_text())
    previous_200step_offline = json.loads(PREVIOUS_200STEP_OFFLINE_PATH.read_text())
    baseline_arm_mae = {k: v for k, v in previous_200step_offline["joint_mae"].items() if k != "gripper"}
    baseline = {
        "checkpoint_step": 200,
        "arm_joint_mae": baseline_arm_mae,
        "arm_joint_mae_mean": sum(baseline_arm_mae.values()) / len(baseline_arm_mae),
        "gripper_open_close_accuracy": previous_200step_offline["gripper_open_close_accuracy"],
        "grasp_established_count": previous_200step_rollout["grasp_established_count"],
        "place_success_count": previous_200step_rollout["success_count"],
    }

    checkpoint_results = {}
    for step in CHECKPOINT_STEPS:
        checkpoint_results[step] = evaluate_checkpoint(step)

    grasp_improved = any(checkpoint_results[s]["grasp_established_count"] > 0 for s in CHECKPOINT_STEPS)
    arm_mae_trend = [baseline["arm_joint_mae_mean"]] + [checkpoint_results[s]["arm_joint_mae_mean"] for s in CHECKPOINT_STEPS]
    arm_mae_improving = arm_mae_trend[-1] < arm_mae_trend[0]
    loss_improving = checkpoint_results[2000]["loss_trajectory"]["min_loss_up_to_step"] < checkpoint_results[500]["loss_trajectory"]["min_loss_up_to_step"]

    if grasp_improved:
        conclusion = "grasp_formed_or_clearly_improved -- pipeline performance verified, recommend proceeding to final-environment real data collection"
    elif loss_improving and arm_mae_improving:
        conclusion = "loss_and_approach_still_improving -- recommend extending to 5000 steps with the same settings"
    else:
        conclusion = "loss_decreasing_only_no_behavioral_improvement -- recommend a LIMITED check of action chunking/inference loop/data representation"

    summary = {
        "baseline_200steps": baseline,
        "checkpoints": {str(s): checkpoint_results[s] for s in CHECKPOINT_STEPS},
        "grasp_established_at_any_checkpoint": grasp_improved,
        "arm_joint_mae_trend_200_to_2000": arm_mae_trend,
        "arm_joint_mae_improving": arm_mae_improving,
        "loss_improving_500_to_2000": loss_improving,
        "conclusion": conclusion,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=== 2000-step checkpoint comparison summary ===")
    print(f"arm_joint_mae_mean trend (200->500->1000->2000): {arm_mae_trend}")
    print(f"grasp_established_at_any_checkpoint: {grasp_improved}")
    print(f"conclusion: {conclusion}")
    print(f"\nResults JSON: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
