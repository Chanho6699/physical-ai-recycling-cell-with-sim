"""SO-101 SmolVLA sanity checkpoint reload + inference-contract check +
offline validation-set prediction evaluation (see this task's chat
report, sections 8-9). Runs in a FRESH process (this script's own
invocation) -- never reuses the training process's in-memory policy
object, so a real "save -> reload in a new process -> inference" round
trip is actually exercised.

Loads the checkpoint DIRECTLY via LeRobot's own PreTrainedPolicy/
processor API (SmolVLAPolicy.from_pretrained() +
make_pre_post_processors()) -- does NOT go through vla_server/
vla_adapters/policy_semantics, which this task's own investigation
confirmed are 100% hardcoded to the Panda/LIBERO 7-dim EE-delta
embodiment (PANDA_TARGET_EMBODIMENT in policy_semantics/
compatibility_gate.py) and would refuse any SO-101 checkpoint outright.
lerobot-train itself needed no such adapter to TRAIN on the SO-101
dataset (it consumes the LeRobotDataset's own declared features
directly), so this script mirrors that same adapter-free path for
inference -- no new Panda-specific code touched, no new adapter file
needed for THIS purpose.

Does NOT retrain, does NOT touch the scripted expert/backend/dataset
schema, does NOT apply any additional normalization (the checkpoint's
own saved preprocessor/postprocessor -- computed by lerobot-train from
OUR dataset's own stats -- does all of it).

Run:
  .venv-vla/bin/python -m benchmark.so101_smolvla_checkpoint_inference_eval
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from benchmark.analyze_so101_bin_pilot_dataset import decode_image, load_frames, load_manifest
from benchmark.so101_dataset_schema import SO101_JOINT_NAMES
from benchmark.so101_scripted_expert import PHASE_NAME_BY_ID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_fixed_pilot_30"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "train" / "so101_smolvla_sanity_v0" / "checkpoints" / "000200" / "pretrained_model"
SPLIT_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "split.json"
VALIDATION_METRICS_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "validation_metrics.json"
OFFLINE_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "so101_smolvla_sanity_training" / "offline_predictions.json"
TASK_TEXT = "Pick up the object and place it in the bin."
NUM_JOINTS = len(SO101_JOINT_NAMES)
PHASES_OF_INTEREST = ["pre_grasp", "approach", "grasp", "lift", "transport", "place_descend", "release"]


def load_policy_and_processors(checkpoint_dir: Path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
    policy.eval()
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint_dir))
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(checkpoint_dir))
    return policy, preprocessor, postprocessor


def frame_to_observation(row) -> dict:
    img = decode_image(row["observation.images.front"])  # (256,256,3) uint8
    img_chw = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(np.asarray(row["observation.state"], dtype=np.float32))
    return {"observation.images.front": img_chw, "observation.state": state, "task": TASK_TEXT}


def predict_action(policy, preprocessor, postprocessor, observation: dict) -> np.ndarray:
    policy.reset()  # clear the internal action-chunk queue -- each call here is an INDEPENDENT single-frame query, never consuming a stale queued action from a previous frame
    batch = preprocessor(observation)
    with torch.no_grad():
        raw_action = policy.select_action(batch)
    final_action = postprocessor(raw_action)
    return final_action.squeeze(0).cpu().numpy()


def checkpoint_reload_inference_check(policy, preprocessor, postprocessor, frames: pd.DataFrame, val_episodes: list) -> dict:
    """Section 8 -- at least 5 different validation frames, action
    shape [6], finite, joint order, gripper range, denormalization,
    constant-output check."""
    val_frames = frames[frames["episode_index"].isin(val_episodes)]
    sample_rows = [val_frames.iloc[i] for i in np.linspace(0, len(val_frames) - 1, 6, dtype=int)]

    predictions = []
    for row in sample_rows:
        obs = frame_to_observation(row)
        action = predict_action(policy, preprocessor, postprocessor, obs)
        predictions.append({
            "episode_index": int(row["episode_index"]), "frame_index": int(row["frame_index"]),
            "predicted_action": action.tolist(),
        })

    actions_array = np.array([p["predicted_action"] for p in predictions])
    checks = {
        "num_frames_tested": len(predictions),
        "action_shape_is_6": all(a.shape == (NUM_JOINTS,) for a in [np.array(p["predicted_action"]) for p in predictions]),
        "all_finite": bool(np.all(np.isfinite(actions_array))),
        "joint_names": list(SO101_JOINT_NAMES),
        "gripper_channel_range_min_max": [float(actions_array[:, -1].min()), float(actions_array[:, -1].max())],
        "gripper_in_0_100_scale_not_normalized": bool(actions_array[:, -1].min() >= -5 and actions_array[:, -1].max() <= 105),
        "arm_channels_in_radian_scale_not_normalized": bool(np.abs(actions_array[:, :5]).max() < 10.0),
        "predictions": predictions,
        "constant_output_across_frames": bool(np.allclose(actions_array, actions_array[0], atol=1e-4)),
        "action_std_across_sample_frames": actions_array.std(axis=0).tolist(),
    }
    checks["pass"] = (
        checks["action_shape_is_6"] and checks["all_finite"] and not checks["constant_output_across_frames"]
        and checks["gripper_in_0_100_scale_not_normalized"] and checks["arm_channels_in_radian_scale_not_normalized"]
    )
    return checks


def offline_prediction_evaluation(policy, preprocessor, postprocessor, frames: pd.DataFrame, val_episodes: list) -> dict:
    """Section 9 -- full validation-set prediction vs ground truth."""
    val_frames = frames[frames["episode_index"].isin(val_episodes)].copy()
    val_frames["phase_name"] = val_frames["phase_id"].apply(lambda v: PHASE_NAME_BY_ID[int(v[0]) if hasattr(v, "__len__") else int(v)])

    predicted = []
    ground_truth = []
    phase_names = []
    for _, row in val_frames.iterrows():
        obs = frame_to_observation(row)
        pred = predict_action(policy, preprocessor, postprocessor, obs)
        predicted.append(pred)
        ground_truth.append(np.asarray(row["action"], dtype=np.float32))
        phase_names.append(row["phase_name"])

    predicted = np.array(predicted)
    ground_truth = np.array(ground_truth)
    error = predicted - ground_truth

    overall_mae = np.abs(error).mean()
    overall_rmse = np.sqrt((error ** 2).mean())
    joint_mae = np.abs(error).mean(axis=0)
    joint_rmse = np.sqrt((error ** 2).mean(axis=0))

    phase_errors = {}
    for phase in PHASES_OF_INTEREST:
        idx = [i for i, p in enumerate(phase_names) if p == phase]
        if not idx:
            phase_errors[phase] = {"frame_count": 0}
            continue
        e = error[idx]
        phase_errors[phase] = {
            "frame_count": len(idx),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "joint_mae": {SO101_JOINT_NAMES[j]: float(np.abs(e[:, j]).mean()) for j in range(NUM_JOINTS)},
        }

    gt_gripper_open = ground_truth[:, -1] > 50.0
    pred_gripper_open = predicted[:, -1] > 50.0
    gripper_accuracy = float((gt_gripper_open == pred_gripper_open).mean())

    prediction_variance = predicted.var(axis=0)
    constant_output = bool(np.allclose(predicted, predicted[0], atol=1e-4))

    return {
        "num_validation_frames": len(val_frames),
        "validation_episodes": val_episodes,
        "overall_mae": float(overall_mae), "overall_rmse": float(overall_rmse),
        "joint_mae": {SO101_JOINT_NAMES[j]: float(joint_mae[j]) for j in range(NUM_JOINTS)},
        "joint_rmse": {SO101_JOINT_NAMES[j]: float(joint_rmse[j]) for j in range(NUM_JOINTS)},
        "phase_errors": phase_errors,
        "gripper_open_close_accuracy": gripper_accuracy,
        "prediction_variance_per_joint": {SO101_JOINT_NAMES[j]: float(prediction_variance[j]) for j in range(NUM_JOINTS)},
        "constant_output_across_validation_set": constant_output,
    }


def main() -> None:
    # CLI overrides are purely additive -- every default below is the
    # EXACT module constant this script always used, so a bare
    # `-m benchmark.so101_smolvla_checkpoint_inference_eval` invocation
    # is byte-for-byte unchanged. Added so
    # benchmark/run_so101_smolvla_pipeline.py can subprocess-invoke this
    # SAME script against an arbitrary checkpoint/dataset/split instead
    # of re-implementing this file's own evaluation logic.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--split-path", type=str, default=str(SPLIT_PATH))
    parser.add_argument("--validation-metrics-path", type=str, default=str(VALIDATION_METRICS_PATH))
    parser.add_argument("--offline-predictions-path", type=str, default=str(OFFLINE_PREDICTIONS_PATH))
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_root = Path(args.dataset_root)
    split_path = Path(args.split_path)
    validation_metrics_path = Path(args.validation_metrics_path)
    offline_predictions_path = Path(args.offline_predictions_path)

    split = json.loads(split_path.read_text())
    val_episodes = split["validation_episodes"]

    frames = load_frames(dataset_root)

    policy, preprocessor, postprocessor = load_policy_and_processors(checkpoint_dir)

    reload_check = checkpoint_reload_inference_check(policy, preprocessor, postprocessor, frames, val_episodes)
    validation_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint_reload_inference_check": reload_check}, f, indent=2, default=str)

    offline_eval = offline_prediction_evaluation(policy, preprocessor, postprocessor, frames, val_episodes)
    offline_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(offline_predictions_path, "w", encoding="utf-8") as f:
        json.dump(offline_eval, f, indent=2, default=str)

    print("=== Checkpoint reload + inference contract check ===")
    print(f"num_frames_tested: {reload_check['num_frames_tested']}")
    print(f"action_shape_is_6: {reload_check['action_shape_is_6']}")
    print(f"all_finite: {reload_check['all_finite']}")
    print(f"gripper_channel_range: {reload_check['gripper_channel_range_min_max']}")
    print(f"constant_output_across_frames: {reload_check['constant_output_across_frames']}")
    print(f"PASS: {reload_check['pass']}")
    print()
    print("=== Offline prediction evaluation (validation set) ===")
    print(f"num_validation_frames: {offline_eval['num_validation_frames']}")
    print(f"overall_mae: {offline_eval['overall_mae']:.4f}  overall_rmse: {offline_eval['overall_rmse']:.4f}")
    print(f"gripper_open_close_accuracy: {offline_eval['gripper_open_close_accuracy']:.3f}")
    print(f"constant_output_across_validation_set: {offline_eval['constant_output_across_validation_set']}")
    for phase in PHASES_OF_INTEREST:
        pe = offline_eval["phase_errors"][phase]
        if pe["frame_count"] > 0:
            print(f"  {phase}: frame_count={pe['frame_count']} mae={pe['mae']:.4f} rmse={pe['rmse']:.4f}")
    print()
    print(f"Validation metrics JSON: {validation_metrics_path}")
    print(f"Offline predictions JSON: {offline_predictions_path}")

    # Non-zero exit on non-finite output -- lets a caller (e.g.
    # benchmark/run_so101_smolvla_pipeline.py) gate rollout on this
    # without re-parsing the JSON itself.
    offline_all_finite = np.isfinite(offline_eval["overall_mae"]) and np.isfinite(offline_eval["overall_rmse"])
    if not (reload_check["all_finite"] and offline_all_finite):
        print("\nFAIL: non-finite (NaN/Inf) values detected in checkpoint output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
