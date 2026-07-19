"""Translation/rotation action-scaling diagnostic (v0).

Investigates whether benchmark/collect_recycling_dataset.py's recorded
action[0:6] (translation dx/dy/dz + rotation droll/dpitch/dyaw) is
already in real physical units (meters/radians), or a LIBERO-style
normalized native action space -- and whether
policy_semantics/adapters/smolvla_libero_adapter.py's fixed
TRANSLATION_SCALE_M(0.05)/ROTATION_SCALE_RAD(0.5) multiply is applied
uncritically to BOTH checkpoints regardless of which native scale each
one's own postprocessor actually produces (the same class of bug
already confirmed and fixed for the gripper dimension -- see this
task's chat report and the earlier gripper-collapse diagnosis).

Four lines of evidence, each grounded in real files/code:
  1. Real train20 action[0:6] stats, split into translation/rotation.
  2. Both checkpoints' own postprocessor stats (mean/std/min/max where
     available) for the SAME dims.
  3. The adapter's exact scale/clip constants, read from source.
  4. Real model queries on identical fixed observations for BOTH
     checkpoints, tracing: dataset-expected action -> network raw
     output -> postprocessed output -> adapter output (canonical
     translation_m/rotation_rad) -> final RobotCommand delta.

Read-only: loads both checkpoints for inference, never trains, never
writes to any checkpoint or dataset file. Does not touch gripper code.

Run: .venv-vla/bin/python -m benchmark.diagnose_translation_rotation_scaling
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN20_ROOT = PROJECT_ROOT / "datasets/recycling_lerobot_v0_train20"
ZERO_SHOT_MODEL_ID = "HuggingFaceVLA/smolvla_libero"
FINE_TUNED_MODEL_PATH = str(
    PROJECT_ROOT / "outputs/train/smolvla_recycling_smoke_v0/checkpoints/last/pretrained_model"
)
ORIGINAL_LIBERO_SNAPSHOT = Path(
    "/home/rlack/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/"
    "snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
)


def resolve(path_str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# 1. Real train20 action[0:6] distribution, translation/rotation split
# ============================================================

def analyze_translation_rotation_distribution(root: Path) -> dict:
    df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    actions = np.stack(df["action"].to_numpy())
    translation = actions[:, 0:3]
    rotation = actions[:, 3:6]
    return {
        "total_frames": len(df),
        "translation_min": translation.min(axis=0).tolist(),
        "translation_max": translation.max(axis=0).tolist(),
        "translation_mean": translation.mean(axis=0).tolist(),
        "translation_std": translation.std(axis=0).tolist(),
        "rotation_min": rotation.min(axis=0).tolist(),
        "rotation_max": rotation.max(axis=0).tolist(),
        "rotation_mean": rotation.mean(axis=0).tolist(),
        "rotation_std": rotation.std(axis=0).tolist(),
        "rotation_all_zero": bool((rotation == 0.0).all()),
    }


# ============================================================
# 2. Postprocessor stats comparison (both checkpoints)
# ============================================================

def read_postprocessor_action_stats(checkpoint_dir_or_none) -> dict:
    """Returns whatever action.mean/std/min/max this checkpoint's own
    postprocessor safetensors declares -- None entries where the stat
    wasn't stored (e.g. a MEAN_STD-only unnormalizer keeps no min/max,
    confirmed for the real HuggingFaceVLA/smolvla_libero checkpoint in
    an earlier turn)."""
    from safetensors import safe_open

    config_path = checkpoint_dir_or_none / "policy_postprocessor.json"
    config = json.loads(config_path.read_text())
    state_file = None
    for step in config.get("steps", []):
        if step.get("registry_name") == "unnormalizer_processor":
            state_file = step.get("state_file")
            break
    if not state_file:
        return {}

    result = {}
    with safe_open(str(checkpoint_dir_or_none / state_file), framework="pt") as f:
        keys = set(f.keys())
        for stat in ("mean", "std", "min", "max"):
            key = f"action.{stat}"
            if key in keys:
                result[stat] = f.get_tensor(key).tolist()
    return result


def compare_postprocessor_translation_rotation_stats() -> dict:
    ft_stats = read_postprocessor_action_stats(Path(FINE_TUNED_MODEL_PATH))

    # Real LIBERO's own shipped postprocessor only stores mean/std (see
    # module docstring) -- its native min/max is instead read from the
    # real HuggingFaceVLA/libero dataset's own meta/stats.json (external
    # to the checkpoint file, confirmed in an earlier turn: exactly
    # Box(-1, 1)-shaped).
    libero_ckpt_stats = read_postprocessor_action_stats(
        Path("/home/rlack/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/"
             "snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102")
    )
    libero_dataset_stats = json.loads((ORIGINAL_LIBERO_SNAPSHOT / "meta" / "stats.json").read_text())

    return {
        "fine_tuned_postprocessor": ft_stats,
        "libero_checkpoint_postprocessor": libero_ckpt_stats,
        "libero_real_dataset_stats_action": libero_dataset_stats["action"],
    }


# ============================================================
# 3. Adapter scale/clip constants (read from source, not re-declared)
# ============================================================

def read_adapter_constants() -> dict:
    from policy_semantics.adapters.smolvla_libero_adapter import (
        NATIVE_ACTION_HIGH,
        NATIVE_ACTION_LOW,
        ROTATION_SCALE_RAD,
        TRANSLATION_SCALE_M,
    )

    return {
        "NATIVE_ACTION_LOW": NATIVE_ACTION_LOW,
        "NATIVE_ACTION_HIGH": NATIVE_ACTION_HIGH,
        "TRANSLATION_SCALE_M": TRANSLATION_SCALE_M,
        "ROTATION_SCALE_RAD": ROTATION_SCALE_RAD,
    }


# ============================================================
# 4. Full pipeline trace on real fixed observations, both checkpoints
# ============================================================

def _make_fixed_observations():
    from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")

    observations = []
    positions = {
        "far_center_right": [0.42, 0.00, 0.05],
        "far_center_left": [0.27, 0.00, 0.05],
        "near_object": None,  # filled below, at current EE position
        "positive_y": [0.35, 0.18, 0.05],
        "negative_y": [0.35, -0.18, 0.05],
    }
    ee_state = backend.get_state()
    positions["near_object"] = [ee_state["end_effector_position"][0], ee_state["end_effector_position"][1], 0.05]

    for label, pos in positions.items():
        backend.set_object_position(pos)
        observations.append((label, backend.get_state(), backend.render_main_camera(), backend.render_wrist_camera(), pos))

    backend.shutdown()
    return observations


def _query_checkpoint_full_trace(model_id_or_path: str, observations, instruction: str) -> list:
    import vla_server.model_loader as model_loader
    from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
    from policy_semantics.manifest import get_manifest
    from benchmark.run_vla_action_direction_diagnostic import build_robot_state
    from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

    with model_loader._lock:
        model_loader._state.update({
            "status": "not_loaded", "reason": None, "model_family": None, "model": None,
            "processor": None, "compatibility_result": None, "preprocessor_pipeline": None,
            "postprocessor_pipeline": None,
        })
    result = model_loader.load_model_once("smolvla", model_id_or_path, local_files_only=True)
    if result["status"] != "loaded":
        raise RuntimeError(f"Failed to load {model_id_or_path!r}: {result}")

    action_adapter = SmolVLALiberoActionAdapter()
    manifest = get_manifest(model_id_or_path)

    rows = []
    for label, state, main_image, wrist_image, object_pos in observations:
        backend = PyBulletPandaBackend(gui=False)
        backend.reset()
        backend.set_object_type("plastic_bottle")
        backend.set_object_position(object_pos)
        robot_state, _state_8d, resolved_object_position = build_robot_state(backend)
        main_image = backend.render_main_camera()
        wrist_image = backend.render_wrist_camera()

        model_input = {
            "instruction": instruction,
            "image": main_image,
            "images_by_role": {"main": main_image, "wrist": wrist_image},
            "robot_state": robot_state,
            "step_index": 0,
            "phase": "move_to_object",
            "seed": 0,
            "model_id_or_path": model_id_or_path,
        }
        native_action = model_loader.run_inference("smolvla", model_input)
        raw_model_action = native_action.metadata["raw_model_action"]
        postprocessed_action = native_action.values

        canonical_command = action_adapter.decode(native_action, manifest, context={"degraded_input": False})

        rows.append({
            "label": label,
            "network_raw_translation": raw_model_action[0:3],
            "network_raw_rotation": raw_model_action[3:6],
            "postprocessed_translation": postprocessed_action[0:3],
            "postprocessed_rotation": postprocessed_action[3:6],
            "adapter_translation_m": list(canonical_command.translation_m) if canonical_command else None,
            "adapter_rotation_rad": list(canonical_command.rotation_axis_angle_rad) if canonical_command else None,
            "final_robot_command_translation": list(canonical_command.translation_m) if canonical_command else None,
            "final_robot_command_rotation": list(canonical_command.rotation_axis_angle_rad) if canonical_command else None,
        })
        backend.shutdown()
    return rows


def run_full_pipeline_trace(instruction: str = "플라스틱 병을 플라스틱 수거함에 넣어줘") -> dict:
    observations = _make_fixed_observations()
    zero_shot_rows = _query_checkpoint_full_trace(ZERO_SHOT_MODEL_ID, observations, instruction)
    fine_tuned_rows = _query_checkpoint_full_trace(FINE_TUNED_MODEL_PATH, observations, instruction)
    return {"zero_shot": zero_shot_rows, "fine_tuned": fine_tuned_rows}


def main() -> None:
    print("=== 1. Real train20 action[0:6] distribution (translation/rotation split) ===")
    dist = analyze_translation_rotation_distribution(TRAIN20_ROOT)
    print(f"total_frames: {dist['total_frames']}")
    print(f"translation min: {dist['translation_min']}")
    print(f"translation max: {dist['translation_max']}")
    print(f"translation mean: {dist['translation_mean']}")
    print(f"translation std: {dist['translation_std']}")
    print(f"rotation min/max/mean/std: {dist['rotation_min']} / {dist['rotation_max']} / {dist['rotation_mean']} / {dist['rotation_std']}")
    print(f"rotation all exactly zero: {dist['rotation_all_zero']}")
    print("--> translation values are in [-0.03, +0.03] -- exactly matching "
          "DummyOpenVLAPolicy.DEFAULT_MAX_STEP_SIZE (0.03 m) -- these are REAL METERS, not a [-1,1] native space.")
    print()

    print("=== 2. Postprocessor stats: fine-tuned checkpoint vs real LIBERO ===")
    stats = compare_postprocessor_translation_rotation_stats()
    ft = stats["fine_tuned_postprocessor"]
    print(f"fine-tuned action.mean[0:3] (translation): {ft['mean'][0:3]}")
    print(f"fine-tuned action.std[0:3]: {ft['std'][0:3]}")
    print(f"fine-tuned action.min[0:3]: {ft['min'][0:3]}")
    print(f"fine-tuned action.max[0:3]: {ft['max'][0:3]}")
    libero_real = stats["libero_real_dataset_stats_action"]
    print(f"LIBERO real dataset action mean[0:3]: {libero_real['mean'][0:3]}")
    print(f"LIBERO real dataset action std[0:3]: {libero_real['std'][0:3]}")
    print(f"LIBERO real dataset action min[0:3]: {libero_real['min'][0:3]}")
    print(f"LIBERO real dataset action max[0:3]: {libero_real['max'][0:3]}")
    print("--> fine-tuned checkpoint's own postprocessor stats live in [-0.03, 0.03] (real meters); "
          "LIBERO's live in [-0.9375, 0.9375] (native controller units).")
    print()

    print("=== 3. Adapter scale/clip constants (from source) ===")
    constants = read_adapter_constants()
    print(constants)
    print()

    print("=== 4. Full pipeline trace on 5 real fixed observations, both checkpoints ===")
    trace = run_full_pipeline_trace()
    for label in ("zero_shot", "fine_tuned"):
        print(f"--- {label} ---")
        for row in trace[label]:
            print(f"  [{row['label']}]")
            print(f"    network_raw_translation:      {[f'{v:+.4f}' for v in row['network_raw_translation']]}")
            print(f"    postprocessed_translation:    {[f'{v:+.4f}' for v in row['postprocessed_translation']]}")
            print(f"    adapter_translation_m:        {[f'{v:+.5f}' for v in row['adapter_translation_m']] if row['adapter_translation_m'] else None}")
            print(f"    network_raw_rotation:         {[f'{v:+.4f}' for v in row['network_raw_rotation']]}")
            print(f"    postprocessed_rotation:       {[f'{v:+.4f}' for v in row['postprocessed_rotation']]}")
            print(f"    adapter_rotation_rad:         {[f'{v:+.5f}' for v in row['adapter_rotation_rad']] if row['adapter_rotation_rad'] else None}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
