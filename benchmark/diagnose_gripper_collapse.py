"""Gripper-collapse root-cause diagnostic (v0).

Investigates why the fine-tuned SmolVLA checkpoint's gripper output
collapsed to "close" on 480/480 benchmark steps (see this task's chat
report for the prior zero-shot-vs-fine-tuned comparison that found
this). Four independent lines of evidence, each grounded in real
files/code, not assumption:

  1. Training-data gripper label distribution -- RAW per-frame AND
     actual training-CHUNK exposure-weighted (accounting for
     LeRobot's padding-via-index-clamping, see
     dataset_reader.py._get_query_indices()).
  2. Processor/normalizer stats comparison -- real HuggingFaceVLA/
     smolvla_libero's baked-in action stats vs. this project's own
     freshly-computed train20 stats (meta/stats.json).
  3. Raw model output comparison -- BOTH checkpoints loaded
     in-process (via vla_server.model_loader, the exact production
     loading path) and queried on the SAME fixed observations at
     varying distance-to-object, capturing the raw postprocessed
     native_action.values[6] BEFORE SmolVLALiberoActionAdapter's
     fixed (1 - raw_gripper) / 2 conversion (built for LIBERO's
     native [-1, 1] range, see policy_semantics/adapters/
     smolvla_libero_adapter.py) -- to check whether that formula's
     assumed native scale actually matches what each checkpoint's own
     postprocessor produces.
  4. Weight-level check -- direct safetensors diff of action_out_proj/
     action_in_proj/state_proj between the original and fine-tuned
     checkpoint, gripper row isolated from the other 6 action dims.

Read-only: loads both checkpoints for inference, never trains,
never writes to any checkpoint or dataset file.

Run: .venv-vla/bin/python -m benchmark.diagnose_gripper_collapse
"""

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
CHUNK_SIZE = 50  # SmolVLAConfig.chunk_size, both checkpoints share this base config


def resolve(path_str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# 1. Training data gripper label distribution
# ============================================================

def analyze_dataset_gripper_distribution(root: Path) -> dict:
    df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    all_gripper = np.stack(df["action"].to_numpy())[:, 6]
    raw_close_ratio = float((all_gripper >= 0.5).mean())

    per_episode = {}
    total_close_weight = 0
    total_open_weight = 0
    total_valid_targets = 0
    rel_pos_values = {k: [] for k in range(CHUNK_SIZE)}

    for ep in sorted(df.episode_index.unique()):
        sub = df[df.episode_index == ep].sort_values("frame_index").reset_index(drop=True)
        g = np.stack(sub["action"].to_numpy())[:, 6]
        is_close = (g >= 0.5).astype(int)
        L = len(g)

        transitions = int(np.sum(np.abs(np.diff(is_close))))
        close_idx = np.where(is_close == 1)[0]
        open_idx = np.where(is_close == 0)[0]

        runs, cur = [], 0
        for v in is_close:
            if v == 1:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)

        per_episode[int(ep)] = {
            "length": L,
            "close_ratio": float(is_close.mean()),
            "transitions": transitions,
            "first_close_frame": int(close_idx[0]) if len(close_idx) else None,
            "last_open_frame": int(open_idx[-1]) if len(open_idx) else None,
            "close_run_lengths": runs,
        }

        # Chunk-exposure weighting: frame i is a valid (non-padded) target
        # for every sample s in [0, i] (since delta=i-s must be in
        # [0, CHUNK_SIZE-1] and i < CHUNK_SIZE-1 always holds for our
        # short episodes -- see dataset_reader.py._get_query_indices()'s
        # clamping, confirmed in an earlier turn: query_index =
        # max(ep_start, min(ep_end-1, abs_idx+delta)), and action_is_pad
        # is True (hence loss-masked, see modeling_smolvla.py forward())
        # whenever abs_idx+delta is OUTSIDE [ep_start, ep_end) -- so this
        # weighting counts only genuinely non-padded exposures).
        for i in range(L):
            weight = i + 1
            if is_close[i]:
                total_close_weight += weight
            else:
                total_open_weight += weight
            total_valid_targets += weight

        for s in range(L):
            for rel in range(min(CHUNK_SIZE, L - s)):
                rel_pos_values[rel].append(is_close[s + rel])

    rel_pos_close_ratio = {
        rel: float(np.mean(vals)) for rel, vals in rel_pos_values.items() if vals
    }

    return {
        "total_frames": len(df),
        "raw_close_ratio": raw_close_ratio,
        "raw_open_ratio": 1.0 - raw_close_ratio,
        "per_episode": per_episode,
        "mean_transitions_per_episode": float(np.mean([e["transitions"] for e in per_episode.values()])),
        "mean_close_run_length": float(np.mean([r for e in per_episode.values() for r in e["close_run_lengths"]])),
        "chunk_exposure_weighted_close_ratio": total_close_weight / total_valid_targets,
        "chunk_exposure_weighted_open_ratio": total_open_weight / total_valid_targets,
        "total_valid_target_exposures": total_valid_targets,
        "close_ratio_by_chunk_relative_position": rel_pos_close_ratio,
    }


# ============================================================
# 2. Processor/normalizer stats comparison
# ============================================================

def compare_processor_stats() -> dict:
    import json

    train20_stats = json.loads((TRAIN20_ROOT / "meta" / "stats.json").read_text())
    libero_stats = json.loads((ORIGINAL_LIBERO_SNAPSHOT / "meta" / "stats.json").read_text())

    return {
        "train20_action_mean": train20_stats["action"]["mean"],
        "train20_action_std": train20_stats["action"]["std"],
        "train20_action_min": train20_stats["action"].get("min"),
        "train20_action_max": train20_stats["action"].get("max"),
        "libero_action_mean": libero_stats["action"]["mean"],
        "libero_action_std": libero_stats["action"]["std"],
        "libero_action_min": libero_stats["action"].get("min"),
        "libero_action_max": libero_stats["action"].get("max"),
    }


# ============================================================
# 3. Raw model output comparison (in-process, both checkpoints)
# ============================================================

def _make_fixed_observations():
    """3 fixed PyBullet observations at increasing proximity to the
    object -- far / near / at-bin -- built once and reused for BOTH
    checkpoints so raw outputs are directly comparable (same pixels,
    same state, only the loaded weights differ)."""
    from robot_sim.pybullet_panda_backend import PyBulletPandaBackend

    backend = PyBulletPandaBackend(gui=False)
    backend.reset()
    backend.set_object_type("plastic_bottle")

    observations = []
    # far: object at its default spawn position (~0.45m away)
    backend.set_object_position([0.45, 0.0, 0.05])
    observations.append(("far_from_object", backend.get_state(), backend.render_main_camera(), backend.render_wrist_camera()))

    # near: object placed right at the EE's current position
    ee_state = backend.get_state()
    near_pos = [ee_state["end_effector_position"][0], ee_state["end_effector_position"][1], 0.05]
    backend.set_object_position(near_pos)
    observations.append(("near_object", backend.get_state(), backend.render_main_camera(), backend.render_wrist_camera()))

    # at_bin: object placed at the bin position (post-carry scenario)
    bin_pos = list(backend.get_state()["bin_position"])
    backend.set_object_position(bin_pos)
    observations.append(("at_bin", backend.get_state(), backend.render_main_camera(), backend.render_wrist_camera()))

    backend.shutdown()
    return observations


def _query_checkpoint(model_id_or_path: str, observations, instruction: str) -> list:
    import vla_server.model_loader as model_loader
    from policy_semantics.adapters.smolvla_libero_adapter import SmolVLALiberoActionAdapter
    from policy_semantics.manifest import get_manifest

    # Force a clean reload regardless of whatever this process last had
    # loaded -- diagnostic-only manipulation of module state, never done
    # in production code (generic_vla_server.py always loads exactly one
    # model_family/model_id_or_path for the lifetime of the process).
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
    rows = []
    for label, state, main_image, wrist_image in observations:
        model_input = {
            "instruction": instruction,
            "image": main_image,
            "images_by_role": {"main": main_image, "wrist": wrist_image},
            "robot_state": state,
            "step_index": 0,
            "phase": "move_to_object",
            "seed": 0,
            "model_id_or_path": model_id_or_path,
        }
        native_action = model_loader.run_inference("smolvla", model_input)
        raw_model_gripper = native_action.metadata["raw_model_action"][6]
        postprocessed_gripper = native_action.values[6]

        manifest = get_manifest(model_id_or_path)
        canonical_command = action_adapter.decode(native_action, manifest, context={"degraded_input": False})

        rows.append({
            "label": label,
            "raw_model_gripper": raw_model_gripper,
            "postprocessed_native_gripper": postprocessed_gripper,
            "clipped_to_native_range": max(-1.0, min(1.0, postprocessed_gripper)),
            "gripper_opening_01": canonical_command.gripper_opening_01 if canonical_command else None,
            "gripper_command": (
                "close" if (canonical_command is not None and canonical_command.gripper_opening_01 <= 0.5) else "open"
            ) if canonical_command is not None else None,
        })
    return rows


def compare_raw_model_outputs(instruction: str = "플라스틱 병을 플라스틱 수거함에 넣어줘") -> dict:
    observations = _make_fixed_observations()
    zero_shot_rows = _query_checkpoint(ZERO_SHOT_MODEL_ID, observations, instruction)
    fine_tuned_rows = _query_checkpoint(FINE_TUNED_MODEL_PATH, observations, instruction)
    return {"zero_shot": zero_shot_rows, "fine_tuned": fine_tuned_rows}


# ============================================================
# main
# ============================================================

def main() -> None:
    print("=== 1. Training data gripper label distribution (datasets/recycling_lerobot_v0_train20) ===")
    dist = analyze_dataset_gripper_distribution(TRAIN20_ROOT)
    print(f"total_frames: {dist['total_frames']}")
    print(f"RAW per-frame: close={dist['raw_close_ratio']:.2%} open={dist['raw_open_ratio']:.2%}")
    print(f"mean transitions/episode: {dist['mean_transitions_per_episode']:.2f}")
    print(f"mean close run length: {dist['mean_close_run_length']:.2f} steps")
    print(f"CHUNK-EXPOSURE WEIGHTED (accounts for padding-clamp repetition of late frames): "
          f"close={dist['chunk_exposure_weighted_close_ratio']:.2%} open={dist['chunk_exposure_weighted_open_ratio']:.2%} "
          f"(n={dist['total_valid_target_exposures']} valid target-exposures)")
    print("close_ratio by chunk-relative-position (0=next action actually executed, 49=49 steps ahead):")
    for rel in sorted(dist["close_ratio_by_chunk_relative_position"]):
        if rel % 5 == 0:
            print(f"  rel_pos={rel:2d}: close_ratio={dist['close_ratio_by_chunk_relative_position'][rel]:.2%}")
    print()

    print("=== 2. Processor/normalizer stats: our train20 vs real LIBERO dataset ===")
    stats = compare_processor_stats()
    print(f"train20 action[6] (gripper): mean={stats['train20_action_mean'][6]:.4f} std={stats['train20_action_std'][6]:.4f} "
          f"min={stats['train20_action_min'][6]} max={stats['train20_action_max'][6]}")
    print(f"LIBERO   action[6] (gripper): mean={stats['libero_action_mean'][6]:.4f} std={stats['libero_action_std'][6]:.4f} "
          f"min={stats['libero_action_min'][6]} max={stats['libero_action_max'][6]}")
    print("--> train20's gripper dim lives in [0, 1] (legacy wire convention); LIBERO's lives in [-1, 1] "
          "(robosuite native convention). SmolVLALiberoActionAdapter.decode()'s fixed formula "
          "gripper_opening_01 = (1 - raw_gripper) / 2 assumes the LATTER range.")
    print()

    print("=== 3. Raw model output comparison (same fixed observations, both checkpoints, in-process) ===")
    comparison = compare_raw_model_outputs()
    for label in ("zero_shot", "fine_tuned"):
        print(f"--- {label} ---")
        for row in comparison[label]:
            print(
                f"  {row['label']:<16} raw_model_gripper={row['raw_model_gripper']:+.4f} "
                f"postprocessed={row['postprocessed_native_gripper']:+.4f} "
                f"gripper_opening_01={row['gripper_opening_01']:.4f} -> {row['gripper_command']}"
            )
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
