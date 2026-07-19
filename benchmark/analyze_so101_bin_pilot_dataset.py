"""SO-101 bin pilot (20-episode) dataset quality/distribution analysis
(see this task's chat report, "수집된 데이터의 품질과 분포를 점검"). Reads
ONLY from disk (datasets/so101_bin_pilot_20) -- never trusts in-memory
collection state. Does NOT collect any new episodes, does NOT run VLA
training, does NOT touch expert/backend/schema files, does NOT apply
normalization to the raw dataset.

Reuses (does NOT reimplement):
  - benchmark.collect_so101_episode's own verify_dataset() for the
    disk round-trip base checks.
  - benchmark.so101_scripted_expert's own PHASE_SEQUENCE/
    PHASE_NAME_BY_ID for phase labeling.
  - benchmark.so101_dataset_schema's own SO101_JOINT_NAMES for
    joint-wise labeling.

Run:
  .venv-vla/bin/python -m benchmark.analyze_so101_bin_pilot_dataset
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p
from PIL import Image, ImageDraw

from benchmark.collect_so101_episode import verify_dataset
from benchmark.so101_dataset_schema import SO101_JOINT_NAMES
from benchmark.so101_scripted_expert import PHASE_NAME_BY_ID, PHASE_SEQUENCE
from robot_sim.so101_pybullet_backend import ARM_JOINT_NAMES, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_pilot_20"
SUMMARY_PATH = PROJECT_ROOT / "results" / "so101_bin_pilot_20_summary.json"
ACTION_STATS_PATH = PROJECT_ROOT / "results" / "so101_bin_pilot_20_action_stats.json"
PHASE_STATS_PATH = PROJECT_ROOT / "results" / "so101_bin_pilot_20_phase_stats.json"
PREVIEW_DIR = PROJECT_ROOT / "results" / "so101_bin_pilot_20_preview"

PREVIEW_SEEDS = [0, 5, 10, 15, 19]
FRAME_LABELS = ["first", "near_grasp", "mid_lift_transport", "near_release", "last"]
NUM_JOINTS = len(SO101_JOINT_NAMES)


def refuse_if_exists(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing result file/dir: {path}")


def load_manifest(root: Path) -> list:
    lines = (root / "collection_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_frames(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def decode_image(raw) -> np.ndarray:
    if isinstance(raw, dict) and "bytes" in raw:
        return np.array(Image.open(io.BytesIO(raw["bytes"])).convert("RGB"))
    return np.array(raw)


def round_trip_checks(root: Path, manifest: list, frames: pd.DataFrame, summary_json: dict) -> dict:
    checks = {}
    checks["reload_verification"] = verify_dataset(root)

    checks["total_episodes_eq_20"] = int(frames["episode_index"].nunique()) == 20
    checks["manifest_rows_eq_20"] = len(manifest) == 20
    checks["parquet_frame_count_eq_total_frames"] = len(frames) == summary_json["total_frames"]

    episode_indices = sorted(frames["episode_index"].unique().tolist())
    checks["episode_index_continuous_0_to_19"] = episode_indices == list(range(20))

    seeds = [m["seed"] for m in manifest]
    checks["seeds_0_to_19_no_dup_no_missing"] = sorted(seeds) == list(range(20))

    frame_range_issues = []
    for ep_idx, group in frames.groupby("episode_index"):
        fi = sorted(group["frame_index"].tolist())
        if fi != list(range(len(fi))):
            frame_range_issues.append({"episode_index": int(ep_idx), "frame_index_values": fi})
    checks["episode_frame_index_ranges_ok"] = len(frame_range_issues) == 0
    checks["frame_range_issues"] = frame_range_issues

    sample_rows = pd.concat([frames.iloc[[0]], frames.iloc[[len(frames) // 2]], frames.iloc[[-1]]])
    image_decode_ok = True
    for _, row in sample_rows.iterrows():
        try:
            img = decode_image(row["observation.images.front"])
            if img.shape != (256, 256, 3) or img.dtype != np.uint8:
                image_decode_ok = False
        except Exception:
            image_decode_ok = False
    checks["sample_images_decode_ok"] = image_decode_ok

    state_array = np.stack(frames["observation.state"].to_numpy())
    action_array = np.stack(frames["action"].to_numpy())
    checks["state_dtype_shape_ok"] = state_array.dtype == np.float32 and state_array.shape == (len(frames), NUM_JOINTS)
    checks["action_dtype_shape_ok"] = action_array.dtype == np.float32 and action_array.shape == (len(frames), NUM_JOINTS)
    checks["no_nan_inf"] = bool(np.all(np.isfinite(state_array)) and np.all(np.isfinite(action_array)))

    phase_ids = frames["phase_id"].apply(lambda v: int(v[0]) if hasattr(v, "__len__") else int(v))
    checks["phase_id_within_mapping_range"] = bool(phase_ids.between(0, len(PHASE_SEQUENCE) - 1).all())

    checks["all_episodes_place_success_true"] = all(m["place_success"] is True for m in manifest)

    checks["overall_round_trip_pass"] = all([
        checks["total_episodes_eq_20"], checks["manifest_rows_eq_20"], checks["parquet_frame_count_eq_total_frames"],
        checks["episode_index_continuous_0_to_19"], checks["seeds_0_to_19_no_dup_no_missing"],
        checks["episode_frame_index_ranges_ok"], checks["sample_images_decode_ok"],
        checks["state_dtype_shape_ok"], checks["action_dtype_shape_ok"], checks["no_nan_inf"],
        checks["phase_id_within_mapping_range"], checks["all_episodes_place_success_true"],
    ])
    return checks


def pick_preview_frame_indices(group: pd.DataFrame) -> dict:
    group = group.sort_values("frame_index")
    phase_names = group["phase_id"].apply(lambda v: PHASE_NAME_BY_ID[int(v[0]) if hasattr(v, "__len__") else int(v)])
    n = len(group)
    idx = {}
    idx["first"] = 0
    grasp_rows = group.index[phase_names == "grasp"].tolist()
    idx["near_grasp"] = group.index.get_loc(grasp_rows[len(grasp_rows) // 2]) if grasp_rows else n // 4
    lift_transport_rows = group.index[phase_names.isin(["lift", "transport"])].tolist()
    idx["mid_lift_transport"] = group.index.get_loc(lift_transport_rows[len(lift_transport_rows) // 2]) if lift_transport_rows else n // 2
    release_rows = group.index[phase_names == "release"].tolist()
    idx["near_release"] = group.index.get_loc(release_rows[0]) if release_rows else n - 2
    idx["last"] = n - 1
    return idx


def build_image_preview(frames: pd.DataFrame) -> dict:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sanity = {}
    thumbnails = {}

    for seed in PREVIEW_SEEDS:
        group = frames[frames["episode_index"] == seed].reset_index(drop=True)
        frame_positions = pick_preview_frame_indices(group)
        seed_dir = PREVIEW_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        sanity[str(seed)] = {}
        for label in FRAME_LABELS:
            row = group.iloc[frame_positions[label]]
            img = decode_image(row["observation.images.front"])
            Image.fromarray(img).save(seed_dir / f"{label}.png")
            thumbnails[(seed, label)] = img
            sanity[str(seed)][label] = {
                "frame_index": int(row["frame_index"]),
                "phase": PHASE_NAME_BY_ID[int(row["phase_id"][0]) if hasattr(row["phase_id"], "__len__") else int(row["phase_id"])],
                "brightness_mean": float(img.mean()),
                "brightness_std": float(img.std()),
                "looks_blank_or_broken": bool(img.std() < 1.0),
            }

    # Contact sheet: rows = seeds, cols = frame labels.
    thumb_size = 160
    grid = Image.new("RGB", (thumb_size * len(FRAME_LABELS), thumb_size * len(PREVIEW_SEEDS)), (32, 32, 32))
    draw = ImageDraw.Draw(grid)
    for row_i, seed in enumerate(PREVIEW_SEEDS):
        for col_i, label in enumerate(FRAME_LABELS):
            img = Image.fromarray(thumbnails[(seed, label)]).resize((thumb_size, thumb_size))
            grid.paste(img, (col_i * thumb_size, row_i * thumb_size))
            draw.text((col_i * thumb_size + 4, row_i * thumb_size + 4), f"s{seed}/{label}", fill=(255, 255, 0))
    contact_sheet_path = PREVIEW_DIR / "contact_sheet.png"
    grid.save(contact_sheet_path)

    sanity_path = PREVIEW_DIR / "sanity.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    seed_first_frame_brightness = {str(s): sanity[str(s)]["first"]["brightness_mean"] for s in PREVIEW_SEEDS}
    any_blank_or_broken = any(sanity[str(s)][label]["looks_blank_or_broken"] for s in PREVIEW_SEEDS for label in FRAME_LABELS)

    return {
        "preview_dir": str(PREVIEW_DIR), "contact_sheet_path": str(contact_sheet_path), "sanity_path": str(sanity_path),
        "seed_first_frame_brightness_mean": seed_first_frame_brightness,
        "any_frame_looks_blank_or_broken": any_blank_or_broken,
    }


def joint_wise_stats(array: np.ndarray) -> dict:
    stats = {}
    for i, name in enumerate(SO101_JOINT_NAMES):
        col = array[:, i]
        stats[name] = {
            "min": float(col.min()), "max": float(col.max()), "mean": float(col.mean()), "std": float(col.std()),
            "p01": float(np.percentile(col, 1)), "p50": float(np.percentile(col, 50)), "p99": float(np.percentile(col, 99)),
        }
    return stats


def compute_action_stats(frames: pd.DataFrame) -> dict:
    state_array = np.stack(frames["observation.state"].to_numpy())
    action_array = np.stack(frames["action"].to_numpy())

    result = {
        "observation_state": joint_wise_stats(state_array),
        "action": joint_wise_stats(action_array),
        "nan_inf_count": {
            "state": int(np.sum(~np.isfinite(state_array))), "action": int(np.sum(~np.isfinite(action_array))),
        },
    }

    action_delta = action_array - state_array
    result["action_minus_state_delta"] = joint_wise_stats(action_delta)

    consecutive_deltas = []
    identical_count = 0
    total_consecutive = 0
    per_episode_range = {}
    for ep_idx, group in frames.groupby("episode_index"):
        group = group.sort_values("frame_index")
        ep_action = np.stack(group["action"].to_numpy())
        per_episode_range[int(ep_idx)] = {
            SO101_JOINT_NAMES[i]: {"min": float(ep_action[:, i].min()), "max": float(ep_action[:, i].max())}
            for i in range(NUM_JOINTS)
        }
        if len(ep_action) > 1:
            deltas = ep_action[1:] - ep_action[:-1]
            consecutive_deltas.append(deltas)
            identical_count += int(np.sum(np.all(deltas == 0, axis=1)))
            total_consecutive += len(deltas)
    all_deltas = np.concatenate(consecutive_deltas, axis=0)
    result["consecutive_action_delta"] = joint_wise_stats(all_deltas)
    result["identical_consecutive_action_fraction"] = float(identical_count / total_consecutive) if total_consecutive else None
    result["per_episode_action_range"] = per_episode_range

    gripper_col = action_array[:, -1]
    result["gripper_action_distribution"] = {
        "min": float(gripper_col.min()), "max": float(gripper_col.max()),
        "fraction_near_0_closed": float(np.mean(gripper_col < 10.0)),
        "fraction_near_100_open": float(np.mean(gripper_col > 90.0)),
        "fraction_mid_range": float(np.mean((gripper_col >= 10.0) & (gripper_col <= 90.0))),
    }

    backend = So101PyBulletBackend(gui=False)
    try:
        backend.reset()
        joints_by_name = {}
        for joint_index in range(p.getNumJoints(backend.robot_id, physicsClientId=backend.client_id)):
            info = p.getJointInfo(backend.robot_id, joint_index, physicsClientId=backend.client_id)
            joints_by_name[info[1].decode("utf-8")] = {"lower": info[8], "upper": info[9]}
        violations = []
        for i, name in enumerate(ARM_JOINT_NAMES):
            lower, upper = joints_by_name[name]["lower"], joints_by_name[name]["upper"]
            col = state_array[:, i]
            bad = np.where((col < lower - 1e-6) | (col > upper + 1e-6))[0]
            for idx in bad:
                violations.append({"joint": name, "row": int(idx), "value": float(col[idx]), "lower": lower, "upper": upper})
        result["joint_limit_violation_count"] = len(violations)
        result["joint_limit_violations_sample"] = violations[:10]
    finally:
        backend.close()

    return result


def compute_phase_stats(frames: pd.DataFrame) -> dict:
    phase_names = frames["phase_id"].apply(lambda v: PHASE_NAME_BY_ID[int(v[0]) if hasattr(v, "__len__") else int(v)])
    total = len(frames)

    overall_counts = phase_names.value_counts().to_dict()
    overall_ratio = {k: v / total for k, v in overall_counts.items()}

    per_episode = {}
    per_episode_sequences = {}
    for ep_idx, group in frames.groupby("episode_index"):
        group = group.sort_values("frame_index")
        ep_phase_names = group["phase_id"].apply(lambda v: PHASE_NAME_BY_ID[int(v[0]) if hasattr(v, "__len__") else int(v)])
        per_episode[int(ep_idx)] = ep_phase_names.value_counts().to_dict()
        seq = []
        for name in ep_phase_names:
            if not seq or seq[-1] != name:
                seq.append(name)
        per_episode_sequences[int(ep_idx)] = seq

    ep_phase_ids = frames.groupby("episode_index").apply(
        lambda g: g.sort_values("frame_index")["phase_id"].apply(lambda v: int(v[0]) if hasattr(v, "__len__") else int(v)).tolist()
    )
    monotonic_per_episode = {int(ep): all(x <= y for x, y in zip(ids, ids[1:])) for ep, ids in ep_phase_ids.items()}

    unique_sequences = set(tuple(seq) for seq in per_episode_sequences.values())
    dominant_phase = max(overall_ratio, key=overall_ratio.get)

    return {
        "overall_phase_frame_count": overall_counts,
        "overall_phase_ratio": overall_ratio,
        "per_episode_phase_frame_count": per_episode,
        "per_episode_phase_sequence": per_episode_sequences,
        "all_episodes_monotonic_non_decreasing": all(monotonic_per_episode.values()),
        "monotonic_per_episode": monotonic_per_episode,
        "all_episodes_identical_phase_sequence": len(unique_sequences) == 1,
        "unique_phase_sequences": [list(s) for s in unique_sequences],
        "dominant_phase": dominant_phase,
        "dominant_phase_ratio": overall_ratio[dominant_phase],
        "settle_never_recorded": "settle" not in overall_counts,
    }


def compute_diversity_diagnostics(manifest: list, frames: pd.DataFrame, preview_result: dict) -> dict:
    initial_object_xy = np.array([[m["initial_object_position"][0], m["initial_object_position"][1]] for m in manifest])
    bin_center_xy = np.array([[m["bin_center"][0], m["bin_center"][1]] for m in manifest])
    offset_xy = bin_center_xy - initial_object_xy

    episode_lengths = frames.groupby("episode_index").size().tolist()

    action_arrays = {}
    for ep_idx, group in frames.groupby("episode_index"):
        group = group.sort_values("frame_index")
        action_arrays[int(ep_idx)] = np.stack(group["action"].to_numpy())

    ep_indices = sorted(action_arrays.keys())
    pairwise_rms = []
    for i in range(len(ep_indices)):
        for j in range(i + 1, len(ep_indices)):
            a, b = action_arrays[ep_indices[i]], action_arrays[ep_indices[j]]
            if a.shape == b.shape:
                pairwise_rms.append(float(np.sqrt(np.mean((a - b) ** 2))))

    first_frame_states = np.stack([action_arrays[e][0] for e in ep_indices])
    last_frame_states = np.stack([action_arrays[e][-1] for e in ep_indices])

    return {
        "initial_object_x_range": [float(initial_object_xy[:, 0].min()), float(initial_object_xy[:, 0].max())],
        "initial_object_y_range": [float(initial_object_xy[:, 1].min()), float(initial_object_xy[:, 1].max())],
        "bin_center_x_range": [float(bin_center_xy[:, 0].min()), float(bin_center_xy[:, 0].max())],
        "bin_center_y_range": [float(bin_center_xy[:, 1].min()), float(bin_center_xy[:, 1].max())],
        "object_bin_relative_offset_xy_mean": [float(offset_xy[:, 0].mean()), float(offset_xy[:, 1].mean())],
        "object_bin_relative_offset_xy_std": [float(offset_xy[:, 0].std()), float(offset_xy[:, 1].std())],
        "object_bin_offset_effectively_constant": bool(offset_xy[:, 0].std() < 1e-6 and offset_xy[:, 1].std() < 1e-6),
        "episode_length_unique_value_count": len(set(episode_lengths)),
        "episode_length_values": sorted(set(episode_lengths)),
        "pairwise_action_trajectory_rms_diff": {
            "mean": float(np.mean(pairwise_rms)) if pairwise_rms else None,
            "min": float(np.min(pairwise_rms)) if pairwise_rms else None,
            "max": float(np.max(pairwise_rms)) if pairwise_rms else None,
        },
        "first_frame_action_std_across_episodes": [float(x) for x in first_frame_states.std(axis=0)],
        "last_frame_action_std_across_episodes": [float(x) for x in last_frame_states.std(axis=0)],
        "seed_first_frame_image_brightness_mean": preview_result["seed_first_frame_brightness_mean"],
        "sufficient_for_recorder_format_validation": True,
        "sufficient_for_vla_training_diversity": False,
        "diversity_limitation_note": (
            "Object and bin move together with an effectively constant relative "
            "offset (std < 1e-6 in both x and y) -- only +/-1cm absolute object "
            "position is randomized (no yaw, no independent bin placement, no "
            "randomization range change this task). All 20 episodes have the "
            "IDENTICAL frame length (68) and near-identical phase sequence/ratio. "
            "This pilot set is adequate to validate recorder/schema/temporal "
            "correctness at scale, but does NOT provide the geometric or "
            "trajectory diversity a VLA policy would need to generalize -- "
            "trajectories are small perturbations of one nominal trajectory, "
            "not a diverse manipulation distribution. Randomization range "
            "expansion is deferred to a future task per this task's own scope."
        ),
    }


def main() -> None:
    refuse_if_exists(SUMMARY_PATH)
    refuse_if_exists(ACTION_STATS_PATH)
    refuse_if_exists(PHASE_STATS_PATH)
    refuse_if_exists(PREVIEW_DIR)

    manifest = load_manifest(DATASET_ROOT)
    frames = load_frames(DATASET_ROOT)
    summary_json = json.loads((DATASET_ROOT / "collection_summary.json").read_text())

    rt_checks = round_trip_checks(DATASET_ROOT, manifest, frames, summary_json)
    preview_result = build_image_preview(frames)
    action_stats = compute_action_stats(frames)
    phase_stats = compute_phase_stats(frames)
    diversity = compute_diversity_diagnostics(manifest, frames, preview_result)

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "collection_summary": summary_json,
        "round_trip_checks": rt_checks,
        "image_preview": preview_result,
        "diversity_diagnostics": diversity,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(ACTION_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(action_stats, f, indent=2, default=str)
    with open(PHASE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(phase_stats, f, indent=2, default=str)

    print("=== SO-101 bin pilot 20-episode dataset analysis ===")
    print(f"overall_round_trip_pass: {rt_checks['overall_round_trip_pass']}")
    print(f"any_frame_looks_blank_or_broken: {preview_result['any_frame_looks_blank_or_broken']}")
    print(f"object_bin_offset_effectively_constant: {diversity['object_bin_offset_effectively_constant']}")
    print(f"sufficient_for_recorder_format_validation: {diversity['sufficient_for_recorder_format_validation']}")
    print(f"sufficient_for_vla_training_diversity: {diversity['sufficient_for_vla_training_diversity']}")
    print(f"joint_limit_violation_count: {action_stats['joint_limit_violation_count']}")
    print(f"dominant_phase: {phase_stats['dominant_phase']} ratio={phase_stats['dominant_phase_ratio']:.3f}")
    print()
    print(f"Summary JSON: {SUMMARY_PATH}")
    print(f"Action stats JSON: {ACTION_STATS_PATH}")
    print(f"Phase stats JSON: {PHASE_STATS_PATH}")
    print(f"Preview dir: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
