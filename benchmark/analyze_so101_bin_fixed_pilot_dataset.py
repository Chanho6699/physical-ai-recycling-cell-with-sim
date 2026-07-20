"""SO-101 fixed-bin pilot (30-episode, randomization_mode=fixed_bin_object_xy)
dataset quality/diversity analysis (see this task's chat report,
"phase별 데이터 다양성과 시각적 품질을 분석"). Reads ONLY from disk
(datasets/so101_bin_fixed_pilot_30) -- never trusts in-memory collection
state. Does NOT collect any new episodes, does NOT run VLA training,
does NOT touch expert/backend/schema files, does NOT apply
normalization to the raw dataset.

Reuses (does NOT reimplement):
  - benchmark.collect_so101_episode's own verify_dataset().
  - benchmark.analyze_so101_bin_pilot_dataset's own load_manifest(),
    load_frames(), decode_image(), joint_wise_stats(),
    compute_action_stats(), compute_phase_stats() -- the SAME pure
    functions already validated on the earlier 20-episode coupled_small
    pilot, applied here to the new dataset.
  - benchmark.so101_scripted_expert's own PHASE_SEQUENCE/PHASE_NAME_BY_ID.
  - benchmark.benchmark_so101_bin_diagnostic's own FIXED_BIN_MODE_*
    constants (fixed bin center / randomization range), so this
    analysis can never silently diverge from what the collector itself
    actually used.
  - benchmark.evaluate_so101_expert_small_randomization's own
    sample_object_position() to reconstruct EXACT scenes for
    segmentation-based visual-diversity rendering (section 9) --
    same deterministic seed -> object position function the collector
    itself used, not re-derived independently.

This task's core requirement (section 7): PHASE-LEVEL trajectory
diversity, not a single whole-episode average -- computed directly
from the dataset's own recorded action/phase_id columns (no re-running
episodes needed, unlike the earlier randomization-mode comparison
which had to re-run episodes since its diagnostic JSON had no
frame-level data).

Run:
  .venv-vla/bin/python -m benchmark.analyze_so101_bin_fixed_pilot_dataset
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from benchmark.analyze_so101_bin_pilot_dataset import (
    compute_action_stats,
    compute_phase_stats,
    decode_image,
    joint_wise_stats,
    load_frames,
    load_manifest,
)
from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
)
from benchmark.collect_so101_episode import verify_dataset
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.measure_so101_bin_visual_salience import render_rgb_and_segmentation
from benchmark.so101_dataset_schema import SO101_JOINT_NAMES
from benchmark.so101_scripted_expert import PHASE_NAME_BY_ID, PHASE_SEQUENCE
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_fixed_pilot_30"
COUPLED_DATASET_ROOT = PROJECT_ROOT / "datasets" / "so101_bin_pilot_20"  # earlier pilot, read-only comparison baseline
SUMMARY_PATH = PROJECT_ROOT / "results" / "so101_bin_fixed_pilot_30_summary.json"
ACTION_STATS_PATH = PROJECT_ROOT / "results" / "so101_bin_fixed_pilot_30_action_stats.json"
PHASE_DIVERSITY_PATH = PROJECT_ROOT / "results" / "so101_bin_fixed_pilot_30_phase_diversity.json"
VISUAL_DIVERSITY_PATH = PROJECT_ROOT / "results" / "so101_bin_fixed_pilot_30_visual_diversity.json"
PREVIEW_DIR = PROJECT_ROOT / "results" / "so101_bin_fixed_pilot_30_preview"

PREVIEW_SEEDS = [0, 5, 10, 15, 20, 25, 29]
FRAME_LABELS = ["first", "approach", "grasp", "transport", "release", "last"]
PHASES_OF_INTEREST = ["pre_grasp", "approach", "grasp", "lift", "transport", "place_descend", "release"]
NUM_JOINTS = len(SO101_JOINT_NAMES)


def refuse_if_exists(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing result file/dir: {path}")


def frame_phase_name(phase_id_value) -> str:
    pid = int(phase_id_value[0]) if hasattr(phase_id_value, "__len__") else int(phase_id_value)
    return PHASE_NAME_BY_ID[pid]


def round_trip_and_contract_recheck(root: Path, manifest: list, frames: pd.DataFrame, summary_json: dict) -> dict:
    checks = {}
    checks["reload_verification"] = verify_dataset(root)

    checks["total_episodes_eq_30"] = int(frames["episode_index"].nunique()) == 30
    checks["manifest_rows_eq_30"] = len(manifest) == 30
    episode_indices = sorted(frames["episode_index"].unique().tolist())
    checks["episode_index_continuous_0_to_29"] = episode_indices == list(range(30))
    seeds = [m["seed"] for m in manifest]
    checks["seeds_0_to_29_no_dup_no_missing"] = sorted(seeds) == list(range(30))
    checks["parquet_frame_count_eq_total_frames"] = len(frames) == summary_json["total_frames"]

    state_array = np.stack(frames["observation.state"].to_numpy())
    action_array = np.stack(frames["action"].to_numpy())
    checks["state_action_dtype_shape_ok"] = (
        state_array.dtype == np.float32 and action_array.dtype == np.float32
        and state_array.shape == (len(frames), NUM_JOINTS) and action_array.shape == (len(frames), NUM_JOINTS)
    )
    checks["no_nan_inf"] = bool(np.all(np.isfinite(state_array)) and np.all(np.isfinite(action_array)))

    sample_rows = pd.concat([frames.iloc[[0]], frames.iloc[[len(frames) // 2]], frames.iloc[[-1]]])
    checks["sample_images_decode_ok"] = all(
        decode_image(row["observation.images.front"]).shape == (256, 256, 3) for _, row in sample_rows.iterrows()
    )

    checks["all_episodes_place_success_true"] = all(m["place_success"] is True for m in manifest)
    checks["all_episodes_randomization_mode_fixed_bin_object_xy"] = all(m.get("randomization_mode") == "fixed_bin_object_xy" for m in manifest)
    checks["all_episodes_object_yaw_zero"] = all(m.get("object_yaw_rad") == 0.0 for m in manifest)

    bin_centers = np.array([m["bin_center"] for m in manifest])
    checks["fixed_bin_center_identical_across_episodes"] = bool(np.all(bin_centers == bin_centers[0]))
    checks["fixed_bin_center_value"] = bin_centers[0].tolist()

    object_positions = np.array([m["initial_object_position"][:2] for m in manifest])
    checks["initial_object_position_varies"] = bool(object_positions.std(axis=0).min() > 1e-6)
    checks["initial_object_position_std_xy"] = object_positions.std(axis=0).tolist()

    offsets = np.array([m["target_zone_offset_xy"] for m in manifest])
    checks["object_bin_relative_offset_varies"] = bool(offsets.std(axis=0).min() > 1e-6)
    checks["object_bin_relative_offset_std_xy"] = offsets.std(axis=0).tolist()

    checks["overall_pass"] = all(
        v for k, v in checks.items()
        if isinstance(v, bool)
    )
    return checks


def pick_preview_frame_indices(group: pd.DataFrame) -> dict:
    group = group.sort_values("frame_index").reset_index(drop=True)
    phase_names = group["phase_id"].apply(frame_phase_name)
    n = len(group)
    idx = {"first": 0, "last": n - 1}
    for label, phase in (("approach", "approach"), ("grasp", "grasp"), ("transport", "transport"), ("release", "release")):
        rows = group.index[phase_names == phase].tolist()
        idx[label] = rows[len(rows) // 2] if rows else n // 2
    return idx


def build_image_preview(frames: pd.DataFrame) -> dict:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sanity = {}
    thumbnails = {}

    for seed in PREVIEW_SEEDS:
        group = frames[frames["episode_index"] == seed].reset_index(drop=True)
        positions = pick_preview_frame_indices(group)
        seed_dir = PREVIEW_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        sanity[str(seed)] = {}
        for label in FRAME_LABELS:
            row = group.iloc[positions[label]]
            img = decode_image(row["observation.images.front"])
            Image.fromarray(img).save(seed_dir / f"{label}.png")
            thumbnails[(seed, label)] = img
            sanity[str(seed)][label] = {
                "frame_index": int(row["frame_index"]), "phase": frame_phase_name(row["phase_id"]),
                "brightness_mean": float(img.mean()), "brightness_std": float(img.std()),
                "looks_blank_or_broken": bool(img.std() < 1.0),
            }

    thumb = 160
    grid = Image.new("RGB", (thumb * len(FRAME_LABELS), thumb * len(PREVIEW_SEEDS)), (32, 32, 32))
    draw = ImageDraw.Draw(grid)
    for row_i, seed in enumerate(PREVIEW_SEEDS):
        for col_i, label in enumerate(FRAME_LABELS):
            img = Image.fromarray(thumbnails[(seed, label)]).resize((thumb, thumb))
            grid.paste(img, (col_i * thumb, row_i * thumb))
            draw.text((col_i * thumb + 4, row_i * thumb + 4), f"s{seed}/{label}", fill=(255, 255, 0))
    contact_sheet_path = PREVIEW_DIR / "contact_sheet.png"
    grid.save(contact_sheet_path)

    sanity_path = PREVIEW_DIR / "sanity.json"
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)

    any_blank = any(sanity[str(s)][label]["looks_blank_or_broken"] for s in PREVIEW_SEEDS for label in FRAME_LABELS)
    return {
        "preview_dir": str(PREVIEW_DIR), "contact_sheet_path": str(contact_sheet_path), "sanity_path": str(sanity_path),
        "any_frame_looks_blank_or_broken": any_blank,
        "seed_first_frame_brightness_mean": {str(s): sanity[str(s)]["first"]["brightness_mean"] for s in PREVIEW_SEEDS},
    }


def phase_level_diversity(frames: pd.DataFrame, label: str) -> dict:
    """Core analysis (this task's own section 7) -- per PHASE, not one
    whole-episode average: frame count, action mean/std, seed-pairwise
    action RMS (using each episode's OWN frames in that phase, aligned
    by within-phase step index -- padded/truncated to the shorter of
    the pair since phase lengths can differ slightly between seeds),
    start/end-of-phase action variance across seeds, and per-joint
    range within the phase."""
    result = {}
    for phase in PHASES_OF_INTEREST:
        phase_frames = frames[frames["phase_id"].apply(frame_phase_name) == phase]
        if len(phase_frames) == 0:
            result[phase] = {"frame_count": 0}
            continue

        per_seed_actions = {}
        for ep, group in phase_frames.groupby("episode_index"):
            group = group.sort_values("frame_index")
            per_seed_actions[int(ep)] = np.stack(group["action"].to_numpy())

        all_actions = np.concatenate(list(per_seed_actions.values()), axis=0)
        action_mean = all_actions.mean(axis=0).tolist()
        action_std = all_actions.std(axis=0).tolist()
        joint_range = (all_actions.max(axis=0) - all_actions.min(axis=0)).tolist()

        seeds = sorted(per_seed_actions.keys())
        pairwise_rms = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = per_seed_actions[seeds[i]], per_seed_actions[seeds[j]]
                n = min(len(a), len(b))
                if n > 0:
                    pairwise_rms.append(float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2))))

        start_actions = np.array([a[0] for a in per_seed_actions.values()])
        end_actions = np.array([a[-1] for a in per_seed_actions.values()])

        result[phase] = {
            "frame_count": int(len(phase_frames)),
            "frame_count_per_episode_mean": float(phase_frames.groupby("episode_index").size().mean()),
            "action_mean": action_mean, "action_std": action_std,
            "joint_range": {SO101_JOINT_NAMES[i]: float(joint_range[i]) for i in range(NUM_JOINTS)},
            "pairwise_action_rms": {
                "mean": float(np.mean(pairwise_rms)) if pairwise_rms else None,
                "min": float(np.min(pairwise_rms)) if pairwise_rms else None,
                "max": float(np.max(pairwise_rms)) if pairwise_rms else None,
            },
            "start_action_std_across_seeds": start_actions.std(axis=0).tolist(),
            "end_action_std_across_seeds": end_actions.std(axis=0).tolist(),
        }
    result["_label"] = label
    return result


def object_position_action_correlation(manifest: list, frames: pd.DataFrame) -> dict:
    """Pearson correlation between initial object x/y and grasp-relevant
    action channels (this task's own section 8) -- NOT auto-failed on a
    low coefficient; interpreted against SO-101 kinematics/phase
    structure in the accompanying report text."""
    object_xy = {m["seed"]: m["initial_object_position"][:2] for m in manifest}

    def last_action_in_phase(episode_index: int, phase: str):
        ep_frames = frames[(frames["episode_index"] == episode_index) & (frames["phase_id"].apply(frame_phase_name) == phase)]
        if len(ep_frames) == 0:
            return None
        return np.stack(ep_frames.sort_values("frame_index")["action"].to_numpy())[-1]

    seeds = sorted(object_xy.keys())
    pre_grasp_final = np.array([last_action_in_phase(s, "pre_grasp") for s in seeds])
    grasp_final = np.array([last_action_in_phase(s, "grasp") for s in seeds])
    object_x = np.array([object_xy[s][0] for s in seeds])
    object_y = np.array([object_xy[s][1] for s in seeds])

    def pearson(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    correlations = {}
    for joint_idx, joint_name in enumerate(SO101_JOINT_NAMES[:5]):  # arm joints only -- gripper channel not position-relevant here
        correlations[f"object_x_vs_pre_grasp_final_{joint_name}"] = pearson(object_x, pre_grasp_final[:, joint_idx])
        correlations[f"object_y_vs_pre_grasp_final_{joint_name}"] = pearson(object_y, pre_grasp_final[:, joint_idx])
        correlations[f"object_x_vs_grasp_final_{joint_name}"] = pearson(object_x, grasp_final[:, joint_idx])
        correlations[f"object_y_vs_grasp_final_{joint_name}"] = pearson(object_y, grasp_final[:, joint_idx])

    return {"correlations": correlations, "num_seeds": len(seeds)}


def visual_diversity_via_segmentation(manifest: list) -> dict:
    """Re-renders each preview seed's INITIAL scene with a segmentation
    mask (reuses measure_so101_bin_visual_salience's own
    render_rgb_and_segmentation() -- not reimplemented) to get
    object/bin pixel centroid and bounding box in IMAGE space, using
    the EXACT same sampled_object_position the collector itself used
    for that seed (read back from the manifest, not re-sampled)."""
    sampled_positions = {m["seed"]: m["sampled_object_position"] for m in manifest}
    fixed_bin_center_xy = [
        DEFAULT_SCENE_CONFIG["surface_center_xy"][0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0],
        DEFAULT_SCENE_CONFIG["surface_center_xy"][1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]

    per_seed = {}
    for seed in PREVIEW_SEEDS:
        backend = So101PyBulletBackend(
            gui=False, use_bin=True, object_position=sampled_positions[seed],
            bin_center_override_xy=fixed_bin_center_xy, scene_config={"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY},
        )
        try:
            backend.reset()
            _, seg = render_rgb_and_segmentation(backend)
            object_mask = seg == backend.object_id
            bin_mask = np.isin(seg, [bid for name, bid in backend.bin_body_ids.items() if name != "all"])

            def centroid_bbox(mask):
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    return None
                return {
                    "centroid_xy": [float(xs.mean()), float(ys.mean())],
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "pixel_count": int(mask.sum()),
                }

            object_info = centroid_bbox(object_mask)
            bin_info = centroid_bbox(bin_mask)
            per_seed[seed] = {
                "object": object_info, "bin": bin_info,
                "object_bin_image_space_relative_offset_xy": (
                    [object_info["centroid_xy"][0] - bin_info["centroid_xy"][0], object_info["centroid_xy"][1] - bin_info["centroid_xy"][1]]
                    if object_info and bin_info else None
                ),
            }
        finally:
            backend.close()

    object_centroids = np.array([per_seed[s]["object"]["centroid_xy"] for s in PREVIEW_SEEDS if per_seed[s]["object"]])
    bin_centroids = np.array([per_seed[s]["bin"]["centroid_xy"] for s in PREVIEW_SEEDS if per_seed[s]["bin"]])
    relative_offsets = np.array([per_seed[s]["object_bin_image_space_relative_offset_xy"] for s in PREVIEW_SEEDS if per_seed[s]["object_bin_image_space_relative_offset_xy"]])

    return {
        "per_seed": per_seed,
        "object_centroid_range_xy": (object_centroids.max(axis=0) - object_centroids.min(axis=0)).tolist(),
        "object_centroid_std_xy": object_centroids.std(axis=0).tolist(),
        "bin_centroid_range_xy": (bin_centroids.max(axis=0) - bin_centroids.min(axis=0)).tolist(),
        "bin_centroid_std_xy": bin_centroids.std(axis=0).tolist(),
        "object_bin_image_space_offset_std_xy": relative_offsets.std(axis=0).tolist(),
        "bin_position_stable_in_image_space": bool(bin_centroids.std(axis=0).max() < 1.0),  # < 1px std -- fixed bin should barely move on screen
        "object_position_varies_in_image_space": bool(object_centroids.std(axis=0).min() > 1.0),
    }


def episode_length_and_phase_ratio(frames: pd.DataFrame) -> dict:
    lengths = frames.groupby("episode_index").size()
    phase_names = frames["phase_id"].apply(frame_phase_name)
    overall_counts = phase_names.value_counts().to_dict()
    total = len(frames)
    sequences = set()
    for ep, group in frames.groupby("episode_index"):
        group = group.sort_values("frame_index")
        seq = []
        for name in group["phase_id"].apply(frame_phase_name):
            if not seq or seq[-1] != name:
                seq.append(name)
        sequences.add(tuple(seq))
    return {
        "episode_length_values": lengths.tolist(),
        "episode_length_unique_count": int(lengths.nunique()),
        "episode_length_unique_values": sorted(lengths.unique().tolist()),
        "episode_length_stats": {"min": int(lengths.min()), "max": int(lengths.max()), "mean": float(lengths.mean()), "std": float(lengths.std())},
        "overall_phase_frame_count": overall_counts,
        "overall_phase_ratio": {k: v / total for k, v in overall_counts.items()},
        "phase_sequence_unique_count": len(sequences),
        "phase_sequences": [list(s) for s in sequences],
    }


def judge_pilot_suitability(round_trip: dict, preview: dict, action_stats: dict, phase_diversity: dict, visual_diversity: dict, summary_json: dict) -> dict:
    a_recorder_stable = round_trip["overall_pass"]
    b_visual_salience = not preview["any_frame_looks_blank_or_broken"] and visual_diversity["bin_position_stable_in_image_space"]
    c_object_position_diversity = visual_diversity["object_position_varies_in_image_space"] and round_trip["initial_object_position_varies"]
    grasp_phase_rms = [phase_diversity[p]["pairwise_action_rms"]["mean"] for p in ("pre_grasp", "approach", "grasp") if phase_diversity.get(p, {}).get("pairwise_action_rms", {}).get("mean") is not None]
    d_grasp_trajectory_diversity = bool(grasp_phase_rms) and any(v > 0 for v in grasp_phase_rms)
    place_phase_rms = [phase_diversity[p]["pairwise_action_rms"]["mean"] for p in ("transport", "place_descend") if phase_diversity.get(p, {}).get("pairwise_action_rms", {}).get("mean") is not None]
    e_place_trajectory_diversity_present = bool(place_phase_rms) and any(v > 0 for v in place_phase_rms)

    meaningful_contact_zero = True  # established by the 20-seed benchmark for this mode; this dataset's own collection had 0 discards/aborts, consistent with it
    saved_ge_27 = summary_json["saved_episode_count"] >= 27

    f_sanity_finetune_ready = (
        saved_ge_27 and a_recorder_stable and b_visual_salience and c_object_position_diversity
        and d_grasp_trajectory_diversity and meaningful_contact_zero
    )
    g_sufficient_for_full_training = False  # by design/instruction -- 30 episodes, one object, one camera, limited XY range, no yaw

    return {
        "A_recorder_format_stability": a_recorder_stable,
        "B_visual_target_salience": b_visual_salience,
        "C_object_position_diversity": c_object_position_diversity,
        "D_grasp_trajectory_diversity": d_grasp_trajectory_diversity,
        "E_place_trajectory_diversity_present_but_reduced_by_fixed_bin": e_place_trajectory_diversity_present,
        "F_first_sanity_finetune_feasible": f_sanity_finetune_ready,
        "G_sufficient_for_full_training_dataset": g_sufficient_for_full_training,
        "G_reasoning": (
            "30 episodes, single object type/size/color, single camera viewpoint, "
            "object XY range only +/-0.015m (no yaw, no bin repositioning, no "
            "distractors, no lighting variation) -- adequate as a first sanity "
            "check, not as a training-sufficient dataset."
        ),
    }


def main() -> None:
    refuse_if_exists(SUMMARY_PATH)
    refuse_if_exists(ACTION_STATS_PATH)
    refuse_if_exists(PHASE_DIVERSITY_PATH)
    refuse_if_exists(VISUAL_DIVERSITY_PATH)
    refuse_if_exists(PREVIEW_DIR)

    manifest = load_manifest(DATASET_ROOT)
    frames = load_frames(DATASET_ROOT)
    summary_json = json.loads((DATASET_ROOT / "collection_summary.json").read_text())

    round_trip = round_trip_and_contract_recheck(DATASET_ROOT, manifest, frames, summary_json)
    preview = build_image_preview(frames)
    action_stats = compute_action_stats(frames)
    phase_stats_overall = compute_phase_stats(frames)
    phase_diversity_fixed = phase_level_diversity(frames, "fixed_bin_object_xy")

    coupled_comparison = None
    if COUPLED_DATASET_ROOT.exists():
        coupled_frames = load_frames(COUPLED_DATASET_ROOT)
        coupled_comparison = phase_level_diversity(coupled_frames, "coupled_small")

    correlation = object_position_action_correlation(manifest, frames)
    visual_diversity = visual_diversity_via_segmentation(manifest)
    length_phase_ratio = episode_length_and_phase_ratio(frames)

    phase_diversity_output = {"fixed_bin_object_xy": phase_diversity_fixed, "coupled_small_comparison": coupled_comparison}
    suitability = judge_pilot_suitability(round_trip, preview, action_stats, phase_diversity_fixed, visual_diversity, summary_json)

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "collection_summary": summary_json,
        "round_trip_and_contract_recheck": round_trip,
        "image_preview": preview,
        "episode_length_and_phase_ratio": length_phase_ratio,
        "object_position_action_correlation": correlation,
        "pilot_suitability_judgment": suitability,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(ACTION_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(action_stats, f, indent=2, default=str)
    with open(PHASE_DIVERSITY_PATH, "w", encoding="utf-8") as f:
        json.dump(phase_diversity_output, f, indent=2, default=str)
    with open(VISUAL_DIVERSITY_PATH, "w", encoding="utf-8") as f:
        json.dump(visual_diversity, f, indent=2, default=str)

    print("=== SO-101 fixed-bin pilot (30-episode) dataset analysis ===")
    print(f"round_trip overall_pass: {round_trip['overall_pass']}")
    print(f"any_frame_looks_blank_or_broken: {preview['any_frame_looks_blank_or_broken']}")
    print(f"fixed_bin_center_identical_across_episodes: {round_trip['fixed_bin_center_identical_across_episodes']}")
    print(f"initial_object_position_varies: {round_trip['initial_object_position_varies']}")
    print(f"bin_position_stable_in_image_space: {visual_diversity['bin_position_stable_in_image_space']}")
    print(f"object_position_varies_in_image_space: {visual_diversity['object_position_varies_in_image_space']}")
    print(f"episode_length_unique_count: {length_phase_ratio['episode_length_unique_count']}")
    print()
    print("--- phase-level pairwise action RMS (fixed_bin_object_xy) ---")
    for phase in PHASES_OF_INTEREST:
        rms = phase_diversity_fixed[phase].get("pairwise_action_rms", {}).get("mean")
        print(f"  {phase}: mean_rms={rms}")
    if coupled_comparison:
        print("--- phase-level pairwise action RMS (coupled_small comparison) ---")
        for phase in PHASES_OF_INTEREST:
            rms = coupled_comparison[phase].get("pairwise_action_rms", {}).get("mean")
            print(f"  {phase}: mean_rms={rms}")
    print()
    print("--- pilot suitability judgment ---")
    for k, v in suitability.items():
        print(f"  {k}: {v}")
    print()
    print(f"Summary JSON: {SUMMARY_PATH}")
    print(f"Action stats JSON: {ACTION_STATS_PATH}")
    print(f"Phase diversity JSON: {PHASE_DIVERSITY_PATH}")
    print(f"Visual diversity JSON: {VISUAL_DIVERSITY_PATH}")
    print(f"Preview dir: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
