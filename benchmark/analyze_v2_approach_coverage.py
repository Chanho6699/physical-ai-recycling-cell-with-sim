"""V2 Dataset Coverage -- Minimal Diagnostic (see this task's chat report).

Read-only analysis of the ALREADY-COLLECTED datasets/recycling_v2_train160
(local/recycling_cell_v2_train160) -- no simulator rollout, no dataset
writes, no production code changes. Single integrated script, no
clustering, no research framework.

Data available directly in the dataset:
  - data/chunk-000/file-000.parquet: per-frame observation.state (8-dim:
    ee_position[3], ee_orientation_axis_angle[3], gripper_qpos[2] -- see
    PyBulletPandaBackend.get_libero_observation_state()) and action
    (7-dim: [dx,dy,dz,drx,dry,drz,gripper] -- see collect_recycling_dataset.py
    / policy/dummy_openvla_policy.py), keyed by episode_index/frame_index.
  - collection_manifest.jsonl: one line per collected episode, in the
    SAME order as episode_index 0..159 (verified: 160 lines, all
    success=true/saved=true -- collect_v2_dataset.py's collect_split()
    increments the dataset's own episode counter and the manifest's
    "attempt" counter in lockstep for an all-success run, so line index
    == episode_index here). Has the episode's OWN object position
    ("position") and bin_position -- NOT stored per-frame anywhere.

What is NOT directly available, and how this script handles it:
  - Per-frame object position: the dataset has no ground-truth object
    pose track (PyBulletPandaBackend's own held/grasp state is not a
    recorded feature either). The object is physically stationary from
    episode start until the moment it is actually grasped (a fixed
    pybullet constraint then moves it with the gripper) -- so
    manifest's constant per-episode "position" is a VALID stand-in for
    ee-object distance ONLY for frames up to and including the episode's
    first close-gripper frame (action[6] >= 0.5, the same threshold
    action_adapter/adapter_v0.py's ActionAdapter uses). Frames after
    that point may already be carrying the object, so a distance
    computed against the STALE initial position there would be
    meaningless (large) noise, not a real proximity measurement -- this
    script deliberately EXCLUDES post-first-close frames from every
    distance-based statistic and says so explicitly in the report,
    rather than silently computing a misleading number.

Run:
  .venv-vla/bin/python -m benchmark.analyze_v2_approach_coverage
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "recycling_v2_train160"
OUTPUT_JSON = PROJECT_ROOT / "results" / "dataset_analysis" / "v2_approach_coverage.json"
OUTPUT_MD = PROJECT_ROOT / "results" / "dataset_analysis" / "v2_approach_coverage.md"
OUTPUT_HIST = PROJECT_ROOT / "results" / "dataset_analysis" / "v2_approach_coverage_distance_histogram.png"

GRIPPER_CLOSE_THRESHOLD = 0.5  # matches action_adapter/adapter_v0.py's ActionAdapter
DISTANCE_BINS = [
    ("> 0.20m", 0.20, float("inf")),
    ("0.10m ~ 0.20m", 0.10, 0.20),
    ("0.05m ~ 0.10m", 0.05, 0.10),
    ("0.03m ~ 0.05m", 0.03, 0.05),
    ("<= 0.03m", -0.001, 0.03),
]
NEAR_DUPLICATE_TRAJ_THRESHOLD_M = 0.01  # smaller than OBJECT_JITTER_RADIUS_M (0.015m) -- see v2_dataset_positions.py
FINE_CORRECTION_BAND = (0.05, 0.10)
FINE_CORRECTION_MIN_STEPS = 3
DIRECTION_REVERSAL_MIN_RISE_M = 0.02
SIGN_FLIP_NOISE_FLOOR_M = 0.005


def _dist(a, b) -> float:
    return float(math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))))


def load_manifest(dataset_root: Path = DATASET_ROOT) -> list:
    """dataset_root is parameterized (default: this module's own
    DATASET_ROOT, so every existing call site/CLI behavior here is
    unchanged) so benchmark/analyze_v3_recovery_smoke.py can reuse this
    exact function against a different dataset root, per this task's
    chat report ('analyze_v2_approach_coverage.py의 재사용 가능 부분')."""
    lines = []
    with open(dataset_root / "collection_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))
    if not all(l["success"] and l["saved"] for l in lines):
        raise RuntimeError("collection_manifest.jsonl has non-saved/failed attempts -- episode_index alignment assumption would break.")
    return lines


def load_frames(dataset_root: Path = DATASET_ROOT) -> pd.DataFrame:
    """Reads and concatenates EVERY parquet file under data/ (LeRobotDataset
    auto-splits into multiple chunk-*/file-*.parquet files once a dataset
    gets large enough -- v2_train160/smoke50 were small enough to fit in a
    single data/chunk-000/file-000.parquet, but a 500-episode v3 dataset
    was found (this task's chat report) to split into 3 files; reading
    only file-000 silently dropped ~50% of episodes, which then crashed
    downstream per-episode-index lookups with an out-of-bounds error --
    fixed here once, for every caller, rather than special-cased per
    dataset size."""
    parquet_paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)


def build_episode_records(manifest: list, frames: pd.DataFrame) -> list:
    records = []
    for episode_index, manifest_entry in enumerate(manifest):
        object_position = manifest_entry["position"]
        bin_position = manifest_entry["bin_position"]
        ep_frames = frames[frames["episode_index"] == episode_index].sort_values("frame_index")

        ee_positions = [list(s[0:3]) for s in ep_frames["observation.state"]]
        gripper_raws = [float(a[6]) for a in ep_frames["action"]]
        distances_full = [_dist(ee, object_position) for ee in ee_positions]

        first_close_frame = next((i for i, g in enumerate(gripper_raws) if g >= GRIPPER_CLOSE_THRESHOLD), None)
        approach_end = (first_close_frame + 1) if first_close_frame is not None else len(ee_positions)

        approach_distances = distances_full[:approach_end]
        approach_ee = ee_positions[:approach_end]

        min_distance = min(approach_distances) if approach_distances else None
        min_distance_step = int(np.argmin(approach_distances)) if approach_distances else None
        first_approach_step = next((i for i, d in enumerate(approach_distances) if d <= 0.10), None)
        first_grasp_zone_step = next((i for i, d in enumerate(approach_distances) if d <= 0.05), None)
        close_distance = distances_full[first_close_frame] if first_close_frame is not None else None

        # close-attempt blocks (item 4.D): contiguous runs of gripper_raw >= threshold,
        # over the FULL episode (a retry could in principle open then re-close later).
        close_blocks = 0
        prev_closed = False
        for g in gripper_raws:
            closed = g >= GRIPPER_CLOSE_THRESHOLD
            if closed and not prev_closed:
                close_blocks += 1
            prev_closed = closed

        records.append({
            "episode_index": episode_index,
            "object_anchor_name": manifest_entry["object_anchor_name"],
            "bin_name": manifest_entry["bin_name"],
            "object_position": object_position,
            "bin_position": bin_position,
            "seed": manifest_entry["seed"],
            "num_frames_total": len(ee_positions),
            "ee_initial_position": ee_positions[0] if ee_positions else None,
            "first_close_frame": first_close_frame,
            "approach_segment_len": approach_end,
            "approach_distances": approach_distances,
            "approach_relative_positions": [
                [ee[i] - object_position[i] for i in range(3)] for ee in approach_ee
            ],
            "min_distance": min_distance,
            "min_distance_step": min_distance_step,
            "first_approach_step": first_approach_step,
            "first_grasp_zone_step": first_grasp_zone_step,
            "close_distance": close_distance,
            "close_blocks": close_blocks,
        })
    return records


def basic_stats(manifest: list, frames: pd.DataFrame, records: list) -> dict:
    frame_counts = [r["num_frames_total"] for r in records]
    unique_object_positions = len({tuple(r["object_position"]) for r in records})
    unique_bin_positions = len({tuple(r["bin_position"]) for r in records})
    ee_initial_raw = [tuple(r["ee_initial_position"]) for r in records if r["ee_initial_position"] is not None]
    unique_ee_initial_exact = len(set(ee_initial_raw))
    unique_ee_initial_rounded_1e4 = len({tuple(round(v, 4) for v in p) for p in ee_initial_raw})
    return {
        "num_episodes": len(records),
        "total_frames": int(sum(frame_counts)),
        "frames_per_episode_mean": float(np.mean(frame_counts)),
        "frames_per_episode_min": int(np.min(frame_counts)),
        "frames_per_episode_max": int(np.max(frame_counts)),
        "unique_object_initial_positions_exact": unique_object_positions,
        "unique_bin_positions_exact": unique_bin_positions,
        "unique_object_anchor_names": len({r["object_anchor_name"] for r in records}),
        "unique_bin_names": len({r["bin_name"] for r in records}),
        "unique_ee_initial_positions_exact": unique_ee_initial_exact,
        "unique_ee_initial_positions_rounded_1e-4m": unique_ee_initial_rounded_1e4,
    }


def distance_band_analysis(records: list) -> dict:
    all_approach_distances = [d for r in records for d in r["approach_distances"]]
    total = len(all_approach_distances)
    bins = {}
    for name, lo, hi in DISTANCE_BINS:
        count = sum(1 for d in all_approach_distances if lo < d <= hi) if hi != float("inf") else sum(1 for d in all_approach_distances if d > lo)
        bins[name] = {"count": count, "fraction": count / total if total else None}

    min_distances = [r["min_distance"] for r in records if r["min_distance"] is not None]
    n_reach_010 = sum(1 for r in records if r["first_approach_step"] is not None)
    n_reach_005 = sum(1 for r in records if r["first_grasp_zone_step"] is not None)
    n_never_close = sum(1 for r in records if r["first_close_frame"] is None)

    close_distances = [r["close_distance"] for r in records if r["close_distance"] is not None]

    # "close 직전 근거리 frame이 충분히 반복되는가" -- for each episode, count how
    # many of the LAST 5 approach frames (immediately before the close frame) are
    # already <= 0.05m.
    near_frames_before_close = []
    for r in records:
        if r["first_close_frame"] is None:
            continue
        window = r["approach_distances"][max(0, r["first_close_frame"] - 4): r["first_close_frame"] + 1]
        near_frames_before_close.append(sum(1 for d in window if d <= 0.05))

    return {
        "total_approach_frames": total,
        "distance_bins": bins,
        "fraction_within_0_05m": (bins["<= 0.03m"]["count"] + bins["0.03m ~ 0.05m"]["count"]) / total if total else None,
        "min_distance_per_episode_mean": float(np.mean(min_distances)) if min_distances else None,
        "min_distance_per_episode_median": float(np.median(min_distances)) if min_distances else None,
        "min_distance_per_episode_std": float(np.std(min_distances)) if min_distances else None,
        "episodes_ever_reaching_0_10m": n_reach_010,
        "episodes_ever_reaching_0_05m": n_reach_005,
        "episodes_ever_reaching_0_05m_fraction": n_reach_005 / len(records),
        "episodes_never_issuing_close": n_never_close,
        "close_distance_mean": float(np.mean(close_distances)) if close_distances else None,
        "close_distance_median": float(np.median(close_distances)) if close_distances else None,
        "close_distance_std": float(np.std(close_distances)) if close_distances else None,
        "near_frames_in_last_5_before_close_mean": float(np.mean(near_frames_before_close)) if near_frames_before_close else None,
        "near_frames_in_last_5_before_close_min": int(np.min(near_frames_before_close)) if near_frames_before_close else None,
    }


def _aligned_last_n(relative_positions: list, n: int) -> list:
    """Last n relative-position vectors (ee - object), time-aligned by
    'steps before close' (index -1 = the close frame itself), padded by
    repeating the first available vector if the episode's approach
    segment is shorter than n -- so every episode contributes a
    length-n sequence for pairwise comparison."""
    tail = relative_positions[-n:]
    if len(tail) < n and tail:
        pad = [tail[0]] * (n - len(tail))
        tail = pad + tail
    return tail


def trajectory_diversity_analysis(records: list) -> dict:
    n_window = 10
    initial_relative = [tuple(round(v, 3) for v in r["approach_relative_positions"][0]) for r in records if r["approach_relative_positions"]]
    unique_initial_relative = len(set(initial_relative))

    tails = {}
    for r in records:
        if not r["approach_relative_positions"]:
            continue
        tails[r["episode_index"]] = _aligned_last_n(r["approach_relative_positions"], n_window)

    indices = list(tails.keys())
    pairwise_dists = []
    near_duplicate_partner = {i: False for i in indices}
    for a_idx in range(len(indices)):
        for b_idx in range(a_idx + 1, len(indices)):
            i, j = indices[a_idx], indices[b_idx]
            seq_i, seq_j = tails[i], tails[j]
            per_step = [math.sqrt(sum((seq_i[k][d] - seq_j[k][d]) ** 2 for d in range(3))) for k in range(n_window)]
            mean_d = float(np.mean(per_step))
            pairwise_dists.append(mean_d)
            if mean_d <= NEAR_DUPLICATE_TRAJ_THRESHOLD_M:
                near_duplicate_partner[i] = True
                near_duplicate_partner[j] = True

    frac_with_near_duplicate = sum(near_duplicate_partner.values()) / len(near_duplicate_partner) if near_duplicate_partner else None

    # within-anchor vs across-anchor mean trajectory distance (does object
    # position actually change the path SHAPE, or is it the same regardless?)
    anchor_of = {r["episode_index"]: r["object_anchor_name"] for r in records}
    within, across = [], []
    for a_idx in range(len(indices)):
        for b_idx in range(a_idx + 1, len(indices)):
            i, j = indices[a_idx], indices[b_idx]
            seq_i, seq_j = tails[i], tails[j]
            per_step = [math.sqrt(sum((seq_i[k][d] - seq_j[k][d]) ** 2 for d in range(3))) for k in range(n_window)]
            mean_d = float(np.mean(per_step))
            (within if anchor_of[i] == anchor_of[j] else across).append(mean_d)

    return {
        "unique_initial_relative_positions_rounded_1mm": unique_initial_relative,
        "num_episode_pairs_compared": len(pairwise_dists),
        "mean_pairwise_last10_trajectory_distance_m": float(np.mean(pairwise_dists)) if pairwise_dists else None,
        "near_duplicate_threshold_m": NEAR_DUPLICATE_TRAJ_THRESHOLD_M,
        "fraction_episodes_with_near_duplicate_trajectory": frac_with_near_duplicate,
        "within_anchor_mean_trajectory_distance_m": float(np.mean(within)) if within else None,
        "across_anchor_mean_trajectory_distance_m": float(np.mean(across)) if across else None,
        "within_vs_across_ratio": (float(np.mean(within)) / float(np.mean(across))) if within and across and np.mean(across) > 0 else None,
    }


def recovery_analysis(records: list) -> dict:
    type_a, type_b, type_c, type_d = [], [], [], []

    for r in records:
        d = r["approach_distances"]
        rel = r["approach_relative_positions"]

        # A: approach, retreat by >= threshold, then approach again below the prior min.
        found_a = False
        if len(d) >= 3:
            running_min = d[0]
            retreated_from = None
            for k in range(1, len(d)):
                if d[k] > running_min + DIRECTION_REVERSAL_MIN_RISE_M and retreated_from is None:
                    retreated_from = running_min
                elif retreated_from is not None and d[k] < retreated_from:
                    found_a = True
                    break
                running_min = min(running_min, d[k])
        if found_a:
            type_a.append(r["episode_index"])

        # B: x or y relative-error sign flip (both sides exceeding the noise floor).
        found_b = False
        for axis in (0, 1):
            vals = [p[axis] for p in rel]
            pos = [v for v in vals if v > SIGN_FLIP_NOISE_FLOOR_M]
            neg = [v for v in vals if v < -SIGN_FLIP_NOISE_FLOOR_M]
            if pos and neg:
                # only counts as a flip if the sign changes over TIME, not just noise scatter
                signs = [1 if v > SIGN_FLIP_NOISE_FLOOR_M else (-1 if v < -SIGN_FLIP_NOISE_FLOOR_M else 0) for v in vals]
                signs = [s for s in signs if s != 0]
                if len(set(signs)) > 1 and any(signs[k] != signs[k - 1] for k in range(1, len(signs))):
                    found_b = True
        if found_b:
            type_b.append(r["episode_index"])

        # C: dwelling in the 0.05-0.10m fine-correction band for >= N steps.
        lo, hi = FINE_CORRECTION_BAND
        n_in_band = sum(1 for v in d if lo <= v <= hi)
        if n_in_band >= FINE_CORRECTION_MIN_STEPS:
            type_c.append(r["episode_index"])

        # D: multiple distinct close-attempt blocks (close, open, close again).
        if r["close_blocks"] >= 2:
            type_d.append(r["episode_index"])

    def summarize(name, episode_indices, records_by_index):
        examples = episode_indices[:3]
        example_details = [
            {
                "episode_index": i,
                "anchor": records_by_index[i]["object_anchor_name"],
                "bin": records_by_index[i]["bin_name"],
                "seed": records_by_index[i]["seed"],
            }
            for i in examples
        ]
        return {
            "name": name, "count": len(episode_indices), "fraction": len(episode_indices) / len(records),
            "example_episodes": example_details,
        }

    records_by_index = {r["episode_index"]: r for r in records}
    return {
        "A_approach_retreat_reapproach": summarize("A: 접근-후퇴-재접근", type_a, records_by_index),
        "B_error_sign_flip_recentering": summarize("B: x/y 오차 방향 반전 재정렬", type_b, records_by_index),
        "C_fine_correction_dwell_0.05_0.10m": summarize("C: 0.05-0.10m 미세보정 체류", type_c, records_by_index),
        "D_close_retry_after_failed_close": summarize("D: close 실패 후 재접근", type_d, records_by_index),
    }


def make_histogram(records: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping histogram (not required, per task spec).")
        return
    all_d = [d for r in records for d in r["approach_distances"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_d, bins=40, color="#4c72b0", edgecolor="white")
    ax.axvline(0.05, color="red", linestyle="--", label="GRASP_THRESHOLD (0.05m)")
    ax.set_xlabel("EE-object distance (m), approach-segment frames only")
    ax.set_ylabel("frame count")
    ax.set_title("v2_train160: EE-object distance distribution (approach segment)")
    ax.legend()
    fig.tight_layout()
    OUTPUT_HIST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_HIST, dpi=120)
    plt.close(fig)


def render_markdown(result: dict) -> str:
    b = result["basic_stats"]
    d = result["distance_band_analysis"]
    t = result["trajectory_diversity"]
    r = result["recovery_analysis"]
    lines = []
    lines.append("# V2 Dataset Coverage -- Minimal Diagnostic\n")
    lines.append("## 1. 기본 통계\n")
    lines.append(f"- episode 수: {b['num_episodes']}")
    lines.append(f"- 총 frame 수: {b['total_frames']}")
    lines.append(f"- episode당 frame 수 평균/최소/최대: {b['frames_per_episode_mean']:.2f} / {b['frames_per_episode_min']} / {b['frames_per_episode_max']}")
    lines.append(f"- 고유 object 초기 위치 수 (exact): {b['unique_object_initial_positions_exact']} / {b['num_episodes']} (jitter로 거의 전부 고유)")
    lines.append(f"- 고유 bin 위치 수: {b['unique_bin_positions_exact']} (object anchor 16개, bin 5개)")
    lines.append(f"- 고유 ee 초기 위치 수 (exact / 1e-4m 반올림): {b['unique_ee_initial_positions_exact']} / {b['unique_ee_initial_positions_rounded_1e-4m']} -- 매 episode 동일한 reset pose에서 시작\n")

    lines.append("## 2. 거리 구간별 frame 비율 (approach segment만, 아래 방법론 참고)\n")
    lines.append("| 구간 | frame 수 | 비율 |")
    lines.append("|---|---|---|")
    for name, stat in d["distance_bins"].items():
        lines.append(f"| {name} | {stat['count']} | {stat['fraction']:.2%} |")
    lines.append(f"\n- 0.05m 이내 비율: {d['fraction_within_0_05m']:.2%}")
    lines.append(f"- 0.10m까지 도달한 episode: {d['episodes_ever_reaching_0_10m']}/{b['num_episodes']}")
    lines.append(f"- 0.05m까지 도달한 episode: {d['episodes_ever_reaching_0_05m']}/{b['num_episodes']} ({d['episodes_ever_reaching_0_05m_fraction']:.2%})")
    lines.append(f"- close 명령이 전혀 없는 episode: {d['episodes_never_issuing_close']}")
    lines.append(f"- episode별 최소거리 평균/중앙값/표준편차: {d['min_distance_per_episode_mean']:.4f} / {d['min_distance_per_episode_median']:.4f} / {d['min_distance_per_episode_std']:.4f} m")
    lines.append(f"- close 당시 거리 평균/중앙값/표준편차: {d['close_distance_mean']:.4f} / {d['close_distance_median']:.4f} / {d['close_distance_std']:.4f} m")
    lines.append(f"- close 직전 마지막 5 frame 중 0.05m 이내 frame 수 평균/최소: {d['near_frames_in_last_5_before_close_mean']:.2f} / {d['near_frames_in_last_5_before_close_min']}\n")

    lines.append("## 3. 접근 경로 다양성\n")
    lines.append(f"- 초기 상대 위치(ee-object, 1mm 반올림) 고유 개수: {t['unique_initial_relative_positions_rounded_1mm']} / {b['num_episodes']}")
    lines.append(f"- close 직전 10-step trajectory 간 평균 pairwise 차이: {t['mean_pairwise_last10_trajectory_distance_m']:.4f} m ({t['num_episode_pairs_compared']} pairs)")
    lines.append(f"- 거의 동일 trajectory({t['near_duplicate_threshold_m']}m 이하) 반복 episode 비율: {t['fraction_episodes_with_near_duplicate_trajectory']:.2%}")
    lines.append(f"- anchor 내부 평균 trajectory 차이: {t['within_anchor_mean_trajectory_distance_m']:.4f} m")
    lines.append(f"- anchor 간(다른 object 위치) 평균 trajectory 차이: {t['across_anchor_mean_trajectory_distance_m']:.4f} m")
    ratio = t['within_vs_across_ratio']
    lines.append(f"- within/across 비율: {ratio:.3f} (1에 가까울수록 object 위치가 달라져도 접근 경로 형태가 사실상 동일함을 의미)\n")

    lines.append("## 4. Recovery 존재 여부\n")
    for key in ["A_approach_retreat_reapproach", "B_error_sign_flip_recentering", "C_fine_correction_dwell_0.05_0.10m", "D_close_retry_after_failed_close"]:
        entry = r[key]
        lines.append(f"### {entry['name']}")
        lines.append(f"- episode 수: {entry['count']} / {b['num_episodes']} ({entry['fraction']:.2%})")
        if entry["example_episodes"]:
            examples_str = ", ".join(f"ep{e['episode_index']}({e['anchor']}+{e['bin']})" for e in entry["example_episodes"])
            lines.append(f"- 대표 사례: {examples_str}")
        else:
            lines.append("- 대표 사례: 없음")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    manifest = load_manifest()
    frames = load_frames()
    records = build_episode_records(manifest, frames)

    result = {
        "dataset": "local/recycling_cell_v2_train160",
        "dataset_root": str(DATASET_ROOT),
        "methodology_note": (
            "Per-frame object position is not stored; the object is stationary until "
            "the first close-gripper frame (action[6]>=0.5), so all distance statistics "
            "below are computed ONLY over each episode's approach segment (frames 0..first_close_frame "
            "inclusive). Frames after grasp are excluded, not estimated."
        ),
        "basic_stats": basic_stats(manifest, frames, records),
        "distance_band_analysis": distance_band_analysis(records),
        "trajectory_diversity": trajectory_diversity_analysis(records),
        "recovery_analysis": recovery_analysis(records),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    md = render_markdown(result)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    make_histogram(records)

    print(md)
    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")
    if OUTPUT_HIST.exists():
        print(f"Histogram: {OUTPUT_HIST}")


if __name__ == "__main__":
    main()
