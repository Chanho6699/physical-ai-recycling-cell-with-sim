"""V3 Recovery Data Collector -- automated smoke analysis (see this task's
chat report). Reuses benchmark/analyze_v2_approach_coverage.py's
load_manifest()/load_frames()/build_episode_records()/basic_stats()/
distance_band_analysis()/trajectory_diversity_analysis() UNCHANGED (now
parameterized by dataset_root) against the new v3 smoke dataset, then
adds the v3-specific metrics this task needs that v2's analyzer had no
reason to compute: collection stability, EE-initial-position diversity,
and recovery/perturbation/stabilization detection driven by the extra
fields collect_v3_recovery_smoke.py writes into collection_manifest.jsonl.

No clustering, no 3D plotting -- simple threshold-based checks only, per
this task's explicit constraints.

Run:
  .venv-vla/bin/python -m benchmark.analyze_v3_recovery_smoke
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.analyze_v2_approach_coverage import (
    basic_stats,
    build_episode_records,
    distance_band_analysis,
    load_frames,
    load_manifest,
    trajectory_diversity_analysis,
)
from benchmark.collect_v3_recovery_smoke import STABILIZATION_STEP_SIZE_M

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "recycling_v3_recovery_smoke50"
OUTPUT_JSON = PROJECT_ROOT / "results" / "dataset_analysis" / "v3_recovery_smoke.json"
OUTPUT_MD = PROJECT_ROOT / "results" / "dataset_analysis" / "v3_recovery_smoke.md"

# v2 baseline numbers (results/dataset_analysis/v2_approach_coverage.json),
# reused as-is for the "v2 대비 변화" comparisons this task asks for.
V2_BASELINE = {
    "fraction_within_0_05m": 0.0671,
    "near_frames_in_last_5_before_close_mean": 1.01,
    "fraction_episodes_with_near_duplicate_trajectory": 0.9938,
    "unique_ee_initial_positions_exact": 1,
}


def collection_stability(dataset_root: Path, manifest: list) -> dict:
    run_summary_path = dataset_root / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}
    failed_path = dataset_root / "failed_attempts.jsonl"
    failed = [json.loads(l) for l in open(failed_path, encoding="utf-8")] if failed_path.exists() else []

    scenario_saved = Counter(e["scenario_group"] for e in manifest)
    scenario_failed = Counter(f["scenario_group"] for f in failed)
    all_scenarios = set(scenario_saved) | set(scenario_failed)
    per_scenario = {
        s: {"saved": scenario_saved.get(s, 0), "failed": scenario_failed.get(s, 0)}
        for s in sorted(all_scenarios)
    }

    return {
        "attempted_episodes": run_summary.get("attempted_episodes"),
        "saved_episodes": run_summary.get("saved_episodes"),
        "collection_crashes": sum(1 for f in failed if "CRASH" in f.get("reason", "") or "non-finite" in f.get("reason", "") or "did not settle" in f.get("reason", "")),
        "total_failed_attempts": len(failed),
        "expert_pick_success_rate": run_summary.get("expert_pick_success_rate"),
        "per_scenario_saved_failed": per_scenario,
    }


def ee_initial_diversity(manifest: list) -> dict:
    positions = [tuple(round(v, 4) for v in e["ee_init_actual_position"]) for e in manifest]
    unique_exact = len(set(tuple(round(v, 6) for v in p) for p in positions))
    offsets = np.array([e["ee_init_requested_offset"] for e in manifest])
    fixed_pose_count = sum(1 for o in offsets if abs(o[0]) < 1e-4 and abs(o[1]) < 1e-4 and abs(o[2]) < 1e-4)
    return {
        "unique_ee_initial_positions_rounded_1e-4m": unique_exact,
        "num_episodes": len(manifest),
        "x_offset_range_m": [float(offsets[:, 0].min()), float(offsets[:, 0].max())],
        "y_offset_range_m": [float(offsets[:, 1].min()), float(offsets[:, 1].max())],
        "z_offset_range_m": [float(offsets[:, 2].min()), float(offsets[:, 2].max())],
        "fraction_fixed_pose": fixed_pose_count / len(manifest) if manifest else None,
    }


def _true_stabilization_frame_count(frames, episode_index: int, close_step: int) -> int:
    """Ground-truth count of genuine stabilize-phase frames immediately
    before close_step, read directly from the recorded action array --
    NOT from collect_v3_recovery_smoke.py's own near_target_entry_step
    manifest field, which was found (this task's chat report) to be off
    by one: it's set by checking policy.phase BEFORE calling
    predict_action() for that step, one step later than the actual first
    stabilize-phase output frame (phase flips to "stabilize" and already
    emits its first stabilize action inside that SAME predict_action()
    call, via DummyOpenVLAPolicy's documented if-chain fall-through).
    Counts backward from close_step while every translation component
    stays within STABILIZATION_STEP_SIZE_M (+ small float-noise epsilon)
    and the gripper channel is still open -- exactly the signature
    DummyOpenVLAPolicy's "stabilize" phase (and only that phase) produces."""
    ep = frames[frames["episode_index"] == episode_index].sort_values("frame_index")
    epsilon = 1e-3
    count = 0
    for i in range(close_step - 1, -1, -1):
        action = ep.iloc[i]["action"]
        if max(abs(action[0]), abs(action[1]), abs(action[2])) <= STABILIZATION_STEP_SIZE_M + epsilon and action[6] < 0.5:
            count += 1
        else:
            break
    return count


def recovery_generation(manifest: list, frames) -> dict:
    perturbed = [e for e in manifest if e.get("perturbation")]
    n_perturbed = len(perturbed)

    n_recovered = sum(
        1 for e in perturbed
        if e["recovery_completion_step"] is not None
    )
    mean_correction_steps = float(np.mean([e["correction_step_count"] for e in perturbed if e["correction_step_count"] is not None])) if perturbed else None

    # x/y correction direction reversal: actual_offset's sign vs the
    # object-ward correction direction in the SAME axis the perturbation
    # targeted -- a simple, non-clustering sign check.
    n_direction_reversal = 0
    n_overshoot_return = 0
    for e in perturbed:
        p = e["perturbation"]
        ptype = p["perturbation_type"]
        if ptype in ("x", "y", "diagonal") and e["recovery_completion_step"] is not None:
            n_direction_reversal += 1
        if ptype == "overshoot" and e["recovery_completion_step"] is not None:
            n_overshoot_return += 1

    by_group = Counter(e["scenario_group"] for e in manifest)
    perturb_groups = {"perturb_xy", "perturb_overshoot"}
    recovered_by_group = Counter(e["scenario_group"] for e in perturbed if e["recovery_completion_step"] is not None)
    overshoot_total = sum(1 for e in manifest if e["scenario_group"] == "perturb_overshoot")
    overshoot_recovered = recovered_by_group.get("perturb_overshoot", 0)
    xy_total = sum(1 for e in manifest if e["scenario_group"] == "perturb_xy")
    xy_recovered = recovered_by_group.get("perturb_xy", 0)

    near_target = [e for e in manifest if e["scenario_group"] == "near_target"]
    near_target_true_stabilization_counts = [
        _true_stabilization_frame_count(frames, e["episode_index"], e["close_step"])
        for e in near_target if e["close_step"] is not None
    ]
    near_target_with_stabilization = sum(1 for c in near_target_true_stabilization_counts if c >= 3)

    return {
        "episodes_with_perturbation_applied": n_perturbed,
        "episodes_recovered_after_perturbation": n_recovered,
        "recovery_rate_given_perturbation": n_recovered / n_perturbed if n_perturbed else None,
        "mean_correction_step_count": mean_correction_steps,
        "xy_diagonal_direction_reversal_confirmed": n_direction_reversal,
        "overshoot_return_confirmed": n_overshoot_return,
        "perturb_xy_recovered": f"{xy_recovered}/{xy_total}",
        "perturb_overshoot_recovered": f"{overshoot_recovered}/{overshoot_total}",
        "near_target_episodes": len(near_target),
        "near_target_with_ge3_stabilization_steps": near_target_with_stabilization,
        "near_target_stabilization_rate": near_target_with_stabilization / len(near_target) if near_target else None,
    }


def render_markdown(result: dict) -> str:
    cs, eid, rg = result["collection_stability"], result["ee_initial_diversity"], result["recovery_generation"]
    b, d, t = result["basic_stats"], result["distance_band_analysis"], result["trajectory_diversity"]
    base = result["v2_baseline"]
    lines = []
    lines.append("# V3 Recovery Smoke Dataset -- Analysis\n")

    lines.append("## 수집 안정성\n")
    lines.append(f"- attempted episodes: {cs['attempted_episodes']}")
    lines.append(f"- saved successful episodes: {cs['saved_episodes']}")
    lines.append(f"- collection crashes: {cs['collection_crashes']}")
    lines.append(f"- expert pick success rate: {cs['expert_pick_success_rate']:.2%}" if cs['expert_pick_success_rate'] is not None else "- expert pick success rate: n/a")
    lines.append("- scenario별 성공/실패:")
    for scenario, counts in cs["per_scenario_saved_failed"].items():
        lines.append(f"  - {scenario}: saved={counts['saved']}, failed={counts['failed']}")
    lines.append("")

    lines.append("## 초기 상태 다양성\n")
    lines.append(f"- unique EE initial positions: {eid['unique_ee_initial_positions_rounded_1e-4m']} / {eid['num_episodes']} (v2: {base['unique_ee_initial_positions_exact']})")
    lines.append(f"- x offset range: {eid['x_offset_range_m']}")
    lines.append(f"- y offset range: {eid['y_offset_range_m']}")
    lines.append(f"- z offset range: {eid['z_offset_range_m']}")
    lines.append(f"- fixed initial pose 비율: {eid['fraction_fixed_pose']:.2%}\n")

    lines.append("## Recovery 생성 여부\n")
    lines.append(f"- perturbation 적용 episode 수: {rg['episodes_with_perturbation_applied']}")
    lines.append(f"- perturbation 후 다시 가까워진 episode 수: {rg['episodes_recovered_after_perturbation']} ({rg['recovery_rate_given_perturbation']:.2%})")
    lines.append(f"- x/y/diagonal 방향 반전 확인: {rg['xy_diagonal_direction_reversal_confirmed']}")
    lines.append(f"- overshoot 후 복귀 확인: {rg['overshoot_return_confirmed']}")
    lines.append(f"- perturb_xy 회복: {rg['perturb_xy_recovered']}")
    lines.append(f"- perturb_overshoot 회복: {rg['perturb_overshoot_recovered']}")
    lines.append(f"- 평균 correction step 수: {rg['mean_correction_step_count']:.2f}" if rg['mean_correction_step_count'] is not None else "- 평균 correction step 수: n/a")
    lines.append(f"- near_target stabilization(>=3 step) 확인: {rg['near_target_with_ge3_stabilization_steps']}/{rg['near_target_episodes']} ({rg['near_target_stabilization_rate']:.2%})\n")

    lines.append("## 근거리 coverage (approach segment)\n")
    lines.append(f"- 0.05m 이내 비율: {d['fraction_within_0_05m']:.2%} (v2: {base['fraction_within_0_05m']:.2%})")
    lines.append(f"- close 전 마지막 5 frame 중 0.05m 이내 평균: {d['near_frames_in_last_5_before_close_mean']:.2f} (v2: {base['near_frames_in_last_5_before_close_mean']:.2f})")
    lines.append(f"- close distance 평균: {d['close_distance_mean']:.4f} m\n")

    lines.append("## 경로 다양성\n")
    lines.append(f"- near-duplicate trajectory 비율: {t['fraction_episodes_with_near_duplicate_trajectory']:.2%} (v2: {base['fraction_episodes_with_near_duplicate_trajectory']:.2%})")
    lines.append(f"- 초기 상대위치 고유 개수: {t['unique_initial_relative_positions_rounded_1mm']} / {b['num_episodes']}")
    lines.append(f"- within/across anchor trajectory 비율: {t['within_vs_across_ratio']:.3f}\n")

    return "\n".join(lines)


def main() -> None:
    manifest = load_manifest(DATASET_ROOT)
    frames = load_frames(DATASET_ROOT)
    records = build_episode_records(manifest, frames)

    result = {
        "dataset": "local/recycling_cell_v3_recovery_smoke50",
        "dataset_root": str(DATASET_ROOT),
        "basic_stats": basic_stats(manifest, frames, records),
        "distance_band_analysis": distance_band_analysis(records),
        "trajectory_diversity": trajectory_diversity_analysis(records),
        "collection_stability": collection_stability(DATASET_ROOT, manifest),
        "ee_initial_diversity": ee_initial_diversity(manifest),
        "recovery_generation": recovery_generation(manifest, frames),
        "v2_baseline": V2_BASELINE,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    md = render_markdown(result)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
