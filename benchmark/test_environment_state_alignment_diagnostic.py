"""Tests for benchmark/run_environment_state_alignment_diagnostic.py (v0).

No live GPU/VLA server needed anywhere in this file -- the real-dataset
loader is exercised against actual downloaded HuggingFaceVLA/libero
parquet samples (cached locally by huggingface_hub after the first
call), and the PyBullet-side collectors use the real backend directly
(no policy server). Covers:

  1. estimate_reach_distances() on synthetic open->close trajectories
  2. compute_distribution_report()'s min/max/mean/std/percentile/overlap
     arithmetic on synthetic data
  3. compute_object_relative_comparison()'s ratio arithmetic
  4. summarize_dimension_verdicts()/decide_verdict()'s OOD/overlap logic
  5. coordinate_semantics_summary() returns the expected structure/citations
  6. collect_pybullet_orientation_reachability_samples() genuinely
     produces MORE orientation variance than
     collect_pybullet_workspace_samples() (empirically confirms this
     module's own docstring claim about DummyOpenVLAPolicy never rotating)
  7. regression: benchmark.test_libero_real_observation (production
     get_libero_observation_state() itself, unaffected by this diagnostic)

Run: python -m benchmark.test_environment_state_alignment_diagnostic
"""

import numpy as np

from benchmark.run_environment_state_alignment_diagnostic import (
    STATE_DIM_NAMES,
    coordinate_semantics_summary,
    collect_pybullet_orientation_reachability_samples,
    collect_pybullet_workspace_samples,
    compute_distribution_report,
    compute_object_relative_comparison,
    decide_verdict,
    estimate_reach_distances,
    summarize_dimension_verdicts,
)

_FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        _FAILURES.append(name)


def make_synthetic_episode(ee_start, ee_grasp, gripper_open=0.04, gripper_closed=0.0, length=20, close_at=10):
    """8D rows: ee_position ramps from ee_start to ee_grasp linearly,
    orientation/gripper fixed except gripper.0 which drops to
    gripper_closed at close_at."""
    rows = []
    for step in range(length):
        t = step / (length - 1)
        ee = [ee_start[i] + t * (ee_grasp[i] - ee_start[i]) for i in range(3)]
        gripper0 = gripper_open if step < close_at else gripper_closed
        rows.append(ee + [0.0, 0.0, 0.0, gripper0, -gripper0])
    return np.array(rows, dtype=np.float64)


def main() -> None:
    print("=== 1. estimate_reach_distances() ===")
    ep1 = make_synthetic_episode([0.0, 0.0, 0.5], [0.3, 0.0, 0.4], close_at=10)
    episodes = {"ep1": ep1}
    distances = estimate_reach_distances(episodes)
    check("1 reach distance computed", len(distances) == 1, f"got {len(distances)}")
    expected = float(np.linalg.norm(np.array(ep1[10, 0:3]) - np.array(ep1[0, 0:3])))
    check("reach distance matches manual EE(start)->EE(first-closed) computation", abs(distances[0] - expected) < 1e-9, f"got {distances[0]} expected {expected}")

    never_closes = make_synthetic_episode([0, 0, 0.5], [0.3, 0, 0.4], close_at=999)
    check("an episode that never closes the gripper contributes no reach distance", estimate_reach_distances({"e": never_closes}) == [])

    already_closed = make_synthetic_episode([0, 0, 0.5], [0.3, 0, 0.4], close_at=0)
    check("an episode already closed at frame 0 contributes no reach distance (nothing to measure)", estimate_reach_distances({"e": already_closed}) == [])
    print()

    print("=== 2. compute_distribution_report() arithmetic ===")
    real = np.array([[float(i) + noise for i in range(8)] for noise in np.linspace(-1, 1, 100)])
    ours = np.array([[float(i) + noise for i in range(8)] for noise in np.linspace(-1, 1, 100)])  # identical distribution
    report = compute_distribution_report(real, ours)
    check("8 dimensions reported", len(report) == 8, f"got {len(report)}")
    first_dim = STATE_DIM_NAMES[0]
    check("mean matches for identical distributions", abs(report[first_dim]["real"]["mean"] - report[first_dim]["ours"]["mean"]) < 1e-9)
    check("range_overlap_fraction is 1.0 for identical distributions", abs(report[first_dim]["range_overlap_fraction"] - 1.0) < 1e-9, f"got {report[first_dim]['range_overlap_fraction']}")
    check("percentiles (p5/p50/p95) are present", all(key in report[first_dim]["real"] for key in ("p5", "p50", "p95")))
    check("histogram is present with 10 bins", len(report[first_dim]["real"]["histogram"]["counts"]) == 10)
    check("sample_values are present (first 5 raw values)", len(report[first_dim]["real"]["sample_values"]) == 5)

    disjoint_ours = ours + 1000.0
    disjoint_report = compute_distribution_report(real, disjoint_ours)
    check("range_overlap_fraction is 0.0 for completely disjoint ranges", disjoint_report[first_dim]["range_overlap_fraction"] == 0.0)
    print()

    print("=== 3. compute_object_relative_comparison() ===")
    result = compute_object_relative_comparison([0.2, 0.2, 0.2], np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]))
    check("mean_ratio is ~1.0 when both distances are equal on average", abs(result["mean_ratio_ours_over_real"] - 1.0) < 1e-6, f"got {result['mean_ratio_ours_over_real']}")
    result_half = compute_object_relative_comparison([0.4, 0.4], np.array([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]))
    check("mean_ratio is ~0.5 when ours is half of real", abs(result_half["mean_ratio_ours_over_real"] - 0.5) < 1e-6, f"got {result_half['mean_ratio_ours_over_real']}")
    print()

    print("=== 4. summarize_dimension_verdicts()/decide_verdict() ===")
    ok_report = {name: {"real": {"mean": 0.0, "std": 1.0}, "ours": {"mean": 0.1}, "range_overlap_fraction": 0.8} for name in STATE_DIM_NAMES}
    ok_verdicts = summarize_dimension_verdicts(ok_report)
    check("all dims flagged OK when means/overlap are close", all(v["flag"] == "OK" for v in ok_verdicts.values()))
    ok_final = decide_verdict(ok_verdicts, {"mean_ratio_ours_over_real": 1.0})
    check("verdict A when nothing is flagged", ok_final["verdict"] == "A", str(ok_final))

    ood_report = dict(ok_report)
    ood_report["ee_position.x"] = {"real": {"mean": 0.0, "std": 0.1}, "ours": {"mean": 5.0}, "range_overlap_fraction": 0.0}
    ood_verdicts = summarize_dimension_verdicts(ood_report)
    check("ee_position.x flagged OOD with a large mean shift + zero overlap", ood_verdicts["ee_position.x"]["flag"] == "OOD")
    ood_final = decide_verdict(ood_verdicts, {"mean_ratio_ours_over_real": 1.0})
    check("verdict B when a position dimension is OOD", ood_final["verdict"] == "B", str(ood_final))

    orientation_ood_report = dict(ok_report)
    orientation_ood_report["ee_orientation_axis_angle.y"] = {"real": {"mean": 0.0, "std": 0.1}, "ours": {"mean": 5.0}, "range_overlap_fraction": 0.0}
    orientation_verdicts = summarize_dimension_verdicts(orientation_ood_report)
    orientation_final = decide_verdict(orientation_verdicts, {"mean_ratio_ours_over_real": 1.0})
    check("verdict B when an orientation dimension is OOD", orientation_final["verdict"] == "B", str(orientation_final))
    print()

    print("=== 5. coordinate_semantics_summary() structure ===")
    semantics = coordinate_semantics_summary()
    check("has pybullet_ee_position citation", "pybullet_ee_position" in semantics and "source" in semantics["pybullet_ee_position"])
    check("has robosuite_robot0_eef_pos citation", "robosuite_robot0_eef_pos" in semantics and "source" in semantics["robosuite_robot0_eef_pos"])
    check("same_reference_frame_convention verdict is True (both world frame)", semantics["same_reference_frame_convention"]["verdict"] is True)
    check("world_origin_placement_difference cites the robosuite table_offset/base_xpos_offset", "0.8" in semantics["world_origin_placement_difference"]["robosuite_table_offset"])
    print()

    print("=== 6. orientation-reachability sample has more variance than the scripted-policy workspace sample ===")
    workspace_data = collect_pybullet_workspace_samples(
        positions={"center_right": [0.42, 0.0, 0.05]}, max_steps_per_episode=15,
    )
    orientation_data = collect_pybullet_orientation_reachability_samples(num_episodes=1)
    workspace_orientation_std = workspace_data["flat_states"][:, 3:6].std(axis=0)
    reachability_orientation_std = orientation_data["flat_states"][:, 3:6].std(axis=0)
    check(
        "rotation-exercising sample has measurably more orientation variance than the (never-rotating) "
        "scripted-policy workspace sample, on at least 2 of 3 axes",
        sum(1 for i in range(3) if reachability_orientation_std[i] > workspace_orientation_std[i]) >= 2,
        f"workspace_std={workspace_orientation_std} reachability_std={reachability_orientation_std}",
    )
    check(
        "the workspace sample's orientation is nearly frozen (std < 0.01 on y/z, confirming "
        "DummyOpenVLAPolicy never commands rotation)",
        workspace_orientation_std[1] < 0.01 and workspace_orientation_std[2] < 0.01,
        f"got {workspace_orientation_std}",
    )
    print()

    print("=== 7. regression: production get_libero_observation_state() itself ===")
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "benchmark.test_libero_real_observation"], capture_output=True, text=True, timeout=300)
    passed = "ALL CHECKS PASSED" in result.stdout
    check("benchmark.test_libero_real_observation -- ALL CHECKS PASSED", passed, result.stdout[-1500:] if not passed else "")
    print()

    print("=" * 60)
    if _FAILURES:
        print(f"FAIL -- {len(_FAILURES)} check(s) failed: {_FAILURES}")
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
