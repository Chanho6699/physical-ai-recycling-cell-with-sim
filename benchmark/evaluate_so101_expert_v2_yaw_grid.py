"""Expert V2 (orientation-aware) standalone yaw-grid evaluation (see
this task's chat report, "Expert 단독 yaw-grid 평가"). Runs
benchmark.so101_expert_v2_orientation.run_pick_and_place_episode_v2
(NEVER V1's own run_pick_and_place_episode) across the yaw grid and
position groups this task specifies, records EVERY attempt (success or
failure -- nothing silently discarded, per this task's own absolute
principle 10), and prints the aggregate breakdowns this task's report
requires (per-yaw, per-position-group, per-yaw-x-position-group,
failure-reason counts).

Position groups reuse benchmark/collect_so101_stage1b_box_dataset.py's
own REGION_DEFS (already-validated Stage 1B region definitions) --
mapped onto this task's own 5 requested categories:
  center            -> REGION_DEFS["center"]
  interior          -> REGION_DEFS["existing_x_min"]  (inside the ORIGINAL Stage-1A/1B range, off-center)
  edge              -> REGION_DEFS["bridge_plus_x"]   (at the edge of the EXPANDED range)
  corner            -> REGION_DEFS["corner_pn"]       (Stage 1B's own empirically WEAKEST corner, 40% held-out test)
  x_min_corridor    -> REGION_DEFS["x_min_corridor"]  (this task's own explicitly-named "known weak -X corridor")

Yaw grid (this task's own exact list): 0, +-15, +-30, +-45, +-60, +-75, 90 degrees (12 values).
3 deterministic seeds (repeats) per (position_group, yaw) combination
-- NOT identical repeats of the same seed (a fixed seed would only
verify determinism, already covered by test_so101_expert_v1_regression.py) --
3 different seeds spread across each region's own already-used Stage 1B
seed block, so each repeat is a genuinely different sampled XY point
within that region.

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_expert_v2_yaw_grid
"""

import json
import math
from pathlib import Path

from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_MODE_ANCHOR_OFFSET_XY, FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY
from benchmark.collect_so101_stage1b_box_dataset import BOX_FOOTPRINT_XY, REGION_DEFS
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_expert_v2_orientation import ExpertExecutionMonitor, run_pick_and_place_episode_v2
from benchmark.so101_scripted_expert import (
    FAILURE_UNKNOWN_PLACE_FAILURE,
    So101ExpertError,
    compute_bin_success_debug,
    evaluate_bin_place_success,
)
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, So101PyBulletBackend

POSITION_GROUPS = {
    "center": ("center", [15000, 15001, 15002]),
    "interior": ("existing_x_min", [15100, 15101, 15102]),
    "edge": ("bridge_plus_x", [16000, 16001, 16002]),
    "corner": ("corner_pn", [15700, 15701, 15702]),
    "x_min_corridor": ("x_min_corridor", [15500, 15501, 15502]),
}

YAW_GRID_DEG = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90]

OUTPUT_DIR = Path("results/so101_expert_v2_yaw_grid")


def build_backend(region_name: str, seed: int, yaw_rad: float):
    region = REGION_DEFS[region_name]
    sampled_object_position = sample_object_position(seed, region["x_range"], region["y_range"])
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY, "object_footprint_xy": BOX_FOOTPRINT_XY}
    backend = So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config=scene_config, object_yaw_rad=yaw_rad,
    )
    return backend, sampled_object_position


def run_one_episode(position_group: str, region_name: str, seed: int, yaw_deg: float) -> dict:
    yaw_rad = math.radians(yaw_deg)
    backend, sampled_object_position = build_backend(region_name, seed, yaw_rad)
    record = {
        "expert_version": "v2", "strategy": "orientation_aware", "position_group": position_group, "region_name": region_name,
        "seed": seed, "object_position": list(sampled_object_position), "object_yaw_rad": yaw_rad, "object_yaw_deg": yaw_deg,
        "object_dimensions": {"half_extent_x_m": BOX_FOOTPRINT_XY[0], "half_extent_y_m": BOX_FOOTPRINT_XY[1]},
    }
    try:
        backend.reset()
        scene = backend.get_scene_state()
        record["scene_valid"] = bool(scene["layout_validation_passed"])
        if not record["scene_valid"]:
            record.update({
                "success": False, "failure_phase": None, "failure_reason": "scene_invalid", "ik_failed": False,
                "joint_limit_violation": False, "max_joint_jump": None, "orientation_error_max_rad": None,
                "object_lift_height": None, "discard": True, "target_gripper_yaw": None, "selected_orientation_candidate": None,
            })
            return record

        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        monitor = ExpertExecutionMonitor()
        try:
            result = run_pick_and_place_episode_v2(
                backend, yaw_rad, BOX_FOOTPRINT_XY[0], BOX_FOOTPRINT_XY[1], transport_delta_xy, monitor=monitor,
            )
            bin_debug = result["bin_place_result"]["debug"]
            final_object_position = backend.get_object_position()
            bin_place_debug_for_success = {
                "rise_reached": bin_debug["rise_reached"], "pre_place_reached": bin_debug["pre_place_reached"],
                "descend_reached": bin_debug["descend_reached"], "retreat_reached": bin_debug["retreat_reached"],
                "object_separated_during_wait": result["bin_place_result"]["object_separated_during_wait"],
            }
            bin_success_debug = compute_bin_success_debug(
                backend, bin_place_debug_for_success, result["bin_place_result"]["release_constraint_removed"],
                final_object_position, True, scene["layout_validation_passed"],
            )
            place_success, failure_reason, failure_phase = evaluate_bin_place_success(bin_success_debug)
            record.update({
                "success": bool(place_success), "failure_phase": failure_phase, "failure_reason": failure_reason,
                "ik_failed": False, "joint_limit_violation": result["monitor"]["joint_limit_violation"],
                "max_joint_jump": result["monitor"]["max_joint_jump"],
                "orientation_error_max_rad": result["monitor"]["orientation_error_max_rad"],
                "object_lift_height": result["lift"]["final_ee_position"][2] - sampled_object_position[2],
                "discard": False, "target_gripper_yaw": result["grasp_plan"]["target_gripper_yaw"],
                "selected_orientation_candidate": result["grasp_plan"]["selected_orientation_candidate"],
                "effective_grasp_width_m": result["grasp_plan"]["effective_grasp_width_m"],
                "collision_detected": result["monitor"]["collision_detected"],
            })
        except So101ExpertError as exc:
            record.update({
                "success": False, "failure_phase": exc.phase, "failure_reason": exc.failure_reason,
                "ik_failed": exc.failure_reason in ("ik_failed", "lift_failed", "orientation_unreachable"),
                "joint_limit_violation": monitor.joint_limit_violation, "max_joint_jump": monitor.max_joint_jump,
                "orientation_error_max_rad": monitor.orientation_error_max, "object_lift_height": None,
                "discard": False, "target_gripper_yaw": None, "selected_orientation_candidate": None,
                "collision_detected": monitor.collision_detected,
            })
    finally:
        backend.close()
    return record


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    total = len(POSITION_GROUPS) * len(YAW_GRID_DEG) * 3
    done = 0
    for position_group, (region_name, seeds) in POSITION_GROUPS.items():
        for yaw_deg in YAW_GRID_DEG:
            for seed in seeds:
                record = run_one_episode(position_group, region_name, seed, yaw_deg)
                all_records.append(record)
                done += 1
                status = "OK" if record["success"] else f"FAIL({record['failure_reason']})"
                print(f"[{done}/{total}] group={position_group:15s} yaw={yaw_deg:+4d}deg seed={seed} -> {status}")

    with open(OUTPUT_DIR / "yaw_grid_records.jsonl", "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    attempted = len(all_records)
    scene_valid = [r for r in all_records if r["scene_valid"]]
    discarded = [r for r in all_records if r["discard"]]
    valid_attempted = [r for r in scene_valid if not r["discard"]]
    successes = [r for r in valid_attempted if r["success"]]
    failures = [r for r in valid_attempted if not r["success"]]

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"attempted={attempted}  scene_valid={len(scene_valid)}  discarded={len(discarded)}")
    print(f"valid_attempted={len(valid_attempted)}  success={len(successes)}  failed={len(failures)}")
    if valid_attempted:
        print(f"overall success rate = {len(successes)}/{len(valid_attempted)} = {100.0*len(successes)/len(valid_attempted):.1f}%")

    print("\n--- failure_reason counts ---")
    reason_counts = {}
    for r in failures:
        reason_counts[r["failure_reason"]] = reason_counts.get(r["failure_reason"], 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:30s} {count}")

    print("\n--- success rate by yaw (deg) ---")
    for yaw_deg in YAW_GRID_DEG:
        subset = [r for r in valid_attempted if r["object_yaw_deg"] == yaw_deg]
        succ = [r for r in subset if r["success"]]
        rate = 100.0 * len(succ) / len(subset) if subset else float("nan")
        print(f"  yaw={yaw_deg:+4d}deg  {len(succ)}/{len(subset)} = {rate:.1f}%")

    print("\n--- success rate by position group ---")
    for position_group in POSITION_GROUPS:
        subset = [r for r in valid_attempted if r["position_group"] == position_group]
        succ = [r for r in subset if r["success"]]
        rate = 100.0 * len(succ) / len(subset) if subset else float("nan")
        print(f"  {position_group:15s}  {len(succ)}/{len(subset)} = {rate:.1f}%")

    print("\n--- success rate by yaw x position group ---")
    header = "yaw\\group".ljust(10) + "".join(g.ljust(16) for g in POSITION_GROUPS)
    print(header)
    for yaw_deg in YAW_GRID_DEG:
        row = f"{yaw_deg:+4d}deg".ljust(10)
        for position_group in POSITION_GROUPS:
            subset = [r for r in valid_attempted if r["object_yaw_deg"] == yaw_deg and r["position_group"] == position_group]
            succ = [r for r in subset if r["success"]]
            cell = f"{len(succ)}/{len(subset)}" if subset else "-"
            row += cell.ljust(16)
        print(row)

    joint_limit_violations = sum(1 for r in valid_attempted if r.get("joint_limit_violation"))
    max_joint_jump_overall = max((r["max_joint_jump"] for r in valid_attempted if r.get("max_joint_jump") is not None), default=None)
    print(f"\njoint_limit_violation count = {joint_limit_violations}")
    print(f"max_joint_jump (overall max) = {max_joint_jump_overall}")

    with open(OUTPUT_DIR / "yaw_grid_summary.json", "w") as f:
        json.dump({
            "attempted": attempted, "scene_valid": len(scene_valid), "discarded": len(discarded),
            "valid_attempted": len(valid_attempted), "success": len(successes), "failed": len(failures),
            "overall_success_rate": (len(successes) / len(valid_attempted)) if valid_attempted else None,
            "failure_reason_counts": reason_counts, "joint_limit_violation_count": joint_limit_violations,
            "max_joint_jump_overall": max_joint_jump_overall,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_DIR / 'yaw_grid_records.jsonl'} and {OUTPUT_DIR / 'yaw_grid_summary.json'}")


if __name__ == "__main__":
    main()
