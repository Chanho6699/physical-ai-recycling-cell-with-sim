"""Expert V2.1 (size-aware) standalone cylinder evaluation (see this
task's chat report, "Cylinder Expert 단독 평가"). Runs
benchmark.so101_expert_v2_size_aware.run_pick_and_place_episode_v2_1
(NEVER V1's own run_pick_and_place_episode, NEVER the orientation-aware
V2 module) across position groups x scene object_yaw values x seeds,
recording every attempt -- nothing silently discarded (this task's own
absolute principle 9).

The yaw values set here are a ROTATIONAL-SYMMETRY sanity check ONLY
(this task's own section 6: "orientation-aware grasp를 시험하는 것이
아니라 회전 대칭성 확인용") -- V2.1's grasp planner never reads object
yaw at all (see so101_expert_v2_size_aware.py's own ObjectMetadata /
SizeAwareGraspPlanner, neither has a yaw field), so this evaluation is
checking that varying the SCENE's object_yaw_rad (a purely visual/
collision-shape rotation for a rotationally-symmetric upright cylinder)
produces no spurious result differences -- NOT that V2.1 does
orientation-aware grasping (this task's own absolute principle 10:
cylinder results must not be read as solving box arbitrary-yaw).

Two success measures are reported per this task's own "성공 판정 보강":
  legacy_success  -- V1's own evaluate_bin_place_success() criterion, unchanged.
  physical_success -- legacy_success AND object actually rose by at least
                       MINIMUM_PHYSICAL_LIFT_HEIGHT_M AND grasp was
                       maintained (no is_grasped()==False sample) through
                       BOTH lift and transport.
contact_count is recorded as a diagnostic field but (see this task's
chat report on candidate screening) is expected to be 0 even for the
already-validated cube/box baseline in this simulation's rigid fixed-
constraint grasp model -- not a discriminating signal here, reported
honestly rather than treated as a hidden pass/fail gate.

Run:
  .venv-vla/bin/python -m benchmark.evaluate_so101_expert_v2_cylinder
"""

import json
import math
from pathlib import Path

import pybullet as p

from benchmark.benchmark_so101_bin_diagnostic import FIXED_BIN_MODE_ANCHOR_OFFSET_XY, FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY
from benchmark.collect_so101_stage1b_box_dataset import REGION_DEFS
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_expert_v2_size_aware import MINIMUM_PHYSICAL_LIFT_HEIGHT_M, ObjectMetadata, run_pick_and_place_episode_v2_1
from benchmark.so101_scripted_expert import (
    So101ExpertError,
    compute_bin_success_debug,
    evaluate_bin_place_success,
)
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, So101PyBulletBackend

# Final Stage 1C candidate (see this task's chat report, "최종 선택한
# cylinder 크기와 근거"): radius=0.02m (diameter 4cm), height=0.04m --
# identical footprint scale to the already-validated cube/box, isolating
# shape as the only new variable.
CYLINDER_RADIUS_M = 0.02
CYLINDER_HEIGHT_M = 0.04
CYLINDER_MASS_KG = 0.05

POSITION_GROUPS = {
    "center": ("center", [15000, 15001, 15002, 15003, 15004]),
    "interior": ("existing_x_min", [15100, 15101, 15102, 15103, 15104]),
    "edge": ("bridge_plus_x", [16000, 16001, 16002, 16003, 16004]),
    "corner": ("corner_pn", [15700, 15701, 15702, 15703, 15704]),
    "x_min_corridor": ("x_min_corridor", [15500, 15501, 15502, 15503, 15504]),
}

YAW_GRID_DEG = [0, 45, 90, 135, 180]  # rotational-symmetry check only -- see this module's own docstring

OUTPUT_DIR = Path("results/so101_expert_v2_cylinder")


def build_backend(region_name: str, seed: int, yaw_rad: float):
    region = REGION_DEFS[region_name]
    sampled_object_position = sample_object_position(seed, region["x_range"], region["y_range"])
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config = {
        "surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
        "object_shape": "cylinder", "object_radius": CYLINDER_RADIUS_M, "object_height": CYLINDER_HEIGHT_M,
    }
    backend = So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config=scene_config, object_yaw_rad=yaw_rad,
    )
    return backend, sampled_object_position


def run_one_episode(position_group: str, region_name: str, seed: int, yaw_deg: float) -> dict:
    yaw_rad = math.radians(yaw_deg)
    backend, sampled_object_position = build_backend(region_name, seed, yaw_rad)
    record = {
        "expert_version": "v2.1", "strategy": "size_aware", "object_shape": "cylinder",
        "object_dimensions": {"radius_m": CYLINDER_RADIUS_M, "height_m": CYLINDER_HEIGHT_M},
        "radius": CYLINDER_RADIUS_M, "height": CYLINDER_HEIGHT_M, "mass": CYLINDER_MASS_KG, "friction": None,
        "position_group": position_group, "region_name": region_name, "seed": seed,
        "object_position": list(sampled_object_position), "object_yaw": yaw_rad, "object_yaw_deg": yaw_deg,
    }
    try:
        backend.reset()
        scene = backend.get_scene_state()
        record["scene_valid"] = bool(scene["layout_validation_passed"])
        if not record["scene_valid"]:
            record.update({
                "legacy_success": False, "physical_success": False, "failure_phase": None, "failure_reason": "scene_invalid",
                "discard": True, "contact_count": None, "object_lift_height": None, "retention_steps": None,
                "inside_bin": None, "settled_in_bin": None, "joint_limit_violation": False, "collision": False,
            })
            return record

        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        metadata = ObjectMetadata(shape="cylinder", position=list(sampled_object_position), height_m=CYLINDER_HEIGHT_M, radius_m=CYLINDER_RADIUS_M, mass_kg=CYLINDER_MASS_KG)
        try:
            result = run_pick_and_place_episode_v2_1(backend, metadata, transport_delta_xy)
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
            legacy_success, failure_reason, failure_phase = evaluate_bin_place_success(bin_success_debug)

            grasp_maintained = result["lift"]["grasp_maintained_all_steps"] and result["transport"]["grasp_maintained_all_steps"]
            object_lift_height = result["object_lift_height"]
            physical_success = bool(legacy_success and grasp_maintained and object_lift_height >= MINIMUM_PHYSICAL_LIFT_HEIGHT_M)
            retention_steps = (result["lift"]["num_steps"] + result["transport"]["num_steps"]) if grasp_maintained else None

            contacts = p.getContactPoints(bodyA=backend.robot_id, bodyB=backend.object_id, physicsClientId=backend.client_id)
            joint_positions = backend.get_joint_positions()
            from benchmark.so101_scripted_expert import check_joint_limits
            joint_limit_violation = len(check_joint_limits(backend, joint_positions)) > 0

            record.update({
                "legacy_success": bool(legacy_success), "physical_success": physical_success,
                "failure_phase": failure_phase, "failure_reason": failure_reason, "discard": False,
                "contact_count": len(contacts), "object_lift_height": object_lift_height, "retention_steps": retention_steps,
                "inside_bin": bin_success_debug["inside_inner_xy"], "settled_in_bin": bin_success_debug["settle_success"],
                "joint_limit_violation": joint_limit_violation, "collision": False,
                "object_slipped": bool(legacy_success is False and result["lift"]["grasp_maintained_all_steps"] is False),
            })
        except So101ExpertError as exc:
            record.update({
                "legacy_success": False, "physical_success": False, "failure_phase": exc.phase, "failure_reason": exc.failure_reason,
                "discard": False, "contact_count": None, "object_lift_height": None, "retention_steps": None,
                "inside_bin": None, "settled_in_bin": None, "joint_limit_violation": False, "collision": False, "object_slipped": False,
            })
    finally:
        backend.close()
    return record


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    total = len(POSITION_GROUPS) * len(YAW_GRID_DEG) * 5
    done = 0
    for position_group, (region_name, seeds) in POSITION_GROUPS.items():
        for yaw_deg in YAW_GRID_DEG:
            for seed in seeds:
                record = run_one_episode(position_group, region_name, seed, yaw_deg)
                all_records.append(record)
                done += 1
                status = f"legacy={record['legacy_success']} physical={record['physical_success']}" if not record["discard"] else "DISCARDED(scene_invalid)"
                if not record["legacy_success"] and not record["discard"]:
                    status += f" reason={record['failure_reason']}"
                print(f"[{done}/{total}] group={position_group:15s} yaw={yaw_deg:4d}deg seed={seed} -> {status}")

    with open(OUTPUT_DIR / "cylinder_expert_records.jsonl", "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    attempted = len(all_records)
    scene_valid = [r for r in all_records if r["scene_valid"]]
    discarded = [r for r in all_records if r["discard"]]
    valid_attempted = [r for r in scene_valid if not r["discard"]]
    legacy_successes = [r for r in valid_attempted if r["legacy_success"]]
    physical_successes = [r for r in valid_attempted if r["physical_success"]]
    mismatches = [r for r in valid_attempted if r["legacy_success"] != r["physical_success"]]

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"attempted={attempted}  scene_valid={len(scene_valid)}  discarded={len(discarded)}  valid_attempted={len(valid_attempted)}")
    if valid_attempted:
        print(f"legacy_success   = {len(legacy_successes)}/{len(valid_attempted)} = {100.0*len(legacy_successes)/len(valid_attempted):.1f}%")
        print(f"physical_success = {len(physical_successes)}/{len(valid_attempted)} = {100.0*len(physical_successes)/len(valid_attempted):.1f}%")
    print(f"legacy != physical mismatches = {len(mismatches)}")
    for r in mismatches:
        print(f"  MISMATCH group={r['position_group']} yaw={r['object_yaw_deg']} seed={r['seed']} legacy={r['legacy_success']} physical={r['physical_success']} lift_height={r['object_lift_height']}")

    print("\n--- failure_reason counts ---")
    reason_counts = {}
    for r in valid_attempted:
        if not r["legacy_success"]:
            reason_counts[r["failure_reason"]] = reason_counts.get(r["failure_reason"], 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:30s} {count}")

    print("\n--- success rate by position group (legacy / physical) ---")
    for position_group in POSITION_GROUPS:
        subset = [r for r in valid_attempted if r["position_group"] == position_group]
        legacy = [r for r in subset if r["legacy_success"]]
        physical = [r for r in subset if r["physical_success"]]
        print(f"  {position_group:15s} legacy={len(legacy)}/{len(subset)} physical={len(physical)}/{len(subset)}")

    print("\n--- success rate by scene yaw (rotational-symmetry check) ---")
    for yaw_deg in YAW_GRID_DEG:
        subset = [r for r in valid_attempted if r["object_yaw_deg"] == yaw_deg]
        legacy = [r for r in subset if r["legacy_success"]]
        print(f"  yaw={yaw_deg:4d}deg  legacy={len(legacy)}/{len(subset)}")

    contact_counts = [r["contact_count"] for r in valid_attempted if r["contact_count"] is not None]
    joint_limit_violations = sum(1 for r in valid_attempted if r.get("joint_limit_violation"))
    print(f"\ncontact_count: min={min(contact_counts) if contact_counts else None} max={max(contact_counts) if contact_counts else None} (diagnostic only, see module docstring)")
    print(f"joint_limit_violation count = {joint_limit_violations}")

    with open(OUTPUT_DIR / "cylinder_expert_summary.json", "w") as f:
        json.dump({
            "attempted": attempted, "scene_valid": len(scene_valid), "discarded": len(discarded), "valid_attempted": len(valid_attempted),
            "legacy_success": len(legacy_successes), "physical_success": len(physical_successes),
            "legacy_success_rate": (len(legacy_successes) / len(valid_attempted)) if valid_attempted else None,
            "physical_success_rate": (len(physical_successes) / len(valid_attempted)) if valid_attempted else None,
            "mismatch_count": len(mismatches), "failure_reason_counts": reason_counts,
            "joint_limit_violation_count": joint_limit_violations,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_DIR / 'cylinder_expert_records.jsonl'} and {OUTPUT_DIR / 'cylinder_expert_summary.json'}")


if __name__ == "__main__":
    main()
