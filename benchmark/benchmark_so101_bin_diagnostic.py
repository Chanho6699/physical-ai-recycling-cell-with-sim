"""SO-101 open-top bin physical-stability diagnostic benchmark (see
this task's chat report, "production 성공 판정을 변경하기 전에 bin
expert의 물리적 안정성을 진단"). Purely a DIAGNOSTIC measurement tool --
does NOT change place_success/target_xy_error/failure_reason/settle
judgment, does NOT collect a dataset, does NOT run SmolVLA training,
does NOT touch expert waypoints or bin-place clearances, does NOT
touch robot_sim/pybullet_panda_backend.py, does NOT touch the existing
flat 20-seed benchmark.

Reuses (does NOT reimplement):
  - benchmark.evaluate_so101_expert_small_randomization's own
    sample_object_position()/DEFAULT_X_RANGE/DEFAULT_Y_RANGE for
    deterministic object-position randomization (same seed -> same
    x/y offset around the backend's own default object position).
  - benchmark.smoke_so101_bin_place's own classify_object_bin_contacts()
    for final object-wall/object-bottom contact classification.
  - benchmark.so101_scripted_expert's own run_pick_and_place_episode()
    (unmodified control flow/thresholds) and its own
    "bin_place_debug" (waypoint targets/errors, phase-tagged contact
    logs, release-moment AABBs) -- all diagnostic fields already
    exposed by that module, not recomputed here.
  - robot_sim.so101_pybullet_backend's own InvalidSceneLayoutError /
    validate_initial_scene_layout() (auto-run by reset() whenever
    use_bin=True) to catch a bad scene BEFORE any expert phase runs.

This file computes an internal-only `diagnostic_outcome` classification
(see CLASSIFY_* constants below) -- this is NEVER written back into
production place_success/failure_reason.

Run:
  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic
  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic --seeds 0,1,2,3,4
  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic --num-seeds 10
  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic --gui --seed 3
  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic --num-seeds 20 \\
    --output-json results/so101_bin_diagnostic_20seeds.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import pybullet as p

from benchmark.evaluate_so101_expert_small_randomization import (
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    sample_object_position,
)
from benchmark.smoke_so101_bin_place import classify_object_bin_contacts
from benchmark.so101_scripted_expert import (
    ANGULAR_SPEED_PASS_RADPS,
    LINEAR_SPEED_PASS_MPS,
    PHASE_APPROACH,
    PHASE_GRASP,
    PHASE_LIFT,
    PHASE_PLACE_DESCEND,
    PHASE_PRE_GRASP,
    PHASE_TRANSPORT,
    So101ExpertError,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, InvalidSceneLayoutError, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = "results/so101_bin_diagnostic_10seeds.json"
DEFAULT_NUM_SEEDS = 10
DEFAULT_SEED_START = 0

# Candidate-diagnostic-only tolerances (see this task's chat report,
# "stuck_on_rim 진단") -- NOT production thresholds, not used to gate
# place_success anywhere.
NEAR_BOUNDARY_TOLERANCE_M = 0.01
RIM_STRADDLE_TOLERANCE_M = 0.01

# Contact-force classification (see this task's chat report, "contact
# 분류 보강") -- diagnostic-benchmark-only bucketing of robot-bin
# contact severity; NEVER used to gate production place_success (that
# stays exactly as evaluate_bin_place_success() already defines it,
# which never even looks at robot-bin contact). 1.0N is small relative
# to this scene's own object weight (~0.05kg * 9.8 ~= 0.5N) -- a
# contact at or below it reads as a light touch, not a real impact.
LOW_FORCE_CONTACT_MAX_N = 1.0

DIAGNOSTIC_OUTCOMES = [
    "success_candidate", "scene_invalid", "grasp_failed", "lift_failed", "transport_failed",
    "waypoint_failed", "robot_bin_contact", "object_missed_bin", "object_stuck_on_rim",
    "object_not_below_rim", "object_not_settled", "unknown",
]

# --- randomization_mode "fixed_bin_object_xy" (see this task's chat
# report, "randomization 설계 변경") -- bin stays at a FIXED world
# position every episode; only the object's own XY is randomized
# (existing "coupled_small" mode -- bin_center = object_position +
# offset, unchanged, still the CLI default -- is untouched). A
# DEDICATED anchor offset/table footprint, separate from
# DEFAULT_BIN_TARGET_ZONE_OFFSET_XY / DEFAULT_BIN_SURFACE_FOOTPRINT_XY
# (robot_sim/so101_pybullet_backend.py), so this new mode cannot alter
# the coupled mode's own bin position or table size.
#
# Derivation (see this task's chat report, "안전한 범위를 확정한다"):
# object half-extent=0.02m, bin outer half-extent=0.07+0.004=0.074m ->
# two axis-aligned boxes need >0.094m clearance on AT LEAST ONE axis to
# never overlap. With the bin's Y anchored well away from the object's
# nominal Y (0.0) and the object independently randomized within
# +/-FIXED_BIN_OBJECT_Y_RANGE, the worst case is the object's Y at its
# own extreme: anchor_y - Y_RANGE must stay > 0.094 with real margin,
# not just barely over. anchor_y=0.13, Y_RANGE=0.015 ->
# 0.13-0.015-0.094=+0.021m margin. The default table's Y half-extent
# (DEFAULT_BIN_SURFACE_FOOTPRINT_XY=0.19) is also too tight for this
# anchor (0.13+0.074=0.204 > 0.19), so this mode ALSO widens
# surface_footprint_xy's Y half via the EXISTING scene_config override
# mechanism (no backend change) -- 0.22 leaves 0.22-0.204=+0.016m
# margin there too. X keeps the small offset the ORIGINAL coupled
# design already validated for good IK convergence (see
# DEFAULT_BIN_TARGET_ZONE_OFFSET_XY's own docstring: a y-dominant
# split reaches with far better precision than an even x/y split).
FIXED_BIN_MODE_ANCHOR_OFFSET_XY = [0.03, 0.13]
FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY = [0.19, 0.22]
FIXED_BIN_OBJECT_X_RANGE = (-0.015, 0.015)
FIXED_BIN_OBJECT_Y_RANGE = (-0.015, 0.015)

# Object footprint is a SQUARE cross-section cube (object_footprint_xy
# half-extents [0.02, 0.02], object_height=0.04 -- see
# robot_sim/so101_pybullet_backend.py's own DEFAULT_SCENE_CONFIG), not
# a cylinder -- yaw is NOT fully symmetric (only period-90-degree
# symmetric), so a small yaw DOES change the rendered silhouette (see
# this task's chat report, section 5). +/-12 degrees chosen as a small
# perturbation well inside that 90-degree period.
FIXED_BIN_OBJECT_YAW_RANGE_RAD = (-0.2094395102393195, 0.2094395102393195)  # +/-12 degrees

RANDOMIZATION_MODE_COUPLED_SMALL = "coupled_small"
RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY = "fixed_bin_object_xy"


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sample_object_yaw_rad(seed: int, yaw_range_rad: tuple) -> float:
    """Same deterministic pattern as
    benchmark.evaluate_so101_expert_small_randomization.sample_object_position()
    -- a fresh random.Random(seed) per call, so the same seed always
    yields the same yaw. A separate Random instance from whatever
    sample_object_position() itself used for this seed's x/y offset --
    no shared state, so this can't disturb that draw."""
    return random.Random(seed).uniform(yaw_range_rad[0], yaw_range_rad[1])


def parse_seed_list(seeds_str: str) -> list:
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def classify_exception_phase(phase: str) -> str:
    if phase in (PHASE_PRE_GRASP, PHASE_APPROACH, PHASE_GRASP):
        return "grasp_failed"
    if phase == PHASE_LIFT:
        return "lift_failed"
    if phase == PHASE_TRANSPORT:
        return "transport_failed"
    if phase == PHASE_PLACE_DESCEND:
        return "waypoint_failed"
    return "unknown"


def run_single_seed_diagnostic(
    seed: int, x_range: tuple, y_range: tuple, gui: bool = False,
    randomization_mode: str = RANDOMIZATION_MODE_COUPLED_SMALL,
    bin_center_override_xy: list = None, object_yaw_rad: float = None,
    scene_config: dict = None,
) -> dict:
    sampled_object_position = sample_object_position(seed, x_range, y_range)
    record = {"seed": seed, "sampled_object_position": sampled_object_position, "randomization_mode": randomization_mode}

    backend_kwargs = {"gui": gui, "use_bin": True, "object_position": sampled_object_position}
    if bin_center_override_xy is not None:
        backend_kwargs["bin_center_override_xy"] = bin_center_override_xy
    if object_yaw_rad is not None:
        backend_kwargs["object_yaw_rad"] = object_yaw_rad
        record["sampled_object_yaw_rad"] = object_yaw_rad
    if scene_config is not None:
        backend_kwargs["scene_config"] = scene_config
    backend = So101PyBulletBackend(**backend_kwargs)
    try:
        # --- reset() + scene validation (see this task's chat report,
        # "randomization으로 invalid layout이 발생한다면... scene_generation_invalid
        # 또는 diagnostic invalid seed로 별도 기록") --- No resampling,
        # no infinite retry -- a bad scene for this seed is simply
        # recorded as such, once.
        try:
            backend.reset()
        except InvalidSceneLayoutError as exc:
            record.update({
                "diagnostic_outcome": "scene_invalid",
                "A_initial_scene": {"scene_invalid_failure_type": exc.failure_type, "scene_invalid_details": exc.details},
            })
            return record

        scene = backend.get_scene_state()
        bin_info = backend.get_bin_debug_info()
        rim_z = bin_info["rim_z"]
        inner_bounds = {
            "x_min": bin_info["inner_x_min"], "x_max": bin_info["inner_x_max"],
            "y_min": bin_info["inner_y_min"], "y_max": bin_info["inner_y_max"],
        }

        # A. initial scene
        record["A_initial_scene"] = {
            "object_initial_xyz": scene["object_position"],
            "bin_center_xyz": scene["bin_center"],
            "target_zone_offset_xy": scene["target_zone_offset_xy"],
            "surface_footprint_xy": backend.scene_config["surface_footprint_xy"],
            "object_initial_aabb": scene["object_aabb_initial"],
            "bin_outer_bounds": scene["bin_outer_bounds"],
            "layout_validation_passed": scene["layout_validation_passed"],
        }

        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])

        try:
            result = run_pick_and_place_episode(backend, transport_delta_xy)
        except So101ExpertError as exc:
            record["exception"] = {"failure_reason": exc.failure_reason, "phase": exc.phase, "message": str(exc)}
            record["diagnostic_outcome"] = classify_exception_phase(exc.phase)
            final_object_position = backend.get_object_position()
            record["F_final_state"] = {"final_object_xyz": final_object_position}
            return record

        bin_debug = result["bin_place_debug"]

        # B. grasp/transport
        record["B_grasp_transport"] = {
            "grasp_success": True, "lift_success": True, "transport_completed": True,
            "transport_final_ee_xyz": result["transport"]["final_ee_position"],
            "transport_final_object_xyz": result["transport"]["object_final_position"],
            "transport_grasp_maintained_all_steps": result["transport"]["grasp_maintained_all_steps"],
            "transport_constraint_valid_all_steps": result["transport"]["constraint_valid_all_steps"],
        }

        # C. bin waypoints
        waypoint_errors = {
            "rise": bin_debug["rise_error_m"], "pre_place": bin_debug["pre_place_error_m"],
            "descend": bin_debug["descend_error_m"], "retreat": bin_debug["retreat_error_m"],
        }
        waypoint_reached = {
            "rise": bin_debug["rise_reached"], "pre_place": bin_debug["pre_place_reached"],
            "descend": bin_debug["descend_reached"], "retreat": bin_debug["retreat_reached"],
        }
        record["C_bin_waypoints"] = {
            "rise_target": bin_debug["rise_target"], "rise_final_ee": bin_debug["rise_final_ee"], "rise_error_m": bin_debug["rise_error_m"],
            "pre_place_target": bin_debug["pre_place_target"], "pre_place_final_ee": bin_debug["pre_place_final_ee"], "pre_place_error_m": bin_debug["pre_place_error_m"],
            "release_target": bin_debug["release_target"], "descend_final_ee": bin_debug["descend_final_ee"], "descend_error_m": bin_debug["descend_error_m"],
            "retreat_target": bin_debug["retreat_target"], "retreat_final_ee": bin_debug["retreat_final_ee"], "retreat_error_m": bin_debug["retreat_error_m"],
            "waypoint_reached": waypoint_reached, "all_waypoints_reached": all(waypoint_reached.values()),
            "max_waypoint_error_m": max(waypoint_errors.values()),
        }

        # D. release moment
        object_bottom_before = bin_debug["object_aabb_before_release"]["min"][2]
        object_top_before = bin_debug["object_aabb_before_release"]["max"][2]
        object_bottom_after = bin_debug["object_aabb_after_release"]["min"][2]
        object_top_after = bin_debug["object_aabb_after_release"]["max"][2]
        release_xy_error = math.sqrt(
            (result["object_release_position"][0] - bin_debug["bin_center"][0]) ** 2
            + (result["object_release_position"][1] - bin_debug["bin_center"][1]) ** 2
        )
        release_inside_inner_bounds = (
            inner_bounds["x_min"] <= result["object_release_position"][0] <= inner_bounds["x_max"]
            and inner_bounds["y_min"] <= result["object_release_position"][1] <= inner_bounds["y_max"]
        )
        record["D_release_moment"] = {
            "ee_xyz_before_release": bin_debug["ee_position_before_release"], "ee_xyz_after_release": bin_debug["ee_position_after_release"],
            "object_center_xyz_before_release": result["object_release_position"],
            "object_center_xyz_after_release": result["object_position_immediately_after_release"],
            "object_aabb_before_release": bin_debug["object_aabb_before_release"], "object_aabb_after_release": bin_debug["object_aabb_after_release"],
            "object_bottom_z_before_release": object_bottom_before, "object_top_z_before_release": object_top_before,
            "object_bottom_z_after_release": object_bottom_after, "object_top_z_after_release": object_top_after,
            "rim_z": rim_z,
            "object_bottom_minus_rim_before_release": object_bottom_before - rim_z,
            "object_bottom_minus_rim_after_release": object_bottom_after - rim_z,
            "object_center_xy_error_from_bin_center": release_xy_error,
            "object_inside_inner_bounds_at_release": release_inside_inner_bounds,
            "grasp_constraint_removed": result["release_constraint_removed"],
            "object_gripper_separated_during_wait": bin_debug["object_separated_during_wait"],
        }

        # E. contact (phase-separated)
        robot_contact_by_subphase = {}
        for entry in bin_debug["robot_bin_contact_log"]:
            sp = entry["sub_phase"]
            agg = robot_contact_by_subphase.setdefault(sp, {"count": 0, "max_normal_force": 0.0, "first_step": entry["step"]})
            agg["count"] += entry["contact_count"]
            agg["max_normal_force"] = max(agg["max_normal_force"], entry["max_normal_force"])
        object_contact_by_subphase = {}
        for entry in bin_debug["object_bin_contact_log"]:
            sp = entry["sub_phase"]
            agg = object_contact_by_subphase.setdefault(sp, {"count": 0, "max_normal_force": 0.0, "first_step": entry["step"]})
            agg["count"] += entry["contact_count"]
            agg["max_normal_force"] = max(agg["max_normal_force"], entry["max_normal_force"])

        final_object_bin_contacts = classify_object_bin_contacts(backend)
        robot_bin_max_normal_force = max((agg["max_normal_force"] for agg in robot_contact_by_subphase.values()), default=0.0)
        robot_bin_contact_count_total = bin_debug["robot_bin_contact_count_total"]
        # --- contact-force classification (see this task's chat
        # report, section 4) -- diagnostic-only bucket, never fed into
        # production place_success. ---
        if robot_bin_contact_count_total == 0:
            contact_classification = "no_contact"
        elif robot_bin_max_normal_force == 0.0:
            contact_classification = "zero_force_grazing"
        elif robot_bin_max_normal_force <= LOW_FORCE_CONTACT_MAX_N:
            contact_classification = "low_force_contact"
        else:
            contact_classification = "meaningful_contact"

        record["E_contacts"] = {
            "robot_bin_contact_by_subphase": robot_contact_by_subphase,
            "robot_bin_contact_count_total": robot_bin_contact_count_total,
            "robot_bin_max_normal_force": robot_bin_max_normal_force,
            "contact_classification": contact_classification,
            "object_bin_contact_by_subphase_during_carry": object_contact_by_subphase,
            "object_bin_contact_before_release_count": bin_debug["object_bin_contact_before_release_count"],
            "object_bin_contact_after_wait_count": bin_debug["object_bin_contact_after_wait_count"],
            "final_object_wall_contacts": final_object_bin_contacts["walls"],
            "final_object_bottom_contact_count": final_object_bin_contacts["bottom_contact_count"],
        }

        # F. final state
        final_object_position = result["final_object_position"]
        final_aabb_min, final_aabb_max = p.getAABB(backend.object_id, physicsClientId=backend.client_id)
        final_object_aabb = {"min": list(final_aabb_min), "max": list(final_aabb_max)}
        final_inside_inner_bounds = (
            inner_bounds["x_min"] <= final_object_position[0] <= inner_bounds["x_max"]
            and inner_bounds["y_min"] <= final_object_position[1] <= inner_bounds["y_max"]
        )
        final_center_below_rim = final_object_position[2] < rim_z
        final_top_below_rim = final_object_aabb["max"][2] < rim_z
        # Pulled from the SAME bin_success_debug production already
        # computed (see benchmark.so101_scripted_expert's own
        # compute_bin_success_debug()) -- not recomputed independently,
        # so this can never silently drift from what production itself
        # used to decide place_success.
        bin_success_debug = result.get("bin_success_debug") or {}
        record["F_final_state"] = {
            "final_object_xyz": final_object_position, "final_object_aabb": final_object_aabb,
            "inside_inner_xy_bounds": final_inside_inner_bounds,
            "center_below_rim": final_center_below_rim, "top_below_rim": final_top_below_rim,
            "center_rim_delta": bin_success_debug.get("center_rim_delta"), "top_rim_delta": bin_success_debug.get("top_rim_delta"),
            "final_linear_speed_mps": result["final_linear_speed"], "final_angular_speed_radps": result["final_angular_speed"],
            "continuous_stable_steps_required": result["continuous_stable_steps_required"],
            "max_consecutive_stable_steps": result["max_consecutive_stable_steps"],
            "settle_steps_used": result["settle_steps_used"], "settle_success": result["settle_success"], "settle_timeout": result["settle_timeout"],
            "bottom_contact": final_object_bin_contacts["bottom_contact_count"] > 0,
            "wall_contact": any(v > 0 for v in final_object_bin_contacts["walls"].values()),
            "production_place_success": result["place_success"], "production_target_xy_error": result["object_target_xy_error_m"],
            "production_failure_reason": result["failure_reason"],
        }

        # --- stuck-on-rim candidate diagnostic (see this task's chat
        # report, section 6) -- NOT a production definition. ---
        near_boundary = (
            abs(final_object_position[0] - inner_bounds["x_min"]) <= NEAR_BOUNDARY_TOLERANCE_M
            or abs(final_object_position[0] - inner_bounds["x_max"]) <= NEAR_BOUNDARY_TOLERANCE_M
            or abs(final_object_position[1] - inner_bounds["y_min"]) <= NEAR_BOUNDARY_TOLERANCE_M
            or abs(final_object_position[1] - inner_bounds["y_max"]) <= NEAR_BOUNDARY_TOLERANCE_M
        )
        straddling_rim = (
            final_object_aabb["min"][2] <= rim_z + RIM_STRADDLE_TOLERANCE_M
            and final_object_aabb["max"][2] >= rim_z - RIM_STRADDLE_TOLERANCE_M
        )
        low_velocity = (
            result["final_linear_speed"] is not None and result["final_linear_speed"] <= LINEAR_SPEED_PASS_MPS
            and result["final_angular_speed"] is not None and result["final_angular_speed"] <= ANGULAR_SPEED_PASS_RADPS
        )
        has_wall_or_rim_contact = record["E_contacts"]["final_object_wall_contacts"] and any(record["E_contacts"]["final_object_wall_contacts"].values())
        stuck_on_rim_candidate = near_boundary and straddling_rim and low_velocity and has_wall_or_rim_contact
        record["stuck_on_rim_candidate_signals"] = {
            "near_boundary": near_boundary, "straddling_rim": straddling_rim, "low_velocity": low_velocity, "has_wall_or_rim_contact": has_wall_or_rim_contact,
        }

        # --- diagnostic_outcome classification (see this task's chat
        # report, section 5) -- priority order, single outcome per
        # seed, NEVER written into production place_success/
        # failure_reason. ---
        if not record["C_bin_waypoints"]["all_waypoints_reached"]:
            diagnostic_outcome = "waypoint_failed"
        elif record["E_contacts"]["robot_bin_contact_count_total"] > 0:
            diagnostic_outcome = "robot_bin_contact"
        elif not final_inside_inner_bounds:
            diagnostic_outcome = "object_missed_bin"
        elif stuck_on_rim_candidate:
            diagnostic_outcome = "object_stuck_on_rim"
        elif not (final_center_below_rim or final_top_below_rim):
            diagnostic_outcome = "object_not_below_rim"
        elif not result["settle_success"]:
            diagnostic_outcome = "object_not_settled"
        elif (
            result["release_constraint_removed"] and bin_debug["object_separated_during_wait"]
            and low_velocity and result["settle_success"]
        ):
            diagnostic_outcome = "success_candidate"
        else:
            diagnostic_outcome = "unknown"

        record["diagnostic_outcome"] = diagnostic_outcome
        return record
    finally:
        backend.close()


def percentile(values: list, p: float):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def stats_block(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"avg": None, "min": None, "max": None, "p50": None, "p95": None, "count": 0}
    return {
        "avg": sum(values) / len(values), "min": min(values), "max": max(values),
        "p50": percentile(values, 50), "p95": percentile(values, 95), "count": len(values),
    }


def summarize(records: list) -> dict:
    total = len(records)
    outcome_counts = {}
    for r in records:
        outcome_counts[r["diagnostic_outcome"]] = outcome_counts.get(r["diagnostic_outcome"], 0) + 1

    with_waypoints = [r for r in records if "C_bin_waypoints" in r]
    waypoint_errors = [r["C_bin_waypoints"]["max_waypoint_error_m"] for r in with_waypoints]
    with_release = [r for r in records if "D_release_moment" in r]
    release_xy_errors = [r["D_release_moment"]["object_center_xy_error_from_bin_center"] for r in with_release]
    release_bottom_rim_deltas = [r["D_release_moment"]["object_bottom_minus_rim_before_release"] for r in with_release]
    with_final = [r for r in records if "F_final_state" in r and r["F_final_state"].get("settle_steps_used") is not None]
    settle_steps = [r["F_final_state"]["settle_steps_used"] for r in with_final]
    final_xy_errors = [r["F_final_state"]["production_target_xy_error"] for r in with_final]
    final_center_rim_deltas = [r["F_final_state"].get("center_rim_delta") for r in with_final]
    with_contacts = [r for r in records if "E_contacts" in r]
    robot_bin_max_forces = [r["E_contacts"]["robot_bin_max_normal_force"] for r in with_contacts]

    def avg(values):
        values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else None

    # --- production failure_reason distribution (see this task's chat
    # report, section 8) -- read straight from run_pick_and_place_episode()'s
    # own result, not recomputed/reclassified here. ---
    production_failure_reason_counts = {}
    for r in with_final:
        if not r["F_final_state"]["production_place_success"]:
            reason = r["F_final_state"]["production_failure_reason"]
            production_failure_reason_counts[reason] = production_failure_reason_counts.get(reason, 0) + 1

    zero_force_grazing_seeds = [r["seed"] for r in with_contacts if r["E_contacts"]["contact_classification"] == "zero_force_grazing"]
    low_force_contact_seeds = [r["seed"] for r in with_contacts if r["E_contacts"]["contact_classification"] == "low_force_contact"]
    meaningful_contact_seeds = [r["seed"] for r in with_contacts if r["E_contacts"]["contact_classification"] == "meaningful_contact"]

    production_failed_seeds = [r["seed"] for r in with_final if not r["F_final_state"]["production_place_success"]]
    # Seeds that raised an exception before F_final_state ever got a
    # production_place_success field at all (grasp/lift/transport/
    # waypoint-exception/scene_invalid) also count as production
    # failures -- they simply never produced a place_success=True.
    production_failed_seeds += [r["seed"] for r in records if "F_final_state" not in r or "production_place_success" not in r["F_final_state"]]

    return {
        "total_seeds": total,
        "outcome_counts": outcome_counts,
        "success_candidate_count": outcome_counts.get("success_candidate", 0),
        "success_candidate_rate": (outcome_counts.get("success_candidate", 0) / total) if total else None,
        "scene_invalid_count": outcome_counts.get("scene_invalid", 0),
        "grasp_lift_transport_failure_count": (
            outcome_counts.get("grasp_failed", 0) + outcome_counts.get("lift_failed", 0) + outcome_counts.get("transport_failed", 0)
        ),
        "waypoint_failed_count": outcome_counts.get("waypoint_failed", 0),
        "robot_bin_contact_count": outcome_counts.get("robot_bin_contact", 0),
        "object_missed_bin_count": outcome_counts.get("object_missed_bin", 0),
        "object_stuck_on_rim_count": outcome_counts.get("object_stuck_on_rim", 0),
        "object_not_settled_count": outcome_counts.get("object_not_settled", 0),
        "avg_max_waypoint_error_m": avg(waypoint_errors), "max_max_waypoint_error_m": max(waypoint_errors) if waypoint_errors else None,
        "avg_release_xy_error_m": avg(release_xy_errors),
        "avg_release_bottom_minus_rim_m": avg(release_bottom_rim_deltas),
        "avg_settle_steps": avg(settle_steps),
        "failed_seeds": [r["seed"] for r in records if r["diagnostic_outcome"] != "success_candidate"],
        # --- production place_success comparison (see this task's chat
        # report, "기존 diagnostic benchmark와 정합성 확인") -- purely
        # additive output; diagnostic_outcome's own definition/logic is
        # completely untouched. production_place_success is read
        # straight from run_pick_and_place_episode()'s own result (now
        # bin-aware -- see benchmark.so101_scripted_expert's own
        # evaluate_bin_place_success()), not recomputed here. ---
        "production_place_success_count": sum(1 for r in with_final if r["F_final_state"]["production_place_success"]),
        "production_place_success_rate": (sum(1 for r in with_final if r["F_final_state"]["production_place_success"]) / total) if total else None,
        "production_failure_count": total - sum(1 for r in with_final if r["F_final_state"]["production_place_success"]),
        "production_failure_reason_counts": production_failure_reason_counts,
        "production_failed_seeds": sorted(set(production_failed_seeds)),
        "production_vs_diagnostic_agreement_count": sum(
            1 for r in with_final
            if r["F_final_state"]["production_place_success"] == (r["diagnostic_outcome"] == "success_candidate")
        ),
        "production_vs_diagnostic_mismatches": [
            {"seed": r["seed"], "diagnostic_outcome": r["diagnostic_outcome"], "production_place_success": r["F_final_state"]["production_place_success"], "production_failure_reason": r["F_final_state"]["production_failure_reason"]}
            for r in with_final
            if r["F_final_state"]["production_place_success"] != (r["diagnostic_outcome"] == "success_candidate")
        ],
        # --- contact-force classification summary (see this task's
        # chat report, section 4/5) -- diagnostic only. ---
        "zero_force_grazing_seeds": zero_force_grazing_seeds,
        "low_force_contact_seeds": low_force_contact_seeds,
        "meaningful_contact_seeds": meaningful_contact_seeds,
        # --- full stats blocks (avg/min/max/p50/p95) (see this task's
        # chat report, section 6) ---
        "stats_waypoint_error_m": stats_block(waypoint_errors),
        "stats_release_xy_error_m": stats_block(release_xy_errors),
        "stats_final_xy_error_m": stats_block(final_xy_errors),
        "stats_release_bottom_minus_rim_m": stats_block(release_bottom_rim_deltas),
        "stats_final_center_rim_delta_m": stats_block(final_center_rim_deltas),
        "stats_settle_steps": stats_block(settle_steps),
        "stats_robot_bin_max_normal_force_n": stats_block(robot_bin_max_forces),
    }


def print_seed_table(records: list) -> None:
    header = (
        f"{'seed':>4} | {'outcome':<22} | {'grasp':<6} | {'waypt':<6} | {'r-bin':<6} | "
        f"{'in-bin':<7} | {'ctr<rim':<8} | {'top<rim':<8} | {'settled':<8} | {'xy_err':<8} | {'bot-rim':<8} | {'prod':<6} | {'prod_reason':<20}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        seed = r["seed"]
        outcome = r["diagnostic_outcome"]
        grasp = "OK" if "B_grasp_transport" in r else "-"
        waypt = str(r.get("C_bin_waypoints", {}).get("all_waypoints_reached", "-"))
        rbin = str(r.get("E_contacts", {}).get("robot_bin_contact_count_total", "-"))
        in_bin = str(r.get("F_final_state", {}).get("inside_inner_xy_bounds", "-"))
        ctr_rim = str(r.get("F_final_state", {}).get("center_below_rim", "-"))
        top_rim = str(r.get("F_final_state", {}).get("top_below_rim", "-"))
        settled = str(r.get("F_final_state", {}).get("settle_success", "-"))
        xy_err = r.get("F_final_state", {}).get("production_target_xy_error")
        xy_err_str = f"{xy_err:.4f}" if isinstance(xy_err, (int, float)) else "-"
        bot_rim = r.get("D_release_moment", {}).get("object_bottom_minus_rim_before_release")
        bot_rim_str = f"{bot_rim:.4f}" if isinstance(bot_rim, (int, float)) else "-"
        prod = str(r.get("F_final_state", {}).get("production_place_success", "-"))
        prod_reason = str(r.get("F_final_state", {}).get("production_failure_reason", "-"))
        print(
            f"{seed:>4} | {outcome:<22} | {grasp:<6} | {waypt:<6} | {rbin:<6} | "
            f"{in_bin:<7} | {ctr_rim:<8} | {top_rim:<8} | {settled:<8} | {xy_err_str:<8} | {bot_rim_str:<8} | {prod:<6} | {prod_reason:<20}"
        )


def print_failure_analysis(records: list, summary: dict) -> None:
    """Per-seed breakdown for every PRODUCTION failure (see this
    task's chat report, section 7) -- if none, explicitly says so and
    analyzes grazing/contact seeds instead, per that section's own
    "실패 seed가 없다면... grazing/contact seed만 별도 분석"."""
    production_failed_seeds = set(summary["production_failed_seeds"])
    if not production_failed_seeds:
        print("실패 없음 (production_place_success=True for every seed).")
    for r in records:
        if r["seed"] not in production_failed_seeds:
            continue
        print(f"\n--- seed {r['seed']} (production FAILED) ---")
        print(f"  initial object position: {r.get('A_initial_scene', {}).get('object_initial_xyz')}")
        print(f"  bin center: {r.get('A_initial_scene', {}).get('bin_center_xyz')}")
        print(f"  production_failure_reason: {r.get('F_final_state', {}).get('production_failure_reason', r.get('exception', {}).get('failure_reason'))}")
        print(f"  diagnostic_outcome: {r['diagnostic_outcome']}")
        print(f"  grasp/lift/transport: {'OK' if 'B_grasp_transport' in r else 'FAILED (exception: ' + str(r.get('exception')) + ')'}")
        print(f"  waypoint status: {r.get('C_bin_waypoints', {}).get('waypoint_reached')}")
        print(f"  release status: separated={r.get('D_release_moment', {}).get('object_gripper_separated_during_wait')}, constraint_removed={r.get('D_release_moment', {}).get('grasp_constraint_removed')}")
        print(f"  contact: {r.get('E_contacts', {}).get('contact_classification')} (max_force={r.get('E_contacts', {}).get('robot_bin_max_normal_force')})")
        print(f"  final object pose: {r.get('F_final_state', {}).get('final_object_xyz')}")
        print(f"  inside-bin: {r.get('F_final_state', {}).get('inside_inner_xy_bounds')}  below-rim(center): {r.get('F_final_state', {}).get('center_below_rim')}  settled: {r.get('F_final_state', {}).get('settle_success')}")
        print(f"  most likely cause: {r.get('F_final_state', {}).get('production_failure_reason', r.get('exception', {}).get('failure_reason', 'unknown'))}")

    # --- grazing/contact seed analysis (always printed, regardless of
    # failures -- see this task's own "grazing 또는 contact seed 분석") ---
    print("\n--- grazing/contact seed analysis ---")
    for label, seeds in (
        ("zero_force_grazing", summary["zero_force_grazing_seeds"]),
        ("low_force_contact", summary["low_force_contact_seeds"]),
        ("meaningful_contact", summary["meaningful_contact_seeds"]),
    ):
        print(f"  {label}: {seeds}")
        for r in records:
            if r["seed"] in seeds:
                print(
                    f"    seed {r['seed']}: max_force={r['E_contacts']['robot_bin_max_normal_force']}, "
                    f"contact_count={r['E_contacts']['robot_bin_contact_count_total']}, "
                    f"production_place_success={r.get('F_final_state', {}).get('production_place_success')}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated explicit seed list, e.g. 0,1,2,3,4")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed", type=int, default=None, help="single seed (used with --gui)")
    parser.add_argument("--gui", action="store_true", help="run exactly ONE seed with a PyBullet GUI window")
    parser.add_argument("--x-range", type=str, default=f"{DEFAULT_X_RANGE[0]},{DEFAULT_X_RANGE[1]}")
    parser.add_argument("--y-range", type=str, default=f"{DEFAULT_Y_RANGE[0]},{DEFAULT_Y_RANGE[1]}")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--mode", type=str, default=RANDOMIZATION_MODE_COUPLED_SMALL,
        choices=[RANDOMIZATION_MODE_COUPLED_SMALL, RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY],
        help="coupled_small (default, unchanged): bin_center = object_position + offset. "
             "fixed_bin_object_xy: bin stays fixed, only object XY is independently randomized (see this task's chat report).",
    )
    parser.add_argument("--apply-yaw", action="store_true", help="only meaningful with --mode fixed_bin_object_xy -- also randomize object yaw within FIXED_BIN_OBJECT_YAW_RANGE_RAD")
    args = parser.parse_args()

    x_range = tuple(float(v) for v in args.x_range.split(","))
    y_range = tuple(float(v) for v in args.y_range.split(","))

    if args.gui:
        seeds = [args.seed if args.seed is not None else (args.seed_start if args.seeds is None else parse_seed_list(args.seeds)[0])]
        print(f"--gui: running exactly ONE seed ({seeds[0]})")
    elif args.seeds is not None:
        seeds = parse_seed_list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    fixed_bin_kwargs = {}
    if args.mode == RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY:
        nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
        fixed_bin_center_xy = [
            nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
        ]
        fixed_bin_kwargs = {
            "bin_center_override_xy": fixed_bin_center_xy,
            "scene_config": {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY},
        }
        if not args.seeds and args.x_range == f"{DEFAULT_X_RANGE[0]},{DEFAULT_X_RANGE[1]}":
            x_range = FIXED_BIN_OBJECT_X_RANGE
        if not args.seeds and args.y_range == f"{DEFAULT_Y_RANGE[0]},{DEFAULT_Y_RANGE[1]}":
            y_range = FIXED_BIN_OBJECT_Y_RANGE
        print(f"--mode fixed_bin_object_xy: fixed_bin_center_xy={fixed_bin_center_xy}, x_range={x_range}, y_range={y_range}, apply_yaw={args.apply_yaw}")

    records = []
    for seed in seeds:
        yaw = sample_object_yaw_rad(seed, FIXED_BIN_OBJECT_YAW_RANGE_RAD) if (args.mode == RANDOMIZATION_MODE_FIXED_BIN_OBJECT_XY and args.apply_yaw) else None
        record = run_single_seed_diagnostic(
            seed, x_range, y_range, gui=args.gui, randomization_mode=args.mode, object_yaw_rad=yaw, **fixed_bin_kwargs,
        )
        records.append(record)
        outcome = record["diagnostic_outcome"]
        print(f"[seed {seed}] diagnostic_outcome={outcome}")

    summary = summarize(records)

    output = {
        "config": {"seeds": seeds, "x_range": list(x_range), "y_range": list(y_range)},
        "records": records,
        "summary": summary,
    }

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print()
    print_seed_table(records)
    print()
    print("=== summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print()
    print_failure_analysis(records, summary)

    # --- GUI reproduction command (see this task's chat report,
    # section 9) -- a PRODUCTION failure takes priority; if there is
    # none, fall back to the grazing/highest-contact-force seed as the
    # representative GUI target instead of printing nothing. ---
    production_failed_seeds = summary["production_failed_seeds"]
    if production_failed_seeds:
        print(f"\nTo reproduce a PRODUCTION-FAILED seed in GUI:")
        print(f"  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic --gui --seed {production_failed_seeds[0]}")
    else:
        contact_records = [r for r in records if r.get("E_contacts", {}).get("robot_bin_contact_count_total", 0) > 0]
        if contact_records:
            representative = max(contact_records, key=lambda r: r["E_contacts"]["robot_bin_max_normal_force"])
            print(f"\nNo production failures -- representative grazing/highest-contact-force seed ({representative['seed']}, {representative['E_contacts']['contact_classification']}) to reproduce in GUI:")
            print(f"  .venv-vla/bin/python -m benchmark.benchmark_so101_bin_diagnostic --gui --seed {representative['seed']}")
        else:
            print("\nNo production failures and no robot-bin contact in any seed -- nothing to reproduce.")

    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
