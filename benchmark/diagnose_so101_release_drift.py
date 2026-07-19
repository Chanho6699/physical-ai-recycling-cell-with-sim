"""SO-101 release-drift diagnosis (see this task's chat report,
"release drift 원인 진단"). Pure diagnostic -- does NOT modify
so101_scripted_expert.py, does NOT change any waypoint/threshold/
transport/gripper value, does NOT touch robot_sim/pybullet_panda_backend.py.

Traces every physics step around release for a fixed set of seeds:
  - 60 steps immediately before the gripper-open command is issued
    (pure passive observation -- no new command issued by this script
    during this window; the arm/object are already at whatever state
    place_descend's own convergence left them in).
  - The ENTIRE gripper-open transition, at per-physics-step resolution.
    Achieved by calling So101PyBulletBackend.set_gripper(1.0,
    settle_steps=1) repeatedly (DEFAULT_SETTLE_STEPS=40 times) instead
    of once with settle_steps=40 -- behaviorally IDENTICAL (same target
    re-issued each call is a no-op change, same total physics steps,
    same _maybe_release_grasp() check after each step, just checked at
    finer time resolution) -- NOT a modification of set_gripper() or
    of expert behavior, purely calling the SAME existing public method
    with its own existing settle_steps parameter.
  - 240 steps after the gripper-open transition completes.

Reuses so101_scripted_expert.py's own gripper_phase()/move_to_target()/
object_offset_in_ee_frame() and waypoint constants EXACTLY as
run_pick_and_place_episode() itself does, to reach the SAME
pre-release state (pre_grasp -> approach -> grasp -> lift -> transport
-> place_descend) -- this script does not reimplement or alter that
sequence, it only takes over AFTER place_descend, before release, to
get finer-grained observation than run_pick_and_place_episode() itself
records.

This SO-101 URDF's gripper is NOT a symmetric two-finger gripper -- it
is one FIXED jaw ("gripper_frame_link", link index discovered at
runtime from joint_info_by_name) and one MOVING jaw
("moving_jaw_so101_v1_link", the actuated "gripper" joint's own link)
-- see this task's chat report for how this was confirmed (URDF/joint
inspection). Contacts are separated by these two links, not by a
"left/right" convention that does not exist in this hardware design.

Run:
  .venv-vla/bin/python -m benchmark.diagnose_so101_release_drift
"""

import argparse
import json
import math
from pathlib import Path

import pybullet as p

from benchmark.evaluate_so101_expert_small_randomization import (
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    TRANSPORT_DELTA_XY,
    sample_object_position,
)
from benchmark.so101_scripted_expert import (
    ANGULAR_SPEED_PASS_RADPS,
    APPROACH_OFFSET_M,
    DRIFT_WINDOW_STEPS,
    FAILURE_GRASP_FAILED,
    FAILURE_IK_FAILED,
    FAILURE_LIFT_FAILED,
    FAILURE_OBJECT_DROPPED,
    LIFT_DISTANCE_M,
    LIFT_MAX_STEPS,
    LINEAR_SPEED_PASS_MPS,
    MAX_MOVE_STEPS,
    PHASE_APPROACH,
    PHASE_GRASP,
    PHASE_LIFT,
    PHASE_PLACE_DESCEND,
    PHASE_PRE_GRASP,
    PHASE_TRANSPORT,
    PLACE_APPROACH_HEIGHT_ABOVE_SURFACE_M,
    PRE_GRASP_OFFSET_M,
    CONTINUOUS_STABLE_STEPS,
    SETTLE_DRIFT_PASS_M,
    TRANSPORT_MAX_STEPS,
    So101ExpertError,
    gripper_phase,
    move_to_target,
    object_offset_in_ee_frame,
)
from robot_sim.so101_pybullet_backend import DEFAULT_SETTLE_STEPS, So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = "results/so101_release_drift_diagnosis.json"

FAILED_SEEDS = [0, 6, 19]
COMPARISON_SUCCESS_SEEDS = [2, 3, 4, 7, 8]

PRE_RELEASE_STEPS = 60
POST_RELEASE_STEPS = 240


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def object_tilt_angle_degrees(object_orientation: list) -> float:
    matrix = p.getMatrixFromQuaternion(object_orientation)
    local_z_dot_world_z = matrix[8]  # rotation matrix (row-major 3x3): local z-axis's world-z component
    local_z_dot_world_z = max(-1.0, min(1.0, local_z_dot_world_z))
    return math.degrees(math.acos(local_z_dot_world_z))


def run_up_to_place_descend(backend: So101PyBulletBackend, transport_delta_xy: list) -> list:
    """Reuses so101_scripted_expert.py's own building blocks EXACTLY as
    run_pick_and_place_episode() itself calls them, in the same order,
    with the same constants -- stops right before the release
    gripper_phase() call so this script can take over with finer-
    grained instrumentation. Not a reimplementation of expert logic,
    not a modification of it -- same functions, same order, same
    values."""
    object_position, _ = backend.get_object_pose()
    scene = backend.get_scene_state()
    surface_height = scene["table_top_z"]
    target_zone_center_xy = scene["target_zone_center_xy"]

    gripper_phase(backend, PHASE_PRE_GRASP, 1.0)
    move_to_target(backend, [object_position[i] + PRE_GRASP_OFFSET_M[i] for i in range(3)], PHASE_PRE_GRASP, MAX_MOVE_STEPS, FAILURE_IK_FAILED)
    move_to_target(backend, [object_position[i] + APPROACH_OFFSET_M[i] for i in range(3)], PHASE_APPROACH, MAX_MOVE_STEPS, FAILURE_IK_FAILED)
    gripper_phase(backend, PHASE_GRASP, 0.0)
    if not backend.is_grasped():
        raise So101ExpertError("grasp was not established", FAILURE_GRASP_FAILED, phase=PHASE_GRASP)

    ee_pre_lift, _ = backend.get_end_effector_pose()
    lift_target = [ee_pre_lift[0], ee_pre_lift[1], ee_pre_lift[2] + LIFT_DISTANCE_M]
    lift_result = move_to_target(backend, lift_target, PHASE_LIFT, LIFT_MAX_STEPS, FAILURE_LIFT_FAILED, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp lost during lift", FAILURE_OBJECT_DROPPED, phase=PHASE_LIFT)

    ee_lift_final = lift_result["final_ee_position"]
    transport_target = [ee_lift_final[0] + transport_delta_xy[0], ee_lift_final[1] + transport_delta_xy[1], ee_lift_final[2]]
    transport_result = move_to_target(backend, transport_target, PHASE_TRANSPORT, TRANSPORT_MAX_STEPS, FAILURE_IK_FAILED, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp lost during transport", FAILURE_OBJECT_DROPPED, phase=PHASE_TRANSPORT)

    ee_transport_final = transport_result["final_ee_position"]
    place_target = [ee_transport_final[0], ee_transport_final[1], surface_height + PLACE_APPROACH_HEIGHT_ABOVE_SURFACE_M]
    descend_result = move_to_target(backend, place_target, PHASE_PLACE_DESCEND, LIFT_MAX_STEPS, FAILURE_IK_FAILED, track_grasp=True)
    if not backend.is_grasped():
        raise So101ExpertError("grasp lost during place_descend", FAILURE_OBJECT_DROPPED, phase=PHASE_PLACE_DESCEND)

    return target_zone_center_xy


def get_object_contacts(backend: So101PyBulletBackend, fixed_jaw_link: int, moving_jaw_link: int) -> list:
    raw = p.getContactPoints(bodyA=backend.object_id, physicsClientId=backend.client_id)
    classified = []
    for c in raw:
        body_b, link_b = c[2], c[4]
        if body_b == backend.robot_id and link_b == fixed_jaw_link:
            label = "fixed_jaw"
        elif body_b == backend.robot_id and link_b == moving_jaw_link:
            label = "moving_jaw"
        elif body_b == backend.table_id:
            label = "table"
        else:
            label = f"body{body_b}_link{link_b}"
        classified.append({
            "label": label, "position_on_object": list(c[5]), "position_on_other": list(c[6]),
            "normal_direction": list(c[7]), "distance": c[8], "normal_force": c[9],
        })
    return classified


def record_entry(backend: So101PyBulletBackend, phase: str, step: int, target_zone_center_xy: list, fixed_jaw_link: int, moving_jaw_link: int) -> dict:
    ee_position, ee_orientation = backend.get_end_effector_pose()
    ee_link_state = p.getLinkState(backend.robot_id, backend.ee_link_index, computeLinkVelocity=True, physicsClientId=backend.client_id)
    ee_linear_velocity, ee_angular_velocity = list(ee_link_state[6]), list(ee_link_state[7])

    object_position, object_orientation = backend.get_object_pose()
    object_linear_velocity, object_angular_velocity = backend.get_object_velocity()

    gripper_state = p.getJointState(backend.robot_id, backend.gripper_joint_index, physicsClientId=backend.client_id)
    gripper_joint_position, gripper_joint_velocity = gripper_state[0], gripper_state[1]

    relative_offset = object_offset_in_ee_frame(ee_position, ee_orientation, object_position)
    to_target_xy = [object_position[0] - target_zone_center_xy[0], object_position[1] - target_zone_center_xy[1]]

    contacts = get_object_contacts(backend, fixed_jaw_link, moving_jaw_link)
    fixed_jaw_contacts = [c for c in contacts if c["label"] == "fixed_jaw"]
    moving_jaw_contacts = [c for c in contacts if c["label"] == "moving_jaw"]
    table_contacts = [c for c in contacts if c["label"] == "table"]

    return {
        "step": step, "phase": phase,
        "ee_position": ee_position, "ee_orientation": ee_orientation,
        "ee_linear_velocity": ee_linear_velocity, "ee_angular_velocity": ee_angular_velocity,
        "object_position": object_position, "object_orientation": object_orientation,
        "object_linear_velocity": object_linear_velocity, "object_angular_velocity": object_angular_velocity,
        "object_linear_speed_mps": math.sqrt(sum(v ** 2 for v in object_linear_velocity)),
        "object_angular_speed_radps": math.sqrt(sum(v ** 2 for v in object_angular_velocity)),
        "object_tilt_deg": object_tilt_angle_degrees(object_orientation),
        "gripper_joint_position": gripper_joint_position, "gripper_joint_velocity": gripper_joint_velocity,
        "object_to_ee_relative_position": relative_offset,
        "object_to_target_xy": to_target_xy,
        "is_grasped": backend.is_grasped(),
        "contacts": contacts,
        "fixed_jaw_contact_count": len(fixed_jaw_contacts),
        "moving_jaw_contact_count": len(moving_jaw_contacts),
        "table_contact_count": len(table_contacts),
        "fixed_jaw_max_normal_force": max((c["normal_force"] for c in fixed_jaw_contacts), default=0.0),
        "moving_jaw_max_normal_force": max((c["normal_force"] for c in moving_jaw_contacts), default=0.0),
        "table_max_normal_force": max((c["normal_force"] for c in table_contacts), default=0.0),
    }


def displacement_vector(start_xy: list, end_xy: list) -> dict:
    dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
    return {"dx": dx, "dy": dy, "magnitude": math.sqrt(dx ** 2 + dy ** 2), "direction_deg": math.degrees(math.atan2(dy, dx))}


def find_settle_completion(entries: list) -> dict:
    """Post-hoc, using the SAME continuous-stability criteria as
    so101_scripted_expert.py's own settle judgment (imported constants,
    not retyped), applied within this script's own recorded window."""
    consecutive = 0
    for i, e in enumerate(entries):
        if i < DRIFT_WINDOW_STEPS:
            passes = False
            drift = None
        else:
            start_pos = entries[i - DRIFT_WINDOW_STEPS]["object_position"]
            end_pos = e["object_position"]
            drift = math.sqrt(sum((end_pos[k] - start_pos[k]) ** 2 for k in range(3)))
            passes = (
                e["object_linear_speed_mps"] <= LINEAR_SPEED_PASS_MPS
                and e["object_angular_speed_radps"] <= ANGULAR_SPEED_PASS_RADPS
                and drift <= SETTLE_DRIFT_PASS_M
            )
        consecutive = consecutive + 1 if passes else 0
        if consecutive >= CONTINUOUS_STABLE_STEPS:
            return {"achieved": True, "step": e["step"], "object_position": e["object_position"]}
    return {"achieved": False, "step": None, "object_position": None}


def diagnose_seed(seed: int) -> dict:
    object_position = sample_object_position(seed, DEFAULT_X_RANGE, DEFAULT_Y_RANGE)
    backend = So101PyBulletBackend(gui=False, object_position=object_position)
    try:
        backend.reset()
        fixed_jaw_link = backend.joint_info_by_name["gripper_frame_joint"]["index"]
        moving_jaw_link = backend.joint_info_by_name["gripper"]["index"]

        try:
            target_zone_center_xy = run_up_to_place_descend(backend, TRANSPORT_DELTA_XY)
        except So101ExpertError as exc:
            return {"seed": seed, "pre_release_failure": True, "failure_reason": exc.failure_reason, "failure_phase": exc.phase}

        object_release_position = backend.get_object_position()  # "release 시작" / point A reference

        step = 0
        pre_release_entries = []
        for _ in range(PRE_RELEASE_STEPS):
            backend.step(1)
            step += 1
            pre_release_entries.append(record_entry(backend, "pre_release", step, target_zone_center_xy, fixed_jaw_link, moving_jaw_link))

        was_grasped_before_open = backend.is_grasped()
        gripper_open_entries = []
        finger_separation_step = None
        finger_separation_entry = None
        for _ in range(DEFAULT_SETTLE_STEPS):
            backend.set_gripper(1.0, settle_steps=1)
            step += 1
            entry = record_entry(backend, "gripper_open", step, target_zone_center_xy, fixed_jaw_link, moving_jaw_link)
            gripper_open_entries.append(entry)
            if finger_separation_step is None and was_grasped_before_open and not backend.is_grasped():
                finger_separation_step = step
                finger_separation_entry = entry

        post_release_entries = []
        first_floor_contact_step = None
        first_floor_contact_entry = None
        for _ in range(POST_RELEASE_STEPS):
            backend.step(1)
            step += 1
            entry = record_entry(backend, "post_release", step, target_zone_center_xy, fixed_jaw_link, moving_jaw_link)
            post_release_entries.append(entry)
            if first_floor_contact_step is None and entry["table_contact_count"] > 0:
                first_floor_contact_step = step
                first_floor_contact_entry = entry

        settle = find_settle_completion(post_release_entries)
        final_entry = post_release_entries[-1]

        # --- Section 3: error decomposition points A-H ---
        point_A_xy = [object_release_position[0], object_release_position[1]]
        point_B_xy = [gripper_open_entries[0]["object_position"][0], gripper_open_entries[0]["object_position"][1]]
        point_D_xy = [finger_separation_entry["object_position"][0], finger_separation_entry["object_position"][1]] if finger_separation_entry else None
        point_F_xy = [first_floor_contact_entry["object_position"][0], first_floor_contact_entry["object_position"][1]] if first_floor_contact_entry else None
        point_H_xy = [final_entry["object_position"][0], final_entry["object_position"][1]]
        point_G_xy = list(settle["object_position"][:2]) if settle["achieved"] else point_H_xy

        vectors = {
            "release_to_separation": displacement_vector(point_A_xy, point_D_xy) if point_D_xy else None,
            "separation_to_floor_contact": displacement_vector(point_D_xy, point_F_xy) if (point_D_xy and point_F_xy) else None,
            "floor_contact_to_settle": displacement_vector(point_F_xy, point_G_xy) if point_F_xy else None,
            "release_to_final": displacement_vector(point_A_xy, point_H_xy),
        }

        # --- Section 5: additional comparison signals ---
        last_pre_release = pre_release_entries[-1]
        ee_speed_before_release = math.sqrt(sum(v ** 2 for v in last_pre_release["ee_linear_velocity"]))
        ee_angular_speed_before_release = math.sqrt(sum(v ** 2 for v in last_pre_release["ee_angular_velocity"]))
        object_angular_speed_before_release = last_pre_release["object_angular_speed_radps"]
        finger_force_asymmetry = None
        if finger_separation_entry:
            finger_force_asymmetry = finger_separation_entry["fixed_jaw_max_normal_force"] - finger_separation_entry["moving_jaw_max_normal_force"]
        # max jaw-force asymmetry observed anywhere during gripper_open (peak, not just at separation)
        max_finger_force_asymmetry_during_open = max(
            (e["fixed_jaw_max_normal_force"] - e["moving_jaw_max_normal_force"] for e in gripper_open_entries), key=abs, default=None,
        )
        separation_object_velocity = finger_separation_entry["object_linear_velocity"] if finger_separation_entry else None
        first_floor_contact_impulse = first_floor_contact_entry["table_max_normal_force"] if first_floor_contact_entry else None

        def displacement_after(step_index_in_post, n_steps):
            if first_floor_contact_step is None:
                return None
            start = first_floor_contact_entry
            end_index = (first_floor_contact_step - 1) + n_steps  # index into post_release_entries (0-based)
            if end_index >= len(post_release_entries):
                return None
            end = post_release_entries[end_index]
            return displacement_vector([start["object_position"][0], start["object_position"][1]], [end["object_position"][0], end["object_position"][1]])

        movement_after_floor_contact = {
            "30_steps": displacement_after(first_floor_contact_step, 30),
            "60_steps": displacement_after(first_floor_contact_step, 60),
            "120_steps": displacement_after(first_floor_contact_step, 120),
        }

        release_height = object_release_position[2]
        object_tilt_at_release = last_pre_release["object_tilt_deg"]

        return {
            "seed": seed,
            "pre_release_failure": False,
            "object_position": object_position,
            "target_zone_center_xy": target_zone_center_xy,
            "points": {
                "A_release_start_xy": point_A_xy,
                "B_gripper_open_start_xy": point_B_xy,
                "D_finger_separation_xy": point_D_xy,
                "F_first_floor_contact_xy": point_F_xy,
                "G_settle_complete_xy": point_G_xy,
                "H_final_xy": point_H_xy,
            },
            "step_markers": {
                "finger_separation_step": finger_separation_step,
                "first_floor_contact_step": first_floor_contact_step,
                "settle_complete_step": settle["step"], "settle_achieved_within_window": settle["achieved"],
            },
            "vectors": vectors,
            "signals": {
                "ee_speed_before_release_mps": ee_speed_before_release,
                "ee_angular_speed_before_release_radps": ee_angular_speed_before_release,
                "object_angular_speed_before_release_radps": object_angular_speed_before_release,
                "finger_force_asymmetry_at_separation_n": finger_force_asymmetry,
                "max_finger_force_asymmetry_during_open_n": max_finger_force_asymmetry_during_open,
                "separation_object_velocity_mps": separation_object_velocity,
                "first_floor_contact_impulse_n": first_floor_contact_impulse,
                "movement_after_floor_contact": movement_after_floor_contact,
                "release_height_m": release_height,
                "object_tilt_deg_at_release": object_tilt_at_release,
                "final_drift_direction_deg": vectors["release_to_final"]["direction_deg"],
            },
            "final_xy_error_from_target": math.sqrt(
                (point_H_xy[0] - target_zone_center_xy[0]) ** 2 + (point_H_xy[1] - target_zone_center_xy[1]) ** 2
            ),
            "full_trace": {"pre_release": pre_release_entries, "gripper_open": gripper_open_entries, "post_release": post_release_entries},
        }
    finally:
        backend.close()


def summarize(diagnoses: list) -> dict:
    with_data = [d for d in diagnoses if not d.get("pre_release_failure")]

    def avg(values):
        values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else None

    def group_avg(field_path_fn, seeds):
        return avg([field_path_fn(d) for d in with_data if d["seed"] in seeds])

    per_seed_total_drift = {str(d["seed"]): d["vectors"]["release_to_final"] for d in with_data}
    per_seed_phase_breakdown = {
        str(d["seed"]): {
            "release_to_separation": d["vectors"]["release_to_separation"],
            "separation_to_floor_contact": d["vectors"]["separation_to_floor_contact"],
            "floor_contact_to_settle": d["vectors"]["floor_contact_to_settle"],
        } for d in with_data
    }
    per_seed_finger_asymmetry = {str(d["seed"]): d["signals"]["max_finger_force_asymmetry_during_open_n"] for d in with_data}
    per_seed_floor_impact = {str(d["seed"]): d["signals"]["first_floor_contact_impulse_n"] for d in with_data}

    return {
        "per_seed_total_release_drift_vector": per_seed_total_drift,
        "per_seed_phase_breakdown": per_seed_phase_breakdown,
        "per_seed_finger_force_asymmetry_n": per_seed_finger_asymmetry,
        "per_seed_first_floor_contact_impulse_n": per_seed_floor_impact,
        "failed_group_avg_total_drift_magnitude": group_avg(lambda d: d["vectors"]["release_to_final"]["magnitude"], FAILED_SEEDS),
        "success_group_avg_total_drift_magnitude": group_avg(lambda d: d["vectors"]["release_to_final"]["magnitude"], COMPARISON_SUCCESS_SEEDS),
        "failed_group_avg_ee_speed_before_release": group_avg(lambda d: d["signals"]["ee_speed_before_release_mps"], FAILED_SEEDS),
        "success_group_avg_ee_speed_before_release": group_avg(lambda d: d["signals"]["ee_speed_before_release_mps"], COMPARISON_SUCCESS_SEEDS),
        "failed_group_avg_object_angular_speed_before_release": group_avg(lambda d: d["signals"]["object_angular_speed_before_release_radps"], FAILED_SEEDS),
        "success_group_avg_object_angular_speed_before_release": group_avg(lambda d: d["signals"]["object_angular_speed_before_release_radps"], COMPARISON_SUCCESS_SEEDS),
        "failed_group_avg_floor_contact_impulse": group_avg(lambda d: d["signals"]["first_floor_contact_impulse_n"], FAILED_SEEDS),
        "success_group_avg_floor_contact_impulse": group_avg(lambda d: d["signals"]["first_floor_contact_impulse_n"], COMPARISON_SUCCESS_SEEDS),
        "failed_group_avg_finger_force_asymmetry": group_avg(lambda d: d["signals"]["max_finger_force_asymmetry_during_open_n"], FAILED_SEEDS),
        "success_group_avg_finger_force_asymmetry": group_avg(lambda d: d["signals"]["max_finger_force_asymmetry_during_open_n"], COMPARISON_SUCCESS_SEEDS),
        "failed_group_drift_directions_deg": {str(d["seed"]): d["vectors"]["release_to_final"]["direction_deg"] for d in with_data if d["seed"] in FAILED_SEEDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    seeds = FAILED_SEEDS + COMPARISON_SUCCESS_SEEDS
    diagnoses = []
    for seed in seeds:
        d = diagnose_seed(seed)
        diagnoses.append(d)
        if d.get("pre_release_failure"):
            print(f"[seed {seed}] PRE-RELEASE FAILURE ({d['failure_reason']} @ {d['failure_phase']})")
        else:
            v = d["vectors"]["release_to_final"]
            print(f"[seed {seed}] total_drift=({v['dx']:.4f},{v['dy']:.4f}) mag={v['magnitude']:.4f} dir={v['direction_deg']:.1f}deg "
                  f"final_xy_error={d['final_xy_error_from_target']:.4f} finger_sep_step={d['step_markers']['finger_separation_step']} "
                  f"floor_contact_step={d['step_markers']['first_floor_contact_step']} settle_step={d['step_markers']['settle_complete_step']}")

    summary = summarize(diagnoses)

    output = {
        "config": {
            "failed_seeds": FAILED_SEEDS, "comparison_success_seeds": COMPARISON_SUCCESS_SEEDS,
            "pre_release_steps": PRE_RELEASE_STEPS, "post_release_steps": POST_RELEASE_STEPS,
        },
        "diagnoses": diagnoses,
        "summary": summary,
    }

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== release drift diagnosis summary ===")
    print(f"per_seed_total_release_drift_vector: {json.dumps(summary['per_seed_total_release_drift_vector'], indent=2, default=str)}")
    print(f"per_seed_phase_breakdown: {json.dumps(summary['per_seed_phase_breakdown'], indent=2, default=str)}")
    print(f"per_seed_finger_force_asymmetry_n: {summary['per_seed_finger_force_asymmetry_n']}")
    print(f"per_seed_first_floor_contact_impulse_n: {summary['per_seed_first_floor_contact_impulse_n']}")
    print(f"failed_group_avg_total_drift_magnitude: {summary['failed_group_avg_total_drift_magnitude']}")
    print(f"success_group_avg_total_drift_magnitude: {summary['success_group_avg_total_drift_magnitude']}")
    print(f"failed_group_avg_ee_speed_before_release: {summary['failed_group_avg_ee_speed_before_release']}")
    print(f"success_group_avg_ee_speed_before_release: {summary['success_group_avg_ee_speed_before_release']}")
    print(f"failed_group_avg_object_angular_speed_before_release: {summary['failed_group_avg_object_angular_speed_before_release']}")
    print(f"success_group_avg_object_angular_speed_before_release: {summary['success_group_avg_object_angular_speed_before_release']}")
    print(f"failed_group_avg_floor_contact_impulse: {summary['failed_group_avg_floor_contact_impulse']}")
    print(f"success_group_avg_floor_contact_impulse: {summary['success_group_avg_floor_contact_impulse']}")
    print(f"failed_group_avg_finger_force_asymmetry: {summary['failed_group_avg_finger_force_asymmetry']}")
    print(f"success_group_avg_finger_force_asymmetry: {summary['success_group_avg_finger_force_asymmetry']}")
    print(f"failed_group_drift_directions_deg: {summary['failed_group_drift_directions_deg']}")
    print(f"\nResult JSON: {output_path}")


if __name__ == "__main__":
    main()
