"""SO-101 bin-aware scripted-expert place smoke test (see this task's
chat report, "bin에 맞는 안전한 place 경로"). Runs a full pick -> lift
-> transport -> BIN pre-place -> descend -> release -> wait -> retreat
-> settle episode with backend.use_bin=True, and checks ONLY structural/
safety conditions (waypoint reached, no wall tunneling, object ends up
near the bin) -- NOT the production place_success/failure_reason
judgment (that stays target_xy_error-based and UNCHANGED in
so101_scripted_expert.py; this file computes its own diagnostic-only
checks, never feeding them back into production code).

Does not touch benchmark/so101_scripted_expert.py's flat-target path,
does not run the 20-seed randomization benchmark, does not re-collect
any dataset, does not touch robot_sim/pybullet_panda_backend.py.

Run (headless):
  .venv-vla/bin/python -m benchmark.smoke_so101_bin_place

Run (GUI, to watch the episode live):
  .venv-vla/bin/python -m benchmark.smoke_so101_bin_place --gui
"""

import argparse
import json
import time
from pathlib import Path

import pybullet as p

from benchmark.so101_scripted_expert import So101ExpertError, run_pick_and_place_episode
from robot_sim.so101_pybullet_backend import So101PyBulletBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "bin_place_smoke.json"

# The scene-layout conflict this file used to work around locally
# (object spawn AABB overlapping the bin's own AABB under the flat
# marker's old default offset) is now handled by
# So101PyBulletBackend itself -- passing use_bin=True with no
# scene_config override now automatically resolves
# target_zone_offset_xy (and surface_footprint_xy) to their bin-safe
# production defaults (see robot_sim/so101_pybullet_backend.py's own
# DEFAULT_BIN_TARGET_ZONE_OFFSET_XY / DEFAULT_BIN_SURFACE_FOOTPRINT_XY
# and reset()'s own automatic validate_initial_scene_layout() call).
# TRANSPORT_DELTA_XY is read back from the backend's OWN resolved
# scene_config after reset() (see main()) rather than hardcoded here a
# second time, so this file can never silently drift from whatever the
# production default actually is.
WALL_NAMES = ["left_wall", "right_wall", "front_wall", "back_wall"]


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def classify_object_bin_contacts(backend: So101PyBulletBackend) -> dict:
    bin_info = backend.get_bin_debug_info()
    per_wall_contact_count = {}
    for wall_name in WALL_NAMES:
        wall_id = bin_info["body_ids"][wall_name]
        contacts = p.getContactPoints(bodyA=backend.object_id, bodyB=wall_id, physicsClientId=backend.client_id)
        per_wall_contact_count[wall_name] = len(contacts)
    bottom_id = bin_info["body_ids"]["bottom"]
    bottom_contacts = p.getContactPoints(bodyA=backend.object_id, bodyB=bottom_id, physicsClientId=backend.client_id)
    return {"walls": per_wall_contact_count, "bottom_contact_count": len(bottom_contacts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="open a PyBullet GUI window and slow down stepping so the episode is watchable")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    crashed = False
    crash_reason = None
    diagnostics = {}

    backend = So101PyBulletBackend(gui=args.gui, use_bin=True)
    if args.gui:
        # Local-to-this-script visualization aid only -- does not touch
        # So101PyBulletBackend's own step() definition; a human watching
        # the GUI window sees: lateral move to pre-place (well above the
        # walls) -> vertical descend -> gripper open -> object drop ->
        # vertical retreat -> settle.
        original_step = backend.step

        def slow_step(steps: int = 1) -> None:
            for _ in range(steps):
                original_step(1)
                time.sleep(1.0 / 240.0)

        backend.step = slow_step

    try:
        backend.reset()  # validates the initial scene layout itself (see reset()'s own call to validate_initial_scene_layout()) -- raises InvalidSceneLayoutError before this script ever proceeds if the scene is bad
        bin_info = backend.get_bin_debug_info()
        diagnostics["bin_info_read_pass"] = bin_info is not None
        diagnostics["layout_validation_passed"] = backend.get_scene_state()["layout_validation_passed"]

        # Transport must land the arm where THIS episode's own bin
        # actually is -- read back the backend's own resolved offset
        # rather than hardcoding it a second time in this file.
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        result = run_pick_and_place_episode(backend, transport_delta_xy)
        bin_debug = result["bin_place_debug"]

        final_object_position = result["final_object_position"]
        inner_bounds_ok = (
            bin_info["inner_x_min"] <= final_object_position[0] <= bin_info["inner_x_max"]
            and bin_info["inner_y_min"] <= final_object_position[1] <= bin_info["inner_y_max"]
        )
        below_rim = final_object_position[2] <= bin_info["rim_z"]
        object_bin_contacts = classify_object_bin_contacts(backend)
        any_object_bin_contact = any(object_bin_contacts["walls"].values()) or object_bin_contacts["bottom_contact_count"] > 0

        diagnostics.update({
            "pre_place_reached_pass": bin_debug["pre_place_reached"],
            "release_waypoint_reached_pass": bin_debug["descend_reached"],
            "gripper_open_performed_pass": result["release_constraint_removed"],
            "object_separated_pass": bin_debug["object_separated_during_wait"],
            "retreat_reached_pass": bin_debug["retreat_reached"],
            "no_wall_tunneling_pass": bin_debug["robot_bin_contact_count_total"] == 0,
            "robot_bin_contact_count_total": bin_debug["robot_bin_contact_count_total"],
            "robot_bin_contact_log": bin_debug["robot_bin_contact_log"],
            "object_near_bin_pass": inner_bounds_ok or any_object_bin_contact,
            "final_object_position": final_object_position,
            "inner_bounds_ok": inner_bounds_ok,
            "below_rim": below_rim,
            "final_linear_speed_mps": result["final_linear_speed"],
            "final_angular_speed_radps": result["final_angular_speed"],
            "object_bin_contact": object_bin_contacts,
            "pre_place_target": bin_debug["pre_place_target"], "pre_place_final_ee": bin_debug["pre_place_final_ee"], "pre_place_error_m": bin_debug["pre_place_error_m"],
            "release_target": bin_debug["release_target"], "descend_final_ee": bin_debug["descend_final_ee"], "descend_error_m": bin_debug["descend_error_m"],
            "retreat_target": bin_debug["retreat_target"], "retreat_final_ee": bin_debug["retreat_final_ee"], "retreat_error_m": bin_debug["retreat_error_m"],
            "release_wait_steps_used": bin_debug["release_wait_steps_used"],
            "bin_center": bin_debug["bin_center"], "rim_z": bin_debug["rim_z"],
            # Diagnostic-only -- NOT used as this test's pass condition
            # (see this task's own "이번 테스트의 핵심 pass 조건으로
            # 사용하지 말 것"), reported for context only.
            "production_place_success_diagnostic_only": result["place_success"],
            "production_target_xy_error_diagnostic_only": result["object_target_xy_error_m"],
        })
    except (So101ExpertError, Exception) as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    overall_pass = (
        not crashed
        and diagnostics.get("layout_validation_passed", False)
        and diagnostics.get("bin_info_read_pass", False)
        and diagnostics.get("pre_place_reached_pass", False)
        and diagnostics.get("release_waypoint_reached_pass", False)
        and diagnostics.get("gripper_open_performed_pass", False)
        and diagnostics.get("object_separated_pass", False)
        and diagnostics.get("retreat_reached_pass", False)
        and diagnostics.get("no_wall_tunneling_pass", False)
        and diagnostics.get("object_near_bin_pass", False)
    )

    output = {"crashed": crashed, "crash_reason": crash_reason, "overall_pass": overall_pass, "diagnostics": diagnostics}

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 bin-aware place smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        for key in (
            "layout_validation_passed", "bin_info_read_pass", "pre_place_reached_pass", "release_waypoint_reached_pass",
            "gripper_open_performed_pass", "object_separated_pass", "retreat_reached_pass",
            "no_wall_tunneling_pass", "object_near_bin_pass",
        ):
            print(f"{key}: {diagnostics[key]}")
        print(f"final_object_position: {diagnostics['final_object_position']}")
        print(f"inner_bounds_ok: {diagnostics['inner_bounds_ok']}  below_rim: {diagnostics['below_rim']}")
        print(f"final_linear_speed_mps: {diagnostics['final_linear_speed_mps']}  final_angular_speed_radps: {diagnostics['final_angular_speed_radps']}")
        print(f"object_bin_contact: {diagnostics['object_bin_contact']}")
        print(f"robot_bin_contact_count_total: {diagnostics['robot_bin_contact_count_total']}")
        print(f"release_wait_steps_used: {diagnostics['release_wait_steps_used']}")
        print(f"(diagnostic only, not this test's pass condition) production_place_success: {diagnostics['production_place_success_diagnostic_only']}, target_xy_error: {diagnostics['production_target_xy_error_diagnostic_only']}")
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
