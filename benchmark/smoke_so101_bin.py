"""SO-101 open-top bin V1 structural/physics smoke test (see this
task's chat report, "target marker를 실제 충돌이 있는 open-top bin으로
확장"). Verifies ONLY the bin's physical structure -- does not touch
benchmark/so101_scripted_expert.py, does not run the 20-seed
benchmark, does not change place/release waypoints or success
judgment, does not touch robot_sim/pybullet_panda_backend.py.

Covers (see this task's own section 11):
  A. structure  -- 5 bodies (1 bottom + 4 walls), all mass=0.0, all
     with a real collision shape
  B. dimensions -- inner/outer width/length, wall height, bottom
     thickness match the requested constants
  C. heights    -- bottom_center_z/bottom_top_z/wall_center_z/rim_z
     match the documented formulas
  D. dynamics   -- lateralFriction/rollingFriction/spinningFriction/
     restitution actually applied (read back via p.getDynamicsInfo(),
     not just asserted from the constants)
  E. drop test  -- a small test object dropped from above bin center
     comes to rest ON the bin bottom, inside the inner footprint, and
     never falls through the table
  F. wall test  -- a small test object pushed horizontally into a wall
     registers a contact against that wall's own body id and does not
     tunnel through it
  G. lifecycle  -- reset() called multiple times never leaves stale/
     duplicated bin bodies

Run:
  .venv-vla/bin/python -m benchmark.smoke_so101_bin
"""

import argparse
import json
import math
from pathlib import Path

import pybullet as p

from robot_sim.so101_pybullet_backend import (
    BIN_BOTTOM_THICKNESS_M,
    BIN_INNER_LENGTH_M,
    BIN_INNER_WIDTH_M,
    BIN_LATERAL_FRICTION,
    BIN_RESTITUTION,
    BIN_ROLLING_FRICTION,
    BIN_SPINNING_FRICTION,
    BIN_WALL_HEIGHT_M,
    BIN_WALL_THICKNESS_M,
    So101PyBulletBackend,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "so101" / "bin_smoke.json"

DIMENSION_TOLERANCE_M = 1e-6
HEIGHT_TOLERANCE_M = 1e-6
DYNAMICS_TOLERANCE = 1e-6

DROP_TEST_HALF_EXTENT_M = 0.01
DROP_TEST_MASS_KG = 0.03
DROP_TEST_START_HEIGHT_ABOVE_RIM_M = 0.05
DROP_TEST_STEPS = 240
DROP_RESTING_HEIGHT_TOLERANCE_M = 0.01

WALL_TEST_HALF_EXTENT_M = 0.01
WALL_TEST_MASS_KG = 0.03
WALL_TEST_GAP_FROM_WALL_M = 0.03
WALL_TEST_INITIAL_SPEED_MPS = 0.4
WALL_TEST_STEPS = 120
WALL_TEST_PENETRATION_TOLERANCE_M = 0.005


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def close_enough(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def check_structure(backend: So101PyBulletBackend) -> dict:
    body_ids = backend.bin_body_ids
    part_names = ["bottom", "left_wall", "right_wall", "front_wall", "back_wall"]
    ids = [body_ids[name] for name in part_names]

    mass_checks = {}
    collision_checks = {}
    for name, body_id in zip(part_names, ids):
        dynamics_info = p.getDynamicsInfo(body_id, -1, physicsClientId=backend.client_id)
        mass_checks[name] = dynamics_info[0]
        collision_shapes = p.getCollisionShapeData(body_id, -1, physicsClientId=backend.client_id)
        collision_checks[name] = len(collision_shapes) > 0

    return {
        "num_bodies": len(set(ids)),
        "num_bodies_pass": len(set(ids)) == 5,
        "has_bottom_and_four_walls": set(part_names) == set(body_ids.keys()) - {"all"},
        "all_ids_valid": all(i is not None and i >= 0 for i in ids),
        "mass_by_part": mass_checks,
        "all_mass_zero_pass": all(m == 0.0 for m in mass_checks.values()),
        "collision_shape_present_by_part": collision_checks,
        "all_have_collision_pass": all(collision_checks.values()),
    }


def check_dimensions(bin_info: dict) -> dict:
    inner_width = bin_info["inner_x_max"] - bin_info["inner_x_min"]
    inner_length = bin_info["inner_y_max"] - bin_info["inner_y_min"]
    outer_width = bin_info["outer_x_max"] - bin_info["outer_x_min"]
    outer_length = bin_info["outer_y_max"] - bin_info["outer_y_min"]

    return {
        "inner_width_m": inner_width, "inner_width_pass": close_enough(inner_width, BIN_INNER_WIDTH_M, DIMENSION_TOLERANCE_M),
        "inner_length_m": inner_length, "inner_length_pass": close_enough(inner_length, BIN_INNER_LENGTH_M, DIMENSION_TOLERANCE_M),
        "outer_width_m": outer_width, "outer_width_pass": close_enough(outer_width, BIN_INNER_WIDTH_M + 2 * BIN_WALL_THICKNESS_M, DIMENSION_TOLERANCE_M),
        "outer_length_m": outer_length, "outer_length_pass": close_enough(outer_length, BIN_INNER_LENGTH_M + 2 * BIN_WALL_THICKNESS_M, DIMENSION_TOLERANCE_M),
        "wall_height_m": bin_info["wall_height"], "wall_height_pass": close_enough(bin_info["wall_height"], BIN_WALL_HEIGHT_M, DIMENSION_TOLERANCE_M),
        "bottom_thickness_m": bin_info["bottom_thickness"], "bottom_thickness_pass": close_enough(bin_info["bottom_thickness"], BIN_BOTTOM_THICKNESS_M, DIMENSION_TOLERANCE_M),
    }


def check_heights(bin_info: dict) -> dict:
    table_surface_z = bin_info["table_surface_z"]
    expected_bottom_center_z = table_surface_z + bin_info["bottom_thickness"] / 2.0
    expected_bottom_top_z = table_surface_z + bin_info["bottom_thickness"]
    expected_wall_center_z = expected_bottom_top_z + bin_info["wall_height"] / 2.0
    expected_rim_z = expected_bottom_top_z + bin_info["wall_height"]

    return {
        "bottom_center_z_pass": close_enough(bin_info["bottom_center_z"], expected_bottom_center_z, HEIGHT_TOLERANCE_M),
        "bottom_top_z_pass": close_enough(bin_info["bottom_top_z"], expected_bottom_top_z, HEIGHT_TOLERANCE_M),
        "wall_center_z_pass": close_enough(bin_info["wall_center_z"], expected_wall_center_z, HEIGHT_TOLERANCE_M),
        "rim_z_pass": close_enough(bin_info["rim_z"], expected_rim_z, HEIGHT_TOLERANCE_M),
        "rim_above_bottom_pass": bin_info["rim_z"] > bin_info["bottom_top_z"],
    }


def check_dynamics(bin_info: dict) -> dict:
    per_part = {}
    all_pass = True
    for name, values in bin_info["applied_dynamics"].items():
        part_pass = (
            close_enough(values["lateral_friction"], BIN_LATERAL_FRICTION, DYNAMICS_TOLERANCE)
            and close_enough(values["rolling_friction"], BIN_ROLLING_FRICTION, DYNAMICS_TOLERANCE)
            and close_enough(values["spinning_friction"], BIN_SPINNING_FRICTION, DYNAMICS_TOLERANCE)
            and close_enough(values["restitution"], BIN_RESTITUTION, DYNAMICS_TOLERANCE)
        )
        per_part[name] = {**values, "pass": part_pass}
        all_pass = all_pass and part_pass
    return {"per_part": per_part, "all_pass": all_pass}


def run_drop_test(backend: So101PyBulletBackend, bin_info: dict) -> dict:
    drop_position = [bin_info["center_x"], bin_info["center_y"], bin_info["rim_z"] + DROP_TEST_START_HEIGHT_ABOVE_RIM_M]
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[DROP_TEST_HALF_EXTENT_M] * 3, physicsClientId=backend.client_id)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[DROP_TEST_HALF_EXTENT_M] * 3, rgbaColor=[1.0, 0.0, 0.0, 1.0], physicsClientId=backend.client_id)
    test_body_id = p.createMultiBody(
        baseMass=DROP_TEST_MASS_KG, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
        basePosition=drop_position, physicsClientId=backend.client_id,
    )

    backend.step(DROP_TEST_STEPS)

    final_position, _ = p.getBasePositionAndOrientation(test_body_id, physicsClientId=backend.client_id)
    p.removeBody(test_body_id, physicsClientId=backend.client_id)

    resting_height_error = abs((final_position[2] - DROP_TEST_HALF_EXTENT_M) - bin_info["bottom_top_z"])
    within_inner_bounds = (
        bin_info["inner_x_min"] <= final_position[0] <= bin_info["inner_x_max"]
        and bin_info["inner_y_min"] <= final_position[1] <= bin_info["inner_y_max"]
    )
    did_not_fall_through_table = final_position[2] > 0.0

    return {
        "drop_start_position": drop_position, "final_position": list(final_position),
        "resting_height_error_m": resting_height_error, "resting_on_bottom_pass": resting_height_error <= DROP_RESTING_HEIGHT_TOLERANCE_M,
        "within_inner_bounds_pass": within_inner_bounds, "did_not_fall_through_table_pass": did_not_fall_through_table,
    }


def run_wall_test(backend: So101PyBulletBackend, bin_info: dict) -> dict:
    """Pushes a small test object horizontally INTO the right wall from
    outside -- deterministic initial position/velocity/step count (see
    this task's own "지나치게 flaky하지 않도록")."""
    right_wall_body_id = backend.bin_body_ids["right_wall"]
    start_x = bin_info["outer_x_max"] + WALL_TEST_GAP_FROM_WALL_M
    start_position = [start_x, bin_info["center_y"], bin_info["wall_center_z"]]

    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[WALL_TEST_HALF_EXTENT_M] * 3, physicsClientId=backend.client_id)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[WALL_TEST_HALF_EXTENT_M] * 3, rgbaColor=[1.0, 1.0, 0.0, 1.0], physicsClientId=backend.client_id)
    test_body_id = p.createMultiBody(
        baseMass=WALL_TEST_MASS_KG, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
        basePosition=start_position, physicsClientId=backend.client_id,
    )
    p.resetBaseVelocity(test_body_id, linearVelocity=[-WALL_TEST_INITIAL_SPEED_MPS, 0.0, 0.0], physicsClientId=backend.client_id)

    contact_with_right_wall_recorded = False
    for _ in range(WALL_TEST_STEPS):
        backend.step(1)
        contacts = p.getContactPoints(bodyA=test_body_id, bodyB=right_wall_body_id, physicsClientId=backend.client_id)
        if contacts:
            contact_with_right_wall_recorded = True

    final_position, _ = p.getBasePositionAndOrientation(test_body_id, physicsClientId=backend.client_id)
    p.removeBody(test_body_id, physicsClientId=backend.client_id)

    right_wall_near_face_x = backend.bin_geometry["right_wall"]["position"][0] - backend.bin_geometry["right_wall"]["half_extents"][0]
    did_not_tunnel_through = final_position[0] >= (right_wall_near_face_x - WALL_TEST_PENETRATION_TOLERANCE_M)

    return {
        "start_position": start_position, "final_position": list(final_position),
        "contact_with_right_wall_recorded_pass": contact_with_right_wall_recorded,
        "right_wall_near_face_x": right_wall_near_face_x, "did_not_tunnel_through_pass": did_not_tunnel_through,
    }


def run_lifecycle_test(num_resets: int = 3) -> dict:
    backend = So101PyBulletBackend(gui=False, use_bin=True)
    try:
        body_id_sets = []
        for _ in range(num_resets):
            backend.reset()
            ids = set(backend.bin_body_ids[name] for name in ("bottom", "left_wall", "right_wall", "front_wall", "back_wall"))
            body_id_sets.append(sorted(ids))
        each_reset_has_exactly_five_unique_ids = all(len(s) == 5 for s in body_id_sets)
        return {
            "num_resets": num_resets, "body_id_sets_by_reset": body_id_sets,
            "each_reset_has_exactly_five_unique_ids_pass": each_reset_has_exactly_five_unique_ids,
        }
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    crashed = False
    crash_reason = None
    results = {}

    backend = So101PyBulletBackend(gui=False, use_bin=True)
    try:
        backend.reset()
        bin_info = backend.get_bin_debug_info()
        results["bin_info"] = bin_info

        results["structure"] = check_structure(backend)
        results["dimensions"] = check_dimensions(bin_info)
        results["heights"] = check_heights(bin_info)
        results["dynamics"] = check_dynamics(bin_info)
        results["drop_test"] = run_drop_test(backend, bin_info)
        results["wall_test"] = run_wall_test(backend, bin_info)
    except Exception as exc:
        crashed = True
        crash_reason = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    if not crashed:
        try:
            results["lifecycle_test"] = run_lifecycle_test()
        except Exception as exc:
            crashed = True
            crash_reason = f"{type(exc).__name__}: {exc}"

    overall_pass = (
        not crashed
        and results.get("structure", {}).get("num_bodies_pass", False)
        and results.get("structure", {}).get("has_bottom_and_four_walls", False)
        and results.get("structure", {}).get("all_ids_valid", False)
        and results.get("structure", {}).get("all_mass_zero_pass", False)
        and results.get("structure", {}).get("all_have_collision_pass", False)
        and results.get("dimensions", {}).get("inner_width_pass", False)
        and results.get("dimensions", {}).get("inner_length_pass", False)
        and results.get("dimensions", {}).get("outer_width_pass", False)
        and results.get("dimensions", {}).get("outer_length_pass", False)
        and results.get("dimensions", {}).get("wall_height_pass", False)
        and results.get("dimensions", {}).get("bottom_thickness_pass", False)
        and results.get("heights", {}).get("bottom_center_z_pass", False)
        and results.get("heights", {}).get("bottom_top_z_pass", False)
        and results.get("heights", {}).get("wall_center_z_pass", False)
        and results.get("heights", {}).get("rim_z_pass", False)
        and results.get("heights", {}).get("rim_above_bottom_pass", False)
        and results.get("dynamics", {}).get("all_pass", False)
        and results.get("drop_test", {}).get("resting_on_bottom_pass", False)
        and results.get("drop_test", {}).get("within_inner_bounds_pass", False)
        and results.get("drop_test", {}).get("did_not_fall_through_table_pass", False)
        and results.get("wall_test", {}).get("contact_with_right_wall_recorded_pass", False)
        and results.get("wall_test", {}).get("did_not_tunnel_through_pass", False)
        and results.get("lifecycle_test", {}).get("each_reset_has_exactly_five_unique_ids_pass", False)
    )

    output = {"crashed": crashed, "crash_reason": crash_reason, "overall_pass": overall_pass, "results": results}

    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("=== SO-101 open-top bin V1 smoke test ===")
    print(f"crashed: {crashed}" + (f" ({crash_reason})" if crashed else ""))
    if not crashed:
        print(f"structure: {results['structure']}")
        print(f"dimensions: {results['dimensions']}")
        print(f"heights: {results['heights']}")
        print(f"dynamics all_pass: {results['dynamics']['all_pass']}")
        print(f"drop_test: {results['drop_test']}")
        print(f"wall_test: {results['wall_test']}")
        print(f"lifecycle_test: {results['lifecycle_test']}")
    print(f"\n=== OVERALL PASS: {overall_pass} ===")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
