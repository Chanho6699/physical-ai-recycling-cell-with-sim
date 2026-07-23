"""Expert V2.1 (size-aware) compatibility test (see this task's chat
report, "V2.1 cube/box 결과" and "호환성 원칙"). Standalone script (plain
asserts + PASS/FAIL summary, matching this project's existing
convention -- not pytest).

Verifies: (1) V1 itself is untouched (re-runs
test_so101_expert_v1_regression.py's own checks are NOT duplicated here
-- run that file separately; this file only checks V2.1-vs-V1
equivalence), (2) feeding V2.1 the EXISTING cube/box dimensions
reproduces V1's own pre_grasp/approach targets within float tolerance
and reaches the SAME place_success outcome, (3) a cylinder of the same
footprint scale (radius=0.02m, matching cube/box's own 0.04m closing-
axis width) also succeeds via V1's unmodified success criterion.

Run:
  .venv-vla/bin/python -m benchmark.test_so101_expert_v2_size_aware
"""

import math

from benchmark.benchmark_so101_bin_diagnostic import (
    FIXED_BIN_MODE_ANCHOR_OFFSET_XY,
    FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY,
    FIXED_BIN_OBJECT_X_RANGE,
    FIXED_BIN_OBJECT_Y_RANGE,
)
from benchmark.collect_so101_stage1b_box_dataset import BOX_FOOTPRINT_XY, REGION_DEFS as STAGE1B_REGION_DEFS
from benchmark.evaluate_so101_expert_small_randomization import sample_object_position
from benchmark.so101_expert_v2_size_aware import ObjectMetadata, run_pick_and_place_episode_v2_1
from benchmark.so101_scripted_expert import (
    APPROACH_OFFSET_M,
    PRE_GRASP_OFFSET_M,
    So101ExpertError,
    compute_bin_success_debug,
    evaluate_bin_place_success,
    run_pick_and_place_episode,
)
from robot_sim.so101_pybullet_backend import DEFAULT_SCENE_CONFIG, So101PyBulletBackend

CUBE_SEED = 1
BOX_SEED = 15000
POSITION_TOLERANCE_M = 1e-6

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))


def build_backend(object_footprint_xy=None, object_shape="box", object_radius=None, seed=CUBE_SEED, x_range=None, y_range=None):
    x_range = x_range if x_range is not None else FIXED_BIN_OBJECT_X_RANGE
    y_range = y_range if y_range is not None else FIXED_BIN_OBJECT_Y_RANGE
    sampled_object_position = sample_object_position(seed, x_range, y_range)
    nominal_object_xy = DEFAULT_SCENE_CONFIG["surface_center_xy"]
    fixed_bin_center_xy = [
        nominal_object_xy[0] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[0], nominal_object_xy[1] + FIXED_BIN_MODE_ANCHOR_OFFSET_XY[1],
    ]
    scene_config = {"surface_footprint_xy": FIXED_BIN_MODE_SURFACE_FOOTPRINT_XY}
    if object_shape == "cylinder":
        scene_config["object_shape"] = "cylinder"
        scene_config["object_radius"] = object_radius
    elif object_footprint_xy is not None:
        scene_config["object_footprint_xy"] = object_footprint_xy
    backend = So101PyBulletBackend(
        gui=False, use_bin=True, object_position=sampled_object_position,
        bin_center_override_xy=fixed_bin_center_xy, scene_config=scene_config,
    )
    return backend, sampled_object_position


def run_v1(object_footprint_xy=None, seed=CUBE_SEED, x_range=None, y_range=None):
    backend, sampled_position = build_backend(object_footprint_xy, "box", None, seed, x_range, y_range)
    try:
        backend.reset()
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        try:
            result = run_pick_and_place_episode(backend, transport_delta_xy)
            return {"place_success": result["place_success"], "pre_grasp_target": result["pre_grasp"]["target"], "approach_target": result["approach"]["target"]}
        except So101ExpertError as exc:
            return {"place_success": False, "failure_reason": exc.failure_reason}
    finally:
        backend.close()


def run_v2_1(object_shape="box", object_footprint_xy=None, object_radius=None, height=0.04, seed=CUBE_SEED, x_range=None, y_range=None):
    backend, sampled_position = build_backend(object_footprint_xy, object_shape, object_radius, seed, x_range, y_range)
    try:
        backend.reset()
        object_position, _ = backend.get_object_pose()
        metadata = ObjectMetadata(
            shape=object_shape, position=list(object_position), height_m=height,
            footprint_xy_half_extents=object_footprint_xy, radius_m=object_radius,
        )
        transport_delta_xy = list(backend.scene_config["target_zone_offset_xy"])
        try:
            result = run_pick_and_place_episode_v2_1(backend, metadata, transport_delta_xy)
            bin_debug = result["bin_place_result"]["debug"]
            scene = backend.get_scene_state()
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
            place_success, failure_reason, _failure_phase = evaluate_bin_place_success(bin_success_debug)
            return {
                "place_success": place_success, "pre_grasp_target": result["pre_grasp"]["target"],
                "approach_target": result["approach"]["target"], "grasp_plan": result["grasp_plan"],
            }
        except So101ExpertError as exc:
            return {"place_success": False, "failure_reason": exc.failure_reason}
    finally:
        backend.close()


def main() -> None:
    # 1. Cube: V2.1 (fed the existing default cube footprint/height) must compute the SAME pre_grasp/approach targets as V1
    v1_cube = run_v1(object_footprint_xy=None, seed=CUBE_SEED)
    v2_cube = run_v2_1(object_shape="box", object_footprint_xy=DEFAULT_SCENE_CONFIG["object_footprint_xy"], height=DEFAULT_SCENE_CONFIG["object_height"], seed=CUBE_SEED)
    check("cube: V1 pre_grasp target == V2.1 pre_grasp target (tolerance)",
          all(abs(a - b) < POSITION_TOLERANCE_M for a, b in zip(v1_cube["pre_grasp_target"], v2_cube["pre_grasp_target"])),
          detail=f"{v1_cube['pre_grasp_target']} vs {v2_cube['pre_grasp_target']}")
    check("cube: V1 approach target == V2.1 approach target (tolerance)",
          all(abs(a - b) < POSITION_TOLERANCE_M for a, b in zip(v1_cube["approach_target"], v2_cube["approach_target"])),
          detail=f"{v1_cube['approach_target']} vs {v2_cube['approach_target']}")
    check("cube: V1 place_success == True", v1_cube["place_success"] is True)
    check("cube: V2.1 place_success == True (matches V1)", v2_cube["place_success"] is True)
    check("cube: V2.1 effective_object_width_m == 0.04 (2x default footprint half-extent)",
          abs(v2_cube["grasp_plan"]["effective_object_width_m"] - 0.04) < 1e-9)

    # 2. Rectangular box yaw=0: same check with Stage 1B box dimensions
    center_region = STAGE1B_REGION_DEFS["center"]
    v1_box = run_v1(object_footprint_xy=BOX_FOOTPRINT_XY, seed=BOX_SEED, x_range=center_region["x_range"], y_range=center_region["y_range"])
    v2_box = run_v2_1(object_shape="box", object_footprint_xy=BOX_FOOTPRINT_XY, height=0.04, seed=BOX_SEED, x_range=center_region["x_range"], y_range=center_region["y_range"])
    check("box: V1 pre_grasp target == V2.1 pre_grasp target (tolerance)",
          all(abs(a - b) < POSITION_TOLERANCE_M for a, b in zip(v1_box["pre_grasp_target"], v2_box["pre_grasp_target"])),
          detail=f"{v1_box['pre_grasp_target']} vs {v2_box['pre_grasp_target']}")
    check("box: V1 approach target == V2.1 approach target (tolerance)",
          all(abs(a - b) < POSITION_TOLERANCE_M for a, b in zip(v1_box["approach_target"], v2_box["approach_target"])),
          detail=f"{v1_box['approach_target']} vs {v2_box['approach_target']}")
    check("box: V1 place_success == True (known baseline)", v1_box["place_success"] is True)
    check("box: V2.1 place_success == True (matches V1)", v2_box["place_success"] is True)
    check("box: V2.1 effective_object_width_m == 0.04 (BOX_FOOTPRINT_XY[0]*2)",
          abs(v2_box["grasp_plan"]["effective_object_width_m"] - 0.04) < 1e-9)

    # 3. Cylinder (final Stage 1C candidate: radius=0.02m, height=0.04m, same footprint scale as cube/box)
    v2_cyl = run_v2_1(object_shape="cylinder", object_radius=0.02, height=0.04, seed=BOX_SEED, x_range=center_region["x_range"], y_range=center_region["y_range"])
    check("cylinder: V2.1 place_success == True (radius=0.02m candidate, center position)", v2_cyl["place_success"] is True)
    check("cylinder: V2.1 effective_object_width_m == 0.04 (2x radius)",
          abs(v2_cyl["grasp_plan"]["effective_object_width_m"] - 0.04) < 1e-9)
    check("cylinder: V2.1 recommended_grasp_axis == rotationally_symmetric",
          v2_cyl["grasp_plan"]["recommended_grasp_axis"] == "rotationally_symmetric")

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Total: {passed}/{len(results)} passed")
    if passed != len(results):
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
